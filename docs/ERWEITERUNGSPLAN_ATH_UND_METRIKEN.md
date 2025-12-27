# 📊 Erweiterungsplan: ATH-Tracking & Metriken-Integration

**Datum:** 2025-01-XX  
**Ziel:** Integration der neuen Metriken aus dem überarbeiteten pump-metric Service in den ML-Training-Service

---

## 📋 Übersicht der neuen Features aus pump-metric

### ✅ Bereits integriert (in `load_training_data()` vorhanden):
- ✅ `dev_sold_amount` - Dev-Tracking (Rug-Pull-Indikator)
- ✅ `buy_pressure_ratio` - Relatives Buy/Sell-Verhältnis
- ✅ `unique_signer_ratio` - Wash-Trading-Erkennung
- ✅ `whale_buy_volume_sol`, `whale_sell_volume_sol` - Whale-Volumen
- ✅ `num_whale_buys`, `num_whale_sells` - Whale-Anzahl
- ✅ `net_volume_sol` - Netto-Volumen (Delta)
- ✅ `volatility_pct` - Volatilität
- ✅ `avg_trade_size_sol` - Durchschnittliche Trade-Größe

### ❌ Noch NICHT integriert:
- ❌ **ATH-Tracking** (`ath_price_sol`, `ath_timestamp`) - **NEU!**
  - Befindet sich in `coin_streams` Tabelle (nicht in `coin_metrics`)
  - Wird live getrackt und alle 5 Sekunden aktualisiert
  - Wichtig für: Preis-Momentum, Breakout-Erkennung, Resistance-Levels

---

## 🎯 Phase 1: ATH-Daten Integration

### 1.1 Datenbank-Schema-Verständnis

**ATH-Daten befinden sich in `coin_streams`:**
```sql
-- coin_streams Tabelle
ath_price_sol NUMERIC DEFAULT 0      -- All-Time High Preis
ath_timestamp TIMESTAMPTZ            -- Timestamp des letzten Updates
```

**Problem:** `coin_metrics` enthält keine ATH-Daten, aber wir brauchen sie für jedes Metrik-Intervall!

**Lösung:** JOIN mit `coin_streams` beim Laden der Trainingsdaten

### 1.2 Erweiterung von `load_training_data()`

**Datei:** `app/training/feature_engineering.py`

**Aktuell:**
```python
async def load_training_data(
    train_start: str | datetime,
    train_end: str | datetime,
    features: List[str],
    phases: Optional[List[int]] = None
) -> pd.DataFrame:
    # Lädt nur aus coin_metrics
    query = f"""
        SELECT {base_columns}
        FROM coin_metrics
        WHERE timestamp >= $1 AND timestamp <= $2
    """
```

**Neu (mit ATH-JOIN):**
```python
async def load_training_data(
    train_start: str | datetime,
    train_end: str | datetime,
    features: List[str],
    phases: Optional[List[int]] = None,
    include_ath: bool = True  # NEU: ATH-Daten optional laden
) -> pd.DataFrame:
    # JOIN mit coin_streams für ATH-Daten
    query = f"""
        SELECT 
            cm.{base_columns},
            cs.ath_price_sol,
            cs.ath_timestamp,
            -- Berechne ATH-Relative-Metriken direkt in SQL
            CASE 
                WHEN cm.price_close > 0 AND cs.ath_price_sol > 0 
                THEN ((cm.price_close / cs.ath_price_sol) - 1.0) * 100
                ELSE 0.0
            END AS price_vs_ath_pct,
            CASE 
                WHEN cs.ath_timestamp IS NOT NULL 
                THEN EXTRACT(EPOCH FROM (cm.timestamp - cs.ath_timestamp)) / 60.0
                ELSE NULL
            END AS minutes_since_ath
        FROM coin_metrics cm
        LEFT JOIN coin_streams cs ON cm.mint = cs.token_address
        WHERE cm.timestamp >= $1 AND cm.timestamp <= $2
    """
```

**Wichtig:**
- LEFT JOIN (falls Stream nicht existiert)
- Berechne ATH-Relative-Metriken direkt in SQL (Performance!)
- `price_vs_ath_pct`: Wie weit ist aktueller Preis vom ATH entfernt? (negativ = unter ATH, positiv = neuer ATH)
- `minutes_since_ath`: Wie lange ist es her, dass ATH erreicht wurde?

### 1.3 ATH-Features zu `base_columns` hinzufügen

**Datei:** `app/training/feature_engineering.py` (Zeile 79-108)

**Erweitern:**
```python
base_columns = """
    timestamp, 
    phase_id_at_time,
    
    -- Basis OHLC
    price_open, price_high, price_low, price_close,
    
    -- Volumen
    volume_sol, buy_volume_sol, sell_volume_sol, net_volume_sol,
    
    -- Market Cap & Phase
    market_cap_close,
    
    -- ⚠️ KRITISCH: Dev-Tracking (Rug-Pull-Indikator)
    dev_sold_amount,
    
    -- Ratio-Metriken (Bot-Spam vs. echtes Interesse)
    buy_pressure_ratio,
    unique_signer_ratio,
    
    -- Whale-Aktivität
    whale_buy_volume_sol,
    whale_sell_volume_sol,
    num_whale_buys,
    num_whale_sells,
    
    -- Volatilität
    volatility_pct,
    avg_trade_size_sol,
    
    -- 🆕 ATH-Tracking (aus coin_streams JOIN)
    ath_price_sol,
    ath_timestamp,
    price_vs_ath_pct,      -- Berechnet: ((price_close / ath_price_sol) - 1.0) * 100
    minutes_since_ath       -- Berechnet: Minuten seit letztem ATH
"""
```

**Hinweis:** `ath_price_sol` und `ath_timestamp` werden automatisch geladen (auch wenn nicht in Features-Liste), damit sie für Feature-Engineering verfügbar sind.

---

## 🎯 Phase 2: ATH-basierte Feature-Engineering

### 2.1 Erweiterung von `create_pump_detection_features()`

**Datei:** `app/training/feature_engineering.py` (Zeile 478-594)

**Neue Features hinzufügen:**

```python
def create_pump_detection_features(
    data: pd.DataFrame,
    window_sizes: list = [5, 10, 15]
) -> pd.DataFrame:
    df = data.copy()
    
    # ... bestehende Features ...
    
    # 🆕 10. ATH-basierte Features (KRITISCH für Breakout-Erkennung!)
    if 'ath_price_sol' in df.columns and 'price_close' in df.columns:
        # ATH-Relative-Metriken (bereits in SQL berechnet, aber für Rolling-Windows nutzen)
        if 'price_vs_ath_pct' not in df.columns:
            # Fallback: Berechne lokal falls nicht in SQL
            df['price_vs_ath_pct'] = (
                (df['price_close'] / df['ath_price_sol'].replace(0, np.nan) - 1.0) * 100
            )
        
        # ATH-Distance (wie weit vom ATH entfernt?)
        df['ath_distance_pct'] = -df['price_vs_ath_pct']  # Negativ = unter ATH, positiv = über ATH
        df['is_near_ath'] = (df['ath_distance_pct'].abs() < 5.0).astype(int)  # Innerhalb 5% vom ATH
        df['is_at_ath'] = (df['price_close'] >= df['ath_price_sol'] * 0.999).astype(int)  # Innerhalb 0.1% vom ATH
        
        # ATH-Breakout-Erkennung
        df['ath_breakout'] = (df['price_close'] > df['ath_price_sol']).astype(int)
        df['ath_breakout_volume'] = df['ath_breakout'] * df['volume_sol']  # Volumen bei Breakout
        
        # Rolling-Windows für ATH-Features
        for window in window_sizes:
            # ATH-Trend (nähert sich Preis dem ATH?)
            df[f'ath_distance_trend_{window}'] = (
                df['ath_distance_pct'].rolling(window, min_periods=1).mean()
            )
            df[f'ath_approach_{window}'] = (
                df['ath_distance_trend_{window}'].diff() < 0
            ).astype(int)  # Nähert sich dem ATH
            
            # ATH-Breakout-Häufigkeit
            df[f'ath_breakout_count_{window}'] = (
                df['ath_breakout'].rolling(window, min_periods=1).sum()
            )
            
            # ATH-Volumen bei Breakouts
            df[f'ath_breakout_volume_ma_{window}'] = (
                df['ath_breakout_volume'].rolling(window, min_periods=1).mean()
            )
    
    # 🆕 11. ATH-Zeit-Features (wie lange ist es her?)
    if 'minutes_since_ath' in df.columns:
        df['ath_age_hours'] = df['minutes_since_ath'] / 60.0
        df['ath_is_recent'] = (df['minutes_since_ath'] < 60).astype(int)  # Innerhalb 1 Stunde
        df['ath_is_old'] = (df['minutes_since_ath'] > 1440).astype(int)  # Älter als 24 Stunden
        
        for window in window_sizes:
            df[f'ath_age_trend_{window}'] = (
                df['minutes_since_ath'].rolling(window, min_periods=1).mean()
            )
    
    # ... NaN-Handling ...
    
    return df
```

**Neue Features (Zusammenfassung):**
1. `ath_distance_pct` - Wie weit vom ATH entfernt? (negativ = unter ATH)
2. `is_near_ath` - Innerhalb 5% vom ATH? (0/1)
3. `is_at_ath` - Innerhalb 0.1% vom ATH? (0/1)
4. `ath_breakout` - Neuer ATH erreicht? (0/1)
5. `ath_breakout_volume` - Volumen bei Breakout
6. `ath_distance_trend_{window}` - Trend: Nähert sich dem ATH?
7. `ath_approach_{window}` - Nähert sich dem ATH? (0/1)
8. `ath_breakout_count_{window}` - Anzahl Breakouts im Fenster
9. `ath_breakout_volume_ma_{window}` - Durchschnittliches Breakout-Volumen
10. `ath_age_hours` - Alter des ATH in Stunden
11. `ath_is_recent` - ATH innerhalb 1 Stunde? (0/1)
12. `ath_is_old` - ATH älter als 24 Stunden? (0/1)
13. `ath_age_trend_{window}` - Trend des ATH-Alters

**Gesamt:** ~30 neue Features (bei 3 window_sizes)

### 2.2 Update von `get_engineered_feature_names()`

**Datei:** `app/training/feature_engineering.py` (Zeile 597-637)

**Erweitern:**
```python
def get_engineered_feature_names(window_sizes: list = [5, 10, 15]) -> list:
    features = []
    
    # ... bestehende Features ...
    
    # 🆕 ATH-basierte Features
    features.extend([
        'ath_distance_pct',
        'is_near_ath',
        'is_at_ath',
        'ath_breakout',
        'ath_breakout_volume',
        'ath_age_hours',
        'ath_is_recent',
        'ath_is_old'
    ])
    
    # ATH Rolling-Windows
    for w in window_sizes:
        features.extend([
            f'ath_distance_trend_{w}',
            f'ath_approach_{w}',
            f'ath_breakout_count_{w}',
            f'ath_breakout_volume_ma_{w}',
            f'ath_age_trend_{w}'
        ])
    
    return features
```

---

## 🎯 Phase 3: Default-Features Update

### 3.1 ATH-Features zu DEFAULT_FEATURES hinzufügen

**Datei:** `app/training/engine.py` (Zeile 18-43)

**Aktuell:**
```python
DEFAULT_FEATURES = [
    # Basis OHLC
    "price_open", "price_high", "price_low", "price_close",
    
    # Volumen
    "volume_sol", "buy_volume_sol", "sell_volume_sol", "net_volume_sol",
    
    # Market Cap & Phase
    "market_cap_close", "phase_id_at_time",
    
    # ⚠️ KRITISCH für Rug-Detection
    "dev_sold_amount",
    
    # Ratio-Metriken
    "buy_pressure_ratio",
    "unique_signer_ratio",
    
    # Whale-Aktivität
    "whale_buy_volume_sol",
    "whale_sell_volume_sol",
    
    # Volatilität
    "volatility_pct",
    "avg_trade_size_sol"
]
```

**Neu (mit ATH):**
```python
DEFAULT_FEATURES = [
    # Basis OHLC
    "price_open", "price_high", "price_low", "price_close",
    
    # Volumen
    "volume_sol", "buy_volume_sol", "sell_volume_sol", "net_volume_sol",
    
    # Market Cap & Phase
    "market_cap_close", "phase_id_at_time",
    
    # ⚠️ KRITISCH für Rug-Detection
    "dev_sold_amount",
    
    # Ratio-Metriken
    "buy_pressure_ratio",
    "unique_signer_ratio",
    
    # Whale-Aktivität
    "whale_buy_volume_sol",
    "whale_sell_volume_sol",
    
    # Volatilität
    "volatility_pct",
    "avg_trade_size_sol",
    
    # 🆕 ATH-Tracking (Breakout-Erkennung)
    "ath_price_sol",
    "price_vs_ath_pct",      # Wie weit vom ATH entfernt?
    "minutes_since_ath"      # Wie lange ist es her?
]
```

**Hinweis:** `ath_timestamp` wird nicht als Feature verwendet (nur für Berechnungen), aber `minutes_since_ath` ist nützlich.

### 3.2 CRITICAL_FEATURES erweitern

**Datei:** `app/training/feature_engineering.py` (Zeile 639-648)

**Aktuell:**
```python
CRITICAL_FEATURES = [
    "dev_sold_amount",  # KRITISCH: Rug-Pull-Indikator
    "buy_pressure_ratio",
    "unique_signer_ratio",
    "whale_buy_volume_sol",
    "whale_sell_volume_sol",
    "net_volume_sol",
    "volatility_pct"
]
```

**Neu (mit ATH):**
```python
CRITICAL_FEATURES = [
    "dev_sold_amount",  # KRITISCH: Rug-Pull-Indikator
    "buy_pressure_ratio",
    "unique_signer_ratio",
    "whale_buy_volume_sol",
    "whale_sell_volume_sol",
    "net_volume_sol",
    "volatility_pct",
    "price_vs_ath_pct"  # 🆕 KRITISCH: ATH-Distance (Breakout-Erkennung)
]
```

---

## 🎯 Phase 4: Datenbank-Validierung

### 4.1 Prüfung ob ATH-Daten verfügbar sind

**Neue Funktion:** `validate_ath_data_availability()`

**Datei:** `app/training/feature_engineering.py`

```python
async def validate_ath_data_availability(
    train_start: datetime,
    train_end: datetime
) -> Dict[str, Any]:
    """
    Prüft ob ATH-Daten für den Zeitraum verfügbar sind.
    
    Returns:
        Dict mit:
        - available: bool
        - coins_with_ath: int
        - coins_without_ath: int
        - coverage_pct: float
    """
    pool = await get_pool()
    
    # Prüfe wie viele Coins ATH-Daten haben
    query = """
        SELECT 
            COUNT(DISTINCT cm.mint) as total_coins,
            COUNT(DISTINCT CASE WHEN cs.ath_price_sol > 0 THEN cm.mint END) as coins_with_ath,
            COUNT(DISTINCT CASE WHEN cs.ath_price_sol = 0 OR cs.ath_price_sol IS NULL THEN cm.mint END) as coins_without_ath
        FROM coin_metrics cm
        LEFT JOIN coin_streams cs ON cm.mint = cs.token_address
        WHERE cm.timestamp >= $1 AND cm.timestamp <= $2
    """
    
    row = await pool.fetchrow(query, train_start, train_end)
    
    total_coins = row['total_coins'] or 0
    coins_with_ath = row['coins_with_ath'] or 0
    coins_without_ath = row['coins_without_ath'] or 0
    
    coverage_pct = (coins_with_ath / total_coins * 100) if total_coins > 0 else 0.0
    
    return {
        "available": coins_with_ath > 0,
        "coins_with_ath": coins_with_ath,
        "coins_without_ath": coins_without_ath,
        "coverage_pct": coverage_pct,
        "total_coins": total_coins
    }
```

**Integration in `train_model()`:**
```python
# Vor dem Training: Prüfe ATH-Daten-Verfügbarkeit
ath_validation = await validate_ath_data_availability(train_start_utc, train_end_utc)
if not ath_validation["available"]:
    logger.warning(f"⚠️ Keine ATH-Daten verfügbar! Coverage: {ath_validation['coverage_pct']:.1f}%")
else:
    logger.info(f"✅ ATH-Daten verfügbar: {ath_validation['coins_with_ath']}/{ath_validation['total_coins']} Coins ({ath_validation['coverage_pct']:.1f}%)")
```

---

## 🎯 Phase 5: Performance-Optimierung

### 5.1 Index für ATH-JOIN

**SQL-Migration:** `sql/migration_add_ath_indexes.sql`

```sql
-- Index für schnellen JOIN zwischen coin_metrics und coin_streams
CREATE INDEX IF NOT EXISTS idx_coin_metrics_mint 
ON coin_metrics(mint);

CREATE INDEX IF NOT EXISTS idx_coin_streams_token_address 
ON coin_streams(token_address);

-- Composite Index für ATH-Abfragen
CREATE INDEX IF NOT EXISTS idx_coin_streams_ath 
ON coin_streams(token_address, ath_price_sol, ath_timestamp) 
WHERE is_active = TRUE;
```

**Ausführen:**
```bash
psql -h localhost -U postgres -d crypto -f sql/migration_add_ath_indexes.sql
```

### 5.2 Caching für ATH-Daten

**Optional:** Wenn viele Coins getrackt werden, kann ATH-Cache helfen.

**Datei:** `app/training/feature_engineering.py`

```python
# Cache für ATH-Daten (optional, für Performance)
_ath_cache: Dict[str, Dict[str, Any]] = {}
_ath_cache_ttl = 60  # 60 Sekunden

async def get_ath_for_mint(mint: str) -> Optional[Dict[str, Any]]:
    """Holt ATH-Daten für einen Coin (mit Cache)."""
    # Prüfe Cache
    if mint in _ath_cache:
        cached_data = _ath_cache[mint]
        if time.time() - cached_data['timestamp'] < _ath_cache_ttl:
            return cached_data['data']
    
    # Lade aus DB
    pool = await get_pool()
    row = await pool.fetchrow(
        "SELECT ath_price_sol, ath_timestamp FROM coin_streams WHERE token_address = $1",
        mint
    )
    
    if row:
        data = {
            'ath_price_sol': row['ath_price_sol'],
            'ath_timestamp': row['ath_timestamp']
        }
        _ath_cache[mint] = {'data': data, 'timestamp': time.time()}
        return data
    
    return None
```

**Hinweis:** Nur implementieren wenn Performance-Probleme auftreten!

---

## 🎯 Phase 6: Dokumentation & Tests

### 6.1 Dokumentation aktualisieren

**Dateien:**
- `docs/COMPLETE_WORKFLOW_DOKUMENTATION.md` - ATH-Integration dokumentieren
- `docs/MODELL_ERSTELLUNG_KOMPLETT_DOKUMENTATION.md` - ATH-Features erklären
- `docs/SCHEMA_PRUEFUNG_ERGEBNIS.md` - ATH-Spalten dokumentieren

### 6.2 Tests erstellen

**Datei:** `tests/test_ath_integration.py`

```python
"""
Tests für ATH-Integration
"""
import pytest
from app.training.feature_engineering import (
    load_training_data,
    validate_ath_data_availability,
    create_pump_detection_features
)

@pytest.mark.asyncio
async def test_load_training_data_with_ath():
    """Test: Lädt Trainingsdaten mit ATH-Daten."""
    data = await load_training_data(
        train_start="2024-01-01T00:00:00Z",
        train_end="2024-01-02T00:00:00Z",
        features=["price_close", "ath_price_sol"],
        include_ath=True
    )
    
    assert 'ath_price_sol' in data.columns
    assert 'price_vs_ath_pct' in data.columns
    assert 'minutes_since_ath' in data.columns

@pytest.mark.asyncio
async def test_ath_feature_engineering():
    """Test: ATH-Features werden korrekt erstellt."""
    # Mock-Daten
    data = pd.DataFrame({
        'price_close': [0.001, 0.0015, 0.002],
        'ath_price_sol': [0.002, 0.002, 0.002],
        'volume_sol': [10, 20, 30]
    })
    
    df = create_pump_detection_features(data, window_sizes=[5])
    
    assert 'ath_distance_pct' in df.columns
    assert 'is_near_ath' in df.columns
    assert 'ath_breakout' in df.columns
```

---

## 📋 Implementierungs-Checkliste

### Phase 1: ATH-Daten Integration
- [ ] `load_training_data()` erweitern mit LEFT JOIN zu `coin_streams`
- [ ] ATH-Relative-Metriken in SQL berechnen (`price_vs_ath_pct`, `minutes_since_ath`)
- [ ] `base_columns` erweitern um ATH-Felder
- [ ] Tests: Prüfe ob ATH-Daten korrekt geladen werden

### Phase 2: ATH-Feature-Engineering
- [ ] `create_pump_detection_features()` erweitern um ATH-Features
- [ ] `get_engineered_feature_names()` aktualisieren
- [ ] Tests: Prüfe ob ATH-Features korrekt erstellt werden

### Phase 3: Default-Features Update
- [ ] `DEFAULT_FEATURES` erweitern um ATH-Features
- [ ] `CRITICAL_FEATURES` erweitern um `price_vs_ath_pct`
- [ ] Tests: Prüfe ob Default-Features korrekt verwendet werden

### Phase 4: Datenbank-Validierung
- [ ] `validate_ath_data_availability()` implementieren
- [ ] Integration in `train_model()` (Warnung wenn keine ATH-Daten)
- [ ] Tests: Prüfe Validierung

### Phase 5: Performance-Optimierung
- [ ] SQL-Migration für Indizes erstellen
- [ ] Indizes in Datenbank erstellen
- [ ] Performance-Tests (optional: Caching)

### Phase 6: Dokumentation & Tests
- [ ] Dokumentation aktualisieren
- [ ] Unit-Tests erstellen
- [ ] Integration-Tests erstellen

---

## ⚠️ Wichtige Hinweise

### 1. LEFT JOIN vs. INNER JOIN
- **LEFT JOIN:** Behält alle `coin_metrics` Zeilen, auch wenn kein Stream existiert
- **INNER JOIN:** Verliert Zeilen ohne Stream (kann zu Datenverlust führen)
- **Empfehlung:** LEFT JOIN verwenden, NULL-Werte durch 0 ersetzen

### 2. ATH-Daten können NULL sein
- Neue Coins haben möglicherweise noch kein ATH (`ath_price_sol = 0` oder NULL)
- Behandlung: `COALESCE(ath_price_sol, 0)` in SQL
- Feature-Engineering: Prüfe auf `ath_price_sol > 0` vor Berechnungen

### 3. Performance bei vielen Coins
- JOIN kann langsam sein bei vielen Coins (1000+)
- Lösung: Indizes auf `coin_metrics.mint` und `coin_streams.token_address`
- Optional: Caching für ATH-Daten (nur wenn nötig)

### 4. Rückwärtskompatibilität
- `include_ath` Parameter ist optional (Default: `True`)
- Falls `False`: Alte Funktionalität bleibt erhalten
- ATH-Features werden nur erstellt wenn ATH-Daten vorhanden sind

---

## 🎯 Zusammenfassung

**Neue Features:**
1. ✅ ATH-Daten werden aus `coin_streams` geladen (JOIN)
2. ✅ ATH-Relative-Metriken werden in SQL berechnet (Performance!)
3. ✅ ~30 neue ATH-basierte Features im Feature-Engineering
4. ✅ ATH-Features in DEFAULT_FEATURES und CRITICAL_FEATURES
5. ✅ Validierung ob ATH-Daten verfügbar sind
6. ✅ Performance-Optimierung durch Indizes

**Geschätzter Aufwand:**
- Phase 1-2: 4-6 Stunden
- Phase 3-4: 2-3 Stunden
- Phase 5-6: 2-3 Stunden
- **Gesamt: 8-12 Stunden**

**Priorität:**
- 🔴 **HOCH** - ATH-Tracking ist wichtig für Breakout-Erkennung
- ATH-Features können Modell-Performance deutlich verbessern
- Besonders wichtig für zeitbasierte Vorhersagen (Pump-Erkennung)

---

**Erstellt:** 2025-01-XX  
**Status:** 📋 Plan erstellt, bereit für Implementierung


