# 🚀 Schritt-für-Schritt Implementierungs-Anleitung: Metriken-Integration

## 📋 Übersicht

Diese Anleitung führt dich durch die komplette Umsetzung des Anpassungsvorschlags. Alle Schritte sind nummeriert und können nacheinander abgearbeitet werden.

**Geschätzte Dauer:** 4-6 Stunden (je nach Erfahrung)

---

## ⚠️ Vorbereitung

### Schritt 0.1: Backup erstellen
```bash
# Backup der Datenbank erstellen (falls vorhanden)
pg_dump -h localhost -U postgres -d ml_training > backup_$(date +%Y%m%d_%H%M%S).sql
```

### Schritt 0.2: Prüfe Datenbank-Verbindung
```bash
# Teste ob Datenbank erreichbar ist
psql -h localhost -U postgres -d ml_training -c "SELECT version();"
```

### Schritt 0.3: Prüfe ob exchange_rates Tabelle existiert
```sql
-- Prüfe ob exchange_rates Tabelle existiert
SELECT EXISTS (
    SELECT FROM information_schema.tables 
    WHERE table_schema = 'public' 
    AND table_name = 'exchange_rates'
);
```

**Falls nicht vorhanden:** Siehe Schritt 1.3

---

## 📊 Phase 1: Datenbank-Schema Updates (KRITISCH)

### Schritt 1.1: Erstelle Migration für rug_detection_metrics

**Datei:** `sql/migration_add_rug_metrics.sql` (NEU erstellen)

```sql
-- ============================================================================
-- Migration: Rug-Detection-Metriken & Marktstimmung
-- Datum: 2024-12-XX
-- Beschreibung: Fügt rug_detection_metrics und market_context_enabled hinzu
-- ============================================================================

-- Erweitere ml_models Tabelle
ALTER TABLE ml_models 
ADD COLUMN IF NOT EXISTS rug_detection_metrics JSONB,
ADD COLUMN IF NOT EXISTS market_context_enabled BOOLEAN DEFAULT FALSE;

-- Erweitere ml_test_results Tabelle
ALTER TABLE ml_test_results
ADD COLUMN IF NOT EXISTS rug_detection_metrics JSONB;

-- Index für schnellere Queries
CREATE INDEX IF NOT EXISTS idx_ml_models_rug_metrics 
ON ml_models USING GIN (rug_detection_metrics);

-- Kommentare hinzufügen
COMMENT ON COLUMN ml_models.rug_detection_metrics IS 'JSONB Object: Rug-Pull-spezifische Metriken {"dev_sold_detection_rate": 0.85, "wash_trading_detection_rate": 0.72, "weighted_cost": 123.45, "precision_at_10": 0.90, ...}';
COMMENT ON COLUMN ml_models.market_context_enabled IS 'True wenn Marktstimmung (SOL-Preis-Kontext) aktiviert wurde';
COMMENT ON COLUMN ml_test_results.rug_detection_metrics IS 'JSONB Object: Rug-Pull-spezifische Metriken für Test-Ergebnisse';
```

**Ausführen:**
```bash
psql -h localhost -U postgres -d ml_training -f sql/migration_add_rug_metrics.sql
```

---

### Schritt 1.2: Prüfe coin_metrics Schema

**Prüfe ob alle neuen Spalten vorhanden sind:**
```sql
-- Prüfe ob alle benötigten Spalten in coin_metrics existieren
SELECT column_name, data_type 
FROM information_schema.columns 
WHERE table_name = 'coin_metrics' 
AND column_name IN (
    'dev_sold_amount',
    'buy_pressure_ratio',
    'unique_signer_ratio',
    'whale_buy_volume_sol',
    'whale_sell_volume_sol',
    'net_volume_sol',
    'volatility_pct',
    'avg_trade_size_sol',
    'num_whale_buys',
    'num_whale_sells'
)
ORDER BY column_name;
```

**Falls Spalten fehlen:** Siehe Schritt 1.3 (Migration für coin_metrics)

---

### Schritt 1.3: Erstelle exchange_rates Tabelle (falls nicht vorhanden)

**Datei:** `sql/migration_create_exchange_rates.sql` (NEU erstellen)

```sql
-- ============================================================================
-- Migration: Exchange Rates Tabelle erstellen
-- Datum: 2024-12-XX
-- Beschreibung: Erstellt exchange_rates Tabelle für Marktstimmung (SOL-Preis)
-- ============================================================================

CREATE TABLE IF NOT EXISTS exchange_rates (
    id SERIAL PRIMARY KEY,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    sol_price_usd NUMERIC(20, 6) NOT NULL,
    usd_to_eur_rate NUMERIC(10, 6),
    native_currency_price_usd NUMERIC(20, 6),
    blockchain_id INTEGER DEFAULT 1,
    source VARCHAR(50)
);

-- Indizes
CREATE INDEX IF NOT EXISTS idx_exchange_rates_created_at ON exchange_rates(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_exchange_rates_blockchain_id ON exchange_rates(blockchain_id);

-- Kommentare
COMMENT ON TABLE exchange_rates IS 'Marktstimmung ("Wasserstand") - Referenztabelle für KI-Training zur Unterscheidung von echten Token-Pumps vs. allgemeinen Marktbewegungen';
COMMENT ON COLUMN exchange_rates.sol_price_usd IS 'WICHTIG: Der "Wasserstand" - Aktueller SOL-Preis in USD (z.B. 145.50)';
COMMENT ON COLUMN exchange_rates.created_at IS 'Zeitstempel des Snapshots';
```

**Ausführen:**
```bash
psql -h localhost -U postgres -d ml_training -f sql/migration_create_exchange_rates.sql
```

---

## 🔧 Phase 2: Backend-Implementierung

### Schritt 2.1: Erweitere load_training_data() Funktion

**Datei:** `app/training/feature_engineering.py`

**Änderung:** Ersetze die SQL-Query in `load_training_data()` (ca. Zeile 91-98)

**ALT:**
```python
query = f"""
    SELECT timestamp, {feature_list}, phase_id_at_time
    FROM coin_metrics
    WHERE timestamp >= $1 AND timestamp <= $2
    {phase_filter}
    ORDER BY timestamp
    LIMIT ${param_count + 1}
"""
```

**NEU:**
```python
# ⚠️ WICHTIG: Lade IMMER alle neuen Metriken (auch wenn nicht in Features-Liste)
# Damit sind sie für Feature-Engineering verfügbar
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
    avg_trade_size_sol
"""

# Zusätzliche Features aus Request (falls nicht bereits in base_columns)
additional_features = [f for f in features if f not in [
    'timestamp', 'phase_id_at_time', 'price_open', 'price_high', 'price_low', 'price_close',
    'volume_sol', 'buy_volume_sol', 'sell_volume_sol', 'net_volume_sol',
    'market_cap_close', 'dev_sold_amount', 'buy_pressure_ratio', 'unique_signer_ratio',
    'whale_buy_volume_sol', 'whale_sell_volume_sol', 'num_whale_buys', 'num_whale_sells',
    'volatility_pct', 'avg_trade_size_sol'
]]

if additional_features:
    additional_list = ", ".join(additional_features)
    query = f"""
        SELECT {base_columns}, {additional_list}
        FROM coin_metrics
        WHERE timestamp >= $1 AND timestamp <= $2
        {phase_filter}
        ORDER BY timestamp
        LIMIT ${param_count + 1}
    """
else:
    query = f"""
        SELECT {base_columns}
        FROM coin_metrics
        WHERE timestamp >= $1 AND timestamp <= $2
        {phase_filter}
        ORDER BY timestamp
        LIMIT ${param_count + 1}
    """
```

**Test:**
```python
# Teste ob neue Spalten geladen werden
python3 -c "
import asyncio
from app.training.feature_engineering import load_training_data
from datetime import datetime, timedelta

async def test():
    end = datetime.now()
    start = end - timedelta(days=1)
    data = await load_training_data(start, end, ['price_close'])
    print(f'Spalten: {list(data.columns)}')
    print(f'dev_sold_amount vorhanden: {\"dev_sold_amount\" in data.columns}')

asyncio.run(test())
"
```

---

### Schritt 2.2: Erstelle enrich_with_market_context() Funktion

**Datei:** `app/training/feature_engineering.py`

**Hinzufügen:** Neue Funktion nach `load_training_data()` (ca. nach Zeile 131)

```python
async def enrich_with_market_context(
    data: pd.DataFrame,
    train_start: datetime,
    train_end: datetime
) -> pd.DataFrame:
    """
    Fügt Marktstimmung (SOL-Preis) zu Trainingsdaten hinzu.
    Merge mit Forward-Fill (nimmt letzten bekannten Wert).
    
    Args:
        data: DataFrame mit Trainingsdaten (muss timestamp als Index haben)
        train_start: Start-Zeitpunkt
        train_end: Ende-Zeitpunkt
    
    Returns:
        DataFrame mit zusätzlichen Spalten: sol_price_usd, sol_price_change_pct, sol_price_ma_5, sol_price_volatility
    """
    pool = await get_pool()
    
    # Konvertiere zu UTC
    train_start_utc = _ensure_utc(train_start)
    train_end_utc = _ensure_utc(train_end)
    
    # Lade Exchange Rates
    sql = """
        SELECT 
            created_at as timestamp,
            sol_price_usd,
            usd_to_eur_rate
        FROM exchange_rates
        WHERE created_at >= $1 AND created_at <= $2
        ORDER BY created_at
    """
    
    try:
        rows = await pool.fetch(sql, train_start_utc, train_end_utc)
        
        if not rows:
            logger.warning("⚠️ Keine Exchange Rates gefunden - Marktstimmung wird nicht hinzugefügt")
            return data
        
        # Konvertiere zu DataFrame
        rates_df = pd.DataFrame([dict(row) for row in rows])
        rates_df['timestamp'] = pd.to_datetime(rates_df['timestamp'])
        rates_df.set_index('timestamp', inplace=True)
        
        # Merge mit Forward-Fill (nehme letzten bekannten Wert)
        data = data.merge(rates_df[['sol_price_usd']], left_index=True, right_index=True, how='left')
        data['sol_price_usd'].fillna(method='ffill', inplace=True)
        
        # ✅ NEUE Features berechnen:
        data['sol_price_change_pct'] = data['sol_price_usd'].pct_change() * 100
        data['sol_price_ma_5'] = data['sol_price_usd'].rolling(5, min_periods=1).mean()
        data['sol_price_volatility'] = data['sol_price_usd'].rolling(10, min_periods=1).std()
        
        # NaN-Werte durch 0 ersetzen (am Anfang der Serie)
        data['sol_price_change_pct'].fillna(0, inplace=True)
        data['sol_price_ma_5'].fillna(data['sol_price_usd'], inplace=True)
        data['sol_price_volatility'].fillna(0, inplace=True)
        
        logger.info("✅ Marktstimmung hinzugefügt: sol_price_usd, sol_price_change_pct, sol_price_ma_5, sol_price_volatility")
        
    except Exception as e:
        logger.warning(f"⚠️ Fehler beim Laden der Exchange Rates: {e} - Marktstimmung wird nicht hinzugefügt")
    
    return data
```

**Test:**
```python
# Teste enrich_with_market_context
python3 -c "
import asyncio
from app.training.feature_engineering import load_training_data, enrich_with_market_context
from datetime import datetime, timedelta

async def test():
    end = datetime.now()
    start = end - timedelta(days=1)
    data = await load_training_data(start, end, ['price_close'])
    data = await enrich_with_market_context(data, start, end)
    print(f'Neue Spalten: {[c for c in data.columns if \"sol_price\" in c]}')

asyncio.run(test())
"
```

---

### Schritt 2.3: Modernisiere create_pump_detection_features() Funktion

**Datei:** `app/training/feature_engineering.py`

**Änderung:** Ersetze die gesamte Funktion `create_pump_detection_features()` (ca. Zeile 318-427)

**WICHTIG:** Die alte Funktion nutzt Spalten die NICHT existieren (`volume_usd`, `order_buy_volume`, etc.)

**NEU:**
```python
def create_pump_detection_features(
    data: pd.DataFrame,
    window_sizes: list = [5, 10, 15]
) -> pd.DataFrame:
    """
    MODERNISIERT: Nutzt neue Metriken aus coin_metrics.
    Erstellt zusätzliche Features für Pump-Detection.
    
    Args:
        data: DataFrame mit coin_metrics Daten (MUSS nach timestamp sortiert sein!)
        window_sizes: Fenstergrößen für Rolling-Berechnungen (in Anzahl Zeilen)
    
    Returns:
        DataFrame mit zusätzlichen Features (ursprüngliche Features bleiben erhalten)
    """
    df = data.copy()
    
    # ⚠️ WICHTIG: Daten müssen nach timestamp sortiert sein!
    if not df.index.is_monotonic_increasing:
        df = df.sort_index()
        logger.warning("⚠️ Daten wurden nach timestamp sortiert für Feature-Engineering")
    
    # ✅ 1. Dev-Tracking Features (KRITISCH!)
    if 'dev_sold_amount' in df.columns:
        df['dev_sold_flag'] = (df['dev_sold_amount'] > 0).astype(int)
        df['dev_sold_cumsum'] = df['dev_sold_amount'].cumsum()
        for window in window_sizes:
            df[f'dev_sold_spike_{window}'] = (
                df['dev_sold_amount'].rolling(window, min_periods=1).sum() > 0
            ).astype(int)
    
    # ✅ 2. Ratio-Features (schon berechnet in coin_metrics!)
    if 'buy_pressure_ratio' in df.columns:
        for window in window_sizes:
            df[f'buy_pressure_ma_{window}'] = (
                df['buy_pressure_ratio'].rolling(window, min_periods=1).mean()
            )
            df[f'buy_pressure_trend_{window}'] = (
                df['buy_pressure_ratio'] - df[f'buy_pressure_ma_{window}']
            )
    
    # ✅ 3. Whale-Aktivität Features
    if 'whale_buy_volume_sol' in df.columns and 'whale_sell_volume_sol' in df.columns:
        df['whale_net_volume'] = (
            df['whale_buy_volume_sol'] - df['whale_sell_volume_sol']
        )
        for window in window_sizes:
            df[f'whale_activity_{window}'] = (
                df['whale_buy_volume_sol'].rolling(window, min_periods=1).sum() +
                df['whale_sell_volume_sol'].rolling(window, min_periods=1).sum()
            )
    
    # ✅ 4. Volatilitäts-Features (nutzt neue volatility_pct Spalte!)
    if 'volatility_pct' in df.columns:
        for window in window_sizes:
            df[f'volatility_ma_{window}'] = (
                df['volatility_pct'].rolling(window, min_periods=1).mean()
            )
            df[f'volatility_spike_{window}'] = (
                df['volatility_pct'] > 
                df[f'volatility_ma_{window}'] * 1.5
            ).astype(int)
    
    # ✅ 5. Wash-Trading Detection
    if 'unique_signer_ratio' in df.columns:
        for window in window_sizes:
            df[f'wash_trading_flag_{window}'] = (
                df['unique_signer_ratio'].rolling(window, min_periods=1).mean() < 0.15
            ).astype(int)
    
    # ✅ 6. Net-Volume Features
    if 'net_volume_sol' in df.columns:
        for window in window_sizes:
            df[f'net_volume_ma_{window}'] = (
                df['net_volume_sol'].rolling(window, min_periods=1).mean()
            )
            df[f'volume_flip_{window}'] = (
                (df['net_volume_sol'] > 0).astype(int).diff().abs()
            )
    
    # ✅ 7. Price Momentum (nutzt price_close)
    if 'price_close' in df.columns:
        for window in window_sizes:
            df[f'price_change_{window}'] = df['price_close'].pct_change(periods=window) * 100
            df[f'price_roc_{window}'] = (
                (df['price_close'] - df['price_close'].shift(window)) / 
                df['price_close'].shift(window).replace(0, np.nan)
            ) * 100
    
    # ✅ 8. Volume Patterns (nutzt volume_sol)
    if 'volume_sol' in df.columns:
        for window in window_sizes:
            rolling_avg = df['volume_sol'].rolling(window=window, min_periods=1).mean()
            df[f'volume_ratio_{window}'] = df['volume_sol'] / rolling_avg.replace(0, np.nan)
            rolling_std = df['volume_sol'].rolling(window=window, min_periods=1).std()
            df[f'volume_spike_{window}'] = (
                (df['volume_sol'] - rolling_avg) / rolling_std.replace(0, np.nan)
            )
    
    # NaN-Werte durch 0 ersetzen (entstehen durch Rolling/Shift)
    df.fillna(0, inplace=True)
    
    # Infinite Werte durch 0 ersetzen
    df.replace([np.inf, -np.inf], 0, inplace=True)
    
    engineered_count = len([c for c in df.columns if c not in data.columns])
    logger.info(f"✅ {engineered_count} zusätzliche Features erstellt")
    
    return df
```

**Test:**
```python
# Teste modernisiertes Feature-Engineering
python3 -c "
import asyncio
import pandas as pd
from app.training.feature_engineering import load_training_data, create_pump_detection_features
from datetime import datetime, timedelta

async def test():
    end = datetime.now()
    start = end - timedelta(days=1)
    data = await load_training_data(start, end, ['price_close'])
    data = create_pump_detection_features(data, window_sizes=[5, 10])
    print(f'Anzahl Features: {len(data.columns)}')
    print(f'Neue Features: {[c for c in data.columns if \"dev_sold\" in c or \"whale\" in c][:5]}')

asyncio.run(test())
"
```

---

### Schritt 2.4: Erstelle validate_critical_features() Funktion

**Datei:** `app/training/feature_engineering.py`

**Hinzufügen:** Neue Funktion nach `get_engineered_feature_names()` (ca. nach Zeile 470)

```python
# Kritische Features für Rug-Detection
CRITICAL_FEATURES = [
    "dev_sold_amount",  # KRITISCH: Rug-Pull-Indikator
    "buy_pressure_ratio",  # Relatives Buy/Sell-Verhältnis
    "unique_signer_ratio",  # Wash-Trading-Erkennung
    "whale_buy_volume_sol",
    "whale_sell_volume_sol",
    "net_volume_sol",
    "volatility_pct"
]

def validate_critical_features(features: List[str]) -> Dict[str, bool]:
    """
    Prüft ob kritische Features verwendet werden.
    
    Args:
        features: Liste der Feature-Namen
    
    Returns:
        Dict mit {feature_name: bool} - True wenn Feature vorhanden
    """
    return {
        feature: feature in features 
        for feature in CRITICAL_FEATURES
    }
```

---

### Schritt 2.5: Integriere Feature-Validierung in train_model_sync()

**Datei:** `app/training/engine.py`

**Änderung:** Füge Feature-Validierung nach `prepare_features_for_training()` hinzu (ca. nach Zeile 228)

**Hinzufügen nach Zeile 228:**
```python
    # ✅ NEUE Validierung: Prüfe kritische Features
    from app.training.feature_engineering import validate_critical_features, CRITICAL_FEATURES
    
    missing_critical = validate_critical_features(features)
    
    if not missing_critical.get('dev_sold_amount'):
        logger.warning(
            "⚠️ KRITISCH: 'dev_sold_amount' fehlt in Features! "
            "Dies ist der wichtigste Rug-Pull-Indikator!"
        )
    
    if not missing_critical.get('buy_pressure_ratio'):
        logger.warning(
            "⚠️ WICHTIG: 'buy_pressure_ratio' fehlt - "
            "Bot-Spam vs. echtes Interesse kann nicht erkannt werden"
        )
    
    if not missing_critical.get('unique_signer_ratio'):
        logger.warning(
            "⚠️ WICHTIG: 'unique_signer_ratio' fehlt - "
            "Wash-Trading kann nicht erkannt werden"
        )
```

---

### Schritt 2.6: Integriere enrich_with_market_context() in train_model()

**Datei:** `app/training/engine.py`

**Änderung:** Füge Marktstimmung-Integration nach `load_training_data()` hinzu (ca. nach Zeile 551)

**Hinzufügen nach Zeile 551:**
```python
    # 3.5. Lade Marktstimmung (SOL-Preis-Kontext) - OPTIONAL
    use_market_context = final_params.get('use_market_context', False)
    
    if use_market_context:
        from app.training.feature_engineering import enrich_with_market_context
        logger.info("🌍 Füge Marktstimmung (SOL-Preis-Kontext) hinzu...")
        data = await enrich_with_market_context(
            data, 
            train_start=train_start, 
            train_end=train_end
        )
        
        # Füge Context-Features zu Features-Liste hinzu
        context_features = [
            "sol_price_usd",
            "sol_price_change_pct",
            "sol_price_ma_5",
            "sol_price_volatility"
        ]
        # Nur hinzufügen wenn nicht bereits vorhanden
        for cf in context_features:
            if cf not in features_for_training and cf in data.columns:
                features_for_training.append(cf)
                logger.info(f"➕ Context-Feature '{cf}' hinzugefügt")
    else:
        logger.info("ℹ️ Marktstimmung deaktiviert (use_market_context=False)")
```

---

### Schritt 2.7: Erstelle calculate_rug_detection_metrics() Funktion

**Datei:** `app/training/engine.py`

**Hinzufügen:** Neue Funktion nach `train_model_sync()` (ca. nach Zeile 587)

```python
def calculate_rug_detection_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_pred_proba: Optional[np.ndarray],
    X_test: np.ndarray,
    features: List[str]
) -> Dict[str, Any]:
    """
    Berechnet Rug-Pull-spezifische Metriken.
    
    Args:
        y_true: Echte Labels
        y_pred: Vorhergesagte Labels
        y_pred_proba: Vorhergesagte Wahrscheinlichkeiten (optional)
        X_test: Test-Features
        features: Liste der Feature-Namen
    
    Returns:
        Dict mit Rug-Detection-Metriken
    """
    from sklearn.metrics import confusion_matrix
    
    metrics = {}
    
    # 1. Dev-Sold Detection Rate (wenn Feature vorhanden)
    if 'dev_sold_amount' in features:
        try:
            dev_sold_idx = features.index('dev_sold_amount')
            dev_sold_mask = X_test[:, dev_sold_idx] > 0
            
            if dev_sold_mask.sum() > 0:
                dev_sold_detected = (y_pred[dev_sold_mask] == 1).sum()
                metrics['dev_sold_detection_rate'] = float(dev_sold_detected / dev_sold_mask.sum())
                logger.info(f"📊 Dev-Sold Detection Rate: {metrics['dev_sold_detection_rate']:.2%}")
        except (ValueError, IndexError) as e:
            logger.warning(f"⚠️ Konnte Dev-Sold Detection Rate nicht berechnen: {e}")
    
    # 2. Wash-Trading Detection (wenn Ratio vorhanden)
    if 'unique_signer_ratio' in features:
        try:
            ratio_idx = features.index('unique_signer_ratio')
            wash_trading_mask = X_test[:, ratio_idx] < 0.15
            
            if wash_trading_mask.sum() > 0:
                wash_detected = (y_pred[wash_trading_mask] == 1).sum()
                metrics['wash_trading_detection_rate'] = float(wash_detected / wash_trading_mask.sum())
                logger.info(f"📊 Wash-Trading Detection Rate: {metrics['wash_trading_detection_rate']:.2%}")
        except (ValueError, IndexError) as e:
            logger.warning(f"⚠️ Konnte Wash-Trading Detection Rate nicht berechnen: {e}")
    
    # 3. False Negative Cost (bei Rug-Pull-Detection ist FN teurer als FP!)
    try:
        cm = confusion_matrix(y_true, y_pred)
        if cm.size == 4:  # 2x2 Matrix
            tn, fp, fn, tp = cm.ravel()
            
            # FN = Rug wurde nicht erkannt (sehr teuer!)
            # FP = False Alarm (weniger schlimm)
            fn_cost = fn * 10.0  # FN ist 10x teurer
            fp_cost = fp * 1.0
            metrics['weighted_cost'] = float(fn_cost + fp_cost)
            logger.info(f"💰 Weighted Cost: {metrics['weighted_cost']:.2f} (FN={fn}, FP={fp})")
    except Exception as e:
        logger.warning(f"⚠️ Konnte Weighted Cost nicht berechnen: {e}")
    
    # 4. Profit @ Top-K (wenn Wahrscheinlichkeiten vorhanden)
    if y_pred_proba is not None:
        try:
            for k in [10, 20, 50, 100]:
                if len(y_pred_proba) >= k:
                    top_k_idx = np.argsort(y_pred_proba)[-k:]
                    precision_at_k = y_true[top_k_idx].sum() / k
                    metrics[f'precision_at_{k}'] = float(precision_at_k)
                    logger.info(f"📊 Precision @ Top-{k}: {precision_at_k:.2%}")
        except Exception as e:
            logger.warning(f"⚠️ Konnte Precision @ Top-K nicht berechnen: {e}")
    
    return metrics
```

---

### Schritt 2.8: Integriere calculate_rug_detection_metrics() in train_model_sync()

**Datei:** `app/training/engine.py`

**Änderung:** Füge Rug-Metriken-Berechnung nach Standard-Metriken hinzu (ca. nach Zeile 418)

**Hinzufügen nach Zeile 418:**
```python
    # 5.6. Rug-spezifische Metriken berechnen
    rug_metrics = {}
    try:
        y_pred_proba = None
        if hasattr(model, 'predict_proba'):
            y_pred_proba = model.predict_proba(X_final_test)[:, 1]
        
        rug_metrics = calculate_rug_detection_metrics(
            y_true=y_final_test,
            y_pred=y_pred,
            y_pred_proba=y_pred_proba,
            X_test=X_final_test,
            features=features
        )
        
        # Merge mit Standard-Metriken
        logger.info(f"📊 Rug-Detection-Metriken: {rug_metrics}")
    except Exception as e:
        logger.warning(f"⚠️ Fehler beim Berechnen der Rug-Detection-Metriken: {e}")
```

**Änderung:** Füge rug_metrics zu result-Dict hinzu (ca. Zeile 438)

**Hinzufügen in result-Dict (nach Zeile 458):**
```python
        "rug_detection_metrics": rug_metrics,  # NEU: Rug-spezifische Metriken
```

---

### Schritt 2.9: Update create_model() in models.py für rug_detection_metrics

**Datei:** `app/database/models.py`

**Änderung:** Erweitere `create_model()` Funktion um neue Parameter (ca. Zeile 63)

**Hinzufügen in Funktions-Signatur (nach Zeile 93):**
```python
    rug_detection_metrics: Optional[Dict[str, Any]] = None,  # NEU: Rug-Detection-Metriken
    market_context_enabled: Optional[bool] = False  # NEU: Marktstimmung aktiviert
```

**Hinzufügen in SQL-INSERT (nach Zeile 150):**
```python
        rug_detection_metrics,
        market_context_enabled
```

**Hinzufügen in VALUES (nach Zeile 200):**
```python
        to_jsonb(rug_detection_metrics),
        market_context_enabled
```

---

### Schritt 2.10: Update Default-Features-Liste

**Datei:** `app/api/schemas.py` oder `app/training/engine.py`

**Option 1:** In `app/api/schemas.py` - Erweitere `TrainModelRequest` Defaults

**Hinzufügen nach Zeile 24:**
```python
    # NEU: Defaults für zeitbasierte Vorhersage
    use_time_based_prediction: bool = Field(True, description="Zeitbasierte Vorhersage aktivieren (Default: True)")
    target_var: Optional[str] = Field("price_close", description="Ziel-Variable (Default: price_close)")
    future_minutes: Optional[int] = Field(10, description="Anzahl Minuten in die Zukunft (Default: 10)")
    min_percent_change: Optional[float] = Field(5.0, description="Mindest-Prozent-Änderung (Default: 5.0)")
    direction: Optional[str] = Field("up", description="Richtung: 'up' oder 'down' (Default: up)")
    
    # NEU: Marktstimmung
    use_market_context: bool = Field(False, description="Marktstimmung (SOL-Preis-Kontext) aktivieren")
    
    # NEU: Features ausschließen
    exclude_features: Optional[List[str]] = Field([], description="Features ausschließen (z.B. ['dev_sold_amount'])")
```

**Option 2:** Erstelle DEFAULT_FEATURES Konstante

**Datei:** `app/training/engine.py`

**Hinzufügen nach Zeile 17:**
```python
# Default-Features (wird verwendet wenn keine Features übergeben werden)
DEFAULT_FEATURES = [
    # Basis OHLC
    "price_open", "price_high", "price_low", "price_close",
    
    # Volumen
    "volume_sol", "buy_volume_sol", "sell_volume_sol", "net_volume_sol",
    
    # Market Cap & Phase
    "market_cap_close", "phase_id_at_time",
    
    # ⚠️ KRITISCH für Rug-Detection
    "dev_sold_amount",  # Wichtigster Indikator!
    
    # Ratio-Metriken (Bot-Spam vs. echtes Interesse)
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

**Integration in `train_model()` (ca. Zeile 479):**
```python
    # Wenn keine Features übergeben wurden, verwende Defaults
    if not features:
        features = DEFAULT_FEATURES.copy()
        logger.info(f"📊 Verwende Default-Features: {len(features)} Features")
    
    # Entferne ausgeschlossene Features
    if exclude_features:
        features = [f for f in features if f not in exclude_features]
        logger.info(f"📊 Features nach Ausschluss: {len(features)} Features (ausgeschlossen: {exclude_features})")
```

---

## 🎨 Phase 3: UI-Vereinfachung (Streamlit)

### Schritt 3.1: Vereinfache page_train() Funktion

**Datei:** `app/streamlit_app.py`

**Änderung:** Ersetze die gesamte `page_train()` Funktion

**WICHTIG:** Dies ist eine große Änderung. Erstelle zuerst eine Backup-Kopie!

**NEU:** Siehe separate Datei `app/streamlit_app_train_simplified.py` (wird in Schritt 3.2 erstellt)

---

### Schritt 3.2: Erstelle vereinfachte Train-UI

**Datei:** `app/streamlit_app_train_simplified.py` (NEU)

**Inhalt:** Vollständige vereinfachte `page_train()` Funktion

**Hinweis:** Diese Datei ist zu groß für diese Anleitung. Siehe separate Implementierung.

**Kern-Funktionalität:**
1. **Minimale Felder:**
   - Modell-Name (Text-Input)
   - Modell-Typ (Selectbox: Random Forest / XGBoost)
   - Trainings-Zeitraum (Date-Inputs)
   - Vorhersage-Ziel (Variable, Zeitraum, Prozent, Richtung)

2. **Erweiterte Optionen (ausklappbar):**
   - Feature-Auswahl mit Kategorien
   - Phasen-Filter
   - Hyperparameter
   - Feature-Engineering
   - Marktstimmung

3. **Label-Erstellung Transparenz:**
   - Info-Box die zeigt, wie Labels erstellt werden
   - Dynamisch aktualisiert basierend auf Einstellungen

---

### Schritt 3.3: Update AVAILABLE_FEATURES in streamlit_app.py

**Datei:** `app/streamlit_app.py`

**Änderung:** Ersetze `AVAILABLE_FEATURES` (ca. Zeile 88-100)

**NEU:**
```python
AVAILABLE_FEATURES = [
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
    "num_whale_buys",
    "num_whale_sells",
    
    # Volatilität
    "volatility_pct",
    "avg_trade_size_sol"
]
```

---

## 🧪 Phase 4: Testing

### Schritt 4.1: Teste Datenbank-Migrationen

```bash
# Teste Migration
psql -h localhost -U postgres -d ml_training -f sql/migration_add_rug_metrics.sql

# Prüfe ob Spalten hinzugefügt wurden
psql -h localhost -U postgres -d ml_training -c "
SELECT column_name, data_type 
FROM information_schema.columns 
WHERE table_name = 'ml_models' 
AND column_name IN ('rug_detection_metrics', 'market_context_enabled')
ORDER BY column_name;
"
```

---

### Schritt 4.2: Teste load_training_data() mit neuen Metriken

```python
# Test-Script: test_new_metrics.py
import asyncio
from app.training.feature_engineering import load_training_data
from datetime import datetime, timedelta

async def test():
    end = datetime.now()
    start = end - timedelta(days=7)
    
    data = await load_training_data(start, end, ['price_close'])
    
    required_columns = [
        'dev_sold_amount',
        'buy_pressure_ratio',
        'unique_signer_ratio',
        'whale_buy_volume_sol',
        'whale_sell_volume_sol',
        'net_volume_sol',
        'volatility_pct'
    ]
    
    missing = [c for c in required_columns if c not in data.columns]
    
    if missing:
        print(f"❌ Fehlende Spalten: {missing}")
    else:
        print("✅ Alle neuen Metriken vorhanden!")
        print(f"📊 Daten-Shape: {data.shape}")
        print(f"📊 Spalten: {len(data.columns)}")

asyncio.run(test())
```

**Ausführen:**
```bash
python3 test_new_metrics.py
```

---

### Schritt 4.3: Teste vollständiges Training mit neuen Features

```python
# Test-Script: test_full_training.py
import asyncio
from app.training.engine import train_model
from datetime import datetime, timedelta

async def test():
    end = datetime.now()
    start = end - timedelta(days=7)
    
    result = await train_model(
        model_type="random_forest",
        features=["price_close", "dev_sold_amount", "buy_pressure_ratio"],
        target_var="price_close",
        target_operator=None,
        target_value=None,
        train_start=start,
        train_end=end,
        use_time_based=True,
        future_minutes=10,
        min_percent_change=5.0,
        direction="up",
        params={
            "use_market_context": True,
            "use_engineered_features": True,
            "n_estimators": 50,  # Reduziert für schnelleren Test
            "max_depth": 5
        }
    )
    
    print(f"✅ Training erfolgreich!")
    print(f"📊 Accuracy: {result['accuracy']:.4f}")
    print(f"📊 F1: {result['f1']:.4f}")
    print(f"📊 Rug-Metriken: {result.get('rug_detection_metrics', {})}")

asyncio.run(test())
```

---

## 📝 Phase 5: Dokumentation

### Schritt 5.1: Update API-Dokumentation

**Datei:** `docs/API_BASE_URL_ERKLAERUNG.md` oder neue Datei

**Hinzufügen:**
- Neue Request-Parameter: `use_market_context`, `exclude_features`
- Neue Response-Felder: `rug_detection_metrics`, `market_context_enabled`
- Beispiele für vereinfachte Requests

---

### Schritt 5.2: Update Modell-Erstellungs-Dokumentation

**Datei:** `docs/KOMPLETTE_KI_MODELL_ANLEITUNG.md`

**Hinzufügen:**
- Abschnitt über neue Metriken
- Abschnitt über Marktstimmung-Integration
- Abschnitt über Rug-Detection-Metriken

---

## ✅ Checkliste

### Phase 1: Datenbank
- [ ] Migration `migration_add_rug_metrics.sql` erstellt und ausgeführt
- [ ] Migration `migration_create_exchange_rates.sql` erstellt und ausgeführt (falls nötig)
- [ ] Spalten `rug_detection_metrics` und `market_context_enabled` in `ml_models` vorhanden
- [ ] Spalte `rug_detection_metrics` in `ml_test_results` vorhanden
- [ ] `exchange_rates` Tabelle existiert

### Phase 2: Backend
- [ ] `load_training_data()` erweitert um neue Metriken
- [ ] `enrich_with_market_context()` Funktion erstellt
- [ ] `create_pump_detection_features()` modernisiert
- [ ] `validate_critical_features()` Funktion erstellt
- [ ] Feature-Validierung in `train_model_sync()` integriert
- [ ] Marktstimmung-Integration in `train_model()` integriert
- [ ] `calculate_rug_detection_metrics()` Funktion erstellt
- [ ] Rug-Metriken in `train_model_sync()` integriert
- [ ] `create_model()` in `models.py` erweitert
- [ ] Default-Features-Liste aktualisiert

### Phase 3: UI
- [ ] `page_train()` vereinfacht
- [ ] Feature-Auswahl mit Kategorien implementiert
- [ ] Label-Erstellung Transparenz implementiert
- [ ] `AVAILABLE_FEATURES` aktualisiert

### Phase 4: Testing
- [ ] Datenbank-Migrationen getestet
- [ ] `load_training_data()` mit neuen Metriken getestet
- [ ] Vollständiges Training getestet
- [ ] UI getestet

### Phase 5: Dokumentation
- [ ] API-Dokumentation aktualisiert
- [ ] Modell-Erstellungs-Dokumentation aktualisiert

---

## 🚨 Wichtige Hinweise

1. **Rückwärtskompatibilität:** Alte Modelle funktionieren weiterhin. Neue Features sind optional.

2. **Fehlerbehandlung:** Alle neuen Funktionen haben try-except Blöcke für Fehlerbehandlung.

3. **Performance:** Neue Features können Training verlangsamen. Teste mit kleinen Datensätzen zuerst.

4. **Datenbank:** Stelle sicher, dass `coin_metrics` alle neuen Spalten hat, bevor du trainierst.

5. **Exchange Rates:** Falls `exchange_rates` leer ist, wird Marktstimmung übersprungen (nur Warnung).

---

## 📞 Support

Bei Problemen:
1. Prüfe Logs: `docker logs ml-training-service`
2. Prüfe Datenbank: `SELECT * FROM ml_models ORDER BY created_at DESC LIMIT 1;`
3. Teste einzelne Funktionen mit Test-Scripts

---

**Status:** 📝 Implementierungs-Anleitung - **BEREIT FÜR UMSETZUNG**

**Nächster Schritt:** Beginne mit Phase 1 (Datenbank-Schema Updates)

