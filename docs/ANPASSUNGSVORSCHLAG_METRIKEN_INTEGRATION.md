# 🎯 Anpassungsvorschlag: Metriken-Integration & UI-Vereinfachung

## 📋 Übersicht

Dieses Dokument beschreibt den **strukturierten Vorschlag** für die Integration der neuen Metriken aus `coin_metrics` und `exchange_rates` sowie die Vereinfachung der UI für die Modell-Erstellung.

---

## 🎯 Ziele

1. **Integration neuer Metriken**: `dev_sold_amount`, `buy_pressure_ratio`, `unique_signer_ratio`, Whale-Metriken, etc.
2. **Marktstimmung-Integration**: SOL-Preis-Kontext aus `exchange_rates`
3. **UI-Vereinfachung**: Nur nötigste Einstellungen, Default: zeitbasierte Modelle
4. **Transparenz**: Klare Darstellung, wie Labels erstellt werden
5. **Flexibilität**: Modelle OHNE `dev_sold_amount` erstellen können

---

## 📊 Phase 1: Daten-Integration (KRITISCH)

### 1.1 Erweiterte `load_training_data()` Funktion

**Datei:** `app/training/feature_engineering.py`

**Änderungen:**
- Erweitere SQL-Query um **alle neuen Spalten** aus `coin_metrics`
- Lade automatisch: `dev_sold_amount`, `buy_pressure_ratio`, `unique_signer_ratio`, `whale_buy_volume_sol`, `whale_sell_volume_sol`, `net_volume_sol`, `volatility_pct`, `avg_trade_size_sol`
- **Wichtig**: Diese Spalten werden **immer** geladen (auch wenn nicht in Features-Liste), damit sie für Feature-Engineering verfügbar sind

**Neue SQL-Query:**
```sql
SELECT 
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
    
    -- Zusätzlich: Features aus Request
    {feature_list}
    
FROM coin_metrics
WHERE timestamp >= $1 AND timestamp <= $2
ORDER BY timestamp
LIMIT 500000
```

**Vorteil:**
- Alle neuen Metriken sind verfügbar
- Keine manuelle Feature-Auswahl nötig für Basis-Metriken
- Feature-Engineering kann auf alle Metriken zugreifen

---

### 1.2 Exchange Rates Integration (Marktstimmung)

**Datei:** `app/training/feature_engineering.py`

**Neue Funktion:** `enrich_with_market_context()`

**Zweck:**
- Fügt SOL-Preis-Kontext zu Trainingsdaten hinzu
- Ermöglicht KI zu lernen: "Token steigt, während SOL stabil ist" vs. "Token steigt, weil SOL steigt"

**Implementierung:**
```python
async def enrich_with_market_context(
    data: pd.DataFrame,
    train_start: datetime,
    train_end: datetime
) -> pd.DataFrame:
    """
    Fügt Marktstimmung (SOL-Preis) zu Trainingsdaten hinzu.
    Merge mit Forward-Fill (nimmt letzten bekannten Wert).
    """
    # Lade Exchange Rates aus exchange_rates Tabelle
    # Merge mit data basierend auf timestamp
    # Berechne: sol_price_change_pct, sol_price_ma_5, sol_price_volatility
    # Return: data mit neuen Spalten
```

**Neue Features:**
- `sol_price_usd` - Aktueller SOL-Preis
- `sol_price_change_pct` - Prozentuale Änderung
- `sol_price_ma_5` - 5-Perioden Moving Average
- `sol_price_volatility` - Volatilität des SOL-Preises

**Integration:**
- Wird **automatisch** aufgerufen in `train_model()` nach `load_training_data()`
- Features werden **automatisch** zu Features-Liste hinzugefügt (wenn aktiviert)

---

## 🔧 Phase 2: Feature-Engineering Modernisierung

### 2.1 Modernisierte `create_pump_detection_features()`

**Datei:** `app/training/feature_engineering.py`

**Problem:**
- Aktuelles Feature-Engineering nutzt **alte Spalten** (z.B. `volume_usd`, `order_buy_volume`)
- Diese existieren **nicht** in der neuen `coin_metrics` Tabelle!

**Lösung:**
- Nutze **neue Metriken** aus `coin_metrics`
- Erstelle Features basierend auf: `dev_sold_amount`, `buy_pressure_ratio`, `whale_buy_volume_sol`, etc.

**Neue Feature-Kategorien:**

1. **Dev-Tracking Features** (KRITISCH für Rug-Detection):
   - `dev_sold_flag` - Binary: `dev_sold_amount > 0`
   - `dev_sold_cumsum` - Kumulative Summe
   - `dev_sold_spike_{window}` - Spike-Erkennung über verschiedene Fenster

2. **Ratio-Features** (schon berechnet in `coin_metrics`):
   - `buy_pressure_ma_{window}` - Moving Average
   - `buy_pressure_trend_{window}` - Trend (aktuell vs. MA)
   - `wash_trading_flag_{window}` - `unique_signer_ratio < 0.15`

3. **Whale-Aktivität Features**:
   - `whale_net_volume` - `whale_buy_volume_sol - whale_sell_volume_sol`
   - `whale_activity_{window}` - Summe über Fenster

4. **Volatilitäts-Features** (nutzt `volatility_pct`):
   - `volatility_ma_{window}` - Moving Average
   - `volatility_spike_{window}` - Spike-Erkennung

5. **Net-Volume Features**:
   - `net_volume_ma_{window}` - Moving Average
   - `volume_flip_{window}` - Wechsel von positiv zu negativ

**Ergebnis:**
- Aus ~6 Basis-Features → ~60-80 erweiterte Features
- Features nutzen **tatsächliche Daten** aus `coin_metrics`
- Keine redundanten Berechnungen mehr

---

### 2.2 Feature-Validierung & Warnings

**Datei:** `app/training/engine.py`

**Neue Funktion:** `validate_critical_features()`

**Zweck:**
- Prüft ob kritische Features verwendet werden
- Gibt Warnungen aus, wenn wichtige Features fehlen

**Kritische Features:**
```python
CRITICAL_FEATURES = [
    "dev_sold_amount",  # KRITISCH: Rug-Pull-Indikator
    "buy_pressure_ratio",  # Relatives Buy/Sell-Verhältnis
    "unique_signer_ratio",  # Wash-Trading-Erkennung
    "whale_buy_volume_sol",
    "whale_sell_volume_sol",
    "net_volume_sol",
    "volatility_pct"
]
```

**Integration:**
- Wird in `train_model_sync()` nach `prepare_features_for_training()` aufgerufen
- Loggt Warnungen, wenn kritische Features fehlen
- **Blockiert NICHT** das Training (nur Warnung)

---

## 📈 Phase 3: Default-Features Update

### 3.1 Neue Default-Features-Liste

**Datei:** `app/training/engine.py` oder `app/api/schemas.py`

**Aktuell:**
```python
DEFAULT_FEATURES = [
    "price_open", "price_high", "price_low", "price_close",
    "volume_sol", "market_cap_close", "phase_id_at_time"
]
```

**Neu:**
```python
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

**Hinweis:**
- Diese Features werden **automatisch** verwendet, wenn keine Features-Liste übergeben wird
- Können in der UI **deaktiviert** werden (z.B. `dev_sold_amount` entfernen)

---

## 🎨 Phase 4: UI-Vereinfachung

### 4.1 Vereinfachte Modell-Erstellung (Streamlit)

**Datei:** `app/streamlit_app.py` → `page_train()`

**Aktuell:**
- Viele Einstellungen: Features, Target-Variable, Operator, Value, Phasen, Hyperparameter, Feature-Engineering, SMOTE, TimeSeriesSplit, etc.
- Komplex und überwältigend

**Neu:**
- **Default-Modus**: Nur zeitbasierte Modelle ("Steigt in X Minuten um Y%")
- **Minimale Einstellungen**:
  1. **Modell-Name** (Pflicht)
  2. **Modell-Typ** (Random Forest / XGBoost) - Default: Random Forest
  3. **Zeitraum** (Train Start/End) - Default: Letzte 30 Tage
  4. **Vorhersage-Ziel**:
     - **Ziel-Variable**: `price_close` (Default, auswählbar)
     - **Zeitraum**: 10 Minuten (Default, anpassbar)
     - **Mindest-Änderung**: 5% (Default, anpassbar)
     - **Richtung**: "up" (Default, auswählbar: "up" / "down")
  5. **Erweiterte Optionen** (ausklappbar):
     - Feature-Auswahl (Checkboxen für kritische Features)
     - Phasen-Filter
     - Hyperparameter (vereinfacht: nur n_estimators, max_depth)
     - Feature-Engineering (ein/aus)
     - Marktstimmung (SOL-Preis-Kontext) (ein/aus)

**UI-Layout:**
```
┌─────────────────────────────────────────────────┐
│ 🚀 Neues Modell erstellen                       │
├─────────────────────────────────────────────────┤
│                                                 │
│ Modell-Name: [________________]                │
│                                                 │
│ Modell-Typ: [Random Forest ▼]                  │
│                                                 │
│ Trainings-Zeitraum:                             │
│   Start: [2024-01-01]  Ende: [2024-01-31]      │
│                                                 │
│ Vorhersage-Ziel:                                │
│   Variable: [price_close ▼]                    │
│   Zeitraum: [10] Minuten                        │
│   Mindest-Änderung: [5.0] %                    │
│   Richtung: [Steigt (up) ▼]                    │
│                                                 │
│ [▶ Erweiterte Optionen]                        │
│                                                 │
│ [✅ Modell trainieren]                         │
└─────────────────────────────────────────────────┘
```

**Erweiterte Optionen (ausklappbar):**
```
┌─────────────────────────────────────────────────┐
│ Erweiterte Optionen                             │
├─────────────────────────────────────────────────┤
│                                                 │
│ Features (kritische Features):                  │
│   ☑ dev_sold_amount (Rug-Pull-Indikator)       │
│   ☑ buy_pressure_ratio (Bot-Spam-Erkennung)   │
│   ☑ unique_signer_ratio (Wash-Trading)          │
│   ☑ whale_buy_volume_sol (Whale-Aktivität)     │
│   ☑ volatility_pct (Volatilität)                │
│   ☑ net_volume_sol (Netto-Volumen)              │
│                                                 │
│ Phasen-Filter:                                  │
│   ☑ Phase 1 (Baby Zone)                        │
│   ☑ Phase 2 (Survival Zone)                    │
│   ☑ Phase 3 (Mature Zone)                     │
│                                                 │
│ Hyperparameter:                                 │
│   n_estimators: [100]                          │
│   max_depth: [10]                              │
│                                                 │
│ Feature-Engineering:                            │
│   ☑ Erweiterte Features aktivieren             │
│                                                 │
│ Marktstimmung:                                  │
│   ☑ SOL-Preis-Kontext hinzufügen               │
└─────────────────────────────────────────────────┘
```

**Vorteile:**
- **Einfach**: Nur 4-5 Felder für Standard-Modell
- **Flexibel**: Erweiterte Optionen für Experten
- **Transparent**: Klare Darstellung, was das Modell vorhersagt

---

### 4.2 Label-Erstellung Transparenz

**Datei:** `app/streamlit_app.py` → `page_train()`

**Neue Sektion:** "Wie werden Labels erstellt?"

**Zweck:**
- Zeigt dem Benutzer **genau**, wie Labels aus den Einstellungen erstellt werden
- Erklärt die Logik in verständlicher Sprache

**UI-Element:**
```python
st.info("""
📊 **Label-Erstellung:**

Für jede Zeile in den Trainingsdaten wird geprüft:

1. **Aktueller Wert**: `price_close` zum Zeitpunkt T
2. **Zukünftiger Wert**: `price_close` zum Zeitpunkt T + 10 Minuten
3. **Prozentuale Änderung**: `((Zukunft - Aktuell) / Aktuell) * 100`

**Label = 1** wenn:
- Änderung >= 5.0% (bei "Steigt")
- Änderung <= -5.0% (bei "Fällt")

**Label = 0** wenn:
- Bedingung nicht erfüllt

**Beispiel:**
- Aktuell: 100 SOL
- Zukunft (10 Min): 106 SOL
- Änderung: +6%
- **Label = 1** ✅ (weil 6% >= 5%)
""")
```

**Dynamisch aktualisiert:**
- Zeigt aktuelle Einstellungen (Variable, Zeitraum, Prozent, Richtung)
- Berechnet Beispiel-Labels basierend auf Einstellungen

---

### 4.3 Feature-Auswahl mit Kategorien

**Datei:** `app/streamlit_app.py` → `page_train()`

**Neue Sektion:** Feature-Auswahl mit Kategorien

**Layout:**
```
Features (kritische Features):

┌─ Dev-Tracking (Rug-Pull-Erkennung) ────────────┐
│ ☑ dev_sold_amount                              │
│   → Wichtigster Indikator für Rug-Pulls        │
└────────────────────────────────────────────────┘

┌─ Ratio-Metriken (Bot-Spam vs. echtes Interesse)┐
│ ☑ buy_pressure_ratio                           │
│ ☑ unique_signer_ratio                          │
└────────────────────────────────────────────────┘

┌─ Whale-Aktivität ──────────────────────────────┐
│ ☑ whale_buy_volume_sol                         │
│ ☑ whale_sell_volume_sol                        │
└────────────────────────────────────────────────┘

┌─ Volatilität ───────────────────────────────────┐
│ ☑ volatility_pct                               │
└────────────────────────────────────────────────┘
```

**Vorteile:**
- **Kategorisiert**: Features sind gruppiert nach Bedeutung
- **Erklärend**: Jede Kategorie hat Beschreibung
- **Flexibel**: Einzelne Features können deaktiviert werden

---

## 📊 Phase 5: Rug-spezifische Metriken

### 5.1 Neue Metriken-Funktion

**Datei:** `app/training/engine.py`

**Neue Funktion:** `calculate_rug_detection_metrics()`

**Zweck:**
- Berechnet Rug-Pull-spezifische Metriken
- Nicht nur generische Accuracy/F1, sondern auch:
  - Dev-Sold Detection Rate
  - Wash-Trading Detection Rate
  - Weighted Cost (FN ist teurer als FP bei Rug-Detection!)

**Metriken:**
1. **Dev-Sold Detection Rate**: Wie viele Rug-Pulls mit `dev_sold_amount > 0` wurden erkannt?
2. **Wash-Trading Detection Rate**: Wie viele Wash-Trading-Fälle wurden erkannt?
3. **Weighted Cost**: `FN * 10.0 + FP * 1.0` (False Negative ist 10x teurer!)
4. **Precision @ Top-K**: Precision bei Top-K Vorhersagen (wichtig für Trading!)

**Integration:**
- Wird in `train_model_sync()` nach Standard-Metriken aufgerufen
- Ergebnisse werden in `ml_models.rug_detection_metrics` (JSONB) gespeichert

---

### 5.2 Database Schema Update

**Datei:** `sql/migration_add_rug_metrics.sql` (NEU)

**Änderungen:**
```sql
-- Erweitere ml_models Tabelle
ALTER TABLE ml_models 
ADD COLUMN rug_detection_metrics JSONB,
ADD COLUMN market_context_enabled BOOLEAN DEFAULT FALSE;

-- Erweitere ml_test_results Tabelle
ALTER TABLE ml_test_results
ADD COLUMN rug_detection_metrics JSONB;

-- Index für schnellere Queries
CREATE INDEX idx_ml_models_rug_metrics 
ON ml_models USING GIN (rug_detection_metrics);
```

**JSONB-Struktur:**
```json
{
  "dev_sold_detection_rate": 0.85,
  "wash_trading_detection_rate": 0.72,
  "weighted_cost": 123.45,
  "precision_at_10": 0.90,
  "precision_at_20": 0.85,
  "precision_at_50": 0.78
}
```

---

## 🔄 Phase 6: API-Anpassungen

### 6.1 Vereinfachte Request-Schemas

**Datei:** `app/api/schemas.py`

**Änderungen:**
- `TrainModelRequest` erweitern um:
  - `use_market_context: bool = False` - Marktstimmung aktivieren
  - `exclude_features: List[str] = []` - Features ausschließen (z.B. `["dev_sold_amount"]`)

**Neue Defaults:**
- `use_time_based_prediction: bool = True` (Default: aktiviert!)
- `target_var: str = "price_close"` (Default)
- `future_minutes: int = 10` (Default)
- `min_percent_change: float = 5.0` (Default)
- `direction: str = "up"` (Default)

**Vorteil:**
- API unterstützt vereinfachte Requests
- Defaults sind sinnvoll gesetzt
- Rückwärtskompatibel (alte Requests funktionieren noch)

---

## 📋 Implementierungs-Roadmap

### Phase 1: KRITISCH (Diese Woche)
1. ✅ Erweitere `load_training_data()` um neue Metriken
2. ✅ Update Default-Features-Liste
3. ✅ Feature-Validierung mit Warnings

### Phase 2: WICHTIG (Nächste Woche)
4. ✅ Exchange Rates Integration (`enrich_with_market_context()`)
5. ✅ Modernisiertes Feature-Engineering (nutzt neue Metriken)
6. ✅ Rug-spezifische Metriken (`calculate_rug_detection_metrics()`)

### Phase 3: UI-VEREINFACHUNG (Parallel zu Phase 2)
7. ✅ Vereinfachte Modell-Erstellung (nur zeitbasierte Modelle)
8. ✅ Label-Erstellung Transparenz
9. ✅ Feature-Auswahl mit Kategorien

### Phase 4: DATABASE & API (Parallel zu Phase 2)
10. ✅ Schema Updates (rug_detection_metrics)
11. ✅ API-Anpassungen (vereinfachte Defaults)

---

## 🎯 Erwartete Verbesserungen

### KI-Performance:
- **+15-25% F1-Score** durch Dev-Tracking
- **+10-15% Precision** durch Ratio-Metriken
- **+5-10% Accuracy** durch Marktstimmung

### False-Negative Reduktion:
- **-40-60% FN-Rate** bei Rug-Pulls mit Dev-Sold
- **-20-30% FN-Rate** durch Wash-Trading-Detection

### Model-Robustheit:
- Weniger Overfitting durch Kontext-Features
- Bessere Generalisierung auf neue Coins

### UX-Verbesserung:
- **90% weniger Einstellungen** für Standard-Modell
- **Klare Transparenz** über Label-Erstellung
- **Flexible Feature-Auswahl** für Experten

---

## ⚠️ Wichtige Hinweise

### Rückwärtskompatibilität:
- **Alte Modelle** funktionieren weiterhin
- **Alte API-Requests** werden unterstützt (mit alten Defaults)
- **Neue Features** sind optional (können deaktiviert werden)

### Migration:
- **Keine Daten-Migration** nötig (nur Schema-Erweiterungen)
- **Bestehende Modelle** bleiben unverändert
- **Neue Modelle** nutzen automatisch neue Metriken

### Flexibilität:
- **Modelle OHNE `dev_sold_amount`** können erstellt werden (über `exclude_features`)
- **Marktstimmung** kann deaktiviert werden
- **Feature-Engineering** kann deaktiviert werden

---

## 📝 Zusammenfassung

### Vorher (Veraltet):
❌ Nutzt NICHT `dev_sold_amount` (wichtigster Indikator!)
❌ Ignoriert Ratio-Metriken (`buy_pressure`, `unique_signer`)
❌ Keine Marktstimmung (SOL-Preis-Kontext)
❌ Generische Metriken (Accuracy, F1)
❌ Feature-Engineering basiert auf alten Spalten
❌ Komplexe UI mit 1000 Einstellungen

### Nachher (Modern):
✅ Dev-Tracking als Kern-Feature
✅ Ratio-Metriken für Bot-Spam-Erkennung
✅ Marktstimmung für Kontext-Awareness
✅ Rug-spezifische Metriken (Dev Detection Rate, Weighted Cost)
✅ Modernisiertes Feature-Engineering (nutzt alle neuen Spalten)
✅ Vereinfachte UI (nur 4-5 Felder für Standard-Modell)
✅ Transparente Label-Erstellung
✅ Flexible Feature-Auswahl

---

## ❓ Offene Fragen

1. **Exchange Rates Tabelle**: Existiert diese bereits in der Datenbank? Falls nicht, muss sie erstellt werden.
2. **Feature-Auswahl**: Sollen alle neuen Features standardmäßig aktiviert sein, oder nur kritische?
3. **UI-Layout**: Soll die vereinfachte UI sofort aktiv sein, oder als "Einfacher Modus" neben dem "Experten-Modus"?
4. **Migration**: Sollen bestehende Modelle automatisch mit neuen Metriken neu trainiert werden, oder nur neue Modelle?

---

**Status:** 📝 Vorschlag - **NOCH NICHT IMPLEMENTIERT**

**Nächster Schritt:** Feedback vom Benutzer, dann Umsetzung nach Priorität.

