# 📚 Komplette Dokumentation: Modellerstellung - UI, API & Backend

## 🎯 Übersicht

Diese Dokumentation beschreibt **alle Möglichkeiten und Bedingungen** zur Erstellung von ML-Modellen über die **Streamlit UI** und die **REST API**, sowie die **Backend-Logik** und **Datenbank-Speicherung**.

---

## 📋 Inhaltsverzeichnis

1. [Modell-Typen](#1-modell-typen)
2. [Vorhersage-Typen](#2-vorhersage-typen)
3. [Verfügbare Features](#3-verfügbare-features)
4. [Verfügbare Target-Variablen](#4-verfügbare-target-variablen)
5. [Coin-Phasen](#5-coin-phasen)
6. [Hyperparameter](#6-hyperparameter)
7. [Feature-Engineering (Optional)](#7-feature-engineering-optional)
8. [SMOTE - Imbalanced Data Handling (Optional)](#8-smote---imbalanced-data-handling-optional)
9. [TimeSeriesSplit - Cross-Validation (Optional)](#9-timeseriessplit---cross-validation-optional)
10. [Modellerstellung über UI](#10-modellerstellung-über-ui)
11. [Modellerstellung über API](#11-modellerstellung-über-api)
12. [Backend-Logik](#12-backend-logik)
13. [Datenbank-Speicherung](#13-datenbank-speicherung)
14. [Validierungen & Bedingungen](#14-validierungen--bedingungen)
15. [Job-Verarbeitung](#15-job-verarbeitung)

---

## 1. Modell-Typen

### 1.1 Unterstützte Modell-Typen

| Modell-Typ | Beschreibung | Standard-Hyperparameter |
|------------|--------------|------------------------|
| `random_forest` | Random Forest Classifier | `n_estimators: 100`, `max_depth: 10`, `min_samples_split: 2` |
| `xgboost` | XGBoost Classifier | `n_estimators: 100`, `max_depth: 6`, `learning_rate: 0.1` |

**⚠️ WICHTIG:** Nur diese beiden Modell-Typen sind aktuell implementiert und funktionsfähig!

---

## 2. Vorhersage-Typen

### 2.1 Klassische Vorhersage (Standard)

**Beschreibung:** Modell lernt, ob eine Variable zu einem bestimmten Zeitpunkt einen Schwellwert überschreitet/unterschreitet.

**Beispiel:** "Ist `market_cap_close` > 50000?"

**Erforderliche Parameter:**
- `target_var`: Ziel-Variable (z.B. `"market_cap_close"`)
- `operator`: Vergleichsoperator (`">"`, `"<"`, `">="`, `"<="`, `"="`)
- `target_value`: Schwellwert (z.B. `50000.0`)

**Label-Erstellung:**
- Für jede Zeile wird geprüft: `target_var operator target_value`
- Ergebnis: `1` (Bedingung erfüllt) oder `0` (nicht erfüllt)

### 2.2 Zeitbasierte Vorhersage

**Beschreibung:** Modell lernt, ob eine Variable innerhalb eines bestimmten Zeitraums (in Minuten) um einen bestimmten Prozentsatz steigt oder fällt.

**Beispiel:** "Steigt `price_close` in den nächsten 10 Minuten um mindestens 5%?"

**Erforderliche Parameter:**
- `use_time_based_prediction`: `true`
- `target_var`: Variable die überwacht wird (z.B. `"price_close"`)
- `future_minutes`: Anzahl Minuten in die Zukunft (z.B. `10`)
- `min_percent_change`: Mindest-Prozent-Änderung (z.B. `5.0` für 5%)
- `direction`: `"up"` (steigt) oder `"down"` (fällt)

**Label-Erstellung:**
- Für jede Zeile wird der Wert nach `future_minutes` Minuten berechnet
- Prozentuale Änderung: `((future_value - current_value) / current_value) * 100`
- Wenn `direction == "up"`: Label = `1` wenn Änderung >= `min_percent_change`, sonst `0`
- Wenn `direction == "down"`: Label = `1` wenn Änderung <= `-min_percent_change`, sonst `0`

**⚠️ WICHTIG - Phase-Intervalle:**
- Das System verwendet `interval_seconds` pro Phase aus `ref_coin_phases`
- Für jede Zeile wird basierend auf `phase_id_at_time` der korrekte `interval_seconds` verwendet
- `rows_to_shift = round(future_minutes / (interval_seconds / 60.0))`
- Fallback: Wenn Phase unbekannt oder `interval_seconds = 0`, wird Durchschnitts-Intervall verwendet

**⚠️ WICHTIG - Data Leakage Prevention:**
- Bei zeitbasierter Vorhersage wird `target_var` **NICHT** als Feature verwendet!
- `target_var` wird nur für Label-Erstellung benötigt, nicht für Training
- Dies verhindert unrealistisch hohe Accuracy-Werte (Data Leakage)

**Beispiel:**
- Phase 1: `interval_seconds = 60` → 1 Minute pro Zeile
- `future_minutes = 10` → `rows_to_shift = 10`
- Phase 2: `interval_seconds = 300` → 5 Minuten pro Zeile
- `future_minutes = 10` → `rows_to_shift = 2`

---

## 3. Verfügbare Features

### 3.1 Feature-Liste

Alle Features stammen aus der `coin_metrics` Tabelle:

| Feature | Beschreibung | Datentyp |
|---------|--------------|----------|
| `price_open` | Eröffnungspreis | NUMERIC |
| `price_high` | Höchstpreis | NUMERIC |
| `price_low` | Tiefstpreis | NUMERIC |
| `price_close` | Schlusspreis | NUMERIC |
| `volume_sol` | Volumen in SOL | NUMERIC |
| `volume_usd` | Volumen in USD | NUMERIC |
| `market_cap_open` | Market Cap bei Eröffnung | NUMERIC |
| `market_cap_high` | Höchstes Market Cap | NUMERIC |
| `market_cap_low` | Tiefstes Market Cap | NUMERIC |
| `market_cap_close` | Market Cap bei Schluss | NUMERIC |
| `order_buy_count` | Anzahl Buy-Orders | INT |
| `order_sell_count` | Anzahl Sell-Orders | INT |
| `order_buy_volume` | Buy-Order Volumen | NUMERIC |
| `order_sell_volume` | Sell-Order Volumen | NUMERIC |
| `whale_buy_count` | Anzahl Whale-Buys | INT |
| `whale_sell_count` | Anzahl Whale-Sells | INT |
| `whale_buy_volume` | Whale-Buy Volumen | NUMERIC |
| `whale_sell_volume` | Whale-Sell Volumen | NUMERIC |

**⚠️ WICHTIG:**
- Mindestens **1 Feature** muss ausgewählt werden
- Features werden als **JSONB Array** in der Datenbank gespeichert: `["price_open", "price_high", "price_low", "volume_sol"]`
- Die `target_var` wird automatisch zu den Features hinzugefügt, falls sie nicht bereits enthalten ist (für Label-Erstellung benötigt)
- **Bei zeitbasierter Vorhersage:** `target_var` wird **NICHT** als Feature verwendet (verhindert Data Leakage)
- **Feature-Engineering:** Zusätzliche ~40 Pump-Detection Features können automatisch erstellt werden (siehe Abschnitt 7)

---

## 4. Verfügbare Target-Variablen

### 4.1 Target-Variable-Liste

| Target-Variable | Beschreibung | Verwendung |
|-----------------|--------------|------------|
| `market_cap_close` | Market Cap bei Schluss | Klassische & Zeitbasierte Vorhersage |
| `price_close` | Schlusspreis | Klassische & Zeitbasierte Vorhersage |
| `volume_sol` | Volumen in SOL | Klassische & Zeitbasierte Vorhersage |
| `volume_usd` | Volumen in USD | Klassische & Zeitbasierte Vorhersage |

**⚠️ WICHTIG:**
- Bei **klassischer Vorhersage**: `target_var` ist erforderlich
- Bei **zeitbasierter Vorhersage**: `target_var` ist erforderlich (für welche Variable wird die prozentuale Änderung berechnet)

---

## 5. Coin-Phasen

### 5.1 Phasen-Dynamik

**Quelle:** `ref_coin_phases` Tabelle

**Felder:**
- `id`: Phase-ID (z.B. `1`, `2`, `3`)
- `name`: Phase-Name (z.B. `"Launch"`, `"Growth"`)
- `interval_seconds`: Intervall in Sekunden (z.B. `60`, `300`)

**Verwendung:**
- **Optional:** Wenn keine Phasen ausgewählt werden, werden **alle Phasen** verwendet
- **Filterung:** Nur Daten mit `phase_id_at_time` in der ausgewählten Liste werden verwendet
- **Zeitbasierte Vorhersage:** `interval_seconds` wird pro Phase verwendet, um `rows_to_shift` zu berechnen

**Speicherung:**
- Als **JSONB Array** in der Datenbank: `[1, 2, 3]`
- `null` wenn keine Phasen ausgewählt wurden (alle Phasen werden verwendet)

**⚠️ WICHTIG:**
- Phasen werden dynamisch aus `ref_coin_phases` geladen
- `interval_seconds` wird in der UI angezeigt: `"Phase 1 (60s)"`
- Bei zeitbasierter Vorhersage wird `interval_seconds` pro Phase verwendet (genauer als Durchschnitt!)

---

## 6. Hyperparameter

### 6.1 Random Forest Hyperparameter

| Parameter | Standard | Beschreibung | Bereich |
|-----------|----------|--------------|---------|
| `n_estimators` | `100` | Anzahl der Entscheidungsbäume | `1` - `∞` |
| `max_depth` | `10` | Maximale Tiefe der Bäume | `1` - `∞` |
| `min_samples_split` | `2` | Mindest-Samples zum Split | `2` - `∞` |
| `random_state` | `42` | Seed für Reproduzierbarkeit | `0` - `∞` |

**⚠️ WICHTIG:**
- Hyperparameter sind **optional**
- Wenn nicht angegeben, werden Standard-Werte verwendet
- Werden als **JSONB Object** gespeichert: `{"n_estimators": 100, "max_depth": 10}`

### 6.2 XGBoost Hyperparameter

| Parameter | Standard | Beschreibung | Bereich |
|-----------|----------|--------------|---------|
| `n_estimators` | `100` | Anzahl der Boosting-Runden | `1` - `∞` |
| `max_depth` | `6` | Maximale Tiefe der Bäume | `1` - `∞` |
| `learning_rate` | `0.1` | Lernrate | `0.01` - `1.0` |
| `random_state` | `42` | Seed für Reproduzierbarkeit | `0` - `∞` |

**⚠️ WICHTIG:**
- Hyperparameter sind **optional**
- Wenn nicht angegeben, werden Standard-Werte verwendet
- Werden als **JSONB Object** gespeichert: `{"n_estimators": 100, "max_depth": 6, "learning_rate": 0.1}`

### 6.3 Zeitbasierte Parameter (Intern)

Bei zeitbasierter Vorhersage werden zusätzliche Parameter in `params` gespeichert:

```json
{
  "n_estimators": 100,
  "max_depth": 10,
  "_time_based": {
    "enabled": true,
    "future_minutes": 10,
    "min_percent_change": 5.0,
    "direction": "up"
  }
}
```

**⚠️ WICHTIG:**
- `_time_based` wird in `params` gespeichert (nicht in separaten Spalten!)
- Wird beim Testen/Vergleichen verwendet, um die richtige Label-Erstellung zu wählen

---

## 7. Feature-Engineering (Optional)

### 7.1 Übersicht

**Beschreibung:** Automatische Erstellung von ~40 zusätzlichen Pump-Detection Features zur Verbesserung der Modell-Performance.

**Erwarteter Impact:** +5-10% Accuracy bei Pump-Detection

### 7.2 Erstellte Features

**Feature-Kategorien:**

1. **Price Momentum:**
   - `price_change_{window}`: Prozentuale Preisänderung über Zeitfenster
   - `price_roc_{window}`: Rate of Change (ROC)

2. **Volume Patterns:**
   - `volume_ratio_{window}`: Volumen vs. Rolling Average
   - `volume_spike_{window}`: Volumen-Spike (Standard Deviation)

3. **Buy/Sell Pressure:**
   - `buy_sell_ratio`: Buy-Order Volumen / Sell-Order Volumen
   - `buy_pressure`: Normalisierter Buy-Druck
   - `sell_pressure`: Normalisierter Sell-Druck

4. **Whale Activity:**
   - `whale_buy_sell_ratio`: Whale Buy / Whale Sell Ratio
   - `whale_activity_spike_{window}`: Whale-Activity Spike

5. **Price Volatility:**
   - `price_volatility_{window}`: Rolling Standard Deviation
   - `price_range_{window}`: High-Low Range

6. **Market Cap Velocity:**
   - `mcap_velocity_{window}`: Market Cap Änderungsrate

7. **Order Book Imbalance:**
   - `order_imbalance`: Buy-Orders vs. Sell-Orders

**Fenstergrößen (Standard):** `[5, 10, 15]` (konfigurierbar)

**Gesamtanzahl Features:** ~40 (abhängig von verfügbaren Basis-Features)

### 7.3 Aktivierung

**Parameter:**
- `use_engineered_features`: `true` / `false` (Default: `false`)
- `feature_engineering_windows`: `[5, 10, 15]` (Standard) oder benutzerdefiniert

**⚠️ WICHTIG:**
- Nur tatsächlich erstellte Features werden verwendet (dynamische Feature-Erstellung)
- Features werden automatisch zu `features`-Liste hinzugefügt
- Feature Importance zeigt auch engineered features

---

## 8. SMOTE - Imbalanced Data Handling (Optional)

### 8.1 Übersicht

**Beschreibung:** Automatische Behandlung von Label-Ungleichgewicht mit SMOTE (Synthetic Minority Over-sampling Technique) + RandomUnderSampler.

**Erwarteter Impact:** +3-5% Accuracy bei imbalanced Data

### 8.2 Funktionsweise

**Aktivierung:**
- Automatisch wenn Label-Balance < 30% oder > 70%
- Standardmäßig aktiviert (`use_smote: true`)

**Prozess:**
1. Label-Balance wird geprüft
2. Wenn unausgewogen:
   - SMOTE erhöht Minority-Klasse auf 50% der Majority-Klasse
   - RandomUnderSampler reduziert Majority-Klasse auf 80% der neuen Minority
3. Training mit ausgewogenen Daten

**Parameter:**
- `use_smote`: `true` / `false` (Default: `true`)

**⚠️ WICHTIG:**
- Edge-Case Handling: Wenn zu wenig positive Samples für SMOTE, wird Training ohne SMOTE fortgesetzt
- Logs zeigen Balance vor/nach SMOTE

---

## 9. TimeSeriesSplit - Cross-Validation (Optional)

### 9.1 Übersicht

**Beschreibung:** Verwendung von TimeSeriesSplit statt einfachem Train-Test-Split für realistischere Performance-Bewertung bei Zeitreihen.

**Erwarteter Impact:** Realistischere Metriken, Overfitting-Erkennung

### 9.2 Funktionsweise

**Aktivierung:**
- Standardmäßig aktiviert (`use_timeseries_split: true`)
- Fallback auf einfachen Train-Test-Split wenn deaktiviert

**Prozess:**
1. TimeSeriesSplit mit `n_splits` (Standard: 5)
2. Cross-Validation mit mehreren Metriken (accuracy, f1, precision, recall)
3. Overfitting-Check: Train-Test Gap > 10% → Warning
4. Finales Modell wird auf letztem Split trainiert

**Parameter:**
- `use_timeseries_split`: `true` / `false` (Default: `true`)
- `cv_splits`: Anzahl Splits (Default: `5`)

**Ergebnisse:**
- `cv_scores`: JSONB mit Train/Test Scores pro Split
- `cv_overfitting_gap`: Differenz zwischen Train- und Test-Accuracy

**⚠️ WICHTIG:**
- CV-Ergebnisse werden in DB gespeichert (`cv_scores`, `cv_overfitting_gap`)
- Overfitting-Warning erscheint in Logs wenn Gap > 10%

---

## 10. Modellerstellung über UI

### 10.1 Streamlit UI - Seite "Neues Modell trainieren"

**URL:** `http://localhost:8501` → Navigation: "➕ Neues Modell trainieren"

### 10.2 Formular-Felder

#### 10.2.1 Basis-Informationen

| Feld | Typ | Erforderlich | Beschreibung |
|------|-----|--------------|--------------|
| `Modell-Name` | Text-Input | ✅ Ja | Eindeutiger Name (z.B. `"PumpDetector_v1"`) |
| `Modell-Typ` | Selectbox | ✅ Ja | `"random_forest"` oder `"xgboost"` |
| `Beschreibung` | Text-Area | ❌ Nein | Optionale Beschreibung |

#### 10.2.2 Features

| Feld | Typ | Erforderlich | Beschreibung |
|------|-----|--------------|--------------|
| `Features auswählen` | Multiselect | ✅ Ja | Mindestens 1 Feature aus `AVAILABLE_FEATURES` |

**Standard-Auswahl:** `["price_open", "price_high", "price_low", "volume_sol"]`

#### 10.2.3 Coin-Phasen

| Feld | Typ | Erforderlich | Beschreibung |
|------|-----|--------------|--------------|
| `Phasen auswählen` | Multiselect | ❌ Nein | Phasen aus `ref_coin_phases` (Leer = alle Phasen) |

**Anzeige-Format:** `"Phase 1 (60s)"` oder `"Phase 1 - Launch (60s)"`

#### 10.2.4 Zeitbasierte Vorhersage (Optional)

**Checkbox:** "Zeitbasierte Vorhersage aktivieren" (außerhalb des Forms für sofortige UI-Reaktion)

**Wenn aktiviert:**
- **Target-Variable Sektion wird ausgeblendet**
- Neue Sektion: "⏰ Zeitbasierte Vorhersage - Konfiguration"

| Feld | Typ | Erforderlich | Beschreibung |
|------|-----|--------------|--------------|
| `Welche Variable wird überwacht?` | Selectbox | ✅ Ja | Aus `AVAILABLE_TARGETS` |
| `Zukunft (Minuten)` | Number-Input | ✅ Ja | `min_value: 1`, `step: 1` |
| `Min. Prozent-Änderung` | Number-Input | ✅ Ja | `min_value: 0.1`, `step: 0.5` |
| `Richtung` | Selectbox | ✅ Ja | `"up"` (Steigt) oder `"down"` (Fällt) |

**Wenn deaktiviert:**
- **Target-Variable Sektion wird angezeigt**

| Feld | Typ | Erforderlich | Beschreibung |
|------|-----|--------------|--------------|
| `Target-Variable` | Selectbox | ✅ Ja | Aus `AVAILABLE_TARGETS` |
| `Operator` | Selectbox | ✅ Ja | `">"`, `"<"`, `">="`, `"<="`, `"="` |
| `Target-Wert` | Number-Input | ✅ Ja | `min_value: 0.0`, `step: 1000.0` |

#### 10.2.5 Training-Zeitraum

| Feld | Typ | Erforderlich | Beschreibung |
|------|-----|--------------|--------------|
| `Start-Datum` | Date-Input | ✅ Ja | UTC-Zeitzone |
| `Start-Uhrzeit` | Time-Input | ✅ Ja | UTC-Zeitzone |
| `Ende-Datum` | Date-Input | ✅ Ja | UTC-Zeitzone |
| `Ende-Uhrzeit` | Time-Input | ✅ Ja | UTC-Zeitzone |

**⚠️ WICHTIG:**
- Datum und Uhrzeit werden zu `datetime` kombiniert
- Zeitzone wird auf `UTC` gesetzt: `.replace(tzinfo=timezone.utc)`
- Format: `ISO 8601` mit UTC (z.B. `"2024-01-01T00:00:00+00:00"`)

#### 10.2.6 Hyperparameter (Optional)

**Checkbox:** "Hyperparameter anpassen" (außerhalb des Forms für sofortige UI-Reaktion)

**Wenn aktiviert:**
- **Random Forest:**
  - `n_estimators`: Number-Input (`min_value: 1`, `step: 10`, `value: 100`)
  - `max_depth`: Number-Input (`min_value: 1`, `step: 1`, `value: 10`)
- **XGBoost:**
  - `n_estimators`: Number-Input (`min_value: 1`, `step: 10`, `value: 100`)
  - `max_depth`: Number-Input (`min_value: 1`, `step: 1`, `value: 6`)
  - `learning_rate`: Number-Input (`min_value: 0.01`, `max_value: 1.0`, `step: 0.01`, `value: 0.1`)

**Wenn deaktiviert:**
- Standard-Hyperparameter werden verwendet (aus `ref_model_types`)

#### 10.2.7 Feature-Engineering (Optional)

**Checkbox:** "Erweiterte Pump-Detection Features verwenden" (außerhalb des Forms für sofortige UI-Reaktion)

**Wenn aktiviert:**
- Info-Box: "Es werden automatisch ~40 zusätzliche Features erstellt (Momentum, Volumen-Patterns, Whale-Activity)"
- Multiselect für Fenstergrößen: `[3, 5, 10, 15, 20, 30]` (Standard: `[5, 10, 15]`)

**Wenn deaktiviert:**
- Nur ausgewählte Basis-Features werden verwendet

#### 10.2.8 SMOTE (Optional)

**Checkbox:** "⚖️ SMOTE für Imbalanced Data (empfohlen)" (außerhalb des Forms, Standard: aktiviert)

**Wenn aktiviert:**
- Info-Box: "SMOTE wird automatisch angewendet wenn Label-Balance < 30% oder > 70%"

#### 10.2.9 TimeSeriesSplit (Optional)

**Info-Box:** "ℹ️ TimeSeriesSplit wird standardmäßig verwendet für realistischere Performance-Bewertung"

**Optional:** Checkbox für `cv_splits` (Standard: 5)

### 10.3 Formular-Validierung (UI)

**Vor dem Absenden:**
1. ✅ `model_name` darf nicht leer sein
2. ✅ Mindestens 1 Feature muss ausgewählt sein
3. ✅ Wenn `use_time_based == False`: `target_var`, `operator`, `target_value` müssen gesetzt sein
4. ✅ Wenn `use_time_based == True`: `future_minutes`, `min_percent_change`, `direction` müssen gesetzt sein
5. ✅ `train_start` < `train_end`

### 10.4 API-Aufruf (UI → Backend)

**Endpoint:** `POST /api/models/create`

**Request Body:**
```json
{
  "name": "PumpDetector_v1",
  "model_type": "random_forest",
  "features": ["price_open", "price_high", "price_low", "volume_sol"],
  "phases": [1, 2, 3],
  "params": {"n_estimators": 100, "max_depth": 10},
  "train_start": "2024-01-01T00:00:00+00:00",
  "train_end": "2024-01-31T23:59:59+00:00",
  "description": "Optional description",
  "use_time_based_prediction": false,
  "target_var": "market_cap_close",
  "operator": ">",
  "target_value": 50000.0
}
```

**Oder für zeitbasierte Vorhersage mit allen Features:**
```json
{
  "name": "PricePumpDetector_v1",
  "model_type": "xgboost",
  "features": ["price_open", "price_high", "price_low", "volume_sol"],
  "phases": [1, 2],
  "params": {
    "n_estimators": 100,
    "max_depth": 6,
    "learning_rate": 0.1,
    "use_engineered_features": true,
    "feature_engineering_windows": [5, 10, 15],
    "use_smote": true,
    "use_timeseries_split": true,
    "cv_splits": 5
  },
  "train_start": "2024-01-01T00:00:00+00:00",
  "train_end": "2024-01-31T23:59:59+00:00",
  "description": "Predicts price pumps",
  "use_time_based_prediction": true,
  "future_minutes": 10,
  "min_percent_change": 5.0,
  "direction": "up",
  "target_var": "price_close"
}
```

**Response:**
```json
{
  "job_id": 123,
  "message": "Job erstellt. Modell 'PumpDetector_v1' wird trainiert.",
  "status": "PENDING"
}
```

**⚠️ WICHTIG:**
- Modell wird **NICHT sofort** erstellt!
- Ein **TRAIN-Job** wird in `ml_jobs` erstellt
- Job wird asynchron vom **Job Worker** verarbeitet
- Status kann über `/api/jobs/{job_id}` abgefragt werden

---

## 11. Modellerstellung über API

### 11.1 REST API Endpoint

**Endpoint:** `POST /api/models/create`

**Base URL:** `http://localhost:8000`

**Content-Type:** `application/json`

### 11.2 Request Schema

**Pydantic Schema:** `TrainModelRequest`

```python
{
  # Erforderlich
  "name": str,                    # Eindeutiger Modell-Name
  "model_type": str,              # "random_forest" oder "xgboost"
  "features": List[str],          # Mindestens 1 Feature
  "train_start": datetime,        # ISO-Format mit UTC
  "train_end": datetime,          # ISO-Format mit UTC
  
  # Optional
  "phases": Optional[List[int]],  # [1, 2, 3] oder null
  "params": Optional[Dict],      # Hyperparameter
  "description": Optional[str],   # Beschreibung
  
  # Zeitbasierte Vorhersage
  "use_time_based_prediction": bool,  # Default: false
  "future_minutes": Optional[int],     # Erforderlich wenn use_time_based_prediction=true
  "min_percent_change": Optional[float], # Erforderlich wenn use_time_based_prediction=true
  "direction": Optional[str],           # "up" oder "down" (Default: "up")
  
  # Klassische Vorhersage (Optional wenn use_time_based_prediction=true)
  "target_var": Optional[str],    # Erforderlich wenn use_time_based_prediction=false
  "operator": Optional[str],       # Erforderlich wenn use_time_based_prediction=false
  "target_value": Optional[float],  # Erforderlich wenn use_time_based_prediction=false
  
  # Feature-Engineering (Optional)
  "use_engineered_features": bool,  # Default: false
  "feature_engineering_windows": Optional[List[int]],  # Default: [5, 10, 15]
  
  # SMOTE (Optional)
  "use_smote": bool,  # Default: true
  
  # TimeSeriesSplit (Optional)
  "use_timeseries_split": bool,  # Default: true
  "cv_splits": Optional[int]  # Default: 5
}
```

### 11.3 Validierungen (API)

**Pydantic Validatoren:**

1. **`model_type`:** Muss `"random_forest"` oder `"xgboost"` sein
2. **`operator`:** Wenn gesetzt, muss `">"`, `"<"`, `">="`, `"<="`, `"="` sein
3. **`target_var`:** Erforderlich wenn `use_time_based_prediction=false`
4. **`operator`:** Erforderlich wenn `use_time_based_prediction=false`
5. **`target_value`:** Erforderlich wenn `use_time_based_prediction=false`
6. **`future_minutes`:** Erforderlich wenn `use_time_based_prediction=true`, muss `> 0` sein
7. **`min_percent_change`:** Erforderlich wenn `use_time_based_prediction=true`, muss `> 0` sein
8. **`direction`:** Muss `"up"` oder `"down"` sein
9. **`train_start`, `train_end`:** Werden zu UTC konvertiert (tz-aware)

### 11.4 Beispiel-Requests

#### 11.4.1 Klassische Vorhersage

```bash
curl -X POST "http://localhost:8000/api/models/create" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "MarketCapDetector_v1",
    "model_type": "random_forest",
    "features": ["price_open", "price_high", "price_low", "volume_sol"],
    "phases": [1, 2, 3],
    "train_start": "2024-01-01T00:00:00+00:00",
    "train_end": "2024-01-31T23:59:59+00:00",
    "target_var": "market_cap_close",
    "operator": ">",
    "target_value": 50000.0,
    "description": "Detects high market cap"
  }'
```

#### 11.4.2 Zeitbasierte Vorhersage mit allen Features

```bash
curl -X POST "http://localhost:8000/api/models/create" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "PricePumpDetector_v1",
    "model_type": "xgboost",
    "features": ["price_open", "price_high", "price_low", "volume_sol"],
    "phases": [1, 2],
    "params": {
      "n_estimators": 150,
      "max_depth": 8,
      "learning_rate": 0.05,
      "use_engineered_features": true,
      "feature_engineering_windows": [5, 10, 15],
      "use_smote": true,
      "use_timeseries_split": true,
      "cv_splits": 5
    },
    "train_start": "2024-01-01T00:00:00+00:00",
    "train_end": "2024-01-31T23:59:59+00:00",
    "use_time_based_prediction": true,
    "future_minutes": 10,
    "min_percent_change": 5.0,
    "direction": "up",
    "target_var": "price_close",
    "description": "Predicts price pumps in 10 minutes"
  }'
```

### 11.5 Response

**Schema:** `CreateJobResponse`

```json
{
  "job_id": 123,
  "message": "Job erstellt. Modell 'PumpDetector_v1' wird trainiert.",
  "status": "PENDING"
}
```

**Status-Codes:**
- `201 Created`: Job erfolgreich erstellt
- `400 Bad Request`: Validierungsfehler
- `500 Internal Server Error`: Server-Fehler

---

## 12. Backend-Logik

### 12.1 Job-Erstellung (API → Database)

**Datei:** `app/api/routes.py` → `create_model_job()`

**Ablauf:**
1. Request wird validiert (Pydantic)
2. Zeitbasierte Parameter werden in `params` gespeichert:
   ```python
   if request.use_time_based_prediction:
       final_params = {
           **request.params,
           "_time_based": {
               "enabled": True,
               "future_minutes": request.future_minutes,
               "min_percent_change": request.min_percent_change,
               "direction": request.direction
           }
       }
   ```
3. Job wird in `ml_jobs` erstellt:
   ```python
   job_id = await create_job(
       job_type="TRAIN",
       train_model_type=request.model_type,
       train_target_var=request.target_var,
       train_operator=request.operator,
       train_value=request.target_value,
       train_start=request.train_start,
       train_end=request.train_end,
       train_features=request.features,  # JSONB
       train_phases=request.phases,      # JSONB
       train_params=final_params,        # JSONB (enthält _time_based)
       progress_msg=request.name          # Name temporär gespeichert
   )
   ```
4. Response mit `job_id` wird zurückgegeben

**⚠️ WICHTIG:**
- Modell wird **NICHT sofort** erstellt!
- Job wird asynchron vom **Job Worker** verarbeitet

### 12.2 Job-Verarbeitung (Worker)

**Datei:** `app/queue/job_manager.py` → `process_train_job()`

**Ablauf:**
1. Job wird aus `ml_jobs` geladen
2. Modell-Name wird aus `progress_msg` geholt
3. Parameter werden extrahiert:
   - `model_type`, `target_var`, `target_operator`, `target_value`
   - `train_start`, `train_end`, `features`, `phases`, `params`
   - Zeitbasierte Parameter aus `params["_time_based"]` (falls vorhanden)
4. `train_model()` wird aufgerufen (async, nutzt intern `run_in_executor`)
5. Training-Ergebnis wird erhalten:
   - `accuracy`, `f1`, `precision`, `recall`
   - `feature_importance`
   - `model_path` (Pfad zur `.joblib` Datei)
6. Modell wird in `ml_models` erstellt:
   ```python
   model_id = await create_model(
       name=model_name,
       model_type=model_type,
       target_variable=target_var,
       train_start=train_start_dt,
       train_end=train_end_dt,
       target_operator=target_operator,  # Kann None sein
       target_value=target_value,        # Kann None sein
       features=features,
       phases=phases,
       params=params,  # Enthält _time_based wenn aktiviert
       training_accuracy=training_result['accuracy'],
       training_f1=training_result['f1'],
       training_precision=training_result['precision'],
       training_recall=training_result['recall'],
       feature_importance=training_result['feature_importance'],
       model_file_path=training_result['model_path'],
       status="READY"
   )
   ```
7. Job-Status wird auf `COMPLETED` gesetzt, `result_model_id` wird gesetzt

### 12.3 Training-Engine

**Datei:** `app/training/engine.py` → `train_model_sync()`

**Ablauf:**
1. **Daten laden:**
   ```python
   # Features für Laden: target_var wird hinzugefügt (für Labels benötigt)
   # Features für Training: target_var wird entfernt bei zeitbasierter Vorhersage (verhindert Data Leakage)
   features_for_loading, features_for_training = prepare_features_for_training(
       features=features,
       target_var=target_var,
       use_time_based=use_time_based
   )
   
   data = await load_training_data(
       train_start=train_start,
       train_end=train_end,
       features=features_for_loading,  # Enthält target_var (für Labels)
       phases=phases
   )
   ```
   - Lädt Daten aus `coin_metrics` mit `LIMIT 500000` (RAM-Management)
   - Filtert nach `timestamp` und `phase_id_at_time` (falls `phases` gesetzt)
   - Entfernt doppelte Timestamps
   - Setzt `timestamp` als Index

2. **Labels erstellen:**
   - **Wenn zeitbasierte Vorhersage:**
     ```python
     labels = create_time_based_labels(
         data,
         target_var,
         future_minutes,
         min_percent_change,
         direction,
         phase_intervals  # {phase_id: interval_seconds}
     )
     ```
   - **Wenn klassische Vorhersage:**
     ```python
     labels = create_labels(
         data,
         target_var,
         target_operator,
         target_value
     )
     ```

3. **Label-Balance prüfen:**
   - Positive Labels (`1`) und negative Labels (`0`) müssen vorhanden sein
   - Warnung wenn Ratio < 0.1

4. **Feature-Engineering (optional):**
   ```python
   if params.get('use_engineered_features', False):
       data = create_pump_detection_features(data, window_sizes=params.get('feature_engineering_windows', [5, 10, 15]))
       # Nur tatsächlich erstellte Features werden zu features-Liste hinzugefügt
       engineered_features = get_engineered_feature_names(window_sizes)
       features_for_training.extend(engineered_features)  # Dynamische Feature-Erstellung
   ```

5. **Features und Labels vorbereiten:**
   ```python
   X = data[features_for_training].values  # Features (target_var NICHT enthalten bei zeitbasierter Vorhersage!)
   y = labels.values                        # Labels
   ```

6. **Train-Test-Split oder TimeSeriesSplit:**
   ```python
   if params.get('use_timeseries_split', True):
       # TimeSeriesSplit mit Cross-Validation
       tscv = TimeSeriesSplit(n_splits=params.get('cv_splits', 5))
       cv_results = cross_validate(model, X, y, cv=tscv, ...)
       # Finales Modell auf letztem Split
   else:
       # Einfacher Train-Test-Split (Rückwärtskompatibilität)
       X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
   ```

7. **SMOTE (optional):**
   ```python
   if params.get('use_smote', True):
       # Prüfe Label-Balance
       if positive_ratio < 0.3 or positive_ratio > 0.7:
           # SMOTE + RandomUnderSampler Pipeline
           X_train, y_train = pipeline.fit_resample(X_train, y_train)
   ```

8. **Modell erstellen:**
   ```python
   model = create_model(model_type, params)  # Filtert _time_based aus params
   ```

9. **Training:**
   ```python
   model.fit(X_final_train, y_final_train)  # Finales Train-Set (aus TimeSeriesSplit oder Train-Test-Split)
   ```

10. **Evaluation:**
    ```python
    y_pred = model.predict(X_final_test)
    accuracy = accuracy_score(y_final_test, y_pred)
    f1 = f1_score(y_final_test, y_pred)
    precision = precision_score(y_final_test, y_pred)
    recall = recall_score(y_final_test, y_pred)
    
    # Zusätzliche Metriken
    roc_auc = roc_auc_score(y_final_test, model.predict_proba(X_final_test)[:, 1]) if hasattr(model, 'predict_proba') else None
    mcc = matthews_corrcoef(y_final_test, y_pred)
    cm = confusion_matrix(y_final_test, y_pred)
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    simulated_profit_pct = ((tp * 0.01) + (fp * -0.005)) / len(y_final_test) * 100
    ```

11. **Feature Importance:**
    ```python
    feature_importance = dict(zip(features_for_training, model.feature_importances_))  # Enthält auch engineered features
    ```

12. **Modell speichern:**
    ```python
    model_path = f"{model_storage_path}/model_{model_id}.joblib"
    joblib.dump(model, model_path)
    ```

13. **Ergebnis zurückgeben:**
    ```python
    return {
        "accuracy": accuracy,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "roc_auc": roc_auc,
        "mcc": mcc,
        "fpr": fpr,
        "fnr": fnr,
        "confusion_matrix": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
        "simulated_profit_pct": simulated_profit_pct,
        "cv_scores": cv_results if use_timeseries_split else None,
        "cv_overfitting_gap": train_test_gap if use_timeseries_split else None,
        "feature_importance": feature_importance,
        "features": features_for_training,  # Finale Feature-Liste (inkl. engineered)
        "model_path": model_path
    }
    ```

### 12.4 Feature Engineering

**Datei:** `app/training/feature_engineering.py`

#### 12.4.1 Daten laden (`load_training_data`)

**SQL Query:**
```sql
SELECT timestamp, {feature_list}, phase_id_at_time
FROM coin_metrics
WHERE timestamp >= $1 AND timestamp <= $2
  AND phase_id_at_time = ANY($3)  -- Falls phases gesetzt
ORDER BY timestamp
LIMIT $4  -- 500000 (RAM-Management)
```

**Verarbeitung:**
- Konvertiert zu UTC
- Entfernt doppelte Timestamps (`drop_duplicates`)
- Setzt `timestamp` als Index
- Gibt `DataFrame` zurück

#### 12.4.2 Labels erstellen - Klassisch (`create_labels`)

**Logik:**
```python
if operator == ">":
    labels = (values > target_value).astype(int)
elif operator == "<":
    labels = (values < target_value).astype(int)
# ... etc.
```

#### 12.4.3 Labels erstellen - Zeitbasiert (`create_time_based_labels`)

**Logik:**
1. Daten müssen nach `timestamp` sortiert sein
2. Für jede Zeile:
   - Hole `phase_id_at_time`
   - Berechne `rows_to_shift` basierend auf `interval_seconds`:
     ```python
     interval_minutes = interval_seconds / 60.0
     rows_to_shift = round(future_minutes / interval_minutes)
     ```
   - Hole zukünftigen Wert nach `rows_to_shift` Zeilen
   - Berechne prozentuale Änderung:
     ```python
     percent_change = ((future_value - current_value) / current_value) * 100
     ```
   - Erstelle Label:
     - Wenn `direction == "up"`: `1` wenn `percent_change >= min_percent_change`, sonst `0`
     - Wenn `direction == "down"`: `1` wenn `percent_change <= -min_percent_change`, sonst `0`

**⚠️ WICHTIG:**
- Verwendet `interval_seconds` pro Phase aus `ref_coin_phases` (genauer als Durchschnitt!)
- Fallback: Durchschnitts-Intervall wenn Phase unbekannt

---

## 13. Datenbank-Speicherung

### 13.1 Tabelle: `ml_models`

**Schema:**
```sql
CREATE TABLE ml_models (
    id BIGINT PRIMARY KEY,
    name VARCHAR(255) UNIQUE NOT NULL,
    model_type VARCHAR(50) NOT NULL,
    status VARCHAR(50) DEFAULT 'TRAINING',
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    is_deleted BOOLEAN DEFAULT FALSE,
    
    -- Training-Konfiguration
    target_variable VARCHAR(100) NOT NULL,
    target_operator VARCHAR(10),      -- Optional (NULL für zeitbasierte Vorhersage)
    target_value NUMERIC(20, 2),      -- Optional (NULL für zeitbasierte Vorhersage)
    train_start TIMESTAMP WITH TIME ZONE NOT NULL,
    train_end TIMESTAMP WITH TIME ZONE NOT NULL,
    
    -- JSONB Felder
    features JSONB NOT NULL,           -- ["price_open", "price_high", ...]
    phases JSONB,                       -- [1, 2, 3] oder NULL
    params JSONB,                       -- {"n_estimators": 100, "_time_based": {...}}
    feature_importance JSONB,           -- {"price_open": 0.35, ...}
    
    -- Performance-Metriken
    training_accuracy NUMERIC(5, 4),
    training_f1 NUMERIC(5, 4),
    training_precision NUMERIC(5, 4),
    training_recall NUMERIC(5, 4),
    
    -- Zusätzliche Metriken (Phase 9)
    roc_auc NUMERIC(5, 4),
    mcc NUMERIC(5, 4),
    fpr NUMERIC(5, 4),
    fnr NUMERIC(5, 4),
    confusion_matrix JSONB,
    simulated_profit_pct NUMERIC(10, 4),
    
    -- Cross-Validation (Phase 9)
    cv_scores JSONB,
    cv_overfitting_gap NUMERIC(5, 4),
    
    -- Zeitbasierte Vorhersage (Phase 8)
    future_minutes INT,
    price_change_percent NUMERIC(10, 4),
    target_direction VARCHAR(10),
    
    -- Sonstiges
    model_file_path TEXT,
    description TEXT
);
```

**Speicherung:**
- **JSONB Felder:** Werden als JSON-Strings gespeichert (`json.dumps()`)
- **Beim Lesen:** Werden zu Python-Objekten konvertiert (`json.loads()`)
- **Zeitbasierte Parameter:** Werden in `params["_time_based"]` gespeichert

**Beispiel-Datensatz:**
```json
{
  "id": 1,
  "name": "PricePumpDetector_v1",
  "model_type": "xgboost",
  "status": "READY",
  "target_variable": "price_close",
  "target_operator": null,
  "target_value": null,
  "train_start": "2024-01-01T00:00:00+00:00",
  "train_end": "2024-01-31T23:59:59+00:00",
  "features": ["price_open", "price_high", "price_low", "volume_sol"],
  "phases": [1, 2],
  "params": {
    "n_estimators": 100,
    "max_depth": 6,
    "learning_rate": 0.1,
    "_time_based": {
      "enabled": true,
      "future_minutes": 10,
      "min_percent_change": 5.0,
      "direction": "up"
    }
  },
  "feature_importance": {
    "price_open": 0.35,
    "price_high": 0.25,
    "price_low": 0.20,
    "volume_sol": 0.20
  },
  "training_accuracy": 0.8523,
  "training_f1": 0.7891,
  "training_precision": 0.8123,
  "training_recall": 0.7654,
  "roc_auc": 0.9123,
  "mcc": 0.7234,
  "fpr": 0.1234,
  "fnr": 0.2345,
  "confusion_matrix": {"tp": 1234, "tn": 5678, "fp": 890, "fn": 198},
  "simulated_profit_pct": 8.45,
  "cv_scores": {
    "train_accuracy": [0.85, 0.86, 0.87, 0.88, 0.89],
    "test_accuracy": [0.82, 0.83, 0.84, 0.85, 0.86],
    "train_f1": [0.78, 0.79, 0.80, 0.81, 0.82],
    "test_f1": [0.75, 0.76, 0.77, 0.78, 0.79]
  },
  "cv_overfitting_gap": 0.03,
  "future_minutes": 10,
  "price_change_percent": 5.0,
  "target_direction": "up",
  "model_file_path": "/app/models/model_1.joblib",
  "description": "Predicts price pumps in 10 minutes"
}
```

### 13.2 Tabelle: `ml_jobs`

**Schema:**
```sql
CREATE TABLE ml_jobs (
    id BIGINT PRIMARY KEY,
    job_type VARCHAR(20) NOT NULL,  -- 'TRAIN', 'TEST', 'COMPARE'
    status VARCHAR(20) DEFAULT 'PENDING',
    priority INT DEFAULT 5,
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    started_at TIMESTAMP WITH TIME ZONE,
    completed_at TIMESTAMP WITH TIME ZONE,
    
    progress NUMERIC(3, 2) DEFAULT 0.0,
    progress_msg TEXT,
    error_msg TEXT,
    
    -- TRAIN-spezifische Felder
    train_target_var VARCHAR(100),
    train_operator VARCHAR(10),
    train_value NUMERIC(20, 2),
    train_start TIMESTAMP WITH TIME ZONE,
    train_end TIMESTAMP WITH TIME ZONE,
    train_features JSONB,
    train_phases JSONB,
    train_params JSONB,
    train_model_type VARCHAR(50),
    result_model_id BIGINT REFERENCES ml_models(id),
    
    -- TEST-spezifische Felder
    test_model_id BIGINT REFERENCES ml_models(id),
    test_start TIMESTAMP WITH TIME ZONE,
    test_end TIMESTAMP WITH TIME ZONE,
    result_test_id BIGINT REFERENCES ml_test_results(id),
    
    -- COMPARE-spezifische Felder
    compare_model_a_id BIGINT REFERENCES ml_models(id),
    compare_model_b_id BIGINT REFERENCES ml_models(id),
    compare_start TIMESTAMP WITH TIME ZONE,
    compare_end TIMESTAMP WITH TIME ZONE,
    result_comparison_id BIGINT REFERENCES ml_comparisons(id)
);
```

**Speicherung:**
- **JSONB Felder:** `train_features`, `train_phases`, `train_params` (wie in `ml_models`)
- **Zeitbasierte Parameter:** In `train_params["_time_based"]`

### 13.3 JSONB-Konvertierung

**Beim Schreiben (Python → PostgreSQL):**
```python
import json
features_jsonb = json.dumps(features)  # ["price_open", ...] → '["price_open", ...]'
params_jsonb = json.dumps(params)       # {...} → '{...}'
```

**Beim Lesen (PostgreSQL → Python):**
```python
import json
if isinstance(model_dict['features'], str):
    model_dict['features'] = json.loads(model_dict['features'])
```

**⚠️ WICHTIG:**
- `asyncpg` konvertiert JSONB automatisch zu Python-Objekten beim Lesen
- Beim Schreiben muss explizit `json.dumps()` verwendet werden

---

## 14. Validierungen & Bedingungen

### 14.1 UI-Validierungen

**Vor Formular-Absenden:**
1. ✅ `model_name` darf nicht leer sein
2. ✅ Mindestens 1 Feature muss ausgewählt sein
3. ✅ Wenn `use_time_based == False`:
   - `target_var` muss gesetzt sein
   - `operator` muss gesetzt sein
   - `target_value` muss gesetzt sein
4. ✅ Wenn `use_time_based == True`:
   - `future_minutes` muss > 0 sein
   - `min_percent_change` muss > 0 sein
   - `direction` muss `"up"` oder `"down"` sein
   - `target_var` muss gesetzt sein (für welche Variable wird Änderung berechnet)
5. ✅ `train_start` < `train_end`

### 14.2 API-Validierungen (Pydantic)

**Request-Validierung:**
1. ✅ `model_type` muss `"random_forest"` oder `"xgboost"` sein
2. ✅ `features` muss mindestens 1 Element enthalten
3. ✅ `operator` (wenn gesetzt) muss `">"`, `"<"`, `">="`, `"<="`, `"="` sein
4. ✅ `target_var` ist erforderlich wenn `use_time_based_prediction=false`
5. ✅ `operator` ist erforderlich wenn `use_time_based_prediction=false`
6. ✅ `target_value` ist erforderlich wenn `use_time_based_prediction=false`
7. ✅ `future_minutes` ist erforderlich wenn `use_time_based_prediction=true`, muss `> 0` sein
8. ✅ `min_percent_change` ist erforderlich wenn `use_time_based_prediction=true`, muss `> 0` sein
9. ✅ `direction` muss `"up"` oder `"down"` sein
10. ✅ `train_start`, `train_end` werden zu UTC konvertiert

### 14.3 Backend-Validierungen

**Während Training:**
1. ✅ Daten müssen vorhanden sein (`len(data) > 0`)
2. ✅ `target_var` muss in Daten vorhanden sein
3. ✅ Labels müssen ausgewogen sein (mindestens 1 positive und 1 negative Label)
4. ✅ Warnung wenn Label-Balance < 0.1

**Fehlerbehandlung:**
- `ValueError` wenn Validierung fehlschlägt
- Job-Status wird auf `FAILED` gesetzt
- Fehlermeldung wird in `error_msg` gespeichert

---

## 15. Job-Verarbeitung

### 15.1 Job-Lifecycle

**Status-Übergänge:**
1. `PENDING` → Job erstellt, wartet auf Verarbeitung
2. `RUNNING` → Job wird verarbeitet
3. `COMPLETED` → Job erfolgreich abgeschlossen
4. `FAILED` → Job fehlgeschlagen (Fehler in `error_msg`)

### 15.2 Job-Worker

**Datei:** `app/queue/job_manager.py`

**Ablauf:**
1. Worker prüft regelmäßig auf `PENDING` Jobs (`JOB_POLL_INTERVAL`)
2. Job mit höchster `priority` und ältestem `created_at` wird ausgewählt
3. `process_job()` wird aufgerufen
4. Je nach `job_type`:
   - `TRAIN` → `process_train_job()`
   - `TEST` → `process_test_job()`
   - `COMPARE` → `process_compare_job()`
5. Status wird aktualisiert (`RUNNING` → `COMPLETED` oder `FAILED`)

### 15.3 Asynchrone Verarbeitung

**⚠️ WICHTIG:**
- Training ist **CPU-bound** und blockiert den Event Loop nicht
- `train_model()` nutzt intern `asyncio.run_in_executor()` für `model.fit()`
- Job-Worker läuft in separatem asyncio-Task

### 15.4 Job-Status abfragen

**Endpoint:** `GET /api/jobs/{job_id}`

**Response:**
```json
{
  "id": 123,
  "job_type": "TRAIN",
  "status": "COMPLETED",
  "progress": 1.0,
  "progress_msg": "Modell PumpDetector_v1 erfolgreich erstellt",
  "result_model": {
    "id": 1,
    "name": "PumpDetector_v1",
    "status": "READY",
    "training_accuracy": 0.8523,
    ...
  }
}
```

---

## 📝 Zusammenfassung

### ✅ Alle Möglichkeiten zur Modellerstellung:

1. **Modell-Typen:**
   - `random_forest` (Standard: `n_estimators=100`, `max_depth=10`)
   - `xgboost` (Standard: `n_estimators=100`, `max_depth=6`, `learning_rate=0.1`)

2. **Vorhersage-Typen:**
   - **Klassisch:** Variable `operator` Schwellwert (z.B. `market_cap_close > 50000`)
   - **Zeitbasiert:** Variable steigt/fällt in X Minuten um X% (z.B. `price_close` steigt in 10 Min um 5%)

3. **Features:**
   - 18 verfügbare Features aus `coin_metrics`
   - Mindestens 1 Feature erforderlich

4. **Target-Variablen:**
   - 4 verfügbare Targets: `market_cap_close`, `price_close`, `volume_sol`, `volume_usd`

5. **Coin-Phasen:**
   - Dynamisch aus `ref_coin_phases` geladen
   - Optional (Leer = alle Phasen)
   - `interval_seconds` wird für zeitbasierte Vorhersage verwendet

6. **Hyperparameter:**
   - Optional (Standard-Werte werden verwendet)
   - Random Forest: `n_estimators`, `max_depth`
   - XGBoost: `n_estimators`, `max_depth`, `learning_rate`

7. **Training-Zeitraum:**
   - Genauer Zeitpunkt (Datum + Uhrzeit) in UTC
   - `train_start` < `train_end`

### ✅ Erstellungswege:

1. **Streamlit UI:**
   - Formular mit allen Feldern
   - Validierung vor Absenden
   - Asynchrone Job-Erstellung
   - Feature-Engineering, SMOTE, TimeSeriesSplit Checkboxen

2. **REST API:**
   - `POST /api/models/create`
   - JSON Request Body
   - Pydantic-Validierung
   - Unterstützt alle neuen Parameter (Feature-Engineering, SMOTE, TimeSeriesSplit)

### ✅ Backend-Logik:

1. **Job-Erstellung:** API erstellt Job in `ml_jobs`
2. **Job-Verarbeitung:** Worker verarbeitet Job asynchron
3. **Training:** 
   - Daten laden (mit Data Leakage Prevention)
   - Labels erstellen
   - Feature-Engineering (optional)
   - SMOTE (optional)
   - TimeSeriesSplit oder Train-Test-Split
   - Modell trainieren
   - Zusätzliche Metriken berechnen
   - Speichern
4. **Speicherung:** Modell in `ml_models`, Datei in `/app/models/`

### ✅ Datenbank-Speicherung:

1. **`ml_models`:** 
   - Modell-Konfiguration, Metriken, Feature Importance
   - Zusätzliche Metriken: ROC-AUC, MCC, FPR, FNR, Confusion Matrix, Profit-Simulation
   - Cross-Validation: CV-Scores, Overfitting-Gap
   - Zeitbasierte Vorhersage: future_minutes, price_change_percent, target_direction
2. **`ml_jobs`:** Job-Status, Parameter, Ergebnisse
3. **JSONB:** Features, Phasen, Params, Feature Importance, CV-Scores, Confusion Matrix als JSONB

### ✅ Neue Features (Phase 9):

1. **Feature-Engineering:** ~40 zusätzliche Pump-Detection Features
2. **SMOTE:** Automatische Behandlung von Label-Ungleichgewicht
3. **TimeSeriesSplit:** Realistischere Cross-Validation für Zeitreihen
4. **Zusätzliche Metriken:** ROC-AUC, MCC, FPR, FNR, Confusion Matrix, Profit-Simulation
5. **Data Leakage Prevention:** target_var wird bei zeitbasierter Vorhersage aus Features entfernt

---

## 🔍 Prüf-Checkliste

Verwende diese Checkliste, um zu prüfen, ob alle benötigten Funktionen vorhanden sind:

- [ ] Modell-Typen: `random_forest`, `xgboost` ✅
- [ ] Vorhersage-Typen: Klassisch, Zeitbasiert ✅
- [ ] Features: 18 verfügbare Features ✅
- [ ] Target-Variablen: 4 verfügbare Targets ✅
- [ ] Coin-Phasen: Dynamisch geladen, optional ✅
- [ ] Hyperparameter: Optional, anpassbar ✅
- [ ] Training-Zeitraum: Genauer Zeitpunkt (Datum + Uhrzeit) ✅
- [ ] UI: Vollständiges Formular mit Validierung ✅
- [ ] API: REST Endpoint mit Validierung ✅
- [ ] Backend: Asynchrone Job-Verarbeitung ✅
- [ ] Datenbank: JSONB-Speicherung ✅
- [ ] Zeitbasierte Vorhersage: Phase-Intervalle berücksichtigt ✅
- [ ] Feature-Engineering: ~40 zusätzliche Features ✅
- [ ] SMOTE: Imbalanced Data Handling ✅
- [ ] TimeSeriesSplit: Cross-Validation ✅
- [ ] Zusätzliche Metriken: ROC-AUC, MCC, FPR, FNR, Confusion Matrix, Profit-Simulation ✅
- [ ] Data Leakage Prevention: target_var wird bei zeitbasierter Vorhersage entfernt ✅

---

**Erstellt:** 2024-01-XX  
**Aktualisiert:** 2024-12-23  
**Version:** 2.0  
**Status:** ✅ Vollständig dokumentiert (inkl. Phase 9 Verbesserungen)

