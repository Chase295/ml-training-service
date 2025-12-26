# 📚 Komplette KI-Modell Anleitung - Wie Modelle erstellt und getestet werden

**Datum:** 2025-12-24  
**Ziel:** Vollständiges Verständnis des gesamten Workflows von Modell-Erstellung bis Testing

---

## 📋 Inhaltsverzeichnis

1. [Übersicht - Der komplette Workflow](#1-übersicht---der-komplette-workflow)
2. [Modell-Erstellung - Schritt für Schritt](#2-modell-erstellung---schritt-für-schritt)
3. [Modell-Testing - Schritt für Schritt](#3-modell-testing---schritt-für-schritt)
4. [Datenfluss-Diagramme](#4-datenfluss-diagramme)
5. [Technische Details](#5-technische-details)
6. [Code-Referenzen](#6-code-referenzen)

---

## 1. Übersicht - Der komplette Workflow

### 1.1 Architektur-Übersicht

```
┌─────────────────┐
│  Streamlit UI   │  ← Benutzer-Interface
└────────┬────────┘
         │ HTTP POST /api/models/create
         ▼
┌─────────────────┐
│  FastAPI Routes  │  ← API-Endpunkte
└────────┬────────┘
         │ Erstellt Job in ml_jobs
         ▼
┌─────────────────┐
│  Job Manager    │  ← Asynchroner Worker
│  (Worker Loop)  │     (läuft kontinuierlich)
└────────┬────────┘
         │ Verarbeitet PENDING Jobs
         ▼
┌─────────────────┐
│ Training Engine │  ← CPU-bound Training
│  (train_model)  │     (läuft in run_in_executor)
└────────┬────────┘
         │ Speichert Modell
         ▼
┌─────────────────┐
│  ml_models DB   │  ← Modell-Metadaten
│  + .pkl Datei   │     + trainiertes Modell
└─────────────────┘
```

### 1.2 Wichtige Konzepte

**1. Asynchrones Job-System:**
- Modelle werden **NICHT sofort** erstellt
- Jobs werden in `ml_jobs` Tabelle eingereiht
- Worker verarbeitet Jobs asynchron (alle 5 Sekunden)
- Max. 2 Jobs parallel (konfigurierbar)

**2. CPU-bound Training:**
- Training läuft in `run_in_executor` (separater Thread)
- Blockiert **NICHT** den Event Loop
- Kann mehrere Minuten dauern

**3. Zwei Arten von Vorhersagen:**
- **Klassische Vorhersage:** "Ist price_close > 50000?"
- **Zeitbasierte Vorhersage:** "Steigt price_close in 10 Minuten um 30%?"

**4. Feature-Engineering:**
- Optional: Erstellt ~40 zusätzliche Features aus Basis-Features
- Aktivierbar über `use_engineered_features` Flag

---

## 2. Modell-Erstellung - Schritt für Schritt

### 2.1 Schritt 1: API Request (Web UI → FastAPI)

**Datei:** `app/api/routes.py` → `create_model_job()` (Zeile 42-114)

**Request-Format:**
```json
{
  "name": "PumpDetector_v1",
  "model_type": "random_forest",  // oder "xgboost"
  "train_start": "2024-01-01T00:00:00Z",
  "train_end": "2024-01-07T00:00:00Z",
  "features": ["price_open", "price_high", "price_low", "price_close", "volume_sol"],
  "phases": [1, 2],  // Optional: Nur bestimmte Coin-Phasen
  "use_time_based_prediction": true,  // NEU: Zeitbasierte Vorhersage
  "target_var": "price_close",
  "future_minutes": 10,
  "min_percent_change": 30.0,
  "direction": "up",  // "up" oder "down"
  "params": {
    "n_estimators": 100,
    "max_depth": 10
  },
  "use_engineered_features": true,  // NEU: Feature-Engineering
  "feature_engineering_windows": [5, 10, 15],  // NEU: Fenstergrößen
  "use_smote": true,  // NEU: Imbalanced Data Handling
  "use_timeseries_split": true,  // NEU: TimeSeriesSplit für CV
  "cv_splits": 5  // NEU: Anzahl CV-Splits
}
```

**Was passiert:**
1. Request wird validiert (Pydantic Schema)
2. Zeitbasierte Parameter werden in `params._time_based` gespeichert
3. Feature-Engineering Parameter werden in `params` gespeichert
4. Job wird in `ml_jobs` erstellt mit `status='PENDING'`
5. Modell-Name wird in `progress_msg` temporär gespeichert
6. Response mit `job_id` wird zurückgegeben

**Code-Referenz:**
```42:114:app/api/routes.py
@router.post("/models/create", response_model=CreateJobResponse, status_code=status.HTTP_201_CREATED)
async def create_model_job(request: TrainModelRequest):
    # ... Job wird erstellt ...
```

### 2.2 Schritt 2: Job wird in Datenbank gespeichert

**Datei:** `app/database/models.py` → `create_job()`

**Tabelle:** `ml_jobs`

**Gespeicherte Felder:**
- `job_type`: "TRAIN"
- `status`: "PENDING"
- `train_model_type`: "random_forest" oder "xgboost"
- `train_target_var`: "price_close" (oder None bei zeitbasierter Vorhersage)
- `train_operator`: ">" (oder None bei zeitbasierter Vorhersage)
- `train_value`: 50000.0 (oder None bei zeitbasierter Vorhersage)
- `train_start`: Timestamp (UTC)
- `train_end`: Timestamp (UTC)
- `train_features`: JSONB Array (z.B. `["price_open", "price_high"]`)
- `train_phases`: JSONB Array (z.B. `[1, 2]`) oder NULL
- `train_params`: JSONB Object (enthält Hyperparameter + interne Flags)
- `progress_msg`: Modell-Name (temporär, wird später überschrieben)
- `train_future_minutes`: 10 (bei zeitbasierter Vorhersage)
- `train_price_change_percent`: 30.0 (bei zeitbasierter Vorhersage)
- `train_target_direction`: "up" (bei zeitbasierter Vorhersage)

**Wichtig:**
- Job wird **sofort** erstellt (nicht das Modell!)
- Status ist `PENDING` (wartet auf Worker)

### 2.3 Schritt 3: Job Worker findet Job

**Datei:** `app/queue/job_manager.py` → `start_worker()` (Zeile 558-599)

**Was passiert:**
1. Worker-Loop läuft kontinuierlich (alle `JOB_POLL_INTERVAL` Sekunden, Standard: 5s)
2. Prüft auf `PENDING` Jobs in `ml_jobs`
3. Wenn Job gefunden UND weniger als `MAX_CONCURRENT_JOBS` aktiv:
   - Status wird auf `RUNNING` gesetzt
   - Job wird asynchron verarbeitet (`process_job()`)

**Code-Referenz:**
```558:599:app/queue/job_manager.py
async def start_worker() -> None:
    """Worker-Loop: Prüft regelmäßig auf neue Jobs"""
    while True:
        if len(active_tasks) < MAX_CONCURRENT_JOBS:
            job = await get_next_pending_job()
            if job:
                task = asyncio.create_task(process_job(job['id']))
                # ...
```

### 2.4 Schritt 4: Job-Verarbeitung startet

**Datei:** `app/queue/job_manager.py` → `process_train_job()` (Zeile 89-275)

**Ablauf:**

#### 4.1 Modell-Name extrahieren
```python
model_name = job.get('progress_msg') or f"Model_{job_id}"
```
- Name wurde beim Job-Erstellen in `progress_msg` gespeichert
- Wird später für Modell-Speicherung verwendet

#### 4.2 Parameter extrahieren
```python
target_var = job['train_target_var']  # Kann None sein
target_operator = job['train_operator']  # Kann None sein
target_value = float(job['train_value']) if job['train_value'] else None
train_start = job['train_start']
train_end = job['train_end']
params = job['train_params']  # JSONB → Python Dict
```

#### 4.3 Zeitbasierte Parameter extrahieren
```python
if params and "_time_based" in params:
    time_based_config = params.get("_time_based", {})
    use_time_based = time_based_config.get("enabled", False)
    future_minutes = time_based_config.get("future_minutes")
    min_percent_change = time_based_config.get("min_percent_change")
    direction = time_based_config.get("direction", "up")
```

#### 4.4 Progress Update (10%)
```python
await update_job_status(job_id, status="RUNNING", progress=0.1, 
                       progress_msg="Lade Trainingsdaten...")
```

### 2.5 Schritt 5: Training wird gestartet

**Datei:** `app/training/engine.py` → `train_model()` (Zeile 479-586)

**Ablauf:**

#### 5.1 Default-Parameter laden
```python
default_params = await get_model_type_defaults(model_type)
final_params = {**default_params, **(params or {})}
```
- Lädt Standard-Hyperparameter aus `ml_model_type_defaults` Tabelle
- Überschreibt mit übergebenen Parametern

#### 5.2 Features vorbereiten (verhindert Data Leakage)
```python
features_for_loading, features_for_training = prepare_features_for_training(
    features=features,
    target_var=target_var,
    use_time_based=use_time_based
)
```

**Wichtig bei zeitbasierter Vorhersage:**
- `features_for_loading`: Enthält `target_var` (für Labels benötigt)
- `features_for_training`: Enthält `target_var` **NICHT** (verhindert Data Leakage!)

**Code-Referenz:**
```95:128:app/training/engine.py
def prepare_features_for_training(
    features: List[str],
    target_var: Optional[str],
    use_time_based: bool
) -> tuple[List[str], List[str]]:
    # Bei zeitbasierter Vorhersage: target_var wird aus Features entfernt!
```

#### 5.3 Trainingsdaten laden
```python
data = await load_training_data(
    train_start=train_start,
    train_end=train_end,
    features=features_for_loading,  # Enthält target_var
    phases=phases
)
```

**Datei:** `app/training/feature_engineering.py` → `load_training_data()` (Zeile 50-131)

**SQL Query:**
```sql
SELECT timestamp, price_open, price_high, price_low, price_close, 
       volume_sol, market_cap_close, phase_id_at_time
FROM coin_metrics
WHERE timestamp >= $1 AND timestamp <= $2
  AND phase_id_at_time = ANY($3)  -- Falls Phasen gefiltert
ORDER BY timestamp
LIMIT 500000  -- ⚠️ RAM-Management: Max 500k Zeilen!
```

**Was passiert:**
1. Daten werden aus `coin_metrics` geladen
2. Nach `timestamp` sortiert (wichtig für zeitbasierte Labels!)
3. `timestamp` wird als Index gesetzt
4. Duplikate werden entfernt
5. Max. 500.000 Zeilen (RAM-Management)

**Code-Referenz:**
```50:131:app/training/feature_engineering.py
async def load_training_data(
    train_start: str | datetime,
    train_end: str | datetime,
    features: List[str],
    phases: Optional[List[int]] = None
) -> pd.DataFrame:
    # ... SQL Query ...
```

### 2.6 Schritt 6: Labels erstellen

**Datei:** `app/training/feature_engineering.py`

#### 6.1 Zeitbasierte Labels (wenn aktiviert)

**Funktion:** `create_time_based_labels()` (Zeile 177-316)

**Was passiert:**
1. Für jede Zeile wird der zukünftige Wert berechnet
2. Berechnung basiert auf `phase_id_at_time` und `interval_seconds` aus `ref_coin_phases`
3. Beispiel: Phase 1 hat `interval_seconds=5` → 10 Minuten = 120 Zeilen
4. Prozentuale Änderung wird berechnet: `((future_value - current_value) / current_value) * 100`
5. Label = 1 wenn Änderung >= `min_percent_change` (bei "up") oder <= `-min_percent_change` (bei "down")

**Code-Beispiel:**
```python
# Für jede Zeile:
current_value = data.loc[idx, target_variable]
future_idx = idx + rows_to_shift  # Basierend auf Phase-Intervall
future_value = data.loc[future_idx, target_variable]

percent_change = ((future_value - current_value) / current_value) * 100

if direction == "up":
    label = 1 if percent_change >= min_percent_change else 0
else:
    label = 1 if percent_change <= -min_percent_change else 0
```

**Code-Referenz:**
```177:316:app/training/feature_engineering.py
def create_time_based_labels(
    data: pd.DataFrame,
    target_variable: str,
    future_minutes: int,
    min_percent_change: float,
    direction: str = "up",
    phase_intervals: Optional[Dict[int, int]] = None
) -> pd.Series:
    # ... Berechnet Labels für zeitbasierte Vorhersage ...
```

#### 6.2 Klassische Labels (wenn zeitbasierte Vorhersage deaktiviert)

**Funktion:** `create_labels()` (Zeile 133-175)

**Was passiert:**
1. Für jede Zeile wird geprüft: `target_variable operator target_value`
2. Beispiel: `price_close > 50000` → Label = 1 wenn True, 0 wenn False

**Code-Referenz:**
```133:175:app/training/feature_engineering.py
def create_labels(
    data: pd.DataFrame,
    target_variable: str,
    target_operator: str,
    target_value: float
) -> pd.Series:
    # ... Erstellt binäre Labels ...
```

### 2.7 Schritt 7: Feature-Engineering (optional)

**Datei:** `app/training/feature_engineering.py` → `create_pump_detection_features()` (Zeile 318-427)

**Wird nur ausgeführt wenn:** `params.get('use_engineered_features', False) == True`

**Was passiert:**
1. Zusätzliche Features werden im DataFrame erstellt:
   - **Price Momentum:** `price_change_5`, `price_change_10`, `price_roc_5`, etc.
   - **Volume Patterns:** `volume_ratio_5`, `volume_spike_10`, etc.
   - **Buy/Sell Pressure:** `buy_sell_ratio`, `buy_pressure`, `sell_pressure`
   - **Whale Activity:** `whale_buy_sell_ratio`, `whale_activity_spike_5`, etc.
   - **Price Volatility:** `price_volatility_5`, `price_range_10`, etc.
   - **Market Cap Velocity:** `mcap_velocity_5`, etc.
   - **Order Book Imbalance:** `order_imbalance`

2. Features werden zu `features`-Liste hinzugefügt
3. Aus ~6 Basis-Features werden ~40 erweiterte Features

**Code-Referenz:**
```318:427:app/training/feature_engineering.py
def create_pump_detection_features(
    data: pd.DataFrame,
    window_sizes: list = [5, 10, 15]
) -> pd.DataFrame:
    # ... Erstellt ~40 zusätzliche Features ...
```

### 2.8 Schritt 8: Cross-Validation (optional)

**Datei:** `app/training/engine.py` → `train_model_sync()` (Zeile 240-275)

**Wird nur ausgeführt wenn:** `params.get('use_timeseries_split', True) == True`

**Was passiert:**
1. **TimeSeriesSplit** wird erstellt (respektiert zeitliche Reihenfolge)
2. Anzahl Splits: `params.get('cv_splits', 5)`
3. Cross-Validation wird durchgeführt:
   ```python
   cv_results = cross_validate(
       estimator=temp_model,
       X=X,
       y=y,
       cv=TimeSeriesSplit(n_splits=5),
       scoring=['accuracy', 'f1', 'precision', 'recall'],
       return_train_score=True
   )
   ```
4. Ergebnisse werden gespeichert:
   - `cv_scores`: Dict mit allen CV-Ergebnissen
   - `cv_overfitting_gap`: Train-Test Accuracy Gap

**Code-Referenz:**
```240:275:app/training/engine.py
if use_timeseries_split:
    from sklearn.model_selection import TimeSeriesSplit, cross_validate
    tscv = TimeSeriesSplit(n_splits=n_splits)
    cv_results = cross_validate(...)
```

### 2.9 Schritt 9: Train-Test-Split

**Datei:** `app/training/engine.py` → `train_model_sync()` (Zeile 282-299)

**Wenn TimeSeriesSplit aktiviert:**
- Verwendet letzten Split für finales Test-Set
- Train-Set: Alle Daten bis zum letzten Split
- Test-Set: Letzter Split

**Wenn TimeSeriesSplit deaktiviert:**
- Einfacher Train-Test-Split (80/20)
- **⚠️ Nicht empfohlen für Zeitreihen!**

### 2.10 Schritt 10: SMOTE (Imbalanced Data Handling)

**Datei:** `app/training/engine.py` → `train_model_sync()` (Zeile 301-361)

**Wird nur ausgeführt wenn:** 
- `params.get('use_smote', True) == True` UND
- Label-Balance < 30% oder > 70%

**Was passiert:**
1. Label-Balance wird geprüft
2. Wenn unausgewogen:
   - **SMOTE** erhöht Minority-Klasse (synthetische Beispiele)
   - **Random Under-Sampling** reduziert Majority-Klasse
   - Ziel: ~50/50 Balance

**Code-Referenz:**
```301:361:app/training/engine.py
if use_smote:
    positive_ratio = y_final_train.sum() / len(y_final_train)
    if positive_ratio < 0.3 or positive_ratio > 0.7:
        # SMOTE anwenden ...
```

### 2.11 Schritt 11: Modell-Training (CPU-bound!)

**Datei:** `app/training/engine.py` → `train_model_sync()` (Zeile 363-367)

**Was passiert:**
1. Modell-Instanz wird erstellt:
   ```python
   model = create_model(model_type, params)
   # RandomForestClassifier oder XGBClassifier
   ```

2. Training wird durchgeführt:
   ```python
   model.fit(X_final_train, y_final_train)
   ```
   - **⚠️ CPU-bound!** Blockiert Event Loop
   - Läuft in `run_in_executor` (separater Thread)

**Code-Referenz:**
```32:93:app/training/engine.py
def create_model(model_type: str, params: Dict[str, Any]) -> Any:
    # Erstellt RandomForestClassifier oder XGBClassifier
```

```363:367:app/training/engine.py
model = create_model(model_type, params)
model.fit(X_final_train, y_final_train)  # ⚠️ Blockiert!
```

### 2.12 Schritt 12: Metriken berechnen

**Datei:** `app/training/engine.py` → `train_model_sync()` (Zeile 369-418)

**Berechnete Metriken:**
- `accuracy`: Anteil korrekter Vorhersagen
- `f1`: Harmonisches Mittel aus Precision und Recall
- `precision`: Anteil korrekter positiver Vorhersagen
- `recall`: Anteil erkannte positive Fälle
- `roc_auc`: Area Under ROC Curve (falls `predict_proba` verfügbar)
- `mcc`: Matthews Correlation Coefficient
- `fpr`: False Positive Rate
- `fnr`: False Negative Rate
- `confusion_matrix`: TP, TN, FP, FN
- `simulated_profit_pct`: Simulierter Profit (vereinfacht)

**Code-Referenz:**
```369:418:app/training/engine.py
y_pred = model.predict(X_final_test)
accuracy = accuracy_score(y_final_test, y_pred)
f1 = f1_score(y_final_test, y_pred)
# ... weitere Metriken ...
```

### 2.13 Schritt 13: Feature Importance extrahieren

**Datei:** `app/training/engine.py` → `train_model_sync()` (Zeile 420-426)

**Was passiert:**
```python
if hasattr(model, 'feature_importances_'):
    importances = model.feature_importances_
    feature_importance = dict(zip(features, importances.tolist()))
```

- Feature Importance wird als Dict gespeichert
- Zeigt welche Features am wichtigsten sind

### 2.14 Schritt 14: Modell speichern

**Datei:** `app/training/engine.py` → `train_model_sync()` (Zeile 428-435)

**Was passiert:**
1. Modell wird als `.pkl` Datei gespeichert:
   ```python
   timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
   model_filename = f"model_{model_type}_{timestamp}.pkl"
   model_path = os.path.join(model_storage_path, model_filename)
   joblib.dump(model, model_path)
   ```

2. Pfad wird zurückgegeben (wird in DB gespeichert)

**Code-Referenz:**
```428:435:app/training/engine.py
os.makedirs(model_storage_path, exist_ok=True)
timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
model_filename = f"model_{model_type}_{timestamp}.pkl"
model_path = os.path.join(model_storage_path, model_filename)
joblib.dump(model, model_path)
```

### 2.15 Schritt 15: Modell in Datenbank speichern

**Datei:** `app/queue/job_manager.py` → `process_train_job()` (Zeile 224-254)

**Tabelle:** `ml_models`

**Gespeicherte Felder:**
- `name`: Modell-Name (aus `progress_msg`)
- `model_type`: "random_forest" oder "xgboost"
- `target_variable`: "price_close"
- `target_operator`: ">" (oder NULL bei zeitbasierter Vorhersage)
- `target_value`: 50000.0 (oder NULL bei zeitbasierter Vorhersage)
- `train_start`, `train_end`: Trainings-Zeitraum
- `features`: JSONB Array (inkl. engineered features!)
- `phases`: JSONB Array (z.B. `[1, 2]`)
- `params`: JSONB Object (Hyperparameter + interne Flags)
- `training_accuracy`, `training_f1`, etc.: Metriken
- `feature_importance`: JSONB Object
- `model_file_path`: Pfad zur `.pkl` Datei
- `status`: "READY"
- `cv_scores`: JSONB Object (CV-Ergebnisse)
- `cv_overfitting_gap`: Float (Overfitting-Gap)
- `roc_auc`, `mcc`, `fpr`, `fnr`: Zusätzliche Metriken
- `confusion_matrix`: JSONB Object
- `simulated_profit_pct`: Float
- `future_minutes`, `price_change_percent`, `target_direction`: Zeitbasierte Parameter

**Code-Referenz:**
```224:254:app/queue/job_manager.py
model_id = await create_model(
    name=original_model_name,
    model_type=model_type,
    # ... alle Parameter ...
)
```

### 2.16 Schritt 16: Job abschließen

**Datei:** `app/queue/job_manager.py` → `process_train_job()` (Zeile 263-275)

**Was passiert:**
1. Job-Status wird auf `COMPLETED` gesetzt
2. `result_model_id` wird gesetzt
3. Progress = 100%

---

## 3. Modell-Testing - Schritt für Schritt

### 3.1 Schritt 1: API Request (Web UI → FastAPI)

**Datei:** `app/api/routes.py` → `test_model_job()` (Zeile 174-208)

**Request-Format:**
```json
{
  "test_start": "2024-01-08T00:00:00Z",
  "test_end": "2024-01-15T00:00:00Z"
}
```

**Was passiert:**
1. Request wird validiert
2. Job wird in `ml_jobs` erstellt mit `job_type='TEST'`
3. `test_model_id`: ID des zu testenden Modells
4. `test_start`, `test_end`: Test-Zeitraum
5. Response mit `job_id` wird zurückgegeben

### 3.2 Schritt 2: Job Worker findet Test-Job

**Gleicher Ablauf wie bei Training:**
- Worker findet `PENDING` Job
- Status wird auf `RUNNING` gesetzt
- `process_test_job()` wird aufgerufen

### 3.3 Schritt 3: Modell laden

**Datei:** `app/training/model_loader.py` → `test_model()` (Zeile 43-351)

**Was passiert:**
1. Modell-Info wird aus `ml_models` geladen
2. Modell-Datei wird geladen:
   ```python
   model_obj = load_model(model['model_file_path'])
   # joblib.load("/app/models/model_random_forest_20241224_120000.pkl")
   ```

3. Features und Phasen werden extrahiert:
   ```python
   features = model['features']  # JSONB → Python List
   phases = model['phases']  # JSONB → Python List
   ```

**Code-Referenz:**
```18:41:app/training/model_loader.py
def load_model(model_path: str) -> Any:
    return joblib.load(model_path)
```

### 3.4 Schritt 4: Feature-Engineering prüfen

**Datei:** `app/training/model_loader.py` → `test_model()` (Zeile 79-104)

**Was passiert:**
1. Prüft ob Modell mit Feature-Engineering trainiert wurde:
   ```python
   use_engineered_features = params.get('use_engineered_features', False)
   feature_engineering_windows = params.get('feature_engineering_windows', [5, 10, 15])
   ```

2. Wenn aktiviert:
   - Basis-Features werden identifiziert (ohne engineered features)
   - Engineered features werden später erstellt

### 3.5 Schritt 5: Test-Daten laden

**Datei:** `app/training/model_loader.py` → `test_model()` (Zeile 106-121)

**Was passiert:**
1. Test-Daten werden geladen (gleiche Funktion wie Training):
   ```python
   test_data = await load_training_data(
       train_start=test_start,  # Wird als test_start verwendet
       train_end=test_end,
       features=features_with_target,  # Enthält target_var für Labels
       phases=phases
   )
   ```

2. **Wichtig:** `target_var` muss in Features sein (für Labels benötigt!)

### 3.6 Schritt 6: Feature-Engineering anwenden (wenn nötig)

**Datei:** `app/training/model_loader.py` → `test_model()` (Zeile 123-139)

**Was passiert:**
1. Wenn Modell mit Feature-Engineering trainiert wurde:
   ```python
   test_data = create_pump_detection_features(
       test_data, 
       window_sizes=feature_engineering_windows
   )
   ```

2. Alle Features (inkl. engineered) müssen vorhanden sein

### 3.7 Schritt 7: Labels erstellen

**Datei:** `app/training/model_loader.py` → `test_model()` (Zeile 141-203)

**Gleiche Logik wie beim Training:**
- **Zeitbasierte Vorhersage:** `create_time_based_labels()`
- **Klassische Vorhersage:** `create_labels()`

**Wichtig:** Labels müssen mit Training-Labels identisch erstellt werden!

### 3.8 Schritt 8: Vorhersagen machen

**Datei:** `app/training/model_loader.py` → `test_model()` (Zeile 205-213)

**Was passiert:**
```python
X_test = test_data[features].values  # Nur Features, nicht target_var!
y_test = labels.values
y_pred = model_obj.predict(X_test)
y_pred_proba = model_obj.predict_proba(X_test)[:, 1]  # Falls verfügbar
```

**Wichtig:**
- `target_var` wird **NICHT** als Feature verwendet (verhindert Data Leakage!)
- Nur Features die beim Training verwendet wurden

### 3.9 Schritt 9: Metriken berechnen

**Datei:** `app/training/model_loader.py` → `test_model()` (Zeile 215-259)

**Berechnete Metriken:**
- `accuracy`, `f1`, `precision`, `recall`
- `roc_auc` (falls `predict_proba` verfügbar)
- `mcc`, `fpr`, `fnr`
- `confusion_matrix`: TP, TN, FP, FN
- `simulated_profit_pct`

### 3.10 Schritt 10: Overlap-Check

**Datei:** `app/training/model_loader.py` → `test_model()` (Zeile 278-287)

**Was passiert:**
```python
overlap_info = check_overlap(
    train_start=model['train_start'],
    train_end=model['train_end'],
    test_start=test_start,
    test_end=test_end
)
```

**Prüft ob Test-Zeitraum sich mit Training überschneidet:**
- Wenn ja: Warnung (Ergebnisse können verfälscht sein)
- Blockiert aber **NICHT** den Test

### 3.11 Schritt 11: Train vs. Test Vergleich

**Datei:** `app/training/model_loader.py` → `test_model()` (Zeile 289-312)

**Was passiert:**
1. Train-Metriken werden aus Modell geladen
2. Degradation wird berechnet:
   ```python
   accuracy_degradation = train_accuracy - test_accuracy
   f1_degradation = train_f1 - test_f1
   ```

3. Overfitting-Indikator:
   ```python
   is_overfitted = accuracy_degradation > 0.1  # > 10% Gap
   ```

### 3.12 Schritt 12: Test-Ergebnis in Datenbank speichern

**Datei:** `app/queue/job_manager.py` → `process_test_job()` (Zeile 321-356)

**Tabelle:** `ml_test_results`

**Gespeicherte Felder:**
- `model_id`: Verweis auf `ml_models`
- `test_start`, `test_end`: Test-Zeitraum
- `accuracy`, `f1_score`, `precision_score`, `recall`
- `roc_auc`, `mcc`, `fpr`, `fnr`
- `confusion_matrix`: JSONB Object
- `tp`, `tn`, `fp`, `fn`: Confusion Matrix als einzelne Felder
- `num_samples`, `num_positive`, `num_negative`
- `has_overlap`, `overlap_note`
- `train_accuracy`, `train_f1`, etc.: Train-Metriken
- `accuracy_degradation`, `f1_degradation`
- `is_overfitted`: Boolean
- `test_duration_days`: Float

---

## 4. Datenfluss-Diagramme

### 4.1 Modell-Erstellung - Kompletter Datenfluss

```
┌─────────────┐
│ Streamlit UI│
└──────┬──────┘
       │ POST /api/models/create
       ▼
┌─────────────┐
│ FastAPI     │ → Erstellt Job in ml_jobs (status=PENDING)
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ Job Worker  │ → Findet PENDING Job
│ (Worker Loop)│ → Status = RUNNING
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ train_model │ → Lädt Daten aus coin_metrics
│   (async)   │ → Erstellt Labels
└──────┬──────┘ → Feature-Engineering (optional)
       │ → Cross-Validation (optional)
       │ → SMOTE (optional)
       ▼
┌─────────────┐
│train_model_ │ → Erstellt Modell (RandomForest/XGBoost)
│   sync      │ → model.fit() (CPU-bound, in run_in_executor)
│ (CPU-bound) │ → Berechnet Metriken
└──────┬──────┘ → Speichert Modell als .pkl
       │
       ▼
┌─────────────┐
│ ml_models   │ → Modell-Metadaten gespeichert
│   + .pkl    │ → Modell-Datei gespeichert
└─────────────┘
```

### 4.2 Modell-Testing - Kompletter Datenfluss

```
┌─────────────┐
│ Streamlit UI│
└──────┬──────┘
       │ POST /api/models/{id}/test
       ▼
┌─────────────┐
│ FastAPI     │ → Erstellt TEST Job in ml_jobs
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ Job Worker  │ → Findet PENDING Job
└──────┬──────┘ → Status = RUNNING
       │
       ▼
┌─────────────┐
│ test_model  │ → Lädt Modell aus ml_models + .pkl
│   (async)   │ → Lädt Test-Daten aus coin_metrics
└──────┬──────┘ → Feature-Engineering (wenn nötig)
       │ → Erstellt Labels (gleiche Logik wie Training)
       │ → model.predict() → Vorhersagen
       │ → Berechnet Metriken
       │ → Overlap-Check
       │ → Train vs. Test Vergleich
       ▼
┌─────────────┐
│ml_test_     │ → Test-Ergebnis gespeichert
│  results    │
└─────────────┘
```

---

## 5. Technische Details

### 5.1 Datenbank-Schema

#### ml_models Tabelle
```sql
CREATE TABLE ml_models (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) UNIQUE NOT NULL,
    model_type VARCHAR(50) NOT NULL,  -- "random_forest" oder "xgboost"
    target_variable VARCHAR(100) NOT NULL,
    target_operator VARCHAR(10),  -- NULL bei zeitbasierter Vorhersage
    target_value NUMERIC,  -- NULL bei zeitbasierter Vorhersage
    train_start TIMESTAMPTZ NOT NULL,
    train_end TIMESTAMPTZ NOT NULL,
    features JSONB NOT NULL,  -- Array von Feature-Namen
    phases JSONB,  -- Array von Phase-IDs oder NULL
    params JSONB,  -- Hyperparameter + interne Flags
    training_accuracy NUMERIC,
    training_f1 NUMERIC,
    training_precision NUMERIC,
    training_recall NUMERIC,
    feature_importance JSONB,  -- Dict: {feature: importance}
    model_file_path VARCHAR(500),
    status VARCHAR(50) DEFAULT 'TRAINING',
    cv_scores JSONB,  -- Cross-Validation Ergebnisse
    cv_overfitting_gap NUMERIC,
    roc_auc NUMERIC,
    mcc NUMERIC,
    fpr NUMERIC,
    fnr NUMERIC,
    confusion_matrix JSONB,
    simulated_profit_pct NUMERIC,
    future_minutes INTEGER,  -- Bei zeitbasierter Vorhersage
    price_change_percent NUMERIC,  -- Bei zeitbasierter Vorhersage
    target_direction VARCHAR(10)  -- "up" oder "down"
);
```

#### ml_jobs Tabelle
```sql
CREATE TABLE ml_jobs (
    id SERIAL PRIMARY KEY,
    job_type VARCHAR(50) NOT NULL,  -- "TRAIN", "TEST", "COMPARE"
    status VARCHAR(50) DEFAULT 'PENDING',
    progress NUMERIC DEFAULT 0.0,
    progress_msg TEXT,
    -- Training-Parameter
    train_model_type VARCHAR(50),
    train_target_var VARCHAR(100),
    train_operator VARCHAR(10),
    train_value NUMERIC,
    train_start TIMESTAMPTZ,
    train_end TIMESTAMPTZ,
    train_features JSONB,
    train_phases JSONB,
    train_params JSONB,
    train_future_minutes INTEGER,
    train_price_change_percent NUMERIC,
    train_target_direction VARCHAR(10),
    -- Test-Parameter
    test_model_id INTEGER,
    test_start TIMESTAMPTZ,
    test_end TIMESTAMPTZ,
    -- Vergleichs-Parameter
    compare_model_a_id INTEGER,
    compare_model_b_id INTEGER,
    compare_start TIMESTAMPTZ,
    compare_end TIMESTAMPTZ,
    -- Ergebnisse
    result_model_id INTEGER,
    result_test_id INTEGER,
    result_comparison_id INTEGER,
    error_msg TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);
```

#### ml_test_results Tabelle
```sql
CREATE TABLE ml_test_results (
    id SERIAL PRIMARY KEY,
    model_id INTEGER REFERENCES ml_models(id),
    test_start TIMESTAMPTZ NOT NULL,
    test_end TIMESTAMPTZ NOT NULL,
    accuracy NUMERIC,
    f1_score NUMERIC,
    precision_score NUMERIC,
    recall NUMERIC,
    roc_auc NUMERIC,
    mcc NUMERIC,
    fpr NUMERIC,
    fnr NUMERIC,
    simulated_profit_pct NUMERIC,
    confusion_matrix JSONB,
    tp INTEGER,
    tn INTEGER,
    fp INTEGER,
    fn INTEGER,
    num_samples INTEGER,
    num_positive INTEGER,
    num_negative INTEGER,
    has_overlap BOOLEAN,
    overlap_note TEXT,
    train_accuracy NUMERIC,
    train_f1 NUMERIC,
    train_precision NUMERIC,
    train_recall NUMERIC,
    accuracy_degradation NUMERIC,
    f1_degradation NUMERIC,
    is_overfitted BOOLEAN,
    test_duration_days NUMERIC,
    created_at TIMESTAMPTZ DEFAULT NOW()
);
```

### 5.2 Wichtige Konzepte erklärt

#### 5.2.1 Data Leakage Prävention

**Problem:** Bei zeitbasierter Vorhersage darf `target_var` **NICHT** als Feature verwendet werden!

**Lösung:**
```python
# Für Daten-Laden: target_var wird benötigt (für Labels)
features_for_loading = features + [target_var]

# Für Training: target_var wird ENTFERNT
features_for_training = features  # Ohne target_var!
```

**Warum?**
- Labels werden basierend auf zukünftigen Werten von `target_var` erstellt
- Wenn `target_var` als Feature verwendet wird, "sieht" das Modell die Antwort vorher
- Das wäre Data Leakage!

**Code-Referenz:**
```95:128:app/training/engine.py
def prepare_features_for_training(
    features: List[str],
    target_var: Optional[str],
    use_time_based: bool
) -> tuple[List[str], List[str]]:
    # Bei zeitbasierter Vorhersage: target_var wird entfernt!
```

#### 5.2.2 Feature-Engineering Workflow

**Beim Training:**
1. Basis-Features werden geladen (z.B. `["price_open", "price_high"]`)
2. Feature-Engineering erstellt zusätzliche Features im DataFrame
3. Features-Liste wird erweitert: `features.extend(engineered_features)`
4. Modell wird mit erweiterten Features trainiert
5. Erweiterte Features-Liste wird in DB gespeichert

**Beim Testing:**
1. Basis-Features werden geladen
2. Feature-Engineering wird angewendet (gleiche Funktion!)
3. Alle Features (inkl. engineered) müssen vorhanden sein
4. Modell macht Vorhersagen mit erweiterten Features

**Code-Referenz:**
```201:227:app/training/engine.py
if use_engineered_features:
    data = create_pump_detection_features(data, window_sizes=window_sizes)
    engineered_features_created = list(new_columns)
    features.extend(engineered_features_created)
```

#### 5.2.3 Zeitbasierte Vorhersage - Phase-Intervalle

**Problem:** Verschiedene Phasen haben verschiedene Intervalle (5s, 30s, 60s)

**Lösung:**
- Phase-Intervalle werden aus `ref_coin_phases` geladen
- Für jede Zeile wird `rows_to_shift` basierend auf `phase_id_at_time` berechnet
- Beispiel: Phase 1 (5s Intervall) → 10 Minuten = 120 Zeilen
- Beispiel: Phase 2 (30s Intervall) → 10 Minuten = 20 Zeilen

**Code-Referenz:**
```218:273:app/training/feature_engineering.py
if 'phase_id_at_time' in data.columns and phase_intervals:
    def calculate_rows_to_shift(phase_id):
        interval_seconds = phase_intervals[phase_id]
        interval_minutes = interval_seconds / 60.0
        return int(round(future_minutes / interval_minutes))
```

#### 5.2.4 SMOTE (Imbalanced Data Handling)

**Problem:** Bei Pump-Detection haben wir oft 99% normale Coins, 1% Pump-Coins

**Lösung:**
- SMOTE erstellt synthetische Pump-Beispiele
- Random Under-Sampling reduziert normale Coins
- Ziel: ~50/50 Balance

**Wird nur angewendet wenn:**
- `use_smote=True` UND
- Label-Balance < 30% oder > 70%

**Code-Referenz:**
```301:361:app/training/engine.py
if use_smote:
    positive_ratio = y_final_train.sum() / len(y_final_train)
    if positive_ratio < 0.3 or positive_ratio > 0.7:
        # SMOTE anwenden ...
```

#### 5.2.5 TimeSeriesSplit vs. normaler Split

**Problem:** Bei Zeitreihen darf die Zukunft nicht zum Training verwendet werden!

**Lösung:**
- **TimeSeriesSplit:** Respektiert zeitliche Reihenfolge
  - Split 1: Train = [0:20%], Test = [20%:40%]
  - Split 2: Train = [0:40%], Test = [40%:60%]
  - Split 3: Train = [0:60%], Test = [60%:80%]
  - Split 4: Train = [0:80%], Test = [80%:100%]
- **Normaler Split:** Zufällige Aufteilung (nicht empfohlen für Zeitreihen!)

**Code-Referenz:**
```240:275:app/training/engine.py
if use_timeseries_split:
    tscv = TimeSeriesSplit(n_splits=n_splits)
    cv_results = cross_validate(estimator=temp_model, X=X, y=y, cv=tscv, ...)
```

---

## 6. Code-Referenzen

### 6.1 Wichtige Dateien

| Datei | Zweck | Wichtige Funktionen |
|-------|-------|---------------------|
| `app/api/routes.py` | API-Endpunkte | `create_model_job()`, `test_model_job()` |
| `app/queue/job_manager.py` | Job-Verarbeitung | `process_train_job()`, `process_test_job()`, `start_worker()` |
| `app/training/engine.py` | Training-Logik | `train_model()`, `train_model_sync()`, `create_model()` |
| `app/training/feature_engineering.py` | Feature-Engineering | `load_training_data()`, `create_labels()`, `create_time_based_labels()`, `create_pump_detection_features()` |
| `app/training/model_loader.py` | Modell-Laden & Testing | `load_model()`, `test_model()` |
| `app/database/models.py` | DB-Operationen | `create_model()`, `create_test_result()`, `create_job()` |

### 6.2 Wichtige Funktionen im Detail

#### 6.2.1 `train_model()` - Async Wrapper

**Datei:** `app/training/engine.py` (Zeile 479-586)

**Zweck:** Async Wrapper für `train_model_sync()`

**Ablauf:**
1. Lädt Default-Parameter aus DB
2. Mergt mit übergebenen Parametern
3. Bereitet Features vor (verhindert Data Leakage)
4. Lädt Trainingsdaten (async)
5. Ruft `train_model_sync()` in `run_in_executor` auf (CPU-bound!)

#### 6.2.2 `train_model_sync()` - CPU-bound Training

**Datei:** `app/training/engine.py` (Zeile 130-477)

**Zweck:** Führt tatsächliches Training durch

**Ablauf:**
1. Erstellt Labels (zeitbasiert oder klassisch)
2. Feature-Engineering (optional)
3. Cross-Validation (optional)
4. Train-Test-Split
5. SMOTE (optional)
6. Modell-Training (`model.fit()`)
7. Metriken berechnen
8. Modell speichern

#### 6.2.3 `create_time_based_labels()` - Zeitbasierte Labels

**Datei:** `app/training/feature_engineering.py` (Zeile 177-316)

**Zweck:** Erstellt Labels für zeitbasierte Vorhersagen

**Ablauf:**
1. Für jede Zeile wird zukünftiger Wert berechnet
2. Berechnung basiert auf Phase-Intervall
3. Prozentuale Änderung wird berechnet
4. Label = 1 wenn Bedingung erfüllt, sonst 0

#### 6.2.4 `test_model()` - Modell-Testing

**Datei:** `app/training/model_loader.py` (Zeile 43-351)

**Zweck:** Testet Modell auf neuen Daten

**Ablauf:**
1. Lädt Modell aus DB + .pkl Datei
2. Lädt Test-Daten
3. Feature-Engineering (wenn nötig)
4. Erstellt Labels (gleiche Logik wie Training)
5. Macht Vorhersagen
6. Berechnet Metriken
7. Overlap-Check
8. Train vs. Test Vergleich

---

## 7. Zusammenfassung - Der komplette Workflow

### 7.1 Modell-Erstellung (Zusammenfassung)

1. **UI Request** → FastAPI erstellt Job in `ml_jobs` (PENDING)
2. **Job Worker** → Findet Job, setzt Status auf RUNNING
3. **Training Engine** → Lädt Daten aus `coin_metrics`
4. **Labels** → Erstellt binäre Labels (zeitbasiert oder klassisch)
5. **Feature-Engineering** → Erstellt zusätzliche Features (optional)
6. **Cross-Validation** → TimeSeriesSplit für realistische Metriken (optional)
7. **SMOTE** → Balanciert unausgewogene Daten (optional)
8. **Training** → `model.fit()` (CPU-bound, in run_in_executor)
9. **Metriken** → Berechnet Accuracy, F1, etc.
10. **Speicherung** → Modell als .pkl + Metadaten in `ml_models`

### 7.2 Modell-Testing (Zusammenfassung)

1. **UI Request** → FastAPI erstellt TEST Job in `ml_jobs` (PENDING)
2. **Job Worker** → Findet Job, setzt Status auf RUNNING
3. **Model Loader** → Lädt Modell aus DB + .pkl Datei
4. **Test-Daten** → Lädt Daten aus `coin_metrics` (neuer Zeitraum)
5. **Feature-Engineering** → Erstellt Features (wenn Modell damit trainiert wurde)
6. **Labels** → Erstellt Labels (gleiche Logik wie Training)
7. **Vorhersagen** → `model.predict()` auf Test-Daten
8. **Metriken** → Berechnet Accuracy, F1, etc.
9. **Overlap-Check** → Prüft Überschneidung mit Training
10. **Vergleich** → Train vs. Test Metriken
11. **Speicherung** → Test-Ergebnis in `ml_test_results`

---

## 8. Wichtige Hinweise für Überarbeitung

### 8.1 Aktuelle Architektur-Stärken

✅ **Asynchrones Job-System:**
- Skaliert gut (mehrere Jobs parallel)
- Blockiert API nicht

✅ **CPU-bound Training in run_in_executor:**
- Blockiert Event Loop nicht
- Kann mehrere Minuten dauern

✅ **Data Leakage Prävention:**
- `target_var` wird bei zeitbasierter Vorhersage entfernt
- Labels werden korrekt erstellt

✅ **Feature-Engineering:**
- Optional, aber mächtig
- Erstellt ~40 zusätzliche Features

### 8.2 Potenzielle Verbesserungen

⚠️ **RAM-Management:**
- Aktuell: Max 500.000 Zeilen
- Könnte bei sehr großen Datensätzen problematisch sein

⚠️ **Feature-Engineering:**
- Verwendet Spalten die möglicherweise nicht existieren
- Fallback-Logik vorhanden, aber könnte robuster sein

⚠️ **Zeitbasierte Vorhersage:**
- Komplexe Logik mit Phase-Intervallen
- Könnte vereinfacht werden

⚠️ **Job-Queue:**
- Aktuell: Polling alle 5 Sekunden
- Könnte mit Message Queue (Redis, RabbitMQ) verbessert werden

---

## 9. Beispiel-Workflow (Konkret)

### 9.1 Beispiel: Modell erstellen

**Request:**
```json
POST /api/models/create
{
  "name": "PumpDetector_v1",
  "model_type": "xgboost",
  "train_start": "2024-01-01T00:00:00Z",
  "train_end": "2024-01-07T00:00:00Z",
  "features": ["price_open", "price_high", "price_low", "price_close", "volume_sol"],
  "use_time_based_prediction": true,
  "target_var": "price_close",
  "future_minutes": 10,
  "min_percent_change": 30.0,
  "direction": "up",
  "use_engineered_features": true,
  "use_smote": true,
  "use_timeseries_split": true
}
```

**Was passiert:**
1. Job wird erstellt (ID: 123, Status: PENDING)
2. Worker findet Job nach ~5 Sekunden
3. Status wird auf RUNNING gesetzt
4. Daten werden geladen: ~100.000 Zeilen aus `coin_metrics`
5. Labels werden erstellt: "Steigt price_close in 10 Minuten um 30%?"
   - Positive Labels: ~5.000 (5%)
   - Negative Labels: ~95.000 (95%)
6. Feature-Engineering erstellt ~40 zusätzliche Features
7. SMOTE wird angewendet (Balance < 30%)
8. TimeSeriesSplit mit 5 Splits
9. XGBoost wird trainiert (dauert ~5 Minuten)
10. Metriken: Accuracy=0.85, F1=0.72
11. Modell wird gespeichert: `/app/models/model_xgboost_20241224_120000.pkl`
12. Modell in DB gespeichert (ID: 456, Status: READY)
13. Job abgeschlossen (Status: COMPLETED)

### 9.2 Beispiel: Modell testen

**Request:**
```json
POST /api/models/456/test
{
  "test_start": "2024-01-08T00:00:00Z",
  "test_end": "2024-01-15T00:00:00Z"
}
```

**Was passiert:**
1. Job wird erstellt (ID: 124, Status: PENDING)
2. Worker findet Job nach ~5 Sekunden
3. Modell wird geladen: ID 456, `.pkl` Datei
4. Test-Daten werden geladen: ~100.000 Zeilen (neuer Zeitraum)
5. Feature-Engineering wird angewendet (gleiche Features wie Training)
6. Labels werden erstellt (gleiche Logik wie Training)
7. Vorhersagen werden gemacht: `model.predict(X_test)`
8. Metriken werden berechnet:
   - Test Accuracy: 0.82
   - Train Accuracy: 0.85
   - Accuracy Degradation: 0.03 (3%)
9. Overlap-Check: Keine Überschneidung ✅
10. Test-Ergebnis wird gespeichert (ID: 789)
11. Job abgeschlossen (Status: COMPLETED)

---

## 10. Fazit

Das System verwendet ein **asynchrones Job-System** mit **CPU-bound Training** in separaten Threads. Modelle werden **nicht sofort** erstellt, sondern in einer Queue verarbeitet. Dies ermöglicht:

- ✅ Skalierbarkeit (mehrere Jobs parallel)
- ✅ Nicht-blockierende API
- ✅ Robuste Fehlerbehandlung
- ✅ Progress-Tracking

**Wichtige Konzepte:**
- **Zeitbasierte Vorhersage:** Verwendet zukünftige Werte für Labels
- **Feature-Engineering:** Erstellt ~40 zusätzliche Features
- **Data Leakage Prävention:** `target_var` wird bei zeitbasierter Vorhersage entfernt
- **TimeSeriesSplit:** Respektiert zeitliche Reihenfolge
- **SMOTE:** Balanciert unausgewogene Daten

**Für Überarbeitung zu beachten:**
- RAM-Management (aktuell: 500k Zeilen Limit)
- Feature-Engineering Robustheit
- Job-Queue könnte mit Message Queue verbessert werden
- Zeitbasierte Vorhersage-Logik könnte vereinfacht werden

