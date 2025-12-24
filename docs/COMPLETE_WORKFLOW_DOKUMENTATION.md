# 📚 Komplette Workflow-Dokumentation - Modell-Erstellung und Testing

**Datum:** 2025-12-24  
**Ziel:** Vollständiges Verständnis des gesamten Workflows von Modell-Erstellung bis Testing

---

## 📋 Inhaltsverzeichnis

1. [Modell-Erstellung - Kompletter Workflow](#1-modell-erstellung---kompletter-workflow)
2. [Modell-Testing - Kompletter Workflow](#2-modell-testing---kompletter-workflow)
3. [Datenfluss-Diagramme](#3-datenfluss-diagramme)
4. [Validierung und Prüfmethoden](#4-validierung-und-prüfmethoden)
5. [Potenzielle Fehlerquellen](#5-potenzielle-fehlerquellen)
6. [Code-Referenzen](#6-code-referenzen)

---

## 1. Modell-Erstellung - Kompletter Workflow

### 1.1 API Request (Web UI oder API)

**Datei:** `app/api/routes.py` → `create_model_job()` (Zeile 42-114)

**Request-Format:**
```json
{
  "name": "Mein Modell",
  "model_type": "random_forest",
  "train_start": "2024-01-01T00:00:00Z",
  "train_end": "2024-01-07T00:00:00Z",
  "features": ["price_open", "price_high", "price_low", "price_close"],
  "phases": [1, 2],
  "use_time_based_prediction": true,
  "target_var": "price_close",
  "future_minutes": 10,
  "min_percent_change": 30.0,
  "direction": "up",
  "params": {
    "n_estimators": 100,
    "max_depth": 5
  }
}
```

**Validierung (Pydantic):**
- ✅ `model_type` muss "random_forest" oder "xgboost" sein
- ✅ `train_start` < `train_end`
- ✅ Wenn `use_time_based_prediction=true`: `future_minutes` und `min_percent_change` müssen gesetzt sein
- ✅ Wenn `use_time_based_prediction=false`: `target_var`, `operator`, `target_value` müssen gesetzt sein

**Was passiert:**
1. Request wird validiert (Pydantic Schema)
2. Zeitbasierte Parameter werden in `params._time_based` gespeichert
3. Job wird in `ml_jobs` erstellt mit `status='PENDING'`
4. Modell-Name wird in `progress_msg` temporär gespeichert
5. Response mit `job_id` wird zurückgegeben

**Code-Referenz:**
```42:114:app/api/routes.py
@router.post("/models/create", response_model=CreateJobResponse, status_code=status.HTTP_201_CREATED)
async def create_model_job(request: TrainModelRequest):
    # ... Validierung und Job-Erstellung
```

---

### 1.2 Job Worker - Job wird verarbeitet

**Datei:** `app/queue/job_manager.py` → `start_worker()` (Zeile 557-600)

**Ablauf:**
1. Worker prüft alle 5 Sekunden auf neue PENDING Jobs
2. `get_next_pending_job()` holt nächsten Job und setzt Status ATOMIC auf RUNNING
3. `process_job()` wird asynchron gestartet

**Code-Referenz:**
```557:600:app/queue/job_manager.py
async def start_worker() -> None:
    while True:
        job = await get_next_pending_job()  # Status wird ATOMIC auf RUNNING gesetzt
        if job:
            task = asyncio.create_task(process_job(job['id']))
```

**WICHTIG:** `get_next_pending_job()` setzt Status sofort auf RUNNING, um Race Conditions zu vermeiden!

---

### 1.3 Job-Verarbeitung - Training startet

**Datei:** `app/queue/job_manager.py` → `process_train_job()` (Zeile 88-275)

**Schritt-für-Schritt:**

#### Schritt 1: Job-Parameter extrahieren
```python
model_name = job.get('progress_msg')  # Name wurde hier gespeichert
model_type = job.get('train_model_type')
features = job.get('train_features')
phases = job.get('train_phases')
target_var = job.get('train_target_var')
target_operator = job.get('train_operator')
target_value = job.get('train_value')
train_start = job.get('train_start')
train_end = job.get('train_end')
params = job.get('train_params')  # JSONB → Python Dict
```

#### Schritt 2: Zeitbasierte Parameter extrahieren
```python
use_time_based = False
if params and "_time_based" in params:
    time_based_config = params.get("_time_based", {})
    use_time_based = time_based_config.get("enabled", False)
    future_minutes = time_based_config.get("future_minutes")
    min_percent_change = time_based_config.get("min_percent_change")
    direction = time_based_config.get("direction", "up")
```

#### Schritt 3: Progress Update (10%)
```python
await update_job_status(job_id, status="RUNNING", progress=0.1, 
                        progress_msg="Lade Trainingsdaten...")
```

#### Schritt 4: Training starten
```python
training_result = await train_model(
    model_type=model_type,
    features=features,
    target_var=target_var,
    target_operator=target_operator,
    target_value=target_value,
    train_start=train_start,
    train_end=train_end,
    phases=phases,
    params=params,
    use_time_based=use_time_based,
    future_minutes=future_minutes,
    min_percent_change=min_percent_change,
    direction=direction
)
```

**Code-Referenz:**
```88:187:app/queue/job_manager.py
async def process_train_job(job: Dict[str, Any]) -> None:
    # ... Job-Parameter extrahieren
    # ... Training starten
```

---

### 1.4 Training Engine - Daten laden

**Datei:** `app/training/engine.py` → `train_model()` (Zeile 479-586)

**Schritt-für-Schritt:**

#### Schritt 1: Default-Parameter laden
```python
default_params = await get_model_type_defaults(model_type)
final_params = {**default_params, **(params or {})}
```

#### Schritt 2: Features vorbereiten (verhindert Data Leakage)
```python
features_for_loading, features_for_training = prepare_features_for_training(
    features=features,
    target_var=target_var,
    use_time_based=use_time_based
)
```

**WICHTIG:** Bei zeitbasierter Vorhersage wird `target_var` nur für Labels verwendet, NICHT für Training!

#### Schritt 3: Trainingsdaten laden
```python
data = await load_training_data(
    train_start=train_start,
    train_end=train_end,
    features=features_for_loading,  # Enthält target_var (für Labels)
    phases=phases
)
```

**Code-Referenz:**
```479:586:app/training/engine.py
async def train_model(...):
    # ... Daten laden
    # ... Training in run_in_executor
```

---

### 1.5 Training Engine - Daten laden (Datenbank)

**Datei:** `app/training/feature_engineering.py` → `load_training_data()` (Zeile 50-130)

**SQL Query:**
```sql
SELECT timestamp, price_open, price_high, price_low, price_close, phase_id_at_time
FROM coin_metrics
WHERE timestamp >= $1 AND timestamp <= $2
  AND phase_id_at_time = ANY($3)  -- Falls phases gesetzt
ORDER BY timestamp
LIMIT 500000  -- RAM-Management
```

**Was passiert:**
1. Zeitstempel werden zu UTC konvertiert
2. Daten werden aus `coin_metrics` geladen
3. Duplikate werden entfernt (basierend auf `timestamp` + `coin_id`)
4. DataFrame wird zurückgegeben

**Code-Referenz:**
```50:130:app/training/feature_engineering.py
async def load_training_data(...):
    # ... SQL Query
    # ... Duplikat-Entfernung
    # ... DataFrame zurückgeben
```

---

### 1.6 Training Engine - Labels erstellen

**Datei:** `app/training/engine.py` → `train_model_sync()` (Zeile 130-450)

#### Option A: Zeitbasierte Labels
```python
if use_time_based:
    labels = create_time_based_labels(
        data,
        target_var,  # z.B. "price_close"
        future_minutes,  # z.B. 10
        min_percent_change,  # z.B. 30.0
        direction,  # "up" oder "down"
        phase_intervals  # {phase_id: interval_seconds}
    )
```

**Was passiert:**
1. Für jede Zeile wird geprüft, ob `target_var` in `future_minutes` um `min_percent_change%` steigt/fällt
2. Phase-Intervalle werden berücksichtigt (z.B. Phase 1 = 60 Sekunden)
3. Label = 1 wenn Bedingung erfüllt, sonst 0

**Code-Referenz:**
```177:250:app/training/feature_engineering.py
def create_time_based_labels(...):
    # ... Berechnet zukünftige Werte
    # ... Prüft ob Änderung >= min_percent_change
    # ... Gibt binäre Labels zurück
```

#### Option B: Normale Labels
```python
else:
    labels = create_labels(
        data,
        target_var,  # z.B. "market_cap_close"
        target_operator,  # z.B. ">"
        target_value  # z.B. 10000
    )
```

**Was passiert:**
1. Für jede Zeile wird geprüft: `data[target_var] operator target_value`
2. Label = 1 wenn Bedingung erfüllt, sonst 0

**Code-Referenz:**
```133:175:app/training/feature_engineering.py
def create_labels(...):
    # ... Prüft Bedingung
    # ... Gibt binäre Labels zurück
```

#### Validierung:
```python
positive_count = labels.sum()
negative_count = len(labels) - positive_count

if positive_count == 0:
    raise ValueError("Keine positiven Labels gefunden!")
if negative_count == 0:
    raise ValueError("Keine negativen Labels gefunden!")
```

---

### 1.7 Training Engine - Feature Engineering (optional)

**Datei:** `app/training/engine.py` → `train_model_sync()` (Zeile 201-228)

**Wenn aktiviert:**
```python
if use_engineered_features:
    data = create_pump_detection_features(data, window_sizes=[5, 10, 15])
    # Neue Features werden erstellt:
    # - price_change_5, price_change_10, price_change_15
    # - volume_ratio_5, volume_ratio_10, volume_ratio_15
    # - price_volatility_5, price_volatility_10, price_volatility_15
    # - mcap_velocity_5, mcap_velocity_10, mcap_velocity_15
    # - price_range_5, price_range_10, price_range_15
    # - price_roc_5, price_roc_10, price_roc_15
    # - order_book_imbalance_5, order_book_imbalance_10, order_book_imbalance_15
    
    features.extend(engineered_features_created)
```

**Code-Referenz:**
```201:228:app/training/engine.py
if use_engineered_features:
    # ... Feature Engineering
```

---

### 1.8 Training Engine - Cross-Validation (optional)

**Datei:** `app/training/engine.py` → `train_model_sync()` (Zeile 240-300)

**Wenn TimeSeriesSplit aktiviert:**
```python
if use_timeseries_split:
    tscv = TimeSeriesSplit(n_splits=5)
    cv_results = cross_validate(
        estimator=temp_model,
        X=X,
        y=y,
        cv=tscv,
        scoring=['accuracy', 'f1', 'precision', 'recall'],
        return_train_score=True
    )
    
    # Overfitting-Check
    train_test_gap = cv_results['train_accuracy'].mean() - cv_results['test_accuracy'].mean()
    if train_test_gap > 0.1:
        logger.warning("⚠️ OVERFITTING erkannt!")
```

**Code-Referenz:**
```240:300:app/training/engine.py
if use_timeseries_split:
    # ... Cross-Validation
```

---

### 1.9 Training Engine - SMOTE (optional)

**Datei:** `app/training/engine.py` → `train_model_sync()` (Zeile 301-350)

**Wenn aktiviert und Label-Ungleichgewicht:**
```python
if use_smote and (positive_ratio < 0.3 or positive_ratio > 0.7):
    smote = SMOTE(k_neighbors=k_neighbors, sampling_strategy=0.5)
    under = RandomUnderSampler(sampling_strategy=0.8)
    pipeline = ImbPipeline([('smote', smote), ('under', under)])
    X_final_train, y_final_train = pipeline.fit_resample(X_final_train, y_final_train)
```

**Code-Referenz:**
```301:350:app/training/engine.py
if use_smote:
    # ... SMOTE anwenden
```

---

### 1.10 Training Engine - Modell trainieren

**Datei:** `app/training/engine.py` → `train_model_sync()` (Zeile 350-420)

**Ablauf:**
```python
# Modell erstellen
model = create_model(model_type, params)

# Training
model.fit(X_final_train, y_final_train)

# Vorhersagen auf Test-Set
y_pred = model.predict(X_final_test)
y_pred_proba = model.predict_proba(X_final_test)[:, 1]

# Metriken berechnen
accuracy = accuracy_score(y_final_test, y_pred)
f1 = f1_score(y_final_test, y_pred)
precision = precision_score(y_final_test, y_pred)
recall = recall_score(y_final_test, y_pred)
roc_auc = roc_auc_score(y_final_test, y_pred_proba)

# Confusion Matrix
cm = confusion_matrix(y_final_test, y_pred)
tn, fp, fn, tp = cm.ravel()

# Zusätzliche Metriken
mcc = matthews_corrcoef(y_final_test, y_pred)
fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

# Profit-Simulation
simulated_profit_pct = ((tp * 0.01) + (fp * -0.005)) / len(y_final_test) * 100

# Feature Importance
feature_importance = dict(zip(features, model.feature_importances_))

# Modell speichern
model_path = f"/app/models/model_{model_type}_{timestamp}.pkl"
joblib.dump(model, model_path)
```

**Code-Referenz:**
```350:450:app/training/engine.py
# ... Modell trainieren
# ... Metriken berechnen
# ... Modell speichern
```

---

### 1.11 Job Manager - Modell in DB speichern

**Datei:** `app/queue/job_manager.py` → `process_train_job()` (Zeile 223-253)

**Ablauf:**
```python
model_id = await create_model(
    name=original_model_name,
    model_type=model_type,
    target_variable=target_var,
    train_start=train_start_dt,
    train_end=train_end_dt,
    target_operator=target_operator,
    target_value=target_value,
    features=final_features,  # Inkl. engineered features
    phases=phases,
    params=params,
    training_accuracy=training_result['accuracy'],
    training_f1=training_result['f1'],
    training_precision=training_result['precision'],
    training_recall=training_result['recall'],
    feature_importance=training_result['feature_importance'],
    model_file_path=training_result['model_path'],
    status="READY",
    cv_scores=training_result.get('cv_scores'),
    cv_overfitting_gap=training_result.get('cv_overfitting_gap'),
    roc_auc=training_result.get('roc_auc'),
    mcc=training_result.get('mcc'),
    fpr=training_result.get('fpr'),
    fnr=training_result.get('fnr'),
    confusion_matrix=training_result.get('confusion_matrix'),
    simulated_profit_pct=training_result.get('simulated_profit_pct'),
    future_minutes=train_future_minutes,
    price_change_percent=train_price_change_percent,
    target_direction=train_target_direction
)
```

**Code-Referenz:**
```223:253:app/queue/job_manager.py
model_id = await create_model(...)
```

---

### 1.12 Database - Modell speichern

**Datei:** `app/database/models.py` → `create_model()` (Zeile 63-220)

**Ablauf:**
1. Eindeutigen Namen generieren (falls nötig)
2. JSONB-Felder konvertieren (features, phases, params, etc.)
3. INSERT in `ml_models` Tabelle
4. Bei Duplikat-Fehler: Neuen Namen generieren und retry (max. 2 Versuche)

**SQL INSERT:**
```sql
INSERT INTO ml_models (
    name, model_type, status,
    target_variable, target_operator, target_value,
    train_start, train_end,
    features, phases, params,
    training_accuracy, training_f1, training_precision, training_recall,
    feature_importance, model_file_path, description,
    cv_scores, cv_overfitting_gap,
    roc_auc, mcc, fpr, fnr, confusion_matrix, simulated_profit_pct,
    future_minutes, price_change_percent, target_direction
) VALUES (
    $1, $2, $3, $4, $5, $6, $7, $8, $9::jsonb, $10::jsonb, $11::jsonb,
    $12, $13, $14, $15, $16::jsonb, $17, $18, $19::jsonb, $20,
    $21, $22, $23, $24, $25::jsonb, $26, $27, $28, $29
) RETURNING id
```

**Code-Referenz:**
```63:220:app/database/models.py
async def create_model(...):
    # ... Eindeutigen Namen generieren
    # ... INSERT in DB
```

---

### 1.13 Job Manager - Job abschließen

**Datei:** `app/queue/job_manager.py` → `process_train_job()` (Zeile 261-275)

**Ablauf:**
```python
await update_job_status(
    job_id,
    status="COMPLETED",
    progress=1.0,
    result_model_id=model_id,
    progress_msg=f"Modell {original_model_name} erfolgreich erstellt"
)
```

**Code-Referenz:**
```261:275:app/queue/job_manager.py
await update_job_status(...)
```

---

## 2. Modell-Testing - Kompletter Workflow

### 2.1 API Request

**Datei:** `app/api/routes.py` → `test_model_job()` (Zeile 173-208)

**Request-Format:**
```json
{
  "test_start": "2024-01-08T00:00:00Z",
  "test_end": "2024-01-09T00:00:00Z"
}
```

**Was passiert:**
1. Modell wird geprüft (existiert, nicht gelöscht)
2. Job wird in `ml_jobs` erstellt mit `job_type='TEST'`
3. Response mit `job_id` wird zurückgegeben

**Code-Referenz:**
```173:208:app/api/routes.py
@router.post("/models/{model_id}/test", ...)
async def test_model_job(model_id: int, request: TestModelRequest):
    # ... Job erstellen
```

---

### 2.2 Job Worker - Test startet

**Datei:** `app/queue/job_manager.py` → `process_test_job()` (Zeile 277-368)

**Schritt-für-Schritt:**

#### Schritt 1: Job-Parameter extrahieren
```python
model_id = job['test_model_id']
test_start = job['test_start']
test_end = job['test_end']
```

#### Schritt 2: Test ausführen
```python
test_result = await test_model(
    model_id=model_id,
    test_start=test_start,
    test_end=test_end,
    model_storage_path=MODEL_STORAGE_PATH
)
```

**Code-Referenz:**
```277:368:app/queue/job_manager.py
async def process_test_job(job: Dict[str, Any]) -> None:
    # ... Test ausführen
    # ... Ergebnis speichern
```

---

### 2.3 Test Engine - Modell laden

**Datei:** `app/training/model_loader.py` → `test_model()` (Zeile 43-353)

**Schritt-für-Schritt:**

#### Schritt 1: Modell-Info aus DB laden
```python
model = await get_model(model_id)
features = model['features']  # JSONB → Python List
phases = model['phases']  # JSONB → Python List
params = model.get('params', {})  # JSONB → Python Dict
```

#### Schritt 2: Modell-Datei laden
```python
model_obj = load_model(model['model_file_path'])
```

#### Schritt 3: Feature-Engineering prüfen
```python
use_engineered_features = params.get('use_engineered_features', False)
if use_engineered_features:
    # Basis-Features extrahieren (ohne engineered features)
    base_features = [f for f in features if f not in engineered_feature_names]
```

#### Schritt 4: Test-Daten laden
```python
test_data = await load_training_data(
    train_start=test_start,  # Wird als "train_start" verwendet (Funktion heißt so)
    train_end=test_end,
    features=features_with_target,  # Enthält target_var für Labels
    phases=phases
)
```

#### Schritt 5: Feature-Engineering anwenden (falls nötig)
```python
if use_engineered_features:
    test_data = create_pump_detection_features(test_data, window_sizes=feature_engineering_windows)
```

**Code-Referenz:**
```43:140:app/training/model_loader.py
async def test_model(...):
    # ... Modell laden
    # ... Test-Daten laden
```

---

### 2.4 Test Engine - Labels erstellen

**Datei:** `app/training/model_loader.py` → `test_model()` (Zeile 141-203)

**Gleiche Logik wie beim Training:**
- Zeitbasierte Vorhersage: `create_time_based_labels()`
- Normale Vorhersage: `create_labels()`

**WICHTIG:** Labels werden mit GLEICHER Logik erstellt wie beim Training!

**Code-Referenz:**
```141:203:app/training/model_loader.py
# ... Labels erstellen (gleiche Logik wie Training)
```

---

### 2.5 Test Engine - Vorhersagen machen

**Datei:** `app/training/model_loader.py` → `test_model()` (Zeile 205-213)

**Ablauf:**
```python
X_test = test_data[features].values  # Alle Features (inkl. engineered)
y_test = labels.values
y_pred = model_obj.predict(X_test)
y_pred_proba = model_obj.predict_proba(X_test)[:, 1]
```

**Code-Referenz:**
```205:213:app/training/model_loader.py
# ... Vorhersagen machen
```

---

### 2.6 Test Engine - Metriken berechnen

**Datei:** `app/training/model_loader.py` → `test_model()` (Zeile 215-259)

**Berechnete Metriken:**
```python
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()

# Zusätzliche Metriken
mcc = matthews_corrcoef(y_test, y_pred)
fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

# Profit-Simulation
simulated_profit_pct = ((tp * 0.01) + (fp * -0.005)) / len(y_test) * 100
```

**Code-Referenz:**
```215:259:app/training/model_loader.py
# ... Metriken berechnen
```

---

### 2.7 Test Engine - Train vs. Test Vergleich

**Datei:** `app/training/model_loader.py` → `test_model()` (Zeile 289-320)

**Ablauf:**
```python
# Train-Metriken aus Modell holen
train_accuracy = model.get('training_accuracy')
train_f1 = model.get('training_f1')
train_precision = model.get('training_precision')
train_recall = model.get('training_recall')

# Degradation berechnen
accuracy_degradation = train_accuracy - accuracy
f1_degradation = train_f1 - f1

# Overfitting-Check
is_overfitted = accuracy_degradation > 0.1
```

**Code-Referenz:**
```289:320:app/training/model_loader.py
# ... Train vs. Test Vergleich
```

---

### 2.8 Test Engine - Overlap-Check

**Datei:** `app/training/model_loader.py` → `test_model()` (Zeile 278-287)

**Ablauf:**
```python
overlap_info = check_overlap(
    train_start=model['train_start'],
    train_end=model['train_end'],
    test_start=test_start,
    test_end=test_end
)
```

**Was wird geprüft:**
- Überlappen sich Train- und Test-Zeitraum?
- Wenn ja: Warnung (Data Leakage möglich!)

**Code-Referenz:**
```278:287:app/training/model_loader.py
# ... Overlap-Check
```

---

### 2.9 Job Manager - Test-Ergebnis speichern

**Datei:** `app/queue/job_manager.py` → `process_test_job()` (Zeile 320-366)

**Ablauf:**
```python
test_id = await get_or_create_test_result(
    model_id=model_id,
    test_start=test_start_dt,
    test_end=test_end_dt,
    accuracy=test_result['accuracy'],
    f1_score=test_result['f1_score'],
    # ... alle Metriken
)
```

**WICHTIG:** `get_or_create_test_result()` prüft zuerst, ob Test bereits existiert (UNIQUE Constraint)!

**Code-Referenz:**
```320:366:app/queue/job_manager.py
test_id = await get_or_create_test_result(...)
```

---

### 2.10 Database - Test-Ergebnis speichern

**Datei:** `app/database/models.py` → `get_or_create_test_result()` (Zeile 359-431)

**Ablauf:**
1. Prüfe ob Test bereits existiert: `SELECT id FROM ml_test_results WHERE model_id=$1 AND test_start=$2 AND test_end=$3`
2. Wenn existiert: Gib existierende ID zurück
3. Wenn nicht: Erstelle neuen Test mit `ON CONFLICT DO UPDATE`

**SQL:**
```sql
INSERT INTO ml_test_results (
    model_id, test_start, test_end,
    accuracy, f1_score, precision_score, recall, roc_auc,
    mcc, fpr, fnr, simulated_profit_pct, confusion_matrix,
    tp, tn, fp, fn,
    num_samples, num_positive, num_negative,
    has_overlap, overlap_note,
    train_accuracy, train_f1, train_precision, train_recall,
    accuracy_degradation, f1_degradation, is_overfitted,
    test_duration_days
) VALUES (...)
ON CONFLICT (model_id, test_start, test_end) 
DO UPDATE SET accuracy = EXCLUDED.accuracy, ...
RETURNING id
```

**Code-Referenz:**
```359:431:app/database/models.py
async def get_or_create_test_result(...):
    # ... Prüfe ob existiert
    # ... Erstelle oder Update
```

---

## 3. Datenfluss-Diagramme

### 3.1 Modell-Erstellung - Datenfluss

```
API Request (Web UI/API)
    ↓
[app/api/routes.py] create_model_job()
    ↓ Validierung (Pydantic)
    ↓ Job in ml_jobs erstellen (status='PENDING')
    ↓
Job Worker [app/queue/job_manager.py] start_worker()
    ↓ get_next_pending_job() → Status ATOMIC auf RUNNING
    ↓ process_train_job()
    ↓
Training Engine [app/training/engine.py] train_model()
    ↓ load_training_data() → Daten aus coin_metrics
    ↓ create_labels() oder create_time_based_labels()
    ↓ Feature Engineering (optional)
    ↓ Cross-Validation (optional)
    ↓ SMOTE (optional)
    ↓ train_model_sync() → Modell trainieren
    ↓ Metriken berechnen
    ↓ Modell speichern (.pkl Datei)
    ↓
Job Manager [app/queue/job_manager.py] process_train_job()
    ↓ create_model() → Modell in ml_models speichern
    ↓ update_job_status(status='COMPLETED')
```

### 3.2 Modell-Testing - Datenfluss

```
API Request (Web UI/API)
    ↓
[app/api/routes.py] test_model_job()
    ↓ Job in ml_jobs erstellen (status='PENDING')
    ↓
Job Worker [app/queue/job_manager.py] start_worker()
    ↓ get_next_pending_job() → Status ATOMIC auf RUNNING
    ↓ process_test_job()
    ↓
Test Engine [app/training/model_loader.py] test_model()
    ↓ get_model() → Modell-Info aus ml_models
    ↓ load_model() → Modell-Datei laden (.pkl)
    ↓ load_training_data() → Test-Daten aus coin_metrics
    ↓ Feature Engineering (falls Modell damit trainiert)
    ↓ create_labels() → Labels erstellen (gleiche Logik wie Training)
    ↓ model.predict() → Vorhersagen machen
    ↓ Metriken berechnen
    ↓ Train vs. Test Vergleich
    ↓ Overlap-Check
    ↓
Job Manager [app/queue/job_manager.py] process_test_job()
    ↓ get_or_create_test_result() → Test in ml_test_results speichern
    ↓ update_job_status(status='COMPLETED')
```

---

## 4. Validierung und Prüfmethoden

### 4.1 Wie prüfe ich, dass Werte richtig sind?

#### 4.1.1 Modell-Metriken prüfen

**1. Training Accuracy sollte realistisch sein:**
- ✅ Zwischen 0.5 (Zufall) und 1.0 (perfekt)
- ⚠️ > 0.95 kann Overfitting bedeuten
- ⚠️ < 0.6 kann bedeuten, dass Modell nicht lernt

**2. F1-Score sollte > 0.5 sein:**
- F1-Score ist wichtiger als Accuracy bei imbalanced data
- Gute Werte: > 0.6

**3. Confusion Matrix prüfen:**
```python
# Aus ml_models oder ml_test_results
tp = model['confusion_matrix']['tp']
tn = model['confusion_matrix']['tn']
fp = model['confusion_matrix']['fp']
fn = model['confusion_matrix']['fn']

# Prüfe:
total = tp + tn + fp + fn
assert total > 0, "Keine Samples!"
assert tp + fn > 0, "Keine positiven Labels!"
assert tn + fp > 0, "Keine negativen Labels!"

# Accuracy sollte stimmen:
calculated_accuracy = (tp + tn) / total
assert abs(calculated_accuracy - model['training_accuracy']) < 0.001
```

**4. Feature Importance prüfen:**
```python
# Feature Importance sollte summiert 1.0 ergeben
feature_importance = model['feature_importance']
total_importance = sum(feature_importance.values())
assert abs(total_importance - 1.0) < 0.01, f"Feature Importance summiert nicht zu 1.0: {total_importance}"
```

#### 4.1.2 Test-Metriken prüfen

**1. Train vs. Test Vergleich:**
```python
# Degradation sollte nicht zu groß sein
accuracy_degradation = test_result['accuracy_degradation']
if accuracy_degradation > 0.1:
    print("⚠️ OVERFITTING: Modell generalisiert schlecht!")
```

**2. Overlap-Check:**
```python
# Test sollte NICHT mit Training überlappen
if test_result['has_overlap']:
    print("⚠️ WARNUNG: Test überlappt mit Training (Data Leakage möglich!)")
```

**3. Test-Zeitraum Validierung:**
```python
# Test sollte mindestens 1 Tag lang sein
test_duration_days = test_result['test_duration_days']
if test_duration_days < 1.0:
    print("⚠️ WARNUNG: Test-Zeitraum zu kurz!")
```

#### 4.1.3 Daten-Validierung

**1. Prüfe ob Daten geladen wurden:**
```python
# In Logs suchen nach:
# "✅ 63893 Zeilen geladen (nach Duplikat-Entfernung)"
# Sollte > 0 sein
```

**2. Prüfe Label-Balance:**
```python
# In Logs suchen nach:
# "📊 Label-Balance: 34.80% positive, 65.20% negative"
# Sollte nicht zu unausgewogen sein (< 10% oder > 90%)
```

**3. Prüfe Features:**
```python
# In Logs suchen nach:
# "📊 Features: ['price_open', 'price_high', ...]"
# Sollte nur Features enthalten, die in coin_metrics existieren
```

---

### 4.2 SQL-Queries zum Prüfen

#### 4.2.1 Prüfe Modell-Details
```sql
SELECT 
    id, name, model_type, status,
    train_start, train_end,
    training_accuracy, training_f1,
    future_minutes, price_change_percent,
    created_at
FROM ml_models
WHERE id = 123;
```

#### 4.2.2 Prüfe Test-Ergebnisse
```sql
SELECT 
    id, model_id, test_start, test_end,
    accuracy, f1_score, precision_score, recall,
    tp, tn, fp, fn,
    num_samples, num_positive, num_negative,
    has_overlap, accuracy_degradation, is_overfitted
FROM ml_test_results
WHERE model_id = 123
ORDER BY created_at DESC;
```

#### 4.2.3 Prüfe auf Duplikate
```sql
-- Sollte 0 Zeilen zurückgeben (UNIQUE Constraint)
SELECT model_id, test_start, test_end, COUNT(*)
FROM ml_test_results
GROUP BY model_id, test_start, test_end
HAVING COUNT(*) > 1;
```

#### 4.2.4 Prüfe Job-Status
```sql
SELECT 
    id, job_type, status, progress,
    result_model_id, result_test_id,
    error_msg, created_at, completed_at
FROM ml_jobs
WHERE job_type = 'TRAIN'
ORDER BY created_at DESC
LIMIT 10;
```

---

### 4.3 Code-Validierung

#### 4.3.1 Prüfe ob Modell korrekt gespeichert wurde
```python
# API Call
response = requests.get(f"{API_BASE_URL}/models/{model_id}")
model = response.json()

# Prüfe:
assert model['status'] == 'READY'
assert model['model_file_path'] is not None
assert os.path.exists(model['model_file_path'])
assert model['training_accuracy'] is not None
assert model['training_f1'] is not None
```

#### 4.3.2 Prüfe ob Test korrekt gespeichert wurde
```python
# API Call
response = requests.get(f"{API_BASE_URL}/test-results/{test_id}")
test = response.json()

# Prüfe:
assert test['model_id'] == model_id
assert test['accuracy'] is not None
assert test['f1_score'] is not None
assert test['num_samples'] > 0
assert test['tp'] + test['tn'] + test['fp'] + test['fn'] == test['num_samples']
```

---

## 5. Potenzielle Fehlerquellen

### 5.1 Data Leakage

**Problem:** `target_var` wird als Feature verwendet

**Lösung:** `prepare_features_for_training()` entfernt `target_var` aus Features bei zeitbasierter Vorhersage

**Code-Referenz:**
```95:128:app/training/engine.py
def prepare_features_for_training(...):
    # ... Entfernt target_var aus Features bei zeitbasierter Vorhersage
```

---

### 5.2 Race Conditions

**Problem:** Mehrere Worker verarbeiten denselben Job

**Lösung:** `get_next_pending_job()` setzt Status ATOMIC auf RUNNING

**Code-Referenz:**
```983:995:app/database/models.py
async def get_next_pending_job():
    # ... UPDATE mit RETURNING (atomic)
```

---

### 5.3 Duplikate

**Problem:** Mehrere Modelle/Tests werden erstellt

**Lösung:** 
- UNIQUE Constraint auf `ml_test_results(model_id, test_start, test_end)`
- `get_or_create_test_result()` prüft zuerst, ob existiert

**Code-Referenz:**
```359:431:app/database/models.py
async def get_or_create_test_result(...):
    # ... Prüfe ob existiert
    # ... ON CONFLICT DO UPDATE
```

---

### 5.4 Fehlende Features

**Problem:** Feature existiert nicht in `coin_metrics`

**Lösung:** Validierung in `load_training_data()` prüft ob Features existieren

**Code-Referenz:**
```50:130:app/training/feature_engineering.py
async def load_training_data(...):
    # ... SQL Query mit Feature-Liste
```

---

### 5.5 Label-Imbalance

**Problem:** Keine positiven oder negativen Labels

**Lösung:** Validierung in `train_model_sync()` prüft Label-Balance

**Code-Referenz:**
```188:199:app/training/engine.py
positive_count = labels.sum()
negative_count = len(labels) - positive_count
if positive_count == 0:
    raise ValueError("Keine positiven Labels gefunden!")
```

---

## 6. Code-Referenzen

### 6.1 Modell-Erstellung - Code-Pfad

1. **API:** `app/api/routes.py:42-114` → `create_model_job()`
2. **Job Worker:** `app/queue/job_manager.py:557-600` → `start_worker()`
3. **Job Processing:** `app/queue/job_manager.py:88-275` → `process_train_job()`
4. **Training Engine:** `app/training/engine.py:479-586` → `train_model()`
5. **Data Loading:** `app/training/feature_engineering.py:50-130` → `load_training_data()`
6. **Label Creation:** `app/training/feature_engineering.py:133-250` → `create_labels()` / `create_time_based_labels()`
7. **Feature Engineering:** `app/training/feature_engineering.py:260-420` → `create_pump_detection_features()`
8. **Model Training:** `app/training/engine.py:130-450` → `train_model_sync()`
9. **Database:** `app/database/models.py:63-220` → `create_model()`

### 6.2 Modell-Testing - Code-Pfad

1. **API:** `app/api/routes.py:173-208` → `test_model_job()`
2. **Job Worker:** `app/queue/job_manager.py:557-600` → `start_worker()`
3. **Job Processing:** `app/queue/job_manager.py:277-368` → `process_test_job()`
4. **Test Engine:** `app/training/model_loader.py:43-353` → `test_model()`
5. **Model Loading:** `app/training/model_loader.py:18-41` → `load_model()`
6. **Data Loading:** `app/training/feature_engineering.py:50-130` → `load_training_data()`
7. **Label Creation:** `app/training/model_loader.py:141-203` → `create_labels()` / `create_time_based_labels()`
8. **Metrics Calculation:** `app/training/model_loader.py:215-320` → Metriken berechnen
9. **Database:** `app/database/models.py:359-431` → `get_or_create_test_result()`

---

## 7. Zusammenfassung - Wichtige Punkte

### ✅ Was funktioniert garantiert:

1. **UNIQUE Constraint:** Verhindert Duplikate in `ml_test_results`
2. **Atomic Status Update:** Verhindert Race Conditions
3. **Data Leakage Prevention:** `target_var` wird nicht als Feature verwendet
4. **Label Validation:** Prüft ob positive/negative Labels vorhanden sind
5. **Feature Validation:** Prüft ob Features in Daten vorhanden sind

### ⚠️ Was zu prüfen ist:

1. **Label-Balance:** Sollte nicht zu unausgewogen sein (< 10% oder > 90%)
2. **Overfitting:** `accuracy_degradation` sollte < 0.1 sein
3. **Test-Overlap:** `has_overlap` sollte `false` sein
4. **Test-Duration:** Sollte mindestens 1 Tag sein
5. **Feature Existence:** Features müssen in `coin_metrics` existieren

### 🔍 Prüfmethoden:

1. **Logs prüfen:** Suche nach Warnungen und Fehlern
2. **API prüfen:** Hole Modell/Test-Details über API
3. **SQL prüfen:** Direkte Datenbank-Abfragen
4. **Metriken prüfen:** Berechne Metriken manuell und vergleiche
5. **Confusion Matrix prüfen:** `tp + tn + fp + fn` sollte `num_samples` entsprechen

---

**Erstellt:** 2025-12-24  
**Status:** ✅ Vollständig dokumentiert

