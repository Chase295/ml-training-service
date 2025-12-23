# 📚 Komplette Dokumentation: Modell-Test & Modell-Vergleich - UI, API & Backend

## 🎯 Übersicht

Diese Dokumentation beschreibt **alle Möglichkeiten und Bedingungen** zum Testen von ML-Modellen und zum Vergleichen von zwei Modellen über die **Streamlit UI** und die **REST API**, sowie die **Backend-Logik** und **Datenbank-Speicherung**.

---

## 📋 Inhaltsverzeichnis

1. [Modell-Test - Übersicht](#1-modell-test---übersicht)
2. [Modell-Test über UI](#2-modell-test-über-ui)
3. [Modell-Test über API](#3-modell-test-über-api)
4. [Modell-Vergleich - Übersicht](#4-modell-vergleich---übersicht)
5. [Modell-Vergleich über UI](#5-modell-vergleich-über-ui)
6. [Modell-Vergleich über API](#6-modell-vergleich-über-api)
7. [Backend-Logik - Test](#7-backend-logik---test)
8. [Backend-Logik - Vergleich](#8-backend-logik---vergleich)
9. [Berechnete Metriken](#9-berechnete-metriken)
10. [Overlap-Check](#10-overlap-check)
11. [Datenbank-Speicherung](#11-datenbank-speicherung)
12. [Validierungen & Bedingungen](#12-validierungen--bedingungen)

---

## 1. Modell-Test - Übersicht

### 1.1 Was ist Modell-Test?

**Beschreibung:** Ein Modell wird auf **neuen Daten** (Test-Daten) evaluiert, die **nicht** für das Training verwendet wurden. Das Modell macht Vorhersagen auf diesen Daten, und die Ergebnisse werden mit den tatsächlichen Labels verglichen.

**Zweck:**
- Bewertung der **Generalisierung** des Modells
- Prüfung der **Performance** auf unbekannten Daten
- Identifikation von **Overfitting** (wenn Test-Accuracy deutlich niedriger als Training-Accuracy)

### 1.2 Voraussetzungen

**Erforderlich:**
- ✅ Modell muss existieren (`ml_models.id`)
- ✅ Modell-Status muss `READY` sein (nicht `TRAINING` oder `FAILED`)
- ✅ Modell-Datei muss vorhanden sein (`model_file_path`)
- ✅ Test-Zeitraum muss definiert sein (`test_start`, `test_end`)
- ✅ Test-Daten müssen im angegebenen Zeitraum vorhanden sein

**Optional:**
- ⚠️ **Overlap-Check:** Prüft ob Test-Daten mit Trainingsdaten überlappen (wird gewarnt, aber Test wird trotzdem ausgeführt)

---

## 2. Modell-Test über UI

### 2.1 Streamlit UI - Seite "Modell testen"

**URL:** `http://localhost:8501` → Navigation: "🧪 Modell testen"

**Alternative Zugriffe:**
- Von Übersichtsseite: Modell auswählen → "🧪 Testen" Button klicken
- Modell wird automatisch vorausgewählt (`st.session_state['test_model_id']`)

### 2.2 Formular-Felder

#### 2.2.1 Modell-Auswahl

| Feld | Typ | Erforderlich | Beschreibung |
|------|-----|--------------|--------------|
| `Modell auswählen` | Selectbox | ✅ Ja | Nur Modelle mit Status `READY` werden angezeigt |

**Anzeige-Format:** `"{name} ({model_type})"` (z.B. `"PumpDetector_v1 (random_forest)"`)

**Filter:**
- Nur Modelle mit `status == 'READY'`
- Nur Modelle mit `is_deleted == False`

#### 2.2.2 Test-Zeitraum

| Feld | Typ | Erforderlich | Beschreibung |
|------|-----|--------------|--------------|
| `Start-Datum` | Date-Input | ✅ Ja | UTC-Zeitzone, Standard: Heute - 7 Tage |
| `Start-Uhrzeit` | Time-Input | ✅ Ja | UTC-Zeitzone |
| `Ende-Datum` | Date-Input | ✅ Ja | UTC-Zeitzone, Standard: Heute |
| `Ende-Uhrzeit` | Time-Input | ✅ Ja | UTC-Zeitzone |

**⚠️ WICHTIG:**
- Datum und Uhrzeit werden zu `datetime` kombiniert
- Zeitzone wird auf `UTC` gesetzt: `.replace(tzinfo=timezone.utc)`
- Format: `ISO 8601` mit UTC (z.B. `"2024-01-01T00:00:00+00:00"`)

#### 2.2.3 Overlap-Warnung

**Wenn Test-Zeitraum mit Trainingsdaten überlappt:**
- ⚠️ Warnung wird angezeigt: `"{X}% Überschneidung mit Trainingsdaten (Test wird trotzdem ausgeführt)"`
- Berechnung:
  ```python
  overlap_duration = min(test_end, train_end) - max(test_start, train_start)
  total_duration = test_end - test_start
  overlap_pct = (overlap_duration / total_duration) * 100
  ```

**⚠️ WICHTIG:**
- Test wird **trotzdem ausgeführt**, auch bei Overlap
- Overlap wird in `ml_test_results.has_overlap` gespeichert
- Detaillierte Overlap-Info in `ml_test_results.overlap_note`

### 2.3 Formular-Validierung (UI)

**Vor dem Absenden:**
1. ✅ Modell muss ausgewählt sein
2. ✅ `test_start` < `test_end`

### 2.4 API-Aufruf (UI → Backend)

**Endpoint:** `POST /api/models/{model_id}/test`

**Request Body:**
```json
{
  "test_start": "2024-02-01T00:00:00Z",
  "test_end": "2024-02-07T23:59:59Z"
}
```

**Response:**
```json
{
  "job_id": 456,
  "message": "Test-Job erstellt für Modell 123",
  "status": "PENDING"
}
```

**⚠️ WICHTIG:**
- Test wird **NICHT sofort** ausgeführt!
- Ein **TEST-Job** wird in `ml_jobs` erstellt
- Job wird asynchron vom **Job Worker** verarbeitet
- Status kann über `/api/jobs/{job_id}` abgefragt werden

### 2.5 UI-Feedback

**Nach erfolgreicher Job-Erstellung:**
- ✅ Erfolgsmeldung: `"Test-Job erstellt! Job-ID: {job_id}"`
- 📊 Status-Info: `"Status: PENDING. Der Test wird jetzt ausgeführt."`
- Button: "📊 Zu Jobs anzeigen" → Navigiert zu Jobs-Seite

---

## 3. Modell-Test über API

### 3.1 REST API Endpoint

**Endpoint:** `POST /api/models/{model_id}/test`

**Base URL:** `http://localhost:8000`

**Content-Type:** `application/json`

### 3.2 Request Schema

**Pydantic Schema:** `TestModelRequest`

```python
{
  # Erforderlich
  "test_start": datetime,  # ISO-Format mit UTC
  "test_end": datetime      # ISO-Format mit UTC
}
```

**⚠️ WICHTIG:**
- `model_id` ist **NICHT** im Request Body, sondern im **URL-Pfad**!

### 3.3 Validierungen (API)

**Pydantic Validatoren:**

1. ✅ `test_start`, `test_end` werden zu UTC konvertiert (tz-aware)
2. ✅ Modell muss existieren (`model_id` in `ml_models`)
3. ✅ Modell darf nicht gelöscht sein (`is_deleted == False`)

### 3.4 Beispiel-Request

```bash
curl -X POST "http://localhost:8000/api/models/123/test" \
  -H "Content-Type: application/json" \
  -d '{
    "test_start": "2024-02-01T00:00:00Z",
    "test_end": "2024-02-07T23:59:59Z"
  }'
```

### 3.5 Response

**Schema:** `CreateJobResponse`

```json
{
  "job_id": 456,
  "message": "Test-Job erstellt für Modell 123",
  "status": "PENDING"
}
```

**Status-Codes:**
- `201 Created`: Job erfolgreich erstellt
- `404 Not Found`: Modell nicht gefunden oder gelöscht
- `400 Bad Request`: Validierungsfehler
- `500 Internal Server Error`: Server-Fehler

---

## 4. Modell-Vergleich - Übersicht

### 4.1 Was ist Modell-Vergleich?

**Beschreibung:** Zwei Modelle werden auf **denselben Test-Daten** evaluiert und ihre Performance-Metriken werden verglichen. Das Modell mit dem **höheren F1-Score** gewinnt.

**Zweck:**
- **Direkter Vergleich** von zwei Modellen
- Identifikation des **besseren Modells** für einen bestimmten Anwendungsfall
- Bewertung von **Modell-Änderungen** (z.B. verschiedene Hyperparameter)

### 4.2 Voraussetzungen

**Erforderlich:**
- ✅ Beide Modelle müssen existieren (`model_a_id`, `model_b_id`)
- ✅ Beide Modelle müssen unterschiedlich sein (`model_a_id != model_b_id`)
- ✅ Beide Modell-Status müssen `READY` sein
- ✅ Beide Modell-Dateien müssen vorhanden sein
- ✅ Test-Zeitraum muss definiert sein (`test_start`, `test_end`)
- ✅ Test-Daten müssen im angegebenen Zeitraum vorhanden sein

**Optional:**
- ⚠️ **Overlap-Check:** Prüft ob Test-Daten mit Trainingsdaten überlappen (für beide Modelle)

---

## 5. Modell-Vergleich über UI

### 5.1 Streamlit UI - Seite "Modelle vergleichen"

**URL:** `http://localhost:8501` → Navigation: "⚔️ Modelle vergleichen"

**Alternative Zugriffe:**
- Von Übersichtsseite: 2 Modelle auswählen → "⚔️ Vergleichen" Button klicken
- Modelle werden automatisch vorausgewählt (`st.session_state['compare_model_a_id']`, `st.session_state['compare_model_b_id']`)

### 5.2 Formular-Felder

#### 5.2.1 Modell-Auswahl

| Feld | Typ | Erforderlich | Beschreibung |
|------|-----|--------------|--------------|
| `Modell A` | Selectbox | ✅ Ja | Erstes Modell (nur `READY` Modelle) |
| `Modell B` | Selectbox | ✅ Ja | Zweites Modell (nur `READY` Modelle, darf nicht Modell A sein) |

**Anzeige-Format:** `"{name} ({model_type})"` (z.B. `"PumpDetector_v1 (random_forest)"`)

**Filter:**
- Nur Modelle mit `status == 'READY'`
- Nur Modelle mit `is_deleted == False`
- Modell B darf nicht Modell A sein (automatisch gefiltert)

**Validierung:**
- Mindestens 2 Modelle müssen verfügbar sein

#### 5.2.2 Test-Zeitraum

| Feld | Typ | Erforderlich | Beschreibung |
|------|-----|--------------|--------------|
| `Start-Datum` | Date-Input | ✅ Ja | UTC-Zeitzone, Standard: Heute - 7 Tage |
| `Start-Uhrzeit` | Time-Input | ✅ Ja | UTC-Zeitzone |
| `Ende-Datum` | Date-Input | ✅ Ja | UTC-Zeitzone, Standard: Heute |
| `Ende-Uhrzeit` | Time-Input | ✅ Ja | UTC-Zeitzone |

**⚠️ WICHTIG:**
- Gleicher Test-Zeitraum für **beide Modelle**
- Datum und Uhrzeit werden zu `datetime` kombiniert
- Zeitzone wird auf `UTC` gesetzt

### 5.3 Formular-Validierung (UI)

**Vor dem Absenden:**
1. ✅ Mindestens 2 Modelle müssen verfügbar sein
2. ✅ Modell A und Modell B müssen unterschiedlich sein
3. ✅ `test_start` < `test_end`

### 5.4 API-Aufruf (UI → Backend)

**Endpoint:** `POST /api/models/compare`

**Request Body:**
```json
{
  "model_a_id": 123,
  "model_b_id": 456,
  "test_start": "2024-02-01T00:00:00Z",
  "test_end": "2024-02-07T23:59:59Z"
}
```

**Response:**
```json
{
  "job_id": 789,
  "message": "Vergleichs-Job erstellt für Modell 123 vs 456",
  "status": "PENDING"
}
```

**⚠️ WICHTIG:**
- Vergleich wird **NICHT sofort** ausgeführt!
- Ein **COMPARE-Job** wird in `ml_jobs` erstellt
- Job wird asynchron vom **Job Worker** verarbeitet
- Status kann über `/api/jobs/{job_id}` abgefragt werden

### 5.5 UI-Feedback

**Nach erfolgreicher Job-Erstellung:**
- ✅ Erfolgsmeldung: `"Vergleichs-Job erstellt! Job-ID: {job_id}"`
- 📊 Status-Info: `"Status: PENDING. Der Vergleich wird jetzt ausgeführt."`
- Button: "📊 Zu Jobs anzeigen" → Navigiert zu Jobs-Seite

---

## 6. Modell-Vergleich über API

### 6.1 REST API Endpoint

**Endpoint:** `POST /api/models/compare`

**Base URL:** `http://localhost:8000`

**Content-Type:** `application/json`

### 6.2 Request Schema

**Pydantic Schema:** `CompareModelsRequest`

```python
{
  # Erforderlich
  "model_a_id": int,        # ID des ersten Modells
  "model_b_id": int,        # ID des zweiten Modells (muss != model_a_id sein)
  "test_start": datetime,   # ISO-Format mit UTC
  "test_end": datetime      # ISO-Format mit UTC
}
```

### 6.3 Validierungen (API)

**Pydantic Validatoren:**

1. ✅ `test_start`, `test_end` werden zu UTC konvertiert (tz-aware)
2. ✅ `model_a_id` und `model_b_id` müssen unterschiedlich sein
3. ✅ Beide Modelle müssen existieren (`model_a_id`, `model_b_id` in `ml_models`)
4. ✅ Beide Modelle dürfen nicht gelöscht sein (`is_deleted == False`)

### 6.4 Beispiel-Request

```bash
curl -X POST "http://localhost:8000/api/models/compare" \
  -H "Content-Type: application/json" \
  -d '{
    "model_a_id": 123,
    "model_b_id": 456,
    "test_start": "2024-02-01T00:00:00Z",
    "test_end": "2024-02-07T23:59:59Z"
  }'
```

### 6.5 Response

**Schema:** `CreateJobResponse`

```json
{
  "job_id": 789,
  "message": "Vergleichs-Job erstellt für Modell 123 vs 456",
  "status": "PENDING"
}
```

**Status-Codes:**
- `201 Created`: Job erfolgreich erstellt
- `404 Not Found`: Eines oder beide Modelle nicht gefunden
- `400 Bad Request`: Validierungsfehler (z.B. `model_a_id == model_b_id`)
- `500 Internal Server Error`: Server-Fehler

---

## 7. Backend-Logik - Test

### 7.1 Job-Erstellung (API → Database)

**Datei:** `app/api/routes.py` → `test_model_job()`

**Ablauf:**
1. Request wird validiert (Pydantic)
2. Modell wird geprüft (existiert, nicht gelöscht)
3. Job wird in `ml_jobs` erstellt:
   ```python
   job_id = await create_job(
       job_type="TEST",
       priority=5,
       test_model_id=model_id,
       test_start=request.test_start,
       test_end=request.test_end
   )
   ```
4. Response mit `job_id` wird zurückgegeben

**⚠️ WICHTIG:**
- Test wird **NICHT sofort** ausgeführt!
- Job wird asynchron vom **Job Worker** verarbeitet

### 7.2 Job-Verarbeitung (Worker)

**Datei:** `app/queue/job_manager.py` → `process_test_job()`

**Ablauf:**
1. Job wird aus `ml_jobs` geladen
2. Parameter werden extrahiert:
   - `model_id`, `test_start`, `test_end`
3. `test_model()` wird aufgerufen (async)
4. Test-Ergebnis wird erhalten:
   - Metriken: `accuracy`, `f1_score`, `precision_score`, `recall`, `roc_auc`
   - Confusion Matrix: `tp`, `tn`, `fp`, `fn`
   - Samples: `num_samples`, `num_positive`, `num_negative`
   - Overlap-Info: `has_overlap`, `overlap_note`
5. Test-Ergebnis wird in `ml_test_results` gespeichert:
   ```python
   test_id = await create_test_result(
       model_id=model_id,
       test_start=test_start_dt,
       test_end=test_end_dt,
       accuracy=test_result['accuracy'],
       f1_score=test_result['f1_score'],
       precision_score=test_result['precision_score'],
       recall=test_result['recall'],
       roc_auc=test_result.get('roc_auc'),
       tp=test_result['tp'],
       tn=test_result['tn'],
       fp=test_result['fp'],
       fn=test_result['fn'],
       num_samples=test_result['num_samples'],
       num_positive=test_result['num_positive'],
       num_negative=test_result['num_negative'],
       has_overlap=test_result['has_overlap'],
       overlap_note=test_result.get('overlap_note')
   )
   ```
6. Job-Status wird auf `COMPLETED` gesetzt, `result_test_id` wird gesetzt

### 7.3 Test-Funktion

**Datei:** `app/training/model_loader.py` → `test_model()`

**Ablauf:**
1. **Modell laden:**
   - Modell-Info aus `ml_models` holen
   - Modell-Datei laden (`joblib.load(model_file_path)`)
   - Features und Phasen aus JSONB extrahieren

2. **Modell-Typ bestimmen:**
   - Prüfe ob zeitbasierte Vorhersage: `target_operator is None or target_value is None`
   - Wenn zeitbasiert: Hole Parameter aus `params["_time_based"]`

3. **Test-Daten laden:**
   ```python
   test_data = await load_training_data(
       train_start=test_start,  # Wird als test_start verwendet
       train_end=test_end,      # Wird als test_end verwendet
       features=features_with_target,  # target_var wird hinzugefügt
       phases=phases
   )
   ```

4. **Labels erstellen:**
   - **Wenn zeitbasierte Vorhersage:**
     ```python
     labels = create_time_based_labels(
         test_data,
         model['target_variable'],
         future_minutes,
         min_percent_change,
         direction,
         phase_intervals  # {phase_id: interval_seconds}
     )
     ```
   - **Wenn klassische Vorhersage:**
     ```python
     labels = create_labels(
         test_data,
         model['target_variable'],
         model['target_operator'],
         float(model['target_value'])
     )
     ```

5. **Vorhersagen machen:**
   ```python
   X_test = test_data[features].values  # Nur Features, nicht target_var
   y_test = labels.values
   y_pred = model_obj.predict(X_test)
   y_pred_proba = model_obj.predict_proba(X_test)[:, 1]  # Falls verfügbar
   ```

6. **Metriken berechnen:**
   - `accuracy_score(y_test, y_pred)`
   - `f1_score(y_test, y_pred)`
   - `precision_score(y_test, y_pred)`
   - `recall_score(y_test, y_pred)`
   - `roc_auc_score(y_test, y_pred_proba)` (falls `predict_proba` verfügbar)

7. **Confusion Matrix:**
   ```python
   cm = confusion_matrix(y_test, y_pred)
   tn, fp, fn, tp = cm.ravel()  # 2x2 Matrix
   ```

8. **Overlap-Check:**
   ```python
   overlap_info = check_overlap(
       train_start=model['train_start'],
       train_end=model['train_end'],
       test_start=test_start,
       test_end=test_end
   )
   ```

9. **Ergebnis zurückgeben:**
   ```python
   return {
       "accuracy": float(accuracy),
       "f1_score": float(f1),
       "precision_score": float(precision),
       "recall": float(recall),
       "roc_auc": float(roc_auc) if roc_auc else None,
       "tp": int(tp),
       "tn": int(tn),
       "fp": int(fp),
       "fn": int(fn),
       "num_samples": len(test_data),
       "num_positive": int(labels.sum()),
       "num_negative": int(len(labels) - labels.sum()),
       "has_overlap": overlap_info['has_overlap'],
       "overlap_note": overlap_info['overlap_note']
   }
   ```

**⚠️ WICHTIG:**
- `target_var` wird zu Features hinzugefügt (für Label-Erstellung benötigt)
- Aber: Vorhersagen werden nur mit **originalen Features** gemacht (ohne `target_var`)
- Zeitbasierte Parameter werden aus `params["_time_based"]` geholt
- Phase-Intervalle werden aus `ref_coin_phases` geladen

---

## 8. Backend-Logik - Vergleich

### 8.1 Job-Erstellung (API → Database)

**Datei:** `app/api/routes.py` → `compare_models_job()`

**Ablauf:**
1. Request wird validiert (Pydantic)
2. Beide Modelle werden geprüft (existieren, nicht gelöscht, unterschiedlich)
3. Job wird in `ml_jobs` erstellt:
   ```python
   job_id = await create_job(
       job_type="COMPARE",
       priority=5,
       compare_model_a_id=request.model_a_id,
       compare_model_b_id=request.model_b_id,
       compare_start=request.test_start,
       compare_end=request.test_end
   )
   ```
4. Response mit `job_id` wird zurückgegeben

**⚠️ WICHTIG:**
- Vergleich wird **NICHT sofort** ausgeführt!
- Job wird asynchron vom **Job Worker** verarbeitet

### 8.2 Job-Verarbeitung (Worker)

**Datei:** `app/queue/job_manager.py` → `process_compare_job()`

**Ablauf:**
1. Job wird aus `ml_jobs` geladen
2. Parameter werden extrahiert:
   - `model_a_id`, `model_b_id`, `test_start`, `test_end`
3. **Modell A testen:**
   ```python
   result_a = await test_model(
       model_id=model_a_id,
       test_start=test_start,
       test_end=test_end,
       model_storage_path=MODEL_STORAGE_PATH
   )
   ```
4. **Modell B testen:**
   ```python
   result_b = await test_model(
       model_id=model_b_id,
       test_start=test_start,
       test_end=test_end,
       model_storage_path=MODEL_STORAGE_PATH
   )
   ```
5. **Gewinner bestimmen:**
   ```python
   winner_id = None
   if result_a['f1_score'] > result_b['f1_score']:
       winner_id = model_a_id
   elif result_b['f1_score'] > result_a['f1_score']:
       winner_id = model_b_id
   # Bei Gleichstand: winner_id bleibt None (Unentschieden)
   ```
6. Vergleich wird in `ml_comparisons` gespeichert:
   ```python
   comparison_id = await create_comparison(
       model_a_id=model_a_id,
       model_b_id=model_b_id,
       test_start=test_start_dt,
       test_end=test_end_dt,
       num_samples=result_a['num_samples'],  # Beide sollten gleich sein
       a_accuracy=result_a['accuracy'],
       a_f1=result_a['f1_score'],
       a_precision=result_a['precision_score'],
       a_recall=result_a['recall'],
       b_accuracy=result_b['accuracy'],
       b_f1=result_b['f1_score'],
       b_precision=result_b['precision_score'],
       b_recall=result_b['recall'],
       winner_id=winner_id
   )
   ```
7. Job-Status wird auf `COMPLETED` gesetzt, `result_comparison_id` wird gesetzt

**⚠️ WICHTIG:**
- Beide Modelle werden auf **denselben Test-Daten** getestet
- Gewinner wird basierend auf **F1-Score** bestimmt
- Bei Gleichstand: `winner_id = None` (Unentschieden)

---

## 9. Berechnete Metriken

### 9.1 Metriken für Modell-Test

**Alle Metriken werden für jedes Test-Ergebnis berechnet:**

| Metrik | Beschreibung | Formel | Bereich |
|--------|--------------|--------|---------|
| `accuracy` | Genauigkeit | `(tp + tn) / (tp + tn + fp + fn)` | `0.0` - `1.0` |
| `f1_score` | F1-Score (Harmonisches Mittel) | `2 * (precision * recall) / (precision + recall)` | `0.0` - `1.0` |
| `precision_score` | Präzision | `tp / (tp + fp)` | `0.0` - `1.0` |
| `recall` | Recall (Sensitivity) | `tp / (tp + fn)` | `0.0` - `1.0` |
| `roc_auc` | ROC-AUC (optional) | Fläche unter ROC-Kurve | `0.0` - `1.0` |

**Confusion Matrix:**
- `tp` (True Positives): Korrekt als positiv vorhergesagt
- `tn` (True Negatives): Korrekt als negativ vorhergesagt
- `fp` (False Positives): Falsch als positiv vorhergesagt
- `fn` (False Negatives): Falsch als negativ vorhergesagt

**Samples:**
- `num_samples`: Gesamtanzahl Test-Samples
- `num_positive`: Anzahl positiver Labels (`1`)
- `num_negative`: Anzahl negativer Labels (`0`)

### 9.2 Metriken für Modell-Vergleich

**Für beide Modelle werden folgende Metriken gespeichert:**

| Metrik | Beschreibung | Modell A | Modell B |
|--------|--------------|----------|----------|
| `a_accuracy` / `b_accuracy` | Genauigkeit | ✅ | ✅ |
| `a_f1` / `b_f1` | F1-Score | ✅ | ✅ |
| `a_precision` / `b_precision` | Präzision | ✅ | ✅ |
| `a_recall` / `b_recall` | Recall | ✅ | ✅ |

**Gewinner:**
- `winner_id`: ID des Modells mit höherem F1-Score
- `None`: Unentschieden (beide haben gleichen F1-Score)

**⚠️ WICHTIG:**
- Gewinner wird **nur** basierend auf **F1-Score** bestimmt
- Andere Metriken werden gespeichert, aber nicht für Gewinner-Entscheidung verwendet

---

## 10. Overlap-Check

### 10.1 Was ist Overlap?

**Beschreibung:** Overlap tritt auf, wenn der **Test-Zeitraum** mit dem **Trainings-Zeitraum** überlappt. Dies kann zu **verfälschten Ergebnissen** führen, da das Modell auf Daten getestet wird, die es bereits während des Trainings gesehen hat.

### 10.2 Overlap-Berechnung

**Datei:** `app/training/feature_engineering.py` → `check_overlap()`

**Logik:**
```python
def check_overlap(train_start, train_end, test_start, test_end):
    # Konvertiere zu UTC
    train_start_utc = _ensure_utc(train_start)
    train_end_utc = _ensure_utc(train_end)
    test_start_utc = _ensure_utc(test_start)
    test_end_utc = _ensure_utc(test_end)
    
    # Prüfe ob Overlap existiert
    if test_start_utc < train_end_utc and test_end_utc > train_start_utc:
        # Berechne Overlap-Dauer
        overlap_start = max(test_start_utc, train_start_utc)
        overlap_end = min(test_end_utc, train_end_utc)
        overlap_duration = overlap_end - overlap_start
        
        # Berechne Test-Dauer
        test_duration = test_end_utc - test_start_utc
        
        # Berechne Overlap-Prozent
        overlap_percent = (overlap_duration.total_seconds() / test_duration.total_seconds()) * 100
        
        return {
            "has_overlap": True,
            "overlap_note": f"⚠️ {overlap_percent:.1f}% Überschneidung mit Trainingsdaten - Ergebnisse können verfälscht sein"
        }
    else:
        return {
            "has_overlap": False,
            "overlap_note": "✅ Keine Überschneidung mit Trainingsdaten"
        }
```

### 10.3 Overlap-Speicherung

**In `ml_test_results`:**
- `has_overlap`: `BOOLEAN` (wird in DB gespeichert)
- `overlap_note`: `TEXT` (wird in DB gespeichert)

**⚠️ WICHTIG:**
- Overlap wird **nur gewarnt**, Test wird **trotzdem ausgeführt**
- Overlap-Info wird in `ml_test_results` gespeichert
- Bei Modell-Vergleich: Overlap wird für **beide Modelle** separat geprüft (aber nicht in `ml_comparisons` gespeichert)

---

## 11. Datenbank-Speicherung

### 11.1 Tabelle: `ml_test_results`

**Schema:**
```sql
CREATE TABLE ml_test_results (
    id BIGINT PRIMARY KEY,
    model_id BIGINT NOT NULL REFERENCES ml_models(id) ON DELETE CASCADE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Zeitraum
    test_start TIMESTAMP WITH TIME ZONE NOT NULL,
    test_end TIMESTAMP WITH TIME ZONE NOT NULL,
    
    -- Metriken
    accuracy NUMERIC(5, 4),
    f1_score NUMERIC(5, 4),
    precision_score NUMERIC(5, 4),
    recall NUMERIC(5, 4),
    roc_auc NUMERIC(5, 4),
    
    -- Confusion Matrix
    tp INT,
    tn INT,
    fp INT,
    fn INT,
    
    -- Samples
    num_samples INT,
    num_positive INT,
    num_negative INT,
    
    -- Overlap-Check
    has_overlap BOOLEAN DEFAULT FALSE,
    overlap_note TEXT,
    
    -- Feature Importance (optional)
    feature_importance JSONB  -- {"price_open": 0.35, ...}
);
```

**Beispiel-Datensatz:**
```json
{
  "id": 1,
  "model_id": 123,
  "created_at": "2024-02-08T10:00:00+00:00",
  "test_start": "2024-02-01T00:00:00+00:00",
  "test_end": "2024-02-07T23:59:59+00:00",
  "accuracy": 0.8523,
  "f1_score": 0.7891,
  "precision_score": 0.8123,
  "recall": 0.7654,
  "roc_auc": 0.9012,
  "tp": 1523,
  "tn": 2845,
  "fp": 356,
  "fn": 476,
  "num_samples": 5200,
  "num_positive": 1999,
  "num_negative": 3201,
  "has_overlap": false,
  "overlap_note": "✅ Keine Überschneidung mit Trainingsdaten",
  "feature_importance": {
    "price_open": 0.35,
    "price_high": 0.25,
    "price_low": 0.20,
    "volume_sol": 0.20
  }
}
```

### 11.2 Tabelle: `ml_comparisons`

**Schema:**
```sql
CREATE TABLE ml_comparisons (
    id BIGINT PRIMARY KEY,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Die beiden Modelle
    model_a_id BIGINT NOT NULL REFERENCES ml_models(id) ON DELETE CASCADE,
    model_b_id BIGINT NOT NULL REFERENCES ml_models(id) ON DELETE CASCADE,
    
    -- Zeitraum (gleich für beide)
    test_start TIMESTAMP WITH TIME ZONE NOT NULL,
    test_end TIMESTAMP WITH TIME ZONE NOT NULL,
    num_samples INT,
    
    -- Metriken Modell A
    a_accuracy NUMERIC(5, 4),
    a_f1 NUMERIC(5, 4),
    a_precision NUMERIC(5, 4),
    a_recall NUMERIC(5, 4),
    
    -- Metriken Modell B
    b_accuracy NUMERIC(5, 4),
    b_f1 NUMERIC(5, 4),
    b_precision NUMERIC(5, 4),
    b_recall NUMERIC(5, 4),
    
    -- Gewinner
    winner_id BIGINT REFERENCES ml_models(id),
    
    CONSTRAINT chk_different_models CHECK (model_a_id != model_b_id)
);
```

**Beispiel-Datensatz:**
```json
{
  "id": 1,
  "created_at": "2024-02-08T10:00:00+00:00",
  "model_a_id": 123,
  "model_b_id": 456,
  "test_start": "2024-02-01T00:00:00+00:00",
  "test_end": "2024-02-07T23:59:59+00:00",
  "num_samples": 5200,
  "a_accuracy": 0.8523,
  "a_f1": 0.7891,
  "a_precision": 0.8123,
  "a_recall": 0.7654,
  "b_accuracy": 0.8456,
  "b_f1": 0.7823,
  "b_precision": 0.8012,
  "b_recall": 0.7634,
  "winner_id": 123
}
```

### 11.3 Tabelle: `ml_jobs`

**Für TEST-Jobs:**
- `job_type`: `"TEST"`
- `test_model_id`: ID des zu testenden Modells
- `test_start`, `test_end`: Test-Zeitraum
- `result_test_id`: ID des erstellten Test-Ergebnisses (nach Abschluss)

**Für COMPARE-Jobs:**
- `job_type`: `"COMPARE"`
- `compare_model_a_id`, `compare_model_b_id`: IDs der beiden Modelle
- `compare_start`, `compare_end`: Test-Zeitraum
- `result_comparison_id`: ID des erstellten Vergleichs (nach Abschluss)

---

## 12. Validierungen & Bedingungen

### 12.1 UI-Validierungen

**Modell-Test:**
1. ✅ Modell muss ausgewählt sein
2. ✅ Modell-Status muss `READY` sein
3. ✅ `test_start` < `test_end`

**Modell-Vergleich:**
1. ✅ Mindestens 2 Modelle müssen verfügbar sein
2. ✅ Modell A und Modell B müssen unterschiedlich sein
3. ✅ Beide Modell-Status müssen `READY` sein
4. ✅ `test_start` < `test_end`

### 12.2 API-Validierungen (Pydantic)

**Modell-Test:**
1. ✅ `test_start`, `test_end` werden zu UTC konvertiert
2. ✅ Modell muss existieren (`model_id` in `ml_models`)
3. ✅ Modell darf nicht gelöscht sein (`is_deleted == False`)

**Modell-Vergleich:**
1. ✅ `test_start`, `test_end` werden zu UTC konvertiert
2. ✅ `model_a_id` und `model_b_id` müssen unterschiedlich sein
3. ✅ Beide Modelle müssen existieren
4. ✅ Beide Modelle dürfen nicht gelöscht sein

### 12.3 Backend-Validierungen

**Während Test:**
1. ✅ Modell-Datei muss vorhanden sein (`model_file_path`)
2. ✅ Test-Daten müssen vorhanden sein (`len(test_data) > 0`)
3. ✅ Labels müssen ausgewogen sein (mindestens 1 positive und 1 negative Label)
4. ✅ Zeitbasierte Parameter müssen vorhanden sein (wenn zeitbasierte Vorhersage)

**Während Vergleich:**
1. ✅ Beide Modell-Dateien müssen vorhanden sein
2. ✅ Beide Tests müssen erfolgreich sein
3. ✅ Beide Tests müssen auf denselben Daten ausgeführt werden

**Fehlerbehandlung:**
- `ValueError` wenn Validierung fehlschlägt
- Job-Status wird auf `FAILED` gesetzt
- Fehlermeldung wird in `error_msg` gespeichert

---

## 📝 Zusammenfassung

### ✅ Alle Möglichkeiten zum Modell-Test:

1. **UI:**
   - Seite "🧪 Modell testen"
   - Modell auswählen (nur `READY` Modelle)
   - Test-Zeitraum definieren (Datum + Uhrzeit in UTC)
   - Overlap-Warnung (wenn Test-Daten mit Trainingsdaten überlappen)

2. **API:**
   - `POST /api/models/{model_id}/test`
   - JSON Request Body mit `test_start`, `test_end`
   - Asynchrone Job-Erstellung

3. **Backend:**
   - Modell laden (aus DB und Datei)
   - Test-Daten laden (aus `coin_metrics`)
   - Labels erstellen (klassisch oder zeitbasiert)
   - Vorhersagen machen
   - Metriken berechnen (Accuracy, F1, Precision, Recall, ROC-AUC)
   - Confusion Matrix erstellen
   - Overlap-Check durchführen
   - Ergebnis in `ml_test_results` speichern

### ✅ Alle Möglichkeiten zum Modell-Vergleich:

1. **UI:**
   - Seite "⚔️ Modelle vergleichen"
   - 2 Modelle auswählen (nur `READY` Modelle, müssen unterschiedlich sein)
   - Test-Zeitraum definieren (gleicher Zeitraum für beide Modelle)

2. **API:**
   - `POST /api/models/compare`
   - JSON Request Body mit `model_a_id`, `model_b_id`, `test_start`, `test_end`
   - Asynchrone Job-Erstellung

3. **Backend:**
   - Beide Modelle testen (auf denselben Test-Daten)
   - Metriken für beide Modelle berechnen
   - Gewinner bestimmen (basierend auf F1-Score)
   - Vergleich in `ml_comparisons` speichern

### ✅ Berechnete Metriken:

- **Accuracy:** Genauigkeit
- **F1-Score:** Harmonisches Mittel aus Precision und Recall
- **Precision:** Präzision (True Positives / (True Positives + False Positives))
- **Recall:** Recall (True Positives / (True Positives + False Negatives))
- **ROC-AUC:** Fläche unter ROC-Kurve (optional)
- **Confusion Matrix:** TP, TN, FP, FN

### ✅ Overlap-Check:

- Prüft ob Test-Daten mit Trainingsdaten überlappen
- Berechnet Overlap-Prozent
- Warnung wird angezeigt, Test wird trotzdem ausgeführt
- Overlap-Info wird in `ml_test_results` gespeichert

### ✅ Datenbank-Speicherung:

1. **`ml_test_results`:** Test-Ergebnisse mit Metriken, Confusion Matrix, Overlap-Info
2. **`ml_comparisons`:** Vergleichs-Ergebnisse mit Metriken für beide Modelle und Gewinner
3. **`ml_jobs`:** Job-Status, Parameter, Ergebnisse

---

## 🔍 Prüf-Checkliste

Verwende diese Checkliste, um zu prüfen, ob alle benötigten Funktionen vorhanden sind:

- [ ] Modell-Test über UI: Vollständiges Formular mit Validierung ✅
- [ ] Modell-Test über API: REST Endpoint mit Validierung ✅
- [ ] Modell-Vergleich über UI: Vollständiges Formular mit Validierung ✅
- [ ] Modell-Vergleich über API: REST Endpoint mit Validierung ✅
- [ ] Backend: Asynchrone Job-Verarbeitung ✅
- [ ] Metriken: Accuracy, F1, Precision, Recall, ROC-AUC ✅
- [ ] Confusion Matrix: TP, TN, FP, FN ✅
- [ ] Overlap-Check: Prüfung und Warnung ✅
- [ ] Datenbank: `ml_test_results`, `ml_comparisons`, `ml_jobs` ✅
- [ ] Zeitbasierte Vorhersage: Unterstützung beim Testen ✅
- [ ] Gewinner-Bestimmung: Basierend auf F1-Score ✅

---

**Erstellt:** 2024-01-XX  
**Version:** 1.0  
**Status:** ✅ Vollständig dokumentiert

