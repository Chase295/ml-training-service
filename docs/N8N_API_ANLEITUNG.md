# 📘 n8n API-Anleitung: ML Training Service Fernsteuerung

**Version:** 1.0  
**Erstellt:** 2025-12-23  
**Status:** ✅ Vollständig

---

## 📋 Inhaltsverzeichnis

1. [Übersicht](#übersicht)
2. [API-Basis-URL](#api-basis-url)
3. [Authentifizierung](#authentifizierung)
4. [Modell-Management](#modell-management)
5. [Modell-Training](#modell-training)
6. [Modell-Testing](#modell-testing)
7. [Modell-Vergleich](#modell-vergleich)
8. [Modell-Download](#modell-download)
9. [Job-Management](#job-management)
10. [Test-Ergebnisse](#test-ergebnisse)
11. [Vergleichs-Ergebnisse](#vergleichs-ergebnisse)
12. [Daten-Verfügbarkeit](#daten-verfügbarkeit)
13. [n8n Workflow-Beispiele](#n8n-workflow-beispiele)
14. [Fehlerbehandlung](#fehlerbehandlung)

---

## 🎯 Übersicht

Diese Anleitung zeigt, wie du das **ML Training Service** vollständig von **n8n** aus fernsteuern kannst. Alle Funktionen sind über REST-API-Endpunkte verfügbar.

### Verfügbare Funktionen:

- ✅ **Modelle erstellen** (Training starten)
- ✅ **Modelle auflisten** (mit Filtern)
- ✅ **Modell-Details abrufen**
- ✅ **Modelle testen** (auf Test-Daten)
- ✅ **Modelle vergleichen** (2 Modelle gegeneinander)
- ✅ **Modelle herunterladen** (.pkl Dateien)
- ✅ **Modelle aktualisieren** (Name, Beschreibung)
- ✅ **Modelle löschen** (Soft-Delete)
- ✅ **Jobs überwachen** (Status, Fortschritt)
- ✅ **Test-Ergebnisse abrufen**
- ✅ **Vergleichs-Ergebnisse abrufen**
- ✅ **Daten-Verfügbarkeit prüfen**

---

## 🌐 API-Basis-URL

```
http://localhost:8000/api
```

**Produktion:** Ersetze `localhost:8000` mit deiner Server-Adresse.

---

## 🔐 Authentifizierung

**Aktuell:** Keine Authentifizierung erforderlich.

⚠️ **Hinweis:** Für Produktion sollte Authentifizierung implementiert werden.

---

## 📦 Modell-Management

### 1. Modelle auflisten

**Endpoint:** `GET /api/models`

**Query-Parameter:**
- `status` (optional): Filter nach Status (`READY`, `TRAINING`, `FAILED`)
- `is_deleted` (optional): `true`/`false` (Standard: `false`)

**n8n HTTP Request Node:**
```
Method: GET
URL: http://localhost:8000/api/models?status=READY&is_deleted=false
```

**Response (200 OK):**
```json
[
  {
    "id": 1,
    "name": "Pump_RF_5min_30pct",
    "model_type": "random_forest",
    "status": "READY",
    "created_at": "2025-12-23T10:00:00Z",
    "updated_at": "2025-12-23T10:05:00Z",
    "is_deleted": false,
    "target_variable": "price_close",
    "target_operator": null,
    "target_value": null,
    "train_start": "2025-12-21T19:42:29Z",
    "train_end": "2025-12-23T09:58:35Z",
    "features": ["price_open", "price_high", "price_low", "volume_sol"],
    "phases": null,
    "params": {
      "cv_splits": 5,
      "_time_based": {
        "enabled": true,
        "direction": "up",
        "future_minutes": 5,
        "min_percent_change": 30
      },
      "use_engineered_features": true,
      "feature_engineering_windows": [5, 10, 15]
    },
    "training_accuracy": 0.8542,
    "training_f1": 0.7234,
    "roc_auc": 0.8912,
    "mcc": 0.6543
  }
]
```

---

### 2. Modell-Details abrufen

**Endpoint:** `GET /api/models/{model_id}`

**n8n HTTP Request Node:**
```
Method: GET
URL: http://localhost:8000/api/models/1
```

**Response (200 OK):**
```json
{
  "id": 1,
  "name": "Pump_RF_5min_30pct",
  "model_type": "random_forest",
  "status": "READY",
  "created_at": "2025-12-23T10:00:00Z",
  "updated_at": "2025-12-23T10:05:00Z",
  "is_deleted": false,
  "target_variable": "price_close",
  "target_operator": null,
  "target_value": null,
  "train_start": "2025-12-21T19:42:29Z",
  "train_end": "2025-12-23T09:58:35Z",
  "features": ["price_open", "price_high", "price_low", "volume_sol"],
  "phases": null,
  "params": {
    "cv_splits": 5,
    "_time_based": {
      "enabled": true,
      "direction": "up",
      "future_minutes": 5,
      "min_percent_change": 30
    },
    "use_engineered_features": true,
    "feature_engineering_windows": [5, 10, 15]
  },
  "training_accuracy": 0.8542,
  "training_f1": 0.7234,
  "roc_auc": 0.8912,
  "mcc": 0.6543,
  "confusion_matrix": {
    "tp": 1234,
    "tn": 5678,
    "fp": 890,
    "fn": 198
  }
}
```

---

### 3. Modell aktualisieren

**Endpoint:** `PATCH /api/models/{model_id}`

**n8n HTTP Request Node:**
```
Method: PATCH
URL: http://localhost:8000/api/models/1
Content-Type: application/json
```

**Request Body:**
```json
{
  "name": "Neuer Modell-Name",
  "description": "Neue Beschreibung"
}
```

**Response (200 OK):**
```json
{
  "id": 1,
  "name": "Neuer Modell-Name",
  "description": "Neue Beschreibung",
  ...
}
```

---

### 4. Modell löschen

**Endpoint:** `DELETE /api/models/{model_id}`

**n8n HTTP Request Node:**
```
Method: DELETE
URL: http://localhost:8000/api/models/1
```

**Response (204 No Content):** Kein Body

⚠️ **Hinweis:** Soft-Delete - Modell wird nur als gelöscht markiert, Datei bleibt erhalten.

---

## 🎓 Modell-Training

### Modell erstellen (Training starten)

**Endpoint:** `POST /api/models/create`

**n8n HTTP Request Node:**
```
Method: POST
URL: http://localhost:8000/api/models/create
Content-Type: application/json
```

**Request Body (Minimal - Normale Vorhersage):**
```json
{
  "name": "Mein_Modell_RF",
  "model_type": "random_forest",
  "features": ["price_open", "price_high", "price_low", "volume_sol"],
  "phases": null,
  "train_start": "2025-12-21T19:42:29Z",
  "train_end": "2025-12-23T09:58:35Z",
  "target_var": "price_close",
  "operator": ">",
  "target_value": 0.0001,
  "description": "Mein erstes Modell"
}
```

**Request Body (Zeitbasierte Vorhersage):**
```json
{
  "name": "Pump_Detector_5min_30pct",
  "model_type": "xgboost",
  "features": ["price_open", "price_high", "price_low", "volume_sol", "buy_volume_sol", "sell_volume_sol"],
  "phases": null,
  "train_start": "2025-12-21T19:42:29Z",
  "train_end": "2025-12-23T09:58:35Z",
  "use_time_based_prediction": true,
  "future_minutes": 5,
  "min_percent_change": 30.0,
  "direction": "up",
  "use_engineered_features": true,
  "feature_engineering_windows": [5, 10, 15],
  "use_smote": true,
  "use_timeseries_split": true,
  "cv_splits": 5,
  "description": "Pump-Detector für 5min, 30% Steigerung"
}
```

**Request Body (Mit Hyperparametern):**
```json
{
  "name": "Optimiertes_Modell",
  "model_type": "random_forest",
  "features": ["price_open", "price_high", "price_low", "volume_sol"],
  "phases": [1, 2, 3],
  "train_start": "2025-12-21T19:42:29Z",
  "train_end": "2025-12-23T09:58:35Z",
  "target_var": "price_close",
  "operator": ">",
  "target_value": 0.0001,
  "params": {
    "n_estimators": 200,
    "max_depth": 15,
    "min_samples_split": 5
  },
  "use_engineered_features": true,
  "use_smote": true,
  "use_timeseries_split": true,
  "cv_splits": 5
}
```

**Response (201 Created):**
```json
{
  "job_id": 42,
  "status": "PENDING",
  "message": "Training-Job erstellt"
}
```

⚠️ **WICHTIG:** Das Modell wird **nicht sofort** erstellt! Der Job wird in die Queue eingereiht und asynchron verarbeitet. Verwende `/api/queue/{job_id}` um den Status zu überwachen.

---

### Verfügbare Parameter:

#### Basis-Parameter:
- `name` (string, **erforderlich**): Eindeutiger Modell-Name
- `model_type` (string, **erforderlich**): `"random_forest"` oder `"xgboost"`
- `features` (array, **erforderlich**): Liste der Feature-Namen
- `phases` (array, optional): Liste der Coin-Phasen (z.B. `[1, 2, 3]`)
- `train_start` (datetime, **erforderlich**): Start-Zeitpunkt (ISO-Format mit UTC)
- `train_end` (datetime, **erforderlich**): Ende-Zeitpunkt (ISO-Format mit UTC)
- `description` (string, optional): Beschreibung

#### Ziel-Variable (für normale Vorhersage):
- `target_var` (string, **erforderlich** wenn `use_time_based_prediction=false`): z.B. `"price_close"`
- `operator` (string, **erforderlich** wenn `use_time_based_prediction=false`): `">"`, `"<"`, `">="`, `"<="`, `"="`
- `target_value` (float, **erforderlich** wenn `use_time_based_prediction=false`): Schwellwert

#### Zeitbasierte Vorhersage:
- `use_time_based_prediction` (boolean, default: `false`): Aktivieren
- `future_minutes` (integer, **erforderlich** wenn `use_time_based_prediction=true`): Minuten in die Zukunft (z.B. `5`)
- `min_percent_change` (float, **erforderlich** wenn `use_time_based_prediction=true`): Mindest-Prozent-Änderung (z.B. `30.0` für 30%)
- `direction` (string, default: `"up"`): `"up"` (steigt) oder `"down"` (fällt)

#### Feature-Engineering:
- `use_engineered_features` (boolean, default: `false`): Erweiterte Pump-Detection Features
- `feature_engineering_windows` (array, optional): Fenstergrößen (z.B. `[5, 10, 15]`)

#### SMOTE (Imbalanced Data Handling):
- `use_smote` (boolean, default: `true`): SMOTE aktivieren

#### Cross-Validation:
- `use_timeseries_split` (boolean, default: `true`): TimeSeriesSplit verwenden
- `cv_splits` (integer, default: `5`): Anzahl Splits

#### Hyperparameter:
- `params` (object, optional): Z.B. `{"n_estimators": 200, "max_depth": 15}`

---

## 🧪 Modell-Testing

### Modell testen

**Endpoint:** `POST /api/models/{model_id}/test`

**n8n HTTP Request Node:**
```
Method: POST
URL: http://localhost:8000/api/models/1/test
Content-Type: application/json
```

**Request Body:**
```json
{
  "test_start": "2025-12-23T10:00:00Z",
  "test_end": "2025-12-23T22:00:00Z"
}
```

**Response (201 Created):**
```json
{
  "job_id": 43,
  "status": "PENDING",
  "message": "Test-Job erstellt"
}
```

⚠️ **WICHTIG:** Der Test wird asynchron ausgeführt. Verwende `/api/queue/{job_id}` um den Status zu überwachen.

---

## ⚔️ Modell-Vergleich

### Zwei Modelle vergleichen

**Endpoint:** `POST /api/models/compare`

**n8n HTTP Request Node:**
```
Method: POST
URL: http://localhost:8000/api/models/compare
Content-Type: application/json
```

**Request Body:**
```json
{
  "model_a_id": 1,
  "model_b_id": 2,
  "test_start": "2025-12-23T10:00:00Z",
  "test_end": "2025-12-23T22:00:00Z"
}
```

**Response (201 Created):**
```json
{
  "job_id": 44,
  "status": "PENDING",
  "message": "Vergleichs-Job erstellt"
}
```

⚠️ **WICHTIG:** Der Vergleich wird asynchron ausgeführt. Verwende `/api/queue/{job_id}` um den Status zu überwachen.

---

## 📥 Modell-Download

### Modell herunterladen

**Endpoint:** `GET /api/models/{model_id}/download`

**n8n HTTP Request Node:**
```
Method: GET
URL: http://localhost:8000/api/models/1/download
Response Format: File
```

**Response (200 OK):**
- Content-Type: `application/octet-stream`
- Body: `.pkl` Datei (Binary)

**n8n Konfiguration:**
- **Response Format:** `File`
- **Binary Property:** `data` (oder wie du es nennen möchtest)

---

## 📊 Job-Management

### Jobs auflisten

**Endpoint:** `GET /api/queue`

**Query-Parameter:**
- `status` (optional): Filter nach Status (`PENDING`, `RUNNING`, `COMPLETED`, `FAILED`)
- `job_type` (optional): Filter nach Typ (`TRAIN`, `TEST`, `COMPARE`)

**n8n HTTP Request Node:**
```
Method: GET
URL: http://localhost:8000/api/queue?status=COMPLETED&job_type=TRAIN
```

**Response (200 OK):**
```json
[
  {
    "id": 42,
    "job_type": "TRAIN",
    "status": "COMPLETED",
    "priority": 5,
    "created_at": "2025-12-23T10:00:00Z",
    "started_at": "2025-12-23T10:00:05Z",
    "completed_at": "2025-12-23T10:05:30Z",
    "progress": 100,
    "progress_msg": "Training abgeschlossen",
    "error_msg": null,
    "result_model_id": 1,
    "result_test_id": null,
    "result_comparison_id": null
  }
]
```

---

### Job-Details abrufen

**Endpoint:** `GET /api/queue/{job_id}`

**n8n HTTP Request Node:**
```
Method: GET
URL: http://localhost:8000/api/queue/42
```

**Response (200 OK):**
```json
{
  "id": 42,
  "job_type": "TRAIN",
  "status": "COMPLETED",
  "priority": 5,
  "created_at": "2025-12-23T10:00:00Z",
  "started_at": "2025-12-23T10:00:05Z",
  "completed_at": "2025-12-23T10:05:30Z",
  "progress": 100,
  "progress_msg": "Training abgeschlossen",
  "error_msg": null,
  "result_model_id": 1,
  "result_model": {
    "id": 1,
    "name": "Mein_Modell_RF",
    "model_type": "random_forest",
    "status": "READY",
    "training_accuracy": 0.8542,
    "training_f1": 0.7234,
    ...
  }
}
```

---

## 📈 Test-Ergebnisse

### Test-Ergebnisse auflisten

**Endpoint:** `GET /api/test-results`

**Query-Parameter:**
- `limit` (optional, default: 100): Anzahl Ergebnisse
- `offset` (optional, default: 0): Offset für Pagination

**n8n HTTP Request Node:**
```
Method: GET
URL: http://localhost:8000/api/test-results?limit=50&offset=0
```

**Response (200 OK):**
```json
[
  {
    "id": 1,
    "model_id": 1,
    "created_at": "2025-12-23T11:00:00Z",
    "test_start": "2025-12-23T10:00:00Z",
    "test_end": "2025-12-23T22:00:00Z",
    "accuracy": 0.8234,
    "f1_score": 0.7123,
    "precision_score": 0.7890,
    "recall": 0.6543,
    "roc_auc": 0.8765,
    "mcc": 0.6234,
    "fpr": 0.1234,
    "fnr": 0.2345,
    "simulated_profit_pct": 5.67,
    "confusion_matrix": {
      "tp": 1234,
      "tn": 5678,
      "fp": 890,
      "fn": 198
    },
    "num_samples": 8000,
    "num_positive": 1432,
    "num_negative": 6568
  }
]
```

---

### Test-Ergebnis-Details abrufen

**Endpoint:** `GET /api/test-results/{test_id}`

**n8n HTTP Request Node:**
```
Method: GET
URL: http://localhost:8000/api/test-results/1
```

**Response (200 OK):** Siehe oben (einzelnes Objekt statt Array)

---

### Test-Ergebnis löschen

**Endpoint:** `DELETE /api/test-results/{test_id}`

**n8n HTTP Request Node:**
```
Method: DELETE
URL: http://localhost:8000/api/test-results/1
```

**Response (200 OK):**
```json
{
  "message": "Test-Ergebnis 1 erfolgreich gelöscht"
}
```

---

## ⚖️ Vergleichs-Ergebnisse

### Vergleichs-Ergebnisse auflisten

**Endpoint:** `GET /api/comparisons`

**Query-Parameter:**
- `limit` (optional, default: 100): Anzahl Ergebnisse
- `offset` (optional, default: 0): Offset für Pagination

**n8n HTTP Request Node:**
```
Method: GET
URL: http://localhost:8000/api/comparisons?limit=50&offset=0
```

**Response (200 OK):**
```json
[
  {
    "id": 1,
    "model_a_id": 1,
    "model_b_id": 2,
    "created_at": "2025-12-23T12:00:00Z",
    "test_start": "2025-12-23T10:00:00Z",
    "test_end": "2025-12-23T22:00:00Z",
    "winner_id": 1,
    "a_accuracy": 0.8234,
    "a_f1": 0.7123,
    "a_roc_auc": 0.8765,
    "a_mcc": 0.6234,
    "b_accuracy": 0.7890,
    "b_f1": 0.6789,
    "b_roc_auc": 0.8456,
    "b_mcc": 0.5678
  }
]
```

---

### Vergleichs-Ergebnis-Details abrufen

**Endpoint:** `GET /api/comparisons/{comparison_id}`

**n8n HTTP Request Node:**
```
Method: GET
URL: http://localhost:8000/api/comparisons/1
```

**Response (200 OK):** Siehe oben (einzelnes Objekt statt Array)

---

### Vergleichs-Ergebnis löschen

**Endpoint:** `DELETE /api/comparisons/{comparison_id}`

**n8n HTTP Request Node:**
```
Method: DELETE
URL: http://localhost:8000/api/comparisons/1
```

**Response (200 OK):**
```json
{
  "message": "Vergleich 1 erfolgreich gelöscht"
}
```

---

## 📅 Daten-Verfügbarkeit

### Verfügbare Daten abrufen

**Endpoint:** `GET /api/data-availability`

**n8n HTTP Request Node:**
```
Method: GET
URL: http://localhost:8000/api/data-availability
```

**Response (200 OK):**
```json
{
  "min_timestamp": "2025-12-21T19:42:29Z",
  "max_timestamp": "2025-12-23T22:48:15Z"
}
```

**Verwendung:** Prüfe vor dem Training/Testing, welche Daten verfügbar sind.

---

## 🔄 n8n Workflow-Beispiele

### Workflow 1: Modell erstellen und Status überwachen

```
1. HTTP Request: GET /api/data-availability
   → Speichere min_timestamp und max_timestamp

2. HTTP Request: POST /api/models/create
   → Body: TrainModelRequest mit train_start/train_end basierend auf verfügbaren Daten
   → Speichere job_id

3. Wait Node (5 Sekunden)

4. HTTP Request: GET /api/queue/{job_id}
   → Prüfe status

5. IF Node: status == "COMPLETED"?
   → Ja: Weiter zu Schritt 6
   → Nein: Zurück zu Schritt 3

6. HTTP Request: GET /api/models/{result_model_id}
   → Modell-Details abrufen
```

---

### Workflow 2: Bestes Modell finden und testen

```
1. HTTP Request: GET /api/models?status=READY&is_deleted=false
   → Alle fertigen Modelle abrufen

2. Code Node: Sortiere nach training_accuracy (höchste zuerst)
   → Wähle bestes Modell (model_id)

3. HTTP Request: GET /api/data-availability
   → Verfügbare Daten prüfen

4. HTTP Request: POST /api/models/{model_id}/test
   → Test starten mit test_start/test_end basierend auf verfügbaren Daten
   → Speichere job_id

5. Wait Node (10 Sekunden)

6. HTTP Request: GET /api/queue/{job_id}
   → Prüfe status

7. IF Node: status == "COMPLETED"?
   → Ja: Weiter zu Schritt 8
   → Nein: Zurück zu Schritt 5

8. HTTP Request: GET /api/test-results/{result_test_id}
   → Test-Ergebnisse abrufen
```

---

### Workflow 3: Zwei Modelle vergleichen

```
1. HTTP Request: GET /api/models?status=READY&is_deleted=false
   → Alle fertigen Modelle abrufen

2. Code Node: Wähle 2 Modelle (z.B. neueste 2)

3. HTTP Request: GET /api/data-availability
   → Verfügbare Daten prüfen

4. HTTP Request: POST /api/models/compare
   → Vergleich starten
   → Body: {
       "model_a_id": model_1_id,
       "model_b_id": model_2_id,
       "test_start": "...",
       "test_end": "..."
     }
   → Speichere job_id

5. Wait Node (15 Sekunden)

6. HTTP Request: GET /api/queue/{job_id}
   → Prüfe status

7. IF Node: status == "COMPLETED"?
   → Ja: Weiter zu Schritt 8
   → Nein: Zurück zu Schritt 5

8. HTTP Request: GET /api/comparisons/{result_comparison_id}
   → Vergleichs-Ergebnisse abrufen

9. Code Node: Bestimme Gewinner (winner_id)
   → Wenn winner_id == model_a_id: Modell A gewinnt
   → Wenn winner_id == model_b_id: Modell B gewinnt
```

---

### Workflow 4: Modell herunterladen und speichern

```
1. HTTP Request: GET /api/models?status=READY&is_deleted=false
   → Alle fertigen Modelle abrufen

2. Code Node: Wähle Modell (z.B. nach Name filtern)

3. HTTP Request: GET /api/models/{model_id}/download
   → Response Format: File
   → Binary Property: data

4. Write Binary File Node
   → Speichere .pkl Datei lokal oder in Cloud Storage
```

---

## ⚠️ Fehlerbehandlung

### HTTP Status Codes:

- **200 OK:** Erfolgreich
- **201 Created:** Ressource erstellt (Job erstellt)
- **204 No Content:** Erfolgreich gelöscht
- **400 Bad Request:** Ungültige Request-Parameter
- **404 Not Found:** Ressource nicht gefunden
- **500 Internal Server Error:** Server-Fehler

### Fehler-Response Format:

```json
{
  "detail": "Fehlerbeschreibung"
}
```

### n8n Fehlerbehandlung:

**In n8n HTTP Request Node:**
- ✅ **Continue On Fail:** Aktivieren
- ✅ **Response Format:** `JSON`
- ✅ **Error Handling:** Prüfe `statusCode` in Response

**Beispiel-Code (n8n Code Node):**
```javascript
const statusCode = $input.item.json.statusCode;
if (statusCode >= 400) {
  throw new Error(`API-Fehler: ${statusCode} - ${$input.item.json.body.detail}`);
}
return $input.item;
```

---

## 📝 Wichtige Hinweise

### 1. Asynchrone Jobs

⚠️ **WICHTIG:** Training, Testing und Vergleich werden **asynchron** ausgeführt!

- Nach `POST /api/models/create` → Job wird erstellt, Modell ist noch nicht fertig
- Verwende `GET /api/queue/{job_id}` um Status zu überwachen
- Warte bis `status == "COMPLETED"` bevor du das Modell verwendest

### 2. Datum/Zeit-Format

**Verwende immer ISO-Format mit UTC:**
```
2025-12-23T10:00:00Z
```

**n8n DateTime Node:**
- Format: `YYYY-MM-DDTHH:mm:ssZ`
- Timezone: UTC

### 3. Eindeutige Modell-Namen

⚠️ Modell-Namen müssen **eindeutig** sein!

**Lösung:** Verwende Timestamp im Namen:
```javascript
const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
const modelName = `Modell_${timestamp}`;
```

### 4. Daten-Verfügbarkeit prüfen

⚠️ Prüfe **immer** zuerst die verfügbaren Daten:
```
GET /api/data-availability
```

Verwende `min_timestamp` und `max_timestamp` für `train_start`/`train_end` und `test_start`/`test_end`.

### 5. Job-Status überwachen

**Mögliche Status-Werte:**
- `PENDING`: Job wartet in Queue
- `RUNNING`: Job wird gerade ausgeführt
- `COMPLETED`: Job erfolgreich abgeschlossen
- `FAILED`: Job fehlgeschlagen (siehe `error_msg`)

**Polling-Intervall:** 5-10 Sekunden

---

## 🎯 Zusammenfassung

### Schnellstart-Checkliste:

1. ✅ **Daten-Verfügbarkeit prüfen:** `GET /api/data-availability`
2. ✅ **Modell erstellen:** `POST /api/models/create`
3. ✅ **Job-Status überwachen:** `GET /api/queue/{job_id}` (wiederholt)
4. ✅ **Modell-Details abrufen:** `GET /api/models/{model_id}`
5. ✅ **Modell testen:** `POST /api/models/{model_id}/test`
6. ✅ **Test-Ergebnisse abrufen:** `GET /api/test-results/{test_id}`
7. ✅ **Modelle vergleichen:** `POST /api/models/compare`
8. ✅ **Modell herunterladen:** `GET /api/models/{model_id}/download`

---

**Erstellt:** 2025-12-23  
**Version:** 1.0  
**Status:** ✅ Vollständig

