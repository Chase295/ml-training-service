# 🧪 End-to-End Test Anleitung

## Übersicht

Diese Anleitung beschreibt, wie du das gesamte ML Training Service Projekt von unten nach oben testest.

## Voraussetzungen

1. ✅ Docker Desktop läuft
2. ✅ Container `ml-training-service` läuft
3. ✅ Datenbank-Verbindung funktioniert
4. ✅ `ref_coin_phases` Tabelle enthält Daten

## Test-Skript ausführen

### Automatischer E2E-Test

```bash
cd ml-training-service
python3 test_e2e.py
```

Das Skript testet automatisch:
1. ✅ Health Check
2. ✅ Phasen laden (ref_coin_phases)
3. ✅ Modelle auflisten
4. ✅ Normales Modell trainieren
5. ✅ Zeitbasiertes Modell trainieren
6. ✅ Job-Status prüfen
7. ✅ Modell testen
8. ✅ Modelle vergleichen
9. ✅ Jobs auflisten
10. ✅ Prometheus Metrics

## Manuelle Tests

### 1. Health Check

```bash
curl http://localhost:8000/api/health | jq
```

**Erwartetes Ergebnis:**
```json
{
  "status": "healthy",
  "db_connected": true,
  "uptime_seconds": 123,
  ...
}
```

### 2. Phasen laden

```bash
curl http://localhost:8000/api/phases | jq
```

**Erwartetes Ergebnis:**
```json
[
  {
    "id": 1,
    "name": "Baby Zone",
    "interval_seconds": 5,
    "max_age_minutes": 20
  },
  ...
]
```

### 3. Normales Modell trainieren (via API)

```bash
curl -X POST http://localhost:8000/api/models/create \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Test_Model_Normal",
    "model_type": "random_forest",
    "target_var": "price_close",
    "operator": ">",
    "target_value": 100.0,
    "features": ["price_open", "price_high", "volume_sol"],
    "phases": [1, 2],
    "train_start": "2024-01-01T00:00:00Z",
    "train_end": "2024-12-22T23:59:59Z",
    "use_time_based_prediction": false
  }' | jq
```

**Erwartetes Ergebnis:**
```json
{
  "job_id": 123,
  "status": "PENDING",
  "message": "Job erstellt..."
}
```

### 4. Zeitbasiertes Modell trainieren (via API)

```bash
curl -X POST http://localhost:8000/api/models/create \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Test_Model_TimeBased",
    "model_type": "random_forest",
    "target_var": "price_close",
    "operator": null,
    "target_value": null,
    "features": ["price_open", "price_high", "volume_sol"],
    "phases": [1, 2],
    "train_start": "2024-01-01T00:00:00Z",
    "train_end": "2024-12-22T23:59:59Z",
    "use_time_based_prediction": true,
    "future_minutes": 10,
    "min_percent_change": 5.0,
    "direction": "up"
  }' | jq
```

### 5. Job-Status prüfen

```bash
curl http://localhost:8000/api/jobs/123 | jq
```

**Mögliche Status:**
- `PENDING` - Job wartet
- `RUNNING` - Job läuft
- `COMPLETED` - Job erfolgreich
- `FAILED` - Job fehlgeschlagen

### 6. Modell testen

```bash
curl -X POST http://localhost:8000/api/models/1/test \
  -H "Content-Type: application/json" \
  -d '{
    "test_start": "2024-12-15T00:00:00Z",
    "test_end": "2024-12-22T23:59:59Z"
  }' | jq
```

### 7. Modelle vergleichen

```bash
curl -X POST http://localhost:8000/api/models/compare \
  -H "Content-Type: application/json" \
  -d '{
    "model_a_id": 1,
    "model_b_id": 2,
    "test_start": "2024-12-15T00:00:00Z",
    "test_end": "2024-12-22T23:59:59Z"
  }' | jq
```

## UI-Tests (Streamlit)

### 1. Öffne Streamlit UI

```
http://localhost:8501
```

### 2. Teste "Neues Modell trainieren"

1. ✅ Navigiere zu "➕ Neues Modell trainieren"
2. ✅ Prüfe: "Zeitbasierte Vorhersage aktivieren" Checkbox reagiert sofort
3. ✅ Prüfe: "Hyperparameter anpassen" Checkbox reagiert sofort
4. ✅ Prüfe: Phasen werden mit `interval_seconds` angezeigt
5. ✅ Prüfe: Bei zeitbasierter Vorhersage werden Ziel-Variablen ausgeblendet
6. ✅ Prüfe: Datum + Uhrzeit können eingegeben werden

**Normales Modell:**
- Modell-Name: `Test_UI_Normal`
- Modell-Typ: `random_forest`
- Features: `price_open`, `price_high`, `volume_sol`
- Phasen: Wähle 1-2 Phasen
- Ziel-Variable: `price_close`
- Operator: `>`
- Target-Wert: `100.0`
- Training-Zeitraum: Letzte 30 Tage
- Submit → Job sollte erstellt werden

**Zeitbasiertes Modell:**
- Aktiviere "Zeitbasierte Vorhersage aktivieren"
- Prüfe: Ziel-Variablen sind ausgeblendet
- Prüfe: Zeitbasierte Konfiguration erscheint
- Modell-Name: `Test_UI_TimeBased`
- Variable überwachen: `price_close`
- Zukunft: `10` Minuten
- Min. Prozent-Änderung: `5.0`
- Richtung: `Steigt`
- Submit → Job sollte erstellt werden

### 3. Teste "Modell testen"

1. ✅ Wähle ein fertiges Modell (Status: READY)
2. ✅ Setze Test-Zeitraum (Datum + Uhrzeit)
3. ✅ Submit → Test-Job sollte erstellt werden

### 4. Teste "Modelle vergleichen"

1. ✅ Wähle 2 fertige Modelle
2. ✅ Setze Test-Zeitraum
3. ✅ Submit → Vergleichs-Job sollte erstellt werden

### 5. Teste "Jobs"

1. ✅ Prüfe: Alle Jobs werden angezeigt
2. ✅ Prüfe: Status wird korrekt angezeigt
3. ✅ Prüfe: Progress wird angezeigt

## Datenbank-Checks

### 1. Prüfe Modelle in DB

```sql
SELECT id, name, status, model_type, 
       use_time_based_prediction, future_minutes, min_percent_change
FROM ml_models 
WHERE name LIKE 'TEST_%' 
ORDER BY created_at DESC;
```

### 2. Prüfe Jobs in DB

```sql
SELECT id, job_type, status, progress, 
       train_model_type, train_phases
FROM ml_jobs 
ORDER BY created_at DESC 
LIMIT 10;
```

### 3. Prüfe Test-Ergebnisse

```sql
SELECT id, model_id, accuracy, f1_score, num_samples
FROM ml_test_results 
ORDER BY created_at DESC 
LIMIT 5;
```

## Checkliste

### API-Endpunkte
- [ ] `GET /api/health` - Health Check
- [ ] `GET /api/phases` - Phasen laden
- [ ] `GET /api/models` - Modelle auflisten
- [ ] `GET /api/models/{id}` - Modell-Details
- [ ] `POST /api/models/create` - Modell trainieren (normal)
- [ ] `POST /api/models/create` - Modell trainieren (zeitbasiert)
- [ ] `POST /api/models/{id}/test` - Modell testen
- [ ] `POST /api/models/compare` - Modelle vergleichen
- [ ] `GET /api/jobs` - Jobs auflisten
- [ ] `GET /api/jobs/{id}` - Job-Details
- [ ] `GET /api/metrics` - Prometheus Metrics

### UI-Funktionalität
- [ ] Phasen werden mit `interval_seconds` angezeigt
- [ ] Zeitbasierte Vorhersage Checkbox reagiert sofort
- [ ] Hyperparameter Checkbox reagiert sofort
- [ ] Ziel-Variablen werden bei zeitbasierter Vorhersage ausgeblendet
- [ ] Datum + Uhrzeit können eingegeben werden
- [ ] Normales Modell kann trainiert werden
- [ ] Zeitbasiertes Modell kann trainiert werden
- [ ] Modell kann getestet werden
- [ ] Modelle können verglichen werden

### Logik
- [ ] `interval_seconds` werden beim Training berücksichtigt
- [ ] Zeitbasierte Labels werden korrekt erstellt
- [ ] Phase-Intervalle werden korrekt verwendet
- [ ] Jobs werden korrekt verarbeitet
- [ ] Modelle werden korrekt gespeichert

## Troubleshooting

### Problem: Health Check schlägt fehl
- Prüfe: Container läuft? `docker-compose ps`
- Prüfe: Datenbank-Verbindung? `docker-compose logs ml-training | grep -i "db\|error"`

### Problem: Phasen werden nicht geladen
- Prüfe: API-Endpoint funktioniert? `curl http://localhost:8000/api/phases`
- Prüfe: Datenbank enthält `ref_coin_phases`? SQL-Query ausführen

### Problem: Training schlägt fehl
- Prüfe: Job-Logs? `docker-compose logs ml-training | grep -i "error\|training"`
- Prüfe: Daten vorhanden? Prüfe `coin_metrics` Tabelle
- Prüfe: Zeitraum korrekt? Prüfe `train_start` und `train_end`

### Problem: UI reagiert nicht
- Prüfe: Streamlit läuft? `docker-compose logs ml-training | grep streamlit`
- Prüfe: Browser-Cache leeren
- Prüfe: Container neu starten? `docker-compose restart ml-training`

## Erfolgskriterien

✅ Alle API-Endpunkte funktionieren  
✅ Phasen werden korrekt geladen und angezeigt  
✅ Normales Modell kann trainiert werden  
✅ Zeitbasiertes Modell kann trainiert werden  
✅ `interval_seconds` werden beim Training berücksichtigt  
✅ UI reagiert sofort auf Checkbox-Änderungen  
✅ Jobs werden korrekt verarbeitet  
✅ Modelle können getestet werden  
✅ Modelle können verglichen werden  

