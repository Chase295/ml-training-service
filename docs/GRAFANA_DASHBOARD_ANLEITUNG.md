# 📊 Grafana Dashboard - ML Training Service

**Prometheus-Metriken für Job-Monitoring**

---

## 🎯 Verfügbare Metriken

### Job-Status Metriken

#### `ml_job_progress_percent`
**Beschreibung:** Fortschritt eines Jobs in Prozent (0-100)

**Labels:**
- `job_id`: Job-ID
- `job_type`: TRAIN, TEST, COMPARE
- `model_type`: random_forest, xgboost, etc.

**Beispiel Query:**
```promql
ml_job_progress_percent{job_type="TRAIN"}
```

---

#### `ml_job_duration_seconds`
**Beschreibung:** Laufzeit eines laufenden Jobs in Sekunden

**Labels:**
- `job_id`: Job-ID
- `job_type`: TRAIN, TEST, COMPARE
- `model_type`: random_forest, xgboost, etc.

**Beispiel Query:**
```promql
ml_job_duration_seconds{job_type="TRAIN", status="RUNNING"}
```

---

#### `ml_job_status`
**Beschreibung:** Job-Status (1=PENDING, 2=RUNNING, 3=COMPLETED, 4=FAILED, 5=CANCELLED)

**Labels:**
- `job_id`: Job-ID
- `job_type`: TRAIN, TEST, COMPARE
- `model_type`: random_forest, xgboost, etc.
- `status`: PENDING, RUNNING, COMPLETED, FAILED, CANCELLED

**Beispiel Query:**
```promql
ml_job_status{job_type="TRAIN", status="RUNNING"}
```

---

#### `ml_job_features_count`
**Beschreibung:** Anzahl Features in einem Job

**Labels:**
- `job_id`: Job-ID
- `job_type`: TRAIN, TEST, COMPARE

**Beispiel Query:**
```promql
ml_job_features_count{job_type="TRAIN"}
```

---

#### `ml_job_phases_count`
**Beschreibung:** Anzahl Phasen in einem Job

**Labels:**
- `job_id`: Job-ID
- `job_type`: TRAIN, TEST, COMPARE

**Beispiel Query:**
```promql
ml_job_phases_count{job_type="TRAIN"}
```

---

### Weitere Metriken

- `ml_active_jobs`: Anzahl aktiver Jobs
- `ml_jobs_total`: Gesamtanzahl Jobs (Counter)
- `ml_jobs_duration_seconds`: Histogram der Job-Dauer
- `ml_models_total`: Anzahl Modelle
- `ml_service_uptime_seconds`: Service-Uptime
- `ml_db_connected`: DB-Verbindungsstatus

---

## 📈 Grafana Dashboard Panels

### Panel 1: Aktive Jobs Progress

**Visualization:** Stat / Gauge

**Query:**
```promql
ml_job_progress_percent{job_type="TRAIN", status="RUNNING"}
```

**Options:**
- **Unit:** Percent (0-100)
- **Thresholds:** 
  - Green: 0-50
  - Yellow: 50-80
  - Red: 80-100

---

### Panel 2: Job-Status Übersicht

**Visualization:** Table

**Query:**
```promql
ml_job_status{job_type="TRAIN"}
```

**Columns:**
- `job_id`
- `job_type`
- `model_type`
- `status`
- `value` (1 = aktiv, 0 = inaktiv)

---

### Panel 3: Job-Progress Timeline

**Visualization:** Time Series

**Query:**
```promql
ml_job_progress_percent{job_type="TRAIN"}
```

**Options:**
- **Legend:** `{{job_id}} - {{model_type}}`
- **Y-Axis:** 0-100 (Percent)

---

### Panel 4: Job-Duration

**Visualization:** Time Series

**Query:**
```promql
ml_job_duration_seconds{job_type="TRAIN", status="RUNNING"}
```

**Options:**
- **Legend:** `Job {{job_id}} - {{model_type}}`
- **Y-Axis:** Seconds
- **Unit:** seconds

---

### Panel 5: Features & Phases

**Visualization:** Stat

**Queries:**
```promql
# Features
ml_job_features_count{job_type="TRAIN"}

# Phasen
ml_job_phases_count{job_type="TRAIN"}
```

---

### Panel 6: Job-Status Verteilung

**Visualization:** Pie Chart

**Query:**
```promql
sum by (status) (ml_job_status{job_type="TRAIN"})
```

---

### Panel 7: Aktive Jobs Count

**Visualization:** Stat

**Query:**
```promql
ml_active_jobs
```

---

## 🔧 Prometheus Scrape Config

**In `prometheus.yml` hinzufügen:**

```yaml
scrape_configs:
  - job_name: 'ml-training-service'
    metrics_path: '/api/metrics'
    static_configs:
      - targets: ['100.76.209.59:8005']
    scrape_interval: 5s
    scrape_timeout: 3s
```

---

## 📊 Beispiel Dashboard JSON

**Grafana Dashboard Import:**

```json
{
  "dashboard": {
    "title": "ML Training Service - Jobs",
    "panels": [
      {
        "title": "Job Progress",
        "targets": [
          {
            "expr": "ml_job_progress_percent{job_type=\"TRAIN\"}",
            "legendFormat": "Job {{job_id}} - {{model_type}}"
          }
        ],
        "type": "timeseries"
      },
      {
        "title": "Job Duration",
        "targets": [
          {
            "expr": "ml_job_duration_seconds{job_type=\"TRAIN\", status=\"RUNNING\"}",
            "legendFormat": "Job {{job_id}}"
          }
        ],
        "type": "timeseries"
      },
      {
        "title": "Active Jobs",
        "targets": [
          {
            "expr": "ml_active_jobs"
          }
        ],
        "type": "stat"
      }
    ]
  }
}
```

---

## 🔄 Metriken-Update

**Update-Intervall:** Alle 5 Sekunden

**Automatisch aktualisiert:**
- Job Progress (%)
- Job Duration (Sekunden)
- Job Status
- Features & Phases Count

**Manuell aktualisiert:**
- Bei Job-Status-Änderungen (PENDING → RUNNING → COMPLETED/FAILED)
- Bei Progress-Updates (0.1% → 50% → 80% → 100%)

---

## 📝 Logging

**Verbessertes Logging:**
- Job-ID in allen Log-Messages
- Detaillierte Parameter-Info
- Progress-Updates mit Timestamps
- Dauer-Tracking

**Beispiel Log:**
```
🎯 Verarbeite TRAIN Job 41
📋 Job-Details: Typ=random_forest, Features=6, Phasen=[1]
📝 Modell-Name: Test-RF-10min-30pct
📊 Job 41: Status auf RUNNING gesetzt, Progress: 0.1%
🔄 Job 41: Starte Training... (kann mehrere Minuten dauern)
✅ Job 41: Training abgeschlossen - Accuracy=0.8523, F1=0.7845
📊 Job 41: Speichere Modell in DB... (Progress: 80%)
🎉 Job 41 erfolgreich abgeschlossen: Modell 47 erstellt (Dauer: 1234.5s)
```

---

---

## 📥 Dashboard Import

**Fertiges Dashboard JSON verfügbar:**
- Datei: `docs/grafana_dashboard.json`
- **Import in Grafana:**
  1. Grafana öffnen → Dashboards → Import
  2. JSON-Datei hochladen oder Inhalt einfügen
  3. Prometheus-Datenquelle auswählen
  4. Dashboard speichern

**Dashboard enthält:**
- ✅ 16 Panels für vollständiges Monitoring
- ✅ Job Progress & Duration Time Series
- ✅ Job Status Tabelle & Pie Chart
- ✅ Service Health (DB, Uptime, Memory)
- ✅ Model Accuracy Tracking
- ✅ Features & Phases Stats
- ✅ Job Counter & Histogram

**Erstellt:** 2025-12-24  
**Version:** 1.0

