# 📊 Grafana Dashboard für ML Training Service

Dieses Dashboard bietet eine vollständige Live-Übersicht über den ML Training Service mit allen wichtigen Metriken.

## 🚀 Installation

### 1. Dashboard importieren

1. Öffne Grafana (normalerweise `http://localhost:3000`)
2. Gehe zu **Dashboards** → **Import**
3. Wähle die Datei `dashboard_ml_training_service.json` aus
4. Oder kopiere den JSON-Inhalt direkt in das Import-Feld

### 2. Prometheus-Datasource konfigurieren

**Wichtig:** Das Dashboard verwendet die Variable `${DS_PROMETHEUS}` für die Prometheus-Datasource.

1. Gehe zu **Configuration** → **Data Sources**
2. Füge eine neue **Prometheus**-Datasource hinzu (falls noch nicht vorhanden)
3. URL: `http://localhost:8012/metrics` (oder die URL deines Prometheus-Servers)
4. Speichere und teste die Verbindung

**Alternative:** Wenn du bereits eine Prometheus-Datasource hast:
1. Öffne das Dashboard
2. Klicke auf das Zahnrad-Symbol (⚙️) oben rechts
3. Gehe zu **Variables**
4. Wähle `DS_PROMETHEUS` aus
5. Wähle deine Prometheus-Datasource aus der Liste

## 📋 Dashboard-Panels

### 🔴 Service Health (Obere Reihe)

- **Datenbank Status** (Gauge): Zeigt an, ob die Datenbank verbunden ist (grün = online, rot = offline)
- **Service Uptime** (Time Series): Zeigt die Laufzeit des Services in Sekunden
- **Gesamt Modelle** (Stat): Anzahl der trainierten Modelle
- **Aktive Jobs** (Stat): Anzahl der aktuell laufenden Jobs
- **Job Progress** (Gauge): Fortschritt des aktuellen Jobs in Prozent

### 💾 System-Ressourcen

- **Memory Usage** (Time Series): Resident und Virtual Memory Verbrauch
- **CPU Usage** (Time Series): CPU-Auslastung (Rate)
- **Python GC Collections** (Time Series): Garbage Collection Statistiken nach Generation
- **File Descriptors** (Time Series): Anzahl offener Datei-Deskriptoren

### 🤖 ML-Metriken

- **Training Accuracy** (Time Series): Training Accuracy aller Modelle (nach model_id)
- **Test Accuracy** (Time Series): Test Accuracy aller Modelle (nach model_id)
- **Job Duration** (Time Series): Laufzeit der Jobs
- **Python GC Objects Collected** (Time Series): Rate der gesammelten GC-Objekte
- **Job Features Count** (Time Series): Anzahl Features pro Job
- **Job Phases Count** (Time Series): Anzahl Phasen pro Job

## ⚙️ Konfiguration

### Refresh-Intervall

Das Dashboard aktualisiert sich automatisch alle **10 Sekunden**. Du kannst dies ändern:

1. Klicke auf das Uhr-Symbol oben rechts
2. Wähle ein anderes Refresh-Intervall (z.B. 5s, 30s, 1m)

### Zeitbereich

Standardmäßig zeigt das Dashboard die letzten **1 Stunde** an. Du kannst den Zeitbereich ändern:

1. Klicke auf das Uhr-Symbol oben rechts
2. Wähle einen anderen Zeitbereich (z.B. "Last 6 hours", "Last 24 hours")

### Alerts (Optional)

Du kannst Alerts für kritische Metriken einrichten:

1. Klicke auf ein Panel
2. Wähle **Edit**
3. Gehe zum Tab **Alert**
4. Erstelle eine Alert-Regel (z.B. "DB disconnected" wenn `ml_db_connected == 0`)

## 📊 Verfügbare Metriken

### Service-Metriken

- `ml_service_uptime_seconds` - Service Laufzeit
- `ml_db_connected` - Datenbank-Verbindungsstatus (1 = verbunden, 0 = getrennt)
- `ml_models_total` - Gesamtanzahl Modelle
- `ml_active_jobs` - Anzahl aktiver Jobs

### Job-Metriken

- `ml_jobs_total{job_type, status}` - Gesamtanzahl Jobs (Counter)
- `ml_jobs_duration_seconds{job_type}` - Job-Dauer (Histogram)
- `ml_job_progress_percent{job_id, job_type, model_type}` - Job-Fortschritt (0-100%)
- `ml_job_duration_seconds{job_id, job_type, model_type}` - Aktuelle Job-Dauer
- `ml_job_status{job_id, job_type, model_type, status}` - Job-Status (1=PENDING, 2=RUNNING, 3=COMPLETED, 4=FAILED, 5=CANCELLED)
- `ml_job_features_count{job_id, job_type}` - Anzahl Features pro Job
- `ml_job_phases_count{job_id, job_type}` - Anzahl Phasen pro Job

### Modell-Metriken

- `ml_training_accuracy{model_id}` - Training Accuracy pro Modell
- `ml_test_accuracy{model_id}` - Test Accuracy pro Modell

### System-Metriken (Standard Prometheus)

- `process_resident_memory_bytes` - Resident Memory
- `process_virtual_memory_bytes` - Virtual Memory
- `process_cpu_seconds_total` - CPU-Zeit (als Rate verwenden)
- `process_open_fds` - Offene File Descriptors
- `process_max_fds` - Maximale File Descriptors
- `python_gc_collections_total{generation}` - GC Collections
- `python_gc_objects_collected_total{generation}` - Gesammelte GC-Objekte

## 🔍 PromQL-Beispiele

### Job Success Rate

```promql
sum(rate(ml_jobs_total{status="COMPLETED"}[5m])) / sum(rate(ml_jobs_total[5m]))
```

### Durchschnittliche Job-Dauer

```promql
avg(ml_job_duration_seconds)
```

### Memory Usage in MB

```promql
process_resident_memory_bytes / 1024 / 1024
```

### CPU Usage in Prozent

```promql
rate(process_cpu_seconds_total[5m]) * 100
```

## 🐛 Troubleshooting

### Dashboard zeigt keine Daten

1. **Prüfe Prometheus-Datasource:**
   - Gehe zu **Configuration** → **Data Sources**
   - Teste die Verbindung zur Prometheus-Datasource
   - Stelle sicher, dass die URL korrekt ist

2. **Prüfe Metriken-Endpoint:**
   - Öffne `http://localhost:8012/metrics` im Browser
   - Stelle sicher, dass Metriken im Prometheus-Format ausgegeben werden

3. **Prüfe Zeitbereich:**
   - Stelle sicher, dass der gewählte Zeitbereich Daten enthält
   - Versuche einen größeren Zeitbereich (z.B. "Last 24 hours")

### Metriken fehlen

1. **Prüfe Service-Status:**
   - Stelle sicher, dass der ML Training Service läuft
   - Prüfe die Logs auf Fehler

2. **Prüfe Metriken-Export:**
   - Der Service muss Metriken unter `/metrics` exportieren
   - Prüfe die `app/utils/metrics.py` Datei

## 📝 Anpassungen

### Weitere Panels hinzufügen

1. Klicke auf **Add** → **Visualization**
2. Wähle einen Panel-Typ (z.B. Time Series, Stat, Gauge)
3. Konfiguriere die PromQL-Query
4. Speichere das Panel

### Dashboard teilen

1. Klicke auf das Teilen-Symbol (📤) oben rechts
2. Wähle **Export** → **Save to file**
3. Oder teile den Link mit anderen Benutzern

## 🎨 Design-Anpassungen

Das Dashboard verwendet das **Dark Theme** standardmäßig. Du kannst das Theme ändern:

1. Klicke auf dein Profil (unten links)
2. Gehe zu **Preferences**
3. Wähle ein anderes Theme (Light, Dark, System)

## 📚 Weitere Ressourcen

- [Grafana Dokumentation](https://grafana.com/docs/grafana/latest/)
- [Prometheus Query Language (PromQL)](https://prometheus.io/docs/prometheus/latest/querying/basics/)
- [ML Training Service Dokumentation](../docs/)

