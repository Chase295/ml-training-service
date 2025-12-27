# 🐳 Docker Test-Anleitung

**Datum:** 2025-12-27  
**Status:** ✅ Docker-Container erfolgreich gebaut und gestartet

---

## ✅ Container-Status

### Container läuft:
- **Container-Name:** `ml-training-service`
- **Status:** `healthy` ✅
- **Ports:**
  - **API:** `http://localhost:8012` (FastAPI)
  - **Web UI:** `http://localhost:8502` (Streamlit)

### Services:
- ✅ FastAPI Service (Port 8000 intern → 8012 extern)
- ✅ Streamlit UI (Port 8501 intern → 8502 extern)
- ✅ Datenbank verbunden (100.118.155.75:5432/beta)
- ✅ Job Worker läuft
- ✅ Health Check erfolgreich

---

## 🚀 Schnellstart

### 1. Container starten
```bash
cd ml-training-service
docker-compose up -d
```

### 2. Container-Status prüfen
```bash
docker-compose ps
```

### 3. Logs anzeigen
```bash
docker-compose logs -f ml-training
```

### 4. Health-Check
```bash
curl http://localhost:8012/api/health
```

### 5. Web UI öffnen
Öffne im Browser: **http://localhost:8502**

---

## 📊 Verfügbare Endpunkte

### API (Port 8012):
- **Health:** `http://localhost:8012/api/health`
- **Metrics:** `http://localhost:8012/api/metrics`
- **Data Availability:** `http://localhost:8012/api/data-availability`
- **Models:** `http://localhost:8012/api/models`
- **Jobs:** `http://localhost:8012/api/queue`

### Web UI (Port 8502):
- **Dashboard:** `http://localhost:8502` (Tab: Dashboard)
- **Modelle:** `http://localhost:8502` (Tab: Modelle)
- **Training:** `http://localhost:8502` (Tab: Training)
- **Info:** `http://localhost:8502` (Tab: Info)

---

## 🔧 Wichtige Befehle

### Container stoppen
```bash
docker-compose stop
```

### Container neu starten
```bash
docker-compose restart
```

### Container entfernen (aber Image behalten)
```bash
docker-compose down
```

### Container entfernen + Volumes löschen
```bash
docker-compose down -v
```

### Image neu bauen
```bash
docker-compose build --no-cache
```

### Logs in Echtzeit
```bash
docker-compose logs -f ml-training
```

### In Container einsteigen
```bash
docker-compose exec ml-training bash
```

---

## 🗄️ Datenbank-Konfiguration

Die Datenbank-Verbindung ist in `docker-compose.yml` konfiguriert:

```yaml
environment:
  - DB_DSN=postgresql://postgres:9HVxi6hN6j7xpmqUx84o@100.118.155.75:5432/beta
```

**Wichtig:** Die Datenbank ist extern (nicht im Docker-Compose). Stelle sicher, dass:
- Die Datenbank erreichbar ist
- Die Zugangsdaten korrekt sind
- Die benötigten Tabellen existieren (`coin_metrics`, `coin_streams`, etc.)

---

## 📁 Volumes

### Models-Verzeichnis
Das `./models` Verzeichnis wird als Volume gemappt:
- **Host:** `./models` (im Projekt-Verzeichnis)
- **Container:** `/app/models`

**Zweck:** Modelle werden persistent gespeichert (bleiben erhalten nach Container-Neustart)

---

## 🧪 Test-Ergebnisse

### ✅ Erfolgreiche Tests:
1. ✅ Docker-Image gebaut
2. ✅ Container gestartet
3. ✅ Health Check erfolgreich
4. ✅ Datenbank verbunden
5. ✅ FastAPI Service läuft
6. ✅ Streamlit UI läuft
7. ✅ Job Worker aktiv

### 📊 Health-Check Response:
```json
{
    "status": "healthy",
    "db_connected": true,
    "uptime_seconds": 3,
    "start_time": 1766850056.2597747,
    "total_jobs_processed": 0,
    "last_error": null
}
```

---

## ⚠️ Bekannte Warnungen

### Pydantic-Warnungen:
```
Field "model_id" has conflict with protected namespace "model_".
```

**Status:** Harmlos, kann ignoriert werden. Funktionalität nicht beeinträchtigt.

---

## 🎯 Nächste Schritte

1. **Web UI öffnen:** http://localhost:8502
2. **Test-Training erstellen:** Tab "Training" → Neues Modell trainieren
3. **ATH-Features testen:** Prüfe ob ATH-Daten korrekt geladen werden
4. **Modell testen:** Tab "Testen" → Modell auf neuen Daten testen

---

## 📝 Troubleshooting

### Container startet nicht:
```bash
# Prüfe Logs
docker-compose logs ml-training

# Prüfe ob Ports belegt sind
lsof -i :8012
lsof -i :8502
```

### Datenbank-Verbindung fehlgeschlagen:
```bash
# Prüfe DB_DSN in docker-compose.yml
# Teste Verbindung manuell:
psql "postgresql://postgres:9HVxi6hN6j7xpmqUx84o@100.118.155.75:5432/beta"
```

### Container ist nicht gesund:
```bash
# Prüfe Health-Check
curl http://localhost:8012/api/health

# Prüfe Logs
docker-compose logs ml-training | tail -50
```

---

**Erstellt:** 2025-12-27  
**Status:** ✅ Container läuft und ist bereit für Tests


