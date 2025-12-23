# 🚀 Deployment Guide - ML Training Service

## Coolify Deployment

### 1. Service erstellen

- **Service-Typ:** Dockerfile
- **Repository:** Dein Git-Repo (oder lokales Verzeichnis)
- **Dockerfile-Pfad:** `ml-training-service/Dockerfile`
- **Build-Kontext:** `ml-training-service/`

### 2. Environment Variables

Setze folgende Environment Variables in Coolify:

```bash
# ⚠️ EXTERNE Datenbank (nicht im Docker-Container!)
DB_DSN=postgresql://postgres:password@100.118.155.75:5432/crypto

# Ports (Standard)
API_PORT=8000
STREAMLIT_PORT=8501
MODEL_STORAGE_PATH=/app/models

# API Base URL für Streamlit
API_BASE_URL=http://localhost:8000

# Job Queue
JOB_POLL_INTERVAL=5
MAX_CONCURRENT_JOBS=2
```

**Wichtig:**
- `DB_DSN` muss die **externe DB-Adresse** enthalten (IP oder Hostname)
- Coolify-Container muss **Netzwerk-Zugriff** zur externen DB haben
- Prüfe Firewall/Netzwerk-Einstellungen

### 3. Volumes konfigurieren

- **Persistent Volume** für Modelle:
  - Host-Pfad: `/app/models`
  - Volume-Name: `ml-training-models` (oder automatisch)
  - **Wichtig:** Modelle bleiben erhalten bei Container-Neustart!

### 4. Ports konfigurieren

- **Port 8000** → FastAPI (Health + Metrics + API)
- **Port 8501** → Streamlit UI
- **Wichtig:** Metrics sind auf Port 8000 unter `/api/metrics` verfügbar

### 5. Health Check

- Coolify nutzt automatisch den HEALTHCHECK aus dem Dockerfile
- Oder manuell: `/api/health` Endpoint

### 6. ⚠️ KRITISCH: Ressourcen-Limits setzen (RAM-Management)!

**In Coolify: Settings → Resources → Memory Limit**

- **RAM-Limit:** Max 80% des Host-RAMs (z.B. 8GB von 10GB)
- **CPU-Limit:** 2-4 Cores für Training
- **Wichtig:** Ohne Limits kann Container bei großen Datensätzen abstürzen (OOM Kill)
- **Zusätzlich:** LIMIT in SQL Queries (500000 Zeilen) verhindert RAM-Überlauf

### 7. Deploy

- Klicke auf "Deploy"
- Warte auf Build (kann einige Minuten dauern)
- Prüfe Logs in Coolify

## Nach Deployment prüfen

### Via Coolify UI:
- Service-Status sollte "Running" sein
- Logs sollten keine Fehler zeigen
- Health Check sollte grün sein

### Via Browser/curl:
```bash
# Health Check
curl http://<coolify-url>:8000/api/health
# Erwartet: {"status": "healthy", "db_connected": true, ...}

# Metrics
curl http://<coolify-url>:8000/api/metrics
# Erwartet: Prometheus-Format mit Metriken

# Streamlit UI
http://<coolify-url>:8501
```

## Docker-spezifische Tipps

- **Logs prüfen:** Coolify zeigt Container-Logs an
- **Volume prüfen:** In Coolify → Volumes → Prüfe ob `ml-training-models` existiert
- **Redeploy:** Bei Code-Änderungen einfach "Redeploy" klicken
- **Rollback:** Coolify speichert alte Images, kann zurückgerollt werden

## Lokales Testing

### Mit Docker Compose:
```bash
cd ml-training-service

# Starten
docker-compose up -d

# Logs
docker-compose logs -f

# Stoppen
docker-compose down
```

### Mit Docker Run:
```bash
# Image bauen
docker build -t ml-training-service .

# Container starten
docker run -d \
  --name ml-training-test \
  -p 8000:8000 \
  -p 8501:8501 \
  -e DB_DSN="postgresql://postgres:password@100.118.155.75:5432/crypto" \
  -v $(pwd)/models:/app/models \
  ml-training-service

# Logs ansehen
docker logs -f ml-training-test

# Container stoppen
docker stop ml-training-test
docker rm ml-training-test
```

## Troubleshooting

### DB-Verbindungsfehler
- Prüfe ob `DB_DSN` korrekt ist
- Prüfe ob externe DB erreichbar ist (Firewall, Netzwerk)
- Prüfe Logs: `docker logs ml-training-service`

### OOM Kill (Out of Memory)
- Setze RAM-Limits in Coolify
- Reduziere `MAX_CONCURRENT_JOBS` (Standard: 2)
- Prüfe SQL LIMIT (500000 Zeilen)

### Health Check schlägt fehl
- Prüfe ob FastAPI läuft: `docker exec ml-training-service ps aux | grep uvicorn`
- Prüfe Logs auf Fehler
- Prüfe ob Port 8000 erreichbar ist

### Streamlit lädt nicht
- Prüfe ob Streamlit läuft: `docker exec ml-training-service ps aux | grep streamlit`
- Prüfe Logs auf Fehler
- Prüfe ob Port 8501 erreichbar ist

