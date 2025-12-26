# ✅ Coolify Deployment Checkliste

**Datum:** 26. Dezember 2025  
**Status:** 🚀 BEREIT FÜR PRODUCTION

---

## 📋 Pre-Deployment Checkliste

### ✅ Code & Dateien
- [x] ✅ Alle Python-Dateien vorhanden (24 Dateien)
- [x] ✅ Alle SQL-Migrationen vorhanden (15 Dateien)
- [x] ✅ Dockerfile vorhanden und korrekt
- [x] ✅ docker-compose.yml vorhanden
- [x] ✅ requirements.txt vollständig
- [x] ✅ Alle kritischen Module importierbar

### ✅ Funktionalität
- [x] ✅ Modell-Erstellung funktioniert zu 100%
- [x] ✅ Labels/Tags werden korrekt gesetzt
- [x] ✅ Feature-Engineering funktioniert
- [x] ✅ Metriken-Berechnung funktioniert
- [x] ✅ API-Endpunkte funktionieren
- [x] ✅ Web UI funktioniert
- [x] ✅ Datenbank-Integration funktioniert

### ✅ Validierung
- [x] ✅ Label-Erstellung validiert (siehe `docs/LABEL_VALIDIERUNGSBERICHT.md`)
- [x] ✅ Data Leakage verhindert
- [x] ✅ Feature-Engineering validiert
- [x] ✅ Metriken-Integration validiert

---

## 🚀 Coolify Deployment Schritte

### Schritt 1: Service in Coolify erstellen

1. **Coolify öffnen** → **"New Resource"** → **"Dockerfile"**

2. **Repository konfigurieren:**
   - **Source:** Git Repository
   - **Repository URL:** `https://github.com/Chase295/ml-training-service.git`
   - **Branch:** `main`
   - **Dockerfile-Pfad:** `Dockerfile`
   - **Build-Kontext:** `.`
   - **Authentication:** GitHub App oder Personal Access Token

3. **Service-Name:** `ml-training-service`

4. **Domain (optional):**
   - **Subdomain:** `ml-training`
   - **Domain:** Deine Domain

---

### Schritt 2: Environment Variables setzen

**In Coolify: Settings → Environment Variables**

```bash
# ⚠️ KRITISCH: Externe Datenbank
DB_DSN=postgresql://postgres:9HVxi6hN6j7xpmqUx84o@100.118.155.75:5432/beta

# Ports (intern - bleiben gleich)
API_PORT=8000
STREAMLIT_PORT=8501

# Modelle-Speicherung
MODEL_STORAGE_PATH=/app/models

# ⚠️ WICHTIG: API Base URL für Streamlit
# Option 1: Mit Domain (empfohlen)
API_BASE_URL=https://ml-training.deine-domain.com/api
# ODER wenn Port direkt verwendet wird:
# API_BASE_URL=https://ml-training.deine-domain.com:8000/api

# Option 2: Mit IP-Adresse (wenn keine Domain)
# API_BASE_URL=http://DEINE_SERVER_IP:8000/api

# ⚠️ WICHTIG: Innerhalb des Containers muss auf localhost:8000 zugegriffen werden!
# Wenn Streamlit und FastAPI im selben Container sind:
# API_BASE_URL=http://localhost:8000

# Job Queue
JOB_POLL_INTERVAL=5
MAX_CONCURRENT_JOBS=2

# Logging (optional)
LOG_LEVEL=INFO
LOG_FORMAT=text
LOG_JSON_INDENT=0

# Coolify-Modus (optional)
COOLIFY_MODE=true
SERVICE_NAME=ml-training-service
```

**⚠️ WICHTIG:**
- `DB_DSN` muss die **externe Datenbank-Adresse** enthalten
- `API_BASE_URL` muss die **öffentliche URL** sein (nicht localhost, außer wenn im selben Container)
- Innerhalb des Containers: `http://localhost:8000`
- Von außen: `https://deine-domain.com/api` oder `http://IP:8000/api`

---

### Schritt 3: Volumes konfigurieren

**Settings → Volumes**

- **Volume Name:** `ml-training-models` (automatisch)
- **Container-Pfad:** `/app/models`
- **Type:** Persistent Volume
- **⚠️ WICHTIG:** Modelle bleiben erhalten bei Container-Neustart!

---

### Schritt 4: Ports konfigurieren

**Settings → Ports**

- **Port 8000:** FastAPI (API, Health, Metrics) → ✅ Public aktivieren
- **Port 8501:** Streamlit UI → ✅ Public aktivieren

**⚠️ HINWEIS:** Coolify kann automatisch Ports zuweisen. Die internen Ports (8000, 8501) bleiben gleich.

---

### Schritt 5: Health Check konfigurieren

**Settings → Health Check**

- **Path:** `/api/health`
- **Port:** `8000`
- **Interval:** `10s`
- **Timeout:** `5s`
- **Retries:** `5`
- **Start Period:** `10s`

**ODER:** Nutze den HEALTHCHECK aus dem Dockerfile (automatisch)

---

### Schritt 6: Ressourcen-Limits setzen

**Settings → Resources**

- **Memory Limit:** `8GB` (oder 80% des verfügbaren RAMs)
- **CPU Limit:** `2-4 Cores`

**⚠️ WICHTIG:** Ohne Limits kann Container bei großen Datensätzen abstürzen (OOM Kill)!

---

### Schritt 7: Deploy!

1. **Klicke auf "Deploy"**
2. **Warte auf Build** (2-5 Minuten)
3. **Prüfe Logs** in Coolify

---

## ✅ Post-Deployment Checkliste

### 1. Health Check prüfen

```bash
curl https://ml-training.deine-domain.com/api/health
# ODER
curl http://DEINE_SERVER_IP:8000/api/health
```

**Erwartet:**
```json
{
  "status": "healthy",
  "db_connected": true,
  "uptime_seconds": 123,
  "start_time": 1234567890.123,
  "total_jobs_processed": 0,
  "last_error": null
}
```

### 2. API-Endpunkte testen

```bash
# Data Availability
curl https://ml-training.deine-domain.com/api/data-availability

# Phases
curl https://ml-training.deine-domain.com/api/phases

# Models
curl https://ml-training.deine-domain.com/api/models
```

### 3. Web UI testen

```
https://ml-training.deine-domain.com:8501
# ODER
http://DEINE_SERVER_IP:8501
```

**Prüfe:**
- [ ] Dashboard lädt
- [ ] Konfiguration-Seite funktioniert
- [ ] Modelle-Übersicht funktioniert
- [ ] Training-Seite funktioniert

### 4. Erste Modell-Erstellung testen

1. **Öffne Web UI**
2. **Gehe zu "➕ Training"**
3. **Erstelle minimales Modell:**
   - Modell-Name: `TEST_PRODUCTION`
   - Modell-Typ: `random_forest`
   - Training-Zeitraum: Verfügbaren Zeitraum wählen
   - Features: `price_open`, `price_close`, `volume_sol`
   - Vorhersage-Ziel: `price_close`, 10 Minuten, 5%, "Steigt"
4. **Klicke auf "🚀 Modell trainieren"**
5. **Warte auf Completion** (1-5 Minuten)
6. **Prüfe ob Modell erstellt wurde**

### 5. Logs prüfen

**In Coolify: Logs**

**Prüfe auf:**
- ✅ Keine kritischen Fehler
- ✅ "Service started successfully"
- ✅ "Database connected"
- ✅ "Job completed successfully"

---

## 🔧 Troubleshooting

### Problem: "Connection refused" in Web UI

**Ursache:** `API_BASE_URL` ist falsch konfiguriert

**Lösung:**
- Innerhalb des Containers: `API_BASE_URL=http://localhost:8000`
- Von außen (wenn Reverse Proxy): `API_BASE_URL=https://deine-domain.com/api`
- Prüfe Environment Variables in Coolify

### Problem: Datenbank-Verbindung fehlgeschlagen

**Ursache:** Firewall oder falsche DB_DSN

**Lösung:**
1. Prüfe `DB_DSN` in Environment Variables
2. Prüfe ob Datenbank vom Coolify-Server erreichbar ist:
   ```bash
   # Vom Coolify-Server aus:
   telnet 100.118.155.75 5432
   ```
3. Prüfe Firewall-Regeln

### Problem: Modell-Erstellung schlägt fehl

**Ursache:** Zu wenig Daten oder falsche Konfiguration

**Lösung:**
1. Prüfe Daten-Verfügbarkeit: `/api/data-availability`
2. Prüfe Logs in Coolify
3. Prüfe ob genug Daten vorhanden sind (>1000 Samples empfohlen)

### Problem: Container stürzt ab (OOM Kill)

**Ursache:** Zu wenig RAM

**Lösung:**
1. Erhöhe Memory Limit in Coolify (Settings → Resources)
2. Reduziere `MAX_TRAINING_ROWS` in `app/training/feature_engineering.py` (aktuell: 500000)
3. Verwende kleinere Datensätze für Training

---

## 📊 Monitoring

### Health Check Endpoint

```bash
curl https://ml-training.deine-domain.com/api/health
```

**Überwache:**
- `status`: Sollte immer `"healthy"` sein
- `db_connected`: Sollte immer `true` sein
- `last_error`: Sollte `null` sein

### Metrics Endpoint

```bash
curl https://ml-training.deine-domain.com/api/metrics
```

**Überwache:**
- Anzahl verarbeiteter Jobs
- Fehlerrate
- Durchschnittliche Job-Dauer

---

## ✅ Finale Checkliste

### Vor Live-Deployment
- [x] ✅ Alle Tests bestanden
- [x] ✅ Code validiert
- [x] ✅ Labels/Tags validiert
- [x] ✅ Dateien vorhanden
- [x] ✅ Dokumentation vollständig

### Nach Live-Deployment
- [ ] Health Check funktioniert
- [ ] API-Endpunkte funktionieren
- [ ] Web UI funktioniert
- [ ] Erste Modell-Erstellung erfolgreich
- [ ] Logs zeigen keine Fehler

---

**Checkliste erstellt am:** 26. Dezember 2025  
**Bereit für Production:** ✅ JA

