# 🚀 Coolify Deployment mit Docker Compose

**Einfache Methode ohne GitHub App/Token**

---

## ⚡ Schnellstart

### 1. Repository kurzzeitig öffentlich machen (nur für Setup)

1. **GitHub Repository:** Settings → Danger Zone → Change visibility → Make public
2. **⚠️ WICHTIG:** Nach dem Setup wieder auf privat setzen!

**ODER:** Nutze einen öffentlichen Gist oder Paste-Service für die docker-compose.yml

### 2. Service in Coolify erstellen

1. **Coolify öffnen** → **"New Resource"** → **"Docker Compose"**
2. **Repository konfigurieren:**
   - **Source:** `Git Repository`
   - **Repository URL:** `https://github.com/Chase295/ml-training-service.git`
   - **Branch:** `main`
   - **Docker Compose File:** `docker-compose.coolify.yml`
   - **Keine Authentifizierung nötig** (wenn Repository öffentlich ist)

3. **Service-Name:** `ml-training-service`

---

### 3. Environment Variables setzen

**In Coolify: Settings → Environment Variables**

```bash
# ⚠️ KRITISCH: Externe Datenbank
DB_DSN=postgresql://postgres:Ycy0qfClGpXPbm3Vulz1jBL0OFfCojITnbST4JBYreS5RkBCTsYc2FkbgyUstE6g@100.76.209.59:5432/crypto

# ⚠️ WICHTIG: Öffentliche URL, nicht localhost!
API_BASE_URL=https://ml-training.deine-domain.com/api
# ODER mit IP:
# API_BASE_URL=http://DEINE_SERVER_IP:8000/api

# Optional (Standard-Werte sind bereits in docker-compose.yml)
JOB_POLL_INTERVAL=5
MAX_CONCURRENT_JOBS=2
LOG_LEVEL=INFO
```

---

### 4. Volumes prüfen

**Coolify erstellt automatisch:**
- Volume: `ml-training-models` → `/app/models` im Container

**Keine manuelle Konfiguration nötig!**

---

### 5. Ports prüfen

**Coolify erkennt automatisch aus docker-compose.yml:**
- Port 8000 → FastAPI
- Port 8501 → Streamlit UI

**Beide Ports:** ✅ Public aktivieren (in Coolify Settings)

---

### 6. Ressourcen-Limits setzen

**Settings → Resources**

- **Memory Limit:** `8GB` (empfohlen)
- **CPU Limit:** `2-4 Cores`

---

### 7. Deploy!

**Klicke auf "Deploy"** und warte auf Build (2-5 Minuten)

**Nach erfolgreichem Deployment:**
- ✅ Repository wieder auf **privat** setzen (GitHub Settings)

---

## 📝 Docker Compose File

**Datei:** `docker-compose.coolify.yml`

**Wichtig:**
- Verwendet Environment Variables (werden von Coolify gesetzt)
- Persistentes Volume für Modelle
- Health Check konfiguriert
- Beide Ports (8000, 8501) freigegeben

---

## ✅ Nach Deployment prüfen

### Health Check:
```bash
curl http://deine-coolify-url:8000/api/health
```

### Streamlit UI:
```
http://deine-coolify-url:8501
```

---

## 🔄 Repository wieder privat machen

**Nach erfolgreichem Deployment:**

1. **GitHub Repository:** Settings → Danger Zone → Change visibility → Make private
2. **Coolify funktioniert weiterhin** (hat bereits den Code geladen)
3. **Bei Updates:** Repository kurzzeitig öffentlich machen → Coolify pullt Updates → Wieder privat

**ODER:** Nutze GitHub App/Token (siehe andere Anleitung) für dauerhaften Zugriff

---

## 🎯 Vorteile dieser Methode

- ✅ Keine GitHub App/Token nötig
- ✅ Einfache Konfiguration
- ✅ Docker Compose ist vertraut
- ✅ Alle Services in einer Datei

## ⚠️ Nachteile

- ❌ Repository muss kurzzeitig öffentlich sein
- ❌ Bei Updates muss Repository wieder öffentlich gemacht werden

---

**Erstellt:** 2025-12-24  
**Version:** 1.0

