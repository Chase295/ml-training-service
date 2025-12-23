# 🚀 Coolify Quick Start - ML Training Service

**Für private GitHub-Repositories**

---

## ⚡ Schnellstart (5 Minuten)

### 1. GitHub-Integration in Coolify konfigurieren

**Option A: GitHub App (Empfohlen - siehe Bild oben)**

1. **In Coolify:** Settings → Source Providers → GitHub
2. **Klicke auf "New GitHub App"** (oder ähnlicher Button)
3. **Konfiguration:**
   - **Name:** Beliebiger Name (z.B. `Coolify ML Training`)
   - **Organization:** Leer lassen (verwendet deinen GitHub-User)
   - **System Wide:** ✅ Aktivieren (für alle Services verfügbar)
4. **Klicke auf "Continue"**
5. **Folge den Anweisungen:**
   - Coolify erstellt automatisch eine GitHub App
   - Du wirst zu GitHub weitergeleitet
   - Autorisiere die App für dein Repository
6. **Fertig!** Die GitHub App ist jetzt konfiguriert

**Option B: Personal Access Token (Alternative)**

1. Gehe zu: https://github.com/settings/tokens
2. Klicke: **"Generate new token" → "Generate new token (classic)"**
3. Konfiguration:
   - **Note:** `Coolify ML Training`
   - **Expiration:** `90 days` (oder "No expiration")
   - **Scopes:** ✅ `repo` (voller Zugriff auf private Repos)
4. Klicke: **"Generate token"**
5. **⚠️ Kopiere den Token sofort!** (Format: `ghp_xxxxxxxxxxxx...`)
6. **In Coolify:** Settings → Source Providers → GitHub → Token einfügen

---

### 2. Service in Coolify erstellen

1. **Coolify öffnen** → **"New Resource"** → **"Dockerfile"**

2. **Repository konfigurieren:**
   - **Source:** `Git Repository`
   - **Repository URL:** `https://github.com/Chase295/ml-training-service.git`
   - **Branch:** `main`
   - **Dockerfile-Pfad:** `Dockerfile`
   - **Build-Kontext:** `.`
   - **GitHub Provider:** Wähle deine konfigurierte GitHub App (wenn Option A verwendet)
   - **ODER GitHub Token:** Füge deinen PAT ein (wenn Option B verwendet)

3. **Service-Name:** `ml-training-service`

---

### 3. Environment Variables setzen

**In Coolify: Settings → Environment Variables**

```bash
# ⚠️ KRITISCH: Externe Datenbank
DB_DSN=postgresql://postgres:Ycy0qfClGpXPbm3Vulz1jBL0OFfCojITnbST4JBYreS5RkBCTsYc2FkbgyUstE6g@100.76.209.59:5432/crypto

# Ports
API_PORT=8000
STREAMLIT_PORT=8501

# Modelle
MODEL_STORAGE_PATH=/app/models

# API Base URL (wichtig für Streamlit)
# ⚠️ WICHTIG: Verwende die öffentliche URL deines Servers!
# Mit Domain:
API_BASE_URL=https://ml-training.deine-domain.com/api
# ODER mit IP:
# API_BASE_URL=http://DEINE_SERVER_IP:8000/api
# ODER wenn Reverse Proxy ohne Port:
# API_BASE_URL=https://ml-training.deine-domain.com/api

# Job Queue
JOB_POLL_INTERVAL=5
MAX_CONCURRENT_JOBS=2
```

---

### 4. Volumes konfigurieren

**Settings → Volumes**

- **Volume Name:** `ml-training-models` (automatisch)
- **Container-Pfad:** `/app/models`
- **Type:** Persistent Volume

---

### 5. Ports konfigurieren

**Settings → Ports**

- **Port 8000:** FastAPI (API, Health, Metrics)
- **Port 8501:** Streamlit UI

**Beide Ports:** ✅ Public aktivieren

---

### 6. Ressourcen-Limits setzen

**Settings → Resources**

- **Memory Limit:** `8GB` (oder 80% des verfügbaren RAMs)
- **CPU Limit:** `2-4 Cores`

---

### 7. Deploy!

**Klicke auf "Deploy"** und warte auf Build (2-5 Minuten)

---

## ✅ Nach Deployment prüfen

### Health Check:
```bash
curl http://deine-coolify-url:8000/api/health
```

**Erwartet:**
```json
{
  "status": "healthy",
  "db_connected": true,
  "uptime_seconds": 123
}
```

### Streamlit UI:
```
http://deine-coolify-url:8501
```

---

## 🔧 Troubleshooting

### Problem: "Repository not found" oder "Authentication failed"

**Lösung:**
1. Prüfe ob GitHub Token korrekt ist
2. Prüfe ob Token `repo` Scope hat
3. Prüfe ob Token nicht abgelaufen ist
4. Erstelle neuen Token falls nötig

### Problem: Build schlägt fehl

**Lösung:**
1. Prüfe Logs in Coolify
2. Prüfe ob Dockerfile-Pfad korrekt ist (`Dockerfile`)
3. Prüfe ob Build-Kontext korrekt ist (`.`)

### Problem: Datenbank-Verbindung fehlgeschlagen

**Lösung:**
1. Prüfe `DB_DSN` Environment Variable
2. Prüfe ob Datenbank vom Coolify-Server erreichbar ist
3. Prüfe Firewall-Regeln (Port 5432)

---

## 📝 Checkliste

- [ ] GitHub Personal Access Token erstellt (mit `repo` Scope)
- [ ] Service in Coolify erstellt (Dockerfile)
- [ ] Repository URL korrekt: `https://github.com/Chase295/ml-training-service.git`
- [ ] GitHub Token in Coolify eingetragen
- [ ] Environment Variables gesetzt (besonders `DB_DSN`)
- [ ] Persistent Volume für `/app/models` konfiguriert
- [ ] Ports 8000 und 8501 konfiguriert
- [ ] RAM-Limits gesetzt (8GB empfohlen)
- [ ] Service deployed
- [ ] Health Check erfolgreich
- [ ] Streamlit UI erreichbar

---

**Erstellt:** 2025-12-24  
**Version:** 1.0

