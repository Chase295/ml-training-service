# 🚀 Coolify Deployment Guide - ML Training Service

**Version:** 1.0  
**Erstellt:** 2025-12-23  
**Status:** ✅ Vollständig

---

## 📋 Inhaltsverzeichnis

1. [Voraussetzungen](#voraussetzungen)
2. [Schritt-für-Schritt Anleitung](#schritt-für-schritt-anleitung)
3. [Environment Variables](#environment-variables)
4. [Volumes konfigurieren](#volumes-konfigurieren)
5. [Ports konfigurieren](#ports-konfigurieren)
6. [Health Checks](#health-checks)
7. [Ressourcen-Limits](#ressourcen-limits)
8. [Externe Datenbank](#externe-datenbank)
9. [Nach Deployment prüfen](#nach-deployment-prüfen)
10. [Troubleshooting](#troubleshooting)

---

## ✅ Voraussetzungen

### 1. Coolify installiert und konfiguriert
- Coolify läuft auf deinem Server
- Du hast Zugriff auf die Coolify-UI
- Git-Repository ist eingerichtet (privat auf GitHub)

### 2. Externe PostgreSQL-Datenbank
- ⚠️ **WICHTIG:** Die Datenbank läuft **EXTERN** (nicht in Coolify!)
- Datenbank ist erreichbar vom Coolify-Server aus
- Firewall-Regeln erlauben Verbindung (Port 5432)
- Datenbank-Schema ist bereits angewendet (`sql/schema.sql`)

### 3. GitHub-Zugriff für privates Repository
- **Option A (Empfohlen):** GitHub Personal Access Token (PAT)
- **Option B:** SSH Key für Git-Zugriff
- Repository: `https://github.com/Chase295/ml-training-service` (privat)

---

## 🎯 Schritt-für-Schritt Anleitung

### Schritt 1: Neuen Service in Coolify erstellen

1. **Öffne Coolify UI**
2. **Klicke auf "New Resource"** → **"Docker Compose"** oder **"Dockerfile"**
3. **Wähle "Dockerfile"** (empfohlen für Single-Container)

### Schritt 2: GitHub-Integration konfigurieren (für privates Repo)

**⚠️ WICHTIG:** Da dein Repository privat ist, benötigt Coolify Zugriff!

**Option A: GitHub App (Empfohlen - bessere Integration)**

1. **In Coolify:** Settings → Source Providers → GitHub
2. **Klicke auf "New GitHub App"** (oder "Add GitHub App")
3. **Konfiguration:**
   - **Name:** Beliebiger Name (z.B. `Coolify ML Training`)
   - **Organization:** Leer lassen (verwendet deinen GitHub-User) oder deine GitHub-Organization
   - **System Wide:** ✅ Aktivieren (empfohlen - für alle Services verfügbar)
4. **Klicke auf "Continue"**
5. **Folge den Anweisungen:**
   - Coolify erstellt automatisch eine GitHub App
   - Du wirst zu GitHub weitergeleitet
   - Autorisiere die App für dein Repository `Chase295/ml-training-service`
   - Wähle die benötigten Berechtigungen (Repository-Zugriff)
6. **Fertig!** Die GitHub App ist jetzt konfiguriert und kann für alle Services verwendet werden

**Vorteile der GitHub App:**
- ✅ Bessere Sicherheit (granulare Berechtigungen)
- ✅ Automatische Updates
- ✅ System-weit verfügbar
- ✅ Keine manuellen Token nötig

**Option B: Personal Access Token (Alternative)**

1. **Gehe zu GitHub:** https://github.com/settings/tokens
2. **Klicke auf "Generate new token" → "Generate new token (classic)"**
3. **Token konfigurieren:**
   - **Note:** `Coolify ML Training Service`
   - **Expiration:** Wähle Ablaufzeit (z.B. 90 Tage oder "No expiration")
   - **Scopes:** Aktiviere `repo` (voller Zugriff auf private Repositories)
4. **Klicke auf "Generate token"**
5. **⚠️ WICHTIG:** Kopiere den Token sofort! (wird nur einmal angezeigt)
   - Format: `ghp_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx`

**Token in Coolify speichern:**
- **Coolify UI:** Settings → Source Providers → GitHub → Token einfügen
- **Oder:** Beim Erstellen des Services direkt eingeben

### Schritt 3: Repository/Quelle konfigurieren

**In Coolify beim Erstellen des Services:**

**Option A: GitHub Repository (Empfohlen)**
- **Source:** Git Repository
- **Repository URL:** `https://github.com/Chase295/ml-training-service.git`
- **Branch:** `main`
- **Dockerfile-Pfad:** `Dockerfile` (liegt im Root des Repos)
- **Build-Kontext:** `.` (Root-Verzeichnis)
- **Authentication:**
  - **GitHub Provider:** Wähle deine konfigurierte GitHub App (wenn Option A verwendet)
  - **ODER GitHub Token:** Füge deinen PAT ein (wenn Option B verwendet)

**Option B: SSH (Alternative)**
- **Source:** Git Repository (SSH)
- **Repository URL:** `git@github.com:Chase295/ml-training-service.git`
- **SSH Key:** Füge deinen privaten SSH-Key in Coolify ein
- **Branch:** `main`
- **Dockerfile-Pfad:** `Dockerfile`
- **Build-Kontext:** `.`

### Schritt 3: Service-Name und Domain

- **Service Name:** `ml-training-service` (oder wie du möchtest)
- **Domain:** Optional - z.B. `ml-training.deine-domain.com`
  - **Subdomain:** `ml-training`
  - **Domain:** `deine-domain.com`
- **Ports:** Siehe [Ports konfigurieren](#ports-konfigurieren)

---

## 🔧 Environment Variables

### In Coolify: Settings → Environment Variables

Setze folgende Environment Variables:

```bash
# ⚠️ KRITISCH: Externe Datenbank-Verbindung
DB_DSN=postgresql://postgres:Ycy0qfClGpXPbm3Vulz1jBL0OFfCojITnbST4JBYreS5RkBCTsYc2FkbgyUstE6g@100.76.209.59:5432/crypto

# Ports (Standard - werden automatisch von Coolify gemappt)
API_PORT=8000
STREAMLIT_PORT=8501

# Modelle-Speicherung
MODEL_STORAGE_PATH=/app/models

# API Base URL für Streamlit (wichtig für interne API-Calls)
# ⚠️ WICHTIG: Verwende die öffentliche URL, über die der Service erreichbar ist!
# Option 1: Mit Domain (empfohlen für Produktion)
API_BASE_URL=https://ml-training.deine-domain.com/api
# ODER wenn Port direkt verwendet wird:
# API_BASE_URL=https://ml-training.deine-domain.com:8000/api
# Option 2: Mit IP-Adresse (wenn keine Domain)
# API_BASE_URL=http://DEINE_SERVER_IP:8000/api
# Option 3: Nur wenn Streamlit und FastAPI im selben Container sind UND keine Domain:
# API_BASE_URL=http://localhost:8000

# Job Queue Konfiguration
JOB_POLL_INTERVAL=5
MAX_CONCURRENT_JOBS=2

# Logging (optional)
LOG_LEVEL=INFO
LOG_FORMAT=text
LOG_JSON_INDENT=0
```

### ⚠️ Wichtige Hinweise:

1. **DB_DSN:**
   - Muss die **externe Datenbank-Adresse** enthalten
   - Format: `postgresql://user:password@host:port/database`
   - Coolify-Container muss **Netzwerk-Zugriff** zur externen DB haben
   - Prüfe Firewall/Netzwerk-Einstellungen

2. **API_BASE_URL:**
   - Wird von Streamlit verwendet, um die FastAPI zu erreichen
   - **⚠️ WICHTIG:** Muss die öffentliche URL sein, über die der Service erreichbar ist!
   - **Mit Domain (empfohlen):** `https://ml-training.deine-domain.com/api` oder `https://ml-training.deine-domain.com:8000/api`
   - **Mit IP:** `http://DEINE_SERVER_IP:8000/api`
   - **Nur lokal (nicht empfohlen):** `http://localhost:8000` (funktioniert nur wenn Streamlit und FastAPI im selben Container sind)
   - **Hinweis:** Streamlit sendet Requests vom Browser des Benutzers, daher muss die URL vom Browser aus erreichbar sein!

---

## 💾 Volumes konfigurieren

### In Coolify: Settings → Volumes

**Persistentes Volume für Modelle:**

- **Volume Name:** `ml-training-models` (oder automatisch generiert)
- **Host-Pfad:** `/app/models` (oder Coolify-Standard-Pfad)
- **Container-Pfad:** `/app/models`
- **Type:** Persistent Volume

⚠️ **WICHTIG:** 
- Modelle bleiben erhalten bei Container-Neustart
- Ohne Volume gehen alle trainierten Modelle verloren!
- Volume wird automatisch von Coolify verwaltet

---

## 🔌 Ports konfigurieren

### In Coolify: Settings → Ports

**Port 8000 - FastAPI (API, Health, Metrics):**
- **Container Port:** `8000`
- **Public Port:** `8000` (oder automatisch)
- **Protocol:** TCP
- **Public:** ✅ Aktiviert (wenn öffentlich erreichbar)

**Port 8501 - Streamlit UI:**
- **Container Port:** `8501`
- **Public Port:** `8501` (oder automatisch)
- **Protocol:** TCP
- **Public:** ✅ Aktiviert (wenn öffentlich erreichbar)

### Alternative: Reverse Proxy (empfohlen für Produktion)

**Mit Traefik/Nginx:**
- Port 8000 → `/api/*` (API-Endpunkte)
- Port 8501 → `/` (Streamlit UI)
- SSL/TLS automatisch via Let's Encrypt

**Coolify Reverse Proxy:**
- Coolify kann automatisch Reverse Proxy konfigurieren
- Domain: `ml-training.deine-domain.com`
- SSL: Automatisch via Let's Encrypt

---

## 🏥 Health Checks

### Automatisch (via Dockerfile)

Das Dockerfile enthält bereits einen HEALTHCHECK:
```dockerfile
HEALTHCHECK --interval=10s --timeout=5s --start-period=10s --retries=5 \
    CMD curl -f http://localhost:8000/api/health || exit 1
```

Coolify nutzt diesen automatisch.

### Manuell konfigurieren (optional)

**In Coolify: Settings → Health Check**

- **Health Check Path:** `/api/health`
- **Health Check Port:** `8000`
- **Interval:** `10s`
- **Timeout:** `5s`
- **Retries:** `5`
- **Start Period:** `10s`

**Health Check Response:**
```json
{
  "status": "healthy",
  "db_connected": true,
  "uptime_seconds": 1234
}
```

---

## 💪 Ressourcen-Limits

### ⚠️ KRITISCH: RAM-Management!

**In Coolify: Settings → Resources**

**Memory Limits:**
- **Memory Limit:** `8GB` (oder 80% des verfügbaren RAMs)
- **Memory Reservation:** `4GB` (empfohlenes Minimum)

**CPU Limits:**
- **CPU Limit:** `2-4 Cores` (für Training)
- **CPU Reservation:** `1 Core`

**Warum wichtig?**
- ML-Training kann sehr RAM-intensiv sein
- Ohne Limits kann Container bei großen Datensätzen abstürzen (OOM Kill)
- SQL Queries haben bereits LIMIT (500000 Zeilen) um RAM-Überlauf zu verhindern

**Zusätzliche Sicherheit:**
- `MAX_CONCURRENT_JOBS=2` verhindert zu viele parallele Trainings
- SQL LIMIT in Queries verhindert zu große Datensätze

---

## 🗄️ Externe Datenbank

### Voraussetzungen

1. **Datenbank läuft extern** (nicht in Coolify)
2. **Netzwerk-Zugriff:** Coolify-Server kann Datenbank erreichen
3. **Firewall:** Port 5432 ist offen (oder VPN-Tunnel)
4. **Schema:** Datenbank-Schema ist angewendet (`sql/schema.sql`)

### Verbindung prüfen

**Von Coolify-Server aus:**
```bash
# Teste Verbindung
psql -h 100.76.209.59 -p 5432 -U postgres -d crypto -c "SELECT 1;"
```

**Von Container aus (nach Deployment):**
```bash
# In Coolify: Logs → Execute Command
curl http://localhost:8000/api/health
# Sollte {"db_connected": true} zurückgeben
```

### Firewall-Regeln

**Wenn Datenbank auf separatem Server:**
- Erlaube Verbindungen von Coolify-Server-IP
- PostgreSQL `pg_hba.conf` konfigurieren
- Oder VPN-Tunnel zwischen Servern

---

## ✅ Nach Deployment prüfen

### 1. Service-Status

**In Coolify UI:**
- Service-Status sollte **"Running"** sein
- Health Check sollte **grün** sein
- Logs sollten keine Fehler zeigen

### 2. Health Check testen

```bash
curl http://deine-domain.com:8000/api/health
# ODER
curl http://localhost:8000/api/health
```

**Erwartete Response:**
```json
{
  "status": "healthy",
  "db_connected": true,
  "uptime_seconds": 123
}
```

### 3. API-Endpunkte testen

```bash
# Modelle auflisten
curl http://deine-domain.com:8000/api/models

# Phasen abrufen
curl http://deine-domain.com:8000/api/phases

# Daten-Verfügbarkeit prüfen
curl http://deine-domain.com:8000/api/data-availability
```

### 4. Streamlit UI testen

**Öffne im Browser:**
```
http://deine-domain.com:8501
# ODER
https://ml-training.deine-domain.com (wenn Reverse Proxy)
```

**Erwartetes Verhalten:**
- UI lädt ohne Fehler
- Seiten sind navigierbar
- Modelle können aufgelistet werden

### 5. Logs prüfen

**In Coolify: Logs**

**Erfolgreiche Logs:**
```
✅ Datenbank-Pool erstellt: 100.76.209.59:5432
✅ FastAPI gestartet auf Port 8000
✅ Streamlit gestartet auf Port 8501
```

**Fehler-Logs:**
```
❌ Fehler beim Erstellen des DB-Pools: ...
❌ Connection refused to database
```

---

## 🔍 Troubleshooting

### Problem 1: Datenbank-Verbindung fehlgeschlagen

**Symptome:**
- Health Check zeigt `"db_connected": false`
- Logs: `❌ Fehler beim Erstellen des DB-Pools`

**Lösungen:**
1. **Prüfe DB_DSN Environment Variable:**
   - Format: `postgresql://user:password@host:port/database`
   - Keine Leerzeichen!
   - Sonderzeichen in Password URL-encoden

2. **Prüfe Netzwerk-Zugriff:**
   ```bash
   # Von Coolify-Server aus
   telnet 100.76.209.59 5432
   # ODER
   nc -zv 100.76.209.59 5432
   ```

3. **Prüfe Firewall:**
   - Port 5432 muss offen sein
   - Coolify-Server-IP muss erlaubt sein

4. **Prüfe PostgreSQL-Konfiguration:**
   - `pg_hba.conf` erlaubt Verbindungen von Coolify-Server
   - `postgresql.conf` hat `listen_addresses = '*'`

### Problem 2: Container startet nicht

**Symptome:**
- Service-Status: "Failed" oder "Restarting"
- Logs zeigen Start-Fehler

**Lösungen:**
1. **Prüfe Dockerfile:**
   - Dockerfile ist korrekt
   - Build-Kontext ist richtig gesetzt

2. **Prüfe Logs:**
   - Coolify → Logs → Zeige alle Logs
   - Suche nach Fehlermeldungen

3. **Prüfe Ressourcen:**
   - RAM/CPU-Limits sind nicht zu niedrig
   - Host hat genug Ressourcen

### Problem 3: Ports nicht erreichbar

**Symptome:**
- Health Check schlägt fehl
- API/UI nicht erreichbar

**Lösungen:**
1. **Prüfe Port-Konfiguration:**
   - Ports sind in Coolify konfiguriert
   - Public Ports sind aktiviert

2. **Prüfe Firewall:**
   - Ports 8000 und 8501 sind offen
   - Coolify-Firewall erlaubt eingehende Verbindungen

3. **Prüfe Reverse Proxy:**
   - Wenn Reverse Proxy verwendet wird, prüfe Konfiguration
   - Domain zeigt auf richtige Ports

### Problem 4: Modelle gehen verloren

**Symptome:**
- Nach Container-Neustart sind Modelle weg

**Lösungen:**
1. **Prüfe Volume-Konfiguration:**
   - Volume ist in Coolify konfiguriert
   - Container-Pfad: `/app/models`
   - Volume ist persistent (nicht ephemeral)

2. **Prüfe Volume-Mount:**
   - In Logs sollte kein Fehler zu `/app/models` sein
   - Volume ist gemountet

### Problem 5: Training-Jobs schlagen fehl

**Symptome:**
- Jobs haben Status "FAILED"
- Logs zeigen Fehler

**Lösungen:**
1. **Prüfe RAM-Limits:**
   - Container hat genug RAM
   - Erhöhe Memory Limit falls nötig

2. **Prüfe Datenbank:**
   - Datenbank ist erreichbar
   - Schema ist korrekt angewendet

3. **Prüfe Logs:**
   - Job-Logs zeigen spezifische Fehler
   - API-Logs zeigen Request-Fehler

---

## 📝 Zusammenfassung

### Checkliste für erfolgreiches Deployment:

- [ ] Coolify ist installiert und läuft
- [ ] Externe Datenbank ist erreichbar
- [ ] Service in Coolify erstellt (Dockerfile)
- [ ] Repository/Quelle konfiguriert
- [ ] Environment Variables gesetzt (besonders `DB_DSN`)
- [ ] Persistent Volume für `/app/models` konfiguriert
- [ ] Ports 8000 und 8501 sind konfiguriert
- [ ] Health Check funktioniert
- [ ] RAM-Limits sind gesetzt (8GB empfohlen)
- [ ] Service läuft und Health Check ist grün
- [ ] API ist erreichbar (`/api/health`)
- [ ] Streamlit UI ist erreichbar (Port 8501)
- [ ] Logs zeigen keine Fehler

---

## 🎯 Quick Start

### Minimal-Konfiguration:

1. **Service erstellen:** Dockerfile, Repository konfigurieren
2. **Environment Variables:**
   ```
   DB_DSN=postgresql://user:pass@host:5432/crypto
   API_BASE_URL=http://localhost:8000
   ```
3. **Volume:** `/app/models` → Persistent
4. **Ports:** 8000, 8501 → Public
5. **Deploy:** Klicke auf "Deploy"

---

**Erstellt:** 2025-12-23  
**Version:** 1.0  
**Status:** ✅ Vollständig

