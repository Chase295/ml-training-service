# 🔗 API_BASE_URL Erklärung

**Wichtig für Coolify/Produktion-Deployment**

---

## ⚠️ Warum nicht `localhost`?

**`API_BASE_URL` wird von Streamlit verwendet, um API-Requests zu machen.**

Diese Requests werden **vom Browser des Benutzers** ausgeführt, nicht vom Server!

### Beispiel:

1. **Benutzer öffnet:** `https://ml-training.deine-domain.com:8501` (Streamlit UI)
2. **Streamlit lädt im Browser**
3. **Streamlit macht API-Request:** `API_BASE_URL + "/api/models"`
4. **Browser sendet Request:** Vom Browser des Benutzers zum Server

**Problem mit `localhost`:**
- Browser des Benutzers versucht `http://localhost:8000/api/models` aufzurufen
- `localhost` im Browser = Benutzer's Computer, nicht der Server!
- Request schlägt fehl ❌

---

## ✅ Richtige Konfiguration

### Option 1: Mit Domain (Empfohlen)

```bash
API_BASE_URL=https://ml-training.deine-domain.com/api
```

**Oder mit Port:**
```bash
API_BASE_URL=https://ml-training.deine-domain.com:8000/api
```

### Option 2: Mit IP-Adresse

```bash
API_BASE_URL=http://100.76.209.59:8000/api
```

### Option 3: Mit Reverse Proxy

**Wenn Coolify Reverse Proxy verwendet:**
```bash
API_BASE_URL=https://ml-training.deine-domain.com/api
```

**Coolify konfiguriert automatisch:**
- Port 8000 → `/api/*` (FastAPI)
- Port 8501 → `/` (Streamlit UI)

---

## 🔍 Wie funktioniert es?

### Request-Flow:

```
Benutzer-Browser
    ↓
Streamlit UI (Port 8501)
    ↓ (macht API-Request mit API_BASE_URL)
FastAPI (Port 8000)
    ↓
Datenbank
```

**Streamlit Code:**
```python
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

# Request wird vom Browser ausgeführt:
response = httpx.get(f"{API_BASE_URL}/api/models")
```

**Wenn `API_BASE_URL=http://localhost:8000`:**
- Browser versucht `http://localhost:8000/api/models` aufzurufen
- `localhost` = Benutzer's Computer
- Server ist nicht erreichbar ❌

**Wenn `API_BASE_URL=https://ml-training.deine-domain.com/api`:**
- Browser ruft `https://ml-training.deine-domain.com/api/models` auf
- Request geht zum Server ✅

---

## 📝 Coolify-Konfiguration

### Mit Domain:

```bash
API_BASE_URL=https://ml-training.deine-domain.com/api
```

**Coolify Reverse Proxy:**
- Domain: `ml-training.deine-domain.com`
- Port 8000 → `/api/*`
- Port 8501 → `/`

### Ohne Domain (nur IP):

```bash
API_BASE_URL=http://DEINE_SERVER_IP:8000/api
```

**Beispiel:**
```bash
API_BASE_URL=http://100.76.209.59:8000/api
```

---

## ⚠️ Wichtige Hinweise

1. **`/api` am Ende:**
   - FastAPI hat Prefix `/api`
   - Endpoints sind: `/api/models`, `/api/health`, etc.
   - Daher: `API_BASE_URL=https://domain.com/api` (mit `/api`)

2. **HTTPS vs HTTP:**
   - Wenn Domain mit SSL: `https://`
   - Wenn nur IP: `http://`

3. **Port:**
   - Mit Reverse Proxy: Kein Port nötig
   - Ohne Reverse Proxy: Port angeben (`:8000`)

---

## 🧪 Testen

### Nach Deployment prüfen:

1. **Öffne Streamlit UI:** `https://ml-training.deine-domain.com:8501`
2. **Öffne Browser DevTools (F12) → Network Tab**
3. **Lade eine Seite (z.B. Übersicht)**
4. **Prüfe Requests:**
   - Sollte Requests zu `https://ml-training.deine-domain.com/api/models` machen
   - **NICHT** zu `http://localhost:8000/api/models`

### Wenn Requests zu `localhost` gehen:

- ❌ `API_BASE_URL` ist falsch konfiguriert
- ✅ Setze `API_BASE_URL` auf öffentliche URL
- ✅ Container neu starten

---

**Erstellt:** 2025-12-24  
**Version:** 1.0

