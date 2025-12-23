# ✅ Phase 2.3: Strukturierte Logs - Implementierung

**Datum:** 2024-12-23  
**Status:** ✅ Implementiert

---

## 📊 Übersicht

Strukturiertes Logging-System mit folgenden Features:
- ✅ JSON-Logging (optional)
- ✅ Konfigurierbares Log-Level
- ✅ Request-ID für Tracing
- ✅ Strukturierte Log-Messages

---

## 🔧 Implementierung

### 1. Logging-Modul erstellt
**Datei:** `app/utils/logging_config.py`

**Features:**
- `StructuredFormatter`: Formatiert Logs zu JSON oder strukturiertem Text
- `setup_logging()`: Konfiguriert Logging für die gesamte Anwendung
- `get_logger()`: Gibt Logger mit korrekter Konfiguration zurück
- `set_request_id()` / `get_request_id()`: Request-ID Management
- `log_with_context()`: Loggt mit zusätzlichen Context-Feldern

### 2. Request-ID Middleware
**Datei:** `app/main.py`

**Features:**
- `RequestIDMiddleware`: Generiert Request-ID für jeden Request
- Request-ID wird in Response-Header zurückgegeben (`X-Request-ID`)
- Request-ID wird in allen Logs automatisch eingefügt

### 3. Konfiguration
**Datei:** `app/utils/config.py`

**Environment Variables:**
- `LOG_LEVEL`: Log-Level (DEBUG, INFO, WARNING, ERROR, CRITICAL) - Default: INFO
- `LOG_FORMAT`: Format ("text" oder "json") - Default: "text"
- `LOG_JSON_INDENT`: JSON-Indentation (0 = kompakt, 2+ = formatiert) - Default: 0

---

## 📝 Verwendung

### Logger erstellen
```python
from app.utils.logging_config import get_logger

logger = get_logger(__name__)
logger.info("Nachricht")
```

### Mit Context loggen
```python
from app.utils.logging_config import log_with_context, get_logger

logger = get_logger(__name__)
log_with_context(
    logger,
    logging.INFO,
    "Job erstellt",
    extra_fields={"job_id": 123, "model_type": "random_forest"}
)
```

### Request-ID verwenden
```python
from app.utils.logging_config import get_request_id

request_id = get_request_id()  # Wird automatisch gesetzt
```

---

## 📊 Log-Formate

### Text-Format (Default)
```
[2024-12-23T21:00:00+00:00] [INFO] [app.api.routes] [req:12345678] ✅ TRAIN-Job erstellt: 5 für Modell 'Test'
```

### JSON-Format
```json
{
  "timestamp": "2024-12-23T21:00:00+00:00",
  "level": "INFO",
  "logger": "app.api.routes",
  "request_id": "12345678-1234-1234-1234-123456789abc",
  "message": "✅ TRAIN-Job erstellt: 5 für Modell 'Test'",
  "job_id": 5,
  "model_name": "Test",
  "model_type": "random_forest"
}
```

---

## 🚀 Konfiguration

### Environment Variables setzen

**Docker Compose:**
```yaml
environment:
  - LOG_LEVEL=DEBUG
  - LOG_FORMAT=json
  - LOG_JSON_INDENT=2
```

**Docker Run:**
```bash
docker run -e LOG_LEVEL=DEBUG -e LOG_FORMAT=json ...
```

**Lokal:**
```bash
export LOG_LEVEL=DEBUG
export LOG_FORMAT=json
python -m app.main
```

---

## ✅ Vorteile

1. **Strukturierte Logs:** Einfacheres Parsing und Analysieren
2. **Request-Tracing:** Jeder Request hat eine eindeutige ID
3. **Konfigurierbar:** Log-Level und Format über Environment Variables
4. **Context-Felder:** Zusätzliche Informationen in Logs
5. **JSON-Support:** Für Log-Aggregation-Tools (ELK, Splunk, etc.)

---

## 🧪 Testing

Nach Container-Rebuild:
1. Logs sollten strukturiert sein
2. Request-ID sollte in Logs und Response-Headers erscheinen
3. JSON-Format sollte funktionieren (wenn aktiviert)

---

## 📝 Nächste Schritte

- [ ] Weitere Module auf strukturiertes Logging umstellen
- [ ] Log-Rotation implementieren (optional)
- [ ] Log-Aggregation-Integration (optional)

