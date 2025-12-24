# 🤖 ML Training Service

Machine Learning Training Service für Kryptowährungs-Datenanalyse.

**Version:** 2.0  
**Status:** ✅ Produktionsreif  
**Stand:** 24. Dezember 2025

## 📋 Übersicht

Dieser Service ermöglicht das Training, Testen und Vergleichen von ML-Modellen (Random Forest, XGBoost) für Kryptowährungs-Daten aus der `coin_metrics` Tabelle.

### Hauptfunktionen

- ✅ **Modell-Training** (Random Forest, XGBoost)
- ✅ **Zeitbasierte Vorhersagen** (z.B. "Steigt in 5 Min um 30%")
- ✅ **Feature-Engineering** für bessere Performance
- ✅ **Modell-Testing** auf neuen Daten
- ✅ **Modell-Vergleich** um das beste zu finden
- ✅ **Asynchrone Job-Verarbeitung**
- ✅ **Web-UI** (Streamlit)
- ✅ **REST API** für Automatisierung
- ✅ **Prometheus Metriken** für Monitoring

## 🚀 Schnellstart

### Voraussetzungen
- Docker Desktop
- PostgreSQL Datenbank (extern)

### Installation

1. **Repository klonen:**
   ```bash
   git clone <repository-url>
   cd ml-training-service
   ```

2. **Docker Container starten:**
   ```bash
   docker-compose up -d
   ```

3. **Service prüfen:**
   - FastAPI: http://localhost:8000
   - Streamlit UI: http://localhost:8501
   - API Docs: http://localhost:8000/docs
   - Health Check: http://localhost:8000/health

4. **Datenbank-Schema anwenden:**
   ```bash
   psql -h <db-host> -U postgres -d crypto_bot -f sql/schema.sql
   ```

### Erste Schritte

1. **Web-UI öffnen:** http://localhost:8501
2. **Modell erstellen:** "Neues Modell erstellen" → Parameter eingeben
3. **Modell testen:** "Modell testen" → Zeitraum wählen
4. **Ergebnisse ansehen:** "Übersicht" → Modell auswählen

## 📁 Projektstruktur

```
ml-training-service/
├── app/                          # Hauptanwendung
│   ├── api/                      # REST API (FastAPI)
│   │   ├── routes.py             # API Endpoints
│   │   ├── schemas.py            # Pydantic Schemas
│   │   └── validators.py         # Validierungs-Logik
│   ├── database/                 # Datenbank-Operationen
│   │   ├── connection.py         # DB-Verbindung
│   │   ├── models.py             # DB-Interaktionen
│   │   └── utils.py              # JSONB-Helper
│   ├── queue/                    # Job-Verarbeitung
│   │   └── job_manager.py        # Job-Queue Manager
│   ├── training/                 # ML Training-Logik
│   │   ├── engine.py             # Training-Engine
│   │   ├── feature_engineering.py # Feature-Engineering
│   │   └── model_loader.py       # Modell-Laden/Testen
│   ├── utils/                    # Utilities
│   │   ├── config.py             # Konfiguration
│   │   ├── exceptions.py         # Custom Exceptions
│   │   ├── logging_config.py     # Logging-Setup
│   │   └── metrics.py            # Prometheus Metriken
│   ├── main.py                   # FastAPI App
│   └── streamlit_app.py          # Streamlit UI
├── docs/                         # Dokumentation
│   └── 00_GESAMT_DOKUMENTATION.md # ⭐ Start hier!
├── tests/                        # Test-Scripts
├── sql/                          # SQL-Schema und Queries
├── models/                       # Gespeicherte ML-Modelle (.pkl)
├── docker-compose.yml            # Docker-Konfiguration
├── Dockerfile                    # Docker-Image
└── requirements.txt              # Python-Abhängigkeiten
```

## 📚 Dokumentation

### ⭐ Start hier: [Gesamt-Dokumentation](docs/00_GESAMT_DOKUMENTATION.md)

**Wichtige Dokumentationen:**

- **[Gesamt-Dokumentation](docs/00_GESAMT_DOKUMENTATION.md)** - Vollständige Übersicht
- **[Modell-Erstellung](docs/MODELL_ERSTELLUNG_KOMPLETT_DOKUMENTATION.md)** - Detaillierte Anleitung
- **[Modell-Test & Vergleich](docs/MODELL_TEST_VERGLEICH_KOMPLETT_DOKUMENTATION.md)** - Testing-Anleitung
- **[Datenbank-Schema](docs/DATABASE_SCHEMA.md)** - Schema-Dokumentation
- **[API-Anleitung](docs/N8N_API_ANLEITUNG.md)** - API-Nutzung
- **[Deployment](docs/COOLIFY_DEPLOYMENT.md)** - Coolify Deployment
- **[Testbericht](docs/TESTBERICHT_VALIDIERUNG.md)** - Vollständiger Testbericht

**Vollständige Übersicht:** Siehe [docs/README.md](docs/README.md)

## 🧪 Tests

Tests befinden sich im `tests/` Ordner:

```bash
# End-to-End Tests ausführen
python tests/test_e2e.py
python tests/test_e2e_xgboost.py
```

## 🔧 Konfiguration

### Umgebungsvariablen

Die Datenbank-Verbindung wird in `app/database/connection.py` konfiguriert:

```python
DB_HOST = "10.0.128.18"
DB_PORT = 5432
DB_NAME = "crypto_bot"
DB_USER = "postgres"
DB_PASSWORD = "your_password"
```

## 📊 Features

- ✅ Modell-Training (Random Forest, XGBoost)
- ✅ Klassische Vorhersagen (Schwellwert-basiert)
- ✅ Zeitbasierte Vorhersagen (Steigt/Fällt in X Minuten um X%)
- ✅ Modell-Testing auf neuen Daten
- ✅ Modell-Vergleich (2 Modelle auf denselben Daten)
- ✅ Asynchrone Job-Verarbeitung
- ✅ Streamlit Web-UI
- ✅ REST API
- ✅ Prometheus Metriken

## 🛠️ Entwicklung

### Lokale Entwicklung

```bash
# Container neu bauen
docker-compose up -d --build

# Logs anzeigen
docker-compose logs -f

# In Container einsteigen
docker-compose exec ml-training bash
```

### Code-Struktur

- **API Routes:** `app/api/routes.py`
- **Schemas:** `app/api/schemas.py`
- **Database Models:** `app/database/models.py`
- **Training Engine:** `app/training/engine.py`
- **Feature Engineering:** `app/training/feature_engineering.py`
- **Job Manager:** `app/queue/job_manager.py`

## 📝 Lizenz

Proprietär

---

**Erstellt:** 2024  
**Version:** 1.0
