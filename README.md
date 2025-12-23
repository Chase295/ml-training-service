# 🤖 ML Training Service

Machine Learning Training Service für Kryptowährungs-Datenanalyse.

## 📋 Übersicht

Dieser Service ermöglicht das Training, Testen und Vergleichen von ML-Modellen (Random Forest, XGBoost) für Kryptowährungs-Daten aus der `coin_metrics` Tabelle.

## 🚀 Schnellstart

### Voraussetzungen
- Docker Desktop
- PostgreSQL Datenbank (extern oder via Docker)

### Installation

1. **Docker Container starten:**
   ```bash
   docker-compose up -d
   ```

2. **Service prüfen:**
   - FastAPI: http://localhost:8000
   - Streamlit UI: http://localhost:8501
   - API Docs: http://localhost:8000/docs

3. **Datenbank-Schema anwenden:**
   ```bash
   psql -h localhost -U postgres -d crypto_bot -f sql/schema.sql
   ```

## 📁 Projektstruktur

```
ml-training-service/
├── app/                    # Hauptanwendung
│   ├── api/                # REST API Endpoints
│   ├── database/           # Datenbank-Operationen
│   ├── queue/              # Job-Verarbeitung
│   ├── training/           # ML Training-Logik
│   ├── utils/              # Utilities
│   └── streamlit_app.py    # Streamlit UI
├── docs/                   # Dokumentation
├── tests/                  # Test-Dateien
├── sql/                    # SQL-Schema und Queries
├── models/                 # Gespeicherte ML-Modelle
├── docker-compose.yml      # Docker-Konfiguration
├── Dockerfile              # Docker-Image
└── requirements.txt        # Python-Abhängigkeiten
```

## 📚 Dokumentation

Alle Dokumentationen befinden sich im `docs/` Ordner:

- **[Modellerstellung](docs/MODELL_ERSTELLUNG_KOMPLETT_DOKUMENTATION.md)** - Vollständige Anleitung zur Modellerstellung
- **[Modell-Test & Vergleich](docs/MODELL_TEST_VERGLEICH_KOMPLETT_DOKUMENTATION.md)** - Anleitung zum Testen und Vergleichen
- **[Deployment](docs/DEPLOYMENT.md)** - Deployment-Anleitung
- **[Datenbank-Schema](docs/DATABASE_SCHEMA.md)** - Datenbank-Dokumentation

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
