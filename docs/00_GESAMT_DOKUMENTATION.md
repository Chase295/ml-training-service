# 📚 ML Training Service - Gesamt-Dokumentation

**Version:** 2.0  
**Stand:** 24. Dezember 2025  
**Status:** ✅ Produktionsreif

---

## 📋 Inhaltsverzeichnis

1. [Projekt-Übersicht](#projekt-übersicht)
2. [Schnellstart](#schnellstart)
3. [Architektur](#architektur)
4. [Funktionen](#funktionen)
5. [Dokumentations-Übersicht](#dokumentations-übersicht)
6. [API-Referenz](#api-referenz)
7. [Datenbank-Schema](#datenbank-schema)
8. [Deployment](#deployment)
9. [Testing & Validierung](#testing--validierung)
10. [Troubleshooting](#troubleshooting)

---

## 🎯 Projekt-Übersicht

### Was ist der ML Training Service?

Ein vollständiger Machine Learning Service für Kryptowährungs-Datenanalyse, der:

- ✅ **ML-Modelle trainiert** (Random Forest, XGBoost)
- ✅ **Modelle testet** auf neuen Daten
- ✅ **Modelle vergleicht** um das beste zu finden
- ✅ **Zeitbasierte Vorhersagen** macht (z.B. "Steigt in 5 Min um 30%")
- ✅ **Feature-Engineering** für bessere Performance
- ✅ **Asynchrone Job-Verarbeitung** für lange Trainings
- ✅ **Web-UI** für einfache Bedienung
- ✅ **REST API** für Automatisierung

### Technologie-Stack

- **Backend:** FastAPI (Python 3.11)
- **Frontend:** Streamlit
- **Datenbank:** PostgreSQL (extern)
- **ML-Frameworks:** Scikit-learn, XGBoost
- **Job-Queue:** Asyncio-basiert
- **Monitoring:** Prometheus Metriken
- **Deployment:** Docker, Coolify

---

## 🚀 Schnellstart

### Voraussetzungen

- Docker Desktop
- PostgreSQL Datenbank (extern)
- Python 3.11+ (für lokale Entwicklung)

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
2. **Modell erstellen:** "Neues Modell erstellen" → Parameter eingeben → "Modell erstellen"
3. **Modell testen:** "Modell testen" → Zeitraum wählen → "Test starten"
4. **Ergebnisse ansehen:** "Übersicht" → Modell auswählen → "Details anzeigen"

---

## 🏗️ Architektur

### Projekt-Struktur

```
ml-training-service/
├── app/                          # Hauptanwendung
│   ├── api/                      # REST API
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
│   │   ├── engine.py              # Training-Engine
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
├── tests/                         # Test-Scripts
├── sql/                          # SQL-Schema und Queries
├── models/                       # Gespeicherte ML-Modelle (.pkl)
├── docker-compose.yml            # Docker-Konfiguration
├── Dockerfile                    # Docker-Image
└── requirements.txt              # Python-Abhängigkeiten
```

### Datenfluss

```
API Request (Web UI / REST API)
    ↓
FastAPI Routes (app/api/routes.py)
    ↓
Job erstellen (ml_jobs Tabelle)
    ↓
Job Queue (app/queue/job_manager.py)
    ↓
Training Engine (app/training/engine.py)
    ↓
Datenbank (PostgreSQL)
    ↓
Ergebnis speichern (ml_models, ml_test_results, ml_comparisons)
```

### Komponenten

#### 1. REST API (FastAPI)
- **Datei:** `app/api/routes.py`
- **Funktion:** REST Endpoints für Modell-Erstellung, Testing, Vergleich
- **Port:** 8000

#### 2. Web-UI (Streamlit)
- **Datei:** `app/streamlit_app.py`
- **Funktion:** Benutzerfreundliche Oberfläche
- **Port:** 8501

#### 3. Job Queue
- **Datei:** `app/queue/job_manager.py`
- **Funktion:** Asynchrone Verarbeitung von Trainings/Test-Jobs
- **Status:** PENDING → RUNNING → COMPLETED/FAILED

#### 4. Training Engine
- **Datei:** `app/training/engine.py`
- **Funktion:** ML-Modell-Training (Random Forest, XGBoost)
- **Features:** Feature-Engineering, SMOTE, TimeSeriesSplit

#### 5. Feature Engineering
- **Datei:** `app/training/feature_engineering.py`
- **Funktion:** Erstellt zusätzliche Features (Momentum, Volatilität, etc.)

---

## ⚙️ Funktionen

### 1. Modell-Erstellung

**Unterstützte Modell-Typen:**
- Random Forest
- XGBoost

**Vorhersage-Typen:**
- **Klassisch:** Schwellwert-basiert (z.B. `market_cap_close > 10000`)
- **Zeitbasiert:** Zukünftige Preisänderung (z.B. "Steigt in 5 Min um 30%")

**Features:**
- ✅ Feature-Engineering (optional)
- ✅ SMOTE (Oversampling, optional)
- ✅ TimeSeriesSplit (Cross-Validation, optional)
- ✅ Hyperparameter-Tuning

**Dokumentation:** [MODELL_ERSTELLUNG_KOMPLETT_DOKUMENTATION.md](MODELL_ERSTELLUNG_KOMPLETT_DOKUMENTATION.md)

### 2. Modell-Testing

**Funktionen:**
- Test auf neuen Daten
- Train vs. Test Vergleich
- Overlap-Prüfung
- Umfassende Metriken

**Metriken:**
- Accuracy, F1-Score, Precision, Recall
- ROC-AUC, MCC, FPR, FNR
- Confusion Matrix
- Simulated Profit

**Dokumentation:** [MODELL_TEST_VERGLEICH_KOMPLETT_DOKUMENTATION.md](MODELL_TEST_VERGLEICH_KOMPLETT_DOKUMENTATION.md)

### 3. Modell-Vergleich

**Funktionen:**
- Zwei Modelle auf denselben Daten testen
- Automatische Gewinner-Bestimmung
- Detaillierter Vergleich

**Dokumentation:** [MODELL_TEST_VERGLEICH_KOMPLETT_DOKUMENTATION.md](MODELL_TEST_VERGLEICH_KOMPLETT_DOKUMENTATION.md)

### 4. Feature-Engineering

**Erstellte Features:**
- Price Momentum (ROC, Change)
- Volatility (Preis-Schwankungen)
- Market Cap Velocity
- Price Range
- Volume Patterns

**Dokumentation:** [VALIDIERUNG_FEATURE_ENGINEERING.md](VALIDIERUNG_FEATURE_ENGINEERING.md)

---

## 📚 Dokumentations-Übersicht

### Kern-Dokumentationen

| Dokument | Beschreibung | Status |
|----------|--------------|--------|
| **[COMPLETE_WORKFLOW_DOKUMENTATION.md](COMPLETE_WORKFLOW_DOKUMENTATION.md)** | Vollständiger Workflow von Erstellung bis Testing | ✅ Aktuell |
| **[MODELL_ERSTELLUNG_KOMPLETT_DOKUMENTATION.md](MODELL_ERSTELLUNG_KOMPLETT_DOKUMENTATION.md)** | Detaillierte Modell-Erstellung | ✅ Aktuell |
| **[MODELL_TEST_VERGLEICH_KOMPLETT_DOKUMENTATION.md](MODELL_TEST_VERGLEICH_KOMPLETT_DOKUMENTATION.md)** | Testing und Vergleich | ✅ Aktuell |
| **[DATABASE_SCHEMA.md](DATABASE_SCHEMA.md)** | Datenbank-Schema | ✅ Aktuell |

### API & Integration

| Dokument | Beschreibung | Status |
|----------|--------------|--------|
| **[N8N_API_ANLEITUNG.md](N8N_API_ANLEITUNG.md)** | API-Nutzung mit n8n | ✅ Aktuell |
| **[API_BASE_URL_ERKLAERUNG.md](API_BASE_URL_ERKLAERUNG.md)** | API-Konfiguration | ✅ Aktuell |

### Deployment

| Dokument | Beschreibung | Status |
|----------|--------------|--------|
| **[COOLIFY_DEPLOYMENT.md](COOLIFY_DEPLOYMENT.md)** | Deployment auf Coolify | ✅ Aktuell |
| **[COOLIFY_QUICK_START.md](COOLIFY_QUICK_START.md)** | Schnellstart Coolify | ✅ Aktuell |
| **[DEPLOYMENT.md](DEPLOYMENT.md)** | Allgemeine Deployment-Anleitung | ✅ Aktuell |

### Testing & Validierung

| Dokument | Beschreibung | Status |
|----------|--------------|--------|
| **[TESTBERICHT_VALIDIERUNG.md](TESTBERICHT_VALIDIERUNG.md)** | Vollständiger Testbericht | ✅ Aktuell |
| **[VALIDIERUNG_FEATURE_ENGINEERING.md](VALIDIERUNG_FEATURE_ENGINEERING.md)** | Feature-Engineering Validierung | ✅ Aktuell |
| **[ERKLAERUNG_F1_SCORE_PROBLEM.md](ERKLAERUNG_F1_SCORE_PROBLEM.md)** | Erklärung F1-Score = 0 | ✅ Aktuell |
| **[VERGLEICH_MODell_1_VS_3.md](VERGLEICH_MODell_1_VS_3.md)** | Modell-Vergleich | ✅ Aktuell |

### Monitoring

| Dokument | Beschreibung | Status |
|----------|--------------|--------|
| **[GRAFANA_DASHBOARD_ANLEITUNG.md](GRAFANA_DASHBOARD_ANLEITUNG.md)** | Grafana Setup | ✅ Aktuell |

### Historische Dokumentationen

Diese Dokumentationen sind für Referenzzwecke behalten, aber möglicherweise veraltet:

- `PHASE1_*.md` - Phase 1 Implementierung
- `PHASE2_*.md` - Phase 2 Implementierung
- `TEST_PHASE*.md` - Test-Ergebnisse
- `STATUS_UEBERSICHT.md` - Alte Status-Übersicht

---

## 🔌 API-Referenz

### Basis-URL

- **Lokal:** `http://localhost:8000/api`
- **Produktion:** `http://100.76.209.59:8005/api`

### Wichtige Endpoints

#### Modell-Erstellung
```
POST /api/models/create
```

#### Modell-Testing
```
POST /api/models/{model_id}/test
```

#### Modell-Vergleich
```
POST /api/models/compare
```

#### Modell-Liste
```
GET /api/models
```

#### Job-Status
```
GET /api/queue/{job_id}
```

**Vollständige API-Dokumentation:** http://localhost:8000/docs (Swagger UI)

**Detaillierte Anleitung:** [N8N_API_ANLEITUNG.md](N8N_API_ANLEITUNG.md)

---

## 🗄️ Datenbank-Schema

### Haupt-Tabellen

#### `ml_models`
Speichert alle trainierten Modelle.

**Wichtige Felder:**
- `id`, `name`, `model_type`
- `train_start`, `train_end`
- `features` (JSONB)
- `params` (JSONB)
- `training_accuracy`, `training_f1`
- `confusion_matrix` (JSONB)
- `feature_importance` (JSONB)

#### `ml_test_results`
Speichert alle Test-Ergebnisse.

**Wichtige Felder:**
- `id`, `model_id`
- `test_start`, `test_end`
- `accuracy`, `f1_score`, `precision_score`, `recall`
- `confusion_matrix` (JSONB)
- `train_accuracy`, `accuracy_degradation`
- `is_overfitted`

#### `ml_comparisons`
Speichert Modell-Vergleiche.

**Wichtige Felder:**
- `id`, `model_a_id`, `model_b_id`
- `test_a_id`, `test_b_id` (Foreign Keys zu ml_test_results)
- `winner_id`
- `test_start`, `test_end`

#### `ml_jobs`
Speichert alle Jobs (Training, Testing, Vergleich).

**Wichtige Felder:**
- `id`, `job_type` (TRAIN, TEST, COMPARE)
- `status` (PENDING, RUNNING, COMPLETED, FAILED)
- `progress` (0.0 - 1.0)
- `result_model_id`, `result_test_id`, `result_comparison_id`

**Vollständiges Schema:** [DATABASE_SCHEMA.md](DATABASE_SCHEMA.md)  
**SQL-Schema:** `sql/schema.sql`

---

## 🚀 Deployment

### Lokal (Docker)

```bash
docker-compose up -d
```

### Coolify

**Anleitung:** [COOLIFY_DEPLOYMENT.md](COOLIFY_DEPLOYMENT.md)

**Schnellstart:** [COOLIFY_QUICK_START.md](COOLIFY_QUICK_START.md)

### Umgebungsvariablen

```bash
DB_HOST=your-db-host
DB_PORT=5432
DB_NAME=crypto_bot
DB_USER=postgres
DB_PASSWORD=your-password
```

---

## 🧪 Testing & Validierung

### Automatisierte Tests

**Validierungs-Scripts:**
- `tests/comprehensive_validation.py` - Systemweite Validierung
- `tests/validate_model_test_results.py` - Einzelne Validierung
- `tests/final_live_test.py` - Live-Test

**Ausführung:**
```bash
API_BASE_URL="http://localhost:8000/api" python3 tests/comprehensive_validation.py
```

### Testbericht

**Vollständiger Testbericht:** [TESTBERICHT_VALIDIERUNG.md](TESTBERICHT_VALIDIERUNG.md)

**Ergebnis:** ✅ Alle Tests bestanden, System zu 100% validiert

---

## 🔧 Troubleshooting

### Häufige Probleme

#### 1. F1-Score = 0

**Problem:** Modell macht keine positiven Vorhersagen.

**Lösung:** 
- Feature-Engineering aktivieren
- Mehr Features verwenden
- SMOTE aktivieren
- Weniger strenge Anforderungen

**Dokumentation:** [ERKLAERUNG_F1_SCORE_PROBLEM.md](ERKLAERUNG_F1_SCORE_PROBLEM.md)

#### 2. Job bleibt bei PENDING

**Problem:** Job wird nicht verarbeitet.

**Lösung:**
- Job Worker prüfen: `docker-compose logs ml-training`
- Datenbank-Verbindung prüfen
- Ressourcen prüfen (CPU, RAM)

#### 3. Feature-Engineering funktioniert nicht

**Problem:** Keine Feature-Engineering Features erstellt.

**Lösung:**
- Parameter prüfen: `use_engineered_features: true`
- Logs prüfen: Sollte "Erstelle Pump-Detection Features" zeigen

**Dokumentation:** [VALIDIERUNG_FEATURE_ENGINEERING.md](VALIDIERUNG_FEATURE_ENGINEERING.md)

#### 4. Datenbank-Verbindungsfehler

**Problem:** Kann nicht zur Datenbank verbinden.

**Lösung:**
- DB-Host, Port, Credentials prüfen
- Firewall-Regeln prüfen
- Datenbank erreichbar?

---

## 📊 Monitoring

### Prometheus Metriken

**Endpoint:** `http://localhost:8000/metrics`

**Wichtige Metriken:**
- `ml_active_jobs` - Anzahl aktiver Jobs
- `ml_job_progress_percent` - Job-Fortschritt
- `ml_models_total` - Anzahl Modelle
- `ml_service_uptime_seconds` - Uptime

### Grafana Dashboard

**Anleitung:** [GRAFANA_DASHBOARD_ANLEITUNG.md](GRAFANA_DASHBOARD_ANLEITUNG.md)

**Dashboard JSON:** `docs/grafana_dashboard.json`

---

## 🔄 Workflow-Beispiel

### 1. Modell erstellen

1. Web-UI öffnen: http://localhost:8501
2. "Neues Modell erstellen"
3. Parameter eingeben:
   - Modell-Typ: XGBoost
   - Features: price_open, price_high, price_low, price_close
   - Feature-Engineering: ✅ Aktiviert
   - Zeitbasierte Vorhersage: 5 Min, 30%
4. "Modell erstellen" klicken
5. Warten bis Job abgeschlossen

### 2. Modell testen

1. "Modell testen"
2. Modell auswählen
3. Test-Zeitraum wählen
4. "Test starten"
5. Ergebnisse ansehen

### 3. Modelle vergleichen

1. "Modelle vergleichen"
2. Zwei Modelle auswählen
3. Test-Zeitraum wählen
4. "Vergleich starten"
5. Gewinner ansehen

---

## 📝 Changelog

### Version 2.0 (24. Dezember 2025)

- ✅ Schema-Verbesserungen (ml_test_results, ml_comparisons)
- ✅ Feature-Engineering implementiert
- ✅ SMOTE (Oversampling) implementiert
- ✅ TimeSeriesSplit implementiert
- ✅ Zusätzliche Metriken (MCC, FPR, FNR, ROC-AUC)
- ✅ Train vs. Test Vergleich
- ✅ Umfassende Validierung
- ✅ Grafana Dashboard
- ✅ Coolify Deployment

### Version 1.0 (Initial)

- ✅ Basis-Funktionalität
- ✅ Modell-Erstellung
- ✅ Modell-Testing
- ✅ Modell-Vergleich

---

## 📞 Support

Bei Fragen oder Problemen:

1. **Dokumentation prüfen:** Siehe [Dokumentations-Übersicht](#dokumentations-übersicht)
2. **Logs prüfen:** `docker-compose logs ml-training`
3. **API testen:** http://localhost:8000/docs

---

**Letzte Aktualisierung:** 24. Dezember 2025  
**Version:** 2.0  
**Status:** ✅ Produktionsreif

