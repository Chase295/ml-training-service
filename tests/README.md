# 🧪 Test-Suite für KI-Modell-Erstellung

## Übersicht

Dieses Verzeichnis enthält automatisierte Tests für die KI-Modell-Erstellung.

## Test-Skripte

### `test_model_creation.py`

Automatisiertes Test-Skript, das folgende Tests durchführt:

1. **Health Check** - Prüft ob API erreichbar ist
2. **Data Availability** - Prüft ob Trainingsdaten verfügbar sind
3. **Phases** - Prüft ob Phasen geladen werden können
4. **Modell erstellen (minimal)** - Erstellt ein minimales Modell
5. **Modell erstellen (vollständig)** - Erstellt ein vollständiges Modell mit allen Features
6. **Job-Completion** - Wartet auf Training-Completion und prüft Ergebnisse
7. **Modell testen** - Testet ein trainiertes Modell

## Verwendung

### Voraussetzungen

- Docker Container läuft
- FastAPI erreichbar auf `http://localhost:8000`
- Python 3.11+ installiert
- `requests` Bibliothek installiert

### Installation

```bash
pip install requests
```

### Ausführung

```bash
# Einfache Ausführung
python tests/test_model_creation.py

# Mit detaillierter Ausgabe
python tests/test_model_creation.py --verbose
```

### Erwartete Ausgabe

```
ℹ️  ============================================================
ℹ️  Starte automatische Tests für KI-Modell-Erstellung
ℹ️  ============================================================
ℹ️  Test 1: Health Check
✅ Health Check erfolgreich
ℹ️  Test 2: Data Availability
✅ Data Availability OK: 12345 Samples
...
✅ 🎉 Alle Tests bestanden!
```

## Exit-Codes

- `0` - Alle Tests bestanden
- `1` - Mindestens ein Test fehlgeschlagen

## Integration in CI/CD

Das Skript kann in CI/CD-Pipelines integriert werden:

```yaml
# Beispiel: GitHub Actions
- name: Run Model Creation Tests
  run: |
    python tests/test_model_creation.py
```

## Manuelle Tests

Für manuelle Tests siehe: `docs/TESTPLAN_KI_MODELL_ERSTELLUNG.md`

## Fehlerbehebung

### "Connection refused"

- Prüfe ob Docker Container läuft: `docker ps`
- Prüfe ob FastAPI erreichbar ist: `curl http://localhost:8000/api/health`

### "No data available"

- Prüfe ob Daten in Datenbank vorhanden sind
- Prüfe `DB_DSN` Konfiguration

### "Job timeout"

- Training kann länger dauern
- Erhöhe `TIMEOUT` in `test_model_creation.py`
