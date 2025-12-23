# 🧪 Tests

Dieser Ordner enthält alle Test-Dateien für das ML Training Service Projekt.

## 📄 Verfügbare Tests

### End-to-End Tests
- **test_e2e.py** - End-to-End Tests für Random Forest Modelle
- **test_e2e_xgboost.py** - End-to-End Tests für XGBoost Modelle
- **test_phase8_e2e.py** - End-to-End Tests für Phase 8

### Phasen-spezifische Tests
- **test_phase2.py** - Tests für Phase 2
- **test_phase3.py** - Tests für Phase 3
- **test_phase4.py** - Tests für Phase 4
- **test_phase5.py** - Tests für Phase 5

## 🚀 Ausführung

```bash
# Alle Tests ausführen
python -m pytest tests/

# Spezifischen Test ausführen
python tests/test_e2e.py

# Mit Docker
docker-compose exec ml-training python tests/test_e2e.py
```

## 📝 Hinweise

- Tests sollten gegen eine laufende Instanz des ML Training Service ausgeführt werden
- Stelle sicher, dass die Datenbank korrekt konfiguriert ist
- Test-Daten sollten in der `coin_metrics` Tabelle vorhanden sein

