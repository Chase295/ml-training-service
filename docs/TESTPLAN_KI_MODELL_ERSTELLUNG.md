# 🧪 Vollständiger Testplan: KI-Modell-Erstellung

## 📋 Übersicht

Dieser Testplan stellt sicher, dass **alle Funktionen** der KI-Modell-Erstellung zu **100%** funktionieren. Er deckt alle Komponenten ab: UI, API, Datenbank, Feature-Engineering, Training, Testing und Evaluation.

---

## 🎯 Test-Phasen

### Phase 1: Vorbereitung & Umgebung
### Phase 2: Daten-Validierung
### Phase 3: UI-Tests (Streamlit)
### Phase 4: API-Tests (FastAPI)
### Phase 5: Feature-Engineering Tests
### Phase 6: Training-Pipeline Tests
### Phase 7: Modell-Evaluation Tests
### Phase 8: Integration-Tests (End-to-End)
### Phase 9: Edge Cases & Fehlerbehandlung
### Phase 10: Performance-Tests

---

## 📝 Phase 1: Vorbereitung & Umgebung

### 1.1 Umgebung prüfen

**Ziel:** Sicherstellen, dass alle Services laufen und erreichbar sind.

**Tests:**
```bash
# 1. Docker Container Status
docker ps | grep ml-training-service
# Erwartet: Container läuft, Status "Up"

# 2. FastAPI Health Check
curl http://localhost:8000/api/health
# Erwartet: {"status": "healthy", ...}

# 3. Streamlit UI erreichbar
curl http://localhost:8501
# Erwartet: HTTP 200

# 4. Datenbank-Verbindung
docker exec ml-training-service python -c "from app.database.connection import get_db; next(get_db())"
# Erwartet: Keine Fehler
```

**Checkliste:**
- [ ] Docker Container läuft
- [ ] FastAPI erreichbar (Port 8000)
- [ ] Streamlit UI erreichbar (Port 8501)
- [ ] Datenbank-Verbindung funktioniert
- [ ] Alle Environment Variables gesetzt

### 1.2 Datenbank-Schema prüfen

**Ziel:** Sicherstellen, dass alle benötigten Tabellen existieren.

**SQL-Tests:**
```sql
-- Prüfe alle benötigten Tabellen
SELECT table_name 
FROM information_schema.tables 
WHERE table_schema = 'public' 
AND table_name IN (
    'ml_models',
    'ml_jobs',
    'ml_test_results',
    'ml_model_comparisons',
    'coin_metrics',
    'exchange_rates',
    'coin_phases'
);

-- Prüfe Spalten in ml_models
SELECT column_name, data_type 
FROM information_schema.columns 
WHERE table_name = 'ml_models'
AND column_name IN (
    'rug_detection_metrics',
    'market_context_enabled'
);
```

**Checkliste:**
- [ ] Alle Tabellen existieren
- [ ] `ml_models` hat `rug_detection_metrics` (JSONB)
- [ ] `ml_models` hat `market_context_enabled` (BOOLEAN)
- [ ] `ml_test_results` hat `rug_detection_metrics` (JSONB)
- [ ] Indizes existieren

### 1.3 Testdaten prüfen

**Ziel:** Sicherstellen, dass ausreichend Trainingsdaten vorhanden sind.

**Tests:**
```python
# Prüfe Daten-Verfügbarkeit
import requests
response = requests.get("http://localhost:8000/api/data-availability")
data = response.json()

# Erwartete Werte:
assert data['min_timestamp'] is not None
assert data['max_timestamp'] is not None
assert data['total_samples'] > 1000  # Mindestens 1000 Samples
```

**Checkliste:**
- [ ] Mindestens 1000 Samples in `coin_metrics`
- [ ] Zeitbereich mindestens 7 Tage
- [ ] Alle kritischen Features vorhanden (`dev_sold_amount`, etc.)
- [ ] `exchange_rates` Daten vorhanden (wenn Market Context verwendet wird)

---

## 📝 Phase 2: Daten-Validierung

### 2.1 Feature-Daten prüfen

**Ziel:** Sicherstellen, dass alle Features korrekt geladen werden.

**Tests:**
```python
# Test: Feature-Loading
from app.training.feature_engineering import load_training_data
import asyncio

async def test_feature_loading():
    df = await load_training_data(
        train_start="2024-01-01T00:00:00Z",
        train_end="2024-01-07T23:59:59Z",
        phases=None
    )
    
    # Prüfe kritische Features
    required_features = [
        'price_open', 'price_high', 'price_low', 'price_close',
        'volume_sol', 'dev_sold_amount', 'buy_pressure_ratio',
        'unique_signer_ratio', 'whale_buy_volume_sol', 'volatility_pct'
    ]
    
    for feature in required_features:
        assert feature in df.columns, f"Feature {feature} fehlt!"
        assert df[feature].notna().sum() > 0, f"Feature {feature} hat nur NaN-Werte!"
    
    print("✅ Alle Features korrekt geladen")

asyncio.run(test_feature_loading())
```

**Checkliste:**
- [ ] Alle Basis-Features vorhanden (OHLC, Volumen)
- [ ] Alle neuen Features vorhanden (`dev_sold_amount`, etc.)
- [ ] Keine unerwarteten NaN-Werte in kritischen Features
- [ ] Daten-Typen korrekt (float, int, etc.)

### 2.2 Market Context Daten prüfen

**Ziel:** Sicherstellen, dass Exchange Rates korrekt geladen werden.

**Tests:**
```python
# Test: Market Context Loading
from app.training.feature_engineering import enrich_with_market_context
import pandas as pd

async def test_market_context():
    # Erstelle Test-Daten
    df = pd.DataFrame({
        'created_at': pd.date_range('2024-01-01', periods=100, freq='5min'),
        'price_close': [100] * 100
    })
    
    # Teste Enrichment
    enriched_df = await enrich_with_market_context(df)
    
    # Prüfe neue Features
    assert 'sol_price_usd' in enriched_df.columns
    assert 'sol_price_change_pct' in enriched_df.columns
    assert 'sol_price_ma_5' in enriched_df.columns
    assert 'sol_price_volatility' in enriched_df.columns
    
    print("✅ Market Context korrekt geladen")

asyncio.run(test_market_context())
```

**Checkliste:**
- [ ] `sol_price_usd` vorhanden
- [ ] `sol_price_change_pct` berechnet
- [ ] `sol_price_ma_5` berechnet
- [ ] `sol_price_volatility` berechnet
- [ ] Keine NaN-Werte in Market Context Features

---

## 📝 Phase 3: UI-Tests (Streamlit)

### 3.1 Basis-Formular Tests

**Ziel:** Sicherstellen, dass alle Formular-Felder funktionieren.

**Manuelle Tests:**
1. **Modell-Name:**
   - [ ] Text-Input akzeptiert Text
   - [ ] Leerer Name zeigt Fehler
   - [ ] Spezielle Zeichen werden akzeptiert

2. **Modell-Typ:**
   - [ ] Dropdown zeigt "random_forest" und "xgboost"
   - [ ] Standard ist "random_forest"
   - [ ] Auswahl wird gespeichert

3. **Training-Zeitraum:**
   - [ ] Start-Datum kann ausgewählt werden
   - [ ] Ende-Datum kann ausgewählt werden
   - [ ] Ende-Datum muss nach Start-Datum sein
   - [ ] Uhrzeit kann ausgewählt werden
   - [ ] Validierung: Start < Ende

4. **Vorhersage-Ziel:**
   - [ ] Variable kann ausgewählt werden
   - [ ] Zeitraum (Minuten) kann eingegeben werden (1-60)
   - [ ] Mindest-Änderung (%) kann eingegeben werden
   - [ ] Richtung (Steigt/Fällt) kann ausgewählt werden

**Checkliste:**
- [ ] Alle Felder funktionieren
- [ ] Validierung funktioniert
- [ ] Fehlermeldungen werden angezeigt

### 3.2 Erweiterte Optionen Tests

**Ziel:** Sicherstellen, dass alle erweiterten Optionen funktionieren.

**Manuelle Tests:**
1. **Feature-Auswahl:**
   - [ ] Expander kann geöffnet/geschlossen werden
   - [ ] Alle Feature-Tabs sind sichtbar
   - [ ] Checkboxen funktionieren in allen Tabs
   - [ ] Features werden korrekt gesammelt (auch beim Tab-Wechsel)
   - [ ] Kritische Features sind markiert
   - [ ] Zusammenfassung zeigt korrekte Anzahl

2. **Phasen-Filter:**
   - [ ] Phasen werden geladen
   - [ ] Multiselect funktioniert
   - [ ] Leere Auswahl = alle Phasen

3. **Hyperparameter:**
   - [ ] Checkbox "Hyperparameter anpassen" funktioniert
   - [ ] Random Forest: n_estimators, max_depth
   - [ ] XGBoost: n_estimators, max_depth, learning_rate
   - [ ] Werte werden korrekt übernommen

4. **Feature-Engineering:**
   - [ ] Checkbox funktioniert
   - [ ] Standard: aktiviert

5. **Marktstimmung:**
   - [ ] Checkbox funktioniert
   - [ ] Standard: deaktiviert

6. **Daten-Handling:**
   - [ ] SMOTE Checkbox funktioniert
   - [ ] TimeSeriesSplit Checkbox funktioniert
   - [ ] Anzahl Splits kann eingegeben werden (nur wenn TimeSeriesSplit aktiv)

**Checkliste:**
- [ ] Alle erweiterten Optionen funktionieren
- [ ] Feature-Auswahl funktioniert stabil (auch bei Tab-Wechsel)
- [ ] Alle Werte werden korrekt übernommen

### 3.3 Form-Submission Tests

**Ziel:** Sicherstellen, dass das Formular korrekt abgesendet wird.

**Manuelle Tests:**
1. **Minimales Modell (Standard):**
   - [ ] Fülle nur Pflichtfelder aus
   - [ ] Klicke "Modell trainieren"
   - [ ] Erfolgsmeldung wird angezeigt
   - [ ] Job-ID wird angezeigt
   - [ ] Weiterleitung zu Jobs funktioniert

2. **Vollständiges Modell (Alle Optionen):**
   - [ ] Fülle alle Felder aus
   - [ ] Aktiviere alle erweiterten Optionen
   - [ ] Wähle spezifische Features
   - [ ] Klicke "Modell trainieren"
   - [ ] Erfolgsmeldung wird angezeigt
   - [ ] Job wird erstellt

3. **Fehlerbehandlung:**
   - [ ] Leerer Modell-Name → Fehler
   - [ ] Start >= Ende → Fehler
   - [ ] Keine Features ausgewählt → Fallback auf alle Features

**Checkliste:**
- [ ] Form-Submission funktioniert
- [ ] Erfolgsmeldungen werden angezeigt
- [ ] Fehlermeldungen werden angezeigt
- [ ] Job wird korrekt erstellt

---

## 📝 Phase 4: API-Tests (FastAPI)

### 4.1 API-Endpunkte prüfen

**Ziel:** Sicherstellen, dass alle API-Endpunkte funktionieren.

**Tests:**
```python
import requests
import json

BASE_URL = "http://localhost:8000"

# Test 1: Health Check
def test_health():
    response = requests.get(f"{BASE_URL}/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data['status'] == 'healthy'
    print("✅ Health Check funktioniert")

# Test 2: Data Availability
def test_data_availability():
    response = requests.get(f"{BASE_URL}/api/data-availability")
    assert response.status_code == 200
    data = response.json()
    assert 'min_timestamp' in data
    assert 'max_timestamp' in data
    print("✅ Data Availability funktioniert")

# Test 3: Phases
def test_phases():
    response = requests.get(f"{BASE_URL}/api/phases")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    print("✅ Phases funktioniert")

# Test 4: Models List
def test_models_list():
    response = requests.get(f"{BASE_URL}/api/models")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    print("✅ Models List funktioniert")
```

**Checkliste:**
- [ ] `/api/health` funktioniert
- [ ] `/api/data-availability` funktioniert
- [ ] `/api/phases` funktioniert
- [ ] `/api/models` funktioniert

### 4.2 Modell-Erstellung API Test

**Ziel:** Sicherstellen, dass Modell-Erstellung über API funktioniert.

**Tests:**
```python
# Test: Modell erstellen (minimal)
def test_create_model_minimal():
    payload = {
        "name": "TEST_MODEL_MINIMAL",
        "model_type": "random_forest",
        "target_var": "price_close",
        "operator": None,
        "target_value": None,
        "features": ["price_open", "price_close", "volume_sol"],
        "phases": None,
        "params": None,
        "train_start": "2024-01-01T00:00:00Z",
        "train_end": "2024-01-07T23:59:59Z",
        "use_time_based_prediction": True,
        "future_minutes": 10,
        "min_percent_change": 5.0,
        "direction": "up",
        "use_engineered_features": True,
        "feature_engineering_windows": [5, 10, 15],
        "use_smote": True,
        "use_timeseries_split": True,
        "cv_splits": 5,
        "use_market_context": False
    }
    
    response = requests.post(f"{BASE_URL}/api/models/create", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert 'job_id' in data
    assert 'status' in data
    print(f"✅ Modell erstellt: Job-ID {data['job_id']}")
    return data['job_id']

# Test: Modell erstellen (vollständig)
def test_create_model_full():
    payload = {
        "name": "TEST_MODEL_FULL",
        "model_type": "xgboost",
        "target_var": "price_close",
        "operator": None,
        "target_value": None,
        "features": [
            "price_open", "price_high", "price_low", "price_close",
            "volume_sol", "dev_sold_amount", "buy_pressure_ratio"
        ],
        "phases": [1, 2],
        "params": {
            "n_estimators": 200,
            "max_depth": 8,
            "learning_rate": 0.15
        },
        "train_start": "2024-01-01T00:00:00Z",
        "train_end": "2024-01-07T23:59:59Z",
        "use_time_based_prediction": True,
        "future_minutes": 15,
        "min_percent_change": 7.5,
        "direction": "up",
        "use_engineered_features": True,
        "feature_engineering_windows": [5, 10, 15],
        "use_smote": True,
        "use_timeseries_split": True,
        "cv_splits": 5,
        "use_market_context": True
    }
    
    response = requests.post(f"{BASE_URL}/api/models/create", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert 'job_id' in data
    print(f"✅ Vollständiges Modell erstellt: Job-ID {data['job_id']}")
    return data['job_id']
```

**Checkliste:**
- [ ] Minimales Modell kann erstellt werden
- [ ] Vollständiges Modell kann erstellt werden
- [ ] Job wird in Datenbank gespeichert
- [ ] Job-Status ist "PENDING" oder "RUNNING"

---

## 📝 Phase 5: Feature-Engineering Tests

### 5.1 Feature-Erstellung Tests

**Ziel:** Sicherstellen, dass alle Features korrekt erstellt werden.

**Tests:**
```python
from app.training.feature_engineering import create_pump_detection_features
import pandas as pd
import numpy as np

def test_feature_engineering():
    # Erstelle Test-Daten
    df = pd.DataFrame({
        'price_open': [100, 105, 110, 115, 120],
        'price_high': [102, 107, 112, 117, 122],
        'price_low': [99, 104, 109, 114, 119],
        'price_close': [101, 106, 111, 116, 121],
        'volume_sol': [1000, 1500, 2000, 2500, 3000],
        'buy_volume_sol': [600, 900, 1200, 1500, 1800],
        'sell_volume_sol': [400, 600, 800, 1000, 1200],
        'dev_sold_amount': [0, 100, 0, 200, 0],
        'buy_pressure_ratio': [0.6, 0.6, 0.6, 0.6, 0.6],
        'unique_signer_ratio': [0.8, 0.75, 0.7, 0.65, 0.6],
        'whale_buy_volume_sol': [100, 150, 200, 250, 300],
        'whale_sell_volume_sol': [50, 75, 100, 125, 150],
        'volatility_pct': [2.0, 2.5, 3.0, 3.5, 4.0],
        'net_volume_sol': [200, 300, 400, 500, 600],
        'created_at': pd.date_range('2024-01-01', periods=5, freq='5min')
    })
    
    # Teste Feature-Engineering
    features_df = create_pump_detection_features(df, windows=[5, 10])
    
    # Prüfe erwartete Features
    expected_features = [
        'price_momentum_5', 'volume_momentum_5',
        'price_volatility_5', 'volume_volatility_5',
        'dev_sold_ratio', 'whale_activity_ratio',
        'buy_sell_imbalance', 'net_volume_trend'
    ]
    
    for feature in expected_features:
        assert feature in features_df.columns, f"Feature {feature} fehlt!"
    
    print("✅ Feature-Engineering funktioniert")
    return features_df
```

**Checkliste:**
- [ ] Alle Basis-Features werden erstellt
- [ ] Momentum-Features werden erstellt
- [ ] Volatilitäts-Features werden erstellt
- [ ] Dev-Tracking-Features werden erstellt
- [ ] Whale-Aktivitäts-Features werden erstellt
- [ ] Keine NaN-Werte in kritischen Features

### 5.2 Feature-Validierung Tests

**Ziel:** Sicherstellen, dass kritische Features validiert werden.

**Tests:**
```python
from app.training.engine import validate_critical_features

def test_feature_validation():
    # Test 1: Alle kritischen Features vorhanden
    features = [
        'price_close', 'volume_sol', 'dev_sold_amount',
        'buy_pressure_ratio', 'unique_signer_ratio'
    ]
    validate_critical_features(features)
    print("✅ Validierung: Alle Features vorhanden")
    
    # Test 2: Fehlende kritische Features
    features_missing = ['price_close', 'volume_sol']
    # Sollte Warnung ausgeben, aber nicht abbrechen
    validate_critical_features(features_missing)
    print("✅ Validierung: Warnung bei fehlenden Features")
```

**Checkliste:**
- [ ] Validierung funktioniert
- [ ] Warnungen werden ausgegeben
- [ ] Training wird nicht abgebrochen (nur Warnung)

---

## 📝 Phase 6: Training-Pipeline Tests

### 6.1 Training-Workflow Test

**Ziel:** Sicherstellen, dass der gesamte Training-Workflow funktioniert.

**Tests:**
```python
from app.training.engine import train_model
import asyncio

async def test_training_workflow():
    # Test-Parameter
    train_params = {
        "name": "TEST_TRAINING",
        "model_type": "random_forest",
        "target_var": "price_close",
        "features": ["price_open", "price_close", "volume_sol"],
        "train_start": "2024-01-01T00:00:00Z",
        "train_end": "2024-01-07T23:59:59Z",
        "use_time_based_prediction": True,
        "future_minutes": 10,
        "min_percent_change": 5.0,
        "direction": "up",
        "use_engineered_features": False,
        "use_smote": False,
        "use_timeseries_split": False,
        "use_market_context": False
    }
    
    # Starte Training
    result = await train_model(train_params)
    
    # Prüfe Ergebnis
    assert result is not None
    assert 'model_id' in result
    assert 'accuracy' in result
    assert 'f1_score' in result
    assert result['accuracy'] > 0
    assert result['f1_score'] > 0
    
    print(f"✅ Training erfolgreich: Accuracy={result['accuracy']:.4f}")
    return result

# Führe Test aus
result = asyncio.run(test_training_workflow())
```

**Checkliste:**
- [ ] Training startet erfolgreich
- [ ] Daten werden geladen
- [ ] Features werden erstellt
- [ ] Modell wird trainiert
- [ ] Metriken werden berechnet
- [ ] Modell wird gespeichert
- [ ] Modell wird in Datenbank gespeichert

### 6.2 Rug-Detection Metriken Test

**Ziel:** Sicherstellen, dass Rug-Detection-Metriken korrekt berechnet werden.

**Tests:**
```python
from app.training.engine import calculate_rug_detection_metrics
import numpy as np

def test_rug_detection_metrics():
    # Erstelle Test-Daten
    y_true = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 0, 0, 1, 0, 1, 0, 1])
    
    # Simuliere dev_sold_amount (für Dev-Sold Detection)
    dev_sold_amount = np.array([0, 1000, 0, 500, 0, 2000, 0, 0, 0, 1500])
    
    # Berechne Metriken
    metrics = calculate_rug_detection_metrics(
        y_true, y_pred, dev_sold_amount
    )
    
    # Prüfe Metriken
    assert 'dev_sold_detection_rate' in metrics
    assert 'wash_trading_detection_rate' in metrics
    assert 'weighted_cost' in metrics
    assert 'precision_at_k' in metrics
    
    assert 0 <= metrics['dev_sold_detection_rate'] <= 1
    assert 0 <= metrics['wash_trading_detection_rate'] <= 1
    assert metrics['weighted_cost'] >= 0
    
    print("✅ Rug-Detection Metriken berechnet")
    print(f"   Dev-Sold Detection Rate: {metrics['dev_sold_detection_rate']:.4f}")
    print(f"   Weighted Cost: {metrics['weighted_cost']:.4f}")
```

**Checkliste:**
- [ ] `dev_sold_detection_rate` wird berechnet
- [ ] `wash_trading_detection_rate` wird berechnet
- [ ] `weighted_cost` wird berechnet (FN 10x FP)
- [ ] `precision_at_k` wird berechnet
- [ ] Metriken werden in Datenbank gespeichert

---

## 📝 Phase 7: Modell-Evaluation Tests

### 7.1 Test-Erstellung Test

**Ziel:** Sicherstellen, dass Modell-Tests korrekt erstellt werden.

**Tests:**
```python
# Test: Modell testen
def test_model_testing():
    # 1. Erstelle Test-Job
    model_id = 1  # Verwende existierendes Modell
    
    payload = {
        "test_start": "2024-01-08T00:00:00Z",
        "test_end": "2024-01-14T23:59:59Z"
    }
    
    response = requests.post(
        f"{BASE_URL}/api/models/{model_id}/test",
        json=payload
    )
    
    assert response.status_code == 200
    data = response.json()
    assert 'job_id' in data
    print(f"✅ Test-Job erstellt: {data['job_id']}")
    
    # 2. Warte auf Completion
    import time
    job_id = data['job_id']
    max_wait = 300  # 5 Minuten
    start_time = time.time()
    
    while time.time() - start_time < max_wait:
        response = requests.get(f"{BASE_URL}/api/queue/{job_id}")
        job_data = response.json()
        
        if job_data['status'] == 'COMPLETED':
            assert 'result_test' in job_data
            test_result = job_data['result_test']
            
            # Prüfe Test-Ergebnisse
            assert 'accuracy' in test_result
            assert 'f1_score' in test_result
            assert 'rug_detection_metrics' in test_result
            
            print("✅ Test erfolgreich abgeschlossen")
            print(f"   Accuracy: {test_result['accuracy']:.4f}")
            return test_result
        
        elif job_data['status'] == 'FAILED':
            raise Exception(f"Test fehlgeschlagen: {job_data.get('error', 'Unknown')}")
        
        time.sleep(5)
    
    raise Exception("Test-Timeout")
```

**Checkliste:**
- [ ] Test-Job wird erstellt
- [ ] Test wird ausgeführt
- [ ] Test-Ergebnisse werden berechnet
- [ ] Rug-Detection-Metriken werden berechnet
- [ ] Ergebnisse werden in Datenbank gespeichert

### 7.2 Vergleichs-Test

**Ziel:** Sicherstellen, dass Modell-Vergleiche funktionieren.

**Tests:**
```python
# Test: Modelle vergleichen
def test_model_comparison():
    payload = {
        "model_a_id": 1,
        "model_b_id": 2,
        "test_start": "2024-01-08T00:00:00Z",
        "test_end": "2024-01-14T23:59:59Z"
    }
    
    response = requests.post(
        f"{BASE_URL}/api/models/compare",
        json=payload
    )
    
    assert response.status_code == 200
    data = response.json()
    assert 'job_id' in data
    
    # Warte auf Completion (wie oben)
    # ...
    
    # Prüfe Vergleichs-Ergebnisse
    assert 'winner_id' in comparison_result
    assert 'a_accuracy' in comparison_result
    assert 'b_accuracy' in comparison_result
    
    print("✅ Vergleich erfolgreich")
```

**Checkliste:**
- [ ] Vergleichs-Job wird erstellt
- [ ] Beide Modelle werden getestet
- [ ] Gewinner wird ermittelt
- [ ] Vergleichs-Ergebnisse werden gespeichert

---

## 📝 Phase 8: Integration-Tests (End-to-End)

### 8.1 Vollständiger Workflow Test

**Ziel:** Teste den kompletten Workflow von Modell-Erstellung bis Evaluation.

**Test-Szenario:**
1. **Modell erstellen** (über UI oder API)
2. **Warten auf Training-Completion**
3. **Modell testen**
4. **Ergebnisse prüfen**
5. **Modell vergleichen** (mit anderem Modell)
6. **Ergebnisse in UI anzeigen**

**Checkliste:**
- [ ] Schritt 1: Modell wird erstellt
- [ ] Schritt 2: Training läuft erfolgreich
- [ ] Schritt 3: Modell-Status wird auf "READY" gesetzt
- [ ] Schritt 4: Test wird erstellt und ausgeführt
- [ ] Schritt 5: Test-Ergebnisse werden angezeigt
- [ ] Schritt 6: Vergleich funktioniert
- [ ] Schritt 7: Alle Ergebnisse sind in UI sichtbar

### 8.2 UI-API-Integration Test

**Ziel:** Sicherstellen, dass UI und API nahtlos zusammenarbeiten.

**Tests:**
1. **UI → API:**
   - [ ] Form-Submission sendet korrekte Daten an API
   - [ ] API-Antworten werden korrekt in UI angezeigt
   - [ ] Fehler werden korrekt behandelt

2. **API → Datenbank:**
   - [ ] Modell wird in Datenbank gespeichert
   - [ ] Job wird in Datenbank gespeichert
   - [ ] Ergebnisse werden in Datenbank gespeichert

3. **Datenbank → UI:**
   - [ ] Modelle werden in UI angezeigt
   - [ ] Jobs werden in UI angezeigt
   - [ ] Ergebnisse werden in UI angezeigt

---

## 📝 Phase 9: Edge Cases & Fehlerbehandlung

### 9.1 Edge Cases

**Ziel:** Teste ungewöhnliche Szenarien.

**Tests:**
1. **Leere Daten:**
   - [ ] Training mit 0 Samples → Fehler
   - [ ] Test mit 0 Samples → Fehler

2. **Extreme Werte:**
   - [ ] Sehr großer Zeitraum (1 Jahr)
   - [ ] Sehr kleiner Zeitraum (1 Stunde)
   - [ ] Sehr hohe Prozent-Änderung (1000%)
   - [ ] Sehr niedrige Prozent-Änderung (0.1%)

3. **Fehlende Features:**
   - [ ] Feature nicht in Datenbank → Fehler
   - [ ] Feature nur NaN-Werte → Warnung

4. **Gleichzeitige Jobs:**
   - [ ] Mehrere Training-Jobs gleichzeitig
   - [ ] Mehrere Test-Jobs gleichzeitig

**Checkliste:**
- [ ] Alle Edge Cases werden behandelt
- [ ] Fehlermeldungen sind aussagekräftig
- [ ] System bleibt stabil

### 9.2 Fehlerbehandlung

**Ziel:** Sicherstellen, dass Fehler korrekt behandelt werden.

**Tests:**
1. **API-Fehler:**
   - [ ] Ungültige Parameter → 400 Bad Request
   - [ ] Fehlende Parameter → 400 Bad Request
   - [ ] Datenbank-Fehler → 500 Internal Server Error

2. **Training-Fehler:**
   - [ ] Training schlägt fehl → Job Status "FAILED"
   - [ ] Fehler wird in Job gespeichert
   - [ ] Fehler wird in UI angezeigt

3. **Daten-Fehler:**
   - [ ] Keine Daten verfügbar → Fehlermeldung
   - [ ] Unvollständige Daten → Warnung

**Checkliste:**
- [ ] Alle Fehler werden abgefangen
- [ ] Fehlermeldungen sind hilfreich
- [ ] System bleibt stabil nach Fehlern

---

## 📝 Phase 10: Performance-Tests

### 10.1 Training-Performance

**Ziel:** Sicherstellen, dass Training in akzeptabler Zeit läuft.

**Tests:**
```python
import time

def test_training_performance():
    start_time = time.time()
    
    # Training mit 10.000 Samples
    # ...
    
    duration = time.time() - start_time
    
    # Erwartung: < 5 Minuten für 10.000 Samples
    assert duration < 300, f"Training zu langsam: {duration:.2f}s"
    print(f"✅ Training-Performance OK: {duration:.2f}s")
```

**Checkliste:**
- [ ] Training < 5 Min für 10.000 Samples
- [ ] Feature-Engineering < 30 Sek
- [ ] Modell-Speicherung < 5 Sek

### 10.2 UI-Performance

**Ziel:** Sicherstellen, dass UI responsiv bleibt.

**Tests:**
- [ ] Seite lädt < 2 Sek
- [ ] Feature-Auswahl reagiert < 100ms
- [ ] Form-Submission < 1 Sek

**Checkliste:**
- [ ] UI bleibt responsiv
- [ ] Keine langen Wartezeiten

---

## ✅ Abschluss-Checkliste

### Vor Produktion:

- [ ] Alle Tests in Phase 1-10 bestanden
- [ ] Keine kritischen Fehler
- [ ] Performance akzeptabel
- [ ] Dokumentation aktualisiert
- [ ] Logs überprüft
- [ ] Backup-Strategie vorhanden

### Test-Report erstellen:

1. **Test-Datum:** _______________
2. **Getestete Version:** _______________
3. **Tester:** _______________
4. **Ergebnisse:**
   - Bestanden: ___ / ___
   - Fehlgeschlagen: ___ / ___
   - Übersprungen: ___ / ___

5. **Kritische Probleme:**
   - _______________
   - _______________

6. **Empfehlungen:**
   - _______________
   - _______________

---

## 🚀 Nächste Schritte

Nach erfolgreichem Abschluss aller Tests:

1. **Produktions-Deployment vorbereiten**
2. **Monitoring einrichten**
3. **Dokumentation finalisieren**
4. **Team-Schulung durchführen**

---

**Viel Erfolg beim Testen! 🎯**

