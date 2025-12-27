"""
Vollständiges System-Test-Skript
Testet alle Funktionen: Modelle, Zeiten, Features, Vergleiche, Web UI
"""
import asyncio
import sys
import os
import time
from datetime import datetime, timedelta
import json
import requests
from typing import Dict, List, Any

# Setze DB_DSN VOR dem Import
DB_DSN = "postgresql://postgres:9HVxi6hN6j7xpmqUx84o@100.118.155.75:5432/beta"
os.environ['DB_DSN'] = DB_DSN

# API Base URL
API_BASE_URL = "http://localhost:8012"
STREAMLIT_URL = "http://localhost:8502"

# Test-Ergebnisse
test_results = {
    "passed": [],
    "failed": [],
    "warnings": []
}

def log_test(test_name: str, passed: bool, message: str = "", warning: bool = False):
    """Loggt Test-Ergebnis"""
    status = "✅ PASS" if passed else "❌ FAIL"
    if warning:
        status = "⚠️ WARN"
    
    print(f"{status} | {test_name}")
    if message:
        print(f"      → {message}")
    
    if passed:
        test_results["passed"].append(test_name)
    elif warning:
        test_results["warnings"].append(f"{test_name}: {message}")
    else:
        test_results["failed"].append(f"{test_name}: {message}")

# ============================================================================
# PHASE 1: API Health & Connectivity
# ============================================================================

def test_api_health():
    """Test 1: API Health Check"""
    try:
        response = requests.get(f"{API_BASE_URL}/api/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            log_test("API Health Check", True, f"Status: {data.get('status')}, DB: {data.get('db_connected')}")
            return data
        else:
            log_test("API Health Check", False, f"Status Code: {response.status_code}")
            return None
    except Exception as e:
        log_test("API Health Check", False, str(e))
        return None

def test_data_availability():
    """Test 2: Data Availability"""
    try:
        response = requests.get(f"{API_BASE_URL}/api/data-availability", timeout=10)
        if response.status_code == 200:
            data = response.json()
            min_ts = data.get('min_timestamp', 'N/A')
            max_ts = data.get('max_timestamp', 'N/A')
            log_test("Data Availability", True, f"Von {min_ts} bis {max_ts}")
            return data
        else:
            log_test("Data Availability", False, f"Status Code: {response.status_code}")
            return None
    except Exception as e:
        log_test("Data Availability", False, str(e))
        return None

# ============================================================================
# PHASE 2: Model Training Tests
# ============================================================================

def test_create_model_random_forest():
    """Test 3: Random Forest Modell erstellen (klassisch)"""
    try:
        # Letzte 3 Tage
        train_end = datetime.now()
        train_start = train_end - timedelta(days=3)
        
        payload = {
            "name": f"TEST_RF_Classic_{int(time.time())}",
            "model_type": "random_forest",
            "train_start": train_start.isoformat(),
            "train_end": train_end.isoformat(),
            "features": ["price_open", "price_high", "price_low", "price_close", "volume_sol"],
            "phases": [1, 2],
            "use_time_based_prediction": False,  # WICHTIG: Klassisch, nicht zeitbasiert!
            "target_var": "price_close",  # WICHTIG: target_var, nicht target_variable!
            "operator": ">",  # WICHTIG: operator, nicht target_operator!
            "target_value": 0.0000001,  # Niedrigerer Wert für bessere Label-Balance
            "params": {
                "n_estimators": 50,  # Reduziert für schnelleres Testing
                "max_depth": 5
            }
        }
        
        response = requests.post(f"{API_BASE_URL}/api/models/create", json=payload, timeout=30)
        if response.status_code == 201:
            data = response.json()
            job_id = data.get('job_id')
            log_test("Random Forest (Classic)", True, f"Job ID: {job_id}")
            return job_id
        else:
            log_test("Random Forest (Classic)", False, f"Status: {response.status_code}, {response.text}")
            return None
    except Exception as e:
        log_test("Random Forest (Classic)", False, str(e))
        return None

def test_create_model_xgboost_time_based():
    """Test 4: XGBoost Modell erstellen (zeitbasiert mit ATH)"""
    try:
        train_end = datetime.now()
        train_start = train_end - timedelta(days=2)
        
        payload = {
            "name": f"TEST_XGB_TimeBased_{int(time.time())}",
            "model_type": "xgboost",
            "train_start": train_start.isoformat(),
            "train_end": train_end.isoformat(),
            "features": [
                "price_open", "price_high", "price_low", "price_close",
                "volume_sol", "buy_volume_sol", "sell_volume_sol",
                "dev_sold_amount", "buy_pressure_ratio",
                "ath_price_sol", "price_vs_ath_pct", "minutes_since_ath"  # ATH-Features!
            ],
            "phases": [1, 2],
            "use_time_based_prediction": True,
            "target_var": "price_close",
            "future_minutes": 10,
            "min_percent_change": 20.0,
            "direction": "up",
            "params": {
                "n_estimators": 50,
                "max_depth": 4,
                "learning_rate": 0.1,
                "use_engineered_features": True,  # Feature-Engineering aktivieren
                "include_ath": True  # ATH-Daten aktivieren
            }
        }
        
        response = requests.post(f"{API_BASE_URL}/api/models/create", json=payload, timeout=30)
        if response.status_code == 201:
            data = response.json()
            job_id = data.get('job_id')
            log_test("XGBoost (Time-Based + ATH)", True, f"Job ID: {job_id}")
            return job_id
        else:
            log_test("XGBoost (Time-Based + ATH)", False, f"Status: {response.status_code}, {response.text}")
            return None
    except Exception as e:
        log_test("XGBoost (Time-Based + ATH)", False, str(e))
        return None

def test_create_model_with_market_context():
    """Test 5: Modell mit Marktstimmung"""
    try:
        train_end = datetime.now()
        train_start = train_end - timedelta(days=2)
        
        payload = {
            "name": f"TEST_MarketContext_{int(time.time())}",
            "model_type": "random_forest",
            "train_start": train_start.isoformat(),
            "train_end": train_end.isoformat(),
            "features": ["price_close", "volume_sol"],
            "phases": [1],
            "use_time_based_prediction": True,
            "target_var": "price_close",
            "future_minutes": 5,
            "min_percent_change": 15.0,
            "direction": "up",
            "params": {
                "n_estimators": 30,
                "use_market_context": True  # Marktstimmung aktivieren
            }
        }
        
        response = requests.post(f"{API_BASE_URL}/api/models/create", json=payload, timeout=30)
        if response.status_code == 201:
            data = response.json()
            job_id = data.get('job_id')
            log_test("Modell mit Marktstimmung", True, f"Job ID: {job_id}")
            return job_id
        else:
            log_test("Modell mit Marktstimmung", False, f"Status: {response.status_code}")
            return None
    except Exception as e:
        log_test("Modell mit Marktstimmung", False, str(e))
        return None

# ============================================================================
# PHASE 3: Job Status & Monitoring
# ============================================================================

def test_job_status(job_id: int):
    """Test 6: Job-Status prüfen"""
    try:
        response = requests.get(f"{API_BASE_URL}/api/queue/{job_id}", timeout=5)
        if response.status_code == 200:
            data = response.json()
            status = data.get('status', 'UNKNOWN')
            progress = data.get('progress', 0)
            log_test(f"Job Status Check (Job {job_id})", True, f"Status: {status}, Progress: {progress*100:.1f}%")
            return data
        else:
            log_test(f"Job Status Check (Job {job_id})", False, f"Status Code: {response.status_code}")
            return None
    except Exception as e:
        log_test(f"Job Status Check (Job {job_id})", False, str(e))
        return None

def wait_for_job_completion(job_id: int, max_wait: int = 300):
    """Wartet auf Job-Abschluss"""
    start_time = time.time()
    while time.time() - start_time < max_wait:
        response = requests.get(f"{API_BASE_URL}/api/queue/{job_id}", timeout=5)
        if response.status_code == 200:
            data = response.json()
            status = data.get('status')
            progress = data.get('progress', 0)
            
            if status == 'COMPLETED':
                return True, data
            elif status == 'FAILED':
                error = data.get('error_msg', 'Unknown error')
                return False, f"Job failed: {error}"
            
            print(f"      ⏳ Job {job_id}: {status} ({progress*100:.1f}%)")
            time.sleep(5)
        else:
            time.sleep(5)
    
    return False, "Timeout"

# ============================================================================
# PHASE 4: Model Testing
# ============================================================================

def test_list_models():
    """Test 7: Modelle auflisten"""
    try:
        response = requests.get(f"{API_BASE_URL}/api/models", timeout=10)
        if response.status_code == 200:
            data = response.json()
            # API gibt direkt eine Liste zurück, nicht ein Dict
            if isinstance(data, list):
                models = data
            else:
                models = data.get('models', [])
            ready_models = [m for m in models if m.get('status') == 'READY']
            log_test("List Models", True, f"{len(models)} Modelle total, {len(ready_models)} READY")
            return models
        else:
            log_test("List Models", False, f"Status Code: {response.status_code}")
            return []
    except Exception as e:
        log_test("List Models", False, str(e))
        return []

def test_model_details(model_id: int):
    """Test 8: Modell-Details abrufen"""
    try:
        response = requests.get(f"{API_BASE_URL}/api/models/{model_id}", timeout=10)
        if response.status_code == 200:
            data = response.json()
            name = data.get('name', 'N/A')
            model_type = data.get('model_type', 'N/A')
            status = data.get('status', 'N/A')
            accuracy = data.get('training_accuracy', 0)
            log_test(f"Model Details (ID {model_id})", True, f"{name} ({model_type}): {status}, Acc: {accuracy:.4f}")
            return data
        else:
            log_test(f"Model Details (ID {model_id})", False, f"Status Code: {response.status_code}")
            return None
    except Exception as e:
        log_test(f"Model Details (ID {model_id})", False, str(e))
        return None

def test_model_testing(model_id: int):
    """Test 9: Modell testen"""
    try:
        # Test auf letzten 12 Stunden
        test_end = datetime.now()
        test_start = test_end - timedelta(hours=12)
        
        payload = {
            "test_start": test_start.isoformat(),
            "test_end": test_end.isoformat()
        }
        
        response = requests.post(f"{API_BASE_URL}/api/models/{model_id}/test", json=payload, timeout=30)
        if response.status_code == 201:
            data = response.json()
            job_id = data.get('job_id')
            log_test(f"Model Testing (Model {model_id})", True, f"Test Job ID: {job_id}")
            return job_id
        else:
            log_test(f"Model Testing (Model {model_id})", False, f"Status: {response.status_code}, {response.text}")
            return None
    except Exception as e:
        log_test(f"Model Testing (Model {model_id})", False, str(e))
        return None

# ============================================================================
# PHASE 5: Model Comparison
# ============================================================================

def test_model_comparison(model_a_id: int, model_b_id: int):
    """Test 10: Modell-Vergleich"""
    try:
        test_end = datetime.now()
        test_start = test_end - timedelta(hours=6)
        
        payload = {
            "model_a_id": model_a_id,
            "model_b_id": model_b_id,
            "test_start": test_start.isoformat(),
            "test_end": test_end.isoformat()
        }
        
        response = requests.post(f"{API_BASE_URL}/api/models/compare", json=payload, timeout=30)
        if response.status_code == 201:
            data = response.json()
            job_id = data.get('job_id')
            log_test(f"Model Comparison ({model_a_id} vs {model_b_id})", True, f"Compare Job ID: {job_id}")
            return job_id
        else:
            log_test(f"Model Comparison ({model_a_id} vs {model_b_id})", False, f"Status: {response.status_code}")
            return None
    except Exception as e:
        log_test(f"Model Comparison ({model_a_id} vs {model_b_id})", False, str(e))
        return None

# ============================================================================
# PHASE 6: Web UI Tests
# ============================================================================

def test_streamlit_ui():
    """Test 11: Streamlit UI erreichbar"""
    try:
        response = requests.get(STREAMLIT_URL, timeout=5)
        if response.status_code == 200:
            log_test("Streamlit UI", True, f"Erreichbar auf {STREAMLIT_URL}")
            return True
        else:
            log_test("Streamlit UI", False, f"Status Code: {response.status_code}")
            return False
    except Exception as e:
        log_test("Streamlit UI", False, str(e))
        return False

def test_api_metrics():
    """Test 12: Prometheus Metrics"""
    try:
        response = requests.get(f"{API_BASE_URL}/api/metrics", timeout=5)
        if response.status_code == 200:
            content = response.text
            if "tracker_" in content or "job_" in content:
                log_test("Prometheus Metrics", True, "Metriken verfügbar")
                return True
            else:
                log_test("Prometheus Metrics", False, "Keine Metriken gefunden")
                return False
        else:
            log_test("Prometheus Metrics", False, f"Status Code: {response.status_code}")
            return False
    except Exception as e:
        log_test("Prometheus Metrics", False, str(e))
        return False

# ============================================================================
# PHASE 7: Edge Cases & Error Handling
# ============================================================================

def test_invalid_model_id():
    """Test 13: Ungültige Model-ID"""
    try:
        response = requests.get(f"{API_BASE_URL}/api/models/999999", timeout=5)
        if response.status_code == 404:
            log_test("Invalid Model ID Handling", True, "404 korrekt zurückgegeben")
            return True
        else:
            log_test("Invalid Model ID Handling", False, f"Erwartet 404, bekam {response.status_code}")
            return False
    except Exception as e:
        log_test("Invalid Model ID Handling", False, str(e))
        return False

def test_invalid_job_request():
    """Test 14: Ungültiger Job-Request"""
    try:
        # Fehlende Parameter
        payload = {
            "name": "TEST_Invalid",
            "model_type": "random_forest"
            # train_start, train_end fehlen!
        }
        
        response = requests.post(f"{API_BASE_URL}/api/models/create", json=payload, timeout=5)
        if response.status_code in [400, 422]:
            log_test("Invalid Job Request Handling", True, f"Fehler korrekt: {response.status_code}")
            return True
        else:
            log_test("Invalid Job Request Handling", False, f"Erwartet 400/422, bekam {response.status_code}")
            return False
    except Exception as e:
        log_test("Invalid Job Request Handling", False, str(e))
        return False

# ============================================================================
# MAIN TEST RUNNER
# ============================================================================

def main():
    """Hauptfunktion - Führt alle Tests aus"""
    print("=" * 80)
    print("VOLLSTÄNDIGES SYSTEM-TEST")
    print("=" * 80)
    print()
    
    # Phase 1: Connectivity
    print("📡 PHASE 1: API Health & Connectivity")
    print("-" * 80)
    health_data = test_api_health()
    data_availability = test_data_availability()
    print()
    
    if not health_data or not health_data.get('db_connected'):
        print("❌ Datenbank nicht verbunden! Abbruch.")
        return
    
    # Phase 2: Model Training
    print("🏋️ PHASE 2: Model Training")
    print("-" * 80)
    job_rf = test_create_model_random_forest()
    job_xgb = test_create_model_xgboost_time_based()
    job_market = test_create_model_with_market_context()
    print()
    
    # Warte auf erste Jobs
    print("⏳ Warte auf Job-Abschluss...")
    print("-" * 80)
    
    completed_models = []
    
    if job_rf:
        print(f"Warte auf Random Forest Job {job_rf}...")
        success, result = wait_for_job_completion(job_rf, max_wait=180)
        if success:
            model_id = result.get('result_model_id')
            if model_id:
                completed_models.append(model_id)
                log_test("Random Forest Training", True, f"Modell {model_id} erstellt")
            else:
                log_test("Random Forest Training", False, "Keine Model-ID zurückgegeben")
        else:
            log_test("Random Forest Training", False, result)
    
    if job_xgb:
        print(f"Warte auf XGBoost Job {job_xgb}...")
        success, result = wait_for_job_completion(job_xgb, max_wait=180)
        if success:
            model_id = result.get('result_model_id')
            if model_id:
                completed_models.append(model_id)
                log_test("XGBoost Training", True, f"Modell {model_id} erstellt")
            else:
                log_test("XGBoost Training", False, "Keine Model-ID zurückgegeben")
        else:
            log_test("XGBoost Training", False, result)
    
    print()
    
    # Phase 3: Model Management
    print("📋 PHASE 3: Model Management")
    print("-" * 80)
    models = test_list_models()
    
    if completed_models:
        for model_id in completed_models[:2]:  # Nur erste 2 testen
            test_model_details(model_id)
    print()
    
    # Phase 4: Model Testing
    print("🧪 PHASE 4: Model Testing")
    print("-" * 80)
    test_jobs = []
    
    if completed_models:
        for model_id in completed_models[:2]:
            test_job_id = test_model_testing(model_id)
            if test_job_id:
                test_jobs.append((model_id, test_job_id))
    
    # Warte auf Test-Jobs
    if test_jobs:
        print("⏳ Warte auf Test-Abschluss...")
        for model_id, test_job_id in test_jobs:
            success, result = wait_for_job_completion(test_job_id, max_wait=120)
            if success:
                log_test(f"Model Test (Model {model_id})", True, "Test abgeschlossen")
            else:
                log_test(f"Model Test (Model {model_id})", False, result)
    print()
    
    # Phase 5: Model Comparison
    print("⚔️ PHASE 5: Model Comparison")
    print("-" * 80)
    if len(completed_models) >= 2:
        compare_job = test_model_comparison(completed_models[0], completed_models[1])
        if compare_job:
            print(f"⏳ Warte auf Vergleich...")
            success, result = wait_for_job_completion(compare_job, max_wait=120)
            if success:
                log_test("Model Comparison", True, "Vergleich abgeschlossen")
            else:
                log_test("Model Comparison", False, result)
    else:
        log_test("Model Comparison", False, "Nicht genug Modelle für Vergleich", warning=True)
    print()
    
    # Phase 6: Web UI
    print("🌐 PHASE 6: Web UI")
    print("-" * 80)
    test_streamlit_ui()
    test_api_metrics()
    print()
    
    # Phase 7: Error Handling
    print("🛡️ PHASE 7: Error Handling")
    print("-" * 80)
    test_invalid_model_id()
    test_invalid_job_request()
    print()
    
    # Zusammenfassung
    print("=" * 80)
    print("ZUSAMMENFASSUNG")
    print("=" * 80)
    print(f"✅ Bestanden: {len(test_results['passed'])}")
    print(f"❌ Fehlgeschlagen: {len(test_results['failed'])}")
    print(f"⚠️ Warnungen: {len(test_results['warnings'])}")
    print()
    
    if test_results['failed']:
        print("❌ FEHLGESCHLAGENE TESTS:")
        for test in test_results['failed']:
            print(f"   - {test}")
        print()
    
    if test_results['warnings']:
        print("⚠️ WARNUNGEN:")
        for test in test_results['warnings']:
            print(f"   - {test}")
        print()
    
    # Gesamt-Status
    total_tests = len(test_results['passed']) + len(test_results['failed'])
    success_rate = (len(test_results['passed']) / total_tests * 100) if total_tests > 0 else 0
    
    print(f"📊 Erfolgsrate: {success_rate:.1f}%")
    
    if len(test_results['failed']) == 0:
        print("\n✅ ALLE TESTS ERFOLGREICH!")
    else:
        print(f"\n⚠️ {len(test_results['failed'])} Test(s) fehlgeschlagen")

if __name__ == "__main__":
    main()

