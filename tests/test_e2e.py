#!/usr/bin/env python3
"""
End-to-End Test für ML Training Service
Testet alle Komponenten von unten nach oben
"""
import requests
import json
import time
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, Optional

# Konfiguration
API_BASE_URL = "http://localhost:8000"
STREAMLIT_URL = "http://localhost:8501"

class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    END = '\033[0m'
    BOLD = '\033[1m'

def print_success(msg: str):
    print(f"{Colors.GREEN}✅ {msg}{Colors.END}")

def print_error(msg: str):
    print(f"{Colors.RED}❌ {msg}{Colors.END}")

def print_warning(msg: str):
    print(f"{Colors.YELLOW}⚠️  {msg}{Colors.END}")

def print_info(msg: str):
    print(f"{Colors.BLUE}ℹ️  {msg}{Colors.END}")

def print_header(msg: str):
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{msg}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.END}\n")

# ============================================================
# Test 1: Health Check
# ============================================================
def test_health_check() -> bool:
    """Test 1: Health Check"""
    print_header("TEST 1: Health Check")
    try:
        response = requests.get(f"{API_BASE_URL}/api/health", timeout=10)
        if response.status_code == 200:
            data = response.json()
            print_success(f"Health Check erfolgreich")
            print_info(f"  Status: {data.get('status')}")
            print_info(f"  DB Connected: {data.get('db_connected')}")
            print_info(f"  Uptime: {data.get('uptime_seconds')}s")
            return data.get('db_connected', False)
        else:
            print_error(f"Health Check fehlgeschlagen: {response.status_code}")
            return False
    except Exception as e:
        print_error(f"Health Check Fehler: {e}")
        return False

# ============================================================
# Test 2: Phasen laden
# ============================================================
def test_load_phases() -> Optional[list]:
    """Test 2: Phasen aus ref_coin_phases laden"""
    print_header("TEST 2: Phasen laden (ref_coin_phases)")
    try:
        response = requests.get(f"{API_BASE_URL}/api/phases", timeout=10)
        if response.status_code == 200:
            phases = response.json()
            print_success(f"{len(phases)} Phasen geladen")
            for phase in phases:
                print_info(f"  Phase {phase['id']}: {phase['name']} ({phase['interval_seconds']}s)")
            return phases
        else:
            print_error(f"Phasen laden fehlgeschlagen: {response.status_code}")
            return None
    except Exception as e:
        print_error(f"Phasen laden Fehler: {e}")
        return None

# ============================================================
# Test 3: Modell-Liste
# ============================================================
def test_list_models() -> list:
    """Test 3: Alle Modelle auflisten"""
    print_header("TEST 3: Modelle auflisten")
    try:
        response = requests.get(f"{API_BASE_URL}/api/models", timeout=10)
        if response.status_code == 200:
            models = response.json()
            print_success(f"{len(models)} Modell(e) gefunden")
            for model in models[:5]:  # Zeige nur erste 5
                print_info(f"  - {model.get('name')} ({model.get('status')})")
            return models
        else:
            print_error(f"Modelle auflisten fehlgeschlagen: {response.status_code}")
            return []
    except Exception as e:
        print_error(f"Modelle auflisten Fehler: {e}")
        return []

# ============================================================
# Test 4: Normales Modell trainieren
# ============================================================
def test_train_normal_model(phases: Optional[list] = None) -> Optional[int]:
    """Test 4: Normales Modell trainieren"""
    print_header("TEST 4: Normales Modell trainieren")
    
    # Verwende erste 2 Phasen wenn verfügbar
    phase_ids = None
    if phases and len(phases) >= 2:
        phase_ids = [phases[0]['id'], phases[1]['id']]
        print_info(f"Verwende Phasen: {phase_ids}")
    
    train_start = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat().replace('+00:00', 'Z')
    train_end = datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')
    
    # Verwende market_cap_close statt price_close (hat wahrscheinlich bessere Daten)
    data = {
        "name": f"TEST_Normal_{int(time.time() * 1000)}",  # Millisekunden für Eindeutigkeit
        "model_type": "random_forest",  # Teste random_forest
        "target_var": "market_cap_close",
        "operator": ">",
        "target_value": 50.0,  # Schwellwert basierend auf Daten (Ø 71.95, Range 25-301)
        "features": ["price_open", "price_high", "price_low", "volume_sol"],
        "phases": phase_ids,
        "params": {"n_estimators": 10, "max_depth": 5},  # Kleine Werte für schnelles Training
        "train_start": train_start,
        "train_end": train_end,
        "description": "E2E Test - Normales Modell",
        "use_time_based_prediction": False
    }
    
    try:
        response = requests.post(f"{API_BASE_URL}/api/models/create", json=data, timeout=30)
        if response.status_code == 201:
            result = response.json()
            job_id = result.get('job_id')
            print_success(f"Training-Job erstellt: Job-ID {job_id}")
            return job_id
        else:
            print_error(f"Training-Job erstellen fehlgeschlagen: {response.status_code}")
            print_error(f"  Response: {response.text}")
            return None
    except Exception as e:
        print_error(f"Training-Job erstellen Fehler: {e}")
        return None

# ============================================================
# Test 5: Zeitbasiertes Modell trainieren
# ============================================================
def test_train_time_based_model(phases: Optional[list] = None) -> Optional[int]:
    """Test 5: Zeitbasiertes Modell trainieren"""
    print_header("TEST 5: Zeitbasiertes Modell trainieren")
    
    # Verwende erste 2 Phasen wenn verfügbar
    phase_ids = None
    if phases and len(phases) >= 2:
        phase_ids = [phases[0]['id'], phases[1]['id']]
        print_info(f"Verwende Phasen: {phase_ids}")
    
    train_start = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat().replace('+00:00', 'Z')
    train_end = datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')
    
    # Verwende market_cap_close statt price_close (hat wahrscheinlich bessere Daten)
    data = {
        "name": f"TEST_TimeBased_{int(time.time() * 1000)}",  # Millisekunden für Eindeutigkeit
        "model_type": "xgboost",  # Teste xgboost für zeitbasierte Vorhersage
        "target_var": "market_cap_close",  # Wird für zeitbasierte Vorhersage verwendet
        "operator": None,  # Optional bei zeitbasierter Vorhersage
        "target_value": None,  # Optional bei zeitbasierter Vorhersage
        "features": ["price_open", "price_high", "price_low", "volume_sol"],
        "phases": phase_ids,
        "params": {"n_estimators": 10, "max_depth": 5},  # Kleine Werte für schnelles Training
        "train_start": train_start,
        "train_end": train_end,
        "description": "E2E Test - Zeitbasiertes Modell",
        "use_time_based_prediction": True,
        "future_minutes": 10,
        "min_percent_change": 2.0,  # Niedrigere Schwelle für bessere Balance
        "direction": "up"
    }
    
    try:
        response = requests.post(f"{API_BASE_URL}/api/models/create", json=data, timeout=30)
        if response.status_code == 201:
            result = response.json()
            job_id = result.get('job_id')
            print_success(f"Zeitbasiertes Training-Job erstellt: Job-ID {job_id}")
            return job_id
        else:
            print_error(f"Zeitbasiertes Training-Job erstellen fehlgeschlagen: {response.status_code}")
            print_error(f"  Response: {response.text}")
            return None
    except Exception as e:
        print_error(f"Zeitbasiertes Training-Job erstellen Fehler: {e}")
        return None

# ============================================================
# Test 6: Job-Status prüfen
# ============================================================
def test_job_status(job_id: int, max_wait: int = 300) -> Optional[Dict[str, Any]]:
    """Test 6: Job-Status prüfen und warten bis fertig"""
    print_header(f"TEST 6: Job-Status prüfen (Job-ID: {job_id})")
    
    start_time = time.time()
    while time.time() - start_time < max_wait:
        try:
            response = requests.get(f"{API_BASE_URL}/api/queue/{job_id}", timeout=10)
            if response.status_code == 200:
                job = response.json()
                status = job.get('status')
                progress = job.get('progress', 0) * 100
                
                print_info(f"  Status: {status}, Progress: {progress:.1f}%")
                
                if status == "COMPLETED":
                    print_success(f"Job {job_id} erfolgreich abgeschlossen!")
                    return job
                elif status == "FAILED":
                    error_msg = job.get('error_msg', 'Unbekannter Fehler')
                    print_error(f"Job {job_id} fehlgeschlagen: {error_msg}")
                    return job
                
                time.sleep(5)  # Warte 5 Sekunden
            else:
                print_error(f"Job-Status abrufen fehlgeschlagen: {response.status_code}")
                return None
        except Exception as e:
            print_error(f"Job-Status prüfen Fehler: {e}")
            return None
    
    print_warning(f"Job {job_id} hat Zeitlimit überschritten ({max_wait}s)")
    return None

# ============================================================
# Test 7: Modell testen
# ============================================================
def test_model_test(model_id: int) -> Optional[int]:
    """Test 7: Modell testen"""
    print_header(f"TEST 7: Modell testen (Modell-ID: {model_id})")
    
    test_start = (datetime.now(timezone.utc) - timedelta(days=7)).isoformat().replace('+00:00', 'Z')
    test_end = datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')
    
    data = {
        "test_start": test_start,
        "test_end": test_end
    }
    
    try:
        response = requests.post(f"{API_BASE_URL}/api/models/{model_id}/test", json=data, timeout=30)
        if response.status_code == 201:
            result = response.json()
            job_id = result.get('job_id')
            print_success(f"Test-Job erstellt: Job-ID {job_id}")
            return job_id
        else:
            print_error(f"Test-Job erstellen fehlgeschlagen: {response.status_code}")
            print_error(f"  Response: {response.text}")
            return None
    except Exception as e:
        print_error(f"Test-Job erstellen Fehler: {e}")
        return None

# ============================================================
# Test 8: Modelle vergleichen
# ============================================================
def test_compare_models(model_a_id: int, model_b_id: int) -> Optional[int]:
    """Test 8: Zwei Modelle vergleichen"""
    print_header(f"TEST 8: Modelle vergleichen (A: {model_a_id}, B: {model_b_id})")
    
    test_start = (datetime.now(timezone.utc) - timedelta(days=7)).isoformat().replace('+00:00', 'Z')
    test_end = datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')
    
    data = {
        "model_a_id": model_a_id,
        "model_b_id": model_b_id,
        "test_start": test_start,
        "test_end": test_end
    }
    
    try:
        response = requests.post(f"{API_BASE_URL}/api/models/compare", json=data, timeout=30)
        if response.status_code == 201:
            result = response.json()
            job_id = result.get('job_id')
            print_success(f"Vergleichs-Job erstellt: Job-ID {job_id}")
            return job_id
        else:
            print_error(f"Vergleichs-Job erstellen fehlgeschlagen: {response.status_code}")
            print_error(f"  Response: {response.text}")
            return None
    except Exception as e:
        print_error(f"Vergleichs-Job erstellen Fehler: {e}")
        return None

# ============================================================
# Test 9: Jobs auflisten
# ============================================================
def test_list_jobs() -> list:
    """Test 9: Alle Jobs auflisten"""
    print_header("TEST 9: Jobs auflisten")
    try:
        response = requests.get(f"{API_BASE_URL}/api/queue", timeout=10)
        if response.status_code == 200:
            jobs = response.json()
            print_success(f"{len(jobs)} Job(s) gefunden")
            
            # Zeige letzte 5 Jobs
            for job in jobs[:5]:
                status = job.get('status')
                job_type = job.get('job_type')
                print_info(f"  - {job_type} Job {job.get('id')}: {status}")
            
            return jobs
        else:
            print_error(f"Jobs auflisten fehlgeschlagen: {response.status_code}")
            return []
    except Exception as e:
        print_error(f"Jobs auflisten Fehler: {e}")
        return []

# ============================================================
# Test 10: Metrics prüfen
# ============================================================
def test_metrics() -> bool:
    """Test 10: Prometheus Metrics prüfen"""
    print_header("TEST 10: Prometheus Metrics")
    try:
        response = requests.get(f"{API_BASE_URL}/api/metrics", timeout=10)
        if response.status_code == 200:
            metrics = response.text
            print_success("Metrics erfolgreich abgerufen")
            print_info(f"  Länge: {len(metrics)} Zeichen")
            # Zeige erste paar Zeilen
            lines = metrics.split('\n')[:10]
            for line in lines:
                if line.strip():
                    print_info(f"  {line}")
            return True
        else:
            print_error(f"Metrics abrufen fehlgeschlagen: {response.status_code}")
            return False
    except Exception as e:
        print_error(f"Metrics abrufen Fehler: {e}")
        return False

# ============================================================
# Haupt-Test-Funktion
# ============================================================
def main():
    """Führt alle Tests aus"""
    print(f"\n{Colors.BOLD}{Colors.BLUE}")
    print("="*60)
    print("  ML TRAINING SERVICE - END-TO-END TEST")
    print("="*60)
    print(f"{Colors.END}\n")
    
    results = {
        "health_check": False,
        "phases_loaded": False,
        "normal_model_trained": False,
        "time_based_model_trained": False,
        "model_tested": False,
        "models_compared": False,
        "jobs_listed": False,
        "metrics_checked": False
    }
    
    model_ids = []
    
    # Test 1: Health Check
    results["health_check"] = test_health_check()
    if not results["health_check"]:
        print_error("Health Check fehlgeschlagen - weitere Tests werden übersprungen")
        return
    
    # Test 2: Phasen laden
    phases = test_load_phases()
    results["phases_loaded"] = phases is not None and len(phases) > 0
    
    # Test 3: Modelle auflisten
    models = test_list_models()
    
    # Test 4: Normales Modell trainieren
    normal_job_id = test_train_normal_model(phases)
    if normal_job_id:
        print_info("Warte auf Training-Abschluss...")
        normal_job = test_job_status(normal_job_id, max_wait=600)  # 10 Minuten
        if normal_job and normal_job.get('status') == 'COMPLETED':
            results["normal_model_trained"] = True
            model_id = normal_job.get('result_model_id')
            if model_id:
                model_ids.append(model_id)
                print_success(f"Modell erstellt: ID {model_id}")
    
    # Test 5: Zeitbasiertes Modell trainieren
    time_based_job_id = test_train_time_based_model(phases)
    if time_based_job_id:
        print_info("Warte auf Training-Abschluss...")
        time_based_job = test_job_status(time_based_job_id, max_wait=600)  # 10 Minuten
        if time_based_job and time_based_job.get('status') == 'COMPLETED':
            results["time_based_model_trained"] = True
            model_id = time_based_job.get('result_model_id')
            if model_id:
                model_ids.append(model_id)
                print_success(f"Zeitbasiertes Modell erstellt: ID {model_id}")
    
    # Test 6-7: Modell testen (wenn Modelle vorhanden)
    if len(model_ids) >= 1:
        test_job_id = test_model_test(model_ids[0])
        if test_job_id:
            print_info("Warte auf Test-Abschluss...")
            test_job = test_job_status(test_job_id, max_wait=300)  # 5 Minuten
            if test_job and test_job.get('status') == 'COMPLETED':
                results["model_tested"] = True
    
    # Test 8: Modelle vergleichen (wenn mindestens 2 Modelle vorhanden)
    if len(model_ids) >= 2:
        compare_job_id = test_compare_models(model_ids[0], model_ids[1])
        if compare_job_id:
            print_info("Warte auf Vergleich-Abschluss...")
            compare_job = test_job_status(compare_job_id, max_wait=300)  # 5 Minuten
            if compare_job and compare_job.get('status') == 'COMPLETED':
                results["models_compared"] = True
    
    # Test 9: Jobs auflisten
    jobs = test_list_jobs()
    results["jobs_listed"] = len(jobs) > 0
    
    # Test 10: Metrics
    results["metrics_checked"] = test_metrics()
    
    # Zusammenfassung
    print_header("TEST-ZUSAMMENFASSUNG")
    
    total_tests = len(results)
    passed_tests = sum(1 for v in results.values() if v)
    
    for test_name, passed in results.items():
        if passed:
            print_success(f"{test_name}: ✅")
        else:
            print_error(f"{test_name}: ❌")
    
    print(f"\n{Colors.BOLD}Ergebnis: {passed_tests}/{total_tests} Tests erfolgreich{Colors.END}\n")
    
    if passed_tests == total_tests:
        print_success("🎉 ALLE TESTS ERFOLGREICH!")
        return 0
    else:
        print_warning(f"⚠️  {total_tests - passed_tests} Test(s) fehlgeschlagen")
        return 1

if __name__ == "__main__":
    exit(main())

