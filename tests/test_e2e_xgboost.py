#!/usr/bin/env python3
"""
Zusätzlicher Test für XGBoost Modelle
Testet sowohl normale als auch zeitbasierte XGBoost-Modelle
"""
import requests
import json
import time
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, Optional

# Konfiguration
API_BASE_URL = "http://localhost:8000"

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

def print_info(msg: str):
    print(f"{Colors.BLUE}ℹ️  {msg}{Colors.END}")

def print_header(msg: str):
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{msg}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.END}\n")

def test_xgboost_normal(phases: Optional[list] = None) -> Optional[int]:
    """Test: Normales XGBoost-Modell trainieren"""
    print_header("TEST: Normales XGBoost-Modell trainieren")
    
    phase_ids = None
    if phases and len(phases) >= 2:
        phase_ids = [phases[0]['id'], phases[1]['id']]
        print_info(f"Verwende Phasen: {phase_ids}")
    
    train_start = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat().replace('+00:00', 'Z')
    train_end = datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')
    
    data = {
        "name": f"TEST_XGBoost_Normal_{int(time.time() * 1000000)}",  # Mikrosekunden für Eindeutigkeit
        "model_type": "xgboost",
        "target_var": "market_cap_close",
        "operator": ">",
        "target_value": 50.0,
        "features": ["price_open", "price_high", "price_low", "volume_sol"],
        "phases": phase_ids,
        "params": {"n_estimators": 10, "max_depth": 3},  # Kleine Werte für schnelles Training
        "train_start": train_start,
        "train_end": train_end,
        "description": "E2E Test - Normales XGBoost-Modell",
        "use_time_based_prediction": False
    }
    
    try:
        response = requests.post(f"{API_BASE_URL}/api/models/create", json=data, timeout=30)
        if response.status_code == 201:
            result = response.json()
            job_id = result.get('job_id')
            print_success(f"XGBoost Training-Job erstellt: Job-ID {job_id}")
            return job_id
        else:
            print_error(f"XGBoost Training-Job erstellen fehlgeschlagen: {response.status_code}")
            print_error(f"  Response: {response.text}")
            return None
    except Exception as e:
        print_error(f"XGBoost Training-Job erstellen Fehler: {e}")
        return None

def test_xgboost_time_based(phases: Optional[list] = None) -> Optional[int]:
    """Test: Zeitbasiertes XGBoost-Modell trainieren"""
    print_header("TEST: Zeitbasiertes XGBoost-Modell trainieren")
    
    phase_ids = None
    if phases and len(phases) >= 2:
        phase_ids = [phases[0]['id'], phases[1]['id']]
        print_info(f"Verwende Phasen: {phase_ids}")
    
    train_start = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat().replace('+00:00', 'Z')
    train_end = datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')
    
    data = {
        "name": f"TEST_XGBoost_TimeBased_{int(time.time() * 1000000)}",  # Mikrosekunden für Eindeutigkeit
        "model_type": "xgboost",
        "target_var": "market_cap_close",
        "operator": None,
        "target_value": None,
        "features": ["price_open", "price_high", "price_low", "volume_sol"],
        "phases": phase_ids,
        "params": {"n_estimators": 10, "max_depth": 3},  # Kleine Werte für schnelles Training
        "train_start": train_start,
        "train_end": train_end,
        "description": "E2E Test - Zeitbasiertes XGBoost-Modell",
        "use_time_based_prediction": True,
        "future_minutes": 10,
        "min_percent_change": 2.0,
        "direction": "up"
    }
    
    try:
        response = requests.post(f"{API_BASE_URL}/api/models/create", json=data, timeout=30)
        if response.status_code == 201:
            result = response.json()
            job_id = result.get('job_id')
            print_success(f"Zeitbasiertes XGBoost Training-Job erstellt: Job-ID {job_id}")
            return job_id
        else:
            print_error(f"Zeitbasiertes XGBoost Training-Job erstellen fehlgeschlagen: {response.status_code}")
            print_error(f"  Response: {response.text}")
            return None
    except Exception as e:
        print_error(f"Zeitbasiertes XGBoost Training-Job erstellen Fehler: {e}")
        return None

def wait_for_job(job_id: int, max_wait: int = 300) -> Optional[Dict[str, Any]]:
    """Wartet auf Job-Abschluss"""
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
                
                time.sleep(5)
            else:
                print_error(f"Job-Status abrufen fehlgeschlagen: {response.status_code}")
                return None
        except Exception as e:
            print_error(f"Job-Status prüfen Fehler: {e}")
            return None
    
    print(f"{Colors.YELLOW}⚠️  Job {job_id} hat Zeitlimit überschritten ({max_wait}s){Colors.END}")
    return None

def main():
    print(f"\n{Colors.BOLD}{Colors.BLUE}")
    print("="*60)
    print("  XGBOOST MODELLE - END-TO-END TEST")
    print("="*60)
    print(f"{Colors.END}\n")
    
    # Lade Phasen
    try:
        response = requests.get(f"{API_BASE_URL}/api/phases", timeout=10)
        if response.status_code == 200:
            phases = response.json()
            print_success(f"{len(phases)} Phasen geladen")
        else:
            phases = None
    except:
        phases = None
    
    results = {
        "xgboost_normal": False,
        "xgboost_time_based": False
    }
    
    # Test 1: Normales XGBoost-Modell
    normal_job_id = test_xgboost_normal(phases)
    if normal_job_id:
        print_info("Warte auf Training-Abschluss...")
        normal_job = wait_for_job(normal_job_id, max_wait=600)
        if normal_job and normal_job.get('status') == 'COMPLETED':
            results["xgboost_normal"] = True
            model_id = normal_job.get('result_model_id')
            if model_id:
                print_success(f"XGBoost-Modell erstellt: ID {model_id}")
    
    # Test 2: Zeitbasiertes XGBoost-Modell
    time_based_job_id = test_xgboost_time_based(phases)
    if time_based_job_id:
        print_info("Warte auf Training-Abschluss...")
        time_based_job = wait_for_job(time_based_job_id, max_wait=600)
        if time_based_job and time_based_job.get('status') == 'COMPLETED':
            results["xgboost_time_based"] = True
            model_id = time_based_job.get('result_model_id')
            if model_id:
                print_success(f"Zeitbasiertes XGBoost-Modell erstellt: ID {model_id}")
    
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
        print_success("🎉 ALLE XGBOOST-TESTS ERFOLGREICH!")
        return 0
    else:
        print(f"{Colors.YELLOW}⚠️  {total_tests - passed_tests} Test(s) fehlgeschlagen{Colors.END}")
        return 1

if __name__ == "__main__":
    exit(main())

