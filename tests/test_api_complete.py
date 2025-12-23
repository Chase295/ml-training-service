#!/usr/bin/env python3
"""
Vollständiger API-Test nach Datenbank-Umstellung und Refactoring
Testet alle wichtigen Endpoints über die API
"""
import httpx
import json
import time
from datetime import datetime, timedelta
from typing import Dict, Any, Optional

API_BASE = "http://localhost:8000/api"

def print_section(title: str):
    """Druckt einen Abschnitts-Titel"""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")

def print_result(success: bool, message: str, data: Any = None):
    """Druckt ein Testergebnis"""
    status = "✅" if success else "❌"
    print(f"{status} {message}")
    if data and not success:
        print(f"   Fehler: {data}")

def test_health():
    """Test 1: Health Check"""
    print_section("1. Health Check")
    try:
        response = httpx.get(f"{API_BASE}/health", timeout=10)
        if response.status_code == 200:
            data = response.json()
            print_result(True, f"Health Check: {data.get('status')}")
            print(f"   DB Connected: {data.get('db_connected')}")
            print(f"   Uptime: {data.get('uptime_seconds')}s")
            return True
        else:
            print_result(False, f"Health Check fehlgeschlagen: {response.status_code}")
            return False
    except Exception as e:
        print_result(False, f"Health Check Exception: {e}")
        return False

def test_list_models():
    """Test 2: Modelle auflisten"""
    print_section("2. Modelle auflisten")
    try:
        response = httpx.get(f"{API_BASE}/models", timeout=10)
        if response.status_code == 200:
            models = response.json()
            print_result(True, f"Modelle gefunden: {len(models)}")
            if models:
                print(f"   Erstes Modell: ID {models[0].get('id')}, Name: {models[0].get('name')}")
            return models
        else:
            print_result(False, f"Modelle auflisten fehlgeschlagen: {response.status_code}")
            return []
    except Exception as e:
        print_result(False, f"Modelle auflisten Exception: {e}")
        return []

def test_list_jobs():
    """Test 3: Jobs auflisten"""
    print_section("3. Jobs auflisten")
    try:
        response = httpx.get(f"{API_BASE}/queue", timeout=10)
        if response.status_code == 200:
            jobs = response.json()
            print_result(True, f"Jobs gefunden: {len(jobs)}")
            if jobs:
                print(f"   Letzter Job: ID {jobs[0].get('id')}, Type: {jobs[0].get('job_type')}, Status: {jobs[0].get('status')}")
            return jobs
        else:
            print_result(False, f"Jobs auflisten fehlgeschlagen: {response.status_code}")
            return []
    except Exception as e:
        print_result(False, f"Jobs auflisten Exception: {e}")
        return []

def test_create_model():
    """Test 4: Modell erstellen (zeitbasierte Vorhersage)"""
    print_section("4. Modell erstellen (zeitbasierte Vorhersage)")
    
    # Berechne Zeiträume (letzte 24h für Training)
    now = datetime.utcnow()
    train_end = now - timedelta(hours=1)
    train_start = train_end - timedelta(hours=24)
    
    request_data = {
        "name": f"API_Test_Model_{int(time.time())}",
        "model_type": "random_forest",
        "train_start": train_start.isoformat() + "Z",
        "train_end": train_end.isoformat() + "Z",
        "features": ["price_open", "price_high", "price_low", "volume_sol", "buy_volume_sol", "sell_volume_sol"],
        "target_var": "price_close",
        "operator": None,  # Zeitbasierte Vorhersage
        "target_value": None,
        "phases": None,
        "use_time_based_prediction": True,
        "future_minutes": 5,
        "min_percent_change": 30,
        "direction": "up",
        "use_engineered_features": True,
        "feature_engineering_windows": [5, 10, 15],
        "n_estimators": 50,  # Reduziert für schnelleren Test
        "max_depth": 10,
        "cv_splits": 3
    }
    
    try:
        response = httpx.post(
            f"{API_BASE}/models/create",
            json=request_data,
            timeout=30
        )
        
        if response.status_code in [200, 201]:
            job = response.json()
            job_id = job.get('job_id')  # API gibt 'job_id' zurück, nicht 'id'
            print_result(True, f"Train-Job erstellt: ID {job_id}")
            print(f"   Message: {job.get('message')}")
            print(f"   Status: {job.get('status')}")
            return job_id
        else:
            error_data = response.text
            print_result(False, f"Modell erstellen fehlgeschlagen: {response.status_code}")
            print(f"   Response: {error_data[:200]}")
            return None
    except Exception as e:
        print_result(False, f"Modell erstellen Exception: {e}")
        return None

def wait_for_job(job_id: int, max_wait: int = 300):
    """Wartet auf Job-Abschluss"""
    print(f"\n⏳ Warte auf Job {job_id} (max. {max_wait}s)...")
    start_time = time.time()
    
    while time.time() - start_time < max_wait:
        try:
            response = httpx.get(f"{API_BASE}/queue/{job_id}", timeout=10)
            if response.status_code == 200:
                job = response.json()
                status = job.get('status')
                progress = job.get('progress', 0)
                
                if status == 'COMPLETED':
                    print(f"✅ Job {job_id} abgeschlossen!")
                    return job
                elif status == 'FAILED':
                    error_msg = job.get('error_msg', 'Unbekannter Fehler')
                    print(f"❌ Job {job_id} fehlgeschlagen: {error_msg}")
                    return job
                else:
                    print(f"   Status: {status}, Progress: {progress}%", end='\r')
                    time.sleep(2)
            else:
                print(f"   ⚠️ Job-Abfrage fehlgeschlagen: {response.status_code}")
                time.sleep(2)
        except Exception as e:
            print(f"   ⚠️ Exception: {e}")
            time.sleep(2)
    
    print(f"\n⏰ Timeout nach {max_wait}s")
    return None

def test_get_model(model_id: int):
    """Test 5: Modell-Details abrufen"""
    print_section(f"5. Modell-Details abrufen (ID: {model_id})")
    try:
        response = httpx.get(f"{API_BASE}/models/{model_id}", timeout=10)
        if response.status_code == 200:
            model = response.json()
            print_result(True, f"Modell gefunden: {model.get('name')}")
            print(f"   Type: {model.get('model_type')}")
            print(f"   Status: {model.get('status')}")
            print(f"   Features: {len(model.get('features', []))}")
            if model.get('training_accuracy'):
                print(f"   Accuracy: {model.get('training_accuracy'):.4f}")
            return model
        else:
            print_result(False, f"Modell abrufen fehlgeschlagen: {response.status_code}")
            return None
    except Exception as e:
        print_result(False, f"Modell abrufen Exception: {e}")
        return None

def test_test_model(model_id: int):
    """Test 6: Modell testen"""
    print_section(f"6. Modell testen (ID: {model_id})")
    
    # Test-Zeitraum: 1 Tag nach Training
    now = datetime.utcnow()
    test_end = now - timedelta(minutes=30)
    test_start = test_end - timedelta(hours=12)
    
    request_data = {
        "test_start": test_start.isoformat() + "Z",
        "test_end": test_end.isoformat() + "Z"
    }
    
    try:
        response = httpx.post(
            f"{API_BASE}/models/{model_id}/test",
            json=request_data,
            timeout=30
        )
        
        if response.status_code in [200, 201]:
            job = response.json()
            job_id = job.get('job_id')  # API gibt 'job_id' zurück
            print_result(True, f"Test-Job erstellt: ID {job_id}")
            print(f"   Message: {job.get('message')}")
            return job_id
        else:
            error_data = response.text
            print_result(False, f"Modell testen fehlgeschlagen: {response.status_code}")
            print(f"   Response: {error_data[:200]}")
            return None
    except Exception as e:
        print_result(False, f"Modell testen Exception: {e}")
        return None

def test_list_test_results(model_id: int):
    """Test 7: Test-Ergebnisse auflisten"""
    print_section(f"7. Test-Ergebnisse auflisten (Model ID: {model_id})")
    try:
        response = httpx.get(f"{API_BASE}/models/{model_id}/tests", timeout=10)
        if response.status_code == 200:
            results = response.json()
            print_result(True, f"Test-Ergebnisse gefunden: {len(results)}")
            if results:
                latest = results[0]
                print(f"   Neuestes Ergebnis: ID {latest.get('id')}")
                if latest.get('accuracy'):
                    print(f"   Accuracy: {latest.get('accuracy'):.4f}")
            return results
        else:
            print_result(False, f"Test-Ergebnisse auflisten fehlgeschlagen: {response.status_code}")
            return []
    except Exception as e:
        print_result(False, f"Test-Ergebnisse auflisten Exception: {e}")
        return []

def test_list_phases():
    """Test 8: Phasen auflisten"""
    print_section("8. Phasen auflisten")
    try:
        response = httpx.get(f"{API_BASE}/phases", timeout=10)
        if response.status_code == 200:
            phases = response.json()
            print_result(True, f"Phasen gefunden: {len(phases)}")
            for phase in phases:
                print(f"   - {phase.get('name')} (ID: {phase.get('id')}, Interval: {phase.get('interval_seconds')}s)")
            return phases
        else:
            print_result(False, f"Phasen auflisten fehlgeschlagen: {response.status_code}")
            return []
    except Exception as e:
        print_result(False, f"Phasen auflisten Exception: {e}")
        return []

def main():
    """Hauptfunktion: Führt alle Tests aus"""
    print("\n" + "="*60)
    print("  VOLLSTÄNDIGER API-TEST")
    print("  Nach Datenbank-Umstellung und Refactoring")
    print("="*60)
    
    results = {
        "health": False,
        "list_models": False,
        "list_jobs": False,
        "create_model": False,
        "get_model": False,
        "test_model": False,
        "list_test_results": False,
        "list_phases": False
    }
    
    # Test 1: Health Check
    results["health"] = test_health()
    if not results["health"]:
        print("\n❌ Health Check fehlgeschlagen - beende Tests")
        return
    
    # Test 2: Modelle auflisten
    models = test_list_models()
    results["list_models"] = models is not None
    
    # Test 3: Jobs auflisten
    jobs = test_list_jobs()
    results["list_jobs"] = jobs is not None
    
    # Test 4: Modell erstellen
    train_job_id = test_create_model()
    results["create_model"] = train_job_id is not None
    
    if train_job_id:
        # Warte auf Train-Job
        train_job = wait_for_job(train_job_id, max_wait=300)
        
        if train_job and train_job.get('status') == 'COMPLETED':
            model_id = train_job.get('result_model_id')
            
            if model_id:
                # Test 5: Modell-Details abrufen
                model = test_get_model(model_id)
                results["get_model"] = model is not None
                
                if model:
                    # Test 6: Modell testen
                    test_job_id = test_test_model(model_id)
                    results["test_model"] = test_job_id is not None
                    
                    if test_job_id:
                        # Warte auf Test-Job
                        test_job = wait_for_job(test_job_id, max_wait=300)
                        
                        if test_job:
                            # Test 7: Test-Ergebnisse auflisten
                            test_results = test_list_test_results(model_id)
                            results["list_test_results"] = test_results is not None
    
    # Test 8: Phasen auflisten
    phases = test_list_phases()
    results["list_phases"] = phases is not None
    
    # Zusammenfassung
    print_section("ZUSAMMENFASSUNG")
    total = len(results)
    passed = sum(1 for v in results.values() if v)
    
    for test_name, success in results.items():
        status = "✅" if success else "❌"
        print(f"{status} {test_name}")
    
    print(f"\n📊 Ergebnis: {passed}/{total} Tests erfolgreich")
    
    if passed == total:
        print("\n🎉 ALLE TESTS ERFOLGREICH!")
    else:
        print(f"\n⚠️ {total - passed} Test(s) fehlgeschlagen")

if __name__ == "__main__":
    main()

