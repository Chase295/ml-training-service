"""
Integrationstest für Verbesserung 1.1: Data Leakage beheben

Testet die vollständige Modell-Erstellung über die API.
"""
import requests
import time
import json
from datetime import datetime, timezone, timedelta

API_BASE = "http://localhost:8000"

def test_data_leakage_integration():
    """Test ob Data Leakage Fix über API funktioniert"""
    
    print("=" * 60)
    print("🧪 Integrationstest: Data Leakage Fix (Verbesserung 1.1)")
    print("=" * 60)
    
    # Test-Parameter
    train_end = datetime.now(timezone.utc)
    train_start = train_end - timedelta(days=7)
    
    # Test 1: Zeitbasierte Vorhersage über API
    print("\n📋 Test 1: Zeitbasierte Vorhersage über API")
    print("-" * 60)
    
    model_name = f"Test_DataLeakageFix_{int(time.time() * 1000)}"
    
    request_data = {
        "name": model_name,
        "model_type": "random_forest",
        "features": ["price_open", "price_high", "volume_sol"],
        "phases": None,
        "train_start": train_start.isoformat().replace('+00:00', 'Z'),
        "train_end": train_end.isoformat().replace('+00:00', 'Z'),
        "use_time_based_prediction": True,
        "future_minutes": 10,
        "min_percent_change": 5.0,
        "direction": "up",
        "target_var": "price_close"
    }
    
    try:
        # Erstelle Modell-Job
        response = requests.post(f"{API_BASE}/api/models/create", json=request_data, timeout=10)
        response.raise_for_status()
        job_data = response.json()
        job_id = job_data.get('job_id')
        
        print(f"✅ Job erstellt: {job_id}")
        print(f"   Status: {job_data.get('status')}")
        
        # Warte auf Job-Abschluss (max 5 Minuten)
        print("\n⏳ Warte auf Job-Abschluss...")
        max_wait = 300  # 5 Minuten
        start_time = time.time()
        
        while time.time() - start_time < max_wait:
            time.sleep(5)
            job_response = requests.get(f"{API_BASE}/api/jobs/{job_id}", timeout=10)
            job_response.raise_for_status()
            job = job_response.json()
            
            status = job.get('status')
            progress = job.get('progress', 0)
            
            print(f"   Status: {status}, Progress: {progress:.1%}")
            
            if status == "COMPLETED":
                print("\n✅ Job erfolgreich abgeschlossen!")
                
                # Prüfe Modell-Details
                if job.get('result_model_id'):
                    model_id = job['result_model_id']
                    model_response = requests.get(f"{API_BASE}/api/models/{model_id}", timeout=10)
                    model_response.raise_for_status()
                    model = model_response.json()
                    
                    print(f"\n📊 Modell-Details:")
                    print(f"   ID: {model.get('id')}")
                    print(f"   Name: {model.get('name')}")
                    print(f"   Features: {len(model.get('features', []))}")
                    print(f"   Accuracy: {model.get('training_accuracy', 'N/A')}")
                    
                    # Prüfe ob target_var NICHT in Features ist
                    features = model.get('features', [])
                    if 'price_close' not in features:
                        print("   ✅ Korrekt: 'price_close' ist NICHT in Features (Data Leakage verhindert)")
                    else:
                        print("   ❌ FEHLER: 'price_close' ist in Features (Data Leakage vorhanden!)")
                        return False
                    
                    return True
                else:
                    print("   ⚠️ Keine result_model_id gefunden")
                    return False
            
            elif status == "FAILED":
                error_msg = job.get('error_msg', 'Unbekannter Fehler')
                print(f"\n❌ Job fehlgeschlagen: {error_msg}")
                return False
        
        print(f"\n⏰ Timeout: Job nicht innerhalb von {max_wait}s abgeschlossen")
        return False
        
    except requests.exceptions.RequestException as e:
        print(f"❌ API-Fehler: {e}")
        return False
    except Exception as e:
        print(f"❌ Unerwarteter Fehler: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_data_leakage_integration()
    exit(0 if success else 1)

