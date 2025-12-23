#!/usr/bin/env python3
"""Test mit echten Daten aus der Datenbank"""
import requests
import time
import json
from datetime import datetime

API_BASE = "http://localhost:8000/api"

def wait_for_job(job_id, max_wait=300):
    """Warte auf Job-Abschluss"""
    waited = 0
    while waited < max_wait:
        time.sleep(5)
        waited += 5
        try:
            response = requests.get(f"{API_BASE}/jobs/{job_id}")
            if response.status_code == 200:
                job_data = response.json()
                status = job_data.get('status')
                progress = job_data.get('progress', 0) * 100
                msg = job_data.get('progress_msg', '')[:50]
                print(f"   [{waited}s] Status: {status}, Progress: {progress:.1f}% - {msg}")
                
                if status == 'COMPLETED':
                    return job_data
                elif status == 'FAILED':
                    error = job_data.get('error_msg', 'Unbekannter Fehler')
                    raise Exception(f"Job fehlgeschlagen: {error}")
        except Exception as e:
            print(f"   ⚠️ Fehler: {e}")
    
    raise Exception(f"Timeout nach {max_wait} Sekunden")

# 1. Erstelle Modell
print("🚀 Erstelle Modell mit echten Daten...")
model_request = {
    "name": f"Test_RealData_{int(datetime.now().timestamp())}",
    "model_type": "random_forest",
    "target_var": "price_close",
    "operator": ">",
    "target_value": 0.0001,
    "train_start": "2025-12-21T19:42:29Z",
    "train_end": "2025-12-23T09:58:35Z",
    "features": ["price_open", "price_high", "volume_sol"],
    "phases": None,
    "use_time_based_prediction": False,
    "use_engineered_features": False,
    "use_smote": False,
    "use_timeseries_split": False
}

print(f"   Name: {model_request['name']}")
print(f"   Train: {model_request['train_start']} bis {model_request['train_end']}")
print(f"   Features: {model_request['features']}")

response = requests.post(f"{API_BASE}/models/create", json=model_request)
if response.status_code not in [200, 201]:
    print(f"❌ Fehler: {response.status_code}")
    print(response.text)
    exit(1)

job = response.json()
job_id = job['job_id']
print(f"\n✅ Job erstellt: ID {job_id}")

# 2. Warte auf Training
print(f"\n⏳ Warte auf Training (max. 5 Minuten)...")
try:
    job_data = wait_for_job(job_id)
    model_id = job_data.get('result_model_id')
    if not model_id:
        print("❌ Keine Modell-ID im Job gefunden")
        exit(1)
    print(f"\n✅ Training abgeschlossen! Modell-ID: {model_id}")
except Exception as e:
    print(f"\n❌ {e}")
    exit(1)

# 3. Teste Modell
print(f"\n🧪 Teste Modell {model_id}...")
test_request = {
    "test_start": "2025-12-23T09:58:35Z",
    "test_end": "2025-12-23T19:32:36Z"
}
print(f"   Test-Zeitraum: {test_request['test_start']} bis {test_request['test_end']}")

response = requests.post(f"{API_BASE}/models/{model_id}/test", json=test_request)
if response.status_code not in [200, 201]:
    print(f"❌ Fehler: {response.status_code}")
    print(response.text)
    exit(1)

test_job = response.json()
test_job_id = test_job['job_id']
print(f"\n✅ Test-Job erstellt: ID {test_job_id}")

# 4. Warte auf Test
print(f"\n⏳ Warte auf Test (max. 5 Minuten)...")
try:
    test_job_data = wait_for_job(test_job_id)
    test_result = test_job_data.get('result_test')
    if test_result:
        print(f"\n✅ Test abgeschlossen!")
        print(f"\n📊 Ergebnisse:")
        if test_result.get('accuracy') is not None:
            print(f"   Accuracy: {test_result.get('accuracy'):.4f}")
        if test_result.get('f1_score') is not None:
            print(f"   F1: {test_result.get('f1_score'):.4f}")
        if test_result.get('mcc') is not None:
            print(f"   MCC: {test_result.get('mcc'):.4f}")
        if test_result.get('simulated_profit_pct') is not None:
            print(f"   Profit: {test_result.get('simulated_profit_pct'):.4f}%")
        if test_result.get('train_accuracy') is not None:
            print(f"\n📊 Train vs. Test:")
            print(f"   Train Accuracy: {test_result.get('train_accuracy'):.4f}")
            print(f"   Test Accuracy: {test_result.get('accuracy'):.4f}")
            if test_result.get('accuracy_degradation') is not None:
                print(f"   Degradation: {test_result.get('accuracy_degradation'):.4f}")
            print(f"   Overfitted: {test_result.get('is_overfitted', False)}")
    else:
        print("⚠️ Keine Test-Ergebnisse gefunden")
except Exception as e:
    print(f"\n❌ {e}")
    exit(1)

print(f"\n✅ Alle Tests erfolgreich abgeschlossen!")

