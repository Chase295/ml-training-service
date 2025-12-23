#!/usr/bin/env python3
"""Test XGBoost und Random Forest Modelle für Pump-Detection"""
import requests
import time
import json
from datetime import datetime

API_BASE = "http://localhost:8000/api"

def wait_for_job(job_id, max_wait=600):
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

# Gemeinsame Parameter für beide Modelle
base_params = {
    "target_var": "price_close",
    "train_start": "2025-12-21T19:42:29Z",
    "train_end": "2025-12-23T09:58:35Z",
    "features": ["price_open", "price_high", "price_low", "volume_sol", "buy_volume_sol", "sell_volume_sol"],
    "phases": None,
    "use_time_based_prediction": True,
    "future_minutes": 5,
    "min_percent_change": 30.0,
    "direction": "up",
    "use_engineered_features": True,
    "feature_engineering_windows": [5, 10, 15],
    "use_smote": True,
    "use_timeseries_split": True,
    "cv_splits": 5
}

test_params = {
    "test_start": "2025-12-23T09:58:35Z",
    "test_end": "2025-12-23T19:32:36Z"
}

# 1. XGBoost Modell
print("=" * 60)
print("🚀 1. Erstelle XGBoost Modell für Pump-Detection")
print("=" * 60)
model_xgb = {
    **base_params,
    "name": f"Pump_XGBoost_5min_30pct_{int(datetime.now().timestamp())}",
    "model_type": "xgboost"
}

print(f"   Name: {model_xgb['name']}")
print(f"   Type: XGBoost")
print(f"   Future Minutes: {model_xgb['future_minutes']}")
print(f"   Min. Prozent-Änderung: {model_xgb['min_percent_change']}%")

response = requests.post(f"{API_BASE}/models/create", json=model_xgb)
if response.status_code not in [200, 201]:
    print(f"❌ Fehler: {response.status_code}")
    print(response.text)
    exit(1)

job_xgb = response.json()
job_xgb_id = job_xgb['job_id']
print(f"\n✅ XGBoost Job erstellt: ID {job_xgb_id}")

# Warte auf XGBoost Training
print(f"\n⏳ Warte auf XGBoost Training (max. 10 Minuten)...")
try:
    job_data = wait_for_job(job_xgb_id, max_wait=600)
    model_xgb_id = job_data.get('result_model_id')
    if not model_xgb_id:
        print("❌ Keine Modell-ID im Job gefunden")
        exit(1)
    print(f"\n✅ XGBoost Training abgeschlossen! Modell-ID: {model_xgb_id}")
    
    # Hole Modell-Details
    model_response = requests.get(f"{API_BASE}/models/{model_xgb_id}")
    if model_response.status_code == 200:
        model = model_response.json()
        print(f"\n📊 XGBoost Modell-Details:")
        print(f"   Name: {model.get('name')}")
        print(f"   Training Accuracy: {model.get('training_accuracy', 'N/A'):.4f}" if model.get('training_accuracy') else "   Training Accuracy: N/A")
        print(f"   Training F1: {model.get('training_f1', 'N/A'):.4f}" if model.get('training_f1') else "   Training F1: N/A")
        if model.get('mcc'):
            print(f"   MCC: {model.get('mcc'):.4f}")
except Exception as e:
    print(f"\n❌ {e}")
    exit(1)

# Teste XGBoost
print(f"\n🧪 Teste XGBoost Modell {model_xgb_id}...")
response = requests.post(f"{API_BASE}/models/{model_xgb_id}/test", json=test_params)
if response.status_code not in [200, 201]:
    print(f"❌ Fehler: {response.status_code}")
    print(response.text)
    exit(1)

test_job_xgb = response.json()
test_job_xgb_id = test_job_xgb['job_id']
print(f"✅ XGBoost Test-Job erstellt: ID {test_job_xgb_id}")

print(f"\n⏳ Warte auf XGBoost Test (max. 5 Minuten)...")
try:
    test_job_data = wait_for_job(test_job_xgb_id, max_wait=300)
    test_result_xgb = test_job_data.get('result_test')
    if test_result_xgb:
        print(f"\n✅ XGBoost Test abgeschlossen!")
        print(f"\n📊 XGBoost Test-Ergebnisse:")
        if test_result_xgb.get('accuracy'):
            print(f"   Accuracy: {test_result_xgb.get('accuracy'):.4f}")
        if test_result_xgb.get('f1_score'):
            print(f"   F1: {test_result_xgb.get('f1_score'):.4f}")
        if test_result_xgb.get('mcc'):
            print(f"   MCC: {test_result_xgb.get('mcc'):.4f}")
        if test_result_xgb.get('simulated_profit_pct'):
            print(f"   Profit: {test_result_xgb.get('simulated_profit_pct'):.4f}%")
        if test_result_xgb.get('confusion_matrix'):
            cm = test_result_xgb['confusion_matrix']
            print(f"   Confusion Matrix: TP={cm.get('tp', 0)}, TN={cm.get('tn', 0)}, FP={cm.get('fp', 0)}, FN={cm.get('fn', 0)}")
except Exception as e:
    print(f"\n❌ {e}")
    exit(1)

# 2. Random Forest Modell
print("\n" + "=" * 60)
print("🚀 2. Erstelle Random Forest Modell für Pump-Detection")
print("=" * 60)
model_rf = {
    **base_params,
    "name": f"Pump_RandomForest_5min_30pct_{int(datetime.now().timestamp())}",
    "model_type": "random_forest"
}

print(f"   Name: {model_rf['name']}")
print(f"   Type: Random Forest")
print(f"   Future Minutes: {model_rf['future_minutes']}")
print(f"   Min. Prozent-Änderung: {model_rf['min_percent_change']}%")

response = requests.post(f"{API_BASE}/models/create", json=model_rf)
if response.status_code not in [200, 201]:
    print(f"❌ Fehler: {response.status_code}")
    print(response.text)
    exit(1)

job_rf = response.json()
job_rf_id = job_rf['job_id']
print(f"\n✅ Random Forest Job erstellt: ID {job_rf_id}")

# Warte auf Random Forest Training
print(f"\n⏳ Warte auf Random Forest Training (max. 10 Minuten)...")
try:
    job_data = wait_for_job(job_rf_id, max_wait=600)
    model_rf_id = job_data.get('result_model_id')
    if not model_rf_id:
        print("❌ Keine Modell-ID im Job gefunden")
        exit(1)
    print(f"\n✅ Random Forest Training abgeschlossen! Modell-ID: {model_rf_id}")
    
    # Hole Modell-Details
    model_response = requests.get(f"{API_BASE}/models/{model_rf_id}")
    if model_response.status_code == 200:
        model = model_response.json()
        print(f"\n📊 Random Forest Modell-Details:")
        print(f"   Name: {model.get('name')}")
        print(f"   Training Accuracy: {model.get('training_accuracy', 'N/A'):.4f}" if model.get('training_accuracy') else "   Training Accuracy: N/A")
        print(f"   Training F1: {model.get('training_f1', 'N/A'):.4f}" if model.get('training_f1') else "   Training F1: N/A")
        if model.get('mcc'):
            print(f"   MCC: {model.get('mcc'):.4f}")
except Exception as e:
    print(f"\n❌ {e}")
    exit(1)

# Teste Random Forest
print(f"\n🧪 Teste Random Forest Modell {model_rf_id}...")
response = requests.post(f"{API_BASE}/models/{model_rf_id}/test", json=test_params)
if response.status_code not in [200, 201]:
    print(f"❌ Fehler: {response.status_code}")
    print(response.text)
    exit(1)

test_job_rf = response.json()
test_job_rf_id = test_job_rf['job_id']
print(f"✅ Random Forest Test-Job erstellt: ID {test_job_rf_id}")

print(f"\n⏳ Warte auf Random Forest Test (max. 5 Minuten)...")
try:
    test_job_data = wait_for_job(test_job_rf_id, max_wait=300)
    test_result_rf = test_job_data.get('result_test')
    if test_result_rf:
        print(f"\n✅ Random Forest Test abgeschlossen!")
        print(f"\n📊 Random Forest Test-Ergebnisse:")
        if test_result_rf.get('accuracy'):
            print(f"   Accuracy: {test_result_rf.get('accuracy'):.4f}")
        if test_result_rf.get('f1_score'):
            print(f"   F1: {test_result_rf.get('f1_score'):.4f}")
        if test_result_rf.get('mcc'):
            print(f"   MCC: {test_result_rf.get('mcc'):.4f}")
        if test_result_rf.get('simulated_profit_pct'):
            print(f"   Profit: {test_result_rf.get('simulated_profit_pct'):.4f}%")
        if test_result_rf.get('confusion_matrix'):
            cm = test_result_rf['confusion_matrix']
            print(f"   Confusion Matrix: TP={cm.get('tp', 0)}, TN={cm.get('tn', 0)}, FP={cm.get('fp', 0)}, FN={cm.get('fn', 0)}")
except Exception as e:
    print(f"\n❌ {e}")
    exit(1)

# 3. Vergleich
print("\n" + "=" * 60)
print("📊 VERGLEICH: XGBoost vs. Random Forest")
print("=" * 60)

if test_result_xgb and test_result_rf:
    print(f"\n📈 Accuracy:")
    print(f"   XGBoost:      {test_result_xgb.get('accuracy', 'N/A'):.4f}" if test_result_xgb.get('accuracy') else "   XGBoost:      N/A")
    print(f"   Random Forest: {test_result_rf.get('accuracy', 'N/A'):.4f}" if test_result_rf.get('accuracy') else "   Random Forest: N/A")
    
    print(f"\n📈 F1-Score:")
    print(f"   XGBoost:      {test_result_xgb.get('f1_score', 'N/A'):.4f}" if test_result_xgb.get('f1_score') else "   XGBoost:      N/A")
    print(f"   Random Forest: {test_result_rf.get('f1_score', 'N/A'):.4f}" if test_result_rf.get('f1_score') else "   Random Forest: N/A")
    
    print(f"\n📈 MCC:")
    print(f"   XGBoost:      {test_result_xgb.get('mcc', 'N/A'):.4f}" if test_result_xgb.get('mcc') else "   XGBoost:      N/A")
    print(f"   Random Forest: {test_result_rf.get('mcc', 'N/A'):.4f}" if test_result_rf.get('mcc') else "   Random Forest: N/A")
    
    print(f"\n📈 Simulierter Profit:")
    print(f"   XGBoost:      {test_result_xgb.get('simulated_profit_pct', 'N/A'):.4f}%" if test_result_xgb.get('simulated_profit_pct') else "   XGBoost:      N/A")
    print(f"   Random Forest: {test_result_rf.get('simulated_profit_pct', 'N/A'):.4f}%" if test_result_rf.get('simulated_profit_pct') else "   Random Forest: N/A")
    
    # Bestes Modell
    if test_result_xgb.get('accuracy') and test_result_rf.get('accuracy'):
        if test_result_xgb.get('accuracy') > test_result_rf.get('accuracy'):
            print(f"\n🏆 XGBoost ist besser (Accuracy: {test_result_xgb.get('accuracy'):.4f} vs. {test_result_rf.get('accuracy'):.4f})")
        else:
            print(f"\n🏆 Random Forest ist besser (Accuracy: {test_result_rf.get('accuracy'):.4f} vs. {test_result_xgb.get('accuracy'):.4f})")

print(f"\n✅ Alle Tests erfolgreich abgeschlossen!")
print(f"   XGBoost Modell-ID: {model_xgb_id}")
print(f"   Random Forest Modell-ID: {model_rf_id}")

