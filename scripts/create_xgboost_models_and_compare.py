#!/usr/bin/env python3
"""
Erstellt 2 XGBoost-Modelle mit optimalen Einstellungen und startet einen Vergleich
mit den letzten 10 Minuten Daten.
"""

import requests
import json
import time
from datetime import datetime, timezone, timedelta

API_BASE_URL = "http://localhost:8012"

def wait_for_api(max_wait=30):
    """Warte bis API erreichbar ist"""
    for i in range(max_wait):
        try:
            response = requests.get(f"{API_BASE_URL}/api/health", timeout=2)
            if response.status_code == 200:
                return True
        except:
            pass
        time.sleep(1)
    return False

def wait_for_training(job1_id, job2_id, max_wait=600):
    """Warte bis beide Modelle fertig trainiert sind"""
    print(f"\n⏳ Warte auf Training-Abschluss...")
    print(f"   Job 1: {job1_id}")
    print(f"   Job 2: {job2_id}\n")
    
    start_time = time.time()
    check_interval = 5
    
    while time.time() - start_time < max_wait:
        job1 = requests.get(f"{API_BASE_URL}/api/queue/{job1_id}").json()
        job2 = requests.get(f"{API_BASE_URL}/api/queue/{job2_id}").json()
        
        job1_status = job1.get('status')
        job1_progress = job1.get('progress', 0) * 100
        job2_status = job2.get('status')
        job2_progress = job2.get('progress', 0) * 100
        
        print(f"   Job 1: {job1_status} ({job1_progress:.1f}%) | Job 2: {job2_status} ({job2_progress:.1f}%)")
        
        if job1_status == "COMPLETED" and job2_status == "COMPLETED":
            # Hole Modell-IDs
            model1_id = job1.get('result_model', {}).get('id') if job1.get('result_model') else None
            model2_id = job2.get('result_model', {}).get('id') if job2.get('result_model') else None
            
            if model1_id and model2_id:
                # Prüfe ob Modelle READY sind
                model1 = requests.get(f"{API_BASE_URL}/api/models/{model1_id}").json()
                model2 = requests.get(f"{API_BASE_URL}/api/models/{model2_id}").json()
                
                if model1.get('status') == 'READY' and model2.get('status') == 'READY':
                    print("\n✅ Beide Modelle sind READY!")
                    return model1_id, model2_id
                else:
                    print(f"   ⏳ Modell 1: {model1.get('status')}, Modell 2: {model2.get('status')}")
            else:
                print("   ⚠️ Konnte Modell-IDs nicht extrahieren")
        elif job1_status == "FAILED" or job2_status == "FAILED":
            print(f"\n❌ Fehler: Job 1: {job1_status}, Job 2: {job2_status}")
            if job1.get('error_msg'):
                print(f"   Job 1 Fehler: {job1.get('error_msg')}")
            if job2.get('error_msg'):
                print(f"   Job 2 Fehler: {job2.get('error_msg')}")
            return None, None
        
        time.sleep(check_interval)
    
    print("\n⏰ Timeout erreicht!")
    return None, None

def main():
    print("🚀 XGBoost Modell-Erstellung mit optimalen Einstellungen\n")
    
    # Prüfe API-Verfügbarkeit
    print("📡 Prüfe API-Verfügbarkeit...")
    if not wait_for_api():
        print("❌ API nicht erreichbar! Bitte Container starten: docker compose up -d")
        return
    print("✅ API erreichbar\n")
    
    # Daten-Verfügbarkeit
    print("📊 Lade Daten-Verfügbarkeit...")
    data_avail = requests.get(f"{API_BASE_URL}/api/data-availability").json()
    min_ts = datetime.fromisoformat(data_avail['min_timestamp'].replace('Z', '+00:00'))
    max_ts = datetime.fromisoformat(data_avail['max_timestamp'].replace('Z', '+00:00'))
    
    print(f"   Verfügbare Daten: {min_ts} bis {max_ts}")
    print(f"   Dauer: {(max_ts - min_ts).total_seconds() / 3600:.2f} Stunden\n")
    
    # Training: Von Anfang bis 10 Minuten vor Ende
    train_end = max_ts - timedelta(minutes=10)
    train_start = min_ts
    
    # Test: Letzte 10 Minuten
    test_start = max_ts - timedelta(minutes=10)
    test_end = max_ts
    
    print(f"📅 Training-Zeitraum: {train_start} bis {train_end}")
    print(f"   Dauer: {(train_end - train_start).total_seconds() / 3600:.2f} Stunden")
    print(f"📅 Test-Zeitraum: {test_start} bis {test_end}")
    print(f"   Dauer: {(test_end - test_start).total_seconds() / 60:.2f} Minuten\n")
    
    # ISO-Format
    train_start_iso = train_start.astimezone(timezone.utc).isoformat().replace('+00:00', 'Z')
    train_end_iso = train_end.astimezone(timezone.utc).isoformat().replace('+00:00', 'Z')
    test_start_iso = test_start.astimezone(timezone.utc).isoformat().replace('+00:00', 'Z')
    test_end_iso = test_end.astimezone(timezone.utc).isoformat().replace('+00:00', 'Z')
    
    # Alle kritischen Features
    all_features = [
        "price_open", "price_high", "price_low", "price_close",
        "volume_sol", "buy_volume_sol", "sell_volume_sol", "net_volume_sol",
        "market_cap_close", "phase_id_at_time",
        "dev_sold_amount",
        "buy_pressure_ratio", "unique_signer_ratio",
        "whale_buy_volume_sol", "whale_sell_volume_sol", "num_whale_buys", "num_whale_sells",
        "volatility_pct", "avg_trade_size_sol"
    ]
    
    # Modell 1: Konservativ (bessere Generalisierung)
    model1_data = {
        "name": f"XGBoost_Optimal_Conservative_{int(datetime.now().timestamp())}",
        "model_type": "xgboost",
        "target_var": "price_close",
        "operator": None,
        "target_value": None,
        "features": all_features,
        "phases": None,
        "params": {
            "n_estimators": 200,
            "max_depth": 6,
            "learning_rate": 0.05
        },
        "train_start": train_start_iso,
        "train_end": train_end_iso,
        "use_time_based_prediction": True,
        "future_minutes": 10,
        "min_percent_change": 5.0,
        "direction": "up",
        "use_engineered_features": True,
        "feature_engineering_windows": [5, 10, 15],
        "use_smote": True,
        "use_timeseries_split": True,
        "cv_splits": 5,
        "use_market_context": True
    }
    
    print("🚀 Erstelle Modell 1 (Konservativ: n_estimators=200, max_depth=6, lr=0.05)...")
    response1 = requests.post(f"{API_BASE_URL}/api/models/create", json=model1_data, timeout=30)
    if response1.status_code == 201:
        result1 = response1.json()
        job1_id = result1.get('job_id')
        print(f"✅ Modell 1 erstellt! Job-ID: {job1_id}")
    else:
        print(f"❌ Fehler: {response1.status_code} - {response1.text}")
        return
    
    # Modell 2: Aggressiver (höhere Performance)
    model2_data = {
        "name": f"XGBoost_Optimal_Aggressive_{int(datetime.now().timestamp())}",
        "model_type": "xgboost",
        "target_var": "price_close",
        "operator": None,
        "target_value": None,
        "features": all_features,
        "phases": None,
        "params": {
            "n_estimators": 300,
            "max_depth": 8,
            "learning_rate": 0.1
        },
        "train_start": train_start_iso,
        "train_end": train_end_iso,
        "use_time_based_prediction": True,
        "future_minutes": 10,
        "min_percent_change": 5.0,
        "direction": "up",
        "use_engineered_features": True,
        "feature_engineering_windows": [5, 10, 15],
        "use_smote": True,
        "use_timeseries_split": True,
        "cv_splits": 5,
        "use_market_context": True
    }
    
    print("\n🚀 Erstelle Modell 2 (Aggressiver: n_estimators=300, max_depth=8, lr=0.1)...")
    response2 = requests.post(f"{API_BASE_URL}/api/models/create", json=model2_data, timeout=30)
    if response2.status_code == 201:
        result2 = response2.json()
        job2_id = result2.get('job_id')
        print(f"✅ Modell 2 erstellt! Job-ID: {job2_id}")
    else:
        print(f"❌ Fehler: {response2.status_code} - {response2.text}")
        return
    
    # Warte auf Training-Abschluss
    model1_id, model2_id = wait_for_training(job1_id, job2_id)
    
    if not model1_id or not model2_id:
        print("\n❌ Training fehlgeschlagen oder Timeout erreicht!")
        return
    
    print(f"\n📊 Modell-IDs:")
    print(f"   Modell 1: {model1_id}")
    print(f"   Modell 2: {model2_id}\n")
    
    # Starte Vergleich
    print("⚔️ Starte Vergleich mit den letzten 10 Minuten Daten...")
    compare_data = {
        "model_a_id": model1_id,
        "model_b_id": model2_id,
        "test_start": test_start_iso,
        "test_end": test_end_iso
    }
    
    response = requests.post(f"{API_BASE_URL}/api/models/compare", json=compare_data, timeout=30)
    if response.status_code == 201:
        result = response.json()
        compare_job_id = result.get('job_id')
        print(f"✅ Vergleichs-Job erstellt! Job-ID: {compare_job_id}")
        print(f"\n📊 Du kannst den Vergleich in der Web UI unter '⚖️ Vergleichs-Übersicht' sehen.")
        print(f"   URL: http://localhost:8502")
    else:
        print(f"❌ Fehler beim Erstellen des Vergleichs: {response.status_code} - {response.text}")

if __name__ == "__main__":
    main()

