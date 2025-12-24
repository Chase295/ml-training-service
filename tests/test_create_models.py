#!/usr/bin/env python3
"""
Test-Script: Erstellt 2 Test-Modelle
"""

import requests
import json
from datetime import datetime, timedelta, timezone
from dateutil import parser

API_BASE_URL = "http://100.76.209.59:8005/api"

def get_data_availability():
    """Holt verfügbare Daten-Zeitraum"""
    try:
        response = requests.get(f"{API_BASE_URL}/data-availability", timeout=10)
        if response.status_code == 200:
            data = response.json()
            return data.get('min_timestamp'), data.get('max_timestamp')
        return None, None
    except Exception as e:
        print(f"❌ Fehler beim Abrufen der Daten-Verfügbarkeit: {e}")
        return None, None

def get_phases():
    """Holt verfügbare Phasen"""
    try:
        response = requests.get(f"{API_BASE_URL}/phases", timeout=10)
        if response.status_code == 200:
            phases = response.json()
            if isinstance(phases, list) and phases:
                # Verwende erste 2 Phasen
                return [phases[0].get('id'), phases[1].get('id')] if len(phases) >= 2 else [phases[0].get('id')]
        return [1, 2]  # Fallback
    except Exception as e:
        print(f"⚠️  Fehler beim Abrufen der Phasen: {e}")
        return [1, 2]  # Fallback

def create_model(model_config):
    """Erstellt ein Modell"""
    print(f"\n📦 Erstelle Modell: {model_config['name']}")
    print(f"   Algorithmus: {model_config['model_type']}")
    print(f"   Trainings-Zeitraum: {model_config['train_start']} bis {model_config['train_end']}")
    
    try:
        response = requests.post(
            f"{API_BASE_URL}/models/create",
            json=model_config,
            timeout=30
        )
        
        if response.status_code == 201:
            data = response.json()
            job_id = data.get('job_id')
            print(f"✅ Modell-Erstellung gestartet!")
            print(f"   Job ID: {job_id}")
            print(f"   Status: {data.get('status')}")
            return True, job_id
        else:
            print(f"❌ Fehler: {response.status_code}")
            print(f"   Response: {response.text}")
            return False, None
    except Exception as e:
        print(f"❌ Fehler: {e}")
        import traceback
        traceback.print_exc()
        return False, None

def main():
    """Hauptfunktion"""
    print("\n" + "="*60)
    print("🚀 Test: 2 Modelle erstellen")
    print("="*60)
    
    # Hole Daten-Verfügbarkeit
    print("\n📊 Prüfe Daten-Verfügbarkeit...")
    min_ts, max_ts = get_data_availability()
    
    if not min_ts or not max_ts:
        print("❌ Keine Daten verfügbar!")
        return
    
    min_dt = parser.isoparse(min_ts)
    max_dt = parser.isoparse(max_ts)
    
    print(f"✅ Daten verfügbar:")
    print(f"   Von: {min_dt}")
    print(f"   Bis: {max_dt}")
    
    # Berechne Trainings-Zeiträume (letzte 2 Tage)
    train_end = max_dt
    train_start = train_end - timedelta(days=2)
    
    # Hole Phasen
    phases = get_phases()
    print(f"\n📋 Verwende Phasen: {phases}")
    
    # Timestamp für eindeutige Namen
    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    
    # Standard-Features
    default_features = [
        "price_open", "price_high", "price_low", "price_close",
        "volume_sol", "volume_usd",
        "market_cap_close",
        "buy_volume_sol", "sell_volume_sol"
    ]
    
    # Modell 1: Random Forest (einfach)
    print("\n" + "="*60)
    model1_config = {
        "name": f"Test-RF-{timestamp}",
        "model_type": "random_forest",
        "features": default_features,
        "train_start": train_start.isoformat(),
        "train_end": train_end.isoformat(),
        "description": "Test-Modell: Random Forest (einfach)",
        "phases": phases,
        "use_engineered_features": False,
        "use_smote": False,
        "use_timeseries_split": False,
        "use_time_based_prediction": False,
        "target_var": "market_cap_close",
        "operator": ">",
        "target_value": 10000.0
    }
    
    success1, job_id1 = create_model(model1_config)
    
    # Modell 2: XGBoost (mit Feature Engineering)
    print("\n" + "="*60)
    model2_config = {
        "name": f"Test-XGB-{timestamp}",
        "model_type": "xgboost",
        "features": default_features,
        "train_start": train_start.isoformat(),
        "train_end": train_end.isoformat(),
        "description": "Test-Modell: XGBoost (mit Feature Engineering)",
        "phases": phases,
        "use_engineered_features": True,
        "use_smote": False,
        "use_timeseries_split": False,
        "use_time_based_prediction": False,
        "target_var": "market_cap_close",
        "operator": ">",
        "target_value": 10000.0,
        "params": {
            "n_estimators": 100,
            "max_depth": 5
        }
    }
    
    success2, job_id2 = create_model(model2_config)
    
    # Zusammenfassung
    print("\n" + "="*60)
    print("📊 Zusammenfassung")
    print("="*60)
    
    if success1:
        print(f"✅ Modell 1 (Random Forest) erstellt - Job ID: {job_id1}")
    else:
        print(f"❌ Modell 1 (Random Forest) fehlgeschlagen")
    
    if success2:
        print(f"✅ Modell 2 (XGBoost) erstellt - Job ID: {job_id2}")
    else:
        print(f"❌ Modell 2 (XGBoost) fehlgeschlagen")
    
    if success1 and success2:
        print("\n🎉 Beide Modelle wurden erfolgreich erstellt!")
        print("\n💡 Tipp: Prüfe den Status der Jobs mit:")
        print(f"   curl {API_BASE_URL}/queue/{job_id1}")
        print(f"   curl {API_BASE_URL}/queue/{job_id2}")
    elif success1 or success2:
        print("\n⚠️  Ein Modell wurde erstellt, das andere fehlgeschlagen")
    else:
        print("\n❌ Beide Modelle konnten nicht erstellt werden")

if __name__ == "__main__":
    main()

