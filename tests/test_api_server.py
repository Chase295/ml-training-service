#!/usr/bin/env python3
"""
Test-Script für Server-API
Testet alle wichtigen API-Endpunkte
"""

import requests
import json
from datetime import datetime, timedelta, timezone

API_BASE_URL = "http://100.76.209.59:8005/api"

def test_health():
    """Test Health Check"""
    print("\n" + "="*60)
    print("1. Health Check")
    print("="*60)
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=10)
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Health: {data.get('status')}")
            print(f"   DB Connected: {data.get('db_connected')}")
            print(f"   Uptime: {data.get('uptime_seconds', 0)}s")
            return True
        else:
            print(f"❌ Fehler: {response.status_code}")
            print(response.text)
            return False
    except Exception as e:
        print(f"❌ Fehler: {e}")
        return False

def test_metrics():
    """Test Metrics Endpoint"""
    print("\n" + "="*60)
    print("2. Metrics")
    print("="*60)
    try:
        response = requests.get(f"{API_BASE_URL}/metrics", timeout=10)
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            print("✅ Metrics erreichbar")
            return True
        else:
            print(f"❌ Fehler: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Fehler: {e}")
        return False

def test_data_availability():
    """Test Data Availability"""
    print("\n" + "="*60)
    print("3. Data Availability")
    print("="*60)
    try:
        response = requests.get(f"{API_BASE_URL}/data-availability", timeout=10)
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print("✅ Data Availability:")
            print(f"   Min Timestamp: {data.get('min_timestamp')}")
            print(f"   Max Timestamp: {data.get('max_timestamp')}")
            return True
        else:
            print(f"❌ Fehler: {response.status_code}")
            print(response.text)
            return False
    except Exception as e:
        print(f"❌ Fehler: {e}")
        return False

def test_models_list():
    """Test Models auflisten"""
    print("\n" + "="*60)
    print("4. Models auflisten")
    print("="*60)
    try:
        response = requests.get(f"{API_BASE_URL}/models", timeout=10)
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            models = response.json()  # Direkt eine Liste, nicht wrapped
            if isinstance(models, list):
                print(f"✅ {len(models)} Modelle gefunden")
                if models:
                    print("\n   Erste 3 Modelle:")
                    for model in models[:3]:
                        print(f"   - {model.get('name')} ({model.get('algorithm')})")
                        print(f"     Status: {model.get('status')}, Erstellt: {model.get('created_at')}")
                return True, models
            else:
                print(f"⚠️  Unerwartetes Format: {type(models)}")
                return False, []
        else:
            print(f"❌ Fehler: {response.status_code}")
            print(response.text)
            return False, []
    except Exception as e:
        print(f"❌ Fehler: {e}")
        return False, []

def test_jobs_list():
    """Test Jobs auflisten (Queue)"""
    print("\n" + "="*60)
    print("5. Jobs auflisten (Queue)")
    print("="*60)
    try:
        response = requests.get(f"{API_BASE_URL}/queue", timeout=10)
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            jobs = response.json()  # Direkt eine Liste, nicht wrapped
            if isinstance(jobs, list):
                print(f"✅ {len(jobs)} Jobs gefunden")
                if jobs:
                    print("\n   Letzte 3 Jobs:")
                    for job in jobs[:3]:
                        print(f"   - {job.get('job_type')} (ID: {job.get('id')})")
                        print(f"     Status: {job.get('status')}, Erstellt: {job.get('created_at')}")
                return True, jobs
            else:
                print(f"⚠️  Unerwartetes Format: {type(jobs)}")
                return False, []
        else:
            print(f"❌ Fehler: {response.status_code}")
            print(response.text)
            return False, []
    except Exception as e:
        print(f"❌ Fehler: {e}")
        return False, []

def test_test_results_list():
    """Test Test-Ergebnisse auflisten"""
    print("\n" + "="*60)
    print("6. Test-Ergebnisse auflisten")
    print("="*60)
    try:
        response = requests.get(f"{API_BASE_URL}/test-results", timeout=10)
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            results = response.json()  # Direkt eine Liste, nicht wrapped
            if isinstance(results, list):
                print(f"✅ {len(results)} Test-Ergebnisse gefunden")
                if results:
                    print("\n   Letzte 3 Ergebnisse:")
                    for result in results[:3]:
                        model_name = result.get('model_name', 'N/A')
                        accuracy = result.get('accuracy', 'N/A')
                        print(f"   - Modell: {model_name}, Accuracy: {accuracy}")
                return True
            else:
                print(f"⚠️  Unerwartetes Format: {type(results)}")
                return False
        else:
            print(f"❌ Fehler: {response.status_code}")
            print(response.text)
            return False
    except Exception as e:
        print(f"❌ Fehler: {e}")
        return False

def test_comparisons_list():
    """Test Vergleichs-Ergebnisse auflisten"""
    print("\n" + "="*60)
    print("7. Vergleichs-Ergebnisse auflisten")
    print("="*60)
    try:
        response = requests.get(f"{API_BASE_URL}/comparisons", timeout=10)
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            comparisons = response.json()  # Direkt eine Liste, nicht wrapped
            if isinstance(comparisons, list):
                print(f"✅ {len(comparisons)} Vergleiche gefunden")
                if comparisons:
                    print("\n   Letzte 3 Vergleiche:")
                    for comp in comparisons[:3]:
                        winner = comp.get('winner', 'N/A')
                        print(f"   - Gewinner: {winner}")
                return True
            else:
                print(f"⚠️  Unerwartetes Format: {type(comparisons)}")
                return False
        else:
            print(f"❌ Fehler: {response.status_code}")
            print(response.text)
            return False
    except Exception as e:
        print(f"❌ Fehler: {e}")
        return False

def test_phases():
    """Test Phases auflisten"""
    print("\n" + "="*60)
    print("8. Phases auflisten")
    print("="*60)
    try:
        response = requests.get(f"{API_BASE_URL}/phases", timeout=10)
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            phases = response.json()
            if isinstance(phases, list):
                print(f"✅ {len(phases)} Phasen gefunden")
                if phases:
                    print("\n   Erste 3 Phasen:")
                    for phase in phases[:3]:
                        print(f"   - {phase.get('name')} (ID: {phase.get('id')})")
                        print(f"     Interval: {phase.get('interval_seconds')}s")
                return True
            else:
                print(f"⚠️  Unerwartetes Format: {type(phases)}")
                return False
        else:
            print(f"❌ Fehler: {response.status_code}")
            print(response.text)
            return False
    except Exception as e:
        print(f"❌ Fehler: {e}")
        return False

def test_model_create():
    """Test Modell erstellen (nur wenn Daten verfügbar)"""
    print("\n" + "="*60)
    print("8. Modell erstellen (Test)")
    print("="*60)
    
    # Prüfe erst Data Availability
    try:
        response = requests.get(f"{API_BASE_URL}/data-availability", timeout=10)
        if response.status_code != 200:
            print("⚠️  Data Availability nicht verfügbar - überspringe Modell-Erstellung")
            return False
        
        data = response.json()
        min_ts = data.get('min_timestamp')
        max_ts = data.get('max_timestamp')
        
        if not min_ts or not max_ts:
            print("⚠️  Keine Daten verfügbar - überspringe Modell-Erstellung")
            return False
        
        # Parse timestamps
        from dateutil import parser
        min_dt = parser.isoparse(min_ts)
        max_dt = parser.isoparse(max_ts)
        
        # Verwende letzten Tag für Training
        train_end = max_dt
        train_start = train_end - timedelta(days=1)
        
        # Erstelle Test-Modell
        model_data = {
            "name": f"API-Test-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
            "algorithm": "random_forest",
            "train_start": train_start.isoformat(),
            "train_end": train_end.isoformat(),
            "description": "API Test Modell",
            "use_feature_engineering": False,
            "use_smote": False,
            "use_time_series_split": False
        }
        
        print(f"   Erstelle Modell: {model_data['name']}")
        print(f"   Trainings-Zeitraum: {train_start.isoformat()} bis {train_end.isoformat()}")
        
        response = requests.post(
            f"{API_BASE_URL}/models/create",
            json=model_data,
            timeout=30
        )
        
        print(f"Status: {response.status_code}")
        if response.status_code == 201:
            data = response.json()
            job_id = data.get('job_id')
            print(f"✅ Modell-Erstellung gestartet (Job ID: {job_id})")
            return True
        else:
            print(f"❌ Fehler: {response.status_code}")
            print(response.text)
            return False
            
    except Exception as e:
        print(f"❌ Fehler: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Hauptfunktion"""
    print("\n" + "="*60)
    print("🚀 API Server Test")
    print(f"   URL: {API_BASE_URL}")
    print("="*60)
    
    results = {}
    
    # Basis-Tests
    results['health'] = test_health()
    results['metrics'] = test_metrics()
    results['data_availability'] = test_data_availability()
    
    # Daten-Tests
    success, models = test_models_list()
    results['models_list'] = success
    
    success, jobs = test_jobs_list()
    results['jobs_list'] = success
    
    results['test_results'] = test_test_results_list()
    results['comparisons'] = test_comparisons_list()
    results['phases'] = test_phases()
    
    # Optional: Modell erstellen (nur wenn gewünscht)
    # results['model_create'] = test_model_create()
    
    # Zusammenfassung
    print("\n" + "="*60)
    print("📊 Zusammenfassung")
    print("="*60)
    
    total = len(results)
    passed = sum(1 for v in results.values() if v)
    
    for test, result in results.items():
        status = "✅" if result else "❌"
        print(f"{status} {test}")
    
    print(f"\n✅ {passed}/{total} Tests erfolgreich")
    
    if passed == total:
        print("\n🎉 Alle Tests erfolgreich!")
    else:
        print(f"\n⚠️  {total - passed} Test(s) fehlgeschlagen")

if __name__ == "__main__":
    main()

