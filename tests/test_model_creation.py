#!/usr/bin/env python3
"""
Automatisierte Tests für KI-Modell-Erstellung
Führt alle kritischen Tests durch, um sicherzustellen, dass alles funktioniert.
"""

import requests
import json
import time
import sys
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, Optional

BASE_URL = "http://localhost:8000"
TIMEOUT = 300  # 5 Minuten für Job-Completion

class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    END = '\033[0m'

def print_success(msg: str):
    print(f"{Colors.GREEN}✅ {msg}{Colors.END}")

def print_error(msg: str):
    print(f"{Colors.RED}❌ {msg}{Colors.END}")

def print_warning(msg: str):
    print(f"{Colors.YELLOW}⚠️  {msg}{Colors.END}")

def print_info(msg: str):
    print(f"{Colors.BLUE}ℹ️  {msg}{Colors.END}")

class ModelCreationTester:
    def __init__(self, base_url: str = BASE_URL):
        self.base_url = base_url
        self.test_results = []
        self.created_model_ids = []
        self.created_job_ids = []
    
    def test_health(self) -> bool:
        """Test 1: Health Check"""
        print_info("Test 1: Health Check")
        try:
            response = requests.get(f"{self.base_url}/api/health", timeout=5)
            if response.status_code == 200:
                data = response.json()
                if data.get('status') == 'healthy':
                    print_success("Health Check erfolgreich")
                    return True
            print_error(f"Health Check fehlgeschlagen: {response.status_code}")
            return False
        except Exception as e:
            print_error(f"Health Check Fehler: {e}")
            return False
    
    def test_data_availability(self) -> bool:
        """Test 2: Data Availability"""
        print_info("Test 2: Data Availability")
        try:
            response = requests.get(f"{self.base_url}/api/data-availability", timeout=5)
            if response.status_code == 200:
                data = response.json()
                if 'min_timestamp' in data and 'max_timestamp' in data:
                    min_ts = data['min_timestamp']
                    max_ts = data['max_timestamp']
                    total = data.get('total_samples', 0)
                    
                    print_success(f"Data Availability OK: {total} Samples")
                    print_info(f"  Zeitraum: {min_ts} bis {max_ts}")
                    
                    if total < 1000:
                        print_warning(f"Weniger als 1000 Samples verfügbar: {total}")
                    
                    return True
            print_error("Data Availability fehlgeschlagen")
            return False
        except Exception as e:
            print_error(f"Data Availability Fehler: {e}")
            return False
    
    def test_phases(self) -> bool:
        """Test 3: Phases"""
        print_info("Test 3: Phases")
        try:
            response = requests.get(f"{self.base_url}/api/phases", timeout=5)
            if response.status_code == 200:
                data = response.json()
                if isinstance(data, list):
                    print_success(f"Phases geladen: {len(data)} Phasen")
                    return True
            print_error("Phases fehlgeschlagen")
            return False
        except Exception as e:
            print_error(f"Phases Fehler: {e}")
            return False
    
    def test_create_model_minimal(self) -> Optional[str]:
        """Test 4: Modell erstellen (minimal)"""
        print_info("Test 4: Modell erstellen (minimal)")
        try:
            # Hole verfügbare Daten
            data_avail = requests.get(f"{self.base_url}/api/data-availability").json()
            min_ts = data_avail['min_timestamp']
            max_ts = data_avail['max_timestamp']
            
            # Verwende ersten und letzten Tag
            train_start = min_ts
            train_end = max_ts
            
            payload = {
                "name": f"TEST_MINIMAL_{int(time.time())}",
                "model_type": "random_forest",
                "target_var": "price_close",
                "operator": None,
                "target_value": None,
                "features": ["price_open", "price_close", "volume_sol"],
                "phases": None,
                "params": None,
                "train_start": train_start,
                "train_end": train_end,
                "use_time_based_prediction": True,
                "future_minutes": 10,
                "min_percent_change": 5.0,
                "direction": "up",
                "use_engineered_features": False,
                "feature_engineering_windows": None,
                "use_smote": False,
                "use_timeseries_split": False,
                "cv_splits": None,
                "use_market_context": False
            }
            
            response = requests.post(
                f"{self.base_url}/api/models/create",
                json=payload,
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                job_id = data.get('job_id')
                if job_id:
                    print_success(f"Modell erstellt: Job-ID {job_id}")
                    self.created_job_ids.append(job_id)
                    return job_id
            else:
                print_error(f"Modell-Erstellung fehlgeschlagen: {response.status_code}")
                print_error(f"Response: {response.text}")
                return None
        except Exception as e:
            print_error(f"Modell-Erstellung Fehler: {e}")
            return None
    
    def test_create_model_full(self) -> Optional[str]:
        """Test 5: Modell erstellen (vollständig)"""
        print_info("Test 5: Modell erstellen (vollständig)")
        try:
            data_avail = requests.get(f"{self.base_url}/api/data-availability").json()
            min_ts = data_avail['min_timestamp']
            max_ts = data_avail['max_timestamp']
            
            payload = {
                "name": f"TEST_FULL_{int(time.time())}",
                "model_type": "xgboost",
                "target_var": "price_close",
                "operator": None,
                "target_value": None,
                "features": [
                    "price_open", "price_high", "price_low", "price_close",
                    "volume_sol", "dev_sold_amount", "buy_pressure_ratio",
                    "unique_signer_ratio", "whale_buy_volume_sol", "volatility_pct"
                ],
                "phases": None,
                "params": {
                    "n_estimators": 100,
                    "max_depth": 6,
                    "learning_rate": 0.1
                },
                "train_start": min_ts,
                "train_end": max_ts,
                "use_time_based_prediction": True,
                "future_minutes": 15,
                "min_percent_change": 7.5,
                "direction": "up",
                "use_engineered_features": True,
                "feature_engineering_windows": [5, 10, 15],
                "use_smote": True,
                "use_timeseries_split": True,
                "cv_splits": 5,
                "use_market_context": True
            }
            
            response = requests.post(
                f"{self.base_url}/api/models/create",
                json=payload,
                timeout=10
            )
            
            if response.status_code in [200, 201]:  # 201 Created ist auch OK
                data = response.json()
                job_id = data.get('job_id')
                if job_id:
                    print_success(f"Vollständiges Modell erstellt: Job-ID {job_id}")
                    self.created_job_ids.append(job_id)
                    return job_id
            else:
                print_error(f"Modell-Erstellung fehlgeschlagen: {response.status_code}")
                print_error(f"Response: {response.text}")
                return None
        except Exception as e:
            print_error(f"Modell-Erstellung Fehler: {e}")
            return None
    
    def wait_for_job(self, job_id: str, max_wait: int = TIMEOUT) -> Optional[Dict[str, Any]]:
        """Warte auf Job-Completion"""
        print_info(f"Warte auf Job {job_id}...")
        start_time = time.time()
        
        while time.time() - start_time < max_wait:
            try:
                response = requests.get(f"{self.base_url}/api/queue/{job_id}", timeout=5)
                if response.status_code == 200:
                    job_data = response.json()
                    status = job_data.get('status')
                    progress = job_data.get('progress', 0)
                    
                    print_info(f"  Status: {status}, Progress: {progress*100:.1f}%")
                    
                    if status == 'COMPLETED':
                        print_success(f"Job {job_id} abgeschlossen")
                        return job_data
                    elif status == 'FAILED':
                        error = job_data.get('error', 'Unknown error')
                        print_error(f"Job {job_id} fehlgeschlagen: {error}")
                        return None
                    
                time.sleep(5)
            except Exception as e:
                print_warning(f"Fehler beim Abfragen des Jobs: {e}")
                time.sleep(5)
        
        print_error(f"Job {job_id} Timeout nach {max_wait}s")
        return None
    
    def test_job_completion(self, job_id: str) -> bool:
        """Test 6: Job-Completion prüfen"""
        print_info(f"Test 6: Job-Completion für {job_id}")
        job_data = self.wait_for_job(job_id)
        
        if not job_data:
            return False
        
        # Prüfe Ergebnis
        if job_data.get('job_type') == 'TRAIN':
            result = job_data.get('result_model')
            if result:
                model_id = result.get('model_id')
                if model_id:
                    print_success(f"Modell erstellt: ID {model_id}")
                    self.created_model_ids.append(model_id)
                    
                    # Prüfe Metriken
                    accuracy = result.get('accuracy')
                    f1_score = result.get('f1_score')
                    
                    if accuracy is not None and f1_score is not None:
                        print_success(f"  Accuracy: {accuracy:.4f}, F1-Score: {f1_score:.4f}")
                        
                        # Prüfe Rug-Detection Metriken
                        rug_metrics = result.get('rug_detection_metrics')
                        if rug_metrics:
                            print_success("  Rug-Detection Metriken vorhanden")
                            print_info(f"    Dev-Sold Detection Rate: {rug_metrics.get('dev_sold_detection_rate', 'N/A')}")
                            print_info(f"    Weighted Cost: {rug_metrics.get('weighted_cost', 'N/A')}")
                        else:
                            print_warning("  Rug-Detection Metriken fehlen")
                        
                        return True
                    else:
                        print_error("Metriken fehlen im Ergebnis")
                        return False
                else:
                    print_error("Model-ID fehlt im Ergebnis")
                    return False
            else:
                print_error("Kein Ergebnis im Job")
                return False
        else:
            print_warning(f"Unbekannter Job-Typ: {job_data.get('job_type')}")
            return True  # Nicht kritisch
    
    def test_model_testing(self, model_id: int) -> bool:
        """Test 7: Modell testen"""
        print_info(f"Test 7: Modell {model_id} testen")
        try:
            data_avail = requests.get(f"{self.base_url}/api/data-availability").json()
            max_ts = data_avail['max_timestamp']
            
            # Test-Zeitraum: Letzte 7 Tage
            test_end = max_ts
            # Parse und subtrahiere 7 Tage
            test_end_dt = datetime.fromisoformat(test_end.replace('Z', '+00:00'))
            test_start_dt = test_end_dt - timedelta(days=7)
            test_start = test_start_dt.isoformat().replace('+00:00', 'Z')
            
            payload = {
                "test_start": test_start,
                "test_end": test_end
            }
            
            response = requests.post(
                f"{self.base_url}/api/models/{model_id}/test",
                json=payload,
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                job_id = data.get('job_id')
                if job_id:
                    print_success(f"Test-Job erstellt: {job_id}")
                    self.created_job_ids.append(job_id)
                    
                    # Warte auf Completion
                    job_data = self.wait_for_job(job_id)
                    if job_data:
                        test_result = job_data.get('result_test')
                        if test_result:
                            accuracy = test_result.get('accuracy')
                            print_success(f"Test abgeschlossen: Accuracy={accuracy:.4f}")
                            return True
                    return False
            else:
                print_error(f"Test-Erstellung fehlgeschlagen: {response.status_code}")
                return False
        except Exception as e:
            print_error(f"Test-Fehler: {e}")
            return False
    
    def run_all_tests(self) -> Dict[str, bool]:
        """Führe alle Tests aus"""
        print_info("=" * 60)
        print_info("Starte automatische Tests für KI-Modell-Erstellung")
        print_info("=" * 60)
        
        results = {}
        
        # Phase 1: Basis-Tests
        results['health'] = self.test_health()
        results['data_availability'] = self.test_data_availability()
        results['phases'] = self.test_phases()
        
        if not all([results['health'], results['data_availability']]):
            print_error("Basis-Tests fehlgeschlagen. Stoppe weitere Tests.")
            return results
        
        # Phase 2: Modell-Erstellung
        minimal_job_id = self.test_create_model_minimal()
        results['create_minimal'] = minimal_job_id is not None
        
        full_job_id = self.test_create_model_full()
        results['create_full'] = full_job_id is not None
        
        # Phase 3: Job-Completion
        if minimal_job_id:
            results['job_completion_minimal'] = self.test_job_completion(minimal_job_id)
        
        if full_job_id:
            results['job_completion_full'] = self.test_job_completion(full_job_id)
        
        # Phase 4: Modell-Testing
        if self.created_model_ids:
            model_id = self.created_model_ids[0]
            results['model_testing'] = self.test_model_testing(model_id)
        
        # Zusammenfassung
        print_info("=" * 60)
        print_info("Test-Zusammenfassung")
        print_info("=" * 60)
        
        passed = sum(1 for v in results.values() if v)
        total = len(results)
        
        for test_name, result in results.items():
            if result:
                print_success(f"{test_name}: BESTANDEN")
            else:
                print_error(f"{test_name}: FEHLGESCHLAGEN")
        
        print_info(f"\nGesamt: {passed}/{total} Tests bestanden")
        
        if passed == total:
            print_success("🎉 Alle Tests bestanden!")
        else:
            print_error(f"⚠️  {total - passed} Test(s) fehlgeschlagen")
        
        return results

def main():
    """Hauptfunktion"""
    tester = ModelCreationTester()
    results = tester.run_all_tests()
    
    # Exit-Code basierend auf Ergebnissen
    if all(results.values()):
        sys.exit(0)
    else:
        sys.exit(1)

if __name__ == "__main__":
    main()

