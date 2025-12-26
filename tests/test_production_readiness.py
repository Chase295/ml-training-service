#!/usr/bin/env python3
"""
Production Readiness Test - Vollständige Prüfung vor Live-Deployment
"""

import sys
import requests
import json
import time
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, List

# Konfiguration
API_BASE_URL = "http://localhost:8012"
TIMEOUT = 60

# Farben für Output
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    RESET = '\033[0m'

def print_success(msg):
    print(f"{Colors.GREEN}✅ {msg}{Colors.RESET}")

def print_error(msg):
    print(f"{Colors.RED}❌ {msg}{Colors.RESET}")

def print_warning(msg):
    print(f"{Colors.YELLOW}⚠️  {msg}{Colors.RESET}")

def print_info(msg):
    print(f"{Colors.BLUE}ℹ️  {msg}{Colors.RESET}")

def print_section(title):
    print(f"\n{'='*60}")
    print(f"{Colors.BLUE}{title}{Colors.RESET}")
    print(f"{'='*60}\n")

class ProductionReadinessTest:
    def __init__(self):
        self.results = {
            "passed": [],
            "failed": [],
            "warnings": []
        }
        self.created_model_ids = []
        self.created_job_ids = []
    
    def test_health_check(self):
        """Test 1: Health Check"""
        print_section("Test 1: Health Check")
        try:
            response = requests.get(f"{API_BASE_URL}/api/health", timeout=10)
            if response.status_code == 200:
                data = response.json()
                print_success(f"Health Check erfolgreich")
                print_info(f"  Status: {data.get('status')}")
                print_info(f"  DB Connected: {data.get('db_connected')}")
                print_info(f"  Uptime: {data.get('uptime_seconds')}s")
                self.results["passed"].append("Health Check")
                return True
            else:
                print_error(f"Health Check fehlgeschlagen: HTTP {response.status_code}")
                self.results["failed"].append("Health Check")
                return False
        except Exception as e:
            print_error(f"Health Check Fehler: {e}")
            self.results["failed"].append("Health Check")
            return False
    
    def test_data_availability(self):
        """Test 2: Daten-Verfügbarkeit"""
        print_section("Test 2: Daten-Verfügbarkeit")
        try:
            response = requests.get(f"{API_BASE_URL}/api/data-availability", timeout=10)
            if response.status_code == 200:
                data = response.json()
                min_ts = data.get('min_timestamp')
                max_ts = data.get('max_timestamp')
                print_success(f"Daten-Verfügbarkeit OK")
                print_info(f"  Zeitraum: {min_ts} bis {max_ts}")
                
                # Prüfe ob genug Daten vorhanden
                if min_ts and max_ts:
                    min_dt = datetime.fromisoformat(min_ts.replace('Z', '+00:00'))
                    max_dt = datetime.fromisoformat(max_ts.replace('Z', '+00:00'))
                    duration = (max_dt - min_dt).total_seconds() / 3600  # Stunden
                    print_info(f"  Dauer: {duration:.1f} Stunden")
                    
                    if duration < 24:
                        print_warning(f"Wenig Daten verfügbar: {duration:.1f} Stunden (empfohlen: >24h)")
                        self.results["warnings"].append(f"Wenig Daten: {duration:.1f}h")
                
                self.results["passed"].append("Daten-Verfügbarkeit")
                return True
            else:
                print_error(f"Daten-Verfügbarkeit fehlgeschlagen: HTTP {response.status_code}")
                self.results["failed"].append("Daten-Verfügbarkeit")
                return False
        except Exception as e:
            print_error(f"Daten-Verfügbarkeit Fehler: {e}")
            self.results["failed"].append("Daten-Verfügbarkeit")
            return False
    
    def test_create_minimal_model(self):
        """Test 3: Minimales Modell erstellen"""
        print_section("Test 3: Minimales Modell erstellen")
        try:
            # Hole Daten-Verfügbarkeit
            data_avail = requests.get(f"{API_BASE_URL}/api/data-availability", timeout=10).json()
            min_ts = data_avail.get('min_timestamp')
            max_ts = data_avail.get('max_timestamp')
            
            if not min_ts or not max_ts:
                print_error("Keine Daten verfügbar für Modell-Erstellung")
                self.results["failed"].append("Minimales Modell erstellen")
                return False
            
            # Erstelle minimales Modell
            payload = {
                "name": f"PROD_TEST_MINIMAL_{int(time.time())}",
                "model_type": "random_forest",
                "target_var": "price_close",
                "operator": None,
                "target_value": None,
                "features": ["price_open", "price_close", "volume_sol"],
                "phases": None,
                "params": {
                    "use_smote": False,
                    "use_timeseries_split": False
                },
                "train_start": min_ts,
                "train_end": max_ts,
                "use_time_based_prediction": True,
                "future_minutes": 10,
                "min_percent_change": 5.0,
                "direction": "up",
                "use_engineered_features": False,
                "use_market_context": False
            }
            
            response = requests.post(
                f"{API_BASE_URL}/api/models/create",
                json=payload,
                timeout=30
            )
            
            if response.status_code in [200, 201]:
                data = response.json()
                job_id = data.get('job_id')
                self.created_job_ids.append(job_id)
                print_success(f"Minimales Modell erstellt: Job-ID {job_id}")
                
                # Warte auf Job-Completion
                print_info("Warte auf Job-Completion...")
                for i in range(60):  # Max 5 Minuten
                    time.sleep(5)
                    job_response = requests.get(f"{API_BASE_URL}/api/queue/{job_id}", timeout=10)
                    if job_response.status_code == 200:
                        job_data = job_response.json()
                        status = job_data.get('status')
                        progress = job_data.get('progress', 0)
                        print_info(f"  Status: {status}, Progress: {progress*100:.1f}%")
                        
                        if status == "COMPLETED":
                            model_id = job_data.get('result_model_id')
                            if model_id:
                                self.created_model_ids.append(model_id)
                                print_success(f"Modell erfolgreich erstellt: ID {model_id}")
                                self.results["passed"].append("Minimales Modell erstellen")
                                return True
                        elif status == "FAILED":
                            error = job_data.get('error_msg', 'Unknown error')
                            print_error(f"Job fehlgeschlagen: {error}")
                            self.results["failed"].append("Minimales Modell erstellen")
                            return False
                
                print_warning("Job nicht innerhalb von 5 Minuten abgeschlossen")
                self.results["warnings"].append("Minimales Modell: Job läuft noch")
                return False
            else:
                print_error(f"Modell-Erstellung fehlgeschlagen: HTTP {response.status_code}")
                print_error(f"Response: {response.text}")
                self.results["failed"].append("Minimales Modell erstellen")
                return False
        except Exception as e:
            print_error(f"Fehler bei Modell-Erstellung: {e}")
            self.results["failed"].append("Minimales Modell erstellen")
            return False
    
    def test_model_labels(self, model_id: int):
        """Test 4: Labels/Tags Validierung"""
        print_section(f"Test 4: Labels/Tags Validierung für Modell {model_id}")
        try:
            response = requests.get(f"{API_BASE_URL}/api/models/{model_id}", timeout=10)
            if response.status_code == 200:
                model = response.json()
                
                # Prüfe Labels
                print_info("Prüfe Labels...")
                
                # Prüfe ob zeitbasierte Vorhersage korrekt gesetzt
                params = model.get('params', {})
                time_based = params.get('_time_based')
                
                if time_based:
                    print_success("Zeitbasierte Vorhersage aktiviert")
                    print_info(f"  Future Minutes: {time_based.get('future_minutes')}")
                    print_info(f"  Min Percent Change: {time_based.get('min_percent_change')}")
                    print_info(f"  Direction: {time_based.get('direction')}")
                else:
                    print_warning("Keine zeitbasierte Vorhersage gefunden")
                
                # Prüfe Metriken
                accuracy = model.get('training_accuracy')
                f1 = model.get('training_f1')
                precision = model.get('training_precision')
                recall = model.get('training_recall')
                
                if accuracy is not None:
                    print_success(f"Metriken vorhanden:")
                    print_info(f"  Accuracy: {accuracy:.4f}")
                    print_info(f"  F1-Score: {f1:.4f}")
                    print_info(f"  Precision: {precision:.4f}")
                    print_info(f"  Recall: {recall:.4f}")
                    
                    # Prüfe ob Metriken sinnvoll sind
                    if accuracy < 0.5:
                        print_warning(f"Accuracy sehr niedrig: {accuracy:.4f}")
                        self.results["warnings"].append(f"Modell {model_id}: Niedrige Accuracy")
                    
                    self.results["passed"].append(f"Labels/Tags Validierung Modell {model_id}")
                    return True
                else:
                    print_error("Keine Metriken gefunden")
                    self.results["failed"].append(f"Labels/Tags Validierung Modell {model_id}")
                    return False
            else:
                print_error(f"Modell nicht gefunden: HTTP {response.status_code}")
                self.results["failed"].append(f"Labels/Tags Validierung Modell {model_id}")
                return False
        except Exception as e:
            print_error(f"Fehler bei Labels-Validierung: {e}")
            self.results["failed"].append(f"Labels/Tags Validierung Modell {model_id}")
            return False
    
    def test_model_testing(self, model_id: int):
        """Test 5: Modell testen"""
        print_section(f"Test 5: Modell testen (ID {model_id})")
        try:
            # Hole Modell-Info
            model_response = requests.get(f"{API_BASE_URL}/api/models/{model_id}", timeout=10)
            if model_response.status_code != 200:
                print_error(f"Modell nicht gefunden: {model_id}")
                self.results["failed"].append(f"Modell testen {model_id}")
                return False
            
            model = model_response.json()
            train_start = model.get('train_start')
            train_end = model.get('train_end')
            
            if not train_start or not train_end:
                print_error("Train-Zeitraum nicht gefunden")
                self.results["failed"].append(f"Modell testen {model_id}")
                return False
            
            # Erstelle Test-Zeitraum (nach Training)
            train_end_dt = datetime.fromisoformat(train_end.replace('Z', '+00:00'))
            test_start_dt = train_end_dt + timedelta(minutes=10)
            test_end_dt = test_start_dt + timedelta(hours=1)
            
            test_start = test_start_dt.isoformat().replace('+00:00', 'Z')
            test_end = test_end_dt.isoformat().replace('+00:00', 'Z')
            
            print_info(f"Test-Zeitraum: {test_start} bis {test_end}")
            
            # Erstelle Test-Job
            payload = {
                "test_start": test_start,
                "test_end": test_end
            }
            
            response = requests.post(
                f"{API_BASE_URL}/api/models/{model_id}/test",
                json=payload,
                timeout=30
            )
            
            if response.status_code in [200, 201]:
                data = response.json()
                test_job_id = data.get('job_id')
                print_success(f"Test-Job erstellt: {test_job_id}")
                
                # Warte auf Completion
                print_info("Warte auf Test-Completion...")
                for i in range(60):
                    time.sleep(5)
                    job_response = requests.get(f"{API_BASE_URL}/api/queue/{test_job_id}", timeout=10)
                    if job_response.status_code == 200:
                        job_data = job_response.json()
                        status = job_data.get('status')
                        
                        if status == "COMPLETED":
                            test_id = job_data.get('result_test_id')
                            if test_id:
                                print_success(f"Test erfolgreich abgeschlossen: Test-ID {test_id}")
                                self.results["passed"].append(f"Modell testen {model_id}")
                                return True
                        elif status == "FAILED":
                            error = job_data.get('error_msg', 'Unknown error')
                            print_error(f"Test fehlgeschlagen: {error}")
                            self.results["failed"].append(f"Modell testen {model_id}")
                            return False
                
                print_warning("Test nicht innerhalb von 5 Minuten abgeschlossen")
                self.results["warnings"].append(f"Modell testen {model_id}: Läuft noch")
                return False
            else:
                print_error(f"Test-Erstellung fehlgeschlagen: HTTP {response.status_code}")
                print_error(f"Response: {response.text}")
                self.results["failed"].append(f"Modell testen {model_id}")
                return False
        except Exception as e:
            print_error(f"Fehler bei Modell-Test: {e}")
            self.results["failed"].append(f"Modell testen {model_id}")
            return False
    
    def test_model_comparison(self, model_id_1: int, model_id_2: int):
        """Test 6: Modelle vergleichen"""
        print_section(f"Test 6: Modelle vergleichen ({model_id_1} vs {model_id_2})")
        try:
            # Hole Modell-Info
            model1_response = requests.get(f"{API_BASE_URL}/api/models/{model_id_1}", timeout=10)
            model2_response = requests.get(f"{API_BASE_URL}/api/models/{model_id_2}", timeout=10)
            
            if model1_response.status_code != 200 or model2_response.status_code != 200:
                print_error("Eines der Modelle nicht gefunden")
                self.results["failed"].append("Modell-Vergleich")
                return False
            
            model1 = model1_response.json()
            train_end = model1.get('train_end')
            
            if not train_end:
                print_error("Train-Zeitraum nicht gefunden")
                self.results["failed"].append("Modell-Vergleich")
                return False
            
            # Erstelle Vergleichs-Zeitraum
            train_end_dt = datetime.fromisoformat(train_end.replace('Z', '+00:00'))
            compare_start_dt = train_end_dt + timedelta(minutes=10)
            compare_end_dt = compare_start_dt + timedelta(hours=1)
            
            compare_start = compare_start_dt.isoformat().replace('+00:00', 'Z')
            compare_end = compare_end_dt.isoformat().replace('+00:00', 'Z')
            
            # Erstelle Vergleich
            payload = {
                "model_a_id": model_id_1,
                "model_b_id": model_id_2,
                "compare_start": compare_start,
                "compare_end": compare_end
            }
            
            response = requests.post(
                f"{API_BASE_URL}/api/models/compare",
                json=payload,
                timeout=30
            )
            
            if response.status_code in [200, 201]:
                data = response.json()
                compare_job_id = data.get('job_id')
                print_success(f"Vergleichs-Job erstellt: {compare_job_id}")
                
                # Warte auf Completion
                print_info("Warte auf Vergleich-Completion...")
                for i in range(60):
                    time.sleep(5)
                    job_response = requests.get(f"{API_BASE_URL}/api/queue/{compare_job_id}", timeout=10)
                    if job_response.status_code == 200:
                        job_data = job_response.json()
                        status = job_data.get('status')
                        
                        if status == "COMPLETED":
                            compare_id = job_data.get('result_comparison_id')
                            if compare_id:
                                print_success(f"Vergleich erfolgreich: Vergleich-ID {compare_id}")
                                self.results["passed"].append("Modell-Vergleich")
                                return True
                        elif status == "FAILED":
                            error = job_data.get('error_msg', 'Unknown error')
                            print_error(f"Vergleich fehlgeschlagen: {error}")
                            self.results["failed"].append("Modell-Vergleich")
                            return False
                
                print_warning("Vergleich nicht innerhalb von 5 Minuten abgeschlossen")
                self.results["warnings"].append("Modell-Vergleich: Läuft noch")
                return False
            else:
                print_error(f"Vergleichs-Erstellung fehlgeschlagen: HTTP {response.status_code}")
                print_error(f"Response: {response.text}")
                self.results["failed"].append("Modell-Vergleich")
                return False
        except Exception as e:
            print_error(f"Fehler bei Modell-Vergleich: {e}")
            self.results["failed"].append("Modell-Vergleich")
            return False
    
    def test_file_structure(self):
        """Test 7: Datei-Struktur prüfen"""
        print_section("Test 7: Datei-Struktur prüfen")
        try:
            import os
            from pathlib import Path
            
            required_files = [
                "docker-compose.yml",
                "Dockerfile",
                "app/api/routes.py",
                "app/training/engine.py",
                "app/training/feature_engineering.py",
                "app/database/models.py",
                "app/streamlit_app.py",
                "sql/complete_schema.sql"
            ]
            
            missing_files = []
            for file_path in required_files:
                if not os.path.exists(file_path):
                    missing_files.append(file_path)
            
            if missing_files:
                print_error(f"Fehlende Dateien: {', '.join(missing_files)}")
                self.results["failed"].append("Datei-Struktur")
                return False
            else:
                print_success("Alle benötigten Dateien vorhanden")
                self.results["passed"].append("Datei-Struktur")
                return True
        except Exception as e:
            print_error(f"Fehler bei Datei-Struktur-Prüfung: {e}")
            self.results["failed"].append("Datei-Struktur")
            return False
    
    def test_sql_migrations(self):
        """Test 8: SQL-Migrationen prüfen"""
        print_section("Test 8: SQL-Migrationen prüfen")
        try:
            import os
            from pathlib import Path
            
            sql_dir = Path("sql")
            if not sql_dir.exists():
                print_error("SQL-Verzeichnis nicht gefunden")
                self.results["failed"].append("SQL-Migrationen")
                return False
            
            required_migrations = [
                "complete_schema.sql",
                "migration_add_rug_metrics.sql",
                "migration_create_exchange_rates.sql"
            ]
            
            missing_migrations = []
            for migration in required_migrations:
                migration_path = sql_dir / migration
                if not migration_path.exists():
                    missing_migrations.append(migration)
            
            if missing_migrations:
                print_error(f"Fehlende Migrationen: {', '.join(missing_migrations)}")
                self.results["failed"].append("SQL-Migrationen")
                return False
            else:
                print_success("Alle SQL-Migrationen vorhanden")
                self.results["passed"].append("SQL-Migrationen")
                return True
        except Exception as e:
            print_error(f"Fehler bei SQL-Migrationen-Prüfung: {e}")
            self.results["failed"].append("SQL-Migrationen")
            return False
    
    def print_summary(self):
        """Zusammenfassung ausgeben"""
        print_section("Test-Zusammenfassung")
        
        total = len(self.results["passed"]) + len(self.results["failed"]) + len(self.results["warnings"])
        
        print_info(f"Gesamt: {total} Tests")
        print_success(f"Bestanden: {len(self.results['passed'])}")
        print_error(f"Fehlgeschlagen: {len(self.results['failed'])}")
        print_warning(f"Warnungen: {len(self.results['warnings'])}")
        
        if self.results["failed"]:
            print("\n❌ Fehlgeschlagene Tests:")
            for test in self.results["failed"]:
                print(f"   - {test}")
        
        if self.results["warnings"]:
            print("\n⚠️  Warnungen:")
            for warning in self.results["warnings"]:
                print(f"   - {warning}")
        
        if len(self.results["failed"]) == 0:
            print_success("\n✅ ALLE KRITISCHEN TESTS BESTANDEN!")
            print_info("Das System ist bereit für Production-Deployment.")
        else:
            print_error("\n❌ EINIGE TESTS SIND FEHLGESCHLAGEN!")
            print_warning("Bitte behebe die Fehler vor dem Production-Deployment.")

def main():
    """Hauptfunktion"""
    print_section("🧪 Production Readiness Test")
    print_info("Starte umfassende Prüfung vor Live-Deployment...")
    
    tester = ProductionReadinessTest()
    
    # Phase 1: Basis-Infrastruktur
    tester.test_health_check()
    tester.test_data_availability()
    
    # Phase 2: Modell-Erstellung
    if tester.test_create_minimal_model():
        if tester.created_model_ids:
            model_id = tester.created_model_ids[0]
            tester.test_model_labels(model_id)
            tester.test_model_testing(model_id)
    
    # Phase 3: Modell-Vergleich (nur wenn 2 Modelle vorhanden)
    if len(tester.created_model_ids) >= 2:
        tester.test_model_comparison(tester.created_model_ids[0], tester.created_model_ids[1])
    
    # Phase 4: Dateien & Konfiguration
    tester.test_file_structure()
    tester.test_sql_migrations()
    
    # Zusammenfassung
    tester.print_summary()

if __name__ == "__main__":
    main()

