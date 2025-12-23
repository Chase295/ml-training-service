#!/usr/bin/env python3
"""
Phase 8: End-to-End Testing
Testet alle Funktionen des ML Training Services
"""
import asyncio
import httpx
import json
from datetime import datetime, timezone, timedelta

# Konfiguration
API_BASE_URL = "http://localhost:8000"
TIMEOUT = 300  # 5 Minuten für Training

# Test-Ergebnisse
test_results = {
    "passed": [],
    "failed": [],
    "warnings": []
}

def log_test(name: str, passed: bool, message: str = ""):
    """Loggt Test-Ergebnis"""
    if passed:
        test_results["passed"].append(name)
        print(f"✅ {name}: {message}")
    else:
        test_results["failed"].append(name)
        print(f"❌ {name}: {message}")

def log_warning(name: str, message: str):
    """Loggt Warnung"""
    test_results["warnings"].append(f"{name}: {message}")
    print(f"⚠️  {name}: {message}")

async def test_health_check():
    """Test 1: Health Check"""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{API_BASE_URL}/api/health", timeout=10)
            if response.status_code == 200:
                data = response.json()
                if data.get("status") == "healthy" and data.get("db_connected"):
                    log_test("Health Check", True, f"Status: {data['status']}, DB: {data['db_connected']}")
                    return True
                else:
                    log_test("Health Check", False, f"Status nicht healthy: {data}")
                    return False
            else:
                log_test("Health Check", False, f"Status Code: {response.status_code}")
                return False
    except Exception as e:
        log_test("Health Check", False, str(e))
        return False

async def test_metrics():
    """Test 2: Prometheus Metrics"""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{API_BASE_URL}/api/metrics", timeout=10)
            if response.status_code == 200:
                content = response.text
                if "ml_service_uptime_seconds" in content and "ml_db_connected" in content:
                    log_test("Prometheus Metrics", True, "Metrics-Endpoint funktioniert")
                    return True
                else:
                    log_test("Prometheus Metrics", False, "Erwartete Metriken fehlen")
                    return False
            else:
                log_test("Prometheus Metrics", False, f"Status Code: {response.status_code}")
                return False
    except Exception as e:
        log_test("Prometheus Metrics", False, str(e))
        return False

async def test_list_models():
    """Test 3: Modelle auflisten"""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{API_BASE_URL}/api/models", timeout=10)
            if response.status_code == 200:
                models = response.json()
                log_test("Liste Modelle", True, f"{len(models)} Modelle gefunden")
                return models
            else:
                log_test("Liste Modelle", False, f"Status Code: {response.status_code}")
                return []
    except Exception as e:
        log_test("Liste Modelle", False, str(e))
        return []

async def test_create_model():
    """Test 4: Modell erstellen (Training)"""
    try:
        # Bereite Test-Daten vor
        train_end = datetime.now(timezone.utc)
        train_start = train_end - timedelta(days=7)  # 7 Tage Training
        
        model_data = {
            "name": f"TestModell_E2E_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}",
            "model_type": "random_forest",
            "target_var": "market_cap_close",
            "operator": ">",
            "target_value": 50000.0,
            "features": ["price_open", "price_high", "volume_sol"],
            "phases": [1, 2, 3],
            "train_start": train_start.isoformat(),
            "train_end": train_end.isoformat(),
            "params": {
                "n_estimators": 10,  # Kleine Werte für schnellen Test
                "max_depth": 5
            },
            "description": "E2E Test Modell"
        }
        
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            response = await client.post(
                f"{API_BASE_URL}/api/models/create",
                json=model_data,
                timeout=TIMEOUT
            )
            
            if response.status_code == 201:
                result = response.json()
                job_id = result.get("job_id")
                log_test("Modell erstellen (Job)", True, f"Job ID: {job_id}")
                
                # Warte auf Job-Abschluss
                await asyncio.sleep(2)  # Kurz warten
                
                # Prüfe Job-Status
                job_response = await client.get(f"{API_BASE_URL}/api/queue/{job_id}", timeout=10)
                if job_response.status_code == 200:
                    job = job_response.json()
                    log_test("Job-Status prüfen", True, f"Status: {job.get('status')}")
                    
                    # Warte maximal 5 Minuten auf Completion
                    max_wait = 300  # 5 Minuten
                    waited = 0
                    while waited < max_wait:
                        await asyncio.sleep(5)
                        waited += 5
                        job_response = await client.get(f"{API_BASE_URL}/api/queue/{job_id}", timeout=10)
                        job = job_response.json()
                        status = job.get("status")
                        
                        if status == "COMPLETED":
                            model_id = job.get("result_model_id")
                            log_test("Training abgeschlossen", True, f"Modell ID: {model_id}")
                            return model_id
                        elif status == "FAILED":
                            error = job.get("error_msg", "Unbekannter Fehler")
                            log_test("Training fehlgeschlagen", False, error)
                            return None
                        elif status in ["PENDING", "RUNNING"]:
                            progress = job.get("progress", 0)
                            log_warning("Training läuft", f"Status: {status}, Progress: {progress:.1%}")
                    
                    log_test("Training Timeout", False, "Job wurde nicht innerhalb von 5 Minuten abgeschlossen")
                    return None
                else:
                    log_test("Job-Status abrufen", False, f"Status Code: {job_response.status_code}")
                    return None
            else:
                error_text = response.text
                log_test("Modell erstellen", False, f"Status Code: {response.status_code}, Error: {error_text}")
                return None
    except Exception as e:
        log_test("Modell erstellen", False, str(e))
        return None

async def test_get_model(model_id: int):
    """Test 5: Modell-Details abrufen"""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{API_BASE_URL}/api/models/{model_id}", timeout=10)
            if response.status_code == 200:
                model = response.json()
                log_test("Modell-Details", True, f"Name: {model.get('name')}, Accuracy: {model.get('training_accuracy', 'N/A')}")
                return model
            else:
                log_test("Modell-Details", False, f"Status Code: {response.status_code}")
                return None
    except Exception as e:
        log_test("Modell-Details", False, str(e))
        return None

async def test_model(model_id: int):
    """Test 6: Modell testen"""
    try:
        # Test-Zeitraum (nach Training)
        test_end = datetime.now(timezone.utc)
        test_start = test_end - timedelta(days=2)  # 2 Tage Test
        
        test_data = {
            "test_start": test_start.isoformat(),
            "test_end": test_end.isoformat()
        }
        
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            response = await client.post(
                f"{API_BASE_URL}/api/models/{model_id}/test",
                json=test_data,
                timeout=TIMEOUT
            )
            
            if response.status_code == 201:
                result = response.json()
                job_id = result.get("job_id")
                log_test("Test-Job erstellen", True, f"Job ID: {job_id}")
                
                # Warte auf Job-Abschluss
                max_wait = 180  # 3 Minuten
                waited = 0
                while waited < max_wait:
                    await asyncio.sleep(5)
                    waited += 5
                    job_response = await client.get(f"{API_BASE_URL}/api/queue/{job_id}", timeout=10)
                    job = job_response.json()
                    status = job.get("status")
                    
                    if status == "COMPLETED":
                        test_id = job.get("result_test_id")
                        log_test("Test abgeschlossen", True, f"Test ID: {test_id}")
                        return test_id
                    elif status == "FAILED":
                        error = job.get("error_msg", "Unbekannter Fehler")
                        log_test("Test fehlgeschlagen", False, error)
                        return None
                
                log_test("Test Timeout", False, "Test wurde nicht innerhalb von 3 Minuten abgeschlossen")
                return None
            else:
                error_text = response.text
                log_test("Modell testen", False, f"Status Code: {response.status_code}, Error: {error_text}")
                return None
    except Exception as e:
        log_test("Modell testen", False, str(e))
        return None

async def test_compare_models(model_id_1: int, model_id_2: int):
    """Test 7: Modelle vergleichen"""
    try:
        # Test-Zeitraum
        test_end = datetime.now(timezone.utc)
        test_start = test_end - timedelta(days=2)
        
        compare_data = {
            "model_a_id": model_id_1,
            "model_b_id": model_id_2,
            "test_start": test_start.isoformat(),
            "test_end": test_end.isoformat()
        }
        
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            response = await client.post(
                f"{API_BASE_URL}/api/models/compare",
                json=compare_data,
                timeout=TIMEOUT
            )
            
            if response.status_code == 201:
                result = response.json()
                job_id = result.get("job_id")
                log_test("Vergleich-Job erstellen", True, f"Job ID: {job_id}")
                
                # Warte auf Job-Abschluss
                max_wait = 300  # 5 Minuten
                waited = 0
                while waited < max_wait:
                    await asyncio.sleep(5)
                    waited += 5
                    job_response = await client.get(f"{API_BASE_URL}/api/queue/{job_id}", timeout=10)
                    job = job_response.json()
                    status = job.get("status")
                    
                    if status == "COMPLETED":
                        comparison_id = job.get("result_comparison_id")
                        log_test("Vergleich abgeschlossen", True, f"Comparison ID: {comparison_id}")
                        return comparison_id
                    elif status == "FAILED":
                        error = job.get("error_msg", "Unbekannter Fehler")
                        log_test("Vergleich fehlgeschlagen", False, error)
                        return None
                
                log_test("Vergleich Timeout", False, "Vergleich wurde nicht innerhalb von 5 Minuten abgeschlossen")
                return None
            else:
                error_text = response.text
                log_test("Modelle vergleichen", False, f"Status Code: {response.status_code}, Error: {error_text}")
                return None
    except Exception as e:
        log_test("Modelle vergleichen", False, str(e))
        return None

async def test_queue_status():
    """Test 8: Queue-Status"""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{API_BASE_URL}/api/queue", timeout=10)
            if response.status_code == 200:
                jobs = response.json()
                log_test("Queue-Status", True, f"{len(jobs)} Jobs gefunden")
                return True
            else:
                log_test("Queue-Status", False, f"Status Code: {response.status_code}")
                return False
    except Exception as e:
        log_test("Queue-Status", False, str(e))
        return False

async def main():
    """Hauptfunktion: Führt alle Tests aus"""
    print("=" * 60)
    print("Phase 8: End-to-End Testing")
    print("=" * 60)
    print()
    
    # Basis-Tests
    print("📋 Basis-Funktionalität:")
    print("-" * 60)
    await test_health_check()
    await test_metrics()
    await test_queue_status()
    models = await test_list_models()
    print()
    
    # Training-Tests
    print("🤖 Training-Workflow:")
    print("-" * 60)
    model_id_1 = await test_create_model()
    print()
    
    if model_id_1:
        await test_get_model(model_id_1)
        print()
        
        # Testing-Tests
        print("🧪 Testing-Workflow:")
        print("-" * 60)
        test_id = await test_model(model_id_1)
        print()
        
        # Vergleich-Tests (nur wenn 2 Modelle vorhanden)
        if len(models) >= 1:
            # Erstelle zweites Modell für Vergleich
            print("🤖 Zweites Modell für Vergleich erstellen:")
            print("-" * 60)
            model_id_2 = await test_create_model()
            print()
            
            if model_id_2:
                print("⚔️ Vergleich-Workflow:")
                print("-" * 60)
                comparison_id = await test_compare_models(model_id_1, model_id_2)
                print()
    
    # Zusammenfassung
    print("=" * 60)
    print("📊 Test-Zusammenfassung:")
    print("=" * 60)
    print(f"✅ Bestanden: {len(test_results['passed'])}")
    print(f"❌ Fehlgeschlagen: {len(test_results['failed'])}")
    print(f"⚠️  Warnungen: {len(test_results['warnings'])}")
    print()
    
    if test_results['passed']:
        print("✅ Bestandene Tests:")
        for test in test_results['passed']:
            print(f"   - {test}")
        print()
    
    if test_results['failed']:
        print("❌ Fehlgeschlagene Tests:")
        for test in test_results['failed']:
            print(f"   - {test}")
        print()
    
    if test_results['warnings']:
        print("⚠️  Warnungen:")
        for warning in test_results['warnings']:
            print(f"   - {warning}")
        print()
    
    # Exit Code
    if test_results['failed']:
        print("❌ Einige Tests sind fehlgeschlagen!")
        exit(1)
    else:
        print("✅ Alle Tests bestanden!")
        exit(0)

if __name__ == "__main__":
    asyncio.run(main())

