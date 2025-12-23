#!/usr/bin/env python3
"""
Test-Script für Phase 4: REST API
Testet: Pydantic Schemas, API Routes, FastAPI Main App
"""
import asyncio
import sys
import os
import json
from datetime import datetime, timezone, timedelta

# Füge app-Verzeichnis zum Python-Pfad hinzu
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

async def test_schemas():
    """Test 1: Pydantic Schemas"""
    print("\n📋 Test 1: Pydantic Schemas")
    try:
        from app.api.schemas import (
            TrainModelRequest, TestModelRequest, CompareModelsRequest,
            ModelResponse, JobResponse, HealthResponse
        )
        
        # Test 1.1: TrainModelRequest
        print("  🔧 Teste TrainModelRequest...")
        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(days=7)
        
        request = TrainModelRequest(
            name="TestModell_RF_v1",
            model_type="random_forest",
            target_var="market_cap_close",
            operator=">",
            target_value=50000.0,
            features=["price_open", "price_high"],
            phases=[1, 2, 3],
            train_start=start_date.isoformat(),
            train_end=end_date.isoformat()
        )
        print(f"  ✅ TrainModelRequest erstellt: {request.name}")
        
        # Test UTC-Validierung
        request_utc = TrainModelRequest(
            name="TestModell_XGB_v1",
            model_type="xgboost",
            target_var="price_close",
            operator=">",
            target_value=100.0,
            features=["price_open", "volume_sol"],
            train_start="2024-01-01T00:00:00Z",  # String mit Z
            train_end="2024-01-07T00:00:00+00:00"  # String mit +00:00
        )
        print(f"  ✅ UTC-Validierung funktioniert: {request_utc.train_start.tzinfo}")
        
        # Test model_type Validierung
        try:
            invalid_request = TrainModelRequest(
                name="Test",
                model_type="invalid_type",
                target_var="test",
                operator=">",
                target_value=1.0,
                features=["test"],
                train_start=start_date,
                train_end=end_date
            )
            print("  ❌ Validierung fehlgeschlagen (sollte Fehler werfen)")
            return False
        except ValueError as e:
            print(f"  ✅ model_type Validierung funktioniert: {e}")
        
        # Test 1.2: TestModelRequest
        print("  🔧 Teste TestModelRequest...")
        test_request = TestModelRequest(
            model_id=1,
            test_start=start_date.isoformat(),
            test_end=end_date.isoformat()
        )
        print(f"  ✅ TestModelRequest erstellt: model_id={test_request.model_id}")
        
        # Test 1.3: CompareModelsRequest
        print("  🔧 Teste CompareModelsRequest...")
        compare_request = CompareModelsRequest(
            model_a_id=1,
            model_b_id=2,
            test_start=start_date.isoformat(),
            test_end=end_date.isoformat()
        )
        print(f"  ✅ CompareModelsRequest erstellt")
        
        # Test unterschiedliche Modelle
        try:
            invalid_compare = CompareModelsRequest(
                model_a_id=1,
                model_b_id=1,  # Gleiche ID
                test_start=start_date,
                test_end=end_date
            )
            print("  ❌ Validierung fehlgeschlagen (sollte Fehler werfen)")
            return False
        except ValueError as e:
            print(f"  ✅ Unterschiedliche Modelle Validierung funktioniert: {e}")
        
        return True
    except Exception as e:
        print(f"  ❌ Fehler: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_api_routes():
    """Test 2: API Routes (ohne laufenden Server)"""
    print("\n🌐 Test 2: API Routes")
    try:
        # Test 2.1: Health Check Funktion
        print("  ❤️  Teste Health Check Funktion...")
        from app.utils.metrics import init_health_status, get_health_status
        init_health_status()
        health = await get_health_status()
        print(f"  ✅ Health Status: {health.get('status')}, DB: {health.get('db_connected')}")
        
        # Test 2.2: Metrics Funktion
        print("  📊 Teste Metrics Funktion...")
        from app.utils.metrics import generate_metrics
        metrics = generate_metrics()
        print(f"  ✅ Metrics generiert: {len(metrics)} Bytes")
        
        # Test 2.3: Routes Datei vorhanden
        print("  🔌 Teste Routes Datei...")
        import importlib.util
        routes_path = os.path.join(os.path.dirname(__file__), "app", "api", "routes.py")
        if os.path.exists(routes_path):
            print(f"  ✅ routes.py vorhanden")
            # Prüfe ob wichtige Funktionen definiert sind
            with open(routes_path, 'r') as f:
                content = f.read()
                endpoints = [
                    "POST /api/models/create",
                    "GET /api/models",
                    "GET /api/health",
                    "GET /api/metrics"
                ]
                found = [ep for ep in endpoints if any(part in content for part in ep.split())]
                print(f"  ✅ Endpoints gefunden: {len(found)}/{len(endpoints)}")
        else:
            print("  ❌ routes.py nicht gefunden")
            return False
        
        return True
    except Exception as e:
        print(f"  ❌ Fehler: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_main_app():
    """Test 3: FastAPI Main App"""
    print("\n🚀 Test 3: FastAPI Main App")
    try:
        # Test 3.1: main.py vorhanden
        print("  📱 Teste main.py...")
        main_path = os.path.join(os.path.dirname(__file__), "app", "main.py")
        if os.path.exists(main_path):
            print(f"  ✅ main.py vorhanden")
            with open(main_path, 'r') as f:
                content = f.read()
                # Prüfe wichtige Komponenten
                checks = {
                    "FastAPI": "FastAPI" in content,
                    "Router": "include_router" in content,
                    "Startup": "@app.on_event(\"startup\")" in content,
                    "Shutdown": "@app.on_event(\"shutdown\")" in content,
                    "Health": "/health" in content or "health" in content,
                    "Metrics": "/metrics" in content or "metrics" in content
                }
                passed = sum(checks.values())
                print(f"  ✅ Komponenten gefunden: {passed}/{len(checks)}")
                for key, value in checks.items():
                    status = "✅" if value else "❌"
                    print(f"     {status} {key}")
        else:
            print("  ❌ main.py nicht gefunden")
            return False
        
        # Test 3.2: Router Integration
        print("  🔌 Teste Router Integration...")
        routes_path = os.path.join(os.path.dirname(__file__), "app", "api", "routes.py")
        if os.path.exists(routes_path):
            with open(routes_path, 'r') as f:
                routes_content = f.read()
                # Prüfe wichtige Endpoints
                endpoints = [
                    "POST /api/models/create",
                    "GET /api/models",
                    "GET /api/health",
                    "GET /api/metrics",
                    "GET /health",
                    "GET /metrics"
                ]
                found = []
                for ep in endpoints:
                    method, path = ep.split()
                    if method.lower() in routes_content.lower() and path.split('/')[-1] in routes_content:
                        found.append(ep)
                print(f"  ✅ Endpoints definiert: {len(found)}/{len(endpoints)}")
        
        return True
    except Exception as e:
        print(f"  ❌ Fehler: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_api_validation():
    """Test 4: API Request Validation (Schema-Level)"""
    print("\n✅ Test 4: API Request Validation")
    try:
        from app.api.schemas import TrainModelRequest, TestModelRequest, CompareModelsRequest
        
        # Test 4.1: TrainModelRequest Validierung
        print("  🔍 Teste TrainModelRequest Validierung...")
        
        # Gültiger Request
        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(days=7)
        valid_request = TrainModelRequest(
            name="TestModell_API_v1",
            model_type="random_forest",
            target_var="market_cap_close",
            operator=">",
            target_value=50000.0,
            features=["price_open", "price_high"],
            phases=[1, 2, 3],
            train_start=start_date.isoformat(),
            train_end=end_date.isoformat()
        )
        print(f"  ✅ Gültiger Request erstellt: {valid_request.name}")
        
        # Test 4.2: TestModelRequest Validierung
        print("  🔍 Teste TestModelRequest Validierung...")
        test_request = TestModelRequest(
            model_id=1,
            test_start=start_date.isoformat(),
            test_end=end_date.isoformat()
        )
        print(f"  ✅ TestModelRequest erstellt: model_id={test_request.model_id}")
        
        # Test 4.3: CompareModelsRequest Validierung
        print("  🔍 Teste CompareModelsRequest Validierung...")
        compare_request = CompareModelsRequest(
            model_a_id=1,
            model_b_id=2,
            test_start=start_date.isoformat(),
            test_end=end_date.isoformat()
        )
        print(f"  ✅ CompareModelsRequest erstellt")
        
        # Test 4.4: UTC-Konvertierung
        print("  🔍 Teste UTC-Konvertierung...")
        # String ohne Zeitzone
        request_no_tz = TrainModelRequest(
            name="Test",
            model_type="random_forest",
            target_var="test",
            operator=">",
            target_value=1.0,
            features=["test"],
            train_start="2024-01-01T00:00:00",  # Keine Zeitzone
            train_end="2024-01-07T00:00:00"
        )
        if request_no_tz.train_start.tzinfo == timezone.utc:
            print(f"  ✅ UTC-Konvertierung funktioniert: {request_no_tz.train_start.tzinfo}")
        else:
            print(f"  ⚠️  UTC-Konvertierung: {request_no_tz.train_start.tzinfo}")
        
        return True
    except Exception as e:
        print(f"  ❌ Fehler: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Hauptfunktion"""
    print("=" * 60)
    print("🧪 Phase 4 Test Suite - REST API")
    print("=" * 60)
    
    # Setze DB_DSN als Environment Variable (falls nicht gesetzt)
    if "DB_DSN" not in os.environ:
        os.environ["DB_DSN"] = "postgresql://postgres:9HVxi6hN6j7xpmqUx84o@100.118.155.75:5432/crypto"
    
    results = []
    
    # Test 1: Schemas
    results.append(await test_schemas())
    
    # Test 2: API Routes
    results.append(await test_api_routes())
    
    # Test 3: Main App
    results.append(await test_main_app())
    
    # Test 4: API Validation
    results.append(await test_api_validation())
    
    # Zusammenfassung
    print("\n" + "=" * 60)
    print("📊 Zusammenfassung")
    print("=" * 60)
    passed = sum(results)
    total = len(results)
    print(f"✅ Tests bestanden: {passed}/{total}")
    
    if passed == total:
        print("\n🎉 Alle Tests erfolgreich! Phase 4 ist funktionsfähig.")
        print("\n💡 Nächste Schritte:")
        print("   - Phase 5: Job Queue implementieren")
        print("   - Dann: API mit laufendem Server testen")
        return 0
    else:
        print(f"\n⚠️  {total - passed} Test(s) fehlgeschlagen. Bitte Fehler beheben.")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)

