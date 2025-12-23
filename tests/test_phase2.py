#!/usr/bin/env python3
"""
Test-Script für Phase 2: Core-Komponenten
Testet: Config, DB-Verbindung, Models, Metrics
"""
import asyncio
import sys
import os

# Füge app-Verzeichnis zum Python-Pfad hinzu
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

async def test_config():
    """Test 1: Config"""
    print("\n🔧 Test 1: Config")
    try:
        from app.utils.config import DB_DSN, API_PORT, STREAMLIT_PORT, MODEL_STORAGE_PATH
        print(f"  ✅ DB_DSN: {DB_DSN.split('@')[1] if '@' in DB_DSN else 'localhost'}")
        print(f"  ✅ API_PORT: {API_PORT}")
        print(f"  ✅ STREAMLIT_PORT: {STREAMLIT_PORT}")
        print(f"  ✅ MODEL_STORAGE_PATH: {MODEL_STORAGE_PATH}")
        return True
    except Exception as e:
        print(f"  ❌ Fehler: {e}")
        return False

async def test_db_connection():
    """Test 2: DB-Verbindung"""
    print("\n🔌 Test 2: Datenbank-Verbindung")
    try:
        from app.database.connection import get_pool, test_connection, close_pool
        
        # Test Verbindung
        connected = await test_connection()
        if connected:
            print("  ✅ DB-Verbindung erfolgreich!")
            
            # Test einfache Query
            pool = await get_pool()
            result = await pool.fetchval("SELECT 1")
            print(f"  ✅ Query-Test erfolgreich: {result}")
            
            # Test ref_model_types
            count = await pool.fetchval("SELECT COUNT(*) FROM ref_model_types")
            print(f"  ✅ ref_model_types gefunden: {count} Einträge")
            
            await close_pool()
            return True
        else:
            print("  ❌ DB-Verbindung fehlgeschlagen!")
            return False
    except Exception as e:
        print(f"  ❌ Fehler: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_models():
    """Test 3: Datenbank-Modelle"""
    print("\n📊 Test 3: Datenbank-Modelle")
    try:
        from app.database.models import get_model_type_defaults, list_models
        
        # Test get_model_type_defaults
        defaults_rf = await get_model_type_defaults("random_forest")
        print(f"  ✅ Random Forest Defaults: {defaults_rf}")
        
        defaults_xgb = await get_model_type_defaults("xgboost")
        print(f"  ✅ XGBoost Defaults: {defaults_xgb}")
        
        # Test list_models (sollte leer sein am Anfang)
        models = await list_models()
        print(f"  ✅ list_models() funktioniert: {len(models)} Modelle gefunden")
        
        return True
    except Exception as e:
        print(f"  ❌ Fehler: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_metrics():
    """Test 4: Metrics"""
    print("\n📈 Test 4: Prometheus Metrics")
    try:
        from app.utils.metrics import (
            init_health_status, get_health_status, generate_metrics,
            increment_job_counter, update_model_count
        )
        
        # Init Health Status
        init_health_status()
        print("  ✅ Health Status initialisiert")
        
        # Test Health Status
        health = await get_health_status()
        print(f"  ✅ Health Status: {health['status']}")
        print(f"     DB Connected: {health['db_connected']}")
        print(f"     Uptime: {health['uptime_seconds']}s")
        
        # Test Metrics Generation
        metrics = generate_metrics()
        print(f"  ✅ Metrics generiert: {len(metrics)} Bytes")
        
        # Test Counter
        increment_job_counter("TRAIN", "PENDING")
        print("  ✅ Job Counter erhöht")
        
        # Test Gauge
        update_model_count(0)
        print("  ✅ Model Count aktualisiert")
        
        return True
    except Exception as e:
        print(f"  ❌ Fehler: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Hauptfunktion"""
    print("=" * 60)
    print("🧪 Phase 2 Test Suite")
    print("=" * 60)
    
    results = []
    
    # Test 1: Config
    results.append(await test_config())
    
    # Test 2: DB-Verbindung
    results.append(await test_db_connection())
    
    # Test 3: Models
    results.append(await test_models())
    
    # Test 4: Metrics
    results.append(await test_metrics())
    
    # Zusammenfassung
    print("\n" + "=" * 60)
    print("📊 Zusammenfassung")
    print("=" * 60)
    passed = sum(results)
    total = len(results)
    print(f"✅ Tests bestanden: {passed}/{total}")
    
    if passed == total:
        print("\n🎉 Alle Tests erfolgreich! Phase 2 ist funktionsfähig.")
        return 0
    else:
        print(f"\n⚠️  {total - passed} Test(s) fehlgeschlagen. Bitte Fehler beheben.")
        return 1

if __name__ == "__main__":
    # Setze DB_DSN als Environment Variable (falls nicht gesetzt)
    if "DB_DSN" not in os.environ:
        os.environ["DB_DSN"] = "postgresql://postgres:9HVxi6hN6j7xpmqUx84o@100.118.155.75:5432/crypto"
    
    exit_code = asyncio.run(main())
    sys.exit(exit_code)

