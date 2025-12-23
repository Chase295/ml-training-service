#!/usr/bin/env python3
"""
Test-Script für Phase 5: Job Queue
Testet: Job Manager, Worker, Integration
"""
import asyncio
import sys
import os
from datetime import datetime, timezone, timedelta

# Füge app-Verzeichnis zum Python-Pfad hinzu
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

async def test_job_manager_imports():
    """Test 1: Job Manager Imports"""
    print("\n📦 Test 1: Job Manager Imports")
    try:
        from app.queue.job_manager import (
            process_job, process_train_job, process_test_job,
            process_compare_job, start_worker
        )
        print("  ✅ Alle Funktionen importiert")
        print(f"     - process_job: {process_job}")
        print(f"     - process_train_job: {process_train_job}")
        print(f"     - process_test_job: {process_test_job}")
        print(f"     - process_compare_job: {process_compare_job}")
        print(f"     - start_worker: {start_worker}")
        return True
    except Exception as e:
        print(f"  ❌ Fehler: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_job_manager_structure():
    """Test 2: Job Manager Struktur"""
    print("\n🏗️  Test 2: Job Manager Struktur")
    try:
        import inspect
        from app.queue.job_manager import process_job, start_worker
        
        # Prüfe process_job Signatur
        sig = inspect.signature(process_job)
        params = list(sig.parameters.keys())
        print(f"  ✅ process_job Parameter: {params}")
        if 'job_id' not in params:
            print("  ❌ process_job sollte 'job_id' Parameter haben")
            return False
        
        # Prüfe start_worker Signatur
        sig = inspect.signature(start_worker)
        params = list(sig.parameters.keys())
        print(f"  ✅ start_worker Parameter: {params}")
        
        # Prüfe ob Funktionen async sind
        if not inspect.iscoroutinefunction(process_job):
            print("  ❌ process_job sollte async sein")
            return False
        if not inspect.iscoroutinefunction(start_worker):
            print("  ❌ start_worker sollte async sein")
            return False
        
        print("  ✅ Alle Funktionen sind async")
        return True
    except Exception as e:
        print(f"  ❌ Fehler: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_job_db_functions():
    """Test 3: Job DB Funktionen"""
    print("\n💾 Test 3: Job DB Funktionen")
    try:
        from app.database.models import (
            get_next_pending_job, get_job, update_job_status, create_job
        )
        
        # Prüfe ob Funktionen vorhanden sind
        print("  ✅ DB-Funktionen vorhanden:")
        print(f"     - get_next_pending_job: {get_next_pending_job}")
        print(f"     - get_job: {get_job}")
        print(f"     - update_job_status: {update_job_status}")
        print(f"     - create_job: {create_job}")
        
        # Prüfe ob get_next_pending_job None zurückgibt (keine Jobs in DB)
        job = await get_next_pending_job()
        if job is None:
            print("  ✅ Keine PENDING Jobs in DB (erwartet)")
        else:
            print(f"  ⚠️  PENDING Job gefunden: {job['id']}")
        
        return True
    except Exception as e:
        print(f"  ❌ Fehler: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_main_integration():
    """Test 4: Main App Integration"""
    print("\n🔌 Test 4: Main App Integration")
    try:
        main_path = os.path.join(os.path.dirname(__file__), "app", "main.py")
        if os.path.exists(main_path):
            with open(main_path, 'r') as f:
                content = f.read()
                
                # Prüfe ob Worker importiert wird
                if "from app.queue.job_manager import start_worker" in content:
                    print("  ✅ Worker Import vorhanden")
                else:
                    print("  ❌ Worker Import fehlt")
                    return False
                
                # Prüfe ob Worker gestartet wird
                if "asyncio.create_task(start_worker())" in content:
                    print("  ✅ Worker wird gestartet")
                else:
                    print("  ❌ Worker wird nicht gestartet")
                    return False
                
                # Prüfe ob TODO-Kommentar entfernt wurde
                if "# TODO: Uncomment wenn Phase 5 fertig ist" in content:
                    print("  ⚠️  TODO-Kommentar noch vorhanden (sollte entfernt werden)")
                else:
                    print("  ✅ TODO-Kommentar entfernt")
                
                return True
        else:
            print("  ❌ main.py nicht gefunden")
            return False
    except Exception as e:
        print(f"  ❌ Fehler: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_job_creation_flow():
    """Test 5: Job Creation Flow (ohne tatsächliche Ausführung)"""
    print("\n🔄 Test 5: Job Creation Flow")
    try:
        from app.database.models import create_job, get_job
        
        # Erstelle Test-Job (wird nicht verarbeitet, da Worker nicht läuft)
        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(days=7)
        
        job_id = await create_job(
            job_type="TRAIN",
            priority=5,
            train_model_type="random_forest",
            train_target_var="market_cap_close",
            train_operator=">",
            train_value=50000.0,
            train_start=start_date,
            train_end=end_date,
            train_features=["price_open", "price_high"],
            train_phases=[1, 2, 3],
            train_params={"n_estimators": 10},
            progress_msg="TestModell_Phase5_v1"  # ⚠️ WICHTIG: Name in progress_msg
        )
        
        print(f"  ✅ Test-Job erstellt: ID {job_id}")
        
        # Hole Job zurück
        job = await get_job(job_id)
        if job:
            print(f"  ✅ Job abgerufen: Status={job['status']}, Name={job['progress_msg']}")
            if job['progress_msg'] == "TestModell_Phase5_v1":
                print("  ✅ Modell-Name korrekt in progress_msg gespeichert")
            else:
                print(f"  ❌ Modell-Name falsch: {job['progress_msg']}")
                return False
        else:
            print("  ❌ Job nicht gefunden")
            return False
        
        # Cleanup: Setze Job auf CANCELLED (für Test)
        from app.database.models import update_job_status
        await update_job_status(job_id, status="CANCELLED", progress_msg="Test-Job")
        print(f"  ✅ Test-Job auf CANCELLED gesetzt (Cleanup)")
        
        return True
    except Exception as e:
        print(f"  ❌ Fehler: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Hauptfunktion"""
    print("=" * 60)
    print("🧪 Phase 5 Test Suite - Job Queue")
    print("=" * 60)
    
    # Setze DB_DSN als Environment Variable (falls nicht gesetzt)
    if "DB_DSN" not in os.environ:
        os.environ["DB_DSN"] = "postgresql://postgres:9HVxi6hN6j7xpmqUx84o@100.118.155.75:5432/crypto"
    
    results = []
    
    # Test 1: Imports
    results.append(await test_job_manager_imports())
    
    # Test 2: Struktur
    results.append(await test_job_manager_structure())
    
    # Test 3: DB Funktionen
    results.append(await test_job_db_functions())
    
    # Test 4: Main Integration
    results.append(await test_main_integration())
    
    # Test 5: Job Creation Flow
    results.append(await test_job_creation_flow())
    
    # Zusammenfassung
    print("\n" + "=" * 60)
    print("📊 Zusammenfassung")
    print("=" * 60)
    passed = sum(results)
    total = len(results)
    print(f"✅ Tests bestanden: {passed}/{total}")
    
    if passed == total:
        print("\n🎉 Alle Tests erfolgreich! Phase 5 ist funktionsfähig.")
        print("\n💡 Nächste Schritte:")
        print("   - Docker-Container bauen und testen")
        print("   - Worker mit echten Jobs testen")
        print("   - Streamlit UI implementieren (Phase 6)")
        return 0
    else:
        print(f"\n⚠️  {total - passed} Test(s) fehlgeschlagen. Bitte Fehler beheben.")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)

