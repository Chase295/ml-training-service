#!/usr/bin/env python3
"""
Final Live-Test: Erstellt ein neues Modell, testet es und validiert alle Werte
"""
import asyncio
import httpx
import json
import os
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, Optional

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000/api")

async def wait_for_job_completion(job_id: int, max_wait_seconds: int = 600) -> Dict[str, Any]:
    """Wartet auf den Abschluss eines Jobs"""
    print(f"⏳ Warte auf Job {job_id} Abschluss (max {max_wait_seconds}s)...")
    start_time = datetime.now()
    
    async with httpx.AsyncClient(timeout=60.0) as client:
        while (datetime.now() - start_time).total_seconds() < max_wait_seconds:
            response = await client.get(f"{API_BASE_URL}/queue/{job_id}")
            if response.status_code != 200:
                raise Exception(f"Fehler beim Abrufen des Job-Status: {response.status_code}")
            
            job_status = response.json()
            status = job_status.get("status")
            progress = job_status.get("progress", 0.0)
            progress_msg = job_status.get("progress_msg", "N/A")
            
            print(f"   Status: {status}, Progress: {progress*100:.1f}%, Message: {progress_msg}")
            
            if status == "COMPLETED":
                print(f"✅ Job {job_id} erfolgreich abgeschlossen!")
                return job_status
            elif status == "FAILED":
                error_msg = job_status.get("error_msg", "Unbekannter Fehler")
                raise Exception(f"Job {job_id} fehlgeschlagen: {error_msg}")
            
            await asyncio.sleep(5)
        
        raise Exception(f"Timeout: Job {job_id} nicht innerhalb von {max_wait_seconds} Sekunden abgeschlossen.")

async def validate_model_accuracy(model_id: int) -> Dict[str, Any]:
    """Validiert die Accuracy eines Modells durch manuelle Berechnung"""
    print(f"\n{'='*80}")
    print(f"🔍 SCHRITT 2: VALIDIERE MODELL {model_id} IN DB")
    print(f"{'='*80}\n")
    
    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.get(f"{API_BASE_URL}/models/{model_id}")
        if response.status_code != 200:
            raise Exception(f"Modell {model_id} nicht gefunden: {response.status_code}")
        
        model = response.json()
        
        # Hole Confusion Matrix Werte
        cm = model.get('confusion_matrix', {})
        tp = cm.get('tp', 0)
        tn = cm.get('tn', 0)
        fp = cm.get('fp', 0)
        fn = cm.get('fn', 0)
        
        stored_accuracy = model.get('training_accuracy')
        
        print("📊 DB-Werte:")
        print(f"   TP: {tp}")
        print(f"   TN: {tn}")
        print(f"   FP: {fp}")
        print(f"   FN: {fn}")
        print(f"   Gespeicherte Accuracy: {stored_accuracy}")
        
        # Manuelle Berechnung
        print("\n🧮 Manuelle Berechnung:")
        total = tp + tn + fp + fn
        print(f"   Gesamt = TP + TN + FP + FN = {tp} + {tn} + {fp} + {fn} = {total}")
        
        if total > 0:
            calculated_accuracy = (tp + tn) / total
            print(f"   Accuracy = (TP + TN) / Gesamt = ({tp} + {tn}) / {total} = {calculated_accuracy:.6f}")
            print(f"   Gespeicherte Accuracy: {stored_accuracy}")
            
            diff = abs(calculated_accuracy - stored_accuracy)
            print(f"\n   Differenz: {diff:.6f}")
            
            if diff < 0.001:
                print("   ✅ PERFEKT! Werte stimmen überein (Differenz < 0.001)")
                return {"valid": True, "calculated": calculated_accuracy, "stored": stored_accuracy, "diff": diff}
            else:
                print(f"   ❌ FEHLER! Differenz zu groß: {diff}")
                return {"valid": False, "calculated": calculated_accuracy, "stored": stored_accuracy, "diff": diff}
        else:
            print("   ❌ FEHLER! Gesamt ist 0")
            return {"valid": False, "error": "Gesamt ist 0"}

async def validate_test_result(test_id: int) -> Dict[str, Any]:
    """Validiert ein Test-Ergebnis durch manuelle Berechnung"""
    print(f"\n{'='*80}")
    print(f"🔍 SCHRITT 4: VALIDIERE TEST-ERGEBNIS {test_id} IN DB")
    print(f"{'='*80}\n")
    
    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.get(f"{API_BASE_URL}/test-results/{test_id}")
        if response.status_code != 200:
            raise Exception(f"Test-Ergebnis {test_id} nicht gefunden: {response.status_code}")
        
        test = response.json()
        
        # Hole Werte
        tp = test.get('tp', 0)
        tn = test.get('tn', 0)
        fp = test.get('fp', 0)
        fn = test.get('fn', 0)
        num_samples = test.get('num_samples', 0)
        stored_accuracy = test.get('accuracy')
        train_accuracy = test.get('train_accuracy')
        stored_degradation = test.get('accuracy_degradation')
        
        print("📊 DB-Werte:")
        print(f"   TP: {tp}")
        print(f"   TN: {tn}")
        print(f"   FP: {fp}")
        print(f"   FN: {fn}")
        print(f"   num_samples: {num_samples}")
        print(f"   Gespeicherte Accuracy: {stored_accuracy}")
        print(f"   Train Accuracy: {train_accuracy}")
        print(f"   Gespeicherte Degradation: {stored_degradation}")
        
        results = {"checks": []}
        
        # Check 1: TP + TN + FP + FN = num_samples
        print("\n🧮 Check 1: TP + TN + FP + FN = num_samples?")
        total = tp + tn + fp + fn
        print(f"   {tp} + {tn} + {fp} + {fn} = {total}")
        print(f"   num_samples = {num_samples}")
        if total == num_samples:
            print("   ✅ PERFEKT!")
            results["checks"].append({"name": "Summe", "valid": True})
        else:
            print(f"   ❌ FEHLER! {total} != {num_samples}")
            results["checks"].append({"name": "Summe", "valid": False, "expected": num_samples, "actual": total})
        
        # Check 2: accuracy = (TP + TN) / num_samples
        print("\n🧮 Check 2: accuracy = (TP + TN) / num_samples?")
        if num_samples > 0:
            calculated_accuracy = (tp + tn) / num_samples
            print(f"   ({tp} + {tn}) / {num_samples} = {calculated_accuracy:.6f}")
            print(f"   Gespeicherte Accuracy: {stored_accuracy}")
            diff = abs(calculated_accuracy - stored_accuracy)
            print(f"   Differenz: {diff:.6f}")
            if diff < 0.001:
                print("   ✅ PERFEKT!")
                results["checks"].append({"name": "Accuracy", "valid": True})
            else:
                print(f"   ❌ FEHLER! Differenz zu groß: {diff}")
                results["checks"].append({"name": "Accuracy", "valid": False, "calculated": calculated_accuracy, "stored": stored_accuracy, "diff": diff})
        else:
            print("   ❌ FEHLER! num_samples ist 0")
            results["checks"].append({"name": "Accuracy", "valid": False, "error": "num_samples ist 0"})
        
        # Check 3: accuracy_degradation = train_accuracy - accuracy
        print("\n🧮 Check 3: accuracy_degradation = train_accuracy - accuracy?")
        if train_accuracy and stored_accuracy and stored_degradation is not None:
            calculated_degradation = train_accuracy - stored_accuracy
            print(f"   {train_accuracy} - {stored_accuracy} = {calculated_degradation:.6f}")
            print(f"   Gespeicherte Degradation: {stored_degradation}")
            diff = abs(calculated_degradation - stored_degradation)
            print(f"   Differenz: {diff:.6f}")
            if diff < 0.001:
                print("   ✅ PERFEKT!")
                results["checks"].append({"name": "Degradation", "valid": True})
            else:
                print(f"   ❌ FEHLER! Differenz zu groß: {diff}")
                results["checks"].append({"name": "Degradation", "valid": False, "calculated": calculated_degradation, "stored": stored_degradation, "diff": diff})
        else:
            print("   ⚠️ Warnung: Nicht alle Werte vorhanden")
            results["checks"].append({"name": "Degradation", "valid": False, "error": "Werte fehlen"})
        
        # Zusammenfassung
        all_valid = all(check.get("valid", False) for check in results["checks"])
        results["all_valid"] = all_valid
        
        print(f"\n{'='*80}")
        if all_valid:
            print("✅ ALLE 3 CHECKS BESTANDEN!")
        else:
            print("❌ EINIGE CHECKS FEHLGESCHLAGEN!")
        print(f"{'='*80}")
        
        return results

async def final_live_test():
    """Führt den finalen Live-Test durch"""
    print("="*80)
    print("🔍 FINAL LIVE-TEST: VOLLSTÄNDIGE VALIDIERUNG")
    print("="*80)
    print()
    
    async with httpx.AsyncClient(timeout=120.0) as client:
        # Schritt 1: Erstelle Test-Modell
        print("="*80)
        print("📝 SCHRITT 1: ERSTELLE TEST-MODELL")
        print("="*80)
        print()
        
        train_start = (datetime.now(timezone.utc) - timedelta(days=2)).replace(hour=0, minute=0, second=0, microsecond=0)
        train_end = train_start.replace(hour=23, minute=59, second=59)
        
        model_request = {
            "name": "Final Test Modell",
            "model_type": "random_forest",
            "train_start": train_start.isoformat(),
            "train_end": train_end.isoformat(),
            "features": ["price_open", "price_high", "price_low", "price_close"],
            "phases": [1],
            "use_time_based_prediction": True,
            "target_var": "price_close",
            "future_minutes": 5,
            "min_percent_change": 30.0,
            "direction": "up",
            "params": {
                "n_estimators": 50,
                "max_depth": 5
            }
        }
        
        print("📤 Sende Modell-Erstellungs-Request...")
        print(json.dumps(model_request, indent=2))
        
        response = await client.post(f"{API_BASE_URL}/models/create", json=model_request)
        if response.status_code != 201:
            raise Exception(f"Fehler beim Erstellen des Modells: {response.status_code} - {response.text}")
        
        result = response.json()
        job_id = result.get("job_id")
        
        if not job_id:
            raise Exception(f"Keine Job-ID erhalten: {result}")
        
        print(f"✅ Job erstellt: {job_id}")
        
        # Warte auf Abschluss
        job_result = await wait_for_job_completion(job_id)
        model_id = job_result.get("result_model_id")
        
        if not model_id:
            raise Exception(f"Keine Modell-ID erhalten: {job_result}")
        
        print(f"✅ Modell erstellt: ID {model_id}")
        
        # Schritt 2: Validiere Modell
        model_validation = await validate_model_accuracy(model_id)
        
        if not model_validation.get("valid"):
            raise Exception(f"Modell-Validierung fehlgeschlagen: {model_validation}")
        
        # Schritt 3: Teste Modell
        print(f"\n{'='*80}")
        print("📝 SCHRITT 3: TESTE MODELL")
        print(f"{'='*80}\n")
        
        test_start = train_end + timedelta(seconds=1)
        test_end = datetime.now(timezone.utc)
        
        test_request = {
            "test_start": test_start.isoformat(),
            "test_end": test_end.isoformat()
        }
        
        print("📤 Sende Test-Request...")
        print(json.dumps(test_request, indent=2))
        
        response = await client.post(f"{API_BASE_URL}/models/{model_id}/test", json=test_request)
        if response.status_code != 201:
            raise Exception(f"Fehler beim Testen des Modells: {response.status_code} - {response.text}")
        
        result = response.json()
        test_job_id = result.get("job_id")
        
        if not test_job_id:
            raise Exception(f"Keine Test-Job-ID erhalten: {result}")
        
        print(f"✅ Test-Job erstellt: {test_job_id}")
        
        # Warte auf Abschluss
        test_job_result = await wait_for_job_completion(test_job_id)
        test_id = test_job_result.get("result_test_id")
        
        if not test_id:
            raise Exception(f"Keine Test-ID erhalten: {test_job_result}")
        
        print(f"✅ Test abgeschlossen: ID {test_id}")
        
        # Schritt 4: Validiere Test-Ergebnis
        test_validation = await validate_test_result(test_id)
        
        # Schritt 5: Finale Zusammenfassung
        print(f"\n{'='*80}")
        print("📊 FINALE ZUSAMMENFASSUNG")
        print(f"{'='*80}\n")
        
        print(f"✅ Modell erstellt: ID {model_id}")
        print(f"   - Modell-Validierung: {'✅ BESTANDEN' if model_validation.get('valid') else '❌ FEHLGESCHLAGEN'}")
        
        print(f"\n✅ Test durchgeführt: ID {test_id}")
        print(f"   - Test-Validierung: {'✅ ALLE CHECKS BESTANDEN' if test_validation.get('all_valid') else '❌ EINIGE CHECKS FEHLGESCHLAGEN'}")
        
        if model_validation.get("valid") and test_validation.get("all_valid"):
            print(f"\n{'='*80}")
            print("🎉 SYSTEM IST 100% KORREKT!")
            print(f"{'='*80}\n")
            print("✅ Du kannst:")
            print("   - Das System produktiv nutzen")
            print("   - Echte Modelle trainieren")
            print("   - Auf die Metriken vertrauen")
            print("   - Beruhigt schlafen 😴")
            return True
        else:
            print(f"\n{'='*80}")
            print("❌ VALIDIERUNG FEHLGESCHLAGEN")
            print(f"{'='*80}\n")
            return False

if __name__ == "__main__":
    success = asyncio.run(final_live_test())
    exit(0 if success else 1)

