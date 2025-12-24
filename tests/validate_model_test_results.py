#!/usr/bin/env python3
"""
Validierungs-Script für Modell-Erstellung und Testing
Prüft ob alle Werte korrekt sind und keine Fehler im Code sind
"""
import asyncio
import httpx
import json
from typing import Dict, Any, Optional

# API Base URL - kann über Umgebungsvariable überschrieben werden
import os
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000/api")

async def validate_model(model_id: int) -> Dict[str, Any]:
    """
    Validiert ein Modell - prüft ob alle Werte korrekt sind
    """
    async with httpx.AsyncClient(timeout=30.0) as client:
        print(f"\n{'='*60}")
        print(f"🔍 VALIDIERE MODELL {model_id}")
        print(f"{'='*60}\n")
        
        # 1. Hole Modell-Details
        response = await client.get(f"{API_BASE_URL}/models/{model_id}")
        if response.status_code != 200:
            return {"error": f"Modell {model_id} nicht gefunden: {response.status_code}"}
        
        model = response.json()
        
        errors = []
        warnings = []
        
        # 2. Prüfe Basis-Felder
        print("1️⃣ Prüfe Basis-Felder...")
        required_fields = ['id', 'name', 'model_type', 'status', 'train_start', 'train_end', 
                         'training_accuracy', 'training_f1', 'model_file_path']
        for field in required_fields:
            if field not in model or model[field] is None:
                errors.append(f"❌ Feld '{field}' fehlt oder ist None")
            else:
                print(f"   ✅ {field}: {model[field]}")
        
        # 3. Prüfe Status
        print("\n2️⃣ Prüfe Status...")
        if model['status'] != 'READY':
            warnings.append(f"⚠️ Status ist nicht 'READY': {model['status']}")
        else:
            print(f"   ✅ Status: {model['status']}")
        
        # 4. Prüfe Metriken
        print("\n3️⃣ Prüfe Training-Metriken...")
        accuracy = model.get('training_accuracy')
        f1 = model.get('training_f1')
        precision = model.get('training_precision')
        recall = model.get('training_recall')
        
        if accuracy is not None:
            if accuracy < 0.5:
                warnings.append(f"⚠️ Accuracy sehr niedrig: {accuracy:.4f} (kann bedeuten, dass Modell nicht lernt)")
            elif accuracy > 0.95:
                warnings.append(f"⚠️ Accuracy sehr hoch: {accuracy:.4f} (kann Overfitting bedeuten)")
            else:
                print(f"   ✅ Accuracy: {accuracy:.4f} (realistisch)")
        
        if f1 is not None:
            if f1 < 0.5:
                warnings.append(f"⚠️ F1-Score niedrig: {f1:.4f}")
            else:
                print(f"   ✅ F1-Score: {f1:.4f}")
        
        # 5. Prüfe Confusion Matrix
        print("\n4️⃣ Prüfe Confusion Matrix...")
        cm = model.get('confusion_matrix')
        if cm:
            tp = cm.get('tp', 0)
            tn = cm.get('tn', 0)
            fp = cm.get('fp', 0)
            fn = cm.get('fn', 0)
            
            total = tp + tn + fp + fn
            if total > 0:
                calculated_accuracy = (tp + tn) / total
                if accuracy and abs(calculated_accuracy - accuracy) > 0.001:
                    errors.append(f"❌ Accuracy stimmt nicht! Berechnet: {calculated_accuracy:.4f}, Gespeichert: {accuracy:.4f}")
                else:
                    print(f"   ✅ Confusion Matrix korrekt: TP={tp}, TN={tn}, FP={fp}, FN={fn}")
                    print(f"   ✅ Berechnete Accuracy: {calculated_accuracy:.4f} (stimmt mit gespeicherter überein)")
            else:
                errors.append("❌ Confusion Matrix summiert zu 0!")
        else:
            warnings.append("⚠️ Confusion Matrix fehlt")
        
        # 6. Prüfe Feature Importance
        print("\n5️⃣ Prüfe Feature Importance...")
        feature_importance = model.get('feature_importance')
        if feature_importance:
            total_importance = sum(feature_importance.values())
            if abs(total_importance - 1.0) > 0.01:
                errors.append(f"❌ Feature Importance summiert nicht zu 1.0: {total_importance:.4f}")
            else:
                print(f"   ✅ Feature Importance summiert zu: {total_importance:.4f}")
                # Zeige Top 5 Features
                sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
                print(f"   📊 Top 5 Features:")
                for feature, importance in sorted_features[:5]:
                    print(f"      - {feature}: {importance:.4f}")
        else:
            warnings.append("⚠️ Feature Importance fehlt")
        
        # 7. Prüfe Features
        print("\n6️⃣ Prüfe Features...")
        features = model.get('features', [])
        if not features:
            errors.append("❌ Keine Features gefunden!")
        else:
            print(f"   ✅ {len(features)} Features gefunden")
            # Prüfe ob engineered features vorhanden sind
            engineered_features = [f for f in features if any(x in f for x in ['_5', '_10', '_15', 'volatility', 'velocity', 'ratio'])]
            if engineered_features:
                print(f"   🔧 {len(engineered_features)} Engineered Features gefunden")
        
        # 8. Prüfe Zeitbasierte Vorhersage
        print("\n7️⃣ Prüfe Zeitbasierte Vorhersage...")
        if model.get('target_operator') is None or model.get('target_value') is None:
            future_minutes = model.get('future_minutes')
            price_change_percent = model.get('price_change_percent')
            target_direction = model.get('target_direction')
            
            if future_minutes and price_change_percent:
                print(f"   ✅ Zeitbasierte Vorhersage: {future_minutes} Min, {price_change_percent}%, Richtung: {target_direction}")
            else:
                warnings.append("⚠️ Zeitbasierte Vorhersage-Parameter fehlen")
        else:
            print(f"   ✅ Normale Vorhersage: {model.get('target_variable')} {model.get('target_operator')} {model.get('target_value')}")
        
        # 9. Prüfe CV-Scores
        print("\n8️⃣ Prüfe Cross-Validation...")
        cv_scores = model.get('cv_scores')
        if cv_scores:
            train_acc = cv_scores.get('train_accuracy_mean')
            test_acc = cv_scores.get('test_accuracy_mean')
            if train_acc and test_acc:
                gap = train_acc - test_acc
                print(f"   ✅ Train Accuracy: {train_acc:.4f}, Test Accuracy: {test_acc:.4f}")
                print(f"   ✅ Gap: {gap:.4f}")
                if gap > 0.1:
                    warnings.append(f"⚠️ OVERFITTING: Train-Test Gap zu groß: {gap:.4f}")
        else:
            warnings.append("⚠️ CV-Scores fehlen")
        
        # 10. Zusammenfassung
        print(f"\n{'='*60}")
        print("📊 VALIDIERUNGS-ERGEBNIS")
        print(f"{'='*60}\n")
        
        if errors:
            print("❌ FEHLER:")
            for error in errors:
                print(f"   {error}")
        
        if warnings:
            print("\n⚠️ WARNUNGEN:")
            for warning in warnings:
                print(f"   {warning}")
        
        if not errors and not warnings:
            print("✅ ALLE PRÜFUNGEN BESTANDEN!")
        
        return {
            "model_id": model_id,
            "errors": errors,
            "warnings": warnings,
            "valid": len(errors) == 0
        }

async def validate_test_result(test_id: int) -> Dict[str, Any]:
    """
    Validiert ein Test-Ergebnis - prüft ob alle Werte korrekt sind
    """
    async with httpx.AsyncClient(timeout=30.0) as client:
        print(f"\n{'='*60}")
        print(f"🔍 VALIDIERE TEST-ERGEBNIS {test_id}")
        print(f"{'='*60}\n")
        
        # 1. Hole Test-Details
        response = await client.get(f"{API_BASE_URL}/test-results/{test_id}")
        if response.status_code != 200:
            return {"error": f"Test {test_id} nicht gefunden: {response.status_code}"}
        
        test = response.json()
        
        errors = []
        warnings = []
        
        # 2. Prüfe Basis-Felder
        print("1️⃣ Prüfe Basis-Felder...")
        required_fields = ['id', 'model_id', 'test_start', 'test_end', 
                         'accuracy', 'f1_score', 'num_samples']
        for field in required_fields:
            if field not in test or test[field] is None:
                errors.append(f"❌ Feld '{field}' fehlt oder ist None")
            else:
                print(f"   ✅ {field}: {test[field]}")
        
        # 3. Prüfe Confusion Matrix
        print("\n2️⃣ Prüfe Confusion Matrix...")
        cm = test.get('confusion_matrix')
        tp = test.get('tp', 0)
        tn = test.get('tn', 0)
        fp = test.get('fp', 0)
        fn = test.get('fn', 0)
        num_samples = test.get('num_samples', 0)
        
        if cm:
            cm_tp = cm.get('tp', 0)
            cm_tn = cm.get('tn', 0)
            cm_fp = cm.get('fp', 0)
            cm_fn = cm.get('fn', 0)
            
            # Prüfe Konsistenz
            if tp != cm_tp or tn != cm_tn or fp != cm_fp or fn != cm_fn:
                errors.append(f"❌ Confusion Matrix Werte stimmen nicht überein!")
            else:
                print(f"   ✅ Confusion Matrix konsistent: TP={tp}, TN={tn}, FP={fp}, FN={fn}")
        else:
            warnings.append("⚠️ Confusion Matrix (JSONB) fehlt")
        
        # Prüfe ob Summe stimmt
        total = tp + tn + fp + fn
        if total != num_samples:
            errors.append(f"❌ Confusion Matrix summiert nicht zu num_samples! {total} != {num_samples}")
        else:
            print(f"   ✅ Summe stimmt: {total} = {num_samples}")
        
        # Prüfe Accuracy
        if total > 0:
            calculated_accuracy = (tp + tn) / total
            stored_accuracy = test.get('accuracy')
            if stored_accuracy and abs(calculated_accuracy - stored_accuracy) > 0.001:
                errors.append(f"❌ Accuracy stimmt nicht! Berechnet: {calculated_accuracy:.4f}, Gespeichert: {stored_accuracy:.4f}")
            else:
                print(f"   ✅ Accuracy korrekt: {calculated_accuracy:.4f}")
        
        # 4. Prüfe Label-Balance
        print("\n3️⃣ Prüfe Label-Balance...")
        num_positive = test.get('num_positive', 0)
        num_negative = test.get('num_negative', 0)
        
        if num_positive + num_negative != num_samples:
            errors.append(f"❌ Label-Balance stimmt nicht! {num_positive} + {num_negative} != {num_samples}")
        else:
            positive_ratio = num_positive / num_samples if num_samples > 0 else 0
            print(f"   ✅ Label-Balance: {positive_ratio:.2%} positive, {1-positive_ratio:.2%} negative")
            
            if positive_ratio < 0.1 or positive_ratio > 0.9:
                warnings.append(f"⚠️ Sehr unausgewogene Labels: {positive_ratio:.2%} positive")
        
        # 5. Prüfe Overlap
        print("\n4️⃣ Prüfe Overlap...")
        has_overlap = test.get('has_overlap', False)
        if has_overlap:
            warnings.append(f"⚠️ Test überlappt mit Training: {test.get('overlap_note', 'Unbekannt')}")
        else:
            print(f"   ✅ Kein Overlap mit Training")
        
        # 6. Prüfe Train vs. Test Vergleich
        print("\n5️⃣ Prüfe Train vs. Test Vergleich...")
        accuracy_degradation = test.get('accuracy_degradation')
        if accuracy_degradation is not None:
            if accuracy_degradation > 0.1:
                warnings.append(f"⚠️ OVERFITTING: Accuracy Degradation zu groß: {accuracy_degradation:.4f}")
            else:
                print(f"   ✅ Accuracy Degradation: {accuracy_degradation:.4f} (akzeptabel)")
        
        is_overfitted = test.get('is_overfitted', False)
        if is_overfitted:
            warnings.append("⚠️ Modell ist overfitted!")
        else:
            print(f"   ✅ Modell ist nicht overfitted")
        
        # 7. Prüfe Test-Dauer
        print("\n6️⃣ Prüfe Test-Dauer...")
        test_duration_days = test.get('test_duration_days')
        if test_duration_days:
            if test_duration_days < 1.0:
                warnings.append(f"⚠️ Test-Zeitraum zu kurz: {test_duration_days:.2f} Tage (empfohlen: mindestens 1 Tag)")
            else:
                print(f"   ✅ Test-Dauer: {test_duration_days:.2f} Tage")
        
        # 8. Prüfe zusätzliche Metriken
        print("\n7️⃣ Prüfe zusätzliche Metriken...")
        mcc = test.get('mcc')
        fpr = test.get('fpr')
        fnr = test.get('fnr')
        roc_auc = test.get('roc_auc')
        
        if mcc is not None:
            print(f"   ✅ MCC: {mcc:.4f}")
        if fpr is not None:
            print(f"   ✅ FPR: {fpr:.4f}")
        if fnr is not None:
            print(f"   ✅ FNR: {fnr:.4f}")
        if roc_auc is not None:
            print(f"   ✅ ROC-AUC: {roc_auc:.4f}")
        
        # 9. Zusammenfassung
        print(f"\n{'='*60}")
        print("📊 VALIDIERUNGS-ERGEBNIS")
        print(f"{'='*60}\n")
        
        if errors:
            print("❌ FEHLER:")
            for error in errors:
                print(f"   {error}")
        
        if warnings:
            print("\n⚠️ WARNUNGEN:")
            for warning in warnings:
                print(f"   {warning}")
        
        if not errors and not warnings:
            print("✅ ALLE PRÜFUNGEN BESTANDEN!")
        
        return {
            "test_id": test_id,
            "errors": errors,
            "warnings": warnings,
            "valid": len(errors) == 0
        }

async def validate_model_and_tests(model_id: int) -> Dict[str, Any]:
    """
    Validiert ein Modell und alle zugehörigen Tests
    """
    async with httpx.AsyncClient(timeout=30.0) as client:
        print(f"\n{'='*60}")
        print(f"🔍 VALIDIERE MODELL {model_id} UND ALLE TESTS")
        print(f"{'='*60}\n")
        
        # 1. Validiere Modell
        model_result = await validate_model(model_id)
        
        # 2. Hole alle Tests für dieses Modell
        response = await client.get(f"{API_BASE_URL}/test-results")
        if response.status_code == 200:
            all_tests = response.json()
            model_tests = [t for t in all_tests if t.get('model_id') == model_id]
            
            print(f"\n{'='*60}")
            print(f"📊 VALIDIERE {len(model_tests)} TEST-ERGEBNISSE")
            print(f"{'='*60}\n")
            
            test_results = []
            for test in model_tests:
                test_id = test['id']
                test_result = await validate_test_result(test_id)
                test_results.append(test_result)
            
            # 3. Zusammenfassung
            total_errors = len(model_result.get('errors', [])) + sum(len(r.get('errors', [])) for r in test_results)
            total_warnings = len(model_result.get('warnings', [])) + sum(len(r.get('warnings', [])) for r in test_results)
            
            print(f"\n{'='*60}")
            print("📊 GESAMT-VALIDIERUNG")
            print(f"{'='*60}\n")
            print(f"✅ Modell validiert: {model_result.get('valid', False)}")
            print(f"✅ Tests validiert: {len([r for r in test_results if r.get('valid', False)])}/{len(test_results)}")
            print(f"❌ Gesamt-Fehler: {total_errors}")
            print(f"⚠️ Gesamt-Warnungen: {total_warnings}")
            
            return {
                "model": model_result,
                "tests": test_results,
                "total_errors": total_errors,
                "total_warnings": total_warnings,
                "all_valid": total_errors == 0
            }
        else:
            return {
                "model": model_result,
                "tests": [],
                "error": "Konnte Tests nicht laden"
            }

async def main():
    """Hauptfunktion"""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python validate_model_test_results.py <model_id> [test_id]")
        print("\nBeispiele:")
        print("  python validate_model_test_results.py 1          # Validiert Modell 1 und alle Tests")
        print("  python validate_model_test_results.py 1 5      # Validiert Modell 1 und Test 5")
        sys.exit(1)
    
    model_id = int(sys.argv[1])
    
    if len(sys.argv) >= 3:
        # Validiere spezifischen Test
        test_id = int(sys.argv[2])
        await validate_test_result(test_id)
    else:
        # Validiere Modell und alle Tests
        await validate_model_and_tests(model_id)

if __name__ == "__main__":
    asyncio.run(main())

