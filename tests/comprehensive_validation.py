#!/usr/bin/env python3
"""
Umfassende Validierung aller Daten - Prüft ob alle Werte korrekt sind
und keine erfundenen Daten angezeigt werden
"""
import asyncio
import httpx
import json
import os
from typing import Dict, Any, List
from datetime import datetime

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000/api")

async def comprehensive_validation():
    """Umfassende Validierung aller Daten"""
    print("="*80)
    print("🔍 UMFASSENDE VALIDIERUNG ALLER DATEN")
    print("="*80)
    print()
    
    async with httpx.AsyncClient(timeout=60.0) as client:
        errors = []
        warnings = []
        
        # 1. Lade alle Modelle
        print("1️⃣ Lade alle Modelle...")
        try:
            response = await client.get(f"{API_BASE_URL}/models")
            response.raise_for_status()
            models = response.json()
            print(f"   ✅ {len(models)} Modelle gefunden")
        except Exception as e:
            errors.append(f"❌ Fehler beim Laden der Modelle: {e}")
            return
        
        # 2. Lade alle Tests
        print("\n2️⃣ Lade alle Test-Ergebnisse...")
        try:
            response = await client.get(f"{API_BASE_URL}/test-results")
            response.raise_for_status()
            tests = response.json()
            print(f"   ✅ {len(tests)} Test-Ergebnisse gefunden")
        except Exception as e:
            errors.append(f"❌ Fehler beim Laden der Tests: {e}")
            return
        
        # 3. Lade alle Vergleiche
        print("\n3️⃣ Lade alle Vergleiche...")
        try:
            response = await client.get(f"{API_BASE_URL}/comparisons")
            response.raise_for_status()
            comparisons = response.json()
            print(f"   ✅ {len(comparisons)} Vergleiche gefunden")
        except Exception as e:
            errors.append(f"❌ Fehler beim Laden der Vergleiche: {e}")
            return
        
        # 4. Validiere jedes Modell
        print("\n" + "="*80)
        print("4️⃣ VALIDIERE JEDES MODELL")
        print("="*80)
        for model in models:
            model_id = model.get('id')
            print(f"\n📊 Modell {model_id}: {model.get('name', 'N/A')}")
            
            # Prüfe Basis-Felder
            required_fields = ['id', 'name', 'model_type', 'status', 'train_start', 'train_end']
            for field in required_fields:
                if field not in model or model[field] is None:
                    errors.append(f"❌ Modell {model_id}: Feld '{field}' fehlt")
            
            # Prüfe Status
            if model.get('status') != 'READY':
                warnings.append(f"⚠️ Modell {model_id}: Status ist nicht READY: {model.get('status')}")
            
            # Prüfe Confusion Matrix
            cm = model.get('confusion_matrix')
            tp = model.get('tp', 0)
            tn = model.get('tn', 0)
            fp = model.get('fp', 0)
            fn = model.get('fn', 0)
            
            if cm:
                if cm.get('tp') != tp or cm.get('tn') != tn or cm.get('fp') != fp or cm.get('fn') != fn:
                    errors.append(f"❌ Modell {model_id}: Confusion Matrix Werte stimmen nicht überein!")
            
            # Prüfe Accuracy-Berechnung
            total = tp + tn + fp + fn
            if total > 0:
                calculated_accuracy = (tp + tn) / total
                stored_accuracy = model.get('training_accuracy')
                if stored_accuracy and abs(calculated_accuracy - stored_accuracy) > 0.001:
                    errors.append(f"❌ Modell {model_id}: Accuracy stimmt nicht! Berechnet: {calculated_accuracy:.4f}, Gespeichert: {stored_accuracy:.4f}")
            
            # Prüfe Feature Importance Summe
            feature_importance = model.get('feature_importance')
            if feature_importance:
                importance_sum = sum(feature_importance.values())
                if abs(importance_sum - 1.0) > 0.01:
                    errors.append(f"❌ Modell {model_id}: Feature Importance summiert nicht zu 1.0: {importance_sum:.4f}")
            
            # Prüfe Zeitstempel
            train_start = model.get('train_start')
            train_end = model.get('train_end')
            if train_start and train_end:
                try:
                    start_dt = datetime.fromisoformat(train_start.replace('Z', '+00:00'))
                    end_dt = datetime.fromisoformat(train_end.replace('Z', '+00:00'))
                    if end_dt <= start_dt:
                        errors.append(f"❌ Modell {model_id}: train_end muss nach train_start sein!")
                except Exception as e:
                    errors.append(f"❌ Modell {model_id}: Ungültige Zeitstempel: {e}")
        
        # 5. Validiere jeden Test
        print("\n" + "="*80)
        print("5️⃣ VALIDIERE JEDEN TEST")
        print("="*80)
        for test in tests:
            test_id = test.get('id')
            model_id = test.get('model_id')
            print(f"\n📊 Test {test_id}: Modell {model_id}")
            
            # Prüfe ob Modell existiert
            model_exists = any(m.get('id') == model_id for m in models)
            if not model_exists:
                errors.append(f"❌ Test {test_id}: Modell {model_id} existiert nicht!")
            
            # Prüfe Confusion Matrix
            cm = test.get('confusion_matrix')
            tp = test.get('tp', 0)
            tn = test.get('tn', 0)
            fp = test.get('fp', 0)
            fn = test.get('fn', 0)
            num_samples = test.get('num_samples', 0)
            
            if cm:
                if cm.get('tp') != tp or cm.get('tn') != tn or cm.get('fp') != fp or cm.get('fn') != fn:
                    errors.append(f"❌ Test {test_id}: Confusion Matrix Werte stimmen nicht überein!")
            
            # Prüfe Summe
            total = tp + tn + fp + fn
            if total != num_samples:
                errors.append(f"❌ Test {test_id}: Confusion Matrix summiert nicht zu num_samples! {total} != {num_samples}")
            
            # Prüfe Accuracy-Berechnung
            if total > 0:
                calculated_accuracy = (tp + tn) / total
                stored_accuracy = test.get('accuracy')
                if stored_accuracy and abs(calculated_accuracy - stored_accuracy) > 0.001:
                    errors.append(f"❌ Test {test_id}: Accuracy stimmt nicht! Berechnet: {calculated_accuracy:.4f}, Gespeichert: {stored_accuracy:.4f}")
            
            # Prüfe Label-Balance
            num_positive = test.get('num_positive', 0)
            num_negative = test.get('num_negative', 0)
            if num_positive + num_negative != num_samples:
                errors.append(f"❌ Test {test_id}: Label-Balance stimmt nicht! {num_positive} + {num_negative} != {num_samples}")
            
            # Prüfe Zeitstempel
            test_start = test.get('test_start')
            test_end = test.get('test_end')
            if test_start and test_end:
                try:
                    start_dt = datetime.fromisoformat(test_start.replace('Z', '+00:00'))
                    end_dt = datetime.fromisoformat(test_end.replace('Z', '+00:00'))
                    if end_dt <= start_dt:
                        errors.append(f"❌ Test {test_id}: test_end muss nach test_start sein!")
                except Exception as e:
                    errors.append(f"❌ Test {test_id}: Ungültige Zeitstempel: {e}")
            
            # Prüfe Train vs. Test Vergleich
            train_accuracy = test.get('train_accuracy')
            test_accuracy = test.get('accuracy')
            accuracy_degradation = test.get('accuracy_degradation')
            
            if train_accuracy and test_accuracy and accuracy_degradation is not None:
                calculated_degradation = train_accuracy - test_accuracy
                if abs(calculated_degradation - accuracy_degradation) > 0.001:
                    errors.append(f"❌ Test {test_id}: Accuracy Degradation stimmt nicht! Berechnet: {calculated_degradation:.4f}, Gespeichert: {accuracy_degradation:.4f}")
        
        # 6. Validiere jeden Vergleich
        print("\n" + "="*80)
        print("6️⃣ VALIDIERE JEDEN VERGLEICH")
        print("="*80)
        for comparison in comparisons:
            comp_id = comparison.get('id')
            model_a_id = comparison.get('model_a_id')
            model_b_id = comparison.get('model_b_id')
            test_a_id = comparison.get('test_a_id')
            test_b_id = comparison.get('test_b_id')
            
            print(f"\n📊 Vergleich {comp_id}: Modell A={model_a_id}, Modell B={model_b_id}")
            
            # Prüfe ob Modelle existieren
            model_a_exists = any(m.get('id') == model_a_id for m in models)
            model_b_exists = any(m.get('id') == model_b_id for m in models)
            if not model_a_exists:
                errors.append(f"❌ Vergleich {comp_id}: Modell A {model_a_id} existiert nicht!")
            if not model_b_exists:
                errors.append(f"❌ Vergleich {comp_id}: Modell B {model_b_id} existiert nicht!")
            
            # Prüfe ob Tests existieren
            if test_a_id:
                test_a_exists = any(t.get('id') == test_a_id for t in tests)
                if not test_a_exists:
                    errors.append(f"❌ Vergleich {comp_id}: Test A {test_a_id} existiert nicht!")
                else:
                    # Prüfe ob Test A zum Modell A gehört
                    test_a = next(t for t in tests if t.get('id') == test_a_id)
                    if test_a.get('model_id') != model_a_id:
                        errors.append(f"❌ Vergleich {comp_id}: Test A {test_a_id} gehört nicht zu Modell A {model_a_id}!")
            
            if test_b_id:
                test_b_exists = any(t.get('id') == test_b_id for t in tests)
                if not test_b_exists:
                    errors.append(f"❌ Vergleich {comp_id}: Test B {test_b_id} existiert nicht!")
                else:
                    # Prüfe ob Test B zum Modell B gehört
                    test_b = next(t for t in tests if t.get('id') == test_b_id)
                    if test_b.get('model_id') != model_b_id:
                        errors.append(f"❌ Vergleich {comp_id}: Test B {test_b_id} gehört nicht zu Modell B {model_b_id}!")
            
            # Prüfe Winner
            winner_id = comparison.get('winner_id')
            if winner_id and winner_id not in [model_a_id, model_b_id]:
                errors.append(f"❌ Vergleich {comp_id}: Winner {winner_id} ist weder Modell A noch B!")
        
        # 7. Zusammenfassung
        print("\n" + "="*80)
        print("📊 VALIDIERUNGS-ERGEBNIS")
        print("="*80)
        print(f"\n✅ Modelle geprüft: {len(models)}")
        print(f"✅ Tests geprüft: {len(tests)}")
        print(f"✅ Vergleiche geprüft: {len(comparisons)}")
        
        if errors:
            print(f"\n❌ FEHLER GEFUNDEN: {len(errors)}")
            for error in errors:
                print(f"   {error}")
        else:
            print("\n✅ KEINE FEHLER GEFUNDEN!")
        
        if warnings:
            print(f"\n⚠️ WARNUNGEN: {len(warnings)}")
            for warning in warnings:
                print(f"   {warning}")
        else:
            print("\n✅ KEINE WARNUNGEN!")
        
        print("\n" + "="*80)
        
        if errors:
            return False
        return True

if __name__ == "__main__":
    success = asyncio.run(comprehensive_validation())
    exit(0 if success else 1)

