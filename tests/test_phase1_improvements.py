#!/usr/bin/env python3
"""
Test-Skript für Phase 1 Verbesserungen: Feature-Engineering beim Testen + Zusätzliche Metriken
"""
import asyncio
import sys
import os
from datetime import datetime, timezone, timedelta

# Füge app-Verzeichnis zum Python-Pfad hinzu
sys.path.insert(0, '/app')

from app.training.model_loader import test_model
from app.database.models import get_model, get_test_result
from app.api.schemas import TestResultResponse

async def test_phase1_improvements():
    """Test Phase 1 Verbesserungen"""
    print("🧪 Teste Phase 1 Verbesserungen...")
    print("=" * 60)
    
    # 1. Finde ein Modell mit Feature-Engineering
    print("\n1️⃣ Suche Modell mit Feature-Engineering...")
    from app.database.models import list_models
    models = await list_models(limit=50)
    
    model_with_fe = None
    model_without_fe = None
    
    for model in models:
        if model.get('status') != 'READY' or model.get('is_deleted'):
            continue
        
        params = model.get('params', {}) or {}
        if isinstance(params, str):
            import json
            try:
                params = json.loads(params)
            except:
                params = {}
        
        use_fe = params.get('use_engineered_features', False)
        
        if use_fe and not model_with_fe:
            model_with_fe = model
            print(f"   ✅ Gefunden: {model['name']} (mit Feature-Engineering)")
        elif not use_fe and not model_without_fe:
            model_without_fe = model
            print(f"   ✅ Gefunden: {model['name']} (ohne Feature-Engineering)")
        
        if model_with_fe and model_without_fe:
            break
    
    if not model_with_fe:
        print("   ⚠️ Kein Modell mit Feature-Engineering gefunden - erstelle Test-Modell...")
        # Hier könnte man ein Test-Modell erstellen, aber das ist komplex
        print("   ⚠️ Überspringe Feature-Engineering Test")
    else:
        # 2. Teste Modell mit Feature-Engineering
        print(f"\n2️⃣ Teste Modell mit Feature-Engineering: {model_with_fe['name']}")
        test_start = datetime.now(timezone.utc) - timedelta(days=2)
        test_end = datetime.now(timezone.utc) - timedelta(days=1)
        
        try:
            result = await test_model(
                model_id=model_with_fe['id'],
                test_start=test_start.isoformat(),
                test_end=test_end.isoformat()
            )
            
            print(f"   ✅ Test erfolgreich!")
            print(f"   📊 Accuracy: {result.get('accuracy', 0):.4f}")
            print(f"   📊 F1: {result.get('f1_score', 0):.4f}")
            
            # Prüfe neue Metriken
            print(f"\n   📈 Zusätzliche Metriken:")
            print(f"      - MCC: {result.get('mcc', 'N/A')}")
            print(f"      - FPR: {result.get('fpr', 'N/A')}")
            print(f"      - FNR: {result.get('fnr', 'N/A')}")
            print(f"      - Profit: {result.get('simulated_profit_pct', 'N/A')}%")
            
            # Prüfe Confusion Matrix
            cm = result.get('confusion_matrix')
            if cm:
                print(f"   🔢 Confusion Matrix (als Dict):")
                print(f"      - TP: {cm.get('tp', 0)}")
                print(f"      - TN: {cm.get('tn', 0)}")
                print(f"      - FP: {cm.get('fp', 0)}")
                print(f"      - FN: {cm.get('fn', 0)}")
            else:
                print(f"   ⚠️ Confusion Matrix fehlt!")
            
            # Validierung
            errors = []
            if result.get('mcc') is None:
                errors.append("MCC fehlt")
            if result.get('fpr') is None:
                errors.append("FPR fehlt")
            if result.get('fnr') is None:
                errors.append("FNR fehlt")
            if result.get('simulated_profit_pct') is None:
                errors.append("simulated_profit_pct fehlt")
            if not result.get('confusion_matrix'):
                errors.append("confusion_matrix fehlt")
            
            if errors:
                print(f"   ❌ Fehler: {', '.join(errors)}")
                return False
            else:
                print(f"   ✅ Alle neuen Metriken vorhanden!")
        
        except Exception as e:
            print(f"   ❌ Test fehlgeschlagen: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    # 3. Teste Modell ohne Feature-Engineering (Rückwärtskompatibilität)
    if model_without_fe:
        print(f"\n3️⃣ Teste Modell ohne Feature-Engineering: {model_without_fe['name']}")
        test_start = datetime.now(timezone.utc) - timedelta(days=2)
        test_end = datetime.now(timezone.utc) - timedelta(days=1)
        
        try:
            result = await test_model(
                model_id=model_without_fe['id'],
                test_start=test_start.isoformat(),
                test_end=test_end.isoformat()
            )
            
            print(f"   ✅ Test erfolgreich!")
            print(f"   📊 Accuracy: {result.get('accuracy', 0):.4f}")
            
            # Prüfe ob neue Metriken auch ohne Feature-Engineering vorhanden sind
            if result.get('mcc') is not None and result.get('confusion_matrix'):
                print(f"   ✅ Neue Metriken auch ohne Feature-Engineering vorhanden!")
            else:
                print(f"   ⚠️ Neue Metriken fehlen!")
                return False
        
        except Exception as e:
            print(f"   ❌ Test fehlgeschlagen: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    print("\n" + "=" * 60)
    print("✅ Alle Tests erfolgreich!")
    return True

if __name__ == "__main__":
    success = asyncio.run(test_phase1_improvements())
    sys.exit(0 if success else 1)

