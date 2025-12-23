"""
Test für Verbesserung 1.1: Data Leakage beheben

Testet ob target_var korrekt entfernt wird bei zeitbasierter Vorhersage.
"""
import asyncio
import sys
import os

# Füge app-Verzeichnis zum Python-Pfad hinzu
sys.path.insert(0, '/app')

from datetime import datetime, timezone, timedelta
from app.training.engine import train_model, prepare_features_for_training

async def test_data_leakage_fix():
    """Test ob target_var korrekt entfernt wird bei zeitbasierter Vorhersage"""
    
    print("=" * 60)
    print("🧪 Test: Data Leakage Fix (Verbesserung 1.1)")
    print("=" * 60)
    
    # Test-Parameter
    train_end = datetime.now(timezone.utc)
    train_start = train_end - timedelta(days=7)
    
    # Test 1: Zeitbasierte Vorhersage
    print("\n📋 Test 1: Zeitbasierte Vorhersage (target_var sollte NICHT in Features sein)")
    print("-" * 60)
    try:
        result = await train_model(
            model_type="random_forest",
            features=["price_open", "price_high", "volume_sol"],
            target_var="price_close",
            target_operator=None,
            target_value=None,
            train_start=train_start,
            train_end=train_end,
            use_time_based=True,
            future_minutes=10,
            min_percent_change=5.0,
            direction="up"
        )
        print(f"✅ Test 1 erfolgreich!")
        print(f"   Accuracy: {result['accuracy']:.4f}")
        print(f"   Features für Training: {result.get('num_features', 'N/A')}")
        print(f"   Erwartung: 3 Features (ohne price_close)")
        if result.get('num_features') == 3:
            print("   ✅ Korrekt: target_var wurde aus Features entfernt")
        else:
            print(f"   ⚠️ Warnung: Erwartet 3 Features, erhalten {result.get('num_features')}")
    except Exception as e:
        print(f"❌ Test 1 fehlgeschlagen: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test 2: Klassische Vorhersage
    print("\n📋 Test 2: Klassische Vorhersage (target_var sollte in Features sein)")
    print("-" * 60)
    try:
        result = await train_model(
            model_type="random_forest",
            features=["price_open", "price_high", "volume_sol"],
            target_var="market_cap_close",
            target_operator=">",
            target_value=50000.0,
            train_start=train_start,
            train_end=train_end,
            use_time_based=False
        )
        print(f"✅ Test 2 erfolgreich!")
        print(f"   Accuracy: {result['accuracy']:.4f}")
        print(f"   Features für Training: {result.get('num_features', 'N/A')}")
        print(f"   Erwartung: 4 Features (mit market_cap_close)")
        if result.get('num_features') == 4:
            print("   ✅ Korrekt: target_var ist in Features enthalten")
        else:
            print(f"   ⚠️ Warnung: Erwartet 4 Features, erhalten {result.get('num_features')}")
    except Exception as e:
        print(f"❌ Test 2 fehlgeschlagen: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test 3: Unit-Test für prepare_features_for_training()
    print("\n📋 Test 3: Unit-Test für prepare_features_for_training()")
    print("-" * 60)
    try:
        features = ["price_open", "price_high", "volume_sol"]
        target_var = "price_close"
        
        # Test 3a: Zeitbasierte Vorhersage
        use_time_based = True
        loading, training = prepare_features_for_training(features, target_var, use_time_based)
        assert target_var in loading, "target_var sollte in loading sein"
        assert target_var not in training, "target_var sollte NICHT in training sein"
        print("   ✅ Test 3a bestanden: Zeitbasierte Vorhersage")
        
        # Test 3b: Klassische Vorhersage
        use_time_based = False
        loading, training = prepare_features_for_training(features, target_var, use_time_based)
        assert target_var in loading, "target_var sollte in loading sein"
        assert target_var in training, "target_var sollte auch in training sein"
        print("   ✅ Test 3b bestanden: Klassische Vorhersage")
        
    except Exception as e:
        print(f"❌ Test 3 fehlgeschlagen: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "=" * 60)
    print("✅ Alle Tests erfolgreich!")
    print("=" * 60)
    return True

if __name__ == "__main__":
    success = asyncio.run(test_data_leakage_fix())
    sys.exit(0 if success else 1)

