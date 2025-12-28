#!/usr/bin/env python3
"""
UMFASSENDE TESTS: Feature-Engineering Bug-Analyse & -Fix

Testet alle Aspekte des Feature-Engineering:
1. Daten laden
2. ATH-Features berechnen
3. Engineered Features erstellen
4. Vollständige Feature-Verfügbarkeit prüfen
5. Training simulieren

Ziel: Sicherstellen, dass alle erwarteten Features erstellt werden
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
import asyncio
import logging
from datetime import datetime, timezone

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('test_feature_engineering.log')
    ]
)
logger = logging.getLogger(__name__)

async def test_feature_engineering_comprehensive():
    """Umfassender Test des Feature-Engineering"""

    logger.info("🧪 STARTE UMFASSENDE FEATURE-ENGINEERING TESTS")
    logger.info("=" * 60)

    try:
        # ==========================================
        # 1. DATEN LADEN
        # ==========================================
        logger.info("📥 1. Lade Trainingsdaten...")

        from app.training.feature_engineering import load_training_data

        data = await load_training_data(
            train_start="2025-12-27T16:30:00Z",
            train_end="2025-12-28T01:00:00Z",
            features=["price_close", "volume_sol", "buy_pressure_ratio"],
            include_ath=True
        )

        logger.info(f"✅ Daten geladen: {len(data)} Zeilen, {len(data.columns)} Spalten")
        logger.info(f"📊 Spalten: {sorted(list(data.columns))[:10]}...")  # Erste 10 Spalten

        # ==========================================
        # 2. DATEN-QUALITÄT PRÜFEN
        # ==========================================
        logger.info("🔍 2. Prüfe Daten-Qualität...")

        critical_columns = [
            'price_close', 'price_high', 'price_low', 'price_open',
            'volume_sol', 'timestamp', 'mint'
        ]

        missing_critical = [col for col in critical_columns if col not in data.columns]
        if missing_critical:
            logger.error(f"❌ KRITISCHE SPALTEN FEHLEN: {missing_critical}")
            return False
        else:
            logger.info("✅ Alle kritischen Spalten vorhanden")

        # Prüfe Datenvollständigkeit
        for col in critical_columns:
            null_count = data[col].isna().sum()
            if null_count > 0:
                logger.warning(f"⚠️ {col}: {null_count} NULL-Werte ({null_count/len(data)*100:.1f}%)")

        # ==========================================
        # 3. ATH-FEATURES TESTEN
        # ==========================================
        logger.info("🧠 3. Teste ATH-Features...")

        ath_features_expected = [
            'rolling_ath', 'ath_distance_pct', 'ath_breakout',
            'minutes_since_ath', 'ath_age_hours', 'ath_is_recent', 'ath_is_old'
        ]

        ath_features_present = [f for f in ath_features_expected if f in data.columns]
        ath_features_missing = [f for f in ath_features_expected if f not in data.columns]

        logger.info(f"🧠 ATH-Features vorhanden: {len(ath_features_present)}/{len(ath_features_expected)}")
        if ath_features_missing:
            logger.warning(f"⚠️ Fehlende ATH-Features: {ath_features_missing}")

        # Prüfe ATH-Feature Qualität
        if 'ath_distance_pct' in data.columns:
            ath_mean = data['ath_distance_pct'].mean()
            ath_std = data['ath_distance_pct'].std()
            logger.info(".2f")

        # ==========================================
        # 4. ENGINEERED FEATURES TESTEN
        # ==========================================
        logger.info("🔧 4. Teste Engineered Features...")

        from app.training.feature_engineering import create_pump_detection_features, get_engineered_feature_names

        window_sizes = [5, 10, 15]
        expected_engineered = get_engineered_feature_names(window_sizes)

        logger.info(f"🎯 Erwartete engineered Features: {len(expected_engineered)}")

        # Erstelle engineered Features
        data_with_engineered = create_pump_detection_features(data.copy(), window_sizes)

        engineered_created = [col for col in data_with_engineered.columns if col not in data.columns]
        logger.info(f"✅ Erstellte engineered Features: {len(engineered_created)}")

        # Prüfe Vollständigkeit
        missing_engineered = [f for f in expected_engineered if f not in data_with_engineered.columns]
        extra_engineered = [f for f in engineered_created if f not in expected_engineered]

        if missing_engineered:
            logger.error(f"❌ FEHLENDE ENGINEERED FEATURES: {len(missing_engineered)} - {missing_engineered}")
        else:
            logger.info("✅ Alle erwarteten engineered Features erstellt")

        if extra_engineered:
            logger.info(f"ℹ️ Zusätzliche Features: {extra_engineered}")

        # ==========================================
        # 5. TRAINING-SIMULATION
        # ==========================================
        logger.info("🎯 5. Simuliere Training-Feature-Auswahl...")

        # Simuliere die Features-Liste aus dem Training
        features = ["price_close", "volume_sol", "buy_pressure_ratio"]

        # ATH-Features hinzufügen (wie im echten Training)
        if 'rolling_ath' in data_with_engineered.columns:
            from app.training.feature_engineering import get_available_ath_features
            ath_features = get_available_ath_features(include_ath=True)
            features.extend(ath_features)
            logger.info(f"🧠 ATH-Features zur Liste hinzugefügt: {len(ath_features)}")

        # Engineered Features hinzufügen (wie im echten Training)
        features.extend(expected_engineered)  # Alle erwarteten!
        logger.info(f"🔧 Engineered Features zur Liste hinzugefügt: {len(expected_engineered)}")

        # Finale Filterung (wie im echten Training)
        available_features = [f for f in features if f in data_with_engineered.columns]
        missing_features = [f for f in features if f not in data_with_engineered.columns]

        logger.info(f"📊 Finale Feature-Zusammenfassung:")
        logger.info(f"   Ursprüngliche Features-Liste: {len(features)}")
        logger.info(f"   Verfügbare Features: {len(available_features)}")
        logger.info(f"   Fehlende Features: {len(missing_features)}")

        if missing_features:
            logger.error(f"❌ FEHLENDE FEATURES IM TRAINING: {missing_features}")

            # Kategorisiere fehlende Features
            missing_ath = [f for f in missing_features if f in ath_features_expected]
            missing_engineered_final = [f for f in missing_features if f in expected_engineered]

            logger.error(f"   Fehlende ATH-Features: {len(missing_ath)} - {missing_ath}")
            logger.error(f"   Fehlende Engineered-Features: {len(missing_engineered_final)} - {missing_engineered_final}")
        else:
            logger.info("✅ ALLE FEATURES VERFÜGBAR - TRAINING SOLLTE FUNKTIONIEREN!")

        # ==========================================
        # 6. QUALITÄTS-METRIKEN
        # ==========================================
        logger.info("📊 6. Qualitäts-Metriken...")

        # Prüfe Feature-Verteilung
        numeric_features = data_with_engineered.select_dtypes(include=[np.number]).columns
        logger.info(f"🔢 Numerische Features: {len(numeric_features)}")

        # Prüfe auf konstante Features (alle Werte gleich)
        constant_features = []
        for col in numeric_features[:10]:  # Prüfe erste 10
            if data_with_engineered[col].nunique() <= 1:
                constant_features.append(col)

        if constant_features:
            logger.warning(f"⚠️ Konstante Features (mögliche Probleme): {constant_features}")

        # ==========================================
        # 7. ERGEBNIS
        # ==========================================
        logger.info("=" * 60)
        logger.info("📋 TEST-ERGEBNIS")

        success = len(missing_features) == 0
        total_features = len(available_features)

        logger.info(f"✅ Features verfügbar: {total_features}")
        logger.info(f"❌ Features fehlen: {len(missing_features)}")
        logger.info(f"🎯 Erfolg: {'JA' if success else 'NEIN'}")

        if success:
            logger.info("🎉 FEATURE-ENGINEERING BUG BEHOBEN!")
        else:
            logger.error("💥 FEATURE-ENGINEERING BUG BESTEHT NOCH!")

        # Speichere Test-Ergebnisse
        results = {
            'success': success,
            'total_features': total_features,
            'missing_features': len(missing_features),
            'data_rows': len(data_with_engineered),
            'data_columns': len(data_with_engineered.columns),
            'ath_features_present': len(ath_features_present),
            'engineered_features_created': len(engineered_created),
            'missing_features_list': missing_features
        }

        import json
        with open('test_results.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)

        logger.info("💾 Testergebnisse gespeichert in test_results.json")

        return success, total_features, missing_features

    except Exception as e:
        logger.error(f"❌ Fehler bei umfassenden Tests: {e}")
        import traceback
        traceback.print_exc()
        return False, 0, [str(e)]

if __name__ == "__main__":
    result = asyncio.run(test_feature_engineering_comprehensive())
    print(f"\n🎯 Finale Zusammenfassung: {'ERFOLG' if result[0] else 'FEHLSCHLAG'} - {result[1]} Features verfügbar, {len(result[2])} fehlen")

    if not result[0]:
        print(f"❌ Fehlende Features: {result[2]}")
        sys.exit(1)
