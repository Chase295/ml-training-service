"""
Modell-Loader und Testing für ML Training Service
Lädt gespeicherte Modelle und testet sie auf neuen Daten
"""
import joblib
import logging
from typing import Dict, Any
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, confusion_matrix, matthews_corrcoef
)
from app.training.feature_engineering import load_training_data, create_labels, create_time_based_labels, check_overlap
from datetime import timedelta, datetime as dt
from app.database.models import get_model

logger = logging.getLogger(__name__)

def load_model(model_path: str) -> Any:
    """
    Lädt gespeichertes Modell aus .pkl Datei.
    
    Verwendet joblib zum Laden von trainierten Scikit-learn oder XGBoost Modellen.
    
    Args:
        model_path: Pfad zur .pkl Datei (z.B. "/app/models/model_random_forest_123.pkl")
    
    Returns:
        Geladenes Modell-Objekt (RandomForestClassifier, XGBClassifier, etc.)
        
    Raises:
        FileNotFoundError: Wenn die Datei nicht existiert
        Exception: Bei Fehlern beim Laden (z.B. korrupte Datei)
        
    Example:
        ```python
        model = load_model("/app/models/model_random_forest_123.pkl")
        predictions = model.predict(X_test)
        ```
    """
    logger.info(f"📂 Lade Modell: {model_path}")
    return joblib.load(model_path)

async def test_model(
    model_id: int,
    test_start: str,
    test_end: str,
    model_storage_path: str = "/app/models"
) -> Dict[str, Any]:
    """
    Testet ein Modell auf neuen Daten
    
    Args:
        model_id: ID des Modells aus ml_models
        test_start: Start-Zeitpunkt für Test-Daten (ISO-Format oder datetime)
        test_end: Ende-Zeitpunkt für Test-Daten (ISO-Format oder datetime)
        model_storage_path: Pfad zum Models-Verzeichnis (wird nicht verwendet, aber für Konsistenz)
    
    Returns:
        Dict mit allen Metriken, Confusion Matrix, Overlap-Info
    """
    logger.info(f"🧪 Starte Test für Modell {model_id}")
    
    # 1. Lade Modell-Info aus DB
    model = await get_model(model_id)
    if not model or model.get('is_deleted'):
        raise ValueError(f"Modell {model_id} nicht gefunden oder gelöscht!")
    
    logger.info(f"📋 Modell: {model['name']} ({model['model_type']})")
    
    # 2. Lade Modell-Datei
    model_obj = load_model(model['model_file_path'])
    
    # 3. Features und Phasen aus JSONB (asyncpg konvertiert automatisch)
    features = model['features']  # JSONB Array → Python List (kann engineered features enthalten!)
    phases = model['phases'] if model['phases'] else None  # JSONB Array → Python List
    
    logger.info(f"📊 Features: {features}, Phasen: {phases}")
    
    # 3.5. Prüfe ob Feature-Engineering verwendet wurde
    params = model.get('params', {}) or {}
    if isinstance(params, str):
        from app.database.utils import from_jsonb
        params = from_jsonb(params) or {}
    
    use_engineered_features = params.get('use_engineered_features', False)
    feature_engineering_windows = params.get('feature_engineering_windows', [5, 10, 15])
    
    if use_engineered_features:
        logger.info(f"🔧 Modell wurde mit Feature-Engineering trainiert (Windows: {feature_engineering_windows})")
    
    # 3.6. Prüfe ob zeitbasierte Vorhersage (target_operator ist NULL)
    is_time_based = model.get('target_operator') is None or model.get('target_value') is None
    
    # 4. Bestimme Basis-Features (ohne engineered features)
    # Engineered features haben Namen wie "price_change_5", "volume_ratio_10", etc.
    from app.training.feature_engineering import get_engineered_feature_names
    if use_engineered_features:
        engineered_feature_names = get_engineered_feature_names(feature_engineering_windows)
        # Basis-Features sind alle Features die NICHT engineered sind
        # ⚠️ WICHTIG: Konvertiere zu Set für schnelleren Lookup
        engineered_set = set(engineered_feature_names)
        base_features = [f for f in features if f not in engineered_set]
        
        # Debug: Prüfe ob Filterung funktioniert
        still_engineered = [f for f in base_features if f in engineered_set]
        if still_engineered:
            logger.warning(f"⚠️ WARNUNG: {len(still_engineered)} engineered Features noch in base_features: {still_engineered[:5]}")
            # Entferne sie manuell
            base_features = [f for f in base_features if f not in engineered_set]
        
        logger.info(f"📊 Basis-Features: {len(base_features)}, Engineered Features: {len(engineered_feature_names)}")
        logger.debug(f"   Basis (erste 10): {base_features[:10]}")
        logger.debug(f"   Engineered (erste 10): {list(engineered_set)[:10]}")
    else:
        base_features = list(features)
        logger.info(f"📊 Basis-Features: {len(base_features)} (kein Feature-Engineering)")
    
    # 5. Lade Test-Daten (target_var muss in Features sein für Labels)
    features_with_target = list(base_features)  # Nur Basis-Features für Daten-Laden
    if model['target_variable'] not in features_with_target:
        features_with_target.append(model['target_variable'])
    
    # 🆕 Prüfe ob ATH-Daten geladen werden sollen (aus params)
    include_ath = params.get('include_ath', True)  # Default: True
    
    test_data = await load_training_data(
        train_start=test_start,
        train_end=test_end,
        features=features_with_target,
        phases=phases,
        include_ath=include_ath  # 🆕 ATH-Daten optional laden
    )
    
    if len(test_data) == 0:
        raise ValueError("Keine Test-Daten gefunden!")
    
    logger.info(f"✅ {len(test_data)} Test-Daten geladen")
    
    # 6. Feature-Engineering anwenden (wenn Modell damit trainiert wurde)
    if use_engineered_features:
        logger.info("🔧 Erstelle engineered features für Test-Daten...")
        from app.training.feature_engineering import create_pump_detection_features
        
        # Erstelle engineered features
        test_data = create_pump_detection_features(test_data, window_sizes=feature_engineering_windows)
        
        # Prüfe ob alle benötigten Features vorhanden sind
        missing_features = [f for f in features if f not in test_data.columns]
        if missing_features:
            logger.warning(f"⚠️ Einige Features fehlen in Test-Daten: {missing_features}")
            # Verwende nur verfügbare Features
            features = [f for f in features if f in test_data.columns]
            logger.info(f"📊 Verwende {len(features)} verfügbare Features (ausgeschlossen: {len(missing_features)})")
        else:
            logger.info(f"✅ Alle {len(features)} Features (inkl. engineered) verfügbar")
    else:
        # Auch ohne Feature-Engineering: Prüfe ob alle Features vorhanden sind
        missing_features = [f for f in features if f not in test_data.columns]
        if missing_features:
            logger.warning(f"⚠️ Einige Features fehlen in Test-Daten: {missing_features}")
            features = [f for f in features if f in test_data.columns]
            logger.info(f"📊 Verwende {len(features)} verfügbare Features (ausgeschlossen: {len(missing_features)})")
    
    # 7. Erstelle Labels (gleiche Logik wie beim Training)
    if is_time_based:
        # Zeitbasierte Vorhersage: Hole Parameter aus params
        params = model.get('params', {}) or {}
        if isinstance(params, str):
            import json
            try:
                from app.database.utils import from_jsonb
                params = from_jsonb(params) or {}
            except:
                params = {}
        
        time_based_config = params.get('_time_based', {})
        
        future_minutes = time_based_config.get('future_minutes')
        min_percent_change = time_based_config.get('min_percent_change')
        direction = time_based_config.get('direction', 'up')
        
        # Fallback: Falls Parameter fehlen, versuche aus Job-Daten zu holen (für ältere Modelle)
        if not future_minutes or min_percent_change is None:
            logger.warning(f"⚠️ Zeitbasierte Parameter fehlen in Modell {model_id}, versuche Fallback...")
            # Versuche aus ml_jobs zu holen (falls Modell kürzlich erstellt wurde)
            from app.database.models import get_job
            # Suche nach TRAIN-Job für dieses Modell
            # Da wir keine direkte Verknüpfung haben, verwenden wir Default-Werte
            if future_minutes is None:
                future_minutes = 10  # Default
                logger.warning(f"⚠️ Verwende Default future_minutes={future_minutes}")
            if min_percent_change is None:
                min_percent_change = 5.0  # Default
                logger.warning(f"⚠️ Verwende Default min_percent_change={min_percent_change}")
            if not direction:
                direction = 'up'
        
        if not future_minutes or min_percent_change is None:
            raise ValueError(f"Zeitbasierte Vorhersage-Parameter fehlen in Modell {model_id}! future_minutes={future_minutes}, min_percent_change={min_percent_change}")
        
        # Lade Phase-Intervalle
        from app.database.models import get_phase_intervals
        phase_intervals = await get_phase_intervals()
        
        from app.training.feature_engineering import create_time_based_labels
        labels = create_time_based_labels(
            test_data,
            model['target_variable'],
            future_minutes,
            min_percent_change,
            direction,
            phase_intervals
        )
        logger.info(f"⏰ Zeitbasierte Labels erstellt: {labels.sum()} positive, {len(labels) - labels.sum()} negative")
    else:
        # Normale Vorhersage
        if model.get('target_operator') is None or model.get('target_value') is None:
            raise ValueError(f"Modell {model_id} hat keine target_operator/target_value für normale Vorhersage!")
        
        labels = create_labels(
            test_data, 
            model['target_variable'],
            model['target_operator'],
            float(model['target_value'])
        )
        logger.info(f"✅ Normale Labels erstellt: {labels.sum()} positive, {len(labels) - labels.sum()} negative")
    
    # 8. Mache Vorhersagen (verwende alle Features, nicht target_var)
    # target_var wurde nur für Labels benötigt, nicht für Features
    # ⚠️ WICHTIG: features enthält jetzt auch engineered features (wenn Feature-Engineering aktiviert)
    # ⚠️ KRITISCH: Prüfe nochmal ob alle Features vorhanden sind (sicherheitshalber)
    available_features = [f for f in features if f in test_data.columns]
    if len(available_features) != len(features):
        missing = [f for f in features if f not in test_data.columns]
        logger.error(f"❌ KRITISCH: {len(missing)} Features fehlen: {missing}")
        raise ValueError(f"Features fehlen in Test-Daten: {missing}")
    
    X_test = test_data[available_features].values
    y_test = labels.values
    y_pred = model_obj.predict(X_test)
    y_pred_proba = model_obj.predict_proba(X_test)[:, 1] if hasattr(model_obj, 'predict_proba') else None
    
    logger.info(f"🔮 Vorhersagen gemacht: {len(y_pred)} Samples mit {len(features)} Features")
    
    # 9. Berechne Basis-Metriken
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else None
    
    logger.info(f"📈 Basis-Metriken: Accuracy={accuracy:.4f}, F1={f1:.4f}, Precision={precision:.4f}, Recall={recall:.4f}")
    
    # 10. Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    if cm.size == 4:  # 2x2 Matrix
        tn, fp, fn, tp = cm.ravel()
    else:
        # Fallback falls Matrix anders strukturiert
        tn, fp, fn, tp = 0, 0, 0, 0
    
    # 11. Zusätzliche Metriken (Phase 9)
    # MCC (Matthews Correlation Coefficient)
    mcc = matthews_corrcoef(y_test, y_pred)
    
    # FPR (False Positive Rate)
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    
    # FNR (False Negative Rate)
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    
    # Profit-Simulation (vereinfacht)
    # Annahme: 1% Gewinn pro richtig erkanntem Pump, 0.5% Verlust pro False Positive
    profit_per_tp = 0.01  # 1%
    loss_per_fp = -0.005  # -0.5%
    simulated_profit = (tp * profit_per_tp) + (fp * loss_per_fp)
    simulated_profit_pct = simulated_profit / len(y_test) * 100 if len(y_test) > 0 else 0.0
    
    # Confusion Matrix als Dict
    confusion_matrix_dict = {
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn)
    }
    
    logger.info(f"💰 Simulierter Profit: {simulated_profit_pct:.2f}% (bei {tp} TP, {fp} FP)")
    roc_auc_str = f"{roc_auc:.4f}" if roc_auc is not None else "N/A"
    logger.info(f"📊 Zusätzliche Metriken: ROC-AUC={roc_auc_str}, MCC={mcc:.4f}, FPR={fpr:.4f}, FNR={fnr:.4f}")
    
    # 12. Test-Zeitraum Validierung (Phase 2)
    if isinstance(test_start, str):
        test_start_dt = dt.fromisoformat(test_start.replace('Z', '+00:00'))
    else:
        test_start_dt = test_start
    if isinstance(test_end, str):
        test_end_dt = dt.fromisoformat(test_end.replace('Z', '+00:00'))
    else:
        test_end_dt = test_end
    
    test_duration = test_end_dt - test_start_dt
    min_test_duration = timedelta(days=1)  # Mindest 1 Tag
    test_duration_days = test_duration.total_seconds() / 86400.0  # In Tagen
    
    if test_duration < min_test_duration:
        logger.warning(f"⚠️ Test-Zeitraum zu kurz: {test_duration_days:.2f} Tage (empfohlen: mindestens 1 Tag)")
    
    # 13. Overlap-Check
    overlap_info = check_overlap(
        train_start=model['train_start'],
        train_end=model['train_end'],
        test_start=test_start,
        test_end=test_end
    )
    
    if overlap_info['has_overlap']:
        logger.warning(f"⚠️ {overlap_info['overlap_note']}")
    
    # 14. Train vs. Test Vergleich (Phase 2)
    train_accuracy = model.get('training_accuracy')
    train_f1 = model.get('training_f1')
    train_precision = model.get('training_precision')
    train_recall = model.get('training_recall')
    
    accuracy_degradation = None
    f1_degradation = None
    is_overfitted = None
    
    if train_accuracy is not None:
        accuracy_degradation = float(train_accuracy) - float(accuracy)
        # Overfitting-Indikator: > 10% Unterschied
        is_overfitted = accuracy_degradation > 0.1
        
        if is_overfitted:
            logger.warning(f"⚠️ OVERFITTING erkannt! Train-Test Accuracy Gap: {accuracy_degradation:.2%}")
            logger.warning(f"   → Modell generalisiert schlecht auf neue Daten")
        else:
            logger.info(f"✅ Train-Test Accuracy Gap: {accuracy_degradation:.2%} (akzeptabel)")
    
    if train_f1 is not None:
        f1_degradation = float(train_f1) - float(f1)
        logger.info(f"📊 Train-Test F1 Gap: {f1_degradation:.2%}")
    
    # 15. Return Ergebnisse
    result = {
        "accuracy": float(accuracy),
        "f1_score": float(f1),
        "precision_score": float(precision),
        "recall": float(recall),
        "roc_auc": float(roc_auc) if roc_auc is not None else None,
        # Zusätzliche Metriken (Phase 9)
        "mcc": float(mcc),
        "fpr": float(fpr),
        "fnr": float(fnr),
        "simulated_profit_pct": float(simulated_profit_pct),
        "confusion_matrix": confusion_matrix_dict,
        # Legacy: Confusion Matrix als einzelne Felder (für Rückwärtskompatibilität)
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "num_samples": len(test_data),
        "num_positive": int(labels.sum()),
        "num_negative": int(len(labels) - labels.sum()),
        "has_overlap": overlap_info['has_overlap'],
        "overlap_note": overlap_info['overlap_note'],
        # Train vs. Test Vergleich (Phase 2)
        "train_accuracy": float(train_accuracy) if train_accuracy is not None else None,
        "train_f1": float(train_f1) if train_f1 is not None else None,
        "train_precision": float(train_precision) if train_precision is not None else None,
        "train_recall": float(train_recall) if train_recall is not None else None,
        "accuracy_degradation": float(accuracy_degradation) if accuracy_degradation is not None else None,
        "f1_degradation": float(f1_degradation) if f1_degradation is not None else None,
        "is_overfitted": bool(is_overfitted) if is_overfitted is not None else None,
        # Test-Zeitraum Info (Phase 2)
        "test_duration_days": float(test_duration_days)
    }
    
    logger.info(f"✅ Test abgeschlossen für Modell {model_id}")
    
    return result

