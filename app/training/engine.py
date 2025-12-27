"""
Training Engine für ML Training Service
Trainiert Random Forest und XGBoost Modelle
"""
import joblib
import pandas as pd
import numpy as np
import logging
import os
from typing import List, Optional, Dict, Any
from datetime import datetime, timezone
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

# Default-Features (wird verwendet wenn keine Features übergeben werden)
DEFAULT_FEATURES = [
    # Basis OHLC
    "price_open", "price_high", "price_low", "price_close",
    
    # Volumen
    "volume_sol", "buy_volume_sol", "sell_volume_sol", "net_volume_sol",
    
    # Market Cap & Phase
    "market_cap_close", "phase_id_at_time",
    
    # ⚠️ KRITISCH für Rug-Detection
    "dev_sold_amount",  # Wichtigster Indikator!
    
    # Ratio-Metriken (Bot-Spam vs. echtes Interesse)
    "buy_pressure_ratio",
    "unique_signer_ratio",
    
    # Whale-Aktivität
    "whale_buy_volume_sol",
    "whale_sell_volume_sol",
    
    # Volatilität
    "volatility_pct",
    "avg_trade_size_sol",
    
    # 🆕 ATH-Tracking (Breakout-Erkennung)
    "ath_price_sol",
    "price_vs_ath_pct",      # Wie weit vom ATH entfernt?
    "minutes_since_ath"      # Wie lange ist es her?
]

# XGBoost optional (für lokales Testing ohne libomp)
XGBOOST_AVAILABLE = False
XGBClassifier = None
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except Exception as e:
    # Fängt ImportError, XGBoostError, OSError, etc.
    XGBOOST_AVAILABLE = False
    logger.warning(f"⚠️ XGBoost nicht verfügbar: {type(e).__name__}. In Docker wird es funktionieren.")

from app.database.models import get_model_type_defaults
from app.training.feature_engineering import load_training_data, create_labels

def create_model(model_type: str, params: Dict[str, Any]) -> Any:
    """
    Erstellt Modell-Instanz basierend auf Typ.
    
    ⚠️ WICHTIG: Nur Random Forest und XGBoost werden unterstützt!
    
    Args:
        model_type: Modell-Typ ("random_forest" oder "xgboost")
        params: Dictionary mit Hyperparametern. JSONB liefert bereits richtige Python-Typen.
                Unterstützte Parameter:
                - Random Forest: n_estimators, max_depth, min_samples_split, random_state
                - XGBoost: n_estimators, max_depth, learning_rate, random_state
    
    Returns:
        Modell-Instanz (RandomForestClassifier oder XGBClassifier)
        
    Raises:
        ValueError: Wenn model_type nicht unterstützt wird oder XGBoost nicht verfügbar ist
        
    Example:
        ```python
        params = {"n_estimators": 100, "max_depth": 10}
        model = create_model("random_forest", params)
        ```
    """
    # JSONB liefert bereits richtige Python-Typen (int, float, etc.)
    # Keine String-Konvertierung nötig!
    
    # ⚠️ WICHTIG: Entferne interne Parameter die nicht für Modell-Erstellung verwendet werden
    excluded_params = ['n_estimators', 'max_depth', 'min_samples_split', 'random_state', 
                       '_time_based', 'use_engineered_features', 'feature_engineering_windows',
                       'use_smote', 'use_timeseries_split', 'cv_splits',
                       'use_market_context', 'exclude_features']  # Phase 2: Neue Parameter
    
    if model_type == "random_forest":
        return RandomForestClassifier(
            n_estimators=params.get('n_estimators', 100),
            max_depth=params.get('max_depth', 10),
            min_samples_split=params.get('min_samples_split', 2),
            random_state=params.get('random_state', 42),
            **{k: v for k, v in params.items() 
               if k not in excluded_params}
        )
    
    elif model_type == "xgboost":
        if not XGBOOST_AVAILABLE:
            raise ValueError("XGBoost ist nicht verfügbar. In Docker wird es funktionieren.")
        # ⚠️ WICHTIG: Entferne interne Parameter die nicht für Modell-Erstellung verwendet werden
        excluded_params = ['n_estimators', 'max_depth', 'learning_rate', 'random_state',
                           '_time_based', 'use_engineered_features', 'feature_engineering_windows',
                           'use_smote', 'use_timeseries_split', 'cv_splits',
                           'use_market_context', 'exclude_features']  # Phase 2: Neue Parameter
        return XGBClassifier(
            n_estimators=params.get('n_estimators', 100),
            max_depth=params.get('max_depth', 6),
            learning_rate=params.get('learning_rate', 0.1),
            random_state=params.get('random_state', 42),
            eval_metric='logloss',  # Für binäre Klassifikation
            **{k: v for k, v in params.items() 
               if k not in excluded_params}
        )
    
    else:
        raise ValueError(f"Unbekannter Modell-Typ: {model_type}. Nur 'random_forest' und 'xgboost' sind unterstützt!")

def prepare_features_for_training(
    features: List[str],
    target_var: Optional[str],
    use_time_based: bool
) -> tuple[List[str], List[str]]:
    """
    Bereitet Features für Training vor.
    
    ⚠️ KRITISCH: Bei zeitbasierter Vorhersage wird target_var NUR für Labels verwendet,
    NICHT für Training! Dies verhindert Data Leakage.
    
    Args:
        features: Liste der ursprünglichen Features
        target_var: Ziel-Variable (z.B. "price_close")
        use_time_based: True wenn zeitbasierte Vorhersage aktiviert
    
    Returns:
        Tuple von (features_for_loading, features_for_training)
        - features_for_loading: Enthält target_var (für Daten-Laden und Labels)
        - features_for_training: Enthält target_var NICHT bei zeitbasierter Vorhersage
    """
    # Für Daten-Laden: target_var wird benötigt (für Labels)
    features_for_loading = list(features)  # Kopie erstellen
    if target_var and target_var not in features_for_loading:
        features_for_loading.append(target_var)
        logger.info(f"➕ target_var '{target_var}' zu Features für Daten-Laden hinzugefügt")
    
    # Für Training: target_var wird ENTFERNT bei zeitbasierter Vorhersage
    features_for_training = list(features)  # Kopie erstellen
    if use_time_based and target_var and target_var in features_for_training:
        features_for_training.remove(target_var)
        logger.warning(f"⚠️ target_var '{target_var}' aus Features entfernt (zeitbasierte Vorhersage - verhindert Data Leakage)")
    
    return features_for_loading, features_for_training

def train_model_sync(
    data: pd.DataFrame,
    model_type: str,
    features: List[str],
    target_var: Optional[str],  # Optional wenn zeitbasierte Vorhersage aktiviert
    target_operator: Optional[str],  # Optional wenn zeitbasierte Vorhersage aktiviert
    target_value: Optional[float],  # Optional wenn zeitbasierte Vorhersage aktiviert
    params: dict,
    model_storage_path: str = "/app/models",
    # NEU: Zeitbasierte Parameter
    use_time_based: bool = False,
    future_minutes: Optional[int] = None,
    min_percent_change: Optional[float] = None,
    direction: str = "up",
    phase_intervals: Optional[Dict[int, int]] = None  # {phase_id: interval_seconds}
) -> Dict[str, Any]:
    """
    Trainiert ein ML-Modell (SYNCHRON - wird in run_in_executor aufgerufen!)
    
    ⚠️ WICHTIG: Diese Funktion ist SYNCHRON, weil model.fit() CPU-bound ist.
    Sie wird vom Job Manager in run_in_executor aufgerufen, damit der Event Loop nicht blockiert.
    
    Args:
        data: Bereits geladene Trainingsdaten (DataFrame)
        model_type: "random_forest" oder "xgboost" (nur diese beiden!)
        features: Liste der Feature-Namen (z.B. ["price_open", "price_high"])
        target_var: Ziel-Variable (z.B. "market_cap_close")
        target_operator: Vergleichsoperator (">", "<", ">=", "<=", "=")
        target_value: Schwellwert
        params: Dict mit Hyperparametern (bereits gemergt mit Defaults)
        model_storage_path: Pfad zum Models-Verzeichnis
    
    Returns:
        Dict mit Metriken, Modell-Pfad, Feature Importance
    """
    logger.info(f"🚀 Starte Training: {model_type} mit {len(data)} Zeilen")
    
    # 1. Erstelle Labels
    if use_time_based:
        from app.training.feature_engineering import create_time_based_labels
        # Bei zeitbasierter Vorhersage muss target_var gesetzt sein (für welche Variable wird die Änderung berechnet)
        if not target_var:
            raise ValueError("target_var muss gesetzt sein für zeitbasierte Vorhersage (z.B. 'price_close')")
        logger.info(f"⏰ Zeitbasierte Vorhersage: {future_minutes} Minuten, {min_percent_change}%, Richtung: {direction}")
        
        # ⚠️ WICHTIG: create_time_based_labels gibt (labels, data) zurück wenn NaN entfernt wurden
        result = create_time_based_labels(
            data, 
            target_var, 
            future_minutes, 
            min_percent_change, 
            direction,
            phase_intervals  # NEU: Phase-Intervalle übergeben
        )
        
        # Prüfe ob Tuple zurückgegeben wurde (labels, data) oder nur labels
        if isinstance(result, tuple):
            labels, data = result  # Daten wurden gefiltert (NaN entfernt)
        else:
            labels = result  # Keine Filterung nötig
    else:
        # Normale Labels (aktuelles System)
        if not target_var or not target_operator or target_value is None:
            raise ValueError("target_var, target_operator und target_value müssen gesetzt sein wenn zeitbasierte Vorhersage nicht aktiviert ist")
        labels = create_labels(data, target_var, target_operator, target_value)
    
    positive_count = labels.sum()
    negative_count = len(labels) - positive_count
    
    if positive_count == 0:
        raise ValueError(f"Labels sind nicht ausgewogen: {positive_count} positive, {negative_count} negative. Keine positiven Labels gefunden - Bedingung wird nie erfüllt!")
    if negative_count == 0:
        raise ValueError(f"Labels sind nicht ausgewogen: {positive_count} positive, {negative_count} negative. Keine negativen Labels gefunden - Bedingung wird immer erfüllt!")
    
    # Warnung wenn sehr unausgewogen
    balance_ratio = min(positive_count, negative_count) / max(positive_count, negative_count)
    if balance_ratio < 0.1:
        logger.warning(f"⚠️ Labels sehr unausgewogen: {positive_count} positive, {negative_count} negative (Ratio: {balance_ratio:.2f})")
    
    # 1.4. ATH-Features zur Features-Liste hinzufügen (nach Daten-Laden)
    # ⚠️ WICHTIG: ATH-Features werden bereits im DataFrame berechnet, müssen aber zur features Liste hinzugefügt werden
    include_ath = params.get('include_ath', True)
    logger.info(f"🔍 ATH-Debug: include_ath={include_ath}, rolling_ath in data.columns={('rolling_ath' in data.columns)}")

    if include_ath and 'rolling_ath' in data.columns:
        from app.training.feature_engineering import get_available_ath_features
        ath_features = get_available_ath_features(include_ath=True)
        # Füge nur Features hinzu, die tatsächlich im DataFrame vorhanden sind
        available_ath_features = [f for f in ath_features if f in data.columns]
        features.extend(available_ath_features)
        logger.info(f"🧠 ATH-Features zur Liste hinzugefügt: {len(available_ath_features)} Features ({available_ath_features})")
    else:
        logger.warning(f"⚠️ ATH-Features nicht verfügbar: include_ath={include_ath}, rolling_ath exists={('rolling_ath' in data.columns)}")

    # 1.5. Feature-Engineering: Erstelle zusätzliche Features im DataFrame (wenn aktiviert)
    # ⚠️ WICHTIG: Muss nach Label-Erstellung, aber vor Feature-Vorbereitung erfolgen!
    use_engineered_features = params.get('use_engineered_features', False)  # Default: False für Rückwärtskompatibilität
    
    if use_engineered_features:
        from app.training.feature_engineering import create_pump_detection_features, get_engineered_feature_names
        logger.info("🔧 Erstelle Pump-Detection Features im DataFrame...")
        
        window_sizes = params.get('feature_engineering_windows', [5, 10, 15])  # Konfigurierbar
        
        # Speichere ursprüngliche Spalten
        original_columns = set(data.columns)
        
        # Erstelle engineered features im DataFrame
        data = create_pump_detection_features(data, window_sizes=window_sizes)
        
        # Finde tatsächlich erstellte Features (nur die, die im DataFrame vorhanden sind)
        new_columns = set(data.columns) - original_columns
        engineered_features_created = list(new_columns)
        
        # Erweitere features-Liste um tatsächlich erstellte Features
        features.extend(engineered_features_created)
        
        logger.info(f"✅ {len(engineered_features_created)} zusätzliche Features erstellt und zu Features-Liste hinzugefügt")
        logger.info(f"📊 Gesamt-Features: {len(features)}")
    else:
        logger.info("ℹ️ Feature-Engineering deaktiviert (Standard-Modus)")
    
    # ✅ NEUE Validierung: Prüfe kritische Features
    from app.training.feature_engineering import validate_critical_features, CRITICAL_FEATURES
    
    missing_critical = validate_critical_features(features)
    
    if not missing_critical.get('dev_sold_amount'):
        logger.warning(
            "⚠️ KRITISCH: 'dev_sold_amount' fehlt in Features! "
            "Dies ist der wichtigste Rug-Pull-Indikator!"
        )
    
    if not missing_critical.get('buy_pressure_ratio'):
        logger.warning(
            "⚠️ WICHTIG: 'buy_pressure_ratio' fehlt - "
            "Bot-Spam vs. echtes Interesse kann nicht erkannt werden"
        )
    
    if not missing_critical.get('unique_signer_ratio'):
        logger.warning(
            "⚠️ WICHTIG: 'unique_signer_ratio' fehlt - "
            "Wash-Trading kann nicht erkannt werden"
        )
    
    # 2. Prepare Features (X) und Labels (y)
    # ⚠️ WICHTIG: features enthält jetzt auch engineered und ATH features
    # Verwende nur Features, die tatsächlich in den Daten vorhanden sind
    available_features = [f for f in features if f in data.columns]
    missing_features = [f for f in features if f not in data.columns]

    if missing_features:
        logger.warning(f"⚠️ Einige Features nicht in Daten gefunden (werden übersprungen): {missing_features}")

    if not available_features:
        raise ValueError("Keine Features in Daten gefunden!")

    X = data[available_features].values
    y = labels.values
    logger.info(f"📊 Training mit {len(available_features)} Features ({available_features}), {len(data)} Samples")
    logger.info(f"⚠️ Übersprungene Features: {missing_features}")
    
    # 3. TimeSeriesSplit für Cross-Validation (bei Zeitreihen wichtig!)
    use_timeseries_split = params.get('use_timeseries_split', True)  # Default: True
    
    cv_results = None
    if use_timeseries_split:
        from sklearn.model_selection import TimeSeriesSplit, cross_validate
        
        logger.info("🔀 Verwende TimeSeriesSplit für Cross-Validation...")
        
        # TimeSeriesSplit konfigurieren
        n_splits = params.get('cv_splits', 5)  # Anzahl Splits
        tscv = TimeSeriesSplit(n_splits=n_splits)
        
        # Cross-Validation durchführen
        logger.info(f"📊 Führe {n_splits}-Fold Cross-Validation durch...")
        
        # Erstelle temporäres Modell für CV
        temp_model = create_model(model_type, params)
        
        cv_results = cross_validate(
            estimator=temp_model,
            X=X,
            y=y,
            cv=tscv,
            scoring=['accuracy', 'f1', 'precision', 'recall'],
            return_train_score=True,
            n_jobs=-1  # Parallelisierung
        )
        
        # Ergebnisse loggen
        logger.info("📊 Cross-Validation Ergebnisse:")
        logger.info(f"   Train Accuracy: {cv_results['train_accuracy'].mean():.4f} ± {cv_results['train_accuracy'].std():.4f}")
        logger.info(f"   Test Accuracy:  {cv_results['test_accuracy'].mean():.4f} ± {cv_results['test_accuracy'].std():.4f}")
        logger.info(f"   Train F1:       {cv_results['train_f1'].mean():.4f} ± {cv_results['train_f1'].std():.4f}")
        logger.info(f"   Test F1:        {cv_results['test_f1'].mean():.4f} ± {cv_results['test_f1'].std():.4f}")
        
        # Overfitting-Check
        train_test_gap = cv_results['train_accuracy'].mean() - cv_results['test_accuracy'].mean()
        if train_test_gap > 0.1:
            logger.warning(f"⚠️ OVERFITTING erkannt! Train-Test Gap: {train_test_gap:.2%}")
            logger.warning("   → Modell generalisiert schlecht auf neue Daten")
        
        # Final Model Training auf allen Daten
        logger.info("🎯 Trainiere finales Modell auf allen Daten...")
        
        # Verwende letzten Split für finales Test-Set
        splits = list(tscv.split(X))
        last_train_idx, last_test_idx = splits[-1]
        
        X_final_train, X_final_test = X[last_train_idx], X[last_test_idx]
        y_final_train, y_final_test = y[last_train_idx], y[last_test_idx]
        
        logger.info(f"📊 Final Train-Set: {len(X_final_train)} Zeilen, Test-Set: {len(X_final_test)} Zeilen")
        
    else:
        # Fallback: Einfacher Train-Test-Split (für Rückwärtskompatibilität)
        logger.info("ℹ️ Verwende einfachen Train-Test-Split (nicht empfohlen für Zeitreihen)")
        X_final_train, X_final_test, y_final_train, y_final_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
    
    # 3.5. Imbalanced Data Handling mit SMOTE (auf Train-Set)
    use_smote = params.get('use_smote', True)  # Default: True für bessere Performance
    
    if use_smote:
        # Label-Balance prüfen
        positive_ratio = y_final_train.sum() / len(y_final_train)
        negative_ratio = 1 - positive_ratio
        
        logger.info(f"📊 Label-Balance: {positive_ratio:.2%} positive, {negative_ratio:.2%} negative")
        
        # SMOTE anwenden wenn starkes Ungleichgewicht
        balance_threshold = 0.3  # Wenn < 30% oder > 70% → SMOTE
        if positive_ratio < balance_threshold or positive_ratio > (1 - balance_threshold):
            logger.info("⚖️ Starkes Label-Ungleichgewicht erkannt - Wende SMOTE an...")
            
            try:
                from imblearn.over_sampling import SMOTE
                from imblearn.under_sampling import RandomUnderSampler
                from imblearn.pipeline import Pipeline as ImbPipeline
                
                # SMOTE + Random Under-Sampling Kombination
                # SMOTE erhöht Minority-Klasse, Under-Sampling reduziert Majority-Klasse
                sampling_strategy_smote = 0.5  # Ziel: Minority-Klasse auf 50% der Majority-Klasse
                sampling_strategy_under = 0.8  # Dann: Majority auf 80% der neuen Minority
                
                # K-Neighbors für SMOTE (muss <= Anzahl positive Samples sein)
                k_neighbors = min(5, max(1, int(y_final_train.sum()) - 1))
                
                smote = SMOTE(
                    sampling_strategy=sampling_strategy_smote,
                    random_state=42,
                    k_neighbors=k_neighbors
                )
                under = RandomUnderSampler(
                    sampling_strategy=sampling_strategy_under,
                    random_state=42
                )
                
                # Pipeline erstellen
                pipeline = ImbPipeline([
                    ('smote', smote),
                    ('under', under)
                ])
                
                X_train_balanced, y_train_balanced = pipeline.fit_resample(X_final_train, y_final_train)
                
                logger.info(f"✅ SMOTE abgeschlossen:")
                logger.info(f"   Vorher: {len(X_final_train)} Samples ({y_final_train.sum()} positive, {len(y_final_train) - y_final_train.sum()} negative)")
                logger.info(f"   Nachher: {len(X_train_balanced)} Samples ({y_train_balanced.sum()} positive, {len(y_train_balanced) - y_train_balanced.sum()} negative)")
                logger.info(f"   Neue Balance: {y_train_balanced.sum() / len(y_train_balanced):.2%} positive")
                
                X_final_train = X_train_balanced
                y_final_train = y_train_balanced
                
            except Exception as e:
                logger.warning(f"⚠️ SMOTE fehlgeschlagen: {e} - Training ohne SMOTE fortsetzen")
                logger.warning("   Mögliche Ursachen: Zu wenig positive Samples für SMOTE")
        else:
            logger.info("✅ Label-Balance akzeptabel - Kein SMOTE nötig")
    else:
        logger.info("ℹ️ SMOTE deaktiviert (use_smote=False)")
    
    # 4. Erstelle und trainiere Modell (CPU-BOUND - blockiert!)
    model = create_model(model_type, params)
    logger.info(f"⚙️ Training läuft... (kann einige Minuten dauern)")
    model.fit(X_final_train, y_final_train)  # ⚠️ Blockiert Event Loop - deshalb run_in_executor!
    logger.info(f"✅ Training abgeschlossen")
    
    # 5. Berechne Metriken auf finalem Test-Set
    y_pred = model.predict(X_final_test)
    accuracy = accuracy_score(y_final_test, y_pred)
    f1 = f1_score(y_final_test, y_pred)
    precision = precision_score(y_final_test, y_pred)
    recall = recall_score(y_final_test, y_pred)
    
    logger.info(f"📈 Metriken: Accuracy={accuracy:.4f}, F1={f1:.4f}, Precision={precision:.4f}, Recall={recall:.4f}")
    
    # 5.5. Zusätzliche Metriken berechnen
    from sklearn.metrics import roc_auc_score, matthews_corrcoef, confusion_matrix
    
    # ROC-AUC (benötigt Wahrscheinlichkeiten)
    roc_auc = None
    if hasattr(model, 'predict_proba'):
        try:
            y_pred_proba = model.predict_proba(X_final_test)[:, 1]
            roc_auc = roc_auc_score(y_final_test, y_pred_proba)
            logger.info(f"📊 ROC-AUC: {roc_auc:.4f}")
        except Exception as e:
            logger.warning(f"⚠️ ROC-AUC konnte nicht berechnet werden: {e}")
    else:
        logger.info("ℹ️ Modell unterstützt keine Wahrscheinlichkeiten (predict_proba) - ROC-AUC nicht verfügbar")
    
    # Confusion Matrix Details
    cm = confusion_matrix(y_final_test, y_pred)
    if cm.size == 4:  # 2x2 Matrix
        tn, fp, fn, tp = cm.ravel()
    else:
        tn, fp, fn, tp = 0, 0, 0, 0
    
    # False Positive Rate (wichtig für Pump-Detection!)
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    
    # False Negative Rate
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    
    # Matthews Correlation Coefficient (besser für imbalanced data)
    mcc = matthews_corrcoef(y_final_test, y_pred)
    
    # Profit-Simulation (vereinfacht)
    # Annahme: 1% Gewinn pro richtig erkanntem Pump, 0.5% Verlust pro False Positive
    profit_per_tp = 0.01  # 1%
    loss_per_fp = -0.005  # -0.5%
    simulated_profit = (tp * profit_per_tp) + (fp * loss_per_fp)
    simulated_profit_pct = simulated_profit / len(y_final_test) * 100 if len(y_final_test) > 0 else 0.0
    
    logger.info(f"💰 Simulierter Profit: {simulated_profit_pct:.2f}% (bei {tp} TP, {fp} FP)")
    roc_auc_str = f"{roc_auc:.4f}" if roc_auc is not None else "N/A"
    logger.info(f"📊 Zusätzliche Metriken: ROC-AUC={roc_auc_str}, MCC={mcc:.4f}, FPR={fpr:.4f}, FNR={fnr:.4f}")
    
    # 5.6. Rug-spezifische Metriken berechnen
    rug_metrics = {}
    try:
        y_pred_proba = None
        if hasattr(model, 'predict_proba'):
            y_pred_proba = model.predict_proba(X_final_test)[:, 1]
        
        rug_metrics = calculate_rug_detection_metrics(
            y_true=y_final_test,
            y_pred=y_pred,
            y_pred_proba=y_pred_proba,
            X_test=X_final_test,
            features=features
        )
        
        # Merge mit Standard-Metriken
        logger.info(f"📊 Rug-Detection-Metriken: {rug_metrics}")
    except Exception as e:
        logger.warning(f"⚠️ Fehler beim Berechnen der Rug-Detection-Metriken: {e}")
    
    # 6. Feature Importance extrahieren (wenn verfügbar)
    feature_importance = {}
    if hasattr(model, 'feature_importances_'):
        # Für Random Forest und XGBoost
        importances = model.feature_importances_
        feature_importance = dict(zip(features, importances.tolist()))
        logger.info(f"🎯 Feature Importance: {feature_importance}")
    
    # 7. Speichere Modell als .pkl
    os.makedirs(model_storage_path, exist_ok=True)
    # ⚠️ WICHTIG: UTC-Zeitstempel verwenden!
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    model_filename = f"model_{model_type}_{timestamp}.pkl"
    model_path = os.path.join(model_storage_path, model_filename)
    joblib.dump(model, model_path)
    logger.info(f"💾 Modell gespeichert: {model_path}")
    
    # 8. Return Ergebnisse
    result = {
        "accuracy": float(accuracy),
        "f1": float(f1),
        "precision": float(precision),
        "recall": float(recall),
        "roc_auc": float(roc_auc) if roc_auc else None,  # NEU
        "mcc": float(mcc),  # NEU
        "fpr": float(fpr),  # NEU
        "fnr": float(fnr),  # NEU
        "confusion_matrix": {  # NEU
            "tp": int(tp),
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn)
        },
        "simulated_profit_pct": float(simulated_profit_pct),  # NEU
        "rug_detection_metrics": rug_metrics,  # NEU: Rug-spezifische Metriken
        "model_path": model_path,
        "feature_importance": feature_importance,  # Als Dict (für JSONB)
        "num_samples": len(data),
        "num_features": len(features),
        "features": features  # ⚠️ WICHTIG: Erweiterte Features-Liste (inkl. engineered features) zurückgeben
    }
    
    # NEU: CV-Ergebnisse hinzufügen (wenn verfügbar)
    if cv_results is not None:
        result["cv_scores"] = {
            "train_accuracy": cv_results['train_accuracy'].tolist(),
            "test_accuracy": cv_results['test_accuracy'].tolist(),
            "train_f1": cv_results['train_f1'].tolist(),
            "test_f1": cv_results['test_f1'].tolist(),
            "train_precision": cv_results['train_precision'].tolist(),
            "test_precision": cv_results['test_precision'].tolist(),
            "train_recall": cv_results['train_recall'].tolist(),
            "test_recall": cv_results['test_recall'].tolist()
        }
        result["cv_overfitting_gap"] = float(
            cv_results['train_accuracy'].mean() - cv_results['test_accuracy'].mean()
        )
    
    return result

async def train_model(
    model_type: str,
    features: List[str],
    target_var: Optional[str],  # Optional wenn zeitbasierte Vorhersage aktiviert
    target_operator: Optional[str],  # Optional wenn zeitbasierte Vorhersage aktiviert
    target_value: Optional[float],  # Optional wenn zeitbasierte Vorhersage aktiviert
    train_start: str | datetime,
    train_end: str | datetime,
    phases: Optional[List[int]] = None,
    params: Optional[Dict[str, Any]] = None,
    model_name: Optional[str] = None,
    model_storage_path: str = "/app/models",
    # NEU: Zeitbasierte Parameter
    use_time_based: bool = False,
    future_minutes: Optional[int] = None,
    min_percent_change: Optional[float] = None,
    direction: str = "up"
) -> Dict[str, Any]:
    """
    Async Wrapper für train_model_sync
    Lädt Daten async, ruft dann sync-Funktion in run_in_executor auf
    
    ⚠️ KRITISCH: CPU-bound Training läuft in run_in_executor!
    
    Args:
        model_type: "random_forest" oder "xgboost" (nur diese beiden!)
        features: Liste der Feature-Namen
        target_var: Ziel-Variable
        target_operator: Vergleichsoperator
        target_value: Schwellwert
        train_start: Start-Zeitpunkt (ISO-Format oder datetime)
        train_end: Ende-Zeitpunkt (ISO-Format oder datetime)
        phases: Liste der Coin-Phasen oder None
        params: Dict mit Hyperparametern (optional, überschreibt Defaults)
        model_name: Name des Modells (optional, wird nicht verwendet)
        model_storage_path: Pfad zum Models-Verzeichnis
    
    Returns:
        Dict mit Metriken, Modell-Pfad, Feature Importance
    """
    import asyncio
    
    logger.info(f"🎯 Starte Modell-Training: {model_type}")
    
    # 0.5. Wenn keine Features übergeben wurden, verwende Defaults
    exclude_features = (params or {}).get('exclude_features', [])
    if not features:
        features = DEFAULT_FEATURES.copy()
        logger.info(f"📊 Verwende Default-Features: {len(features)} Features")
    
    # Entferne ausgeschlossene Features
    if exclude_features:
        features = [f for f in features if f not in exclude_features]
        logger.info(f"📊 Features nach Ausschluss: {len(features)} Features (ausgeschlossen: {exclude_features})")
    
    # 1. Lade Default-Parameter aus DB (async)
    default_params = await get_model_type_defaults(model_type)
    logger.info(f"📋 Default-Parameter: {default_params}")
    
    # 2. Merge mit übergebenen Parametern (übergebene überschreiben Defaults)
    final_params = {**default_params, **(params or {})}
    logger.info(f"⚙️ Finale Parameter: {final_params}")
    
    # 2.3. Prüfe ob Feature-Engineering aktiviert ist (für später in train_model_sync)
    use_engineered_features = final_params.get('use_engineered_features', False)
    if use_engineered_features:
        logger.info("🔧 Feature-Engineering aktiviert (wird nach Daten-Laden durchgeführt)")

    # 2.4. ATH-Features werden später nach Daten-Laden hinzugefügt
    # (nicht hier, da sie erst in Python berechnet werden)
    
    # 2.5. Filtere ATH-Features aus der Loading-Liste (werden später berechnet)
    # ⚠️ WICHTIG: ATH-Features existieren nicht in der Datenbank und werden erst in Python berechnet
    ath_features = ['rolling_ath', 'ath_distance_pct', 'ath_breakout', 'minutes_since_ath', 'ath_age_hours', 'ath_is_recent', 'ath_is_old']
    features_for_db = [f for f in features if f not in ath_features]

    # Bereite Features vor (verhindert Data Leakage bei zeitbasierter Vorhersage)
    features_for_loading, features_for_training = prepare_features_for_training(
        features=features_for_db,  # Basis-Features ohne ATH-Features (nicht in DB)
        target_var=target_var,
        use_time_based=use_time_based
    )
    logger.info(f"📊 Features für Laden: {len(features_for_loading)} - {features_for_loading}")
    logger.info(f"📊 Features für Training: {len(features_for_training)} - {features_for_training}")
    logger.info(f"🧠 ATH-Features werden nach Daten-Laden hinzugefügt: {len(ath_features)} Features - {ath_features}")
    logger.info(f"🧠 Original features: {features}")
    logger.info(f"🧠 features_for_db: {features_for_db}")

    # Aktualisiere die features Liste für weitere Verarbeitung
    features = features_for_training.copy()
    
    # 2.6. Prüfe ATH-Daten-Verfügbarkeit (wenn include_ath aktiviert)
    from app.training.feature_engineering import validate_ath_data_availability
    # Prüfe automatisch, ob ATH-Features in der Features-Liste sind
    ath_feature_names = ['rolling_ath', 'ath_distance_pct', 'ath_breakout', 'minutes_since_ath', 'ath_age_hours', 'ath_is_recent', 'ath_is_old']
    has_ath_features = any(f in features for f in ath_feature_names)
    include_ath = final_params.get('include_ath', has_ath_features)  # Auto-detect ATH-Features
    
    if include_ath:
        ath_validation = await validate_ath_data_availability(train_start, train_end)
        if not ath_validation["available"]:
            logger.warning(f"⚠️ Keine ATH-Daten verfügbar! Coverage: {ath_validation.get('coverage_pct', 0):.1f}%")
        else:
            logger.info(f"✅ ATH-Daten verfügbar: {ath_validation['coins_with_ath']}/{ath_validation['total_coins']} Coins ({ath_validation['coverage_pct']:.1f}%)")
    
    # 3. Lade Trainingsdaten (async) - mit target_var für Labels
    data = await load_training_data(
        train_start=train_start,
        train_end=train_end,
        features=features_for_loading,  # Enthält target_var (für Labels benötigt)
        phases=phases,
        include_ath=include_ath  # 🆕 ATH-Daten optional laden
    )
    
    if len(data) == 0:
        raise ValueError("Keine Trainingsdaten gefunden!")
    
    # 3.5. Lade Marktstimmung (SOL-Preis-Kontext) - OPTIONAL
    use_market_context = final_params.get('use_market_context', False)
    
    if use_market_context:
        from app.training.feature_engineering import enrich_with_market_context
        logger.info("🌍 Füge Marktstimmung (SOL-Preis-Kontext) hinzu...")
        data = await enrich_with_market_context(
            data, 
            train_start=train_start, 
            train_end=train_end
        )
        
        # Füge Context-Features zu Features-Liste hinzu
        context_features = [
            "sol_price_usd",
            "sol_price_change_pct",
            "sol_price_ma_5",
            "sol_price_volatility"
        ]
        # Nur hinzufügen wenn nicht bereits vorhanden
        for cf in context_features:
            if cf not in features and cf in data.columns:
                features.append(cf)
                logger.info(f"➕ Context-Feature '{cf}' hinzugefügt")
    else:
        logger.info("ℹ️ Marktstimmung deaktiviert (use_market_context=False)")
    
    # 3.6. Lade Phase-Intervalle (falls zeitbasierte Vorhersage aktiviert)
    phase_intervals = None
    if use_time_based:
        from app.database.models import get_phase_intervals
        phase_intervals = await get_phase_intervals()
        logger.info(f"📊 {len(phase_intervals)} Phase-Intervalle geladen für zeitbasierte Vorhersage")
    
    # 4. Führe CPU-bound Training in Executor aus (blockiert Event Loop NICHT!)
    loop = asyncio.get_running_loop()
    logger.info(f"🔄 Starte Training in Executor (blockiert Event Loop nicht)...")
    result = await loop.run_in_executor(
        None,  # Nutzt default ThreadPoolExecutor
        train_model_sync,
        data,  # Bereits geladene Daten
        model_type,
        features,  # ✅ Enthält target_var NICHT bei zeitbasierter Vorhersage!
        target_var,
        target_operator,
        target_value,
        final_params,  # Bereits gemergte Parameter
        model_storage_path,
        # NEU: Zeitbasierte Parameter
        use_time_based,
        future_minutes,
        min_percent_change,
        direction,
        phase_intervals  # NEU: Phase-Intervalle übergeben
    )
    
    logger.info(f"✅ Training erfolgreich abgeschlossen!")
    return result

def calculate_rug_detection_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_pred_proba: Optional[np.ndarray],
    X_test: np.ndarray,
    features: List[str]
) -> Dict[str, Any]:
    """
    Berechnet Rug-Pull-spezifische Metriken.
    
    Args:
        y_true: Echte Labels
        y_pred: Vorhergesagte Labels
        y_pred_proba: Vorhergesagte Wahrscheinlichkeiten (optional)
        X_test: Test-Features
        features: Liste der Feature-Namen
    
    Returns:
        Dict mit Rug-Detection-Metriken
    """
    from sklearn.metrics import confusion_matrix
    
    metrics = {}
    
    # 1. Dev-Sold Detection Rate (wenn Feature vorhanden)
    if 'dev_sold_amount' in features:
        try:
            dev_sold_idx = features.index('dev_sold_amount')
            dev_sold_mask = X_test[:, dev_sold_idx] > 0
            
            if dev_sold_mask.sum() > 0:
                dev_sold_detected = (y_pred[dev_sold_mask] == 1).sum()
                metrics['dev_sold_detection_rate'] = float(dev_sold_detected / dev_sold_mask.sum())
                logger.info(f"📊 Dev-Sold Detection Rate: {metrics['dev_sold_detection_rate']:.2%}")
        except (ValueError, IndexError) as e:
            logger.warning(f"⚠️ Konnte Dev-Sold Detection Rate nicht berechnen: {e}")
    
    # 2. Wash-Trading Detection (wenn Ratio vorhanden)
    if 'unique_signer_ratio' in features:
        try:
            ratio_idx = features.index('unique_signer_ratio')
            wash_trading_mask = X_test[:, ratio_idx] < 0.15
            
            if wash_trading_mask.sum() > 0:
                wash_detected = (y_pred[wash_trading_mask] == 1).sum()
                metrics['wash_trading_detection_rate'] = float(wash_detected / wash_trading_mask.sum())
                logger.info(f"📊 Wash-Trading Detection Rate: {metrics['wash_trading_detection_rate']:.2%}")
        except (ValueError, IndexError) as e:
            logger.warning(f"⚠️ Konnte Wash-Trading Detection Rate nicht berechnen: {e}")
    
    # 3. False Negative Cost (bei Rug-Pull-Detection ist FN teurer als FP!)
    try:
        cm = confusion_matrix(y_true, y_pred)
        if cm.size == 4:  # 2x2 Matrix
            tn, fp, fn, tp = cm.ravel()
            
            # FN = Rug wurde nicht erkannt (sehr teuer!)
            # FP = False Alarm (weniger schlimm)
            fn_cost = fn * 10.0  # FN ist 10x teurer
            fp_cost = fp * 1.0
            metrics['weighted_cost'] = float(fn_cost + fp_cost)
            logger.info(f"💰 Weighted Cost: {metrics['weighted_cost']:.2f} (FN={fn}, FP={fp})")
    except Exception as e:
        logger.warning(f"⚠️ Konnte Weighted Cost nicht berechnen: {e}")
    
    # 4. Profit @ Top-K (wenn Wahrscheinlichkeiten vorhanden)
    if y_pred_proba is not None:
        try:
            for k in [10, 20, 50, 100]:
                if len(y_pred_proba) >= k:
                    top_k_idx = np.argsort(y_pred_proba)[-k:]
                    precision_at_k = y_true[top_k_idx].sum() / k
                    metrics[f'precision_at_{k}'] = float(precision_at_k)
                    logger.info(f"📊 Precision @ Top-{k}: {precision_at_k:.2%}")
        except Exception as e:
            logger.warning(f"⚠️ Konnte Precision @ Top-K nicht berechnen: {e}")
    
    return metrics

