"""
Datenbank-Modelle für ML Training Service
Alle CRUD-Operationen für ml_models, ml_test_results, ml_comparisons, ml_jobs
"""
import logging
from typing import Optional, List, Dict, Any
from datetime import datetime
from app.database.connection import get_pool
from app.database.utils import to_jsonb, from_jsonb, convert_jsonb_fields

logger = logging.getLogger(__name__)

# ============================================================
# ml_models - CRUD Operationen
# ============================================================

async def ensure_unique_model_name(name: str) -> str:
    """
    Stellt sicher, dass der Modell-Name eindeutig ist.
    Falls der Name bereits existiert, wird ein eindeutiger Name generiert.
    
    Args:
        name: Ursprünglicher Modell-Name
    
    Returns:
        Eindeutiger Modell-Name
    """
    pool = await get_pool()
    original_name = name
    
    # Prüfe ob Name bereits existiert
    existing = await pool.fetchval(
        "SELECT id FROM ml_models WHERE name = $1 AND is_deleted = FALSE",
        name
    )
    
    if existing is None:
        # Name ist eindeutig
        return name
    
    # Name existiert bereits - generiere eindeutigen Namen
    import time
    timestamp = int(time.time() * 1000)  # Millisekunden für bessere Eindeutigkeit
    counter = 1
    
    while True:
        new_name = f"{original_name}_{timestamp}_{counter}"
        existing = await pool.fetchval(
            "SELECT id FROM ml_models WHERE name = $1 AND is_deleted = FALSE",
            new_name
        )
        if existing is None:
            logger.warning(f"⚠️ Modell-Name '{original_name}' bereits vorhanden - verwende '{new_name}'")
            return new_name
        counter += 1
        if counter > 1000:  # Sicherheitsgrenze
            # Fallback: Verwende nur Timestamp
            new_name = f"{original_name}_{timestamp}"
            logger.warning(f"⚠️ Konnte keinen eindeutigen Namen generieren - verwende '{new_name}'")
            return new_name

async def create_model(
    name: str,
    model_type: str,
    target_variable: str,
    train_start: datetime,
    train_end: datetime,
    features: List[str],
    target_operator: Optional[str] = None,  # Optional für zeitbasierte Vorhersagen
    target_value: Optional[float] = None,  # Optional für zeitbasierte Vorhersagen
    phases: Optional[List[int]] = None,
    params: Optional[Dict[str, Any]] = None,
    training_accuracy: Optional[float] = None,
    training_f1: Optional[float] = None,
    training_precision: Optional[float] = None,
    training_recall: Optional[float] = None,
    feature_importance: Optional[Dict[str, float]] = None,
    model_file_path: Optional[str] = None,
    description: Optional[str] = None,
    status: str = "TRAINING",
    cv_scores: Optional[Dict[str, Any]] = None,  # NEU: CV-Ergebnisse
    cv_overfitting_gap: Optional[float] = None,  # NEU: Overfitting-Gap
    roc_auc: Optional[float] = None,  # NEU: ROC-AUC
    mcc: Optional[float] = None,  # NEU: MCC
    fpr: Optional[float] = None,  # NEU: False Positive Rate
    fnr: Optional[float] = None,  # NEU: False Negative Rate
    confusion_matrix: Optional[Dict[str, int]] = None,  # NEU: Confusion Matrix
    simulated_profit_pct: Optional[float] = None,  # NEU: Simulierter Profit
    # Zeitbasierte Vorhersage-Parameter
    future_minutes: Optional[int] = None,
    price_change_percent: Optional[float] = None,
    target_direction: Optional[str] = None
) -> int:
    """Erstellt ein neues Modell in ml_models"""
    pool = await get_pool()
    
    # ✅ SICHERHEIT: Stelle sicher, dass der Name eindeutig ist
    # ⚠️ WICHTIG: Rufe ensure_unique_model_name NUR EINMAL auf, um Race Conditions zu vermeiden!
    original_name = name
    name = await ensure_unique_model_name(name)
    if name != original_name:
        logger.info(f"📝 Modell-Name angepasst: '{original_name}' → '{name}'")
    
    # Konvertiere Listen/Dicts zu JSONB-Strings (refactored: nutze Helper-Funktion)
    features_jsonb = to_jsonb(features)
    phases_jsonb = to_jsonb(phases)
    params_jsonb = to_jsonb(params)
    feature_importance_jsonb = to_jsonb(feature_importance)
    cv_scores_jsonb = to_jsonb(cv_scores)  # NEU
    confusion_matrix_jsonb = to_jsonb(confusion_matrix)  # NEU
    
    # Prüfe ob neue Spalten existieren (für Rückwärtskompatibilität)
    # Falls nicht, wird sie ignoriert
    try:
        model_id = await pool.fetchval(
            """
            INSERT INTO ml_models (
                name, model_type, status,
                target_variable, target_operator, target_value,
                train_start, train_end,
                features, phases, params,
                training_accuracy, training_f1, training_precision, training_recall,
                feature_importance, model_file_path, description,
                cv_scores, cv_overfitting_gap,
                roc_auc, mcc, fpr, fnr, confusion_matrix, simulated_profit_pct,
                future_minutes, price_change_percent, target_direction
            ) VALUES (
                $1, $2, $3, $4, $5, $6, $7, $8, $9::jsonb, $10::jsonb, $11::jsonb, 
                $12, $13, $14, $15, $16::jsonb, $17, $18, $19::jsonb, $20,
                $21, $22, $23, $24, $25::jsonb, $26, $27, $28, $29
            ) RETURNING id
            """,
            name, model_type, status,
            target_variable, target_operator, target_value,
            train_start, train_end,
            features_jsonb, phases_jsonb, params_jsonb,
            training_accuracy, training_f1, training_precision, training_recall,
            feature_importance_jsonb, model_file_path, description,
            cv_scores_jsonb, cv_overfitting_gap,
            roc_auc, mcc, fpr, fnr, confusion_matrix_jsonb, simulated_profit_pct,
            future_minutes, price_change_percent, target_direction
        )
    except Exception as e:
        # ✅ SICHERHEIT: Prüfe zuerst auf Duplikat-Fehler (Race Condition)
        error_str = str(e).lower()
        if 'duplicate key' in error_str and 'ml_models_name_key' in error_str:
            # Name wurde zwischen Prüfung und INSERT von anderem Prozess verwendet
            # ⚠️ WICHTIG: Verwende den ORIGINALEN Namen, nicht den bereits angepassten!
            logger.warning(f"⚠️ Duplikat-Fehler erkannt (Race Condition) - generiere neuen eindeutigen Namen...")
            name = await ensure_unique_model_name(original_name if 'original_name' in locals() else name)
            # Versuche erneut mit neuem Namen
            try:
                model_id = await pool.fetchval(
                    """
                    INSERT INTO ml_models (
                        name, model_type, status,
                        target_variable, target_operator, target_value,
                        train_start, train_end,
                        features, phases, params,
                        training_accuracy, training_f1, training_precision, training_recall,
                        feature_importance, model_file_path, description,
                        cv_scores, cv_overfitting_gap,
                        roc_auc, mcc, fpr, fnr, confusion_matrix, simulated_profit_pct,
                        future_minutes, price_change_percent, target_direction
                    ) VALUES (
                        $1, $2, $3, $4, $5, $6, $7, $8, $9::jsonb, $10::jsonb, $11::jsonb, 
                        $12, $13, $14, $15, $16::jsonb, $17, $18, $19::jsonb, $20,
                        $21, $22, $23, $24, $25::jsonb, $26, $27, $28, $29
                    ) RETURNING id
                    """,
                    name, model_type, status,
                    target_variable, target_operator, target_value,
                    train_start, train_end,
                    features_jsonb, phases_jsonb, params_jsonb,
                    training_accuracy, training_f1, training_precision, training_recall,
                    feature_importance_jsonb, model_file_path, description,
                    cv_scores_jsonb, cv_overfitting_gap,
                    roc_auc, mcc, fpr, fnr, confusion_matrix_jsonb, simulated_profit_pct,
                    future_minutes, price_change_percent, target_direction
                )
            except Exception as e3:
                # Falls auch das fehlschlägt, verwende Fallback-Versionen
                logger.error(f"❌ Fehler beim Erstellen mit eindeutigem Namen: {e3}")
                raise
        elif any(col in error_str for col in ['cv_scores', 'cv_overfitting_gap', 'roc_auc', 'mcc', 'fpr', 'fnr', 'confusion_matrix', 'simulated_profit_pct']):
            logger.warning(f"⚠️ Neue Metriken-Spalten nicht gefunden - verwende Fallback (ohne zusätzliche Metriken)")
            try:
                # Versuche mit CV-Spalten (falls vorhanden)
                model_id = await pool.fetchval(
                    """
                    INSERT INTO ml_models (
                        name, model_type, status,
                        target_variable, target_operator, target_value,
                        train_start, train_end,
                        features, phases, params,
                        training_accuracy, training_f1, training_precision, training_recall,
                        feature_importance, model_file_path, description,
                        cv_scores, cv_overfitting_gap
                    ) VALUES (
                        $1, $2, $3, $4, $5, $6, $7, $8, $9::jsonb, $10::jsonb, $11::jsonb, 
                        $12, $13, $14, $15, $16::jsonb, $17, $18, $19::jsonb, $20
                    ) RETURNING id
                    """,
                    name, model_type, status,
                    target_variable, target_operator, target_value,
                    train_start, train_end,
                    features_jsonb, phases_jsonb, params_jsonb,
                    training_accuracy, training_f1, training_precision, training_recall,
                    feature_importance_jsonb, model_file_path, description,
                    cv_scores_jsonb, cv_overfitting_gap
                )
            except Exception as e2:
                # ✅ SICHERHEIT: Prüfe auch hier auf Duplikat-Fehler
                error_str2 = str(e2).lower()
                if 'duplicate key' in error_str2 and 'ml_models_name_key' in error_str2:
                    logger.warning(f"⚠️ Duplikat-Fehler im Fallback-Block - generiere neuen eindeutigen Namen...")
                    name = await ensure_unique_model_name(original_name if 'original_name' in locals() else name)
                    # Versuche erneut mit neuem Namen
                    try:
                        model_id = await pool.fetchval(
                            """
                            INSERT INTO ml_models (
                                name, model_type, status,
                                target_variable, target_operator, target_value,
                                train_start, train_end,
                                features, phases, params,
                                training_accuracy, training_f1, training_precision, training_recall,
                                feature_importance, model_file_path, description,
                                cv_scores, cv_overfitting_gap
                            ) VALUES (
                                $1, $2, $3, $4, $5, $6, $7, $8, $9::jsonb, $10::jsonb, $11::jsonb, 
                                $12, $13, $14, $15, $16::jsonb, $17, $18, $19::jsonb, $20
                            ) RETURNING id
                            """,
                            name, model_type, status,
                            target_variable, target_operator, target_value,
                            train_start, train_end,
                            features_jsonb, phases_jsonb, params_jsonb,
                            training_accuracy, training_f1, training_precision, training_recall,
                            feature_importance_jsonb, model_file_path, description,
                            cv_scores_jsonb, cv_overfitting_gap
                        )
                    except Exception as e3:
                        # Falls auch das fehlschlägt, verwende Standard-Spalten
                        error_str3 = str(e3).lower()
                        if 'duplicate key' in error_str3 and 'ml_models_name_key' in error_str3:
                            name = await ensure_unique_model_name(original_name if 'original_name' in locals() else name)
                        model_id = await pool.fetchval(
                            """
                            INSERT INTO ml_models (
                                name, model_type, status,
                                target_variable, target_operator, target_value,
                                train_start, train_end,
                                features, phases, params,
                                training_accuracy, training_f1, training_precision, training_recall,
                                feature_importance, model_file_path, description
                            ) VALUES (
                                $1, $2, $3, $4, $5, $6, $7, $8, $9::jsonb, $10::jsonb, $11::jsonb, $12, $13, $14, $15, $16::jsonb, $17, $18
                            ) RETURNING id
                            """,
                            name, model_type, status,
                            target_variable, target_operator, target_value,
                            train_start, train_end,
                            features_jsonb, phases_jsonb, params_jsonb,
                            training_accuracy, training_f1, training_precision, training_recall,
                            feature_importance_jsonb, model_file_path, description
                        )
                else:
                    # Fallback: Nur Standard-Spalten
                    try:
                        model_id = await pool.fetchval(
                            """
                            INSERT INTO ml_models (
                                name, model_type, status,
                                target_variable, target_operator, target_value,
                                train_start, train_end,
                                features, phases, params,
                                training_accuracy, training_f1, training_precision, training_recall,
                                feature_importance, model_file_path, description
                            ) VALUES (
                                $1, $2, $3, $4, $5, $6, $7, $8, $9::jsonb, $10::jsonb, $11::jsonb, $12, $13, $14, $15, $16::jsonb, $17, $18
                            ) RETURNING id
                            """,
                            name, model_type, status,
                            target_variable, target_operator, target_value,
                            train_start, train_end,
                            features_jsonb, phases_jsonb, params_jsonb,
                            training_accuracy, training_f1, training_precision, training_recall,
                            feature_importance_jsonb, model_file_path, description
                        )
                    except Exception as e4:
                        # ✅ SICHERHEIT: Prüfe auch im Standard-Fallback auf Duplikat-Fehler
                        error_str4 = str(e4).lower()
                        if 'duplicate key' in error_str4 and 'ml_models_name_key' in error_str4:
                            logger.warning(f"⚠️ Duplikat-Fehler im Standard-Fallback - generiere neuen eindeutigen Namen...")
                            name = await ensure_unique_model_name(original_name if 'original_name' in locals() else name)
                            # Versuche erneut mit neuem Namen
                            model_id = await pool.fetchval(
                                """
                                INSERT INTO ml_models (
                                    name, model_type, status,
                                    target_variable, target_operator, target_value,
                                    train_start, train_end,
                                    features, phases, params,
                                    training_accuracy, training_f1, training_precision, training_recall,
                                    feature_importance, model_file_path, description
                                ) VALUES (
                                    $1, $2, $3, $4, $5, $6, $7, $8, $9::jsonb, $10::jsonb, $11::jsonb, $12, $13, $14, $15, $16::jsonb, $17, $18
                                ) RETURNING id
                                """,
                                name, model_type, status,
                                target_variable, target_operator, target_value,
                                train_start, train_end,
                                features_jsonb, phases_jsonb, params_jsonb,
                                training_accuracy, training_f1, training_precision, training_recall,
                                feature_importance_jsonb, model_file_path, description
                            )
                        else:
                            raise
        else:
            raise
    logger.info(f"✅ Modell erstellt: {name} (ID: {model_id})")
    return model_id

async def get_model(model_id: int) -> Optional[Dict[str, Any]]:
    """Holt ein Modell aus der DB"""
    pool = await get_pool()
    row = await pool.fetchrow(
        "SELECT * FROM ml_models WHERE id = $1",
        model_id
    )
    if not row:
        return None
    
    # Konvertiere JSONB-Felder von Strings zu Python-Objekten (refactored: nutze Helper-Funktion)
    model_dict = dict(row)
    jsonb_fields = ['features', 'phases', 'params', 'feature_importance', 'cv_scores', 'confusion_matrix']  # NEU: confusion_matrix
    model_dict = convert_jsonb_fields(model_dict, jsonb_fields, direction="from")
    return model_dict

async def update_model(
    model_id: int,
    status: Optional[str] = None,
    name: Optional[str] = None,
    description: Optional[str] = None,
    training_accuracy: Optional[float] = None,
    training_f1: Optional[float] = None,
    training_precision: Optional[float] = None,
    training_recall: Optional[float] = None,
    feature_importance: Optional[Dict[str, float]] = None,
    model_file_path: Optional[str] = None,
    **kwargs
) -> bool:
    """Aktualisiert ein Modell"""
    pool = await get_pool()
    
    updates = []
    values = []
    param_num = 1
    
    if name is not None:
        updates.append(f"name = ${param_num}")
        values.append(name)
        param_num += 1
    if description is not None:
        updates.append(f"description = ${param_num}")
        values.append(description)
        param_num += 1
    if status is not None:
        updates.append(f"status = ${param_num}")
        values.append(status)
        param_num += 1
    if training_accuracy is not None:
        updates.append(f"training_accuracy = ${param_num}")
        values.append(training_accuracy)
        param_num += 1
    if training_f1 is not None:
        updates.append(f"training_f1 = ${param_num}")
        values.append(training_f1)
        param_num += 1
    if training_precision is not None:
        updates.append(f"training_precision = ${param_num}")
        values.append(training_precision)
        param_num += 1
    if training_recall is not None:
        updates.append(f"training_recall = ${param_num}")
        values.append(training_recall)
        param_num += 1
    if feature_importance is not None:
        updates.append(f"feature_importance = ${param_num}")
        values.append(feature_importance)
        param_num += 1
    if model_file_path is not None:
        updates.append(f"model_file_path = ${param_num}")
        values.append(model_file_path)
        param_num += 1
    
    if not updates:
        return False
    
    updates.append(f"updated_at = NOW()")
    values.append(model_id)
    
    query = f"UPDATE ml_models SET {', '.join(updates)} WHERE id = ${param_num}"
    await pool.execute(query, *values)
    logger.info(f"✅ Modell aktualisiert: ID {model_id}")
    return True

async def list_models(
    status: Optional[str] = None,
    is_deleted: bool = False,
    limit: int = 100,
    offset: int = 0
) -> List[Dict[str, Any]]:
    """Listet Modelle mit optionalen Filtern"""
    pool = await get_pool()
    
    conditions = ["is_deleted = $1"]
    params = [is_deleted]
    param_num = 2
    
    if status:
        conditions.append(f"status = ${param_num}")
        params.append(status)
        param_num += 1
    
    where_clause = " AND ".join(conditions)
    params.extend([limit, offset])
    
    rows = await pool.fetch(
        f"""
        SELECT * FROM ml_models 
        WHERE {where_clause}
        ORDER BY created_at DESC
        LIMIT ${param_num} OFFSET ${param_num + 1}
        """,
        *params
    )
    # Konvertiere JSONB-Felder von Strings zu Python-Objekten (refactored: nutze Helper-Funktion)
    result = []
    jsonb_fields = ['features', 'phases', 'params', 'feature_importance', 'cv_scores', 'confusion_matrix']  # NEU: cv_scores, confusion_matrix
    for row in rows:
        model_dict = dict(row)
        model_dict = convert_jsonb_fields(model_dict, jsonb_fields, direction="from")
        result.append(model_dict)
    return result

async def delete_model(model_id: int) -> bool:
    """Soft-Delete: Markiert Modell als gelöscht"""
    pool = await get_pool()
    await pool.execute(
        "UPDATE ml_models SET is_deleted = TRUE, deleted_at = NOW() WHERE id = $1",
        model_id
    )
    logger.info(f"✅ Modell gelöscht (soft): ID {model_id}")
    return True

# ============================================================
# ml_test_results - CRUD Operationen
# ============================================================

async def create_test_result(
    model_id: int,
    test_start: datetime,
    test_end: datetime,
    accuracy: Optional[float] = None,
    f1_score: Optional[float] = None,
    precision_score: Optional[float] = None,
    recall: Optional[float] = None,
    roc_auc: Optional[float] = None,
    # Zusätzliche Metriken (Phase 9)
    mcc: Optional[float] = None,
    fpr: Optional[float] = None,
    fnr: Optional[float] = None,
    simulated_profit_pct: Optional[float] = None,
    confusion_matrix: Optional[Dict[str, int]] = None,
    # Legacy: Confusion Matrix als einzelne Felder
    tp: Optional[int] = None,
    tn: Optional[int] = None,
    fp: Optional[int] = None,
    fn: Optional[int] = None,
    num_samples: Optional[int] = None,
    num_positive: Optional[int] = None,
    num_negative: Optional[int] = None,
    has_overlap: bool = False,
    overlap_note: Optional[str] = None,
    feature_importance: Optional[Dict[str, float]] = None,
    # Train vs. Test Vergleich (Phase 2)
    train_accuracy: Optional[float] = None,
    train_f1: Optional[float] = None,
    train_precision: Optional[float] = None,
    train_recall: Optional[float] = None,
    accuracy_degradation: Optional[float] = None,
    f1_degradation: Optional[float] = None,
    is_overfitted: Optional[bool] = None,
    # Test-Zeitraum Info (Phase 2)
    test_duration_days: Optional[float] = None
) -> int:
    """Erstellt ein Test-Ergebnis"""
    pool = await get_pool()
    
    # Konvertiere JSONB-Felder (refactored: nutze Helper-Funktion)
    confusion_matrix_jsonb = to_jsonb(confusion_matrix)
    feature_importance_jsonb = to_jsonb(feature_importance)
    
    test_id = await pool.fetchval(
        """
        INSERT INTO ml_test_results (
            model_id, test_start, test_end,
            accuracy, f1_score, precision_score, recall, roc_auc,
            mcc, fpr, fnr, simulated_profit_pct, confusion_matrix,
            tp, tn, fp, fn,
            num_samples, num_positive, num_negative,
            has_overlap, overlap_note, feature_importance,
            train_accuracy, train_f1, train_precision, train_recall,
            accuracy_degradation, f1_degradation, is_overfitted,
            test_duration_days
        ) VALUES (
            $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13::jsonb, $14, $15, $16, $17, $18, $19, $20, $21, $22, $23::jsonb,
            $24, $25, $26, $27, $28, $29, $30, $31
        ) RETURNING id
        """,
        model_id, test_start, test_end,
        accuracy, f1_score, precision_score, recall, roc_auc,
        mcc, fpr, fnr, simulated_profit_pct, confusion_matrix_jsonb,
        tp, tn, fp, fn,
        num_samples, num_positive, num_negative,
        has_overlap, overlap_note, feature_importance_jsonb,
        train_accuracy, train_f1, train_precision, train_recall,
        accuracy_degradation, f1_degradation, is_overfitted,
        test_duration_days
    )
    logger.info(f"✅ Test-Ergebnis erstellt: ID {test_id} für Modell {model_id}")
    return test_id

async def get_test_result(test_id: int) -> Optional[Dict[str, Any]]:
    """Holt ein einzelnes Test-Ergebnis"""
    pool = await get_pool()
    row = await pool.fetchrow(
        "SELECT * FROM ml_test_results WHERE id = $1",
        test_id
    )
    if not row:
        return None
    # Konvertiere JSONB-Felder (refactored: nutze Helper-Funktion)
    test_dict = dict(row)
    jsonb_fields = ['feature_importance', 'confusion_matrix']
    test_dict = convert_jsonb_fields(test_dict, jsonb_fields, direction="from")
    return test_dict

async def list_all_test_results(
    limit: int = 100,
    offset: int = 0
) -> List[Dict[str, Any]]:
    """Listet alle Test-Ergebnisse"""
    import json
    pool = await get_pool()
    
    rows = await pool.fetch(
        """
        SELECT * FROM ml_test_results 
        ORDER BY created_at DESC
        LIMIT $1 OFFSET $2
        """,
        limit, offset
    )
    
    # Konvertiere JSONB-Felder von Strings zu Python-Objekten
    result = []
    for row in rows:
        test_dict = dict(row)
        jsonb_fields = ['feature_importance', 'confusion_matrix']
        for field in jsonb_fields:
            if field in test_dict and test_dict[field] is not None:
                if isinstance(test_dict[field], str):
                    try:
                        test_dict[field] = json.loads(test_dict[field])
                    except (json.JSONDecodeError, TypeError):
                        pass  # Falls bereits Python-Objekt oder Parsing fehlschlägt
        result.append(test_dict)
    
    return result

async def get_test_results(model_id: int) -> List[Dict[str, Any]]:
    """Holt alle Test-Ergebnisse für ein Modell"""
    pool = await get_pool()
    rows = await pool.fetch(
        "SELECT * FROM ml_test_results WHERE model_id = $1 ORDER BY created_at DESC",
        model_id
    )
    # Konvertiere JSONB-Felder (refactored: nutze Helper-Funktion)
    result = []
    jsonb_fields = ['feature_importance', 'confusion_matrix']
    for row in rows:
        test_dict = dict(row)
        test_dict = convert_jsonb_fields(test_dict, jsonb_fields, direction="from")
        result.append(test_dict)
    return result

# ============================================================
# ml_comparisons - CRUD Operationen
# ============================================================

async def create_comparison(
    model_a_id: int,
    model_b_id: int,
    test_start: datetime,
    test_end: datetime,
    num_samples: Optional[int] = None,
    a_accuracy: Optional[float] = None,
    a_f1: Optional[float] = None,
    a_precision: Optional[float] = None,
    a_recall: Optional[float] = None,
    b_accuracy: Optional[float] = None,
    b_f1: Optional[float] = None,
    b_precision: Optional[float] = None,
    b_recall: Optional[float] = None,
    winner_id: Optional[int] = None,
    # NEU: Zusätzliche Metriken für Modell A
    a_mcc: Optional[float] = None,
    a_fpr: Optional[float] = None,
    a_fnr: Optional[float] = None,
    a_simulated_profit_pct: Optional[float] = None,
    a_confusion_matrix: Optional[Dict[str, int]] = None,
    a_train_accuracy: Optional[float] = None,
    a_train_f1: Optional[float] = None,
    a_accuracy_degradation: Optional[float] = None,
    a_f1_degradation: Optional[float] = None,
    a_is_overfitted: Optional[bool] = None,
    a_test_duration_days: Optional[float] = None,
    # NEU: Zusätzliche Metriken für Modell B
    b_mcc: Optional[float] = None,
    b_fpr: Optional[float] = None,
    b_fnr: Optional[float] = None,
    b_simulated_profit_pct: Optional[float] = None,
    b_confusion_matrix: Optional[Dict[str, int]] = None,
    b_train_accuracy: Optional[float] = None,
    b_train_f1: Optional[float] = None,
    b_accuracy_degradation: Optional[float] = None,
    b_f1_degradation: Optional[float] = None,
    b_is_overfitted: Optional[bool] = None,
    b_test_duration_days: Optional[float] = None
) -> int:
    """Erstellt einen Modell-Vergleich mit allen Metriken"""
    import json
    pool = await get_pool()
    
    # Konvertiere Confusion Matrix zu JSONB (refactored: nutze Helper-Funktion)
    a_confusion_matrix_jsonb = to_jsonb(a_confusion_matrix)
    b_confusion_matrix_jsonb = to_jsonb(b_confusion_matrix)
    
    try:
        comparison_id = await pool.fetchval(
            """
            INSERT INTO ml_comparisons (
                model_a_id, model_b_id, test_start, test_end, num_samples,
                a_accuracy, a_f1, a_precision, a_recall,
                b_accuracy, b_f1, b_precision, b_recall,
                winner_id,
                a_mcc, a_fpr, a_fnr, a_simulated_profit_pct, a_confusion_matrix,
                a_train_accuracy, a_train_f1, a_accuracy_degradation, a_f1_degradation,
                a_is_overfitted, a_test_duration_days,
                b_mcc, b_fpr, b_fnr, b_simulated_profit_pct, b_confusion_matrix,
                b_train_accuracy, b_train_f1, b_accuracy_degradation, b_f1_degradation,
                b_is_overfitted, b_test_duration_days
            ) VALUES (
                $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14,
                $15, $16, $17, $18, $19::jsonb,
                $20, $21, $22, $23, $24, $25,
                $26, $27, $28, $29, $30::jsonb,
                $31, $32, $33, $34, $35, $36
            ) RETURNING id
            """,
            model_a_id, model_b_id, test_start, test_end, num_samples,
            a_accuracy, a_f1, a_precision, a_recall,
            b_accuracy, b_f1, b_precision, b_recall,
            winner_id,
            a_mcc, a_fpr, a_fnr, a_simulated_profit_pct, a_confusion_matrix_jsonb,
            a_train_accuracy, a_train_f1, a_accuracy_degradation, a_f1_degradation,
            a_is_overfitted, a_test_duration_days,
            b_mcc, b_fpr, b_fnr, b_simulated_profit_pct, b_confusion_matrix_jsonb,
            b_train_accuracy, b_train_f1, b_accuracy_degradation, b_f1_degradation,
            b_is_overfitted, b_test_duration_days
        )
    except Exception as e:
        error_str = str(e).lower()
        if any(col in error_str for col in ['a_mcc', 'b_mcc', 'a_fpr', 'b_fpr', 'a_confusion_matrix', 'b_confusion_matrix']):
            # Fallback: Alte Spaltenstruktur (ohne neue Metriken)
            logger.warning(f"⚠️ Neue Vergleichs-Metriken-Spalten nicht gefunden - verwende Fallback (ohne zusätzliche Metriken)")
            comparison_id = await pool.fetchval(
                """
                INSERT INTO ml_comparisons (
                    model_a_id, model_b_id, test_start, test_end, num_samples,
                    a_accuracy, a_f1, a_precision, a_recall,
                    b_accuracy, b_f1, b_precision, b_recall,
                    winner_id
                ) VALUES (
                    $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14
                ) RETURNING id
                """,
                model_a_id, model_b_id, test_start, test_end, num_samples,
                a_accuracy, a_f1, a_precision, a_recall,
                b_accuracy, b_f1, b_precision, b_recall,
                winner_id
            )
        else:
            raise
    
    logger.info(f"✅ Vergleich erstellt: ID {comparison_id} (Modell {model_a_id} vs {model_b_id})")
    return comparison_id

async def get_comparison(comparison_id: int) -> Optional[Dict[str, Any]]:
    """Holt einen Vergleich und konvertiert JSONB-Felder"""
    pool = await get_pool()
    row = await pool.fetchrow(
        "SELECT * FROM ml_comparisons WHERE id = $1",
        comparison_id
    )
    if not row:
        return None
    
    comparison_dict = dict(row)
    # Konvertiere JSONB-Felder von Strings zu Python-Objekten (refactored: nutze Helper-Funktion)
    jsonb_fields = ['a_confusion_matrix', 'b_confusion_matrix']
    comparison_dict = convert_jsonb_fields(comparison_dict, jsonb_fields, direction="from")
    
    return comparison_dict

async def list_comparisons(
    limit: int = 100,
    offset: int = 0
) -> List[Dict[str, Any]]:
    """Listet alle Vergleichs-Ergebnisse"""
    pool = await get_pool()
    
    rows = await pool.fetch(
        """
        SELECT * FROM ml_comparisons 
        ORDER BY created_at DESC
        LIMIT $1 OFFSET $2
        """,
        limit, offset
    )
    
    # Konvertiere JSONB-Felder von Strings zu Python-Objekten (refactored: nutze Helper-Funktion)
    result = []
    jsonb_fields = ['a_confusion_matrix', 'b_confusion_matrix']
    for row in rows:
        comparison_dict = dict(row)
        comparison_dict = convert_jsonb_fields(comparison_dict, jsonb_fields, direction="from")
        result.append(comparison_dict)
    
    return result

async def delete_comparison(comparison_id: int) -> bool:
    """Löscht einen Vergleich"""
    pool = await get_pool()
    deleted = await pool.execute(
        "DELETE FROM ml_comparisons WHERE id = $1",
        comparison_id
    )
    logger.info(f"✅ Vergleich gelöscht: ID {comparison_id}")
    return deleted == "DELETE 1"

async def delete_test_result(test_id: int) -> bool:
    """Löscht ein Test-Ergebnis"""
    pool = await get_pool()
    deleted = await pool.execute(
        "DELETE FROM ml_test_results WHERE id = $1",
        test_id
    )
    logger.info(f"✅ Test-Ergebnis gelöscht: ID {test_id}")
    return deleted == "DELETE 1"

# ============================================================
# ml_jobs - CRUD Operationen
# ============================================================

async def create_job(
    job_type: str,
    priority: int = 5,
    train_model_type: Optional[str] = None,
    train_target_var: Optional[str] = None,
    train_operator: Optional[str] = None,
    train_value: Optional[float] = None,
    train_start: Optional[datetime] = None,
    train_end: Optional[datetime] = None,
    train_features: Optional[List[str]] = None,
    train_phases: Optional[List[int]] = None,
    train_params: Optional[Dict[str, Any]] = None,
    test_model_id: Optional[int] = None,
    test_start: Optional[datetime] = None,
    test_end: Optional[datetime] = None,
    compare_model_a_id: Optional[int] = None,
    compare_model_b_id: Optional[int] = None,
    compare_start: Optional[datetime] = None,
    compare_end: Optional[datetime] = None,
    progress_msg: Optional[str] = None,
    # Zeitbasierte Vorhersage-Parameter für Training
    train_future_minutes: Optional[int] = None,
    train_price_change_percent: Optional[float] = None,
    train_target_direction: Optional[str] = None
) -> int:
    """Erstellt einen neuen Job"""
    pool = await get_pool()
    
    # Konvertiere Listen/Dicts zu JSONB-kompatiblen Werten (refactored: nutze Helper-Funktion)
    train_features_jsonb = to_jsonb(train_features)
    train_phases_jsonb = to_jsonb(train_phases)
    train_params_jsonb = to_jsonb(train_params)
    
    job_id = await pool.fetchval(
        """
        INSERT INTO ml_jobs (
            job_type, priority,
            train_model_type, train_target_var, train_operator, train_value,
            train_start, train_end, train_features, train_phases, train_params,
            test_model_id, test_start, test_end,
            compare_model_a_id, compare_model_b_id, compare_start, compare_end,
            progress_msg,
            train_future_minutes, train_price_change_percent, train_target_direction
        ) VALUES (
            $1, $2, $3, $4, $5, $6, $7, $8, $9::jsonb, $10::jsonb, $11::jsonb, $12, $13, $14, $15, $16, $17, $18, $19, $20, $21, $22
        ) RETURNING id
        """,
        job_type, priority,
        train_model_type, train_target_var, train_operator, train_value,
        train_start, train_end, train_features_jsonb, train_phases_jsonb, train_params_jsonb,
        test_model_id, test_start, test_end,
        compare_model_a_id, compare_model_b_id, compare_start, compare_end,
        progress_msg,
        train_future_minutes, train_price_change_percent, train_target_direction
    )
    logger.info(f"✅ Job erstellt: {job_type} (ID: {job_id})")
    return job_id

async def get_job(job_id: int) -> Optional[Dict[str, Any]]:
    """Holt einen Job"""
    pool = await get_pool()
    row = await pool.fetchrow(
        "SELECT * FROM ml_jobs WHERE id = $1",
        job_id
    )
    if not row:
        return None
    # Konvertiere JSONB-Felder von Strings zu Python-Objekten
    job_dict = dict(row)
    # Konvertiere JSONB-Felder von Strings zu Python-Objekten (refactored: nutze Helper-Funktion)
    jsonb_fields = ['train_features', 'train_phases', 'train_params']
    job_dict = convert_jsonb_fields(job_dict, jsonb_fields, direction="from")
    return job_dict

async def update_job_status(
    job_id: int,
    status: str,
    progress: Optional[float] = None,
    error_msg: Optional[str] = None,
    result_model_id: Optional[int] = None,
    result_test_id: Optional[int] = None,
    result_comparison_id: Optional[int] = None,
    progress_msg: Optional[str] = None
) -> bool:
    """Aktualisiert Job-Status"""
    pool = await get_pool()
    
    updates = ["status = $1"]
    values = [status]
    param_num = 2
    
    if progress is not None:
        updates.append(f"progress = ${param_num}")
        values.append(progress)
        param_num += 1
    if error_msg is not None:
        updates.append(f"error_msg = ${param_num}")
        values.append(error_msg)
        param_num += 1
    if result_model_id is not None:
        updates.append(f"result_model_id = ${param_num}")
        values.append(result_model_id)
        param_num += 1
    if result_test_id is not None:
        updates.append(f"result_test_id = ${param_num}")
        values.append(result_test_id)
        param_num += 1
    if result_comparison_id is not None:
        updates.append(f"result_comparison_id = ${param_num}")
        values.append(result_comparison_id)
        param_num += 1
    if progress_msg is not None:
        updates.append(f"progress_msg = ${param_num}")
        values.append(progress_msg)
        param_num += 1
    
    # Timestamps
    if status == "RUNNING":
        updates.append("started_at = NOW()")
    elif status in ("COMPLETED", "FAILED", "CANCELLED"):
        updates.append("completed_at = NOW()")
    
    values.append(job_id)
    query = f"UPDATE ml_jobs SET {', '.join(updates)} WHERE id = ${param_num}"
    await pool.execute(query, *values)
    logger.info(f"✅ Job aktualisiert: ID {job_id} → {status}")
    return True

async def list_jobs(
    status: Optional[str] = None,
    job_type: Optional[str] = None,
    limit: int = 100,
    offset: int = 0
) -> List[Dict[str, Any]]:
    """Listet Jobs mit optionalen Filtern"""
    pool = await get_pool()
    
    conditions = []
    params = []
    param_num = 1
    
    if status:
        conditions.append(f"status = ${param_num}")
        params.append(status)
        param_num += 1
    
    if job_type:
        conditions.append(f"job_type = ${param_num}")
        params.append(job_type)
        param_num += 1
    
    where_clause = " AND ".join(conditions) if conditions else "1=1"
    params.extend([limit, offset])
    
    rows = await pool.fetch(
        f"""
        SELECT * FROM ml_jobs 
        WHERE {where_clause}
        ORDER BY created_at DESC
        LIMIT ${param_num} OFFSET ${param_num + 1}
        """,
        *params
    )
    # Konvertiere JSONB-Felder von Strings zu Python-Objekten
    import json
    result = []
    for row in rows:
        job_dict = dict(row)
        # JSONB-Felder konvertieren
        jsonb_fields = ['train_features', 'train_phases', 'train_params']
        for field in jsonb_fields:
            if field in job_dict and job_dict[field] is not None:
                if isinstance(job_dict[field], str):
                    try:
                        job_dict[field] = json.loads(job_dict[field])
                    except (json.JSONDecodeError, TypeError):
                        pass  # Falls bereits Python-Objekt oder Parsing fehlschlägt
        result.append(job_dict)
    return result

async def get_next_pending_job() -> Optional[Dict[str, Any]]:
    """Holt nächsten PENDING Job (ORDER BY priority, created_at)"""
    pool = await get_pool()
    row = await pool.fetchrow(
        """
        SELECT * FROM ml_jobs 
        WHERE status = 'PENDING'
        ORDER BY priority DESC, created_at ASC
        LIMIT 1
        FOR UPDATE SKIP LOCKED
        """
    )
    return dict(row) if row else None

# ============================================================
# ref_model_types - Helper Funktionen
# ============================================================

async def get_model_type_defaults(model_type: str) -> Dict[str, Any]:
    """Lade Default-Parameter für Modell-Typ aus ref_model_types"""
    import json
    pool = await get_pool()
    row = await pool.fetchrow(
        "SELECT default_params FROM ref_model_types WHERE name = $1",
        model_type
    )
    if row and row["default_params"]:
        # Refactored: nutze Helper-Funktion
        params = from_jsonb(row["default_params"])
        return params if params is not None else {}
    return {}

async def get_coin_phases() -> List[Dict[str, Any]]:
    """
    Lade alle Coin-Phasen aus ref_coin_phases mit interval_seconds
    
    Returns:
        Liste von Dicts mit: id, name, interval_seconds, max_age_minutes
    """
    pool = await get_pool()
    try:
        rows = await pool.fetch(
            "SELECT id, name, interval_seconds, max_age_minutes FROM ref_coin_phases ORDER BY id ASC"
        )
        phases = []
        for row in rows:
            phases.append({
                "id": row["id"],
                "name": row["name"],
                "interval_seconds": row["interval_seconds"],
                "max_age_minutes": row["max_age_minutes"]
            })
        logger.info(f"✅ {len(phases)} Phasen geladen")
        return phases
    except Exception as e:
        logger.error(f"❌ Fehler beim Laden der Phasen: {e}")
        # Fallback: Leere Liste oder Standard-Phasen
        return []

async def get_phase_intervals() -> Dict[int, int]:
    """
    Lade interval_seconds pro Phase als Dict (für schnellen Zugriff)
    
    Returns:
        Dict: {phase_id: interval_seconds}
    """
    pool = await get_pool()
    try:
        rows = await pool.fetch(
            "SELECT id, interval_seconds FROM ref_coin_phases"
        )
        intervals = {row["id"]: row["interval_seconds"] for row in rows}
        logger.debug(f"✅ {len(intervals)} Phase-Intervalle geladen")
        return intervals
    except Exception as e:
        logger.error(f"❌ Fehler beim Laden der Phase-Intervalle: {e}")
        return {}

