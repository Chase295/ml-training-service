"""
Job Manager für ML Training Service
Verarbeitet Jobs aus der ml_jobs Tabelle (TRAIN, TEST, COMPARE)
"""
import asyncio
import logging
from typing import Dict, Any, Optional
from datetime import datetime, timezone
from app.database.models import (
    get_next_pending_job, update_job_status, get_job,
    create_model, create_test_result, create_comparison
)
from app.training.engine import train_model
from app.training.model_loader import test_model
from app.utils.config import JOB_POLL_INTERVAL, MAX_CONCURRENT_JOBS, MODEL_STORAGE_PATH
from app.utils.metrics import (
    ml_jobs_total, increment_job_counter, update_active_jobs,
    update_job_metrics, update_job_feature_metrics
)

logger = logging.getLogger(__name__)

async def process_job(job_id: int) -> None:
    """
    Verarbeitet einen einzelnen Job
    
    ⚠️ WICHTIG: CPU-bound Training läuft in run_in_executor!
    
    Args:
        job_id: ID des Jobs aus ml_jobs
    """
    job = await get_job(job_id)
    if not job:
        logger.error(f"❌ Job {job_id} nicht gefunden!")
        return
    
    job_type = job['job_type']
    logger.info(f"🚀 Starte Job {job_id}: {job_type}")
    
    # Metriken aktualisieren
    increment_job_counter(job_type=job_type, status="running")
    update_active_jobs(1)
    
    try:
        # Status auf RUNNING setzen
        await update_job_status(job_id, status="RUNNING", progress=0.0)
        
        if job_type == "TRAIN":
            await process_train_job(job)
        elif job_type == "TEST":
            await process_test_job(job)
        elif job_type == "COMPARE":
            await process_compare_job(job)
        else:
            raise ValueError(f"Unbekannter Job-Typ: {job_type}")
        
        # Job erfolgreich abgeschlossen
        await update_job_status(job_id, status="COMPLETED", progress=1.0)
        increment_job_counter(job_type=job_type, status="completed")
        logger.info(f"✅ Job {job_id} erfolgreich abgeschlossen")
        
    except Exception as e:
        # Job fehlgeschlagen
        error_msg = str(e)
        logger.error(f"❌ Job {job_id} fehlgeschlagen: {error_msg}", exc_info=True)
        
        # Update Metriken: Job fehlgeschlagen
        model_type = job.get('train_model_type') or job.get('test_model_id') or 'unknown'
        update_job_metrics(
            job_id=job_id,
            job_type=job_type,
            model_type=model_type,
            status='FAILED',
            progress=0.0,
            duration_seconds=None
        )
        
        await update_job_status(
            job_id,
            status="FAILED",
            progress=0.0,
            error_msg=error_msg
        )
        increment_job_counter(job_type=job_type, status="failed")
    finally:
        update_active_jobs(-1)

async def process_train_job(job: Dict[str, Any]) -> None:
    """
    Verarbeitet TRAIN Job: Erstellt neues Modell
    
    ⚠️ KRITISCH: Training läuft in run_in_executor (CPU-bound)!
    """
    job_id = job['id']
    logger.info(f"🎯 Verarbeite TRAIN Job {job_id}")
    logger.info(f"📋 Job-Details: Typ={job.get('train_model_type')}, Features={len(job.get('train_features', []))}, Phasen={job.get('train_phases')}")
    
    # 1. ⚠️ KRITISCH: Hole Modell-Name aus progress_msg BEVOR es überschrieben wird!
    # Der Name wurde beim Job-Erstellen in progress_msg gespeichert
    model_name = job.get('progress_msg') or f"Model_{job_id}"
    logger.info(f"📝 Modell-Name: {model_name}")
    
    # ⚠️ WICHTIG: Speichere den ursprünglichen Namen, falls progress_msg später überschrieben wird
    original_model_name = model_name
    
    # Update Metriken: Job gestartet
    model_type = job.get('train_model_type', 'unknown')
    features = job.get('train_features', [])
    phases = job.get('train_phases', [])
    update_job_metrics(
        job_id=job_id,
        job_type='TRAIN',
        model_type=model_type,
        status='RUNNING',
        progress=0.1,
        duration_seconds=0
    )
    update_job_feature_metrics(
        job_id=job_id,
        job_type='TRAIN',
        features_count=len(features) if isinstance(features, list) else 0,
        phases_count=len(phases) if isinstance(phases, list) else 0
    )
    
    # 2. Hole Job-Parameter (features und phases bereits oben geholt)
    target_var = job['train_target_var']  # Kann None sein wenn zeitbasierte Vorhersage aktiviert
    target_operator = job['train_operator']  # Kann None sein wenn zeitbasierte Vorhersage aktiviert
    target_value = float(job['train_value']) if job['train_value'] is not None else None  # Kann None sein
    train_start = job['train_start']
    train_end = job['train_end']
    params = job['train_params']  # JSONB → Python Dict oder None
    
    # NEU: Extrahiere zeitbasierte Parameter aus train_params
    use_time_based = False
    future_minutes = None
    min_percent_change = None
    direction = "up"
    
    if params and isinstance(params, dict) and "_time_based" in params:
        time_based_config = params.get("_time_based", {})
        use_time_based = time_based_config.get("enabled", False)
        future_minutes = time_based_config.get("future_minutes")
        min_percent_change = time_based_config.get("min_percent_change")
        direction = time_based_config.get("direction", "up")
        
        # ⚠️ WICHTIG: Behalte _time_based in params, damit es beim Testen verfügbar ist!
        # params wird so an create_model übergeben und in der DB gespeichert
        
        logger.info(f"⏰ Zeitbasierte Vorhersage aktiviert: {future_minutes} Min, {min_percent_change}%, {direction}")
    
    logger.info(f"📋 Training-Parameter: {model_type}, Features: {len(features) if features else 0}")
    
    # 3. Update Progress (10%)
    await update_job_status(job_id, status="RUNNING", progress=0.1, progress_msg="Lade Trainingsdaten...", update_metrics=True)
    logger.info(f"📊 Job {job_id}: Status auf RUNNING gesetzt, Progress: 10%")
    
    # 4. ⚠️ KRITISCH: Training in run_in_executor ausführen (CPU-bound!)
    # ⚠️ WICHTIG: Während des Trainings können wir keine Progress-Updates machen,
    # da es in einem separaten Thread läuft. Progress wird nach dem Training aktualisiert.
    # train_model() nutzt bereits intern run_in_executor für model.fit(),
    # aber sicherheitshalber auch hier nutzen
    logger.info(f"🔄 Starte Training (blockiert Event Loop nicht)...")
    
    # 5. Update Progress (20% - Training startet)
    await update_job_status(job_id, status="RUNNING", progress=0.2, progress_msg="Training startet...", update_metrics=True)
    logger.info(f"📊 Job {job_id}: Training startet (Progress: 20%)")
    
    # 6. Führe Training aus (async, nutzt intern run_in_executor)
    logger.info(f"🔄 Job {job_id}: Starte Training... (kann mehrere Minuten dauern)")
    training_result = await train_model(
        model_type=model_type,
        features=features,
        target_var=target_var,
        target_operator=target_operator,
        target_value=target_value,
        train_start=train_start,
        train_end=train_end,
        phases=phases,
        params=params,
        model_storage_path=MODEL_STORAGE_PATH,
        # NEU: Zeitbasierte Parameter
        use_time_based=use_time_based,
        future_minutes=future_minutes,
        min_percent_change=min_percent_change,
        direction=direction
    )
    
    logger.info(f"✅ Job {job_id}: Training abgeschlossen - Accuracy={training_result['accuracy']:.4f}, F1={training_result['f1']:.4f}")
    
    # 7. Update Progress (60% - Training abgeschlossen)
    await update_job_status(job_id, status="RUNNING", progress=0.6, progress_msg="Training abgeschlossen, speichere Modell...", update_metrics=True)
    logger.info(f"📊 Job {job_id}: Training abgeschlossen (Progress: 60%)")
    
    # 8. Update Progress (80% - Modell wird in DB gespeichert)
    await update_job_status(job_id, status="RUNNING", progress=0.8, progress_msg="Speichere Modell in DB...", update_metrics=True)
    logger.info(f"📊 Job {job_id}: Speichere Modell in DB... (Progress: 80%)")
    
    # 6.5. Hole erweiterte Features aus Training-Result (inkl. engineered features)
    # ⚠️ WICHTIG: Wenn Feature-Engineering aktiviert war, enthält training_result['features'] die erweiterte Liste!
    final_features = training_result.get('features', features)  # Fallback auf ursprüngliche Features
    logger.info(f"📊 Features für Modell-Speicherung: {len(final_features)} (ursprünglich: {len(features)})")
    
    # 7. Erstelle Eintrag in ml_models Tabelle
    # ⚠️ WICHTIG: UTC-Zeitstempel verwenden!
    train_start_dt = train_start
    train_end_dt = train_end
    if isinstance(train_start_dt, str):
        train_start_dt = datetime.fromisoformat(train_start_dt.replace('Z', '+00:00'))
    if isinstance(train_end_dt, str):
        train_end_dt = datetime.fromisoformat(train_end_dt.replace('Z', '+00:00'))
    
    # Stelle sicher, dass Zeitzone UTC ist
    if train_start_dt.tzinfo is None:
        train_start_dt = train_start_dt.replace(tzinfo=timezone.utc)
    if train_end_dt.tzinfo is None:
        train_end_dt = train_end_dt.replace(tzinfo=timezone.utc)
    
    # Hole zeitbasierte Parameter aus Job (falls vorhanden)
    train_future_minutes = job.get('train_future_minutes')
    train_price_change_percent = job.get('train_price_change_percent')
    train_target_direction = job.get('train_target_direction')
    
    model_id = await create_model(
        name=original_model_name,  # ⚠️ Verwende den ursprünglichen Namen!
        model_type=model_type,
        target_variable=target_var,
        train_start=train_start_dt,
        train_end=train_end_dt,
        target_operator=target_operator,
        target_value=target_value,
        features=final_features,  # ✅ Erweiterte Features (inkl. engineered features)
        phases=phases,
        params=params,
        training_accuracy=training_result['accuracy'],
        training_f1=training_result['f1'],
        training_precision=training_result['precision'],
        training_recall=training_result['recall'],
        feature_importance=training_result['feature_importance'],
        model_file_path=training_result['model_path'],
        status="READY",
        cv_scores=training_result.get('cv_scores'),  # NEU: CV-Ergebnisse
        cv_overfitting_gap=training_result.get('cv_overfitting_gap'),  # NEU: Overfitting-Gap
        roc_auc=training_result.get('roc_auc'),  # NEU: ROC-AUC
        mcc=training_result.get('mcc'),  # NEU: MCC
        fpr=training_result.get('fpr'),  # NEU: False Positive Rate
        fnr=training_result.get('fnr'),  # NEU: False Negative Rate
        confusion_matrix=training_result.get('confusion_matrix'),  # NEU: Confusion Matrix
        simulated_profit_pct=training_result.get('simulated_profit_pct'),  # NEU: Simulierter Profit
        # Zeitbasierte Vorhersage-Parameter
        future_minutes=train_future_minutes,
        price_change_percent=train_price_change_percent,
        target_direction=train_target_direction
    )
    
    logger.info(f"✅ Job {job_id}: Modell erstellt - ID {model_id}, Name: {original_model_name}")
    
    # 9. Update Progress (90% - Modell gespeichert)
    await update_job_status(job_id, status="RUNNING", progress=0.9, progress_msg="Modell gespeichert, finalisiere...", update_metrics=True)
    logger.info(f"📊 Job {job_id}: Modell gespeichert (Progress: 90%)")
    
    # 10. Setze result_model_id im Job (100% - COMPLETED)
    await update_job_status(
        job_id,
        status="COMPLETED",
        progress=1.0,
        result_model_id=model_id,
        progress_msg=f"Modell {original_model_name} erfolgreich erstellt",
        update_metrics=True
    )
    
    # Erhöhe Job-Counter
    increment_job_counter('TRAIN', 'COMPLETED')
    
    logger.info(f"🎉 Job {job_id} erfolgreich abgeschlossen: Modell {model_id} erstellt (Progress: 100%)")

async def process_test_job(job: Dict[str, Any]) -> None:
    """
    Verarbeitet TEST Job: Testet Modell auf neuen Daten
    """
    logger.info(f"🧪 Verarbeite TEST Job {job['id']}")
    
    # 1. Hole Job-Parameter
    model_id = job['test_model_id']
    test_start = job['test_start']
    test_end = job['test_end']
    
    logger.info(f"📋 Test-Parameter: Modell {model_id}, Zeitraum: {test_start} bis {test_end}")
    
    # 2. Update Progress
    await update_job_status(job['id'], status="RUNNING", progress=0.2, progress_msg="Lade Test-Daten...")
    
    # 3. Führe Test aus
    test_result = await test_model(
        model_id=model_id,
        test_start=test_start,
        test_end=test_end,
        model_storage_path=MODEL_STORAGE_PATH
    )
    
    logger.info(f"✅ Test abgeschlossen: Accuracy={test_result['accuracy']:.4f}")
    
    # 4. Update Progress
    await update_job_status(job['id'], status="RUNNING", progress=0.8, progress_msg="Speichere Test-Ergebnis...")
    
    # 5. Erstelle Eintrag in ml_test_results Tabelle
    # ⚠️ WICHTIG: UTC-Zeitstempel verwenden!
    test_start_dt = test_start
    test_end_dt = test_end
    if isinstance(test_start_dt, str):
        test_start_dt = datetime.fromisoformat(test_start_dt.replace('Z', '+00:00'))
    if isinstance(test_end_dt, str):
        test_end_dt = datetime.fromisoformat(test_end_dt.replace('Z', '+00:00'))
    
    # Stelle sicher, dass Zeitzone UTC ist
    if test_start_dt.tzinfo is None:
        test_start_dt = test_start_dt.replace(tzinfo=timezone.utc)
    if test_end_dt.tzinfo is None:
        test_end_dt = test_end_dt.replace(tzinfo=timezone.utc)
    
    test_id = await create_test_result(
        model_id=model_id,
        test_start=test_start_dt,
        test_end=test_end_dt,
        accuracy=test_result['accuracy'],
        f1_score=test_result['f1_score'],
        precision_score=test_result['precision_score'],
        recall=test_result['recall'],
        roc_auc=test_result.get('roc_auc'),
        # Zusätzliche Metriken (Phase 9)
        mcc=test_result.get('mcc'),
        fpr=test_result.get('fpr'),
        fnr=test_result.get('fnr'),
        simulated_profit_pct=test_result.get('simulated_profit_pct'),
        confusion_matrix=test_result.get('confusion_matrix'),
        # Legacy: Confusion Matrix als einzelne Felder
        tp=test_result['tp'],
        tn=test_result['tn'],
        fp=test_result['fp'],
        fn=test_result['fn'],
        num_samples=test_result['num_samples'],
        num_positive=test_result['num_positive'],
        num_negative=test_result['num_negative'],
        has_overlap=test_result['has_overlap'],
        overlap_note=test_result.get('overlap_note'),
        # Train vs. Test Vergleich (Phase 2)
        train_accuracy=test_result.get('train_accuracy'),
        train_f1=test_result.get('train_f1'),
        train_precision=test_result.get('train_precision'),
        train_recall=test_result.get('train_recall'),
        accuracy_degradation=test_result.get('accuracy_degradation'),
        f1_degradation=test_result.get('f1_degradation'),
        is_overfitted=test_result.get('is_overfitted'),
        # Test-Zeitraum Info (Phase 2)
        test_duration_days=test_result.get('test_duration_days')
    )
    
    logger.info(f"✅ Test-Ergebnis erstellt: ID {test_id}")
    
    # 6. Setze result_test_id im Job
    await update_job_status(
        job['id'],
        status="COMPLETED",
        progress=1.0,
        result_test_id=test_id,
        progress_msg=f"Test erfolgreich abgeschlossen"
    )
    
    logger.info(f"🎉 TEST Job {job['id']} erfolgreich: Test {test_id} erstellt")

async def process_compare_job(job: Dict[str, Any]) -> None:
    """
    Verarbeitet COMPARE Job: Vergleicht zwei Modelle
    """
    logger.info(f"⚖️ Verarbeite COMPARE Job {job['id']}")
    
    # 1. Hole Job-Parameter
    model_a_id = job['compare_model_a_id']
    model_b_id = job['compare_model_b_id']
    test_start = job['compare_start']
    test_end = job['compare_end']
    
    logger.info(f"📋 Vergleich: Modell {model_a_id} vs {model_b_id}")
    
    # 2. Update Progress
    await update_job_status(job['id'], status="RUNNING", progress=0.1, progress_msg="Teste Modell A...")
    
    # 3. Teste Modell A
    result_a = await test_model(
        model_id=model_a_id,
        test_start=test_start,
        test_end=test_end,
        model_storage_path=MODEL_STORAGE_PATH
    )
    
    logger.info(f"✅ Modell A: Accuracy={result_a['accuracy']:.4f}, F1={result_a['f1_score']:.4f}")
    
    # 4. Update Progress
    await update_job_status(job['id'], status="RUNNING", progress=0.5, progress_msg="Teste Modell B...")
    
    # 5. Teste Modell B
    result_b = await test_model(
        model_id=model_b_id,
        test_start=test_start,
        test_end=test_end,
        model_storage_path=MODEL_STORAGE_PATH
    )
    
    logger.info(f"✅ Modell B: Accuracy={result_b['accuracy']:.4f}, F1={result_b['f1_score']:.4f}")
    
    # 6. Bestimme Gewinner (verbessert: berücksichtigt F1, MCC und Profit)
    winner_id = None
    winner_reason = []
    
    # Vergleiche F1-Score
    if result_a['f1_score'] > result_b['f1_score']:
        winner_id = model_a_id
        winner_reason.append(f"F1: {result_a['f1_score']:.4f} > {result_b['f1_score']:.4f}")
    elif result_b['f1_score'] > result_a['f1_score']:
        winner_id = model_b_id
        winner_reason.append(f"F1: {result_b['f1_score']:.4f} > {result_a['f1_score']:.4f}")
    else:
        # Bei Gleichstand: MCC als Tie-Breaker
        if result_a.get('mcc') and result_b.get('mcc'):
            if result_a['mcc'] > result_b['mcc']:
                winner_id = model_a_id
                winner_reason.append(f"MCC (Tie-Breaker): {result_a['mcc']:.4f} > {result_b['mcc']:.4f}")
            elif result_b['mcc'] > result_a['mcc']:
                winner_id = model_b_id
                winner_reason.append(f"MCC (Tie-Breaker): {result_b['mcc']:.4f} > {result_a['mcc']:.4f}")
        # Falls immer noch Gleichstand: Profit als Tie-Breaker
        if winner_id is None:
            if result_a.get('simulated_profit_pct') and result_b.get('simulated_profit_pct'):
                if result_a['simulated_profit_pct'] > result_b['simulated_profit_pct']:
                    winner_id = model_a_id
                    winner_reason.append(f"Profit (Tie-Breaker): {result_a['simulated_profit_pct']:.4f}% > {result_b['simulated_profit_pct']:.4f}%")
                elif result_b['simulated_profit_pct'] > result_a['simulated_profit_pct']:
                    winner_id = model_b_id
                    winner_reason.append(f"Profit (Tie-Breaker): {result_b['simulated_profit_pct']:.4f}% > {result_a['simulated_profit_pct']:.4f}%")
    
    logger.info(f"🏆 Gewinner: Modell {winner_id if winner_id else 'Unentschieden'} ({', '.join(winner_reason) if winner_reason else 'Gleichstand'})")
    
    # 7. Update Progress
    await update_job_status(job['id'], status="RUNNING", progress=0.8, progress_msg="Speichere Vergleich...")
    
    # 8. Erstelle Eintrag in ml_comparisons Tabelle
    # ⚠️ WICHTIG: UTC-Zeitstempel verwenden!
    test_start_dt = test_start
    test_end_dt = test_end
    if isinstance(test_start_dt, str):
        test_start_dt = datetime.fromisoformat(test_start_dt.replace('Z', '+00:00'))
    if isinstance(test_end_dt, str):
        test_end_dt = datetime.fromisoformat(test_end_dt.replace('Z', '+00:00'))
    
    # Stelle sicher, dass Zeitzone UTC ist
    if test_start_dt.tzinfo is None:
        test_start_dt = test_start_dt.replace(tzinfo=timezone.utc)
    if test_end_dt.tzinfo is None:
        test_end_dt = test_end_dt.replace(tzinfo=timezone.utc)
    
    comparison_id = await create_comparison(
        model_a_id=model_a_id,
        model_b_id=model_b_id,
        test_start=test_start_dt,
        test_end=test_end_dt,
        num_samples=result_a['num_samples'],  # Beide sollten gleich sein
        # Basis-Metriken Modell A
        a_accuracy=result_a['accuracy'],
        a_f1=result_a['f1_score'],
        a_precision=result_a['precision_score'],
        a_recall=result_a['recall'],
        # Basis-Metriken Modell B
        b_accuracy=result_b['accuracy'],
        b_f1=result_b['f1_score'],
        b_precision=result_b['precision_score'],
        b_recall=result_b['recall'],
        # Zusätzliche Metriken Modell A
        a_mcc=result_a.get('mcc'),
        a_fpr=result_a.get('fpr'),
        a_fnr=result_a.get('fnr'),
        a_simulated_profit_pct=result_a.get('simulated_profit_pct'),
        a_confusion_matrix=result_a.get('confusion_matrix'),
        a_train_accuracy=result_a.get('train_accuracy'),
        a_train_f1=result_a.get('train_f1'),
        a_accuracy_degradation=result_a.get('accuracy_degradation'),
        a_f1_degradation=result_a.get('f1_degradation'),
        a_is_overfitted=result_a.get('is_overfitted'),
        a_test_duration_days=result_a.get('test_duration_days'),
        # Zusätzliche Metriken Modell B
        b_mcc=result_b.get('mcc'),
        b_fpr=result_b.get('fpr'),
        b_fnr=result_b.get('fnr'),
        b_simulated_profit_pct=result_b.get('simulated_profit_pct'),
        b_confusion_matrix=result_b.get('confusion_matrix'),
        b_train_accuracy=result_b.get('train_accuracy'),
        b_train_f1=result_b.get('train_f1'),
        b_accuracy_degradation=result_b.get('accuracy_degradation'),
        b_f1_degradation=result_b.get('f1_degradation'),
        b_is_overfitted=result_b.get('is_overfitted'),
        b_test_duration_days=result_b.get('test_duration_days'),
        winner_id=winner_id
    )
    
    logger.info(f"✅ Vergleich erstellt: ID {comparison_id}")
    
    # 9. Setze result_comparison_id im Job
    await update_job_status(
        job['id'],
        status="COMPLETED",
        progress=1.0,
        result_comparison_id=comparison_id,
        progress_msg=f"Vergleich erfolgreich abgeschlossen"
    )
    
    logger.info(f"🎉 COMPARE Job {job['id']} erfolgreich: Vergleich {comparison_id} erstellt")

async def start_worker() -> None:
    """
    Worker-Loop: Prüft regelmäßig auf neue Jobs und verarbeitet sie
    
    ⚠️ WICHTIG: Läuft in Endlosschleife, prüft alle JOB_POLL_INTERVAL Sekunden
    """
    logger.info(f"🔄 Starte Job Worker (Poll-Interval: {JOB_POLL_INTERVAL}s, Max Concurrent: {MAX_CONCURRENT_JOBS})")
    
    active_tasks = set()  # Track laufende Jobs
    
    while True:
        try:
            # Prüfe ob noch Platz für neue Jobs
            if len(active_tasks) < MAX_CONCURRENT_JOBS:
                # Hole nächsten PENDING Job
                job = await get_next_pending_job()
                
                if job:
                    logger.info(f"📥 Neuer Job gefunden: {job['id']} ({job['job_type']})")
                    
                    # Starte Job asynchron (nicht blockierend!)
                    task = asyncio.create_task(process_job(job['id']))
                    active_tasks.add(task)
                    
                    # Entferne Task wenn fertig
                    def remove_task(t):
                        active_tasks.discard(t)
                    task.add_done_callback(remove_task)
                else:
                    # Kein Job gefunden, warte
                    await asyncio.sleep(JOB_POLL_INTERVAL)
            else:
                # Max Concurrent Jobs erreicht, warte
                logger.debug(f"⏳ Max Concurrent Jobs erreicht ({MAX_CONCURRENT_JOBS}), warte...")
                await asyncio.sleep(JOB_POLL_INTERVAL)
            
        except Exception as e:
            logger.error(f"⚠️ Worker Error: {e}")
            import traceback
            traceback.print_exc()
            await asyncio.sleep(JOB_POLL_INTERVAL)

