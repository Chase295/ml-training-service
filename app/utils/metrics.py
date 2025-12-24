"""
Prometheus Metrics und Health Status für ML Training Service
"""
import time
import logging
from typing import Dict, Any
from prometheus_client import Counter, Gauge, Histogram, generate_latest
from app.database.connection import get_pool, test_connection

logger = logging.getLogger(__name__)

# ============================================================
# Prometheus Metrics
# ============================================================

# Job Metrics
ml_jobs_total = Counter(
    'ml_jobs_total',
    'Total number of ML jobs',
    ['job_type', 'status']
)

ml_jobs_duration_seconds = Histogram(
    'ml_jobs_duration_seconds',
    'Duration of ML jobs in seconds',
    ['job_type'],
    buckets=[1, 5, 10, 30, 60, 120, 300, 600]
)

# Model Metrics
ml_models_total = Gauge(
    'ml_models_total',
    'Total number of ML models'
)

ml_training_accuracy = Gauge(
    'ml_training_accuracy',
    'Training accuracy of ML models',
    ['model_id']
)

ml_test_accuracy = Gauge(
    'ml_test_accuracy',
    'Test accuracy of ML models',
    ['model_id']
)

# Service Metrics
ml_service_uptime_seconds = Gauge(
    'ml_service_uptime_seconds',
    'Service uptime in seconds'
)

ml_db_connected = Gauge(
    'ml_db_connected',
    'Database connection status (1=connected, 0=disconnected)'
)

ml_active_jobs = Gauge(
    'ml_active_jobs',
    'Number of currently active jobs'
)

# Job Progress & Status Metrics (für Grafana)
ml_job_progress = Gauge(
    'ml_job_progress_percent',
    'Current progress of a job in percent (0-100)',
    ['job_id', 'job_type', 'model_type']
)

ml_job_duration_seconds = Gauge(
    'ml_job_duration_seconds',
    'Current duration of a running job in seconds',
    ['job_id', 'job_type', 'model_type']
)

ml_job_status = Gauge(
    'ml_job_status',
    'Job status (1=PENDING, 2=RUNNING, 3=COMPLETED, 4=FAILED, 5=CANCELLED)',
    ['job_id', 'job_type', 'model_type', 'status']
)

ml_job_features_count = Gauge(
    'ml_job_features_count',
    'Number of features used in a job',
    ['job_id', 'job_type']
)

ml_job_phases_count = Gauge(
    'ml_job_phases_count',
    'Number of phases used in a job',
    ['job_id', 'job_type']
)

# ============================================================
# Health Status
# ============================================================

health_status: Dict[str, Any] = {
    "db_connected": False,
    "last_error": None,
    "start_time": None,
    "total_jobs_processed": 0
}

def init_health_status():
    """Initialisiert Health Status beim Service-Start"""
    health_status["start_time"] = time.time()
    health_status["db_connected"] = False
    health_status["last_error"] = None
    health_status["total_jobs_processed"] = 0

async def get_health_status() -> Dict[str, Any]:
    """
    Prüft Health Status und gibt Status-Dict zurück
    Returns: {"status": "healthy"/"degraded", "db_connected": bool, ...}
    """
    try:
        # Prüfe DB-Verbindung
        db_connected = await test_connection()
        health_status["db_connected"] = db_connected
        ml_db_connected.set(1 if db_connected else 0)
        
        # Berechne Uptime
        if health_status["start_time"]:
            uptime = time.time() - health_status["start_time"]
            ml_service_uptime_seconds.set(uptime)
        else:
            uptime = 0
        
        # Bestimme Gesamt-Status
        if db_connected:
            status = "healthy"
        else:
            status = "degraded"
        
        return {
            "status": status,
            "db_connected": db_connected,
            "uptime_seconds": int(uptime),
            "start_time": health_status["start_time"],
            "total_jobs_processed": health_status["total_jobs_processed"],
            "last_error": health_status["last_error"]
        }
    except Exception as e:
        logger.error(f"❌ Fehler beim Health Check: {e}")
        health_status["last_error"] = str(e)
        health_status["db_connected"] = False
        ml_db_connected.set(0)
        return {
            "status": "degraded",
            "db_connected": False,
            "uptime_seconds": int(time.time() - health_status["start_time"]) if health_status["start_time"] else 0,
            "start_time": health_status["start_time"],
            "total_jobs_processed": health_status["total_jobs_processed"],
            "last_error": str(e)
        }

def generate_metrics() -> bytes:
    """
    Generiert Prometheus Metrics als String
    Returns: Metrics im Prometheus-Format
    """
    return generate_latest()

def update_model_count(count: int):
    """Aktualisiert Anzahl der Modelle"""
    ml_models_total.set(count)

def increment_job_counter(job_type: str, status: str):
    """Erhöht Job-Counter"""
    ml_jobs_total.labels(job_type=job_type, status=status).inc()

def record_job_duration(job_type: str, duration: float):
    """Zeichnet Job-Dauer auf"""
    ml_jobs_duration_seconds.labels(job_type=job_type).observe(duration)

def update_active_jobs(count: int):
    """Aktualisiert Anzahl aktiver Jobs"""
    ml_active_jobs.set(count)

def update_training_accuracy(model_id: int, accuracy: float):
    """Aktualisiert Training Accuracy für ein Modell"""
    ml_training_accuracy.labels(model_id=str(model_id)).set(accuracy)

def update_test_accuracy(model_id: int, accuracy: float):
    """Aktualisiert Test Accuracy für ein Modell"""
    ml_test_accuracy.labels(model_id=str(model_id)).set(accuracy)

def increment_jobs_processed():
    """Erhöht Counter für verarbeitete Jobs"""
    health_status["total_jobs_processed"] += 1

# ============================================================
# Job Progress & Status Update Functions
# ============================================================

def update_job_metrics(job_id: int, job_type: str, model_type: str, status: str, progress: float, duration_seconds: float = None):
    """
    Aktualisiert Prometheus-Metriken für einen Job
    
    Args:
        job_id: Job-ID
        job_type: TRAIN, TEST, COMPARE
        model_type: random_forest, xgboost, etc.
        status: PENDING, RUNNING, COMPLETED, FAILED, CANCELLED
        progress: Progress in Prozent (0-100)
        duration_seconds: Laufzeit in Sekunden (optional)
    """
    # Status-Mapping für Metrik
    status_map = {
        'PENDING': 1,
        'RUNNING': 2,
        'COMPLETED': 3,
        'FAILED': 4,
        'CANCELLED': 5
    }
    status_value = status_map.get(status, 0)
    
    # Job Progress
    ml_job_progress.labels(
        job_id=str(job_id),
        job_type=job_type,
        model_type=model_type or 'unknown'
    ).set(progress)
    
    # Job Status (setze alle auf 0, dann aktuelle auf 1)
    for s in ['PENDING', 'RUNNING', 'COMPLETED', 'FAILED', 'CANCELLED']:
        ml_job_status.labels(
            job_id=str(job_id),
            job_type=job_type,
            model_type=model_type or 'unknown',
            status=s
        ).set(1 if s == status else 0)
    
    # Job Duration (nur wenn RUNNING)
    if duration_seconds is not None and status == 'RUNNING':
        ml_job_duration_seconds.labels(
            job_id=str(job_id),
            job_type=job_type,
            model_type=model_type or 'unknown'
        ).set(duration_seconds)

def update_job_feature_metrics(job_id: int, job_type: str, features_count: int, phases_count: int = None):
    """
    Aktualisiert Feature- und Phase-Metriken für einen Job
    
    Args:
        job_id: Job-ID
        job_type: TRAIN, TEST, COMPARE
        features_count: Anzahl Features
        phases_count: Anzahl Phasen (optional)
    """
    ml_job_features_count.labels(
        job_id=str(job_id),
        job_type=job_type
    ).set(features_count)
    
    if phases_count is not None:
        ml_job_phases_count.labels(
            job_id=str(job_id),
            job_type=job_type
        ).set(phases_count)

async def update_all_job_metrics():
    """
    Aktualisiert Metriken für alle aktiven Jobs
    Wird regelmäßig aufgerufen (z.B. alle 5 Sekunden)
    """
    try:
        from app.database.models import list_jobs
        from datetime import datetime, timezone
        
        # Hole alle RUNNING und PENDING Jobs
        running_jobs = await list_jobs(status='RUNNING', limit=100)
        pending_jobs = await list_jobs(status='PENDING', limit=100)
        all_active_jobs = running_jobs + pending_jobs
        
        now = datetime.now(timezone.utc)
        
        for job in all_active_jobs:
            job_id = job.get('id')
            job_type = job.get('job_type', 'UNKNOWN')
            model_type = job.get('train_model_type') or job.get('test_model_id') or 'unknown'
            status = job.get('status', 'UNKNOWN')
            progress = job.get('progress', 0.0)
            
            # Berechne Duration
            started_at = job.get('started_at')
            duration_seconds = None
            if started_at:
                try:
                    if isinstance(started_at, str):
                        started_dt = datetime.fromisoformat(started_at.replace('Z', '+00:00'))
                    else:
                        started_dt = started_at
                    
                    if started_dt.tzinfo is None:
                        started_dt = started_dt.replace(tzinfo=timezone.utc)
                    
                    duration_seconds = (now - started_dt).total_seconds()
                except:
                    pass
            
            # ✅ NEU: Progress-Schätzung während des Trainings (wenn Progress bei 20% steht)
            # Wenn Job RUNNING ist und Progress zwischen 20% und 60%, schätze Progress basierend auf Dauer
            estimated_progress = progress
            if status == 'RUNNING' and job_type == 'TRAIN' and 0.2 <= progress < 0.6 and duration_seconds:
                # Schätze Progress basierend auf typischer Trainingsdauer
                # Annahme: Training dauert typischerweise 10-30 Minuten
                # Progress: 20% (Start) → 60% (Ende) = 40% Progress während Training
                # Schätze basierend auf Dauer: Je länger, desto höher der Progress
                
                # Typische Trainingsdauer: 10-30 Minuten (600-1800 Sekunden)
                # Wenn bereits 5 Minuten vergangen sind, schätze 30% Progress
                # Wenn bereits 15 Minuten vergangen sind, schätze 50% Progress
                
                if duration_seconds < 300:  # < 5 Minuten
                    estimated_progress = 0.2 + (duration_seconds / 300) * 0.1  # 20% → 30%
                elif duration_seconds < 900:  # 5-15 Minuten
                    estimated_progress = 0.3 + ((duration_seconds - 300) / 600) * 0.2  # 30% → 50%
                else:  # > 15 Minuten
                    estimated_progress = 0.5 + min(0.1, (duration_seconds - 900) / 1800)  # 50% → 60%
                
                # Begrenze auf max 60% (bis Training wirklich abgeschlossen ist)
                estimated_progress = min(estimated_progress, 0.6)
            
            # Update Metriken
            update_job_metrics(
                job_id=job_id,
                job_type=job_type,
                model_type=model_type,
                status=status,
                progress=estimated_progress * 100.0,  # Konvertiere zu Prozent
                duration_seconds=duration_seconds
            )
            
            # Update Feature/Phase Metriken (nur für TRAIN Jobs)
            if job_type == 'TRAIN':
                features = job.get('train_features', [])
                phases = job.get('train_phases', [])
                update_job_feature_metrics(
                    job_id=job_id,
                    job_type=job_type,
                    features_count=len(features) if isinstance(features, list) else 0,
                    phases_count=len(phases) if isinstance(phases, list) else 0
                )
        
        # Update active jobs count
        update_active_jobs(len(all_active_jobs))
        
    except Exception as e:
        logger.error(f"Fehler beim Aktualisieren der Job-Metriken: {e}", exc_info=True)

