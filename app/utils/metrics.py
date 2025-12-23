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

