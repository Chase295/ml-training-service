from fastapi import APIRouter, HTTPException
from app.database.models import get_all_jobs, get_job_by_id, create_job_record
from app.queue.job_manager import get_job_status
from typing import List, Dict, Any
from datetime import datetime

ui_router = APIRouter()

@ui_router.get("/jobs")
async def get_jobs():
    """Alle Jobs für UI"""
    try:
        jobs = await get_all_jobs()
        return {"jobs": jobs, "total": len(jobs)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Fehler beim Laden der Jobs: {str(e)}")

@ui_router.get("/jobs/{job_id}")
async def get_job(job_id: str):
    """Einzelner Job für UI"""
    try:
        job = await get_job_by_id(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job nicht gefunden")
        return job
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Fehler beim Laden des Jobs: {str(e)}")

@ui_router.post("/jobs")
async def create_job(job_config: Dict[str, Any]):
    """Neuen Job erstellen"""
    try:
        job_id = await create_job_record(job_config)
        return {"job_id": job_id, "status": "created", "message": "Job erfolgreich erstellt"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Fehler beim Erstellen des Jobs: {str(e)}")

@ui_router.get("/models")
async def get_models():
    """Alle Models für UI"""
    # TODO: Implementiere echte Model-Abfrage
    # Hier deine bestehende Model-Logik integrieren
    models = [
        {
            "id": "model_1",
            "name": "XGBoost SOL Model v1",
            "created_at": "2024-01-01T10:00:00Z",
            "status": "active",
            "metrics": {
                "accuracy": 0.85,
                "precision": 0.88,
                "recall": 0.82
            }
        }
    ]
    return {"models": models, "total": len(models)}

@ui_router.get("/models/compare")
async def compare_models(model_ids: List[str]):
    """Models vergleichen"""
    # TODO: Implementiere echte Vergleichslogik
    comparison = {
        "models": model_ids,
        "metrics": {
            "accuracy": [0.85, 0.82],
            "precision": [0.88, 0.79],
            "recall": [0.82, 0.85]
        }
    }
    return comparison
