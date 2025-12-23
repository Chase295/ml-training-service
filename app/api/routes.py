"""
FastAPI Routes für ML Training Service
"""
import os
import logging
from typing import List, Optional
from fastapi import APIRouter, HTTPException, Depends, Response, status
from fastapi.responses import FileResponse, JSONResponse
from fastapi.responses import FileResponse
from app.api.schemas import (
    TrainModelRequest, TestModelRequest, CompareModelsRequest, UpdateModelRequest,
    ModelResponse, TestResultResponse, ComparisonResponse,
    JobResponse, CreateJobResponse, HealthResponse
)
from app.database.models import (
    create_model, get_model, update_model, list_models, delete_model,
    create_test_result, get_test_result, get_test_results, list_all_test_results, delete_test_result,
    create_comparison, get_comparison, list_comparisons, delete_comparison,
    create_job, get_job, list_jobs,
    get_coin_phases
)
from app.database.connection import get_pool
from app.training.engine import train_model
from app.training.model_loader import test_model
from app.utils.metrics import get_health_status, generate_metrics
from app.utils.config import MODEL_STORAGE_PATH
from app.utils.exceptions import (
    ModelNotFoundError, InvalidModelParametersError, DatabaseError,
    JobNotFoundError, JobProcessingError, TrainingError, TestError,
    ComparisonError, DataError, ValidationError, MLTrainingError
)
from app.utils.logging_config import get_logger, log_with_context, get_request_id

logger = get_logger(__name__)

router = APIRouter(prefix="/api", tags=["ML Training Service"])

# ============================================================
# Models Endpoints
# ============================================================

@router.post("/models/create", response_model=CreateJobResponse, status_code=status.HTTP_201_CREATED)
async def create_model_job(request: TrainModelRequest):
    """
    Erstellt einen TRAIN-Job (Modell wird asynchron trainiert)
    
    ⚠️ WICHTIG: Modell wird NICHT sofort erstellt!
    Job wird in Queue eingereiht und später vom Worker verarbeitet.
    """
    try:
        # Erstelle Job in ml_jobs mit job_type: TRAIN
        # Modell-Name wird in progress_msg temporär gespeichert
        
        # NEU: Zeitbasierte Parameter in train_params speichern (falls aktiviert)
        final_params = request.params or {}
        if request.use_time_based_prediction:
            # Speichere zeitbasierte Parameter in train_params
            final_params = {
                **final_params,
                "_time_based": {
                    "enabled": True,
                    "future_minutes": request.future_minutes,
                    "min_percent_change": request.min_percent_change,
                    "direction": request.direction
                }
            }
        
        # NEU: Feature-Engineering Parameter hinzufügen
        if request.use_engineered_features:
            final_params['use_engineered_features'] = True
            if request.feature_engineering_windows:
                final_params['feature_engineering_windows'] = request.feature_engineering_windows
            else:
                final_params['feature_engineering_windows'] = [5, 10, 15]  # Default
        
        # NEU: SMOTE Parameter
        if not request.use_smote:
            final_params['use_smote'] = False
        
        # NEU: TimeSeriesSplit Parameter
        if not request.use_timeseries_split:
            final_params['use_timeseries_split'] = False
        if request.cv_splits:
            final_params['cv_splits'] = request.cv_splits
        
        job_id = await create_job(
            job_type="TRAIN",
            priority=5,
            train_model_type=request.model_type,
            train_target_var=request.target_var,
            train_operator=request.operator,
            train_value=request.target_value,
            train_start=request.train_start,
            train_end=request.train_end,
            train_features=request.features,  # JSONB
            train_phases=request.phases,  # JSONB
            train_params=final_params,  # JSONB (enthält jetzt auch zeitbasierte Parameter)
            progress_msg=request.name,  # ⚠️ WICHTIG: Name temporär in progress_msg speichern!
            # Zeitbasierte Vorhersage-Parameter
            train_future_minutes=request.future_minutes if request.use_time_based_prediction else None,
            train_price_change_percent=request.min_percent_change if request.use_time_based_prediction else None,
            train_target_direction=request.direction if request.use_time_based_prediction else None
        )
        
        logger.info(f"✅ TRAIN-Job erstellt: {job_id} für Modell '{request.name}'")
        
        return CreateJobResponse(
            job_id=job_id,
            message=f"Job erstellt. Modell '{request.name}' wird trainiert.",
            status="PENDING"
        )
    except Exception as e:
        logger.error(f"❌ Fehler beim Erstellen des TRAIN-Jobs: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/models", response_model=List[ModelResponse])
async def list_models_endpoint(
    status: Optional[str] = None,
    is_deleted: bool = False
):
    """Listet alle Modelle (mit optionalen Filtern)"""
    try:
        models = await list_models(status=status, is_deleted=is_deleted)
        # Konvertiere JSONB-Felder und extrahiere Confusion Matrix Werte für Legacy-Felder
        result = []
        for model in models:
            model_dict = dict(model)
            # NEU: Extrahiere Confusion Matrix Werte für Legacy-Felder (tp, tn, fp, fn)
            if model_dict.get('confusion_matrix') and isinstance(model_dict['confusion_matrix'], dict):
                cm = model_dict['confusion_matrix']
                model_dict['tp'] = cm.get('tp')
                model_dict['tn'] = cm.get('tn')
                model_dict['fp'] = cm.get('fp')
                model_dict['fn'] = cm.get('fn')
            result.append(ModelResponse(**model_dict))
        return result
    except DatabaseError as e:
        logger.error(f"❌ Datenbank-Fehler beim Auflisten der Modelle: {e.message}")
        raise HTTPException(status_code=500, detail=e.to_dict())
    except Exception as e:
        logger.error(f"❌ Unerwarteter Fehler beim Auflisten der Modelle: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail={"error": "InternalServerError", "message": "Ein unerwarteter Fehler ist aufgetreten"})

@router.get("/models/{model_id}", response_model=ModelResponse)
async def get_model_endpoint(model_id: int):
    """Holt Modell-Details"""
    try:
        model = await get_model(model_id)
        if not model:
            raise ModelNotFoundError(model_id)
        if model.get('is_deleted'):
            raise ModelNotFoundError(model_id)
        
        # NEU: Extrahiere Confusion Matrix Werte für Legacy-Felder (tp, tn, fp, fn)
        if model.get('confusion_matrix') and isinstance(model['confusion_matrix'], dict):
            cm = model['confusion_matrix']
            model['tp'] = cm.get('tp')
            model['tn'] = cm.get('tn')
            model['fp'] = cm.get('fp')
            model['fn'] = cm.get('fn')
        
        return ModelResponse(**dict(model))
    except ModelNotFoundError as e:
        logger.warning(f"⚠️ Modell nicht gefunden: {e.message}")
        raise HTTPException(status_code=404, detail=e.to_dict())
    except DatabaseError as e:
        logger.error(f"❌ Datenbank-Fehler beim Abrufen des Modells: {e.message}")
        raise HTTPException(status_code=500, detail=e.to_dict())
    except Exception as e:
        logger.error(f"❌ Unerwarteter Fehler beim Abrufen des Modells: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail={"error": "InternalServerError", "message": "Ein unerwarteter Fehler ist aufgetreten"})

@router.post("/models/{model_id}/test", response_model=CreateJobResponse, status_code=status.HTTP_201_CREATED)
async def test_model_job(model_id: int, request: TestModelRequest):
    """
    Erstellt einen TEST-Job (Modell wird asynchron getestet)
    """
    try:
        # Prüfe ob Modell existiert
        model = await get_model(model_id)
        if not model or model.get('is_deleted'):
            raise ModelNotFoundError(model_id)
        
        # Erstelle TEST-Job
        job_id = await create_job(
            job_type="TEST",
            priority=5,
            test_model_id=model_id,
            test_start=request.test_start,
            test_end=request.test_end
        )
        
        logger.info(f"✅ TEST-Job erstellt: {job_id} für Modell {model_id}")
        
        return CreateJobResponse(
            job_id=job_id,
            message=f"Test-Job erstellt für Modell {model_id}",
            status="PENDING"
        )
    except ModelNotFoundError as e:
        logger.warning(f"⚠️ Modell nicht gefunden: {e.message}")
        raise HTTPException(status_code=404, detail=e.to_dict())
    except DatabaseError as e:
        logger.error(f"❌ Datenbank-Fehler beim Erstellen des TEST-Jobs: {e.message}")
        raise HTTPException(status_code=500, detail=e.to_dict())
    except Exception as e:
        logger.error(f"❌ Unerwarteter Fehler beim Erstellen des TEST-Jobs: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail={"error": "InternalServerError", "message": "Ein unerwarteter Fehler ist aufgetreten"})

@router.post("/models/compare", response_model=CreateJobResponse, status_code=status.HTTP_201_CREATED)
async def compare_models_job(request: CompareModelsRequest):
    """
    Erstellt einen COMPARE-Job (Zwei Modelle werden asynchron verglichen)
    """
    try:
        # Prüfe ob beide Modelle existieren
        model_a = await get_model(request.model_a_id)
        model_b = await get_model(request.model_b_id)
        
        if not model_a or model_a.get('is_deleted'):
            raise ModelNotFoundError(request.model_a_id)
        if not model_b or model_b.get('is_deleted'):
            raise ModelNotFoundError(request.model_b_id)
        
        # Erstelle COMPARE-Job
        job_id = await create_job(
            job_type="COMPARE",
            priority=5,
            compare_model_a_id=request.model_a_id,
            compare_model_b_id=request.model_b_id,
            compare_start=request.test_start,
            compare_end=request.test_end
        )
        
        logger.info(f"✅ COMPARE-Job erstellt: {job_id} (Modell {request.model_a_id} vs {request.model_b_id})")
        
        return CreateJobResponse(
            job_id=job_id,
            message=f"Vergleichs-Job erstellt für Modell {request.model_a_id} vs {request.model_b_id}",
            status="PENDING"
        )
    except ModelNotFoundError as e:
        logger.warning(f"⚠️ Modell nicht gefunden: {e.message}")
        raise HTTPException(status_code=404, detail=e.to_dict())
    except DatabaseError as e:
        logger.error(f"❌ Datenbank-Fehler beim Erstellen des COMPARE-Jobs: {e.message}")
        raise HTTPException(status_code=500, detail=e.to_dict())
    except Exception as e:
        logger.error(f"❌ Unerwarteter Fehler beim Erstellen des COMPARE-Jobs: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail={"error": "InternalServerError", "message": "Ein unerwarteter Fehler ist aufgetreten"})

@router.patch("/models/{model_id}", response_model=ModelResponse)
async def update_model_endpoint(model_id: int, request: UpdateModelRequest):
    """
    Aktualisiert Modell-Name oder Beschreibung
    """
    try:
        model = await get_model(model_id)
        if not model:
            raise HTTPException(status_code=404, detail=f"Modell {model_id} nicht gefunden")
        if model.get('is_deleted'):
            raise HTTPException(status_code=404, detail=f"Modell {model_id} wurde gelöscht")
        
        # Prüfe ob Name bereits existiert (wenn geändert)
        if request.name and request.name != model.get('name'):
            existing = await list_models()
            if any(m.get('name') == request.name and m.get('id') != model_id for m in existing):
                raise HTTPException(status_code=400, detail=f"Modell-Name '{request.name}' existiert bereits")
        
        # Aktualisiere nur gesetzte Felder
        update_data = request.dict(exclude_unset=True)
        if not update_data:
            raise HTTPException(status_code=400, detail="Keine Update-Daten bereitgestellt")
        
        await update_model(model_id, name=update_data.get('name'), description=update_data.get('description'))
        logger.info(f"✅ Modell aktualisiert: {model_id} (name={update_data.get('name')}, description={update_data.get('description')})")
        
        # Lade aktualisiertes Modell
        updated_model = await get_model(model_id)
        return ModelResponse(**dict(updated_model))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Fehler beim Aktualisieren des Modells: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/models/{model_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_model_endpoint(model_id: int):
    """
    Soft-Delete: Markiert Modell als gelöscht (löscht Datei nicht!)
    """
    try:
        model = await get_model(model_id)
        if not model:
            raise HTTPException(status_code=404, detail=f"Modell {model_id} nicht gefunden")
        
        await delete_model(model_id)
        logger.info(f"✅ Modell gelöscht (soft): {model_id}")
        
        return Response(status_code=status.HTTP_204_NO_CONTENT)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Fehler beim Löschen des Modells: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/models/{model_id}/download")
async def download_model(model_id: int):
    """
    Download .pkl Datei des Modells
    """
    try:
        model = await get_model(model_id)
        if not model or model.get('is_deleted'):
            raise HTTPException(status_code=404, detail=f"Modell {model_id} nicht gefunden")
        
        model_path = model.get('model_file_path')
        if not model_path or not os.path.exists(model_path):
            raise HTTPException(status_code=404, detail=f"Modell-Datei nicht gefunden: {model_path}")
        
        return FileResponse(
            path=model_path,
            filename=os.path.basename(model_path),
            media_type='application/octet-stream'
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Fehler beim Download des Modells: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================
# Queue Endpoints
# ============================================================

def convert_jsonb_fields(job_dict: dict) -> dict:
    """Konvertiert JSONB-Strings zu Python-Objekten (refactored: nutze Helper-Funktion)"""
    from app.database.utils import convert_jsonb_fields as convert_jsonb
    
    jsonb_fields = ['train_features', 'train_phases', 'train_params']
    return convert_jsonb(job_dict, jsonb_fields, direction="from")

@router.get("/queue", response_model=List[JobResponse])
async def list_jobs_endpoint(
    status: Optional[str] = None,
    job_type: Optional[str] = None
):
    """Listet alle Jobs (mit optionalen Filtern)"""
    try:
        jobs = await list_jobs(status=status, job_type=job_type)
        converted_jobs = []
        for job in jobs:
            job_dict = dict(job)
            converted = convert_jsonb_fields(job_dict)
            
            # Lade Ergebnisse wenn verfügbar (nur für COMPLETED Jobs, um Performance zu sparen)
            result_model = None
            result_test = None
            result_comparison = None
            
            if converted.get('status') == 'COMPLETED':
                if converted.get('result_model_id'):
                    model = await get_model(converted['result_model_id'])
                    if model:
                        result_model = ModelResponse(**dict(model))
                
                if converted.get('result_test_id'):
                    test_result = await get_test_result(converted['result_test_id'])
                    if test_result:
                        result_test = TestResultResponse(**dict(test_result))
                
                if converted.get('result_comparison_id'):
                    comparison = await get_comparison(converted['result_comparison_id'])
                    if comparison:
                        result_comparison = ComparisonResponse(**dict(comparison))
            
            job_response = JobResponse(**converted)
            job_response.result_model = result_model
            job_response.result_test = result_test
            job_response.result_comparison = result_comparison
            converted_jobs.append(job_response)
        
        return converted_jobs
    except Exception as e:
        logger.error(f"❌ Fehler beim Auflisten der Jobs: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/queue/{job_id}", response_model=JobResponse)
async def get_job_endpoint(job_id: int):
    """Holt Job-Details mit Ergebnissen"""
    try:
        job = await get_job(job_id)
        if not job:
            raise HTTPException(status_code=404, detail=f"Job {job_id} nicht gefunden")
        converted_job = convert_jsonb_fields(dict(job))
        
        # Lade Ergebnisse wenn verfügbar
        result_model = None
        result_test = None
        result_comparison = None
        
        if converted_job.get('result_model_id'):
            model = await get_model(converted_job['result_model_id'])
            if model:
                result_model = ModelResponse(**dict(model))
        
        if converted_job.get('result_test_id'):
            test_result = await get_test_result(converted_job['result_test_id'])
            if test_result:
                result_test = TestResultResponse(**dict(test_result))
        
        if converted_job.get('result_comparison_id'):
            comparison = await get_comparison(converted_job['result_comparison_id'])
            if comparison:
                result_comparison = ComparisonResponse(**dict(comparison))
        
        # Erstelle Response mit Ergebnissen
        job_response = JobResponse(**converted_job)
        job_response.result_model = result_model
        job_response.result_test = result_test
        job_response.result_comparison = result_comparison
        
        return job_response
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Fehler beim Abrufen des Jobs: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================
# System Endpoints
# ============================================================

@router.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health Check Endpoint
    Gibt Status, DB-Verbindung, Uptime zurück
    Status Code: 200 wenn healthy, 503 wenn degraded
    """
    try:
        health = await get_health_status()
        health_response = HealthResponse(**health)
        
        # Status Code basierend auf Health
        if health["status"] == "healthy":
            return health_response
        else:
            # Degraded: 503 Service Unavailable
            return JSONResponse(
                content=health_response.dict(),
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE
            )
    except Exception as e:
        logger.error(f"❌ Fehler beim Health Check: {e}")
        health_response = HealthResponse(
            status="degraded",
            db_connected=False,
            uptime_seconds=0,
            start_time=None,
            total_jobs_processed=0,
            last_error=str(e)
        )
        return JSONResponse(
            content=health_response.dict(),
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE
        )

@router.get("/metrics")
async def metrics_endpoint():
    """
    Prometheus Metrics Endpoint
    Gibt Metrics im Prometheus-Format zurück
    """
    try:
        metrics = generate_metrics()
        return Response(
            content=metrics,
            media_type="text/plain; version=0.0.4; charset=utf-8"
        )
    except Exception as e:
        logger.error(f"❌ Fehler beim Generieren der Metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/phases")
async def get_phases_endpoint():
    """
    Lade alle Coin-Phasen aus ref_coin_phases mit interval_seconds
    
    Returns:
        Liste von Phasen mit: id, name, interval_seconds, max_age_minutes
    """
    try:
        phases = await get_coin_phases()
        return phases
    except Exception as e:
        logger.error(f"❌ Fehler beim Laden der Phasen: {e}")
        raise HTTPException(status_code=500, detail=f"Fehler beim Laden der Phasen: {str(e)}")

# ============================================================
# Comparison Endpoints
# ============================================================

@router.get("/comparisons", response_model=List[ComparisonResponse])
async def list_comparisons_endpoint(limit: int = 100, offset: int = 0):
    """
    Listet alle Vergleichs-Ergebnisse
    """
    try:
        comparisons = await list_comparisons(limit=limit, offset=offset)
        # Konvertiere zu ComparisonResponse
        result = []
        for comp in comparisons:
            try:
                result.append(ComparisonResponse(**dict(comp)))
            except Exception as e:
                logger.warning(f"⚠️ Fehler beim Konvertieren von Vergleich {comp.get('id')}: {e}")
                continue
        return result
    except Exception as e:
        logger.error(f"❌ Fehler beim Auflisten der Vergleiche: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/comparisons/{comparison_id}", response_model=ComparisonResponse)
async def get_comparison_endpoint(comparison_id: int):
    """
    Holt einen einzelnen Vergleich
    """
    try:
        comparison = await get_comparison(comparison_id)
        if not comparison:
            raise HTTPException(status_code=404, detail=f"Vergleich {comparison_id} nicht gefunden")
        return ComparisonResponse(**dict(comparison))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Fehler beim Abrufen des Vergleichs: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/comparisons/{comparison_id}")
async def delete_comparison_endpoint(comparison_id: int):
    """
    Löscht einen Vergleich
    """
    try:
        comparison = await get_comparison(comparison_id)
        if not comparison:
            raise HTTPException(status_code=404, detail=f"Vergleich {comparison_id} nicht gefunden")
        
        deleted = await delete_comparison(comparison_id)
        if deleted:
            logger.info(f"✅ Vergleich gelöscht: {comparison_id}")
            return {"message": f"Vergleich {comparison_id} erfolgreich gelöscht"}
        else:
            raise HTTPException(status_code=500, detail="Fehler beim Löschen des Vergleichs")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Fehler beim Löschen des Vergleichs: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================
# Test Results Endpoints
# ============================================================

@router.get("/test-results", response_model=List[TestResultResponse])
async def list_test_results_endpoint(limit: int = 100, offset: int = 0):
    """
    Listet alle Test-Ergebnisse
    """
    try:
        test_results = await list_all_test_results(limit=limit, offset=offset)
        # Konvertiere zu TestResultResponse
        result = []
        for test in test_results:
            try:
                result.append(TestResultResponse(**dict(test)))
            except Exception as e:
                logger.warning(f"⚠️ Fehler beim Konvertieren von Test-Ergebnis {test.get('id')}: {e}")
                continue
        return result
    except Exception as e:
        logger.error(f"❌ Fehler beim Auflisten der Test-Ergebnisse: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/test-results/{test_id}", response_model=TestResultResponse)
async def get_test_result_endpoint(test_id: int):
    """
    Holt ein einzelnes Test-Ergebnis
    """
    try:
        test_result = await get_test_result(test_id)
        if not test_result:
            raise HTTPException(status_code=404, detail=f"Test-Ergebnis {test_id} nicht gefunden")
        return TestResultResponse(**dict(test_result))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Fehler beim Abrufen des Test-Ergebnisses: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/test-results/{test_id}")
async def delete_test_result_endpoint(test_id: int):
    """
    Löscht ein Test-Ergebnis
    """
    try:
        test_result = await get_test_result(test_id)
        if not test_result:
            raise HTTPException(status_code=404, detail=f"Test-Ergebnis {test_id} nicht gefunden")
        
        deleted = await delete_test_result(test_id)
        if deleted:
            logger.info(f"✅ Test-Ergebnis gelöscht: {test_id}")
            return {"message": f"Test-Ergebnis {test_id} erfolgreich gelöscht"}
        else:
            raise HTTPException(status_code=500, detail="Fehler beim Löschen des Test-Ergebnisses")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Fehler beim Löschen des Test-Ergebnisses: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/data-availability")
async def get_data_availability(pool = Depends(get_pool)):
    """
    Gibt zurück, wann die ältesten und neuesten Einträge in coin_metrics vorhanden sind.
    
    Returns:
        {
            "min_timestamp": "2025-12-20T10:00:00Z",
            "max_timestamp": "2025-12-23T15:30:00Z"
        }
    """
    try:
        # Einfache Abfrage: Nur Min/Max Timestamps
        query = """
            SELECT 
                MIN(timestamp) as min_ts,
                MAX(timestamp) as max_ts
            FROM coin_metrics
        """
        row = await pool.fetchrow(query)
        
        if not row or not row['min_ts']:
            return {
                "min_timestamp": None,
                "max_timestamp": None
            }
        
        min_ts = row['min_ts']
        max_ts = row['max_ts']
        
        # Konvertiere zu ISO-Format Strings
        # Stelle sicher, dass Zeitzone UTC ist
        from datetime import timezone
        if min_ts.tzinfo is None:
            min_ts = min_ts.replace(tzinfo=timezone.utc)
        if max_ts.tzinfo is None:
            max_ts = max_ts.replace(tzinfo=timezone.utc)
        
        return {
            "min_timestamp": min_ts.isoformat().replace('+00:00', 'Z'),
            "max_timestamp": max_ts.isoformat().replace('+00:00', 'Z')
        }
    except Exception as e:
        logger.error(f"Fehler beim Abrufen der Daten-Verfügbarkeit: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

# Zusätzliche Endpoints für Coolify-Kompatibilität (ohne /api Prefix)
coolify_router = APIRouter()

@coolify_router.get("/health")
async def health_check_coolify():
    """Health Check für Coolify (ohne /api Prefix)"""
    return await health_check()

@coolify_router.get("/metrics")
async def metrics_coolify():
    """Metrics für Coolify (ohne /api Prefix)"""
    return await metrics_endpoint()

