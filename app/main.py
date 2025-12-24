"""
FastAPI Main App für ML Training Service
"""
import asyncio
import time
import logging
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from app.api.routes import router, coolify_router
from app.database.connection import get_pool, close_pool
from app.utils.metrics import init_health_status
from app.utils.config import API_PORT
from app.utils.logging_config import setup_logging, get_logger, set_request_id, get_request_id

# Strukturiertes Logging konfigurieren
setup_logging()
logger = get_logger(__name__)

# FastAPI App erstellen
app = FastAPI(
    title="ML Training Service",
    description="Machine Learning Training Service für Coin-Bot",
    version="1.0.0"
)

# Request-ID Middleware (muss vor CORS sein)
class RequestIDMiddleware(BaseHTTPMiddleware):
    """Middleware für Request-ID Tracking"""
    
    async def dispatch(self, request: Request, call_next):
        # Generiere oder hole Request-ID aus Header
        request_id = request.headers.get("X-Request-ID")
        if not request_id:
            request_id = set_request_id()
        else:
            set_request_id(request_id)
        
        # Response mit Request-ID Header
        response = await call_next(request)
        response.headers["X-Request-ID"] = get_request_id() or request_id
        return response

app.add_middleware(RequestIDMiddleware)

# CORS (falls nötig)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In Produktion: spezifische Origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Router einbinden
app.include_router(router)
app.include_router(coolify_router)

@app.on_event("startup")
async def startup():
    """Startup Event: Initialisiert DB-Pool, Health Status, Worker"""
    logger.info("🚀 Starte ML Training Service...")
    
    try:
        # DB-Pool erstellen
        await get_pool()  # Initialisiert Pool
        logger.info("✅ Datenbank-Pool erstellt")
        
        # Health Status initialisieren
        init_health_status()
        logger.info("✅ Health Status initialisiert")
        
        # Starte Job Worker
        from app.queue.job_manager import start_worker
        asyncio.create_task(start_worker())
        logger.info("✅ Job Worker gestartet")
        
        # Starte Background-Task für Job-Metriken (alle 5 Sekunden)
        from app.utils.metrics import update_all_job_metrics
        async def metrics_updater():
            """Aktualisiert regelmäßig Job-Metriken für Prometheus/Grafana"""
            while True:
                try:
                    await update_all_job_metrics()
                    await asyncio.sleep(5)  # Alle 5 Sekunden aktualisieren
                except Exception as e:
                    logger.error(f"Fehler beim Aktualisieren der Job-Metriken: {e}", exc_info=True)
                    await asyncio.sleep(5)
        
        asyncio.create_task(metrics_updater())
        logger.info("✅ Job-Metriken-Updater gestartet (alle 5 Sekunden)")
        
        logger.info("✅ Service gestartet: DB verbunden, Worker läuft, Metriken-Updater aktiv")
    except Exception as e:
        logger.error(f"❌ Fehler beim Startup: {e}")
        raise

@app.on_event("shutdown")
async def shutdown():
    """Shutdown Event: Schließt DB-Pool graceful"""
    logger.info("👋 Beende ML Training Service...")
    
    try:
        await close_pool()
        logger.info("✅ Datenbank-Pool geschlossen")
        logger.info("👋 Service beendet")
    except Exception as e:
        logger.error(f"❌ Fehler beim Shutdown: {e}")

@app.get("/")
async def root():
    """Root Endpoint"""
    return {
        "service": "ML Training Service",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=API_PORT,
        reload=False  # In Docker: False
    )

