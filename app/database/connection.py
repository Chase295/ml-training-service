"""
Datenbank-Verbindungsmanagement für ML Training Service

Verwaltet den asyncpg Connection Pool für PostgreSQL-Verbindungen.
Die Datenbank ist EXTERN und wird über DB_DSN konfiguriert.
"""
import asyncpg
import logging
from typing import Optional
from app.utils.config import DB_DSN, get_runtime_config
from app.utils.logging_config import get_logger

logger = get_logger(__name__)

# Globaler Connection Pool (wird beim ersten Aufruf erstellt)
pool: Optional[asyncpg.Pool] = None

async def get_pool() -> asyncpg.Pool:
    """
    Erstellt oder gibt existierenden Connection Pool zurück.
    
    Der Pool wird beim ersten Aufruf erstellt und danach wiederverwendet.
    Dies verbessert die Performance durch Connection-Reuse.
    
    Returns:
        asyncpg.Pool: Connection Pool für Datenbank-Operationen
        
    Raises:
        Exception: Wenn die Verbindung zur Datenbank fehlschlägt
        
    Example:
        ```python
        pool = await get_pool()
        result = await pool.fetchval("SELECT COUNT(*) FROM ml_models")
        ```
    """
    global pool
    if pool is None:
        try:
            # Verwende runtime config oder fallback zu DB_DSN
            current_dsn = get_runtime_config('db_dsn', DB_DSN)

            # DB_DSN enthält externe DB-Adresse, z.B.:
            # postgresql://user:pass@100.118.155.75:5432/crypto
            pool = await asyncpg.create_pool(
                current_dsn,
                min_size=1,
                max_size=10,
                command_timeout=60
            )
            logger.info(f"✅ Datenbank-Pool erstellt: {DB_DSN.split('@')[1] if '@' in DB_DSN else 'localhost'}")
        except Exception as e:
            logger.error(f"❌ Fehler beim Erstellen des DB-Pools: {e}")
            raise
    return pool

async def close_pool() -> None:
    """
    Schließt den Connection Pool graceful.
    
    Wird typischerweise beim Service-Shutdown aufgerufen, um alle
    offenen Verbindungen ordnungsgemäß zu schließen.
    
    Example:
        ```python
        await close_pool()
        ```
    """
    global pool
    if pool:
        await pool.close()
        pool = None
        logger.info("✅ Datenbank-Pool geschlossen")

async def test_connection() -> bool:
    """
    Testet die Datenbank-Verbindung durch eine einfache Query.
    
    Returns:
        bool: True wenn Verbindung erfolgreich, False bei Fehler
        
    Example:
        ```python
        is_connected = await test_connection()
        if not is_connected:
            logger.error("Datenbank nicht erreichbar")
        ```
    """
    try:
        pool = await get_pool()
        result = await pool.fetchval("SELECT 1")
        return result == 1
    except Exception as e:
        logger.error(f"❌ DB-Verbindungstest fehlgeschlagen: {e}")
        return False

