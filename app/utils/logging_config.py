"""
Strukturiertes Logging für ML Training Service

Features:
- JSON-Logging (optional)
- Konfigurierbares Log-Level
- Request-ID für Tracing
- Strukturierte Log-Messages
"""
import os
import json
import logging
import uuid
from typing import Optional, Dict, Any
from datetime import datetime, timezone
from contextvars import ContextVar

# Context-Variable für Request-ID (Thread-safe)
request_id_var: ContextVar[Optional[str]] = ContextVar('request_id', default=None)

# Logging-Konfiguration aus Environment Variables
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
LOG_FORMAT = os.getenv("LOG_FORMAT", "text")  # "text" oder "json"
LOG_JSON_INDENT = int(os.getenv("LOG_JSON_INDENT", "0"))  # 0 = kompakt, 2+ = formatiert


class StructuredFormatter(logging.Formatter):
    """Formatter für strukturierte Logs (JSON oder Text)"""
    
    def __init__(self, use_json: bool = False, json_indent: int = 0):
        super().__init__()
        self.use_json = use_json
        self.json_indent = json_indent
    
    def format(self, record: logging.LogRecord) -> str:
        """Formatiert Log-Record zu strukturiertem Format"""
        # Basis-Informationen
        log_data = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        
        # Request-ID hinzufügen (wenn vorhanden)
        request_id = request_id_var.get()
        if request_id:
            log_data["request_id"] = request_id
        
        # Exception-Informationen hinzufügen
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        
        # Zusätzliche Felder aus record (wenn vorhanden)
        if hasattr(record, "extra_fields"):
            log_data.update(record.extra_fields)
        
        # JSON oder Text-Format
        if self.use_json:
            return json.dumps(log_data, indent=self.json_indent, ensure_ascii=False)
        else:
            # Text-Format: Strukturiert aber lesbar
            parts = [
                f"[{log_data['timestamp']}]",
                f"[{log_data['level']}]",
                f"[{log_data['logger']}]"
            ]
            if request_id:
                parts.append(f"[req:{request_id[:8]}]")
            parts.append(log_data['message'])
            
            if record.exc_info:
                parts.append(f"\n{log_data['exception']}")
            
            return " ".join(parts)


def setup_logging():
    """
    Konfiguriert strukturiertes Logging für die gesamte Anwendung
    
    Verwendet Environment Variables:
    - LOG_LEVEL: Log-Level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    - LOG_FORMAT: "text" oder "json"
    - LOG_JSON_INDENT: JSON-Indentation (0 = kompakt, 2+ = formatiert)
    """
    # Log-Level validieren
    valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    if LOG_LEVEL not in valid_levels:
        logging.warning(f"⚠️ Ungültiges LOG_LEVEL '{LOG_LEVEL}', verwende 'INFO'")
        level = logging.INFO
    else:
        level = getattr(logging, LOG_LEVEL)
    
    # Format bestimmen
    use_json = LOG_FORMAT.lower() == "json"
    
    # Root-Logger konfigurieren
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Entferne vorhandene Handler
    root_logger.handlers.clear()
    
    # Console Handler erstellen
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    
    # Formatter setzen
    formatter = StructuredFormatter(use_json=use_json, json_indent=LOG_JSON_INDENT)
    console_handler.setFormatter(formatter)
    
    # Handler hinzufügen
    root_logger.addHandler(console_handler)
    
    # Logging-Konfiguration loggen
    logger = logging.getLogger(__name__)
    logger.info(
        f"📝 Logging konfiguriert: Level={LOG_LEVEL}, Format={'JSON' if use_json else 'Text'}"
    )


def get_logger(name: str) -> logging.Logger:
    """
    Gibt einen Logger mit dem angegebenen Namen zurück
    
    Args:
        name: Logger-Name (normalerweise __name__)
    
    Returns:
        Logger-Instanz
    """
    return logging.getLogger(name)


def set_request_id(request_id: Optional[str] = None) -> str:
    """
    Setzt Request-ID für aktuellen Context
    
    Args:
        request_id: Optionale Request-ID (wird generiert wenn None)
    
    Returns:
        Request-ID (String)
    """
    if request_id is None:
        request_id = str(uuid.uuid4())
    request_id_var.set(request_id)
    return request_id


def get_request_id() -> Optional[str]:
    """
    Gibt aktuelle Request-ID zurück
    
    Returns:
        Request-ID oder None
    """
    return request_id_var.get()


def log_with_context(
    logger: logging.Logger,
    level: int,
    message: str,
    extra_fields: Optional[Dict[str, Any]] = None,
    exc_info: Optional[Exception] = None
):
    """
    Loggt eine Nachricht mit zusätzlichen Context-Feldern
    
    Args:
        logger: Logger-Instanz
        level: Log-Level (logging.INFO, logging.ERROR, etc.)
        message: Log-Nachricht
        extra_fields: Zusätzliche Felder für strukturierte Logs
        exc_info: Exception-Info (optional)
    """
    # Erstelle LogRecord mit extra_fields
    record = logger.makeRecord(
        logger.name,
        level,
        "",  # filename
        0,   # lineno
        message,
        (),  # args
        exc_info
    )
    
    # Extra-Felder hinzufügen
    if extra_fields:
        record.extra_fields = extra_fields
    
    logger.handle(record)

