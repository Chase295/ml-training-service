"""
Custom Exceptions für ML Training Service

Strukturierte Fehlerbehandlung mit spezifischen Exception-Klassen
für bessere Fehlerbehandlung und Logging.
"""
from typing import Optional, Dict, Any


class MLTrainingError(Exception):
    """Basis-Exception für alle ML Training Service Fehler"""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        self.message = message
        self.details = details or {}
        super().__init__(self.message)
    
    def to_dict(self) -> Dict[str, Any]:
        """Konvertiert Exception zu Dictionary für API-Responses"""
        return {
            "error": self.__class__.__name__,
            "message": self.message,
            "details": self.details
        }


class ModelNotFoundError(MLTrainingError):
    """Modell wurde nicht gefunden"""
    
    def __init__(self, model_id: int):
        super().__init__(
            f"Modell {model_id} nicht gefunden",
            {"model_id": model_id}
        )


class InvalidModelParametersError(MLTrainingError):
    """Ungültige Modell-Parameter"""
    
    def __init__(self, message: str, parameters: Optional[Dict[str, Any]] = None):
        super().__init__(
            f"Ungültige Modell-Parameter: {message}",
            {"parameters": parameters or {}}
        )


class DatabaseError(MLTrainingError):
    """Datenbank-Fehler"""
    
    def __init__(self, message: str, operation: Optional[str] = None):
        super().__init__(
            f"Datenbank-Fehler: {message}",
            {"operation": operation}
        )


class JobNotFoundError(MLTrainingError):
    """Job wurde nicht gefunden"""
    
    def __init__(self, job_id: int):
        super().__init__(
            f"Job {job_id} nicht gefunden",
            {"job_id": job_id}
        )


class JobProcessingError(MLTrainingError):
    """Fehler bei der Job-Verarbeitung"""
    
    def __init__(self, job_id: int, message: str, job_type: Optional[str] = None):
        super().__init__(
            f"Job {job_id} Verarbeitungsfehler: {message}",
            {"job_id": job_id, "job_type": job_type}
        )


class TrainingError(MLTrainingError):
    """Fehler beim Training"""
    
    def __init__(self, message: str, model_type: Optional[str] = None):
        super().__init__(
            f"Training-Fehler: {message}",
            {"model_type": model_type}
        )


class TestError(MLTrainingError):
    """Fehler beim Testen"""
    
    def __init__(self, message: str, model_id: Optional[int] = None):
        super().__init__(
            f"Test-Fehler: {message}",
            {"model_id": model_id}
        )


class ComparisonError(MLTrainingError):
    """Fehler beim Vergleichen"""
    
    def __init__(self, message: str, model_a_id: Optional[int] = None, model_b_id: Optional[int] = None):
        super().__init__(
            f"Vergleichs-Fehler: {message}",
            {"model_a_id": model_a_id, "model_b_id": model_b_id}
        )


class DataError(MLTrainingError):
    """Fehler bei Datenverarbeitung"""
    
    def __init__(self, message: str, data_info: Optional[Dict[str, Any]] = None):
        super().__init__(
            f"Daten-Fehler: {message}",
            {"data_info": data_info or {}}
        )


class ValidationError(MLTrainingError):
    """Validierungs-Fehler"""
    
    def __init__(self, message: str, field: Optional[str] = None, value: Optional[Any] = None):
        super().__init__(
            f"Validierungs-Fehler: {message}",
            {"field": field, "value": str(value) if value is not None else None}
        )

