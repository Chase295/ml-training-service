"""
API-Validierungslogik für Request-Parameter
"""
from typing import Optional
from datetime import datetime
from app.utils.exceptions import ValidationError

# ============================================================
# Datumsvalidierung
# ============================================================

def validate_date_range(
    start: Optional[datetime],
    end: Optional[datetime],
    field_name: str = "Zeitraum"
) -> None:
    """
    Validiert dass start < end.
    
    Args:
        start: Start-Datum
        end: End-Datum
        field_name: Name des Feldes für Fehlermeldung
    
    Raises:
        ValidationError: Wenn start >= end
    
    Examples:
        >>> from datetime import datetime
        >>> start = datetime(2024, 1, 1)
        >>> end = datetime(2024, 1, 2)
        >>> validate_date_range(start, end)  # OK
        >>> validate_date_range(end, start)  # Raises ValidationError
    """
    if start is None or end is None:
        return  # None-Werte werden von Pydantic validiert
    
    if start >= end:
        raise ValidationError(
            f"{field_name}: start muss vor end liegen",
            {"start": str(start), "end": str(end)}
        )

def validate_test_period_overlap(
    train_start: datetime,
    train_end: datetime,
    test_start: datetime,
    test_end: datetime
) -> tuple[bool, Optional[str]]:
    """
    Prüft ob Test-Zeitraum mit Trainings-Zeitraum überlappt.
    
    Args:
        train_start: Trainings-Start
        train_end: Trainings-Ende
        test_start: Test-Start
        test_end: Test-Ende
    
    Returns:
        Tuple (has_overlap: bool, overlap_note: Optional[str])
    
    Examples:
        >>> from datetime import datetime
        >>> train_start = datetime(2024, 1, 1)
        >>> train_end = datetime(2024, 1, 10)
        >>> test_start = datetime(2024, 1, 11)
        >>> test_end = datetime(2024, 1, 20)
        >>> validate_test_period_overlap(train_start, train_end, test_start, test_end)
        (False, None)
        >>> test_start = datetime(2024, 1, 5)
        >>> validate_test_period_overlap(train_start, train_end, test_start, test_end)
        (True, 'Test-Zeitraum überlappt mit Trainings-Zeitraum')
    """
    # Prüfe ob Test-Zeitraum mit Trainings-Zeitraum überlappt
    if test_start < train_end and test_end > train_start:
        overlap_days = (min(test_end, train_end) - max(test_start, train_start)).days
        return True, f"Test-Zeitraum überlappt mit Trainings-Zeitraum ({overlap_days} Tage)"
    
    return False, None

def validate_minimum_test_duration(
    test_start: datetime,
    test_end: datetime,
    min_days: float = 1.0
) -> tuple[bool, Optional[str]]:
    """
    Prüft ob Test-Zeitraum mindestens min_days lang ist.
    
    Args:
        test_start: Test-Start
        test_end: Test-Ende
        min_days: Mindest-Dauer in Tagen
    
    Returns:
        Tuple (is_valid: bool, warning: Optional[str])
    
    Examples:
        >>> from datetime import datetime, timedelta
        >>> test_start = datetime(2024, 1, 1)
        >>> test_end = test_start + timedelta(days=2)
        >>> validate_minimum_test_duration(test_start, test_end, min_days=1.0)
        (True, None)
        >>> test_end = test_start + timedelta(hours=12)
        >>> validate_minimum_test_duration(test_start, test_end, min_days=1.0)
        (False, 'Test-Zeitraum ist zu kurz: 0.5 Tage (empfohlen: mindestens 1.0 Tage)')
    """
    duration_days = (test_end - test_start).total_seconds() / 86400.0
    
    if duration_days < min_days:
        return False, f"Test-Zeitraum ist zu kurz: {duration_days:.1f} Tage (empfohlen: mindestens {min_days} Tage)"
    
    return True, None

# ============================================================
# Modell-Validierung
# ============================================================

def validate_model_type(model_type: str) -> None:
    """
    Validiert dass model_type gültig ist.
    
    Args:
        model_type: Modell-Typ
    
    Raises:
        ValidationError: Wenn model_type ungültig
    
    Examples:
        >>> validate_model_type("random_forest")  # OK
        >>> validate_model_type("invalid")  # Raises ValidationError
    """
    valid_types = ["random_forest", "xgboost"]
    
    if model_type not in valid_types:
        raise ValidationError(
            f"Ungültiger Modell-Typ: {model_type}",
            {"model_type": model_type, "valid_types": valid_types}
        )

def validate_target_operator(operator: Optional[str]) -> None:
    """
    Validiert dass target_operator gültig ist.
    
    Args:
        operator: Operator (">", "<", ">=", "<=", "==")
    
    Raises:
        ValidationError: Wenn operator ungültig
    
    Examples:
        >>> validate_target_operator(">")  # OK
        >>> validate_target_operator("invalid")  # Raises ValidationError
    """
    if operator is None:
        return  # None ist erlaubt (für zeitbasierte Vorhersagen)
    
    valid_operators = [">", "<", ">=", "<=", "=="]
    
    if operator not in valid_operators:
        raise ValidationError(
            f"Ungültiger Operator: {operator}",
            {"operator": operator, "valid_operators": valid_operators}
        )

