"""
Datenbank-Helper-Funktionen für JSONB-Konvertierung und häufige Patterns
"""
import json
import logging
from typing import Any, Optional, Dict, List, Union

logger = logging.getLogger(__name__)

# ============================================================
# JSONB-Konvertierung
# ============================================================

def to_jsonb(value: Optional[Union[Dict, List, str]]) -> Optional[str]:
    """
    Konvertiert Python-Objekt (Dict/List) zu JSONB-String für PostgreSQL.
    
    Args:
        value: Python-Objekt (Dict, List) oder bereits String/None
    
    Returns:
        JSON-String oder None
    
    Examples:
        >>> to_jsonb({"key": "value"})
        '{"key": "value"}'
        >>> to_jsonb([1, 2, 3])
        '[1, 2, 3]'
        >>> to_jsonb(None)
        None
        >>> to_jsonb("already_string")
        'already_string'
    """
    if value is None:
        return None
    
    # Falls bereits String, direkt zurückgeben
    if isinstance(value, str):
        return value
    
    # Konvertiere zu JSON-String
    try:
        return json.dumps(value)
    except (TypeError, ValueError) as e:
        logger.warning(f"⚠️ Konnte Wert nicht zu JSONB konvertieren: {e}")
        return None

def from_jsonb(value: Optional[Union[str, Dict, List]]) -> Optional[Union[Dict, List]]:
    """
    Konvertiert JSONB-String von PostgreSQL zu Python-Objekt (Dict/List).
    
    Args:
        value: JSON-String oder bereits Python-Objekt/None
    
    Returns:
        Python-Objekt (Dict/List) oder None
    
    Examples:
        >>> from_jsonb('{"key": "value"}')
        {'key': 'value'}
        >>> from_jsonb('[1, 2, 3]')
        [1, 2, 3]
        >>> from_jsonb(None)
        None
        >>> from_jsonb({"already": "dict"})
        {'already': 'dict'}
    """
    if value is None:
        return None
    
    # Falls bereits Python-Objekt (Dict/List), direkt zurückgeben
    if isinstance(value, (dict, list)):
        return value
    
    # Falls String, parse zu Python-Objekt
    if isinstance(value, str):
        try:
            return json.loads(value)
        except (json.JSONDecodeError, TypeError) as e:
            # Fallback: Versuche als Python-Literal zu evaluieren
            try:
                import ast
                return ast.literal_eval(value)
            except (ValueError, SyntaxError):
                logger.warning(f"⚠️ Konnte JSONB-String nicht parsen: {value[:100]}...")
                return None
    
    # Unbekannter Typ
    logger.warning(f"⚠️ Unbekannter Typ für JSONB-Konvertierung: {type(value)}")
    return None

def convert_jsonb_fields(
    data: Dict[str, Any],
    fields: List[str],
    direction: str = "from"
) -> Dict[str, Any]:
    """
    Konvertiert mehrere JSONB-Felder in einem Dictionary.
    
    Args:
        data: Dictionary mit Daten
        fields: Liste von Feldnamen, die konvertiert werden sollen
        direction: "from" (String → Python-Objekt) oder "to" (Python-Objekt → String)
    
    Returns:
        Dictionary mit konvertierten Feldern
    
    Examples:
        >>> data = {"features": '["price_open", "price_close"]', "params": {"n_estimators": 100}}
        >>> convert_jsonb_fields(data, ["features"], direction="from")
        {'features': ['price_open', 'price_close'], 'params': {'n_estimators': 100}}
        >>> convert_jsonb_fields(data, ["params"], direction="to")
        {'features': '["price_open", "price_close"]', 'params': '{"n_estimators": 100}'}
    """
    result = data.copy()
    
    for field in fields:
        if field in result and result[field] is not None:
            if direction == "from":
                result[field] = from_jsonb(result[field])
            elif direction == "to":
                result[field] = to_jsonb(result[field])
            else:
                raise ValueError(f"Ungültige direction: {direction}. Muss 'from' oder 'to' sein.")
    
    return result

# ============================================================
# Query-Helper
# ============================================================

def build_where_clause(
    conditions: Dict[str, Any],
    operator: str = "AND"
) -> tuple[str, List[Any]]:
    """
    Baut eine WHERE-Klausel aus Bedingungen.
    
    Args:
        conditions: Dictionary mit Feldname → Wert Mappings
        operator: Logischer Operator ("AND" oder "OR")
    
    Returns:
        Tuple (WHERE-Klausel, Parameter-Liste)
    
    Examples:
        >>> build_where_clause({"status": "READY", "model_type": "random_forest"})
        ('WHERE status = $1 AND model_type = $2', ['READY', 'random_forest'])
    """
    if not conditions:
        return ("", [])
    
    where_parts = []
    params = []
    param_index = 1
    
    for field, value in conditions.items():
        if value is not None:
            where_parts.append(f"{field} = ${param_index}")
            params.append(value)
            param_index += 1
    
    if not where_parts:
        return ("", [])
    
    where_clause = f"WHERE {' ' + operator + ' '.join(where_parts)}" if len(where_parts) > 1 else f"WHERE {where_parts[0]}"
    return (where_clause, params)

