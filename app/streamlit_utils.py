"""
Streamlit Utility Functions
Hilfsfunktionen für API-Calls, Konfiguration und gemeinsame Operationen
"""
import streamlit as st
import os
import httpx
import yaml
import json
import subprocess
import re
from typing import List, Dict, Any, Optional
from pathlib import Path
from urllib.parse import urlparse

# Konfiguration
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
CONFIG_FILE = "/app/config/config.yaml"
ENV_FILE = "/app/config/.env"
SERVICE_NAME = os.getenv("SERVICE_NAME", "ml-training-service")
COOLIFY_MODE = os.getenv("COOLIFY_MODE", "false").lower() == "true"

# Verfügbare Target-Variablen für Training
AVAILABLE_TARGETS = [
    "market_cap_close", "price_close", "volume_sol"
]

# Verfügbare Features für Training
AVAILABLE_FEATURES = [
    # Basis OHLC
    "price_open", "price_high", "price_low", "price_close",

    # Volumen
    "volume_sol", "buy_volume_sol", "sell_volume_sol", "net_volume_sol",

    # Market Cap & Phase
    "market_cap_close", "phase_id_at_time",

    # ⚠️ KRITISCH für Rug-Detection
    "dev_sold_amount",  # Wichtigster Indikator für Rug-Pulls!

    # Ratio-Metriken (Bot-Spam vs. echtes Interesse)
    "buy_pressure_ratio",
    "unique_signer_ratio",

    # Whale-Aktivität
    "whale_buy_volume_sol",
    "whale_sell_volume_sol",
    "num_whale_buys",
    "num_whale_sells",

    # Volatilität
    "volatility_pct",
    "avg_trade_size_sol"
]

# Feature-Kategorien für UI
FEATURE_CATEGORIES = {
    "Basis OHLC": ["price_open", "price_high", "price_low", "price_close"],
    "Volumen": ["volume_sol", "buy_volume_sol", "sell_volume_sol", "net_volume_sol"],
    "Market Cap & Phase": ["market_cap_close", "phase_id_at_time"],
    "Dev-Tracking (Rug-Pull-Erkennung)": ["dev_sold_amount"],
    "Ratio-Metriken (Bot-Spam vs. echtes Interesse)": ["buy_pressure_ratio", "unique_signer_ratio"],
    "Whale-Aktivität": ["whale_buy_volume_sol", "whale_sell_volume_sol", "num_whale_buys", "num_whale_sells"],
    "Volatilität": ["volatility_pct", "avg_trade_size_sol"]
}

# Kritische Features (empfohlen für Rug-Detection)
CRITICAL_FEATURES = [
    "dev_sold_amount",  # Wichtigster Indikator!
    "buy_pressure_ratio",
    "unique_signer_ratio",
    "whale_buy_volume_sol",
    "volatility_pct",
    "net_volume_sol"
]

# ============================================================
# API Functions
# ============================================================

def api_get(endpoint: str, show_errors: bool = False) -> Any:
    """GET Request zur API (kann Dict oder List zurückgeben)"""
    try:
        response = httpx.get(f"{API_BASE_URL}{endpoint}", timeout=30.0)
        response.raise_for_status()
        return response.json()
    except httpx.HTTPError as e:
        if show_errors:
            st.error(f"❌ API-Fehler: {e}")
        return [] if 'comparisons' in endpoint or 'models' in endpoint else {}

def api_post(endpoint: str, data: Dict[str, Any], show_errors: bool = True) -> Optional[Dict[str, Any]]:
    """POST Request zur API"""
    try:
        response = httpx.post(f"{API_BASE_URL}{endpoint}", json=data, timeout=30.0)
        response.raise_for_status()
        return response.json()
    except httpx.HTTPError as e:
        if show_errors:
            st.error(f"❌ API-Fehler: {e}")
            if hasattr(e, 'response') and hasattr(e.response, 'text'):
                st.error(f"Details: {e.response.text}")
        return None

def api_delete(endpoint: str) -> bool:
    """DELETE Request zur API"""
    try:
        response = httpx.delete(f"{API_BASE_URL}{endpoint}", timeout=30.0)
        response.raise_for_status()
        return True
    except httpx.HTTPError as e:
        # Fehler wird nicht hier angezeigt, sondern in der aufrufenden Funktion
        # damit mehrere Löschungen nicht zu vielen Fehlermeldungen führen
        return False

def api_patch(endpoint: str, data: Dict[str, Any], show_errors: bool = True) -> Optional[Dict[str, Any]]:
    """PATCH Request zur API"""
    try:
        response = httpx.patch(f"{API_BASE_URL}{endpoint}", json=data, timeout=30.0)
        response.raise_for_status()
        return response.json()
    except httpx.HTTPError as e:
        if show_errors:
            st.error(f"❌ API-Fehler: {e}")
            if hasattr(e, 'response') and hasattr(e.response, 'text'):
                st.error(f"Details: {e.response.text}")
        return None

# ============================================================
# Feature Definitions
# ============================================================

# Verfügbare Features (aus coin_metrics)
# ⚠️ WICHTIG: Nur Spalten die tatsächlich in der Datenbank existieren!
AVAILABLE_FEATURES = [
    # Basis OHLC
    "price_open", "price_high", "price_low", "price_close",
    
    # Volumen
    "volume_sol", "buy_volume_sol", "sell_volume_sol", "net_volume_sol",
    
    # Market Cap & Phase
    "market_cap_close", "phase_id_at_time",
    
    # ⚠️ KRITISCH für Rug-Detection
    "dev_sold_amount",  # Wichtigster Indikator für Rug-Pulls!
    
    # Ratio-Metriken (Bot-Spam vs. echtes Interesse)
    "buy_pressure_ratio",
    "unique_signer_ratio",
    
    # Whale-Aktivität
    "whale_buy_volume_sol",
    "whale_sell_volume_sol",
    "num_whale_buys",
    "num_whale_sells",
    
    # Volatilität
    "volatility_pct",
    "avg_trade_size_sol"
]

# Feature-Kategorien für UI
FEATURE_CATEGORIES = {
    "Basis OHLC": ["price_open", "price_high", "price_low", "price_close"],
    "Volumen": ["volume_sol", "buy_volume_sol", "sell_volume_sol", "net_volume_sol"],
    "Market Cap & Phase": ["market_cap_close", "phase_id_at_time"],
    "Dev-Tracking (Rug-Pull-Erkennung)": ["dev_sold_amount"],
    "Ratio-Metriken (Bot-Spam vs. echtes Interesse)": ["buy_pressure_ratio", "unique_signer_ratio"],
    "Whale-Aktivität": ["whale_buy_volume_sol", "whale_sell_volume_sol", "num_whale_buys", "num_whale_sells"],
    "Volatilität": ["volatility_pct", "avg_trade_size_sol"]
}

# Kritische Features (empfohlen für Rug-Detection)
CRITICAL_FEATURES = [
    "dev_sold_amount",  # Wichtigster Indikator!
    "price_vs_ath_pct",  # ATH-Relative-Metrik
    "buy_pressure_ratio",
    "unique_signer_ratio"
]

# ============================================================
# Configuration Functions
# ============================================================

def load_phases() -> List[Dict[str, Any]]:
    """Lädt verfügbare Phasen aus der Datenbank"""
    phases = api_get("/api/phases", show_errors=False)
    if not phases or not isinstance(phases, list):
        return []
    return phases

def load_config():
    """Lädt Konfiguration aus YAML-Datei und merged mit aktuellen Umgebungsvariablen"""
    # Starte mit Defaults
    config = get_default_config()

    # Lade aus YAML-Datei falls vorhanden
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r') as f:
                yaml_config = yaml.safe_load(f) or {}
                config.update(yaml_config)
        except Exception as e:
            st.error(f"❌ Fehler beim Laden der Konfiguration: {e}")

    # Überschreibe mit aktuellen Umgebungsvariablen (höchste Priorität)
    env_vars = ['DB_DSN', 'API_BASE_URL', 'MODEL_STORAGE_PATH', 'API_PORT', 'STREAMLIT_PORT']
    for env_var in env_vars:
        env_value = os.getenv(env_var)
        if env_value:
            config[env_var] = env_value

    return config

def save_config(config):
    """Speichert Konfiguration in YAML-Datei"""
    try:
        os.makedirs(os.path.dirname(CONFIG_FILE), exist_ok=True)
        with open(CONFIG_FILE, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        return True
    except Exception as e:
        st.error(f"❌ Fehler beim Speichern der Konfiguration: {e}")
        return False

def get_default_config():
    """Gibt Standard-Konfiguration zurück"""
    return {
        "api_base_url": API_BASE_URL,
        "service_name": SERVICE_NAME,
        "coolify_mode": COOLIFY_MODE,
        "DB_DSN": os.getenv("DB_DSN", ""),
        "API_PORT": os.getenv("API_PORT", "8000"),
        "STREAMLIT_PORT": os.getenv("STREAMLIT_PORT", "8501"),
        "MODEL_STORAGE_PATH": os.getenv("MODEL_STORAGE_PATH", "/app/models"),
        "JOB_POLL_INTERVAL": os.getenv("JOB_POLL_INTERVAL", "5"),
        "MAX_CONCURRENT_JOBS": os.getenv("MAX_CONCURRENT_JOBS", "2")
    }

def validate_url(url, allow_empty=False):
    """Validiert URL-Format"""
    if allow_empty and not url:
        return True, None
    if not url:
        return False, "URL darf nicht leer sein"
    try:
        result = urlparse(url)
        if not result.scheme or not result.netloc:
            return False, "Ungültige URL-Format"
        if result.scheme not in ["http", "https", "postgresql"]:
            return False, f"Nicht unterstütztes Protokoll: {result.scheme}"
        return True, None
    except Exception as e:
        return False, f"URL-Validierung fehlgeschlagen: {str(e)}"

def validate_port(port):
    """Validiert Port-Nummer"""
    if not port:
        return False, "Port darf nicht leer sein"
    try:
        port_int = int(port)
        if 1 <= port_int <= 65535:
            return True, None
        else:
            return False, f"Port muss zwischen 1 und 65535 liegen (ist: {port_int})"
    except ValueError:
        return False, f"Port muss eine Zahl sein (ist: {port})"

def reload_config():
    """Lädt die Konfiguration im Service neu (ohne Neustart)"""
    try:
        import httpx
        response = httpx.post(f"{API_BASE_URL}/api/reload-config", timeout=5)
        if response.status_code == 200:
            data = response.json()
            return True, data.get("message", "Konfiguration wurde neu geladen")
        else:
            return False, f"Fehler: HTTP {response.status_code}"
    except Exception as e:
        return False, f"Fehler beim Neuladen: {str(e)}"

# ============================================================
# Service Management Functions
# ============================================================

def restart_service():
    """Startet Service neu"""
    if COOLIFY_MODE:
        success, message = reload_config()
        if success:
            return True, f"✅ {message} (ohne Neustart - funktioniert in Coolify!)"
        else:
            return False, f"⚠️ Coolify-Modus: {message}. Bitte Service im Coolify-Dashboard neu starten."
    
    try:
        # In Docker wird der Service über Docker Compose neu gestartet
        result = subprocess.run(
            ["docker", "compose", "restart", SERVICE_NAME],
            capture_output=True,
            text=True,
            timeout=30
        )
        if result.returncode == 0:
            return True, "✅ Service wird neu gestartet..."
        else:
            return False, f"❌ Fehler beim Neustart: {result.stderr}"
    except Exception as e:
        return False, f"❌ Fehler beim Neustart: {e}"

def _restart_via_subprocess():
    """Interne Funktion für Service-Neustart"""
    if not COOLIFY_MODE:
        return False
    
    try:
        subprocess.run(
            ["docker", "compose", "restart", SERVICE_NAME],
            capture_output=True,
            timeout=30
        )
        return True
    except:
        return False

def get_service_logs(lines=100):
    """Holt Service-Logs"""
    if not COOLIFY_MODE:
        return "⚠️ Logs nur im Coolify-Modus verfügbar"
    
    try:
        result = subprocess.run(
            ["docker", "compose", "logs", "--tail", str(lines), SERVICE_NAME],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode == 0:
            return result.stdout
        else:
            return f"❌ Fehler beim Abrufen der Logs: {result.stderr}"
    except Exception as e:
        return f"❌ Fehler: {e}"

