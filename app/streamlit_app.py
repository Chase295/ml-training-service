"""
Streamlit UI für ML Training Service
Web-Interface für Modell-Management mit Tab-basiertem Layout
"""
import streamlit as st
import os
import httpx
import pandas as pd
import plotly.express as px
import yaml
import json
import time
import subprocess
import re
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any, Optional
from pathlib import Path
from urllib.parse import urlparse

# Konfiguration
# ⚠️ WICHTIG: Innerhalb des Containers muss auf Port 8000 zugegriffen werden (interner Port)
# Von außen ist die API auf Port 8012 erreichbar (externer Port)
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
CONFIG_FILE = "/app/config/config.yaml"
ENV_FILE = "/app/config/.env"
SERVICE_NAME = os.getenv("SERVICE_NAME", "ml-training-service")
COOLIFY_MODE = os.getenv("COOLIFY_MODE", "false").lower() == "true"

# Page Config
st.set_page_config(
    page_title="ML Training Service - Control Panel",
    page_icon="🤖",
    layout="wide"
)

# ============================================================
# Helper Functions
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

# Verfügbare Features (aus coin_metrics)
# ⚠️ WICHTIG: Nur Spalten die tatsächlich in der Datenbank existieren!
# Diese Liste muss mit den tatsächlichen Spalten in coin_metrics übereinstimmen!
# ⚠️ HINWEIS: market_cap_open, market_cap_high, market_cap_low existieren NICHT!
# Nur market_cap_close ist verfügbar!
# Verfügbare Features aus coin_metrics (kategorisiert)
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

# Verfügbare Target-Variablen
AVAILABLE_TARGETS = [
    "market_cap_close", "price_close", "volume_sol"  # volume_usd existiert nicht!
]

def load_phases() -> List[Dict[str, Any]]:
    """Lade Phasen aus der API"""
    phases_data = api_get("/api/phases")
    if isinstance(phases_data, list):
        return phases_data
    return []

# ============================================================
# Konfigurationssystem (wie in vorherigen Services)
# ============================================================

def load_config():
    """Lädt Konfiguration aus YAML-Datei oder .env"""
    config = {}
    
    # Lade aus Environment Variables
    env_vars = {
        'DB_DSN': os.getenv('DB_DSN'),
        'API_PORT': os.getenv('API_PORT', '8000'),
        'STREAMLIT_PORT': os.getenv('STREAMLIT_PORT', '8501'),
        'MODEL_STORAGE_PATH': os.getenv('MODEL_STORAGE_PATH', '/app/models'),
        'API_BASE_URL': os.getenv('API_BASE_URL', 'http://localhost:8000'),  # Interner Port
        'JOB_POLL_INTERVAL': os.getenv('JOB_POLL_INTERVAL', '5'),
        'MAX_CONCURRENT_JOBS': os.getenv('MAX_CONCURRENT_JOBS', '2'),
        'LOG_LEVEL': os.getenv('LOG_LEVEL', 'INFO'),
        'LOG_FORMAT': os.getenv('LOG_FORMAT', 'text'),
        'LOG_JSON_INDENT': os.getenv('LOG_JSON_INDENT', '0'),
    }
    
    for key, value in env_vars.items():
        if value is not None:
            if value.isdigit():
                config[key] = int(value)
            else:
                try:
                    config[key] = float(value)
                except:
                    config[key] = value
    
    # Lade aus .env Datei
    env_paths = ["/app/config/.env", "/app/.env", "/app/../.env", ".env"]
    for env_path in env_paths:
        if os.path.exists(env_path):
            try:
                with open(env_path, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#') and '=' in line:
                            key, value = line.split('=', 1)
                            key = key.strip()
                            value = value.strip().strip('"').strip("'")
                            if value:
                                if value.isdigit():
                                    config[key] = int(value)
                                else:
                                    try:
                                        config[key] = float(value)
                                    except:
                                        config[key] = value
                break
            except Exception:
                continue
    
    # Lade aus YAML-Datei
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r') as f:
                file_config = yaml.safe_load(f)
                if file_config:
                    config.update(file_config)
        except Exception:
            pass
    
    if not config:
        return get_default_config()
    
    return config

def save_config(config):
    """Speichert Konfiguration in YAML-Datei UND .env Datei"""
    try:
        os.makedirs(os.path.dirname(CONFIG_FILE), exist_ok=True)
    except (OSError, PermissionError):
        raise
    
    try:
        with open(CONFIG_FILE, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
    except (OSError, PermissionError) as e:
        raise OSError(f"Config-Datei kann nicht geschrieben werden: {e}")
    
    env_content = f"""# ============================================================================
# ML TRAINING SERVICE - Umgebungsvariablen
# ============================================================================
DB_DSN={config.get('DB_DSN', 'postgresql://user:pass@localhost:5432/crypto')}
API_PORT={config.get('API_PORT', 8000)}
STREAMLIT_PORT={config.get('STREAMLIT_PORT', 8501)}
MODEL_STORAGE_PATH={config.get('MODEL_STORAGE_PATH', '/app/models')}
API_BASE_URL={config.get('API_BASE_URL', 'http://localhost:8000')}
JOB_POLL_INTERVAL={config.get('JOB_POLL_INTERVAL', 5)}
MAX_CONCURRENT_JOBS={config.get('MAX_CONCURRENT_JOBS', 2)}
LOG_LEVEL={config.get('LOG_LEVEL', 'INFO')}
LOG_FORMAT={config.get('LOG_FORMAT', 'text')}
LOG_JSON_INDENT={config.get('LOG_JSON_INDENT', 0)}
"""
    try:
        os.makedirs(os.path.dirname(ENV_FILE), exist_ok=True)
        with open(ENV_FILE, 'w') as f:
            f.write(env_content)
    except (OSError, PermissionError):
        pass
    
    return True

def get_default_config():
    """Gibt Standard-Konfiguration zurück"""
    return {
        "DB_DSN": "postgresql://user:pass@localhost:5432/crypto",
        "API_PORT": 8000,
        "STREAMLIT_PORT": 8501,
        "MODEL_STORAGE_PATH": "/app/models",
        "API_BASE_URL": "http://localhost:8000",  # Interner Port (Container-intern)
        "JOB_POLL_INTERVAL": 5,
        "MAX_CONCURRENT_JOBS": 2,
        "LOG_LEVEL": "INFO",
        "LOG_FORMAT": "text",
        "LOG_JSON_INDENT": 0
    }

def validate_url(url, allow_empty=False):
    """Validiert eine URL"""
    if allow_empty and not url:
        return True, None
    if not url:
        return False, "URL darf nicht leer sein"
    try:
        result = urlparse(url)
        if not result.scheme or not result.netloc:
            return False, "Ungültige URL-Format"
        if result.scheme not in ["http", "https", "postgresql"]:
            return False, f"Ungültiges Protokoll: {result.scheme}"
        return True, None
    except Exception as e:
        return False, f"URL-Validierungsfehler: {str(e)}"

def validate_port(port):
    """Validiert einen Port"""
    try:
        port_int = int(port)
        if 1 <= port_int <= 65535:
            return True, None
        return False, "Port muss zwischen 1 und 65535 liegen"
    except ValueError:
        return False, "Port muss eine Zahl sein"

def reload_config():
    """Lädt die Konfiguration im Service neu (ohne Neustart)"""
    try:
        response = httpx.post(f"{API_BASE_URL}/api/reload-config", timeout=5)
        if response.status_code == 200:
            data = response.json()
            return True, data.get("message", "Konfiguration wurde neu geladen")
        else:
            return False, f"Fehler: HTTP {response.status_code}"
    except Exception as e:
        return False, f"Fehler beim Neuladen: {str(e)}"

def restart_service():
    """Startet Service neu (über Docker API, damit .env neu geladen wird)"""
    # Coolify-Modus: Versuche zuerst Config-Neuladen über API
    if COOLIFY_MODE:
        success, message = reload_config()
        if success:
            return True, f"✅ {message} (ohne Neustart - funktioniert in Coolify!)"
        else:
            return False, f"⚠️ Coolify-Modus: {message}. Falls das nicht funktioniert, starte den Service im Coolify-Dashboard neu."
    
    try:
        import docker
        client = docker.from_env()
        
        # Versuche verschiedene Container-Namen
        container_names = [SERVICE_NAME, "ml-training-service", "ml-training"]
        container = None
        for name in container_names:
            try:
                container = client.containers.get(name)
                break
            except docker.errors.NotFound:
                continue
        
        # Falls nicht gefunden: Suche nach Containern mit "ml-training" im Namen
        if not container:
            try:
                all_containers = client.containers.list(all=True)
                for cont in all_containers:
                    if "ml-training" in cont.name.lower() or "training" in cont.name.lower():
                        container = cont
                        break
            except Exception:
                pass
        
        if not container:
            return False, "Container 'ml-training-service' nicht gefunden. Bitte prüfe ob der Service läuft."
        
        # Stoppe Container
        container.stop(timeout=10)
        
        # Starte Container neu (lädt .env neu)
        container.start()
        
        return True, "✅ Service erfolgreich neu gestartet! Neue Environment Variables werden geladen."
        
    except ImportError:
        # Docker Python Client nicht verfügbar - versuche über Docker Socket direkt
        return _restart_via_subprocess()
    except Exception as e:
        return _restart_via_subprocess()

def _restart_via_subprocess():
    """Versucht Service über subprocess neu zu starten"""
    try:
        import subprocess
        import os
        
        # Prüfe ob docker compose verfügbar ist
        docker_compose_cmd = None
        for cmd in ["docker", "docker-compose"]:
            try:
                result = subprocess.run(
                    [cmd, "--version"],
                    capture_output=True,
                    timeout=5
                )
                if result.returncode == 0:
                    docker_compose_cmd = cmd
                    break
            except:
                continue
        
        if not docker_compose_cmd:
            return False, "Docker/Docker Compose nicht gefunden. Bitte manuell neu starten: docker compose restart ml-training"
        
        # Versuche über Docker Socket zu arbeiten
        # Finde das Projekt-Verzeichnis (wo docker-compose.yml ist)
        compose_file = "/app/../docker-compose.yml"
        if not os.path.exists(compose_file):
            compose_file = "/app/docker-compose.yml"
        
        if os.path.exists(compose_file):
            work_dir = os.path.dirname(compose_file)
            result = subprocess.run(
                [docker_compose_cmd, "restart", "ml-training"],
                cwd=work_dir,
                capture_output=True,
                text=True,
                timeout=30
            )
            if result.returncode == 0:
                return True, "✅ Service neu gestartet (via docker compose)"
            else:
                return False, f"❌ Docker Compose Fehler: {result.stderr}"
        else:
            return False, "⚠️ docker-compose.yml nicht gefunden. Bitte manuell neu starten: `docker compose restart ml-training`"
            
    except Exception as e:
        return False, f"❌ Fehler: {str(e)}. Bitte manuell neu starten: `docker compose restart ml-training`"

def get_service_logs(lines=100):
    """Holt Logs vom Service"""
    try:
        import docker
        client = docker.from_env()
        container_names = [SERVICE_NAME, "ml-training-service", "ml-training"]
        container = None
        for name in container_names:
            try:
                container = client.containers.get(name)
                break
            except:
                continue
        
        if container:
            logs = container.logs(tail=lines, timestamps=True).decode('utf-8')
            log_lines = logs.strip().split('\n')
            return '\n'.join(reversed(log_lines))
        else:
            return "❌ Container nicht gefunden"
    except ImportError:
        try:
            import subprocess
            compose_paths = ["/app/../docker-compose.yml", "/app/docker-compose.yml"]
            work_dir = "/app"
            for path in compose_paths:
                if os.path.exists(path):
                    work_dir = os.path.dirname(path)
                    break
            
            result = subprocess.run(
                ["docker", "compose", "logs", "--tail", str(lines), "ml-training"],
                cwd=work_dir,
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0:
                log_lines = result.stdout.strip().split('\n')
                return '\n'.join(reversed(log_lines))
            else:
                return f"Fehler: {result.stderr}"
        except Exception as e:
            return f"Fehler: {str(e)}"
    except Exception as e:
        return f"Fehler: {str(e)}"

# ============================================================
# Seiten
# ============================================================

def page_overview():
    """Übersicht: Liste aller Modelle"""
    st.title("🏠 Übersicht - ML Modelle")
    
    # Initialisiere selected_models in session_state
    if 'selected_model_ids' not in st.session_state:
        st.session_state['selected_model_ids'] = []
    
    # Lade Modelle ZUERST (wird für "Alle auswählen" benötigt)
    models = api_get("/api/models")
    if not models:
        st.warning("⚠️ Keine Modelle gefunden oder API-Fehler")
        return
    
    # Filter
    col1, col2, col3 = st.columns(3)
    with col1:
        status_filter = st.selectbox(
            "Status Filter",
            ["Alle", "READY", "TRAINING", "FAILED"],
            key="status_filter"
        )
    with col2:
        model_type_filter = st.selectbox(
            "Modell-Typ Filter",
            ["Alle", "random_forest", "xgboost"],
            key="model_type_filter"
        )
    
    # Filter anwenden
    filtered_models = models
    if status_filter != "Alle":
        filtered_models = [m for m in filtered_models if m.get('status') == status_filter]
    if model_type_filter != "Alle":
        filtered_models = [m for m in filtered_models if m.get('model_type') == model_type_filter]
    
    st.info(f"📊 {len(filtered_models)} Modell(e) gefunden")
    
    # Kompakte Karten-Ansicht
    if filtered_models:
        st.subheader("📋 Modelle")
        
        # Erstelle Karten in einem Grid (2 Spalten)
        cols = st.columns(2)
        
        for idx, model in enumerate(filtered_models):
            model_id = model.get('id')
            model_name = model.get('name', f"ID: {model_id}")
            model_type = model.get('model_type', 'N/A')
            status = model.get('status', 'N/A')
            accuracy = model.get('training_accuracy')
            f1 = model.get('training_f1')
            created_raw = model.get('created_at', '')
            train_start = model.get('train_start')
            train_end = model.get('train_end')
            
            # Checkbox
            is_selected = model_id in st.session_state.get('selected_model_ids', [])
            checkbox_key = f"checkbox_{model_id}"
            
            # Wähle Spalte (abwechselnd)
            col = cols[idx % 2]
            
            with col:
                # Karte mit Border
                card_style = """
                <style>
                .model-card {
                    border: 2px solid #e0e0e0;
                    border-radius: 8px;
                    padding: 12px;
                    margin-bottom: 12px;
                    background: white;
                }
                .model-card.selected {
                    border-color: #1f77b4;
                    background: #f0f8ff;
                }
                </style>
                """
                st.markdown(card_style, unsafe_allow_html=True)
                
                # Header mit Checkbox und Name
                header_col1, header_col2, header_col3, header_col4 = st.columns([0.3, 4, 0.6, 0.6])
                with header_col1:
                    checked = st.checkbox("", value=is_selected, key=checkbox_key, label_visibility="collapsed")
                    # Update session_state ohne st.rerun() - Streamlit rendert automatisch neu
                    if checked and model_id not in st.session_state.get('selected_model_ids', []):
                        if 'selected_model_ids' not in st.session_state:
                            st.session_state['selected_model_ids'] = []
                        st.session_state['selected_model_ids'].append(model_id)
                    elif not checked and model_id in st.session_state.get('selected_model_ids', []):
                        st.session_state['selected_model_ids'].remove(model_id)
                
                with header_col2:
                    # Name mit Umbenennen
                    if st.session_state.get(f'renaming_{model_id}', False):
                        # Popup-ähnliches Verhalten mit Expander
                        with st.expander("✏️ Modell bearbeiten", expanded=True):
                            new_name = st.text_input("Name *", value=model_name, key=f"new_name_{model_id}")
                            new_desc = st.text_area("Beschreibung", value=model.get('description', '') or '', key=f"new_desc_{model_id}", height=80)
                            
                            # Buttons nebeneinander ohne verschachtelte Spalten
                            if st.button("💾 Speichern", key=f"save_{model_id}", use_container_width=True, type="primary"):
                                if new_name and new_name.strip():
                                    data = {"name": new_name.strip()}
                                    if new_desc and new_desc.strip():
                                        data["description"] = new_desc.strip()
                                    result = api_patch(f"/api/models/{model_id}", data)
                                    if result:
                                        st.session_state[f'renaming_{model_id}'] = False
                                        st.success("✅ Modell erfolgreich umbenannt")
                                        st.rerun()
                                else:
                                    st.warning("⚠️ Name darf nicht leer sein")
                            
                            if st.button("❌ Abbrechen", key=f"cancel_{model_id}", use_container_width=True):
                                st.session_state[f'renaming_{model_id}'] = False
                                st.rerun()
                    else:
                        st.markdown(f"**{model_name}**")
                
                with header_col3:
                    if st.button("📋", key=f"details_{model_id}", help="Details anzeigen", use_container_width=True):
                        st.session_state['details_model_id'] = model_id
                        st.session_state['page'] = 'details'
                        st.rerun()
                
                with header_col4:
                    if not st.session_state.get(f'renaming_{model_id}', False):
                        if st.button("✏️", key=f"rename_{model_id}", help="Umbenennen", use_container_width=True):
                            st.session_state[f'renaming_{model_id}'] = True
                            st.rerun()
                
                # Kompakte Info-Zeile
                info_col1, info_col2, info_col3 = st.columns(3)
                
                with info_col1:
                    type_emoji = "🌲" if model_type == "random_forest" else "🚀" if model_type == "xgboost" else "🤖"
                    st.caption(f"{type_emoji} {model_type}")
                
                with info_col2:
                    if status == "READY":
                        st.caption("✅ READY")
                    elif status == "TRAINING":
                        st.caption("🔄 TRAINING")
                    else:
                        st.caption(f"❌ {status}")
                
                with info_col3:
                    st.caption(f"#{model_id}")
                
                # Metriken kompakt - Erweitert
                metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
                with metric_col1:
                    if accuracy:
                        st.metric("Accuracy", f"{accuracy:.3f}", help="Anteil korrekter Vorhersagen (0-1, höher = besser)", label_visibility="visible")
                    else:
                        st.caption("Accuracy: N/A")
                with metric_col2:
                    if f1:
                        st.metric("F1-Score", f"{f1:.3f}", help="Harmonisches Mittel aus Precision und Recall (0-1, höher = besser)", label_visibility="visible")
                    else:
                        st.caption("F1-Score: N/A")
                with metric_col3:
                    roc_auc = model.get('roc_auc')
                    if roc_auc:
                        st.metric("ROC-AUC", f"{roc_auc:.3f}", help="Area Under ROC Curve (0-1, >0.7 = gut)", label_visibility="visible")
                    else:
                        st.caption("ROC-AUC: N/A")
                with metric_col4:
                    mcc = model.get('mcc')
                    if mcc:
                        st.metric("MCC", f"{mcc:.3f}", help="Matthews Correlation Coefficient (-1 bis +1, höher = besser)", label_visibility="visible")
                    else:
                        st.caption("MCC: N/A")
                
                # Zusätzliche Infos
                info_row1, info_row2 = st.columns(2)
                
                with info_row1:
                    # Features
                    features_list = model.get('features', [])
                    if features_list:
                        num_features = len(features_list)
                        st.caption(f"📊 {num_features} Features")
                    
                    # Zeitbasierte Vorhersage Info - DEUTLICH SICHTBAR
                    # Prüfe zuerst direkte Felder aus der DB
                    future_minutes = model.get('future_minutes')
                    price_change = model.get('price_change_percent') or model.get('min_percent_change')
                    direction = model.get('target_direction')
                    
                    # Falls nicht in direkten Feldern, prüfe in params
                    if not future_minutes or not price_change:
                        params = model.get('params', {})
                        if isinstance(params, str):
                            import json
                            try:
                                params = json.loads(params)
                            except:
                                params = {}
                        
                        time_based_params = params.get('_time_based', {})
                        if time_based_params and time_based_params.get('enabled'):
                            future_minutes = future_minutes or time_based_params.get('future_minutes')
                            price_change = price_change or time_based_params.get('min_percent_change')
                            direction = direction or time_based_params.get('direction')
                    
                    # Zeige zeitbasierte Vorhersage wenn vorhanden
                    if future_minutes and price_change:
                        direction_emoji = "📈" if direction == "up" else "📉" if direction == "down" else ""
                        direction_text = "steigt" if direction == "up" else "fällt" if direction == "down" else ""
                        # Größere, sichtbarere Anzeige
                        st.markdown(f"**⏰ Zeitbasierte Vorhersage:** {future_minutes}min, {price_change}% {direction_text} {direction_emoji}")
                    else:
                        # Normale Ziel-Variable
                        target_var = model.get('target_variable', 'N/A')
                        target_operator = model.get('target_operator')
                        target_value = model.get('target_value')
                        if target_operator and target_value is not None:
                            st.markdown(f"**🎯 Ziel:** {target_var} {target_operator} {target_value}")
                        else:
                            st.caption("🎯 Ziel: Nicht konfiguriert")
                    
                    # Feature-Engineering Info
                    params = model.get('params', {})
                    if isinstance(params, str):
                        import json
                        try:
                            params = json.loads(params)
                        except:
                            params = {}
                    if params.get('use_engineered_features'):
                        st.caption("🔧 Feature-Engineering: ✅")
                
                with info_row2:
                    # Trainingszeitraum mit Uhrzeit
                    if train_start and train_end:
                        try:
                            start_dt = train_start if isinstance(train_start, str) else train_start
                            end_dt = train_end if isinstance(train_end, str) else train_end
                            if isinstance(start_dt, str):
                                start_dt = datetime.fromisoformat(start_dt.replace('Z', '+00:00'))
                            if isinstance(end_dt, str):
                                end_dt = datetime.fromisoformat(end_dt.replace('Z', '+00:00'))
                            
                            start_str = start_dt.strftime("%d.%m.%Y %H:%M")
                            end_str = end_dt.strftime("%d.%m.%Y %H:%M")
                            days = (end_dt - start_dt).total_seconds() / 86400.0
                            st.caption(f"📅 Training: {start_str} → {end_str} ({days:.1f} Tage)")
                        except:
                            st.caption("📅 Training: Zeitraum verfügbar")
                    else:
                        st.caption("📅 Training: Zeitraum nicht verfügbar")
                    
                    # Erstellt-Datum mit Uhrzeit
                    if created_raw:
                        try:
                            if isinstance(created_raw, str):
                                created_dt = datetime.fromisoformat(created_raw.replace('Z', '+00:00'))
                            else:
                                created_dt = created_raw
                            created_str = created_dt.strftime("%d.%m.%Y %H:%M")
                            st.caption(f"🕐 Erstellt: {created_str}")
                        except:
                            st.caption(f"🕐 Erstellt: {str(created_raw)[:19] if len(str(created_raw)) > 19 else str(created_raw)}")
                    else:
                        st.caption("🕐 Erstellt: N/A")
                
                # Dünne graue Linie zur Trennung
                if idx < len(filtered_models) - 1:
                    st.markdown("<hr style='margin: 10px 0; border: none; border-top: 1px solid #e0e0e0;'>", unsafe_allow_html=True)
        
        # Zeige ausgewählte Modelle
        selected_model_ids = st.session_state.get('selected_model_ids', [])
        # Filtere nur existierende Modelle
        selected_model_ids = [mid for mid in selected_model_ids if any(m.get('id') == mid for m in filtered_models)]
        # Aktualisiere session_state falls Modelle entfernt wurden
        if len(selected_model_ids) != len(st.session_state.get('selected_model_ids', [])):
            st.session_state['selected_model_ids'] = selected_model_ids
        
        selected_count = len(selected_model_ids)
        if selected_count > 0:
            st.divider()
            st.subheader(f"🔧 Aktionen ({selected_count} Modell(e) ausgewählt)")
            
            selected_models = [m for m in filtered_models if m.get('id') in selected_model_ids]
            
            # Zeige ausgewählte Modelle
            if selected_count <= 3:
                selected_names = [f"{m.get('name')} (ID: {m.get('id')})" for m in selected_models]
                st.info(f"📌 Ausgewählt: {', '.join(selected_names)}")
            
            # Aktionen basierend auf Anzahl
            if selected_count == 1:
                # 1 Modell: Testen, Löschen, Details
                model_id = selected_model_ids[0]
                selected_model = selected_models[0]
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    test_clicked = st.button("🧪 Testen", key="btn_test", use_container_width=True)
                    if test_clicked:
                        st.session_state['test_model_id'] = model_id
                        st.session_state['page'] = 'test'
                        st.rerun()
                
                with col2:
                    details_clicked = st.button("📋 Details", key="btn_details", use_container_width=True)
                    if details_clicked:
                        st.session_state['details_model_id'] = model_id
                        st.session_state['page'] = 'details'
                        st.rerun()
                
                with col3:
                    download_clicked = st.button("📥 Download", key="btn_download", use_container_width=True)
                    if download_clicked:
                        download_url = f"{API_BASE_URL}/api/models/{model_id}/download"
                        st.markdown(f"[⬇️ Modell herunterladen]({download_url})")
                
                with col4:
                    delete_clicked = st.button("🗑️ Löschen", key="btn_delete", use_container_width=True, type="secondary")
                    if delete_clicked:
                        if api_delete(f"/api/models/{model_id}"):
                            st.success("✅ Modell gelöscht")
                            # Entferne aus session_state
                            if model_id in st.session_state.get('selected_model_ids', []):
                                st.session_state['selected_model_ids'].remove(model_id)
                            st.rerun()
                        else:
                            st.error("❌ Fehler beim Löschen des Modells")
            
            elif selected_count == 2:
                # 2 Modelle: Vergleichen, Löschen
                model_a_id = selected_model_ids[0]
                model_b_id = selected_model_ids[1]
                
                col1, col2 = st.columns(2)
                
                with col1:
                    if st.button("⚔️ Vergleichen", key="btn_compare", use_container_width=True, type="primary"):
                        st.session_state['compare_model_a_id'] = model_a_id
                        st.session_state['compare_model_b_id'] = model_b_id
                        st.session_state['page'] = 'compare'
                        st.rerun()
            
                with col2:
                    if st.button("🗑️ Beide löschen", key="btn_delete_both", use_container_width=True, type="secondary"):
                        # Lösche beide Modelle
                        ids_to_delete = list(selected_model_ids)
                        deleted_count = 0
                        failed_count = 0
                        
                        for model_id in ids_to_delete:
                            if api_delete(f"/api/models/{model_id}"):
                                deleted_count += 1
                                if model_id in st.session_state.get('selected_model_ids', []):
                                    st.session_state['selected_model_ids'].remove(model_id)
                            else:
                                failed_count += 1
                        
                        if deleted_count > 0:
                            st.success(f"✅ {deleted_count} Modell(e) gelöscht")
                        if failed_count > 0:
                            st.error(f"❌ {failed_count} Fehler")
                        
                        if deleted_count > 0:
                            st.rerun()
            
            else:
                # Mehr als 2: Nur Löschen
                if st.button("🗑️ Alle ausgewählten löschen", key="btn_delete_all", use_container_width=True, type="secondary"):
                    deleted_count = 0
                    failed_count = 0
                    # Kopiere Liste um während Iteration zu ändern - verwende die tatsächliche Liste aus session_state
                    ids_to_delete = list(st.session_state.get('selected_model_ids', []))
                    for model_id in ids_to_delete:
                        if api_delete(f"/api/models/{model_id}"):
                            deleted_count += 1
                            # Entferne aus session_state
                            if model_id in st.session_state.get('selected_model_ids', []):
                                st.session_state['selected_model_ids'].remove(model_id)
                        else:
                            failed_count += 1
                    
                    # Immer rerun wenn etwas passiert ist
                    if deleted_count > 0 or failed_count > 0:
                        if deleted_count > 0:
                            if failed_count > 0:
                                st.warning(f"⚠️ {deleted_count} Modell(e) gelöscht, {failed_count} Fehler")
                            else:
                                st.success(f"✅ {deleted_count} Modell(e) gelöscht")
                        if failed_count > 0 and deleted_count == 0:
                            st.error(f"❌ Fehler beim Löschen von {failed_count} Modell(en)")
                        st.rerun()
        else:
            st.info("💡 Wähle ein oder mehrere Modelle aus, um Aktionen auszuführen")
    else:
        st.info("ℹ️ Keine Modelle gefunden")

def page_train():
    """Neues Modell trainieren - ÜBERARBEITETE VERSION"""
    st.title("🚀 Neues Modell erstellen")
    
    st.info("""
    **📖 Schnellstart:** 
    Fülle die minimalen Felder aus und klicke auf "Modell trainieren".
    Erweiterte Optionen findest du im ausklappbaren Bereich unten.
    """)
    
    # Initialisiere session_state für Features (einmalig, außerhalb des Forms)
    if 'train_features_initialized' not in st.session_state:
        for category, features_in_category in FEATURE_CATEGORIES.items():
            for feature in features_in_category:
                if f"feature_{feature}" not in st.session_state:
                    st.session_state[f"feature_{feature}"] = True  # Default: aktiviert
        st.session_state['train_features_initialized'] = True
    
    # Lade verfügbare Daten (einmalig, außerhalb des Forms)
    data_availability = api_get("/api/data-availability")
    min_timestamp = data_availability.get('min_timestamp') if data_availability else None
    max_timestamp = data_availability.get('max_timestamp') if data_availability else None
    
    min_date = None
    max_date = None
    min_datetime = None
    max_datetime = None
    
    if min_timestamp and max_timestamp:
        try:
            min_datetime = datetime.fromisoformat(min_timestamp.replace('Z', '+00:00'))
            max_datetime = datetime.fromisoformat(max_timestamp.replace('Z', '+00:00'))
            min_date = min_datetime.date()
            max_date = max_datetime.date()
            st.info(f"📊 **Verfügbare Daten:** Von {min_datetime.strftime('%d.%m.%Y %H:%M')} bis {max_datetime.strftime('%d.%m.%Y %H:%M')}")
        except Exception as e:
            st.warning(f"⚠️ Konnte Datumsbereich nicht parsen: {e}")
    else:
        st.warning("⚠️ Keine Trainingsdaten in der Datenbank gefunden!")
    
    # ============================================================
    # FORMULAR
    # ============================================================
    with st.form("train_model_form", clear_on_submit=False):
        # Basis-Informationen
        st.subheader("📝 Basis-Informationen")
        model_name = st.text_input("Modell-Name *", placeholder="z.B. PumpDetector_v1", key="train_model_name")
        model_type = st.selectbox(
            "Modell-Typ *",
            ["random_forest", "xgboost"],
            index=0,
            help="Random Forest: Robust, schnell. XGBoost: Beste Performance",
            key="train_model_type"
        )
        
        st.divider()
        
        # Training-Zeitraum
        st.subheader("📅 Training-Zeitraum")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**🕐 Start-Zeitpunkt**")
            if min_date:
                default_start_date = min_date
            else:
                default_start_date = datetime.now(timezone.utc).date() - timedelta(days=30)
            
            train_start_date = st.date_input(
                "Start-Datum *",
                value=default_start_date,
                min_value=min_date,
                max_value=max_date,
                key="train_start_date"
            )
            
            if min_datetime and train_start_date == min_date:
                default_start_time = min_datetime.time()
            else:
                default_start_time = datetime.now(timezone.utc).time().replace(hour=0, minute=0, second=0, microsecond=0)
            
            train_start_time = st.time_input(
                "Start-Uhrzeit *",
                value=default_start_time,
                key="train_start_time"
            )
        
        with col2:
            st.markdown("**🕐 Ende-Zeitpunkt**")
            if max_date:
                default_end_date = max_date
            else:
                default_end_date = datetime.now(timezone.utc).date()
            
            train_end_date = st.date_input(
                "Ende-Datum *",
                value=default_end_date,
                min_value=train_start_date if train_start_date else None,
                max_value=max_date,
                key="train_end_date"
            )
            
            if max_datetime and train_end_date == max_date:
                default_end_time = max_datetime.time()
            else:
                default_end_time = datetime.now(timezone.utc).time().replace(hour=23, minute=59, second=59, microsecond=0)
            
            train_end_time = st.time_input(
                "Ende-Uhrzeit *",
                value=default_end_time,
                key="train_end_time"
            )
        
        st.divider()
        
        # Vorhersage-Ziel (zeitbasiert - Standard)
        st.subheader("⏰ Vorhersage-Ziel")
        st.caption("Das Modell lernt: 'Steigt/Fällt die Variable in X Minuten um Y%?'")
        
        time_based_target_var = st.selectbox(
            "Variable überwachen *",
            AVAILABLE_TARGETS, 
            index=0,
            help="Welche Variable soll für die prozentuale Änderung verwendet werden?",
            key="train_target_var"
        )
        
        col1, col2, col3 = st.columns(3)
        with col1:
            future_minutes = st.number_input(
                "Zeitraum (Minuten) *", 
                min_value=1, 
                max_value=60,
                value=10,
                step=1,
                help="In wie vielen Minuten soll die Änderung stattfinden?",
                key="train_future_minutes"
            )
        with col2:
            min_percent_change = st.number_input(
                "Mindest-Änderung (%) *",
                min_value=0.1, 
                max_value=1000.0,
                value=5.0,
                step=0.5,
                help="Mindest-Prozentuale Änderung",
                key="train_min_percent"
            )
        with col3:
            direction = st.selectbox(
                "Richtung *",
                ["up", "down"],
                format_func=lambda x: "Steigt" if x == "up" else "Fällt",
                help="Steigt oder fällt die Variable?",
                key="train_direction"
            )
        
        # Label-Erstellung Transparenz
        st.info(f"""
        📊 **Label-Erstellung:**
        
        Für jede Zeile in den Trainingsdaten wird geprüft:
        
        1. **Aktueller Wert**: `{time_based_target_var}` zum Zeitpunkt T
        2. **Zukünftiger Wert**: `{time_based_target_var}` zum Zeitpunkt T + {future_minutes} Minuten
        3. **Prozentuale Änderung**: `((Zukunft - Aktuell) / Aktuell) * 100`
        
        **Label = 1** wenn:
        - Änderung >= {min_percent_change}% (bei "Steigt")
        - Änderung <= -{min_percent_change}% (bei "Fällt")
        
        **Label = 0** wenn:
        - Bedingung nicht erfüllt
        
        **Beispiel:**
        - Aktuell: 100 SOL
        - Zukunft ({future_minutes} Min): 106 SOL
        - Änderung: +6%
        - **Label = 1** ✅ (weil 6% >= {min_percent_change}%)
        """)
        
        st.divider()
        
        # ============================================================
        # ERWEITERTE OPTIONEN (ausklappbar)
        # ============================================================
        with st.expander("⚙️ Erweiterte Optionen", expanded=False):
            # Feature-Auswahl mit Kategorien
            st.subheader("📊 Features")
            st.caption("Wähle Features aus verschiedenen Kategorien. Kritische Features sind für Rug-Detection empfohlen.")
            
            # Verwende Tabs für Feature-Kategorien
            feature_tabs = st.tabs(list(FEATURE_CATEGORIES.keys()))
            
            for tab_idx, (category, features_in_category) in enumerate(FEATURE_CATEGORIES.items()):
                with feature_tabs[tab_idx]:
                    st.markdown(f"**{category}**")
                    for feature in features_in_category:
                        is_critical = feature in CRITICAL_FEATURES
                        st.checkbox(
                            feature,
                            value=st.session_state.get(f"feature_{feature}", True),
                            key=f"feature_{feature}",
                            help=f"{'⚠️ KRITISCH für Rug-Detection!' if is_critical else ''}"
                        )
            
            # Sammle Features aus ALLEN Kategorien (nach Tabs)
            selected_features = []
            for category, features_in_category in FEATURE_CATEGORIES.items():
                for feature in features_in_category:
                    if st.session_state.get(f"feature_{feature}", False):
                        selected_features.append(feature)
            
            # Fallback: Wenn keine Features ausgewählt wurden, verwende alle
            if not selected_features:
                st.warning("⚠️ Keine Features ausgewählt! Alle Features werden verwendet.")
                selected_features = AVAILABLE_FEATURES.copy()
            else:
                st.info(f"✅ {len(selected_features)} Feature(s) ausgewählt")
            
            st.divider()
            
            # Phasen-Filter
            st.subheader("🪙 Coin-Phasen (optional)")
            phases_list = load_phases()
            phases = None
            
            if phases_list:
                phase_options = {}
                for phase in phases_list:
                    phase_id = phase.get("id")
                    phase_name = phase.get("name", f"Phase {phase_id}")
                    interval_sec = phase.get("interval_seconds", 0)
                    if phase_name and phase_name != f"Phase {phase_id}":
                        display_name = f"Phase {phase_id} - {phase_name} ({interval_sec}s)"
                    else:
                        display_name = f"Phase {phase_id} ({interval_sec}s)"
                    phase_options[phase_id] = display_name
                
                sorted_phases = sorted(phase_options.items())
                phase_labels = [label for _, label in sorted_phases]
                phase_ids = [pid for pid, _ in sorted_phases]
                
                selected_labels = st.multiselect(
                    "Phasen auswählen (optional)",
                    phase_labels,
                    help="Welche Coin-Phasen sollen einbezogen werden? (Leer = alle)",
                    key="train_phases"
                )
                
                phases = [phase_ids[phase_labels.index(label)] for label in selected_labels] if selected_labels else None
            else:
                st.warning("⚠️ Phasen konnten nicht geladen werden.")
            
            st.divider()
            
            # Hyperparameter
            st.subheader("⚙️ Hyperparameter (optional)")
            use_custom_params = st.checkbox("Hyperparameter anpassen", value=False, key="train_use_custom_params")
            params = None
            
            if use_custom_params:
                if model_type == "random_forest":
                    col1, col2 = st.columns(2)
                    with col1:
                        n_estimators = st.number_input("n_estimators", min_value=10, max_value=1000, value=100, step=10, key="train_rf_n_estimators")
                    with col2:
                        max_depth = st.number_input("max_depth", min_value=1, max_value=50, value=10, step=1, key="train_rf_max_depth")
                    params = {"n_estimators": int(n_estimators), "max_depth": int(max_depth)}
                else:  # xgboost
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        n_estimators = st.number_input("n_estimators", min_value=10, max_value=1000, value=100, step=10, key="train_xgb_n_estimators")
                    with col2:
                        max_depth = st.number_input("max_depth", min_value=1, max_value=20, value=6, step=1, key="train_xgb_max_depth")
                    with col3:
                        learning_rate = st.number_input("learning_rate", min_value=0.01, max_value=1.0, value=0.1, step=0.01, key="train_xgb_learning_rate")
                    params = {
                        "n_estimators": int(n_estimators),
                        "max_depth": int(max_depth),
                        "learning_rate": float(learning_rate)
                    }
            
            st.divider()
            
            # Feature-Engineering
            st.subheader("🔧 Feature-Engineering (optional)")
            use_engineered_features = st.checkbox(
                "Erweiterte Pump-Detection Features verwenden",
                value=True,
                help="Erstellt ~40 zusätzliche Features aus den Basis-Features",
                key="train_use_engineered_features"
            )
            feature_engineering_windows = [5, 10, 15] if use_engineered_features else None
            
            st.divider()
            
            # Marktstimmung
            st.subheader("📈 Marktstimmung (optional)")
            use_market_context = st.checkbox(
                "SOL-Preis-Kontext hinzufügen",
                value=False,
                help="Hilft dem Modell zu unterscheiden: 'Token steigt, während SOL stabil ist' vs. 'Token steigt, weil SOL steigt'",
                key="train_use_market_context"
            )
            
            st.divider()
            
            # SMOTE & Cross-Validation
            st.subheader("⚖️ Daten-Handling (optional)")
            use_smote = st.checkbox("SMOTE für Imbalanced Data (empfohlen)", value=True, key="train_use_smote")
            use_timeseries_split = st.checkbox("TimeSeriesSplit für Cross-Validation (empfohlen)", value=True, key="train_use_timeseries_split")
            cv_splits = st.number_input("Anzahl Splits", min_value=3, max_value=10, value=5, step=1, key="train_cv_splits") if use_timeseries_split else 5
        
        # Submit Button
        submitted = st.form_submit_button("🚀 Modell trainieren", type="primary", use_container_width=True)
    
    # ============================================================
    # VERARBEITUNG NACH FORM-SUBMISSION
    # ============================================================
    if submitted:
        # Erstelle datetime-Objekte
        try:
            train_start_dt = datetime.combine(train_start_date, train_start_time).replace(tzinfo=timezone.utc)
            train_end_dt = datetime.combine(train_end_date, train_end_time).replace(tzinfo=timezone.utc)
        except Exception as e:
            st.error(f"❌ Fehler beim Erstellen der Datetime-Objekte: {e}")
            return
        
        # Validierung
        errors = []
        
        if not model_name or not model_name.strip():
            errors.append("❌ Modell-Name ist erforderlich!")
        
        # Sammle Features erneut (sicherstellen, dass alle erfasst werden)
        selected_features = []
        for category, features_in_category in FEATURE_CATEGORIES.items():
            for feature in features_in_category:
                if st.session_state.get(f"feature_{feature}", False):
                    selected_features.append(feature)
        
        if not selected_features:
            selected_features = AVAILABLE_FEATURES.copy()  # Fallback: Alle Features
        
        if train_start_dt >= train_end_dt:
            errors.append("❌ Start-Zeitpunkt muss vor End-Zeitpunkt liegen!")
        
        if errors:
            for error in errors:
                st.error(error)
            return
        
        # API-Call
        with st.spinner("🔄 Erstelle Training-Job..."):
            try:
                train_start_iso = train_start_dt.astimezone(timezone.utc).isoformat().replace('+00:00', 'Z')
                train_end_iso = train_end_dt.astimezone(timezone.utc).isoformat().replace('+00:00', 'Z')
                
                data = {
                    "name": model_name.strip(),
                    "model_type": model_type,
                    "target_var": time_based_target_var,
                    "operator": None,
                    "target_value": None,
                    "features": selected_features,
                    "phases": phases if phases else None,
                    "params": params,
                    "train_start": train_start_iso,
                    "train_end": train_end_iso,
                    "use_time_based_prediction": True,
                    "future_minutes": int(future_minutes),
                    "min_percent_change": float(min_percent_change),
                    "direction": direction,
                    "use_engineered_features": use_engineered_features,
                    "feature_engineering_windows": feature_engineering_windows,
                    "use_smote": use_smote,
                    "use_timeseries_split": use_timeseries_split,
                    "cv_splits": int(cv_splits) if use_timeseries_split else None,
                    "use_market_context": use_market_context
                }
                
                result = api_post("/api/models/create", data)
                
                if result:
                    st.success(f"✅ Job erstellt! Job-ID: {result.get('job_id')}")
                    st.info(f"📊 Status: {result.get('status')}. Das Modell wird jetzt trainiert.")
                    st.balloons()
                    st.session_state['last_created_job_id'] = result.get('job_id')
                else:
                    st.error("❌ Fehler beim Erstellen des Jobs. Bitte prüfe die Logs.")
            except Exception as e:
                st.error(f"❌ Fehler beim Erstellen des Jobs: {str(e)}")
                st.exception(e)
    
    # Weiterleitung zu Jobs-Seite
    if st.session_state.get('last_created_job_id'):
        if st.button("📊 Zu Jobs anzeigen", key="goto_jobs_after_train"):
            st.session_state['page'] = 'jobs'
            st.session_state.pop('last_created_job_id', None)
            st.rerun()

def page_test():
    """Modell testen"""
    st.title("🧪 Modell testen")
    
    # Prüfe ob Modell-ID aus Übersicht übergeben wurde
    test_model_id = st.session_state.get('test_model_id')
    
    # Lade Modelle
    models = api_get("/api/models")
    if not models:
        st.warning("⚠️ Keine Modelle gefunden")
        return
    
    # Filter: Nur READY Modelle
    ready_models = [m for m in models if m.get('status') == 'READY' and not m.get('is_deleted')]
    
    if not ready_models:
        st.warning("⚠️ Keine fertigen Modelle zum Testen verfügbar")
        return
    
    # Modell auswählen - verwende test_model_id wenn gesetzt
    model_options = {m.get('id'): f"{m.get('name')} ({m.get('model_type')})" for m in ready_models}
    
    # Bestimme initialen Index basierend auf test_model_id
    initial_index = 0
    if test_model_id and test_model_id in model_options:
        initial_index = list(model_options.keys()).index(test_model_id)
    
    selected_model_id = st.selectbox(
        "Modell auswählen *",
        options=list(model_options.keys()),
        format_func=lambda x: model_options[x],
        index=initial_index,
        key="test_model_selectbox"
    )
    
    # Aktualisiere test_model_id wenn sich die Auswahl ändert
    if selected_model_id != test_model_id:
        st.session_state['test_model_id'] = selected_model_id
    
    if selected_model_id:
        selected_model = next((m for m in ready_models if m.get('id') == selected_model_id), None)
        
        if selected_model:
            st.info(f"📋 Modell: {selected_model.get('name')} ({selected_model.get('model_type')})")
            
            # Test-Zeitraum
            st.subheader("📅 Test-Zeitraum")
            
            # Lade verfügbare Daten (nur Min/Max Timestamps)
            data_availability = api_get("/api/data-availability")
            
            min_timestamp = data_availability.get('min_timestamp') if data_availability else None
            max_timestamp = data_availability.get('max_timestamp') if data_availability else None
            
            # Parse Timestamps
            min_date = None
            max_date = None
            min_datetime = None
            max_datetime = None
            
            if min_timestamp and max_timestamp:
                try:
                    min_datetime = datetime.fromisoformat(min_timestamp.replace('Z', '+00:00'))
                    max_datetime = datetime.fromisoformat(max_timestamp.replace('Z', '+00:00'))
                    min_date = min_datetime.date()
                    max_date = max_datetime.date()
                    
                    st.info(f"📊 **Verfügbare Daten:** Von {min_datetime.strftime('%d.%m.%Y %H:%M')} bis {max_datetime.strftime('%d.%m.%Y %H:%M')}")
                except Exception as e:
                    st.warning(f"⚠️ Konnte Datumsbereich nicht parsen: {e}")
            else:
                st.warning("⚠️ Keine Testdaten in der Datenbank gefunden!")
            
            st.divider()
            
            # Datum- und Uhrzeit-Auswahl
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**🕐 Start-Zeitpunkt**")
                
                # Default-Start-Datum: Min-Datum oder heute - 7 Tage
                if min_date:
                    default_start_date = min_date
                else:
                    default_start_date = datetime.now(timezone.utc).date() - timedelta(days=7)
                
                test_start_date = st.date_input(
                    "Start-Datum *",
                    value=default_start_date,
                    min_value=min_date,
                    max_value=max_date,
                    key="test_start_date",
                    help=f"Wähle ein Datum zwischen {min_date.strftime('%d.%m.%Y') if min_date else 'N/A'} und {max_date.strftime('%d.%m.%Y') if max_date else 'N/A'}"
                )
                
                # Default-Start-Uhrzeit: Min-Uhrzeit oder 00:00
                if min_datetime and test_start_date == min_date:
                    default_start_time = min_datetime.time()
                else:
                    default_start_time = datetime.now(timezone.utc).time().replace(hour=0, minute=0, second=0, microsecond=0)
                
                test_start_time = st.time_input(
                    "Start-Uhrzeit *",
                    value=default_start_time,
                    key="test_start_time",
                    help="Wähle eine Uhrzeit"
                )
                
                test_start_dt = datetime.combine(test_start_date, test_start_time).replace(tzinfo=timezone.utc)
                
                # Warnung wenn außerhalb des verfügbaren Bereichs
                if min_datetime and test_start_dt < min_datetime:
                    st.warning(f"⚠️ Start-Zeitpunkt liegt vor dem ältesten Eintrag ({min_datetime.strftime('%d.%m.%Y %H:%M')})")
                elif max_datetime and test_start_dt > max_datetime:
                    st.warning(f"⚠️ Start-Zeitpunkt liegt nach dem neuesten Eintrag ({max_datetime.strftime('%d.%m.%Y %H:%M')})")
            
            with col2:
                st.markdown("**🕐 Ende-Zeitpunkt**")
                
                # Default-End-Datum: Max-Datum oder heute
                if max_date:
                    default_end_date = max_date
                else:
                    default_end_date = datetime.now(timezone.utc).date()
                
                test_end_date = st.date_input(
                    "Ende-Datum *",
                    value=default_end_date,
                    min_value=test_start_date,  # Ende muss nach Start sein
                    max_value=max_date,
                    key="test_end_date",
                    help=f"Wähle ein Datum nach dem Start-Datum (max. {max_date.strftime('%d.%m.%Y') if max_date else 'N/A'})"
                )
                
                # Default-End-Uhrzeit: Max-Uhrzeit oder 23:59
                if max_datetime and test_end_date == max_date:
                    default_end_time = max_datetime.time()
                else:
                    default_end_time = datetime.now(timezone.utc).time().replace(hour=23, minute=59, second=59, microsecond=0)
                
                test_end_time = st.time_input(
                    "Ende-Uhrzeit *",
                    value=default_end_time,
                    key="test_end_time",
                    help="Wähle eine Uhrzeit"
                )
                
                test_end_dt = datetime.combine(test_end_date, test_end_time).replace(tzinfo=timezone.utc)
                
                # Warnung wenn außerhalb des verfügbaren Bereichs
                if min_datetime and test_end_dt < min_datetime:
                    st.warning(f"⚠️ Ende-Zeitpunkt liegt vor dem ältesten Eintrag ({min_datetime.strftime('%d.%m.%Y %H:%M')})")
                elif max_datetime and test_end_dt > max_datetime:
                    st.warning(f"⚠️ Ende-Zeitpunkt liegt nach dem neuesten Eintrag ({max_datetime.strftime('%d.%m.%Y %H:%M')})")
            
            # Overlap-Check Info
            if selected_model.get('train_start') and selected_model.get('train_end'):
                train_start_dt = datetime.fromisoformat(selected_model['train_start'].replace('Z', '+00:00'))
                train_end_dt = datetime.fromisoformat(selected_model['train_end'].replace('Z', '+00:00'))
                
                # Prüfe Overlap
                if test_start_dt < train_end_dt and test_end_dt > train_start_dt:
                    overlap_duration = min(test_end_dt, train_end_dt) - max(test_start_dt, train_start_dt)
                    total_duration = test_end_dt - test_start_dt
                    overlap_pct = (overlap_duration.total_seconds() / total_duration.total_seconds() * 100) if total_duration.total_seconds() > 0 else 0
                    st.warning(f"⚠️ {overlap_pct:.1f}% Überschneidung mit Trainingsdaten (Test wird trotzdem ausgeführt)")
            
            # Submit
            if st.button("🧪 Test starten", type="primary", use_container_width=True):
                if test_start_dt >= test_end_dt:
                    st.error("❌ Start-Zeitpunkt muss vor End-Zeitpunkt liegen!")
                else:
                    with st.spinner("🔄 Erstelle Test-Job..."):
                        # Konvertiere datetime zu UTC ISO-Format
                        test_start_iso = test_start_dt.astimezone(timezone.utc).isoformat().replace('+00:00', 'Z')
                        test_end_iso = test_end_dt.astimezone(timezone.utc).isoformat().replace('+00:00', 'Z')
                        
                        data = {
                            "test_start": test_start_iso,
                            "test_end": test_end_iso
                        }
                        
                        result = api_post(f"/api/models/{selected_model_id}/test", data)
                        
                        if result:
                            st.success(f"✅ Test-Job erstellt! Job-ID: {result.get('job_id')}")
                            st.info(f"📊 Status: {result.get('status')}. Der Test wird jetzt ausgeführt.")
                            
                            if st.button("📊 Zu Jobs anzeigen"):
                                st.session_state['page'] = 'jobs'
                                st.rerun()

def page_compare():
    """Modelle vergleichen"""
    st.title("⚔️ Modelle vergleichen")
    
    # Lade Modelle
    models = api_get("/api/models")
    if not models:
        st.warning("⚠️ Keine Modelle gefunden")
        return
    
    # Filter: Nur READY Modelle
    ready_models = [m for m in models if m.get('status') == 'READY' and not m.get('is_deleted')]
    
    if len(ready_models) < 2:
        st.warning("⚠️ Mindestens 2 fertige Modelle benötigt für Vergleich")
        return
    
    # Modell-Auswahl
    model_options = {m.get('id'): f"{m.get('name')} ({m.get('model_type')})" for m in ready_models}
    
    col1, col2 = st.columns(2)
    with col1:
        model_a_id = st.selectbox(
            "Modell A *",
            options=list(model_options.keys()),
            format_func=lambda x: model_options[x],
            index=0 if st.session_state.get('compare_model_a_id') not in model_options else list(model_options.keys()).index(st.session_state.get('compare_model_a_id'))
        )
    with col2:
        # Filter: Modell B darf nicht Modell A sein
        model_b_options = {k: v for k, v in model_options.items() if k != model_a_id}
        model_b_id = st.selectbox(
            "Modell B *",
            options=list(model_b_options.keys()),
            format_func=lambda x: model_b_options[x]
        )
    
    # Test-Zeitraum
    st.subheader("📅 Vergleichs-Zeitraum")
    
    # Lade verfügbare Daten (nur Min/Max Timestamps)
    data_availability = api_get("/api/data-availability")
    
    min_timestamp = data_availability.get('min_timestamp') if data_availability else None
    max_timestamp = data_availability.get('max_timestamp') if data_availability else None
    
    # Parse Timestamps
    min_date = None
    max_date = None
    min_datetime = None
    max_datetime = None
    
    if min_timestamp and max_timestamp:
        try:
            min_datetime = datetime.fromisoformat(min_timestamp.replace('Z', '+00:00'))
            max_datetime = datetime.fromisoformat(max_timestamp.replace('Z', '+00:00'))
            min_date = min_datetime.date()
            max_date = max_datetime.date()
            
            st.info(f"📊 **Verfügbare Daten:** Von {min_datetime.strftime('%d.%m.%Y %H:%M')} bis {max_datetime.strftime('%d.%m.%Y %H:%M')}")
        except Exception as e:
            st.warning(f"⚠️ Konnte Datumsbereich nicht parsen: {e}")
    else:
        st.warning("⚠️ Keine Vergleichsdaten in der Datenbank gefunden!")
    
    st.divider()
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**🕐 Start-Zeitpunkt**")
        
        # Default-Start-Datum: Min-Datum oder heute - 7 Tage
        if min_date:
            default_start_date = min_date
        else:
            default_start_date = datetime.now(timezone.utc).date() - timedelta(days=7)
        
        compare_start_date = st.date_input(
            "Start-Datum *",
            value=default_start_date,
            min_value=min_date,
            max_value=max_date,
            key="compare_start_date",
            help=f"Wähle ein Datum zwischen {min_date.strftime('%d.%m.%Y') if min_date else 'N/A'} und {max_date.strftime('%d.%m.%Y') if max_date else 'N/A'}"
        )
        
        # Default-Start-Uhrzeit: Min-Uhrzeit oder 00:00
        if min_datetime and compare_start_date == min_date:
            default_start_time = min_datetime.time()
        else:
            default_start_time = datetime.now(timezone.utc).time().replace(hour=0, minute=0, second=0, microsecond=0)
        
        compare_start_time = st.time_input(
            "Start-Uhrzeit *",
            value=default_start_time,
            key="compare_start_time",
            help="Wähle eine Uhrzeit"
        )
        
        test_start_dt = datetime.combine(compare_start_date, compare_start_time).replace(tzinfo=timezone.utc)
        
        # Warnung wenn außerhalb des verfügbaren Bereichs
        if min_datetime and test_start_dt < min_datetime:
            st.warning(f"⚠️ Start-Zeitpunkt liegt vor dem ältesten Eintrag ({min_datetime.strftime('%d.%m.%Y %H:%M')})")
        elif max_datetime and test_start_dt > max_datetime:
            st.warning(f"⚠️ Start-Zeitpunkt liegt nach dem neuesten Eintrag ({max_datetime.strftime('%d.%m.%Y %H:%M')})")
    
    with col2:
        st.markdown("**🕐 Ende-Zeitpunkt**")
        
        # Default-End-Datum: Max-Datum oder heute
        if max_date:
            default_end_date = max_date
        else:
            default_end_date = datetime.now(timezone.utc).date()
        
        compare_end_date = st.date_input(
            "Ende-Datum *",
            value=default_end_date,
            min_value=compare_start_date,  # Ende muss nach Start sein
            max_value=max_date,
            key="compare_end_date",
            help=f"Wähle ein Datum nach dem Start-Datum (max. {max_date.strftime('%d.%m.%Y') if max_date else 'N/A'})"
        )
        
        # Default-End-Uhrzeit: Max-Uhrzeit oder 23:59
        if max_datetime and compare_end_date == max_date:
            default_end_time = max_datetime.time()
        else:
            default_end_time = datetime.now(timezone.utc).time().replace(hour=23, minute=59, second=59, microsecond=0)
        
        compare_end_time = st.time_input(
            "Ende-Uhrzeit *",
            value=default_end_time,
            key="compare_end_time",
            help="Wähle eine Uhrzeit"
        )
        
        test_end_dt = datetime.combine(compare_end_date, compare_end_time).replace(tzinfo=timezone.utc)
        
        # Warnung wenn außerhalb des verfügbaren Bereichs
        if min_datetime and test_end_dt < min_datetime:
            st.warning(f"⚠️ Ende-Zeitpunkt liegt vor dem ältesten Eintrag ({min_datetime.strftime('%d.%m.%Y %H:%M')})")
        elif max_datetime and test_end_dt > max_datetime:
            st.warning(f"⚠️ Ende-Zeitpunkt liegt nach dem neuesten Eintrag ({max_datetime.strftime('%d.%m.%Y %H:%M')})")
    
    # Submit
    if st.button("⚔️ Vergleich starten", type="primary", use_container_width=True):
        if test_start_dt >= test_end_dt:
            st.error("❌ Start-Zeitpunkt muss vor End-Zeitpunkt liegen!")
        else:
            with st.spinner("🔄 Erstelle Vergleichs-Job..."):
                # Konvertiere datetime zu UTC ISO-Format
                test_start_iso = test_start_dt.astimezone(timezone.utc).isoformat().replace('+00:00', 'Z')
                test_end_iso = test_end_dt.astimezone(timezone.utc).isoformat().replace('+00:00', 'Z')
                
                data = {
                    "model_a_id": model_a_id,
                    "model_b_id": model_b_id,
                    "test_start": test_start_iso,
                    "test_end": test_end_iso
                }
                
                result = api_post("/api/models/compare", data)
                
                if result:
                    st.success(f"✅ Vergleichs-Job erstellt! Job-ID: {result.get('job_id')}")
                    st.info(f"📊 Status: {result.get('status')}. Der Vergleich wird jetzt ausgeführt.")
                    
                    if st.button("📊 Zu Jobs anzeigen"):
                        st.session_state['page'] = 'jobs'
                        st.rerun()

def page_jobs():
    """Jobs-Übersicht"""
    st.title("📊 Jobs")
    
    # Lade Jobs
    jobs = api_get("/api/queue")
    if not jobs:
        st.info("ℹ️ Keine Jobs gefunden")
        return
    
    # Status-Filter
    status_filter = st.selectbox("Status Filter", ["Alle", "PENDING", "RUNNING", "COMPLETED", "FAILED"])
    
    # Filter anwenden
    filtered_jobs = jobs
    if status_filter != "Alle":
        filtered_jobs = [j for j in jobs if j.get('status') == status_filter]
    
    st.info(f"📊 {len(filtered_jobs)} Job(s) gefunden")
    
    # Tabelle
    if filtered_jobs:
        df = pd.DataFrame([
            {
                "ID": j.get('id'),
                "Typ": j.get('job_type'),
                "Status": j.get('status'),
                "Progress": f"{j.get('progress', 0) * 100:.1f}%" if j.get('progress') else "N/A",
                "Erstellt": j.get('created_at', '')[:19] if j.get('created_at') else "N/A",
                "Nachricht": j.get('progress_msg', '')[:50] if j.get('progress_msg') else "N/A"
            }
            for j in filtered_jobs
        ])
        st.dataframe(df, use_container_width=True, hide_index=True)
        
        # Details
        selected_job_id = st.selectbox("Job auswählen für Details", options=[j.get('id') for j in filtered_jobs])
        if selected_job_id:
            selected_job = next((j for j in filtered_jobs if j.get('id') == selected_job_id), None)
            if selected_job:
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Status", selected_job.get('status', 'N/A'))
                with col2:
                    st.metric("Progress", f"{selected_job.get('progress', 0) * 100:.1f}%")
                with col3:
                    st.metric("Typ", selected_job.get('job_type', 'N/A'))
                
                # Test-Ergebnisse anzeigen
                if selected_job.get('result_test'):
                    st.subheader("🧪 Test-Ergebnisse")
                    test = selected_job['result_test']
                    
                    # Basis-Metriken
                    st.markdown("**📊 Basis-Metriken**")
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Accuracy", f"{test.get('accuracy', 0):.4f}")
                    with col2:
                        st.metric("F1-Score", f"{test.get('f1_score', 0):.4f}")
                    with col3:
                        st.metric("Precision", f"{test.get('precision_score', 0):.4f}")
                    with col4:
                        st.metric("Recall", f"{test.get('recall', 0):.4f}")
                    
                    # Zusätzliche Metriken (Phase 9)
                    if test.get('roc_auc') is not None or test.get('mcc') is not None:
                        st.markdown("**📈 Zusätzliche Metriken**")
                        col1, col2, col3, col4, col5 = st.columns(5)
                        with col1:
                            if test.get('roc_auc') is not None:
                                st.metric("ROC-AUC", f"{test.get('roc_auc', 0):.4f}")
                        with col2:
                            if test.get('mcc') is not None:
                                st.metric("MCC", f"{test.get('mcc', 0):.4f}")
                        with col3:
                            if test.get('fpr') is not None:
                                st.metric("FPR", f"{test.get('fpr', 0):.4f}")
                        with col4:
                            if test.get('fnr') is not None:
                                st.metric("FNR", f"{test.get('fnr', 0):.4f}")
                        with col5:
                            if test.get('simulated_profit_pct') is not None:
                                st.metric("💰 Profit", f"{test.get('simulated_profit_pct', 0):.2f}%")
                    
                    # Confusion Matrix
                    confusion_matrix = test.get('confusion_matrix')
                    if confusion_matrix:
                        st.markdown("**🔢 Confusion Matrix**")
                        cm_col1, cm_col2, cm_col3, cm_col4 = st.columns(4)
                        with cm_col1:
                            st.metric("True Positive (TP)", confusion_matrix.get('tp', 0))
                        with cm_col2:
                            st.metric("True Negative (TN)", confusion_matrix.get('tn', 0))
                        with cm_col3:
                            st.metric("False Positive (FP)", confusion_matrix.get('fp', 0))
                        with cm_col4:
                            st.metric("False Negative (FN)", confusion_matrix.get('fn', 0))
                        
                        # Visualisierung als Tabelle
                        cm_data = {
                            'Tatsächlich': ['Negativ', 'Positiv'],
                            'Vorhergesagt: Negativ': [confusion_matrix.get('tn', 0), confusion_matrix.get('fn', 0)],
                            'Vorhergesagt: Positiv': [confusion_matrix.get('fp', 0), confusion_matrix.get('tp', 0)]
                        }
                        cm_df = pd.DataFrame(cm_data)
                        st.dataframe(cm_df, use_container_width=True, hide_index=True)
                    elif test.get('tp') is not None:
                        # Fallback: Legacy-Format (einzelne Felder)
                        st.markdown("**🔢 Confusion Matrix**")
                        cm_col1, cm_col2, cm_col3, cm_col4 = st.columns(4)
                        with cm_col1:
                            st.metric("True Positive (TP)", test.get('tp', 0))
                        with cm_col2:
                            st.metric("True Negative (TN)", test.get('tn', 0))
                        with cm_col3:
                            st.metric("False Positive (FP)", test.get('fp', 0))
                        with cm_col4:
                            st.metric("False Negative (FN)", test.get('fn', 0))
                    
                    # Samples
                    st.markdown("**📊 Daten-Info**")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Samples", test.get('num_samples', 0))
                    with col2:
                        st.metric("Positive", test.get('num_positive', 0))
                    with col3:
                        st.metric("Negative", test.get('num_negative', 0))
                    
                    # Train vs. Test Vergleich (Phase 2)
                    if test.get('train_accuracy') is not None:
                        st.markdown("**📊 Train vs. Test Vergleich**")
                        train_acc = test.get('train_accuracy')
                        test_acc = test.get('accuracy', 0)
                        acc_degradation = test.get('accuracy_degradation')
                        is_overfitted = test.get('is_overfitted', False)
                        
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Train Accuracy", f"{train_acc:.4f}")
                        with col2:
                            st.metric("Test Accuracy", f"{test_acc:.4f}")
                        with col3:
                            if acc_degradation is not None:
                                degradation_color = "🔴" if is_overfitted else "🟢"
                                st.metric(f"{degradation_color} Degradation", f"{acc_degradation:.2%}")
                        with col4:
                            if is_overfitted:
                                st.error("⚠️ OVERFITTING!")
                            else:
                                st.success("✅ OK")
                        
                        if is_overfitted:
                            st.warning("⚠️ **Overfitting erkannt!** Das Modell generalisiert schlecht auf neue Daten. Train-Test Gap > 10%.")
                        
                        # F1 Vergleich
                        if test.get('train_f1') is not None:
                            train_f1 = test.get('train_f1')
                            test_f1 = test.get('f1_score', 0)
                            f1_degradation = test.get('f1_degradation')
                            if f1_degradation is not None:
                                st.caption(f"F1: Train={train_f1:.4f}, Test={test_f1:.4f}, Gap={f1_degradation:.2%}")
                    
                    # Test-Zeitraum Info (Phase 2)
                    test_duration_days = test.get('test_duration_days')
                    if test_duration_days is not None:
                        if test_duration_days < 1.0:
                            st.warning(f"⚠️ Test-Zeitraum zu kurz: {test_duration_days:.2f} Tage (empfohlen: mindestens 1 Tag)")
                        else:
                            st.caption(f"📅 Test-Zeitraum: {test_duration_days:.2f} Tage")
                    
                    if test.get('has_overlap'):
                        st.warning(f"⚠️ {test.get('overlap_note', 'Überschneidung mit Trainingsdaten')}")
                
                # Vergleichs-Ergebnisse anzeigen
                if selected_job.get('result_comparison'):
                    st.subheader("⚖️ Vergleichs-Ergebnisse")
                    comp = selected_job['result_comparison']
                    
                    # Gewinner
                    winner_id = comp.get('winner_id')
                    if winner_id:
                        st.success(f"🏆 Gewinner: Modell {winner_id}")
                    else:
                        st.info("🤝 Unentschieden")
                    
                    # Basis-Metriken: Modell A vs. Modell B
                    st.markdown("### 📊 Basis-Metriken")
                    col_a1, col_a2, col_a3, col_a4, col_b1, col_b2, col_b3, col_b4 = st.columns(8)
                    with col_a1:
                        st.markdown("**Modell A**")
                        st.metric("Accuracy", f"{comp.get('a_accuracy', 0):.4f}" if comp.get('a_accuracy') else "N/A")
                    with col_a2:
                        st.markdown("&nbsp;")
                        st.metric("F1-Score", f"{comp.get('a_f1', 0):.4f}" if comp.get('a_f1') else "N/A")
                    with col_a3:
                        st.markdown("&nbsp;")
                        st.metric("Precision", f"{comp.get('a_precision', 0):.4f}" if comp.get('a_precision') else "N/A")
                    with col_a4:
                        st.markdown("&nbsp;")
                        st.metric("Recall", f"{comp.get('a_recall', 0):.4f}" if comp.get('a_recall') else "N/A")
                    with col_b1:
                        st.markdown("**Modell B**")
                        st.metric("Accuracy", f"{comp.get('b_accuracy', 0):.4f}" if comp.get('b_accuracy') else "N/A")
                    with col_b2:
                        st.markdown("&nbsp;")
                        st.metric("F1-Score", f"{comp.get('b_f1', 0):.4f}" if comp.get('b_f1') else "N/A")
                    with col_b3:
                        st.markdown("&nbsp;")
                        st.metric("Precision", f"{comp.get('b_precision', 0):.4f}" if comp.get('b_precision') else "N/A")
                    with col_b4:
                        st.markdown("&nbsp;")
                        st.metric("Recall", f"{comp.get('b_recall', 0):.4f}" if comp.get('b_recall') else "N/A")
                    
                    # Zusätzliche Metriken
                    if comp.get('a_mcc') or comp.get('b_mcc') or comp.get('a_simulated_profit_pct') or comp.get('b_simulated_profit_pct'):
                        st.markdown("### 📈 Zusätzliche Metriken")
                        col_a1, col_a2, col_a3, col_a4, col_b1, col_b2, col_b3, col_b4 = st.columns(8)
                        with col_a1:
                            st.markdown("**Modell A**")
                            st.metric("MCC", f"{comp.get('a_mcc', 0):.4f}" if comp.get('a_mcc') else "N/A")
                        with col_a2:
                            st.markdown("&nbsp;")
                            st.metric("FPR", f"{comp.get('a_fpr', 0):.4f}" if comp.get('a_fpr') else "N/A")
                        with col_a3:
                            st.markdown("&nbsp;")
                            st.metric("FNR", f"{comp.get('a_fnr', 0):.4f}" if comp.get('a_fnr') else "N/A")
                        with col_a4:
                            st.markdown("&nbsp;")
                            st.metric("Profit", f"{comp.get('a_simulated_profit_pct', 0):.4f}%" if comp.get('a_simulated_profit_pct') else "N/A")
                        with col_b1:
                            st.markdown("**Modell B**")
                            st.metric("MCC", f"{comp.get('b_mcc', 0):.4f}" if comp.get('b_mcc') else "N/A")
                        with col_b2:
                            st.markdown("&nbsp;")
                            st.metric("FPR", f"{comp.get('b_fpr', 0):.4f}" if comp.get('b_fpr') else "N/A")
                        with col_b3:
                            st.markdown("&nbsp;")
                            st.metric("FNR", f"{comp.get('b_fnr', 0):.4f}" if comp.get('b_fnr') else "N/A")
                        with col_b4:
                            st.markdown("&nbsp;")
                            st.metric("Profit", f"{comp.get('b_simulated_profit_pct', 0):.4f}%" if comp.get('b_simulated_profit_pct') else "N/A")
                    
                    # Confusion Matrix
                    if comp.get('a_confusion_matrix') or comp.get('b_confusion_matrix'):
                        st.markdown("### 📊 Confusion Matrix")
                        col_a, col_b = st.columns(2)
                        with col_a:
                            st.markdown("**Modell A**")
                            if comp.get('a_confusion_matrix'):
                                cm_a = comp['a_confusion_matrix']
                                st.write(f"TP: {cm_a.get('tp', 0)} | TN: {cm_a.get('tn', 0)}")
                                st.write(f"FP: {cm_a.get('fp', 0)} | FN: {cm_a.get('fn', 0)}")
                            else:
                                st.write("N/A")
                        with col_b:
                            st.markdown("**Modell B**")
                            if comp.get('b_confusion_matrix'):
                                cm_b = comp['b_confusion_matrix']
                                st.write(f"TP: {cm_b.get('tp', 0)} | TN: {cm_b.get('tn', 0)}")
                                st.write(f"FP: {cm_b.get('fp', 0)} | FN: {cm_b.get('fn', 0)}")
                            else:
                                st.write("N/A")
                    
                    # Train vs. Test Vergleich
                    if comp.get('a_train_accuracy') or comp.get('b_train_accuracy'):
                        st.markdown("### 📊 Train vs. Test Vergleich")
                        col_a1, col_a2, col_a3, col_b1, col_b2, col_b3 = st.columns(6)
                        with col_a1:
                            st.markdown("**Modell A**")
                            if comp.get('a_train_accuracy'):
                                st.metric("Train Acc", f"{comp.get('a_train_accuracy'):.4f}")
                            else:
                                st.write("Train Acc: N/A")
                        with col_a2:
                            st.markdown("&nbsp;")
                            if comp.get('a_accuracy'):
                                st.metric("Test Acc", f"{comp.get('a_accuracy'):.4f}")
                            else:
                                st.write("Test Acc: N/A")
                        with col_a3:
                            st.markdown("&nbsp;")
                            if comp.get('a_accuracy_degradation') is not None:
                                degradation = comp.get('a_accuracy_degradation')
                                st.metric("Degradation", f"{degradation:.4f}", 
                                         delta=f"{'⚠️ Overfitting' if degradation > 0.1 else '✅ OK'}" if degradation else None)
                            else:
                                st.write("Degradation: N/A")
                        with col_b1:
                            st.markdown("**Modell B**")
                            if comp.get('b_train_accuracy'):
                                st.metric("Train Acc", f"{comp.get('b_train_accuracy'):.4f}")
                            else:
                                st.write("Train Acc: N/A")
                        with col_b2:
                            st.markdown("&nbsp;")
                            if comp.get('b_accuracy'):
                                st.metric("Test Acc", f"{comp.get('b_accuracy'):.4f}")
                            else:
                                st.write("Test Acc: N/A")
                        with col_b3:
                            st.markdown("&nbsp;")
                            if comp.get('b_accuracy_degradation') is not None:
                                degradation = comp.get('b_accuracy_degradation')
                                st.metric("Degradation", f"{degradation:.4f}",
                                         delta=f"{'⚠️ Overfitting' if degradation > 0.1 else '✅ OK'}" if degradation else None)
                            else:
                                st.write("Degradation: N/A")
                        
                        # Overfitting Warnung
                        if comp.get('a_is_overfitted') or comp.get('b_is_overfitted'):
                            if comp.get('a_is_overfitted'):
                                st.warning(f"⚠️ Modell A ist möglicherweise overfitted!")
                            if comp.get('b_is_overfitted'):
                                st.warning(f"⚠️ Modell B ist möglicherweise overfitted!")
                    
                    # Test-Zeitraum Info
                    if comp.get('a_test_duration_days') or comp.get('b_test_duration_days'):
                        st.markdown("### 📅 Test-Zeitraum")
                        duration_a = comp.get('a_test_duration_days', 0) if comp.get('a_test_duration_days') else 0
                        duration_b = comp.get('b_test_duration_days', 0) if comp.get('b_test_duration_days') else 0
                        st.write(f"Dauer: {duration_a:.2f} Tage (beide Modelle getestet auf demselben Zeitraum)")
                        if duration_a < 1:
                            st.warning("⚠️ Test-Zeitraum zu kurz (empfohlen: mindestens 1 Tag)")
                
                # Vollständige Details als JSON (erweiterbar)
                with st.expander("📋 Vollständige Job-Details (JSON)", expanded=False):
                    st.json(selected_job)
    else:
        st.info("ℹ️ Keine Jobs gefunden")

def page_comparisons():
    """Vergleichs-Übersicht: Liste aller Modell-Vergleiche"""
    st.title("⚖️ Vergleichs-Übersicht")
    
    # Initialisiere selected_comparisons in session_state
    if 'selected_comparison_ids' not in st.session_state:
        st.session_state['selected_comparison_ids'] = []
    
    # Lade Vergleichs-Ergebnisse
    comparisons = api_get("/api/comparisons")
    if not comparisons:
        st.warning("⚠️ Keine Vergleichs-Ergebnisse gefunden")
        return
    
    st.info(f"📊 {len(comparisons)} Vergleich(e) gefunden")
    
    # Kompakte Karten-Ansicht
    if comparisons:
        st.subheader("📋 Vergleichs-Ergebnisse")
        
        # Erstelle Karten in einem Grid (2 Spalten)
        cols = st.columns(2)
        
        for idx, comp in enumerate(comparisons):
            comp_id = comp.get('id')
            model_a_id = comp.get('model_a_id')
            model_b_id = comp.get('model_b_id')
            winner_id = comp.get('winner_id')
            created = comp.get('created_at', '')[:10] if comp.get('created_at') else "N/A"
            test_start = comp.get('test_start')
            test_end = comp.get('test_end')
            
            # Hole Modell-Namen und Trainings-Zeiträume
            model_a = api_get(f"/api/models/{model_a_id}")
            model_b = api_get(f"/api/models/{model_b_id}")
            model_a_name = model_a.get('name', f"ID: {model_a_id}") if model_a else f"ID: {model_a_id}"
            model_b_name = model_b.get('name', f"ID: {model_b_id}") if model_b else f"ID: {model_b_id}"
            model_a_train_start = model_a.get('train_start') if model_a else None
            model_a_train_end = model_a.get('train_end') if model_a else None
            model_b_train_start = model_b.get('train_start') if model_b else None
            model_b_train_end = model_b.get('train_end') if model_b else None
            
            # Metriken
            a_accuracy = comp.get('a_accuracy', 0)
            a_f1 = comp.get('a_f1', 0)
            a_roc_auc = comp.get('a_roc_auc')
            a_mcc = comp.get('a_mcc')
            a_profit = comp.get('a_simulated_profit_pct')
            b_accuracy = comp.get('b_accuracy', 0)
            b_f1 = comp.get('b_f1', 0)
            b_roc_auc = comp.get('b_roc_auc')
            b_mcc = comp.get('b_mcc')
            b_profit = comp.get('b_simulated_profit_pct')
            
            # Checkbox
            is_selected = comp_id in st.session_state.get('selected_comparison_ids', [])
            checkbox_key = f"comp_checkbox_{comp_id}"
            
            # Wähle Spalte (abwechselnd)
            col = cols[idx % 2]
            
            with col:
                # Karte mit Border
                card_style = """
                <style>
                .comparison-card {
                    border: 2px solid #e0e0e0;
                    border-radius: 8px;
                    padding: 12px;
                    margin-bottom: 12px;
                    background: white;
                }
                .comparison-card.selected {
                    border-color: #1f77b4;
                    background: #f0f8ff;
                }
                </style>
                """
                st.markdown(card_style, unsafe_allow_html=True)
                
                # Header mit Checkbox und Titel
                header_col1, header_col2, header_col3 = st.columns([0.3, 4, 0.6])
                with header_col1:
                    checked = st.checkbox("", value=is_selected, key=checkbox_key, label_visibility="collapsed")
                    # Update session_state ohne st.rerun() - Streamlit rendert automatisch neu
                    if checked and comp_id not in st.session_state.get('selected_comparison_ids', []):
                        if 'selected_comparison_ids' not in st.session_state:
                            st.session_state['selected_comparison_ids'] = []
                        st.session_state['selected_comparison_ids'].append(comp_id)
                    elif not checked and comp_id in st.session_state.get('selected_comparison_ids', []):
                        st.session_state['selected_comparison_ids'].remove(comp_id)
                
                with header_col2:
                    st.markdown(f"**{model_a_name} vs {model_b_name}**")
                
                with header_col3:
                    if st.button("📋", key=f"comp_details_{comp_id}", help="Details anzeigen", use_container_width=True):
                        st.session_state['comparison_details_id'] = comp_id
                        st.session_state['page'] = 'comparison_details'
                        st.rerun()
                
                # Gewinner
                if winner_id:
                    if winner_id == model_a_id:
                        st.success(f"🏆 Gewinner: {model_a_name}")
                    elif winner_id == model_b_id:
                        st.success(f"🏆 Gewinner: {model_b_name}")
                else:
                    st.info("🤝 Unentschieden")
                
                # Metriken kompakt - Erweitert
                st.markdown("**📊 Metriken:**")
                
                # Modell A Metriken
                st.markdown(f"**Modell A ({model_a_name}):**")
                metric_a_col1, metric_a_col2, metric_a_col3, metric_a_col4 = st.columns(4)
                with metric_a_col1:
                    st.metric("Accuracy", f"{a_accuracy:.3f}", help="Anteil korrekter Vorhersagen (0-1, höher = besser)", label_visibility="visible")
                with metric_a_col2:
                    st.metric("F1-Score", f"{a_f1:.3f}", help="Harmonisches Mittel aus Precision und Recall (0-1, höher = besser)", label_visibility="visible")
                with metric_a_col3:
                    if a_roc_auc:
                        st.metric("ROC-AUC", f"{a_roc_auc:.3f}", help="Area Under ROC Curve (0-1, >0.7 = gut)", label_visibility="visible")
                    else:
                        st.caption("ROC-AUC: N/A")
                with metric_a_col4:
                    if a_mcc:
                        st.metric("MCC", f"{a_mcc:.3f}", help="Matthews Correlation Coefficient (-1 bis +1, höher = besser)", label_visibility="visible")
                    elif a_profit:
                        st.metric("Profit", f"{a_profit:.2f}%", help="Simulierter Profit basierend auf TP/FP", label_visibility="visible")
                    else:
                        st.caption("MCC/Profit: N/A")
                
                # Modell B Metriken
                st.markdown(f"**Modell B ({model_b_name}):**")
                metric_b_col1, metric_b_col2, metric_b_col3, metric_b_col4 = st.columns(4)
                with metric_b_col1:
                    st.metric("Accuracy", f"{b_accuracy:.3f}", help="Anteil korrekter Vorhersagen (0-1, höher = besser)", label_visibility="visible")
                with metric_b_col2:
                    st.metric("F1-Score", f"{b_f1:.3f}", help="Harmonisches Mittel aus Precision und Recall (0-1, höher = besser)", label_visibility="visible")
                with metric_b_col3:
                    if b_roc_auc:
                        st.metric("ROC-AUC", f"{b_roc_auc:.3f}", help="Area Under ROC Curve (0-1, >0.7 = gut)", label_visibility="visible")
                    else:
                        st.caption("ROC-AUC: N/A")
                with metric_b_col4:
                    if b_mcc:
                        st.metric("MCC", f"{b_mcc:.3f}", help="Matthews Correlation Coefficient (-1 bis +1, höher = besser)", label_visibility="visible")
                    elif b_profit:
                        st.metric("Profit", f"{b_profit:.2f}%", help="Simulierter Profit basierend auf TP/FP", label_visibility="visible")
                    else:
                        st.caption("MCC/Profit: N/A")
                
                # Zusätzliche Infos
                info_row1, info_row2 = st.columns(2)
                
                with info_row1:
                    # Trainings-Zeiträume der Modelle
                    if model_a_train_start and model_a_train_end:
                        try:
                            train_start_dt = model_a_train_start if isinstance(model_a_train_start, str) else model_a_train_start
                            train_end_dt = model_a_train_end if isinstance(model_a_train_end, str) else model_a_train_end
                            if isinstance(train_start_dt, str):
                                train_start_dt = datetime.fromisoformat(train_start_dt.replace('Z', '+00:00'))
                            if isinstance(train_end_dt, str):
                                train_end_dt = datetime.fromisoformat(train_end_dt.replace('Z', '+00:00'))
                            
                            train_start_str = train_start_dt.strftime("%d.%m.%Y %H:%M")
                            train_end_str = train_end_dt.strftime("%d.%m.%Y %H:%M")
                            train_days = (train_end_dt - train_start_dt).total_seconds() / 86400.0
                            st.caption(f"🎓 Modell A Training: {train_start_str} → {train_end_str} ({train_days:.1f} Tage)")
                        except:
                            st.caption("🎓 Modell A Training: Zeitraum verfügbar")
                    
                    if model_b_train_start and model_b_train_end:
                        try:
                            train_start_dt = model_b_train_start if isinstance(model_b_train_start, str) else model_b_train_start
                            train_end_dt = model_b_train_end if isinstance(model_b_train_end, str) else model_b_train_end
                            if isinstance(train_start_dt, str):
                                train_start_dt = datetime.fromisoformat(train_start_dt.replace('Z', '+00:00'))
                            if isinstance(train_end_dt, str):
                                train_end_dt = datetime.fromisoformat(train_end_dt.replace('Z', '+00:00'))
                            
                            train_start_str = train_start_dt.strftime("%d.%m.%Y %H:%M")
                            train_end_str = train_end_dt.strftime("%d.%m.%Y %H:%M")
                            train_days = (train_end_dt - train_start_dt).total_seconds() / 86400.0
                            st.caption(f"🎓 Modell B Training: {train_start_str} → {train_end_str} ({train_days:.1f} Tage)")
                        except:
                            st.caption("🎓 Modell B Training: Zeitraum verfügbar")
                
                with info_row2:
                    # Test-Zeitraum mit Uhrzeit
                    if test_start and test_end:
                        try:
                            start_dt = test_start if isinstance(test_start, str) else test_start
                            end_dt = test_end if isinstance(test_end, str) else test_end
                            if isinstance(start_dt, str):
                                start_dt = datetime.fromisoformat(start_dt.replace('Z', '+00:00'))
                            if isinstance(end_dt, str):
                                end_dt = datetime.fromisoformat(end_dt.replace('Z', '+00:00'))
                            
                            start_str = start_dt.strftime("%d.%m.%Y %H:%M")
                            end_str = end_dt.strftime("%d.%m.%Y %H:%M")
                            days = (end_dt - start_dt).total_seconds() / 86400.0
                            st.caption(f"📅 Test: {start_str} → {end_str} ({days:.1f} Tage)")
                        except:
                            st.caption("📅 Test: Zeitraum verfügbar")
                    else:
                        st.caption("📅 Test: Zeitraum nicht verfügbar")
                    
                    # Erstellt-Datum mit Uhrzeit
                    created_raw = comp.get('created_at', '')
                    if created_raw:
                        try:
                            if isinstance(created_raw, str):
                                created_dt = datetime.fromisoformat(created_raw.replace('Z', '+00:00'))
                            else:
                                created_dt = created_raw
                            created_str = created_dt.strftime("%d.%m.%Y %H:%M")
                            st.caption(f"🕐 Erstellt: {created_str}")
                        except:
                            st.caption(f"🕐 Erstellt: {str(created_raw)[:19] if len(str(created_raw)) > 19 else str(created_raw)}")
                    else:
                        st.caption("🕐 Erstellt: N/A")
                
                # Dünne graue Linie zur Trennung
                if idx < len(comparisons) - 1:
                    st.markdown("<hr style='margin: 10px 0; border: none; border-top: 1px solid #e0e0e0;'>", unsafe_allow_html=True)
        
        # Zeige ausgewählte Vergleiche
        selected_comparison_ids = st.session_state.get('selected_comparison_ids', [])
        # Filtere nur existierende Vergleiche
        selected_comparison_ids = [cid for cid in selected_comparison_ids if any(c.get('id') == cid for c in comparisons)]
        # Aktualisiere session_state falls Vergleiche entfernt wurden
        if len(selected_comparison_ids) != len(st.session_state.get('selected_comparison_ids', [])):
            st.session_state['selected_comparison_ids'] = selected_comparison_ids
        
        selected_count = len(selected_comparison_ids)
        if selected_count > 0:
            st.divider()
            st.subheader(f"🔧 Aktionen ({selected_count} Vergleich(e) ausgewählt)")
            
            selected_comparisons = [c for c in comparisons if c.get('id') in selected_comparison_ids]
            
            # Zeige ausgewählte Vergleiche
            if selected_count <= 3:
                selected_names = [f"Vergleich {c.get('id')}" for c in selected_comparisons]
                st.info(f"📌 Ausgewählt: {', '.join(selected_names)}")
            
            # Aktionen
            col1, col2 = st.columns(2)
            
            with col1:
                if selected_count == 1:
                    comp_id = selected_comparison_ids[0]
                    if st.button("📋 Details anzeigen", key="btn_comp_details", use_container_width=True, type="primary"):
                        st.session_state['comparison_details_id'] = comp_id
                        st.session_state['page'] = 'comparison_details'
                        st.rerun()
            
            with col2:
                if st.button("🗑️ Löschen", key="btn_delete_comparisons", use_container_width=True, type="secondary"):
                    deleted_count = 0
                    failed_count = 0
                    ids_to_delete = list(selected_comparison_ids)
                    for comp_id in ids_to_delete:
                        if api_delete(f"/api/comparisons/{comp_id}"):
                            deleted_count += 1
                            if comp_id in st.session_state.get('selected_comparison_ids', []):
                                st.session_state['selected_comparison_ids'].remove(comp_id)
                        else:
                            failed_count += 1
                    
                    if deleted_count > 0:
                        st.success(f"✅ {deleted_count} Vergleich(e) gelöscht")
                    if failed_count > 0:
                        st.error(f"❌ {failed_count} Fehler")
                    
                    if deleted_count > 0:
                        st.rerun()

def page_test_results():
    """Test-Ergebnisse Übersicht: Liste aller Modell-Tests"""
    st.title("🧪 Test-Ergebnisse Übersicht")
    
    # Initialisiere selected_tests in session_state
    if 'selected_test_ids' not in st.session_state:
        st.session_state['selected_test_ids'] = []
    
    # Lade Test-Ergebnisse
    test_results = api_get("/api/test-results")
    if not test_results or not isinstance(test_results, list):
        st.warning("⚠️ Keine Test-Ergebnisse gefunden")
        return
    
    st.info(f"📊 {len(test_results)} Test-Ergebnis(se) gefunden")
    
    # Kompakte Karten-Ansicht
    if test_results:
        st.subheader("📋 Test-Ergebnisse")
        
        # Erstelle Karten in einem Grid (2 Spalten)
        cols = st.columns(2)
        
        for idx, test in enumerate(test_results):
            test_id = test.get('id')
            model_id = test.get('model_id')
            created = test.get('created_at', '')[:10] if test.get('created_at') else "N/A"
            test_start = test.get('test_start')
            test_end = test.get('test_end')
            
            # Hole Modell-Name und Trainings-Zeitraum
            model = api_get(f"/api/models/{model_id}")
            model_name = model.get('name', f"ID: {model_id}") if model else f"ID: {model_id}"
            model_train_start = model.get('train_start') if model else None
            model_train_end = model.get('train_end') if model else None
            
            # Metriken
            accuracy = test.get('accuracy', 0)
            f1 = test.get('f1_score', 0)
            mcc = test.get('mcc')
            profit = test.get('simulated_profit_pct')
            
            # Checkbox
            is_selected = test_id in st.session_state.get('selected_test_ids', [])
            checkbox_key = f"test_checkbox_{test_id}"
            
            # Wähle Spalte (abwechselnd)
            col = cols[idx % 2]
            
            with col:
                # Karte mit Border
                card_style = """
                <style>
                .test-card {
                    border: 2px solid #e0e0e0;
                    border-radius: 8px;
                    padding: 12px;
                    margin-bottom: 12px;
                    background: white;
                }
                .test-card.selected {
                    border-color: #1f77b4;
                    background: #f0f8ff;
                }
                </style>
                """
                st.markdown(card_style, unsafe_allow_html=True)
                
                # Header mit Checkbox und Titel
                header_col1, header_col2, header_col3 = st.columns([0.3, 4, 0.6])
                with header_col1:
                    checked = st.checkbox("", value=is_selected, key=checkbox_key, label_visibility="collapsed")
                    # Update session_state ohne st.rerun() - Streamlit rendert automatisch neu
                    if checked and test_id not in st.session_state.get('selected_test_ids', []):
                        if 'selected_test_ids' not in st.session_state:
                            st.session_state['selected_test_ids'] = []
                        st.session_state['selected_test_ids'].append(test_id)
                    elif not checked and test_id in st.session_state.get('selected_test_ids', []):
                        st.session_state['selected_test_ids'].remove(test_id)
                
                with header_col2:
                    st.markdown(f"**{model_name}**")
                
                with header_col3:
                    if st.button("📋", key=f"test_details_{test_id}", help="Details anzeigen", use_container_width=True):
                        st.session_state['test_details_id'] = test_id
                        st.session_state['page'] = 'test_details'
                        st.rerun()
                
                # Metriken kompakt - Erweitert
                metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
                with metric_col1:
                    st.metric("Accuracy", f"{accuracy:.3f}", help="Anteil korrekter Vorhersagen (0-1, höher = besser)", label_visibility="visible")
                with metric_col2:
                    st.metric("F1-Score", f"{f1:.3f}", help="Harmonisches Mittel aus Precision und Recall (0-1, höher = besser)", label_visibility="visible")
                with metric_col3:
                    roc_auc = test.get('roc_auc')
                    if roc_auc:
                        st.metric("ROC-AUC", f"{roc_auc:.3f}", help="Area Under ROC Curve (0-1, >0.7 = gut)", label_visibility="visible")
                    else:
                        st.caption("ROC-AUC: N/A")
                with metric_col4:
                    if mcc:
                        st.metric("MCC", f"{mcc:.3f}", help="Matthews Correlation Coefficient (-1 bis +1, höher = besser)", label_visibility="visible")
                    elif profit:
                        st.metric("Profit", f"{profit:.2f}%", help="Simulierter Profit basierend auf TP/FP", label_visibility="visible")
                    else:
                        st.caption("MCC/Profit: N/A")
                
                # Zusätzliche Infos
                info_row1, info_row2 = st.columns(2)
                
                with info_row1:
                    # Trainings-Zeitraum des Modells (mit Uhrzeit)
                    if model_train_start and model_train_end:
                        try:
                            train_start_dt = model_train_start if isinstance(model_train_start, str) else model_train_start
                            train_end_dt = model_train_end if isinstance(model_train_end, str) else model_train_end
                            if isinstance(train_start_dt, str):
                                train_start_dt = datetime.fromisoformat(train_start_dt.replace('Z', '+00:00'))
                            if isinstance(train_end_dt, str):
                                train_end_dt = datetime.fromisoformat(train_end_dt.replace('Z', '+00:00'))
                            
                            train_start_str = train_start_dt.strftime("%d.%m.%Y %H:%M")
                            train_end_str = train_end_dt.strftime("%d.%m.%Y %H:%M")
                            train_days = (train_end_dt - train_start_dt).total_seconds() / 86400.0
                            st.caption(f"🎓 Training: {train_start_str} → {train_end_str} ({train_days:.1f} Tage)")
                        except:
                            st.caption("🎓 Training: Zeitraum verfügbar")
                    else:
                        st.caption("🎓 Training: Zeitraum nicht verfügbar")
                    
                    # Test-Zeitraum mit Uhrzeit
                    if test_start and test_end:
                        try:
                            start_dt = test_start if isinstance(test_start, str) else test_start
                            end_dt = test_end if isinstance(test_end, str) else test_end
                            if isinstance(start_dt, str):
                                start_dt = datetime.fromisoformat(start_dt.replace('Z', '+00:00'))
                            if isinstance(end_dt, str):
                                end_dt = datetime.fromisoformat(end_dt.replace('Z', '+00:00'))
                            
                            start_str = start_dt.strftime("%d.%m.%Y %H:%M")
                            end_str = end_dt.strftime("%d.%m.%Y %H:%M")
                            days = (end_dt - start_dt).total_seconds() / 86400.0
                            st.caption(f"📅 Test: {start_str} → {end_str} ({days:.1f} Tage)")
                        except:
                            st.caption("📅 Test: Zeitraum verfügbar")
                    else:
                        st.caption("📅 Test: Zeitraum nicht verfügbar")
                    
                    # Overfitting Warnung
                    if test.get('is_overfitted'):
                        st.warning("⚠️ Möglicherweise overfitted")
                    elif test.get('accuracy_degradation'):
                        degradation = test.get('accuracy_degradation')
                        if degradation and degradation > 0.1:
                            st.warning(f"⚠️ Performance-Degradation: {degradation:.2%}")
                
                with info_row2:
                    # Anzahl Samples
                    num_samples = test.get('num_samples')
                    if num_samples:
                        st.caption(f"📊 {num_samples} Test-Samples")
                    
                    # Train vs. Test Vergleich
                    train_accuracy = test.get('train_accuracy')
                    if train_accuracy and accuracy:
                        degradation = test.get('accuracy_degradation', 0)
                        if degradation:
                            quality = "✅ OK" if degradation < 0.1 else "⚠️ Degradation"
                            st.caption(f"📈 Train: {train_accuracy:.3f} → Test: {accuracy:.3f} ({quality})")
                    
                    # Erstellt-Datum mit Uhrzeit
                    created_raw = test.get('created_at', '')
                    if created_raw:
                        try:
                            if isinstance(created_raw, str):
                                created_dt = datetime.fromisoformat(created_raw.replace('Z', '+00:00'))
                            else:
                                created_dt = created_raw
                            created_str = created_dt.strftime("%d.%m.%Y %H:%M")
                            st.caption(f"🕐 Erstellt: {created_str}")
                        except:
                            st.caption(f"🕐 Erstellt: {str(created_raw)[:19] if len(str(created_raw)) > 19 else str(created_raw)}")
                    else:
                        st.caption("🕐 Erstellt: N/A")
                
                # Dünne graue Linie zur Trennung
                if idx < len(test_results) - 1:
                    st.markdown("<hr style='margin: 10px 0; border: none; border-top: 1px solid #e0e0e0;'>", unsafe_allow_html=True)
        
        # Zeige ausgewählte Tests
        selected_test_ids = st.session_state.get('selected_test_ids', [])
        # Filtere nur existierende Tests
        selected_test_ids = [tid for tid in selected_test_ids if any(t.get('id') == tid for t in test_results)]
        # Aktualisiere session_state falls Tests entfernt wurden
        if len(selected_test_ids) != len(st.session_state.get('selected_test_ids', [])):
            st.session_state['selected_test_ids'] = selected_test_ids
        
        selected_count = len(selected_test_ids)
        if selected_count > 0:
            st.divider()
            st.subheader(f"🔧 Aktionen ({selected_count} Test(s) ausgewählt)")
            
            selected_tests = [t for t in test_results if t.get('id') in selected_test_ids]
            
            # Zeige ausgewählte Tests
            if selected_count <= 3:
                selected_names = [f"Test {t.get('id')}" for t in selected_tests]
                st.info(f"📌 Ausgewählt: {', '.join(selected_names)}")
            
            # Aktionen
            col1, col2 = st.columns(2)
            
            with col1:
                if selected_count == 1:
                    test_id = selected_test_ids[0]
                    if st.button("📋 Details anzeigen", key="btn_test_details", use_container_width=True, type="primary"):
                        st.session_state['test_details_id'] = test_id
                        st.session_state['page'] = 'test_details'
                        st.rerun()
            
            with col2:
                if st.button("🗑️ Löschen", key="btn_delete_tests", use_container_width=True, type="secondary"):
                    deleted_count = 0
                    failed_count = 0
                    ids_to_delete = list(selected_test_ids)
                    for test_id in ids_to_delete:
                        if api_delete(f"/api/test-results/{test_id}"):
                            deleted_count += 1
                            if test_id in st.session_state.get('selected_test_ids', []):
                                st.session_state['selected_test_ids'].remove(test_id)
                        else:
                            failed_count += 1
                    
                    if deleted_count > 0:
                        st.success(f"✅ {deleted_count} Test(s) gelöscht")
                    if failed_count > 0:
                        st.error(f"❌ {failed_count} Fehler")
                    
                    if deleted_count > 0:
                        st.rerun()

def page_test_details():
    """Details-Seite für ein Test-Ergebnis"""
    test_id = st.session_state.get('test_details_id')
    if not test_id:
        st.error("❌ Keine Test-ID gefunden")
        return
    
    test = api_get(f"/api/test-results/{test_id}")
    if not test:
        st.error("❌ Test-Ergebnis nicht gefunden")
        return
    
    # Hole Modell-Name und Details
    model_id = test.get('model_id')
    model = api_get(f"/api/models/{model_id}")
    model_name = model.get('name', f"ID: {model_id}") if model else f"ID: {model_id}"
    model_train_start = model.get('train_start') if model else None
    model_train_end = model.get('train_end') if model else None
    
    st.title(f"📋 Test-Details: {model_name}")
    
    # Info-Box am Anfang
    st.info("""
    **📖 Anleitung:** 
    Diese Seite zeigt alle Details und Metriken des Test-Ergebnisses. Nutze die ℹ️-Icons für Erklärungen zu jedem Wert.
    """)
    
    # Basis-Informationen
    st.subheader("📝 Basis-Informationen")
    info_col1, info_col2, info_col3, info_col4 = st.columns(4)
    with info_col1:
        st.markdown("**Modell-ID**")
        st.write(f"#{model_id}")
    with info_col2:
        st.markdown("**Modell-Name**")
        st.write(model_name)
    with info_col3:
        st.markdown("**Test-ID**")
        st.write(f"#{test_id}")
    with info_col4:
        st.markdown("**Erstellt**")
        created = test.get('created_at', '')
        if created:
            try:
                if isinstance(created, str):
                    created_dt = datetime.fromisoformat(created.replace('Z', '+00:00'))
                else:
                    created_dt = created
                st.write(created_dt.strftime("%d.%m.%Y %H:%M"))
            except:
                st.write(str(created)[:19] if len(str(created)) > 19 else str(created))
        else:
            st.write("N/A")
    
    st.divider()
    
    # Trainings-Zeitraum des Modells
    st.subheader("🎓 Trainings-Zeitraum des Modells")
    st.markdown("""
    **Was ist der Trainings-Zeitraum?**
    
    Der Trainings-Zeitraum zeigt, mit welchen historischen Daten das Modell trainiert wurde.
    Diese Daten wurden verwendet, um das Modell zu erstellen.
    """)
    
    if model_train_start and model_train_end:
        try:
            if isinstance(model_train_start, str):
                train_start_dt = datetime.fromisoformat(model_train_start.replace('Z', '+00:00'))
            else:
                train_start_dt = model_train_start
            if isinstance(model_train_end, str):
                train_end_dt = datetime.fromisoformat(model_train_end.replace('Z', '+00:00'))
            else:
                train_end_dt = model_train_end
            
            train_start_str = train_start_dt.strftime("%d.%m.%Y %H:%M")
            train_end_str = train_end_dt.strftime("%d.%m.%Y %H:%M")
            train_days = (train_end_dt - train_start_dt).total_seconds() / 86400.0
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.write(f"**Start:** {train_start_str}")
            with col2:
                st.write(f"**Ende:** {train_end_str}")
            with col3:
                st.write(f"**Dauer:** {train_days:.1f} Tage")
        except Exception as e:
            st.write(f"Start: {model_train_start}")
            st.write(f"Ende: {model_train_end}")
    else:
        st.write("Trainings-Zeitraum nicht verfügbar")
    
    st.divider()
    
    # Standard-Metriken mit Erklärungen
    st.subheader("📊 Standard-Metriken")
    st.markdown("""
    **Was bedeuten diese Metriken?**
    
    - **Accuracy:** Anteil korrekter Vorhersagen auf den Test-Daten (0-1). Beispiel: 0.85 = 85% der Vorhersagen sind richtig.
    - **F1-Score:** Harmonisches Mittel aus Precision und Recall (0-1). Gut für unausgewogene Daten.
    - **Precision:** Von allen "Positiv"-Vorhersagen, wie viele waren wirklich positiv? (0-1, höher = weniger False Positives)
    - **Recall:** Von allen echten Positiven, wie viele hat das Modell gefunden? (0-1, höher = weniger False Negatives)
    """)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        accuracy = test.get('accuracy')
        if accuracy:
            st.metric("Accuracy", f"{accuracy:.4f}", help="Anteil korrekter Vorhersagen auf Test-Daten (0-1, höher = besser)")
        else:
            st.caption("Accuracy: N/A")
    with col2:
        f1 = test.get('f1_score')
        if f1:
            st.metric("F1-Score", f"{f1:.4f}", help="Harmonisches Mittel aus Precision und Recall (0-1, höher = besser)")
        else:
            st.caption("F1-Score: N/A")
    with col3:
        precision = test.get('precision_score')
        if precision:
            st.metric("Precision", f"{precision:.4f}", help="Von allen 'Positiv'-Vorhersagen, wie viele waren wirklich positiv? (0-1, höher = weniger False Positives)")
        else:
            st.caption("Precision: N/A")
    with col4:
        recall = test.get('recall')
        if recall:
            st.metric("Recall", f"{recall:.4f}", help="Von allen echten Positiven, wie viele hat das Modell gefunden? (0-1, höher = weniger False Negatives)")
        else:
            st.caption("Recall: N/A")
    
    st.divider()
    
    # Zusätzliche Metriken mit Erklärungen
    if test.get('roc_auc') or test.get('mcc') or test.get('fpr') or test.get('fnr') or test.get('simulated_profit_pct'):
        st.subheader("📈 Erweiterte Metriken")
        st.markdown("""
        **Was bedeuten diese Metriken?**
        
        - **ROC-AUC:** Area Under ROC Curve (0-1). Misst die Fähigkeit, zwischen Positiv und Negativ zu unterscheiden. >0.7 = gut, >0.9 = sehr gut.
        - **MCC:** Matthews Correlation Coefficient (-1 bis +1). Berücksichtigt alle 4 Confusion-Matrix-Werte. 0 = zufällig, +1 = perfekt, -1 = perfekt falsch.
        - **FPR (False Positive Rate):** Wie viele Negatives wurden fälschlicherweise als Positiv klassifiziert? (0-1, niedriger = besser)
        - **FNR (False Negative Rate):** Wie viele Positives wurden fälschlicherweise als Negativ klassifiziert? (0-1, niedriger = besser)
        """)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            roc_auc = test.get('roc_auc')
            if roc_auc:
                quality = "Sehr gut" if roc_auc > 0.9 else "Gut" if roc_auc > 0.7 else "Mäßig" if roc_auc > 0.5 else "Schlecht"
                st.metric("ROC-AUC", f"{roc_auc:.4f}", help=f"Area Under ROC Curve (0-1). {quality} (>0.7 = gut, >0.9 = sehr gut)")
            else:
                st.caption("ROC-AUC: N/A")
        with col2:
            mcc = test.get('mcc')
            if mcc:
                quality = "Sehr gut" if mcc > 0.5 else "Gut" if mcc > 0.3 else "Mäßig" if mcc > 0 else "Schlecht"
                st.metric("MCC", f"{mcc:.4f}", help=f"Matthews Correlation Coefficient (-1 bis +1). {quality} (0 = zufällig, +1 = perfekt)")
            else:
                st.caption("MCC: N/A")
        with col3:
            fpr = test.get('fpr')
            if fpr is not None:
                quality = "Gut" if fpr < 0.1 else "Mäßig" if fpr < 0.3 else "Schlecht"
                st.metric("False Positive Rate", f"{fpr:.4f}", help=f"Wie viele Negatives wurden fälschlicherweise als Positiv klassifiziert? {quality} (niedriger = besser)")
            else:
                st.caption("FPR: N/A")
        with col4:
            fnr = test.get('fnr')
            if fnr is not None:
                quality = "Gut" if fnr < 0.1 else "Mäßig" if fnr < 0.3 else "Schlecht"
                st.metric("False Negative Rate", f"{fnr:.4f}", help=f"Wie viele Positives wurden fälschlicherweise als Negativ klassifiziert? {quality} (niedriger = besser)")
            else:
                st.caption("FNR: N/A")
        
        # Profit-Simulation
        simulated_profit = test.get('simulated_profit_pct')
        if simulated_profit is not None:
            st.divider()
            st.markdown("**💰 Profit-Simulation:**")
            st.markdown("""
            **Was ist Profit-Simulation?**
            
            Simuliert den Profit, den das Modell erzielt hätte:
            - **True Positive (TP):** +1% Gewinn (korrekt erkannte Pumps)
            - **False Positive (FP):** -0.5% Verlust (fälschlicherweise als Pump erkannt)
            - **True Negative (TN):** 0% (korrekt als "kein Pump" erkannt)
            - **False Negative (FN):** 0% (verpasste Pumps)
            
            **Auswirkung:** Zeigt, wie profitabel das Modell in der Praxis wäre.
            """)
            
            col1, col2 = st.columns(2)
            with col1:
                profit_quality = "Sehr profitabel" if simulated_profit > 5 else "Profitabel" if simulated_profit > 0 else "Verlust"
                st.metric("💰 Simulierter Profit", f"{simulated_profit:.2f}%", help=f"Simulierter Profit basierend auf TP/FP. {profit_quality}")
            with col2:
                st.caption("**Berechnung:** 1% Gewinn pro TP, -0.5% Verlust pro FP")
    
    st.divider()
    
    # Confusion Matrix mit Erklärung
    confusion_matrix = test.get('confusion_matrix')
    if confusion_matrix:
        st.subheader("🔢 Confusion Matrix")
        st.markdown("""
        **Was ist eine Confusion Matrix?**
        
        Zeigt, wie viele Vorhersagen korrekt und falsch waren:
        - **TP (True Positive):** ✅ Korrekt als "Positiv" erkannt (z.B. Pump erkannt, war wirklich Pump)
        - **TN (True Negative):** ✅ Korrekt als "Negativ" erkannt (z.B. kein Pump erkannt, war wirklich kein Pump)
        - **FP (False Positive):** ❌ Fälschlicherweise als "Positiv" erkannt (z.B. Pump erkannt, war aber kein Pump) → Verluste!
        - **FN (False Negative):** ❌ Fälschlicherweise als "Negativ" erkannt (z.B. kein Pump erkannt, war aber Pump) → Verpasste Chancen!
        
        **Auswirkung:** 
        - Viele TP = Modell erkennt Pumps gut
        - Viele FP = Modell ist zu optimistisch (viele Fehlalarme)
        - Viele FN = Modell verpasst viele Pumps
        """)
        
        cm = confusion_matrix
        cm_col1, cm_col2, cm_col3, cm_col4 = st.columns(4)
        with cm_col1:
            tp = cm.get('tp', 0)
            st.metric("True Positive (TP)", tp, help="✅ Korrekt als 'Positiv' erkannt (z.B. Pump erkannt, war wirklich Pump)")
        with cm_col2:
            tn = cm.get('tn', 0)
            st.metric("True Negative (TN)", tn, help="✅ Korrekt als 'Negativ' erkannt (z.B. kein Pump erkannt, war wirklich kein Pump)")
        with cm_col3:
            fp = cm.get('fp', 0)
            st.metric("False Positive (FP)", fp, help="❌ Fälschlicherweise als 'Positiv' erkannt (z.B. Pump erkannt, war aber kein Pump) → Verluste!")
        with cm_col4:
            fn = cm.get('fn', 0)
            st.metric("False Negative (FN)", fn, help="❌ Fälschlicherweise als 'Negativ' erkannt (z.B. kein Pump erkannt, war aber Pump) → Verpasste Chancen!")
        
        # Visualisierung als Tabelle
        st.markdown("**Confusion Matrix Tabelle:**")
        cm_data = {
            'Tatsächlich': ['Negativ', 'Positiv'],
            'Vorhergesagt: Negativ': [tn, fn],
            'Vorhergesagt: Positiv': [fp, tp]
        }
        cm_df = pd.DataFrame(cm_data)
        st.dataframe(cm_df, use_container_width=True, hide_index=True)
        
        # Interpretation
        total = tp + tn + fp + fn
        if total > 0:
            tp_rate = (tp / total) * 100
            fp_rate = (fp / total) * 100
            fn_rate = (fn / total) * 100
            st.caption(f"ℹ️ Verteilung: {tp_rate:.1f}% TP, {tn/total*100:.1f}% TN, {fp_rate:.1f}% FP, {fn_rate:.1f}% FN")
    
    st.divider()
    
    # Trainings-Zeitraum des Modells (wenn noch nicht angezeigt)
    if model_train_start and model_train_end:
        st.subheader("🎓 Trainings-Zeitraum des Modells")
        st.markdown("""
        **Was ist der Trainings-Zeitraum?**
        
        Der Trainings-Zeitraum zeigt, mit welchen historischen Daten das Modell trainiert wurde.
        Diese Daten wurden verwendet, um das Modell zu erstellen.
        """)
        
        try:
            if isinstance(model_train_start, str):
                train_start_dt = datetime.fromisoformat(model_train_start.replace('Z', '+00:00'))
            else:
                train_start_dt = model_train_start
            if isinstance(model_train_end, str):
                train_end_dt = datetime.fromisoformat(model_train_end.replace('Z', '+00:00'))
            else:
                train_end_dt = model_train_end
            
            train_start_str = train_start_dt.strftime("%d.%m.%Y %H:%M")
            train_end_str = train_end_dt.strftime("%d.%m.%Y %H:%M")
            train_days = (train_end_dt - train_start_dt).total_seconds() / 86400.0
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.write(f"**Start:** {train_start_str}")
            with col2:
                st.write(f"**Ende:** {train_end_str}")
            with col3:
                st.write(f"**Dauer:** {train_days:.1f} Tage")
        except Exception as e:
            st.write(f"Start: {model_train_start}")
            st.write(f"Ende: {model_train_end}")
        
        st.divider()
    
    # Train vs. Test Vergleich mit Erklärung
    if test.get('train_accuracy') is not None:
        st.subheader("📊 Train vs. Test Vergleich")
        st.markdown("""
        **Was bedeutet Train vs. Test Vergleich?**
        
        Vergleicht die Performance auf Trainings- und Test-Daten:
        - **Train Accuracy:** Performance auf den Daten, mit denen das Modell trainiert wurde
        - **Test Accuracy:** Performance auf neuen, ungesehenen Daten
        - **Degradation:** Unterschied zwischen Train- und Test-Accuracy
        
        **Auswirkung:** 
        - Große Degradation (>10%) = Modell ist möglicherweise overfitted (lernt zu spezifisch)
        - Kleine Degradation (<10%) = Modell generalisiert gut auf neue Daten
        """)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            train_acc = test.get('train_accuracy')
            if train_acc:
                st.metric("Train Accuracy", f"{train_acc:.4f}", help="Performance auf Trainingsdaten (0-1, höher = besser)")
            else:
                st.caption("Train Accuracy: N/A")
        with col2:
            test_acc = test.get('accuracy')
            if test_acc:
                st.metric("Test Accuracy", f"{test_acc:.4f}", help="Performance auf Test-Daten (0-1, höher = besser)")
            else:
                st.caption("Test Accuracy: N/A")
        with col3:
            degradation = test.get('accuracy_degradation')
            if degradation is not None:
                quality = "✅ OK" if degradation < 0.1 else "⚠️ Overfitting-Risiko"
                st.metric("Degradation", f"{degradation:.4f}", 
                         delta=quality,
                         help=f"Unterschied zwischen Train- und Test-Accuracy. {quality} (niedriger = besser)")
            else:
                st.caption("Degradation: N/A")
        
        if test.get('is_overfitted'):
            st.warning("⚠️ Modell ist möglicherweise overfitted! Die Performance auf Test-Daten ist deutlich schlechter als auf Trainingsdaten.")
    
    st.divider()
    
    # Test-Zeitraum mit Erklärung
    st.subheader("📅 Test-Zeitraum")
    st.markdown("""
    **Was ist der Test-Zeitraum?**
    
    Der Test-Zeitraum definiert, welche Daten zum Testen des Modells verwendet wurden.
    Diese Daten wurden **nicht** zum Training verwendet.
    
    **Empfehlung:** Mindestens 1 Tag Test-Daten für realistische Ergebnisse.
    """)
    
    test_start = test.get('test_start')
    test_end = test.get('test_end')
    test_duration_days = test.get('test_duration_days')
    
    if test_start and test_end:
        try:
            if isinstance(test_start, str):
                start_dt = datetime.fromisoformat(test_start.replace('Z', '+00:00'))
            else:
                start_dt = test_start
            if isinstance(test_end, str):
                end_dt = datetime.fromisoformat(test_end.replace('Z', '+00:00'))
            else:
                end_dt = test_end
            
            start_str = start_dt.strftime("%d.%m.%Y %H:%M")
            end_str = end_dt.strftime("%d.%m.%Y %H:%M")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.write(f"**Start:** {start_str}")
            with col2:
                st.write(f"**Ende:** {end_str}")
            with col3:
                if test_duration_days:
                    st.write(f"**Dauer:** {test_duration_days:.2f} Tage")
                    if test_duration_days < 1:
                        st.warning("⚠️ Test-Zeitraum zu kurz (empfohlen: mindestens 1 Tag)")
                else:
                    days = (end_dt - start_dt).total_seconds() / 86400.0
                    st.write(f"**Dauer:** {days:.2f} Tage")
                    if days < 1:
                        st.warning("⚠️ Test-Zeitraum zu kurz (empfohlen: mindestens 1 Tag)")
        except Exception as e:
            st.write(f"Start: {test_start}")
            st.write(f"Ende: {test_end}")
    else:
        st.write("Test-Zeitraum nicht verfügbar")
    
    # Anzahl Samples
    num_samples = test.get('num_samples')
    num_positive = test.get('num_positive')
    num_negative = test.get('num_negative')
    if num_samples:
        st.markdown("**Test-Daten:**")
        st.write(f"**Anzahl Samples:** {num_samples}")
        if num_positive is not None and num_negative is not None:
            positive_pct = (num_positive / num_samples) * 100 if num_samples > 0 else 0
            st.write(f"**Positiv:** {num_positive} ({positive_pct:.1f}%)")
            st.write(f"**Negativ:** {num_negative} ({100-positive_pct:.1f}%)")
    
    st.divider()
    
    # Overlap Warnung
    if test.get('has_overlap'):
        st.warning(f"⚠️ **Überschneidung mit Trainingsdaten:** {test.get('overlap_note', 'Die Test-Daten überschneiden sich mit den Trainingsdaten. Dies kann zu unrealistisch guten Ergebnissen führen!')}")
    
    st.divider()
    
    # Vollständige Details
    with st.expander("📋 Vollständige Details (JSON)", expanded=False):
        st.json(test)
    
    # Zurück-Button
    col1, col2 = st.columns([1, 3])
    with col1:
        if st.button("← Zurück zur Test-Übersicht", use_container_width=True):
            st.session_state['page'] = 'test_results'
            st.session_state.pop('test_details_id', None)
            st.rerun()

def page_comparison_details():
    """Details-Seite für einen Vergleich"""
    comparison_id = st.session_state.get('comparison_details_id')
    if not comparison_id:
        st.error("❌ Keine Vergleichs-ID gefunden")
        return
    
    comparison = api_get(f"/api/comparisons/{comparison_id}")
    if not comparison:
        st.error("❌ Vergleich nicht gefunden")
        return
    
    # Hole Modell-Namen und Details
    model_a_id = comparison.get('model_a_id')
    model_b_id = comparison.get('model_b_id')
    model_a = api_get(f"/api/models/{model_a_id}")
    model_b = api_get(f"/api/models/{model_b_id}")
    model_a_name = model_a.get('name', f"ID: {model_a_id}") if model_a else f"ID: {model_a_id}"
    model_b_name = model_b.get('name', f"ID: {model_b_id}") if model_b else f"ID: {model_b_id}"
    model_a_train_start = model_a.get('train_start') if model_a else None
    model_a_train_end = model_a.get('train_end') if model_a else None
    model_b_train_start = model_b.get('train_start') if model_b else None
    model_b_train_end = model_b.get('train_end') if model_b else None
    
    st.title(f"📋 Vergleichs-Details: {model_a_name} vs {model_b_name}")
    
    # Info-Box am Anfang
    st.info("""
    **📖 Anleitung:** 
    Diese Seite zeigt alle Details und Metriken des Modell-Vergleichs. Nutze die ℹ️-Icons für Erklärungen zu jedem Wert.
    """)
    
    # Basis-Informationen
    st.subheader("📝 Basis-Informationen")
    info_col1, info_col2, info_col3, info_col4 = st.columns(4)
    with info_col1:
        st.markdown("**Vergleichs-ID**")
        st.write(f"#{comparison_id}")
    with info_col2:
        st.markdown("**Modell A**")
        st.write(f"{model_a_name} (#{model_a_id})")
    with info_col3:
        st.markdown("**Modell B**")
        st.write(f"{model_b_name} (#{model_b_id})")
    with info_col4:
        st.markdown("**Erstellt**")
        created = comparison.get('created_at', '')
        if created:
            try:
                if isinstance(created, str):
                    created_dt = datetime.fromisoformat(created.replace('Z', '+00:00'))
                else:
                    created_dt = created
                st.write(created_dt.strftime("%d.%m.%Y %H:%M"))
            except:
                st.write(str(created)[:19] if len(str(created)) > 19 else str(created))
        else:
            st.write("N/A")
    
    st.divider()
    
    # Gewinner
    winner_id = comparison.get('winner_id')
    if winner_id:
        if winner_id == model_a_id:
            st.success(f"🏆 **Gewinner: {model_a_name}** - Dieses Modell hat die bessere Performance auf den Test-Daten erzielt.")
        elif winner_id == model_b_id:
            st.success(f"🏆 **Gewinner: {model_b_name}** - Dieses Modell hat die bessere Performance auf den Test-Daten erzielt.")
    else:
        st.info("🤝 **Unentschieden** - Beide Modelle haben eine ähnliche Performance erzielt.")
    
    st.divider()
    
    # Trainings-Zeiträume beider Modelle
    st.subheader("🎓 Trainings-Zeiträume der Modelle")
    st.markdown("""
    **Was sind die Trainings-Zeiträume?**
    
    Die Trainings-Zeiträume zeigen, mit welchen historischen Daten die Modelle trainiert wurden.
    Diese Daten wurden verwendet, um die Modelle zu erstellen.
    """)
    
    train_col1, train_col2 = st.columns(2)
    
    with train_col1:
        st.markdown(f"**Modell A ({model_a_name}):**")
        if model_a_train_start and model_a_train_end:
            try:
                if isinstance(model_a_train_start, str):
                    train_start_dt = datetime.fromisoformat(model_a_train_start.replace('Z', '+00:00'))
                else:
                    train_start_dt = model_a_train_start
                if isinstance(model_a_train_end, str):
                    train_end_dt = datetime.fromisoformat(model_a_train_end.replace('Z', '+00:00'))
                else:
                    train_end_dt = model_a_train_end
                
                train_start_str = train_start_dt.strftime("%d.%m.%Y %H:%M")
                train_end_str = train_end_dt.strftime("%d.%m.%Y %H:%M")
                train_days = (train_end_dt - train_start_dt).total_seconds() / 86400.0
                
                st.write(f"**Start:** {train_start_str}")
                st.write(f"**Ende:** {train_end_str}")
                st.write(f"**Dauer:** {train_days:.1f} Tage")
            except:
                st.write(f"Start: {model_a_train_start}")
                st.write(f"Ende: {model_a_train_end}")
        else:
            st.write("Trainings-Zeitraum nicht verfügbar")
    
    with train_col2:
        st.markdown(f"**Modell B ({model_b_name}):**")
        if model_b_train_start and model_b_train_end:
            try:
                if isinstance(model_b_train_start, str):
                    train_start_dt = datetime.fromisoformat(model_b_train_start.replace('Z', '+00:00'))
                else:
                    train_start_dt = model_b_train_start
                if isinstance(model_b_train_end, str):
                    train_end_dt = datetime.fromisoformat(model_b_train_end.replace('Z', '+00:00'))
                else:
                    train_end_dt = model_b_train_end
                
                train_start_str = train_start_dt.strftime("%d.%m.%Y %H:%M")
                train_end_str = train_end_dt.strftime("%d.%m.%Y %H:%M")
                train_days = (train_end_dt - train_start_dt).total_seconds() / 86400.0
                
                st.write(f"**Start:** {train_start_str}")
                st.write(f"**Ende:** {train_end_str}")
                st.write(f"**Dauer:** {train_days:.1f} Tage")
            except:
                st.write(f"Start: {model_b_train_start}")
                st.write(f"Ende: {model_b_train_end}")
        else:
            st.write("Trainings-Zeitraum nicht verfügbar")
    
    st.divider()
    
    # Standard-Metriken mit Erklärungen
    st.subheader("📊 Standard-Metriken")
    st.markdown("""
    **Was bedeuten diese Metriken?**
    
    - **Accuracy:** Anteil korrekter Vorhersagen auf den Test-Daten (0-1). Beispiel: 0.85 = 85% der Vorhersagen sind richtig.
    - **F1-Score:** Harmonisches Mittel aus Precision und Recall (0-1). Gut für unausgewogene Daten.
    - **Precision:** Von allen "Positiv"-Vorhersagen, wie viele waren wirklich positiv? (0-1, höher = weniger False Positives)
    - **Recall:** Von allen echten Positiven, wie viele hat das Modell gefunden? (0-1, höher = weniger False Negatives)
    """)
    
    # Modell A Metriken
    st.markdown("**Modell A:**")
    col_a1, col_a2, col_a3, col_a4 = st.columns(4)
    with col_a1:
        a_acc = comparison.get('a_accuracy')
        if a_acc:
            st.metric("Accuracy", f"{a_acc:.4f}", help="Anteil korrekter Vorhersagen auf Test-Daten (0-1, höher = besser)")
        else:
            st.caption("Accuracy: N/A")
    with col_a2:
        a_f1 = comparison.get('a_f1')
        if a_f1:
            st.metric("F1-Score", f"{a_f1:.4f}", help="Harmonisches Mittel aus Precision und Recall (0-1, höher = besser)")
        else:
            st.caption("F1-Score: N/A")
    with col_a3:
        a_prec = comparison.get('a_precision')
        if a_prec:
            st.metric("Precision", f"{a_prec:.4f}", help="Von allen 'Positiv'-Vorhersagen, wie viele waren wirklich positiv? (0-1, höher = weniger False Positives)")
        else:
            st.caption("Precision: N/A")
    with col_a4:
        a_rec = comparison.get('a_recall')
        if a_rec:
            st.metric("Recall", f"{a_rec:.4f}", help="Von allen echten Positiven, wie viele hat das Modell gefunden? (0-1, höher = weniger False Negatives)")
        else:
            st.caption("Recall: N/A")
    
    st.divider()
    
    # Modell B Metriken
    st.markdown("**Modell B:**")
    col_b1, col_b2, col_b3, col_b4 = st.columns(4)
    with col_b1:
        b_acc = comparison.get('b_accuracy')
        if b_acc:
            st.metric("Accuracy", f"{b_acc:.4f}", help="Anteil korrekter Vorhersagen auf Test-Daten (0-1, höher = besser)")
        else:
            st.caption("Accuracy: N/A")
    with col_b2:
        b_f1 = comparison.get('b_f1')
        if b_f1:
            st.metric("F1-Score", f"{b_f1:.4f}", help="Harmonisches Mittel aus Precision und Recall (0-1, höher = besser)")
        else:
            st.caption("F1-Score: N/A")
    with col_b3:
        b_prec = comparison.get('b_precision')
        if b_prec:
            st.metric("Precision", f"{b_prec:.4f}", help="Von allen 'Positiv'-Vorhersagen, wie viele waren wirklich positiv? (0-1, höher = weniger False Positives)")
        else:
            st.caption("Precision: N/A")
    with col_b4:
        b_rec = comparison.get('b_recall')
        if b_rec:
            st.metric("Recall", f"{b_rec:.4f}", help="Von allen echten Positiven, wie viele hat das Modell gefunden? (0-1, höher = weniger False Negatives)")
        else:
            st.caption("Recall: N/A")
    
    st.divider()
    
    # Zusätzliche Metriken mit Erklärungen
    if comparison.get('a_roc_auc') or comparison.get('a_mcc') or comparison.get('a_fpr') or comparison.get('a_fnr') or comparison.get('a_simulated_profit_pct') or comparison.get('b_roc_auc') or comparison.get('b_mcc') or comparison.get('b_fpr') or comparison.get('b_fnr') or comparison.get('b_simulated_profit_pct'):
        st.subheader("📈 Erweiterte Metriken")
        st.markdown("""
        **Was bedeuten diese Metriken?**
        
        - **ROC-AUC:** Area Under ROC Curve (0-1). Misst die Fähigkeit, zwischen Positiv und Negativ zu unterscheiden. >0.7 = gut, >0.9 = sehr gut.
        - **MCC:** Matthews Correlation Coefficient (-1 bis +1). Berücksichtigt alle 4 Confusion-Matrix-Werte. 0 = zufällig, +1 = perfekt, -1 = perfekt falsch.
        - **FPR (False Positive Rate):** Wie viele Negatives wurden fälschlicherweise als Positiv klassifiziert? (0-1, niedriger = besser)
        - **FNR (False Negative Rate):** Wie viele Positives wurden fälschlicherweise als Negativ klassifiziert? (0-1, niedriger = besser)
        """)
        
        # Modell A erweiterte Metriken
        st.markdown("**Modell A:**")
        col_a1, col_a2, col_a3, col_a4 = st.columns(4)
        with col_a1:
            a_roc_auc = comparison.get('a_roc_auc')
            if a_roc_auc:
                quality = "Sehr gut" if a_roc_auc > 0.9 else "Gut" if a_roc_auc > 0.7 else "Mäßig" if a_roc_auc > 0.5 else "Schlecht"
                st.metric("ROC-AUC", f"{a_roc_auc:.4f}", help=f"Area Under ROC Curve (0-1). {quality} (>0.7 = gut, >0.9 = sehr gut)")
            else:
                st.caption("ROC-AUC: N/A")
        with col_a2:
            a_mcc = comparison.get('a_mcc')
            if a_mcc:
                quality = "Sehr gut" if a_mcc > 0.5 else "Gut" if a_mcc > 0.3 else "Mäßig" if a_mcc > 0 else "Schlecht"
                st.metric("MCC", f"{a_mcc:.4f}", help=f"Matthews Correlation Coefficient (-1 bis +1). {quality} (0 = zufällig, +1 = perfekt)")
            else:
                st.caption("MCC: N/A")
        with col_a3:
            a_fpr = comparison.get('a_fpr')
            if a_fpr is not None:
                quality = "Gut" if a_fpr < 0.1 else "Mäßig" if a_fpr < 0.3 else "Schlecht"
                st.metric("False Positive Rate", f"{a_fpr:.4f}", help=f"Wie viele Negatives wurden fälschlicherweise als Positiv klassifiziert? {quality} (niedriger = besser)")
            else:
                st.caption("FPR: N/A")
        with col_a4:
            a_fnr = comparison.get('a_fnr')
            if a_fnr is not None:
                quality = "Gut" if a_fnr < 0.1 else "Mäßig" if a_fnr < 0.3 else "Schlecht"
                st.metric("False Negative Rate", f"{a_fnr:.4f}", help=f"Wie viele Positives wurden fälschlicherweise als Negativ klassifiziert? {quality} (niedriger = besser)")
            else:
                st.caption("FNR: N/A")
        
        # Profit für Modell A
        a_profit = comparison.get('a_simulated_profit_pct')
        if a_profit is not None:
            profit_quality = "Sehr profitabel" if a_profit > 5 else "Profitabel" if a_profit > 0 else "Verlust"
            st.caption(f"💰 Simulierter Profit Modell A: {a_profit:.2f}% ({profit_quality})")
        
        st.divider()
        
        # Modell B erweiterte Metriken
        st.markdown("**Modell B:**")
        col_b1, col_b2, col_b3, col_b4 = st.columns(4)
        with col_b1:
            b_roc_auc = comparison.get('b_roc_auc')
            if b_roc_auc:
                quality = "Sehr gut" if b_roc_auc > 0.9 else "Gut" if b_roc_auc > 0.7 else "Mäßig" if b_roc_auc > 0.5 else "Schlecht"
                st.metric("ROC-AUC", f"{b_roc_auc:.4f}", help=f"Area Under ROC Curve (0-1). {quality} (>0.7 = gut, >0.9 = sehr gut)")
            else:
                st.caption("ROC-AUC: N/A")
        with col_b2:
            b_mcc = comparison.get('b_mcc')
            if b_mcc:
                quality = "Sehr gut" if b_mcc > 0.5 else "Gut" if b_mcc > 0.3 else "Mäßig" if b_mcc > 0 else "Schlecht"
                st.metric("MCC", f"{b_mcc:.4f}", help=f"Matthews Correlation Coefficient (-1 bis +1). {quality} (0 = zufällig, +1 = perfekt)")
            else:
                st.caption("MCC: N/A")
        with col_b3:
            b_fpr = comparison.get('b_fpr')
            if b_fpr is not None:
                quality = "Gut" if b_fpr < 0.1 else "Mäßig" if b_fpr < 0.3 else "Schlecht"
                st.metric("False Positive Rate", f"{b_fpr:.4f}", help=f"Wie viele Negatives wurden fälschlicherweise als Positiv klassifiziert? {quality} (niedriger = besser)")
            else:
                st.caption("FPR: N/A")
        with col_b4:
            b_fnr = comparison.get('b_fnr')
            if b_fnr is not None:
                quality = "Gut" if b_fnr < 0.1 else "Mäßig" if b_fnr < 0.3 else "Schlecht"
                st.metric("False Negative Rate", f"{b_fnr:.4f}", help=f"Wie viele Positives wurden fälschlicherweise als Negativ klassifiziert? {quality} (niedriger = besser)")
            else:
                st.caption("FNR: N/A")
        
        # Profit für Modell B
        b_profit = comparison.get('b_simulated_profit_pct')
        if b_profit is not None:
            profit_quality = "Sehr profitabel" if b_profit > 5 else "Profitabel" if b_profit > 0 else "Verlust"
            st.caption(f"💰 Simulierter Profit Modell B: {b_profit:.2f}% ({profit_quality})")
        
        # Profit-Vergleich
        if a_profit is not None and b_profit is not None:
            st.divider()
            st.markdown("**💰 Profit-Vergleich:**")
            st.markdown("""
            **Was ist Profit-Simulation?**
            
            Simuliert den Profit, den das Modell erzielt hätte:
            - **True Positive (TP):** +1% Gewinn (korrekt erkannte Pumps)
            - **False Positive (FP):** -0.5% Verlust (fälschlicherweise als Pump erkannt)
            - **True Negative (TN):** 0% (korrekt als "kein Pump" erkannt)
            - **False Negative (FN):** 0% (verpasste Pumps)
            
            **Auswirkung:** Zeigt, welches Modell in der Praxis profitabler wäre.
            """)
            
            profit_diff = a_profit - b_profit
            if profit_diff > 0:
                st.success(f"📈 **Modell A ist profitabler:** {profit_diff:.2f}% mehr Profit als Modell B")
            elif profit_diff < 0:
                st.success(f"📈 **Modell B ist profitabler:** {abs(profit_diff):.2f}% mehr Profit als Modell A")
            else:
                st.info("🤝 **Gleich profitabel:** Beide Modelle hätten den gleichen Profit erzielt")
    
    st.divider()
    
    # Confusion Matrix mit Erklärung
    if comparison.get('a_confusion_matrix') or comparison.get('b_confusion_matrix'):
        st.subheader("🔢 Confusion Matrix")
        st.markdown("""
        **Was ist eine Confusion Matrix?**
        
        Zeigt, wie viele Vorhersagen korrekt und falsch waren:
        - **TP (True Positive):** ✅ Korrekt als "Positiv" erkannt (z.B. Pump erkannt, war wirklich Pump)
        - **TN (True Negative):** ✅ Korrekt als "Negativ" erkannt (z.B. kein Pump erkannt, war wirklich kein Pump)
        - **FP (False Positive):** ❌ Fälschlicherweise als "Positiv" erkannt (z.B. Pump erkannt, war aber kein Pump) → Verluste!
        - **FN (False Negative):** ❌ Fälschlicherweise als "Negativ" erkannt (z.B. kein Pump erkannt, war aber Pump) → Verpasste Chancen!
        
        **Auswirkung:** 
        - Viele TP = Modell erkennt Pumps gut
        - Viele FP = Modell ist zu optimistisch (viele Fehlalarme)
        - Viele FN = Modell verpasst viele Pumps
        """)
        
        col_a, col_b = st.columns(2)
        
        with col_a:
            st.markdown(f"**Modell A ({model_a_name}):**")
            if comparison.get('a_confusion_matrix'):
                cm_a = comparison['a_confusion_matrix']
                cm_a_col1, cm_a_col2, cm_a_col3, cm_a_col4 = st.columns(4)
                with cm_a_col1:
                    tp_a = cm_a.get('tp', 0)
                    st.metric("TP", tp_a, help="✅ Korrekt als 'Positiv' erkannt")
                with cm_a_col2:
                    tn_a = cm_a.get('tn', 0)
                    st.metric("TN", tn_a, help="✅ Korrekt als 'Negativ' erkannt")
                with cm_a_col3:
                    fp_a = cm_a.get('fp', 0)
                    st.metric("FP", fp_a, help="❌ Fälschlicherweise als 'Positiv' erkannt → Verluste!")
                with cm_a_col4:
                    fn_a = cm_a.get('fn', 0)
                    st.metric("FN", fn_a, help="❌ Fälschlicherweise als 'Negativ' erkannt → Verpasste Chancen!")
                
                # Tabelle
                cm_a_data = {
                    'Tatsächlich': ['Negativ', 'Positiv'],
                    'Vorhergesagt: Negativ': [tn_a, fn_a],
                    'Vorhergesagt: Positiv': [fp_a, tp_a]
                }
                cm_a_df = pd.DataFrame(cm_a_data)
                st.dataframe(cm_a_df, use_container_width=True, hide_index=True)
            else:
                st.write("N/A")
        
        with col_b:
            st.markdown(f"**Modell B ({model_b_name}):**")
            if comparison.get('b_confusion_matrix'):
                cm_b = comparison['b_confusion_matrix']
                cm_b_col1, cm_b_col2, cm_b_col3, cm_b_col4 = st.columns(4)
                with cm_b_col1:
                    tp_b = cm_b.get('tp', 0)
                    st.metric("TP", tp_b, help="✅ Korrekt als 'Positiv' erkannt")
                with cm_b_col2:
                    tn_b = cm_b.get('tn', 0)
                    st.metric("TN", tn_b, help="✅ Korrekt als 'Negativ' erkannt")
                with cm_b_col3:
                    fp_b = cm_b.get('fp', 0)
                    st.metric("FP", fp_b, help="❌ Fälschlicherweise als 'Positiv' erkannt → Verluste!")
                with cm_b_col4:
                    fn_b = cm_b.get('fn', 0)
                    st.metric("FN", fn_b, help="❌ Fälschlicherweise als 'Negativ' erkannt → Verpasste Chancen!")
                
                # Tabelle
                cm_b_data = {
                    'Tatsächlich': ['Negativ', 'Positiv'],
                    'Vorhergesagt: Negativ': [tn_b, fn_b],
                    'Vorhergesagt: Positiv': [fp_b, tp_b]
                }
                cm_b_df = pd.DataFrame(cm_b_data)
                st.dataframe(cm_b_df, use_container_width=True, hide_index=True)
            else:
                st.write("N/A")
    
    st.divider()
    
    # Train vs. Test Vergleich mit Erklärung
    if comparison.get('a_train_accuracy') or comparison.get('b_train_accuracy'):
        st.subheader("📊 Train vs. Test Vergleich")
        st.markdown("""
        **Was bedeutet Train vs. Test Vergleich?**
        
        Vergleicht die Performance auf Trainings- und Test-Daten:
        - **Train Accuracy:** Performance auf den Daten, mit denen das Modell trainiert wurde
        - **Test Accuracy:** Performance auf neuen, ungesehenen Daten
        - **Degradation:** Unterschied zwischen Train- und Test-Accuracy
        
        **Auswirkung:** 
        - Große Degradation (>10%) = Modell ist möglicherweise overfitted (lernt zu spezifisch)
        - Kleine Degradation (<10%) = Modell generalisiert gut auf neue Daten
        """)
        
        # Modell A
        st.markdown("**Modell A:**")
        col_a1, col_a2, col_a3 = st.columns(3)
        with col_a1:
            a_train_acc = comparison.get('a_train_accuracy')
            if a_train_acc:
                st.metric("Train Accuracy", f"{a_train_acc:.4f}", help="Performance auf Trainingsdaten (0-1, höher = besser)")
            else:
                st.caption("Train Accuracy: N/A")
        with col_a2:
            a_test_acc = comparison.get('a_accuracy')
            if a_test_acc:
                st.metric("Test Accuracy", f"{a_test_acc:.4f}", help="Performance auf Test-Daten (0-1, höher = besser)")
            else:
                st.caption("Test Accuracy: N/A")
        with col_a3:
            a_degradation = comparison.get('a_accuracy_degradation')
            if a_degradation is not None:
                quality = "✅ OK" if a_degradation < 0.1 else "⚠️ Overfitting-Risiko"
                st.metric("Degradation", f"{a_degradation:.4f}", 
                         delta=quality,
                         help=f"Unterschied zwischen Train- und Test-Accuracy. {quality} (niedriger = besser)")
            else:
                st.caption("Degradation: N/A")
        
        st.divider()
        
        # Modell B
        st.markdown("**Modell B:**")
        col_b1, col_b2, col_b3 = st.columns(3)
        with col_b1:
            b_train_acc = comparison.get('b_train_accuracy')
            if b_train_acc:
                st.metric("Train Accuracy", f"{b_train_acc:.4f}", help="Performance auf Trainingsdaten (0-1, höher = besser)")
            else:
                st.caption("Train Accuracy: N/A")
        with col_b2:
            b_test_acc = comparison.get('b_accuracy')
            if b_test_acc:
                st.metric("Test Accuracy", f"{b_test_acc:.4f}", help="Performance auf Test-Daten (0-1, höher = besser)")
            else:
                st.caption("Test Accuracy: N/A")
        with col_b3:
            b_degradation = comparison.get('b_accuracy_degradation')
            if b_degradation is not None:
                quality = "✅ OK" if b_degradation < 0.1 else "⚠️ Overfitting-Risiko"
                st.metric("Degradation", f"{b_degradation:.4f}", 
                         delta=quality,
                         help=f"Unterschied zwischen Train- und Test-Accuracy. {quality} (niedriger = besser)")
            else:
                st.caption("Degradation: N/A")
        
        # Overfitting-Warnungen
        if a_degradation and a_degradation > 0.1:
            st.warning(f"⚠️ **Modell A ist möglicherweise overfitted!** Die Performance auf Test-Daten ist deutlich schlechter als auf Trainingsdaten (Degradation: {a_degradation:.2%})")
        if b_degradation and b_degradation > 0.1:
            st.warning(f"⚠️ **Modell B ist möglicherweise overfitted!** Die Performance auf Test-Daten ist deutlich schlechter als auf Trainingsdaten (Degradation: {b_degradation:.2%})")
    
    st.divider()
    
    # Test-Zeitraum mit Erklärung
    st.subheader("📅 Test-Zeitraum")
    st.markdown("""
    **Was ist der Test-Zeitraum?**
    
    Der Test-Zeitraum definiert, welche Daten zum Testen beider Modelle verwendet wurden.
    Diese Daten wurden **nicht** zum Training verwendet.
    
    **Empfehlung:** Mindestens 1 Tag Test-Daten für realistische Ergebnisse.
    """)
    
    test_start = comparison.get('test_start')
    test_end = comparison.get('test_end')
    test_duration_days = comparison.get('a_test_duration_days') or comparison.get('test_duration_days')
    
    if test_start and test_end:
        try:
            if isinstance(test_start, str):
                start_dt = datetime.fromisoformat(test_start.replace('Z', '+00:00'))
            else:
                start_dt = test_start
            if isinstance(test_end, str):
                end_dt = datetime.fromisoformat(test_end.replace('Z', '+00:00'))
            else:
                end_dt = test_end
            
            start_str = start_dt.strftime("%d.%m.%Y %H:%M")
            end_str = end_dt.strftime("%d.%m.%Y %H:%M")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.write(f"**Start:** {start_str}")
            with col2:
                st.write(f"**Ende:** {end_str}")
            with col3:
                if test_duration_days:
                    st.write(f"**Dauer:** {test_duration_days:.2f} Tage")
                    if test_duration_days < 1:
                        st.warning("⚠️ Test-Zeitraum zu kurz (empfohlen: mindestens 1 Tag)")
                else:
                    days = (end_dt - start_dt).total_seconds() / 86400.0
                    st.write(f"**Dauer:** {days:.2f} Tage")
                    if days < 1:
                        st.warning("⚠️ Test-Zeitraum zu kurz (empfohlen: mindestens 1 Tag)")
        except Exception as e:
            st.write(f"Start: {test_start}")
            st.write(f"Ende: {test_end}")
    else:
        st.write("Test-Zeitraum nicht verfügbar")
    
    st.divider()
    
    # Vollständige Details
    with st.expander("📋 Vollständige Details (JSON)", expanded=False):
        st.json(comparison)
    
    # Zurück-Button
    col1, col2 = st.columns([1, 3])
    with col1:
        if st.button("← Zurück zur Vergleichs-Übersicht", use_container_width=True):
            st.session_state['page'] = 'comparisons'
            st.session_state.pop('comparison_details_id', None)
            st.rerun()

def page_details():
    """Modell-Details"""
    model_id = st.session_state.get('details_model_id')
    if not model_id:
        st.warning("⚠️ Kein Modell ausgewählt")
        return
    
    model = api_get(f"/api/models/{model_id}")
    if not model:
        st.error("❌ Modell nicht gefunden")
        return
    
    st.title(f"📋 Modell-Details: {model.get('name')}")
    
    # Info-Box am Anfang
    st.info("""
    **📖 Anleitung:** 
    Diese Seite zeigt alle Details und Metriken des Modells. Nutze die ℹ️-Icons für Erklärungen zu jedem Wert.
    """)
    
    # Basis-Informationen
    st.subheader("📝 Basis-Informationen")
    info_col1, info_col2, info_col3, info_col4 = st.columns(4)
    with info_col1:
        st.markdown("**Modell-Typ**")
        model_type = model.get('model_type', 'N/A')
        type_emoji = "🌲" if model_type == "random_forest" else "🚀" if model_type == "xgboost" else "🤖"
        st.write(f"{type_emoji} {model_type}")
    with info_col2:
        st.markdown("**Status**")
        status = model.get('status', 'N/A')
        if status == "READY":
            st.success("✅ READY")
        elif status == "TRAINING":
            st.info("🔄 TRAINING")
        else:
            st.error(f"❌ {status}")
    with info_col3:
        st.markdown("**Modell-ID**")
        st.write(f"#{model_id}")
    with info_col4:
        st.markdown("**Erstellt**")
        created = model.get('created_at', '')
        if created:
            try:
                if isinstance(created, str):
                    created_dt = datetime.fromisoformat(created.replace('Z', '+00:00'))
                else:
                    created_dt = created
                st.write(created_dt.strftime("%d.%m.%Y %H:%M"))
            except:
                st.write(str(created)[:19] if len(str(created)) > 19 else str(created))
        else:
            st.write("N/A")
    
    # Beschreibung
    description = model.get('description')
    if description:
        st.markdown("**Beschreibung**")
        st.info(description)
    
    st.divider()
    
    # Standard-Metriken mit Erklärungen
    st.subheader("📊 Standard-Metriken")
    st.markdown("""
    **Was bedeuten diese Metriken?**
    
    - **Accuracy:** Anteil korrekter Vorhersagen (0-1). Beispiel: 0.85 = 85% der Vorhersagen sind richtig.
    - **F1-Score:** Harmonisches Mittel aus Precision und Recall (0-1). Gut für unausgewogene Daten.
    - **Precision:** Von allen "Positiv"-Vorhersagen, wie viele waren wirklich positiv? (0-1, höher = weniger False Positives)
    - **Recall:** Von allen echten Positiven, wie viele hat das Modell gefunden? (0-1, höher = weniger False Negatives)
    """)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        accuracy = model.get('training_accuracy')
        if accuracy:
            st.metric("Accuracy", f"{accuracy:.4f}", help="Anteil korrekter Vorhersagen (0-1, höher = besser)")
        else:
            st.caption("Accuracy: N/A")
    with col2:
        f1 = model.get('training_f1')
        if f1:
            st.metric("F1-Score", f"{f1:.4f}", help="Harmonisches Mittel aus Precision und Recall (0-1, höher = besser)")
        else:
            st.caption("F1-Score: N/A")
    with col3:
        precision = model.get('training_precision')
        if precision:
            st.metric("Precision", f"{precision:.4f}", help="Von allen 'Positiv'-Vorhersagen, wie viele waren wirklich positiv? (0-1, höher = weniger False Positives)")
        else:
            st.caption("Precision: N/A")
    with col4:
        recall = model.get('training_recall')
        if recall:
            st.metric("Recall", f"{recall:.4f}", help="Von allen echten Positiven, wie viele hat das Modell gefunden? (0-1, höher = weniger False Negatives)")
        else:
            st.caption("Recall: N/A")
    
    st.divider()
    
    # Zusätzliche Metriken mit Erklärungen
    st.subheader("📈 Erweiterte Metriken")
    st.markdown("""
    **Was bedeuten diese Metriken?**
    
    - **ROC-AUC:** Area Under ROC Curve (0-1). Misst die Fähigkeit, zwischen Positiv und Negativ zu unterscheiden. >0.7 = gut, >0.9 = sehr gut.
    - **MCC:** Matthews Correlation Coefficient (-1 bis +1). Berücksichtigt alle 4 Confusion-Matrix-Werte. 0 = zufällig, +1 = perfekt, -1 = perfekt falsch.
    - **FPR (False Positive Rate):** Wie viele Negatives wurden fälschlicherweise als Positiv klassifiziert? (0-1, niedriger = besser)
    - **FNR (False Negative Rate):** Wie viele Positives wurden fälschlicherweise als Negativ klassifiziert? (0-1, niedriger = besser)
    """)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        roc_auc = model.get('roc_auc')
        if roc_auc:
            quality = "Sehr gut" if roc_auc > 0.9 else "Gut" if roc_auc > 0.7 else "Mäßig" if roc_auc > 0.5 else "Schlecht"
            st.metric("ROC-AUC", f"{roc_auc:.4f}", help=f"Area Under ROC Curve (0-1). {quality} (>0.7 = gut, >0.9 = sehr gut)")
        else:
            st.caption("ROC-AUC: N/A")
    with col2:
        mcc = model.get('mcc')
        if mcc:
            quality = "Sehr gut" if mcc > 0.5 else "Gut" if mcc > 0.3 else "Mäßig" if mcc > 0 else "Schlecht"
            st.metric("MCC", f"{mcc:.4f}", help=f"Matthews Correlation Coefficient (-1 bis +1). {quality} (0 = zufällig, +1 = perfekt)")
        else:
            st.caption("MCC: N/A")
    with col3:
        fpr = model.get('fpr')
        if fpr is not None:
            quality = "Gut" if fpr < 0.1 else "Mäßig" if fpr < 0.3 else "Schlecht"
            st.metric("False Positive Rate", f"{fpr:.4f}", help=f"Wie viele Negatives wurden fälschlicherweise als Positiv klassifiziert? {quality} (niedriger = besser)")
        else:
            st.caption("FPR: N/A")
    with col4:
        fnr = model.get('fnr')
        if fnr is not None:
            quality = "Gut" if fnr < 0.1 else "Mäßig" if fnr < 0.3 else "Schlecht"
            st.metric("False Negative Rate", f"{fnr:.4f}", help=f"Wie viele Positives wurden fälschlicherweise als Negativ klassifiziert? {quality} (niedriger = besser)")
        else:
            st.caption("FNR: N/A")
    
    st.divider()
    
    # Profit-Simulation mit Erklärung
    simulated_profit = model.get('simulated_profit_pct')
    if simulated_profit is not None:
        st.subheader("💰 Profit-Simulation")
        st.markdown("""
        **Was ist Profit-Simulation?**
        
        Simuliert den Profit, den das Modell erzielt hätte:
        - **True Positive (TP):** +1% Gewinn (korrekt erkannte Pumps)
        - **False Positive (FP):** -0.5% Verlust (fälschlicherweise als Pump erkannt)
        - **True Negative (TN):** 0% (korrekt als "kein Pump" erkannt)
        - **False Negative (FN):** 0% (verpasste Pumps)
        
        **Auswirkung:** Zeigt, wie profitabel das Modell in der Praxis wäre.
        """)
        
        col1, col2 = st.columns(2)
        with col1:
            profit_quality = "Sehr profitabel" if simulated_profit > 5 else "Profitabel" if simulated_profit > 0 else "Verlust"
            st.metric("💰 Simulierter Profit", f"{simulated_profit:.2f}%", help=f"Simulierter Profit basierend auf TP/FP. {profit_quality}")
        with col2:
            st.caption("**Berechnung:** 1% Gewinn pro TP, -0.5% Verlust pro FP")
    
    st.divider()
    
    # Confusion Matrix mit Erklärung
    confusion_matrix = model.get('confusion_matrix')
    if confusion_matrix:
        st.subheader("🔢 Confusion Matrix")
        st.markdown("""
        **Was ist eine Confusion Matrix?**
        
        Zeigt, wie viele Vorhersagen korrekt und falsch waren:
        - **TP (True Positive):** ✅ Korrekt als "Positiv" erkannt (z.B. Pump erkannt, war wirklich Pump)
        - **TN (True Negative):** ✅ Korrekt als "Negativ" erkannt (z.B. kein Pump erkannt, war wirklich kein Pump)
        - **FP (False Positive):** ❌ Fälschlicherweise als "Positiv" erkannt (z.B. Pump erkannt, war aber kein Pump) → Verluste!
        - **FN (False Negative):** ❌ Fälschlicherweise als "Negativ" erkannt (z.B. kein Pump erkannt, war aber Pump) → Verpasste Chancen!
        
        **Auswirkung:** 
        - Viele TP = Modell erkennt Pumps gut
        - Viele FP = Modell ist zu optimistisch (viele Fehlalarme)
        - Viele FN = Modell verpasst viele Pumps
        """)
        
        cm_col1, cm_col2, cm_col3, cm_col4 = st.columns(4)
        with cm_col1:
            tp = confusion_matrix.get('tp', 0)
            st.metric("True Positive (TP)", tp, help="✅ Korrekt als 'Positiv' erkannt (z.B. Pump erkannt, war wirklich Pump)")
        with cm_col2:
            tn = confusion_matrix.get('tn', 0)
            st.metric("True Negative (TN)", tn, help="✅ Korrekt als 'Negativ' erkannt (z.B. kein Pump erkannt, war wirklich kein Pump)")
        with cm_col3:
            fp = confusion_matrix.get('fp', 0)
            st.metric("False Positive (FP)", fp, help="❌ Fälschlicherweise als 'Positiv' erkannt (z.B. Pump erkannt, war aber kein Pump) → Verluste!")
        with cm_col4:
            fn = confusion_matrix.get('fn', 0)
            st.metric("False Negative (FN)", fn, help="❌ Fälschlicherweise als 'Negativ' erkannt (z.B. kein Pump erkannt, war aber Pump) → Verpasste Chancen!")
        
        # Visualisierung als Tabelle
        st.markdown("**Confusion Matrix Tabelle:**")
        cm_data = {
            'Tatsächlich': ['Negativ', 'Positiv'],
            'Vorhergesagt: Negativ': [tn, fn],
            'Vorhergesagt: Positiv': [fp, tp]
        }
        cm_df = pd.DataFrame(cm_data)
        st.dataframe(cm_df, use_container_width=True, hide_index=True)
        
        # Interpretation
        total = tp + tn + fp + fn
        if total > 0:
            tp_rate = (tp / total) * 100
            fp_rate = (fp / total) * 100
            fn_rate = (fn / total) * 100
            st.caption(f"ℹ️ Verteilung: {tp_rate:.1f}% TP, {tn/total*100:.1f}% TN, {fp_rate:.1f}% FP, {fn_rate:.1f}% FN")
    
    st.divider()
    
    # Feature Importance Chart mit Erklärung
    if model.get('feature_importance'):
        st.subheader("🎯 Feature Importance")
        st.markdown("""
        **Was ist Feature Importance?**
        
        Zeigt, welche Features am wichtigsten für die Vorhersage sind:
        - **Höhere Werte** = Feature ist wichtiger für die Vorhersage
        - **Niedrigere Werte** = Feature ist weniger wichtig
        
        **Auswirkung:** 
        - Wichtige Features sollten beibehalten werden
        - Unwichtige Features könnten entfernt werden (Feature Selection)
        - Hilft zu verstehen, worauf das Modell basiert
        """)
        
        fi = model['feature_importance']
        if isinstance(fi, dict):
            df_fi = pd.DataFrame(list(fi.items()), columns=['Feature', 'Importance'])
            df_fi = df_fi.sort_values('Importance', ascending=False)
            
            # Zeige Top 10 Features
            st.markdown("**Top 10 wichtigste Features:**")
            st.dataframe(df_fi.head(10), use_container_width=True, hide_index=True)
            
            # Visualisierung
            fig = px.bar(df_fi.head(15), x='Feature', y='Importance', title="Feature Importance (Top 15)")
            fig.update_xaxes(tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
    
    st.divider()
    
    # Modell-Konfiguration
    st.subheader("⚙️ Modell-Konfiguration")
    
    config_col1, config_col2 = st.columns(2)
    
    with config_col1:
        st.markdown("**Training-Zeitraum:**")
        train_start = model.get('train_start')
        train_end = model.get('train_end')
        if train_start and train_end:
            try:
                if isinstance(train_start, str):
                    start_dt = datetime.fromisoformat(train_start.replace('Z', '+00:00'))
                else:
                    start_dt = train_start
                if isinstance(train_end, str):
                    end_dt = datetime.fromisoformat(train_end.replace('Z', '+00:00'))
                else:
                    end_dt = train_end
                
                start_str = start_dt.strftime("%d.%m.%Y %H:%M")
                end_str = end_dt.strftime("%d.%m.%Y %H:%M")
                duration_days = (end_dt - start_dt).total_seconds() / 86400.0
                st.write(f"**Start:** {start_str}")
                st.write(f"**Ende:** {end_str}")
                st.write(f"**Dauer:** {duration_days:.1f} Tage")
            except Exception as e:
                st.write(f"Start: {train_start}")
                st.write(f"Ende: {train_end}")
        else:
            st.write("Zeitraum nicht verfügbar")
        
        st.markdown("**Features:**")
        features_list = model.get('features', [])
        if features_list:
            st.write(f"{len(features_list)} Features ausgewählt")
            with st.expander("Alle Features anzeigen"):
                for feat in features_list:
                    st.write(f"- {feat}")
        else:
            st.write("Keine Features verfügbar")
        
        st.markdown("**Phasen:**")
        phases_list = model.get('phases')
        if phases_list:
            st.write(f"{len(phases_list)} Phase(n) ausgewählt")
            with st.expander("Phasen anzeigen"):
                for phase_id in phases_list:
                    st.write(f"- Phase {phase_id}")
        else:
            st.write("Alle Phasen verwendet")
    
    with config_col2:
        st.markdown("**Ziel-Variable:**")
        target_var = model.get('target_variable', 'N/A')
        target_operator = model.get('target_operator')
        target_value = model.get('target_value')
        
        # Zeitbasierte Vorhersage?
        future_minutes = model.get('future_minutes')
        price_change = model.get('price_change_percent')
        direction = model.get('target_direction')
        
        if future_minutes and price_change:
            st.write(f"**Typ:** ⏰ Zeitbasierte Vorhersage")
            st.write(f"**Variable:** {target_var}")
            st.write(f"**Zeitraum:** {future_minutes} Minuten")
            st.write(f"**Min. Änderung:** {price_change}%")
            direction_text = "📈 Steigt" if direction == "up" else "📉 Fällt" if direction == "down" else "N/A"
            st.write(f"**Richtung:** {direction_text}")
            st.write(f"**Ziel:** {target_var} {direction_text.lower()} in {future_minutes} Minuten um mindestens {price_change}%")
        else:
            st.write(f"**Variable:** {target_var}")
            if target_operator and target_value is not None:
                st.write(f"**Bedingung:** {target_var} {target_operator} {target_value}")
            else:
                st.write("**Bedingung:** Nicht konfiguriert")
        
        st.markdown("**Parameter:**")
        params = model.get('params', {})
        if isinstance(params, str):
            import json
            try:
                params = json.loads(params)
            except:
                params = {}
        
        if params:
            # Zeige wichtige Parameter
            if params.get('use_engineered_features'):
                st.write("🔧 Feature-Engineering: ✅ Aktiviert")
                windows = params.get('feature_engineering_windows', [])
                if windows:
                    st.write(f"   Fenster: {windows}")
            
            if params.get('_time_based', {}).get('enabled'):
                time_based_params = params.get('_time_based', {})
                tb_future_minutes = time_based_params.get('future_minutes') or future_minutes
                tb_min_percent = time_based_params.get('min_percent_change') or price_change
                tb_direction = time_based_params.get('direction') or direction
                if tb_future_minutes and tb_min_percent:
                    direction_text = "steigt" if tb_direction == "up" else "fällt" if tb_direction == "down" else ""
                    st.write(f"⏰ Zeitbasierte Vorhersage: ✅ Aktiviert ({tb_future_minutes}min, {tb_min_percent}% {direction_text})")
                else:
                    st.write("⏰ Zeitbasierte Vorhersage: ✅ Aktiviert")
            
            if params.get('use_smote') is False:
                st.write("⚖️ SMOTE: ❌ Deaktiviert")
            else:
                st.write("⚖️ SMOTE: ✅ Aktiviert (oder automatisch)")
            
            if params.get('use_timeseries_split') is False:
                st.write("🔀 TimeSeriesSplit: ❌ Deaktiviert")
            else:
                st.write("🔀 TimeSeriesSplit: ✅ Aktiviert (oder automatisch)")
            
            cv_splits = params.get('cv_splits')
            if cv_splits:
                st.write(f"🔀 Cross-Validation: {cv_splits} Splits")
            
            # Hyperparameter
            n_estimators = params.get('n_estimators')
            max_depth = params.get('max_depth')
            if n_estimators or max_depth:
                st.write("⚙️ Hyperparameter:")
                if n_estimators:
                    st.write(f"   n_estimators: {n_estimators}")
                if max_depth:
                    st.write(f"   max_depth: {max_depth}")
                learning_rate = params.get('learning_rate')
                if learning_rate:
                    st.write(f"   learning_rate: {learning_rate}")
        else:
            st.write("Keine Parameter verfügbar")
    
    st.divider()
    
    # CV-Scores
    cv_scores = model.get('cv_scores')
    if cv_scores:
        st.subheader("🔀 Cross-Validation Ergebnisse")
        st.markdown("""
        **Was sind CV-Scores?**
        
        Cross-Validation teilt die Daten in mehrere Teile auf und testet das Modell auf jedem Teil.
        Dies gibt eine realistischere Einschätzung der Modell-Performance.
        
        **Auswirkung:** 
        - Zeigt, wie konsistent das Modell über verschiedene Daten-Teile ist
        - Niedrige Standardabweichung = Modell ist stabil
        - Hohe Standardabweichung = Modell ist instabil
        """)
        
        if isinstance(cv_scores, dict):
            cv_col1, cv_col2, cv_col3 = st.columns(3)
            with cv_col1:
                mean_score = cv_scores.get('mean_score')
                if mean_score is not None:
                    st.metric("Durchschnittlicher Score", f"{mean_score:.4f}", help="Durchschnittliche Performance über alle CV-Splits")
            with cv_col2:
                std_score = cv_scores.get('std_score')
                if std_score is not None:
                    st.metric("Standardabweichung", f"{std_score:.4f}", help="Wie stark variiert die Performance? (niedriger = stabiler)")
            with cv_col3:
                cv_overfitting = model.get('cv_overfitting_gap')
                if cv_overfitting is not None:
                    quality = "OK" if cv_overfitting < 0.1 else "⚠️ Overfitting-Risiko"
                    st.metric("Overfitting-Gap", f"{cv_overfitting:.4f}", help=f"Unterschied zwischen Train- und CV-Score. {quality} (niedriger = besser)")
            
            # Einzelne Scores
            individual_scores = cv_scores.get('scores', [])
            if individual_scores:
                st.markdown("**Einzelne CV-Scores:**")
                st.write(f"Scores: {[f'{s:.4f}' for s in individual_scores]}")
    
    st.divider()
    
    # Vollständige Details
    with st.expander("📋 Vollständige Details (JSON)", expanded=False):
        st.json(model)
    
    # Zurück-Button (auch in Sidebar verfügbar)
    col1, col2 = st.columns([1, 3])
    with col1:
        if st.button("← Zurück zur Übersicht", use_container_width=True):
            st.session_state['page'] = 'overview'
            st.session_state.pop('details_model_id', None)
            st.rerun()

# ============================================================
# Tab-Funktionen
# ============================================================

def tab_dashboard():
    """Dashboard Tab"""
    st.title("📊 Dashboard")
    
    # Health Status
    health = api_get("/api/health")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if health:
            status = "🟢 Healthy" if health.get("status") == "healthy" else "🔴 Degraded"
            st.metric("Status", status)
        else:
            st.metric("Status", "❌ Nicht erreichbar")
    
    with col2:
        if health:
            st.metric("Jobs verarbeitet", health.get("total_jobs_processed", 0))
        else:
            st.metric("Jobs verarbeitet", "-")
    
    with col3:
        if health:
            db_status = "✅ Verbunden" if health.get("db_connected") else "❌ Getrennt"
            st.metric("Datenbank", db_status)
        else:
            st.metric("Datenbank", "-")
    
    with col4:
        if health:
            uptime = health.get("uptime_seconds", 0)
            hours = uptime // 3600
            minutes = (uptime % 3600) // 60
            st.metric("Uptime", f"{int(hours)}h {int(minutes)}m")
        else:
            st.metric("Uptime", "-")
    
    # Modelle-Übersicht
    st.subheader("📋 Modelle-Übersicht")
    models = api_get("/api/models")
    if models:
        st.info(f"📊 {len(models)} Modell(e) gefunden")
    else:
        st.info("ℹ️ Keine Modelle gefunden")
    
    # Service-Management
    st.subheader("🔧 Service-Management")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🔄 Konfiguration neu laden", type="primary"):
            with st.spinner("Konfiguration wird neu geladen..."):
                success, message = reload_config()
                if success:
                    st.success(message)
                    time.sleep(2)
                else:
                    st.error(message)
                st.rerun()
    
    with col2:
        if st.button("🔄 Seite aktualisieren"):
            st.rerun()
    
    # Auto-Refresh - ohne time.sleep() um UI nicht zu blockieren
    auto_refresh_enabled = st.checkbox("🔄 Auto-Refresh (5s)", key="auto_refresh_dashboard")
    if auto_refresh_enabled:
        # Verwende st.empty() und st.rerun() ohne time.sleep() - Streamlit wird automatisch neu rendern
        placeholder = st.empty()
        placeholder.info("⏳ Auto-Refresh aktiv...")
        st.rerun()
    
def tab_configuration():
    """Konfiguration Tab"""
    st.title("⚙️ Konfiguration")
    
    try:
        config = load_config()
    except Exception as e:
        st.error(f"⚠️ Fehler beim Laden der Config: {e}")
        config = get_default_config()
    
    using_env_vars = bool(os.getenv('DB_DSN'))
    
    if using_env_vars:
        st.info("🌐 **Coolify-Modus erkannt:** Environment Variables haben Priorität, aber du kannst die Konfiguration trotzdem hier speichern.")
    else:
        st.info("💡 Änderungen werden in der Konfigurationsdatei gespeichert. Nutze den 'Konfiguration neu laden' Button, um Änderungen ohne Neustart zu übernehmen.")
    
    with st.form("config_form"):
        st.subheader("🗄️ Datenbank Einstellungen")
        config["DB_DSN"] = st.text_input("DB DSN", value=config.get("DB_DSN", ""), help="PostgreSQL Connection String")
        if config["DB_DSN"]:
            db_valid, db_error = validate_url(config["DB_DSN"], allow_empty=False)
            if not db_valid:
                st.error(f"❌ {db_error}")
        
        st.subheader("🔌 Port Einstellungen")
        config["API_PORT"] = st.number_input("API Port", min_value=1, max_value=65535, value=config.get("API_PORT", 8000))
        config["STREAMLIT_PORT"] = st.number_input("Streamlit Port", min_value=1, max_value=65535, value=config.get("STREAMLIT_PORT", 8501))
        
        st.subheader("📁 Pfad Einstellungen")
        config["MODEL_STORAGE_PATH"] = st.text_input("Model Storage Path", value=config.get("MODEL_STORAGE_PATH", "/app/models"))
        config["API_BASE_URL"] = st.text_input("API Base URL", value=config.get("API_BASE_URL", "http://localhost:8000"), help="Innerhalb des Containers: localhost:8000, von außen: localhost:8012")
        if config["API_BASE_URL"]:
            api_valid, api_error = validate_url(config["API_BASE_URL"], allow_empty=False)
            if not api_valid:
                st.error(f"❌ {api_error}")
        
        st.subheader("⚙️ Job Queue Einstellungen")
        config["JOB_POLL_INTERVAL"] = st.number_input("Job Poll Interval (Sekunden)", min_value=1, max_value=300, value=config.get("JOB_POLL_INTERVAL", 5))
        config["MAX_CONCURRENT_JOBS"] = st.number_input("Max Concurrent Jobs", min_value=1, max_value=10, value=config.get("MAX_CONCURRENT_JOBS", 2))
        
        st.subheader("📝 Logging Einstellungen")
        config["LOG_LEVEL"] = st.selectbox("Log Level", ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], index=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"].index(config.get("LOG_LEVEL", "INFO")))
        config["LOG_FORMAT"] = st.selectbox("Log Format", ["text", "json"], index=["text", "json"].index(config.get("LOG_FORMAT", "text")))
        config["LOG_JSON_INDENT"] = st.number_input("Log JSON Indent", min_value=0, max_value=4, value=config.get("LOG_JSON_INDENT", 0))
        
        col1, col2 = st.columns(2)
        with col1:
            save_button = st.form_submit_button("💾 Konfiguration speichern", type="primary")
        with col2:
            reset_button = st.form_submit_button("🔄 Auf Standard zurücksetzen")
        
        if save_button:
            errors = []
            
            db_valid, db_error = validate_url(config["DB_DSN"], allow_empty=False)
            if not db_valid:
                errors.append(f"DB DSN: {db_error}")
            
            api_valid, api_error = validate_url(config["API_BASE_URL"], allow_empty=False)
            if not api_valid:
                errors.append(f"API Base URL: {api_error}")
            
            if errors:
                st.error("❌ **Validierungsfehler:**")
                for error in errors:
                    st.error(f"  - {error}")
            else:
                try:
                    result = save_config(config)
                    if result:
                        st.session_state.config_saved = True
                        st.success("✅ Konfiguration gespeichert!")
                        if using_env_vars:
                            st.info("💡 **Tipp:** Nutze den 'Konfiguration neu laden' Button unten, um die Änderungen ohne Neustart zu übernehmen.")
                        else:
                            st.info("💡 **Tipp:** Nutze den 'Konfiguration neu laden' Button unten, um die Änderungen ohne Neustart zu übernehmen.")
                        st.session_state.config_just_saved = True
                except Exception as e:
                    st.error(f"❌ **Fehler beim Speichern:** {e}")
        
        if reset_button:
            try:
                default_config = get_default_config()
                if save_config(default_config):
                    st.session_state.config_saved = True
                    st.success("✅ Konfiguration auf Standard zurückgesetzt!")
                    st.warning("⚠️ Bitte Service neu starten oder 'Konfiguration neu laden' Button unten verwenden!")
                    st.session_state.config_just_saved = True
            except Exception as e:
                st.error(f"❌ **Fehler beim Zurücksetzen:** {e}")
    
    # Reload-Button
    st.divider()
    st.subheader("🔄 Konfiguration neu laden")
    st.caption("Lädt die gespeicherte Konfiguration im Service neu (ohne Neustart)")
    if st.button("🔄 Konfiguration neu laden", type="primary", key="reload_config_button"):
        with st.spinner("Konfiguration wird neu geladen..."):
            success, message = reload_config()
            if success:
                st.success(f"✅ {message}")
                st.info("💡 Die neue Konfiguration ist jetzt aktiv! Kein Neustart nötig.")
            else:
                st.error(f"❌ {message}")
                st.info("💡 Falls der Reload fehlschlägt, starte den Service manuell neu.")
    
    # Neustart-Button
    if st.session_state.get("config_saved", False):
        st.divider()
        st.subheader("🔄 Service-Neustart")
        col1, col2 = st.columns([2, 1])
        with col1:
            st.info("💡 Die Konfiguration wurde gespeichert. Starte den Service neu, damit die neuen Werte geladen werden.")
        with col2:
            if st.button("🔄 Service neu starten", type="primary", use_container_width=True):
                with st.spinner("Service wird neu gestartet..."):
                    success, message = restart_service()
                    if success:
                        st.success(message)
                        st.info("⏳ Bitte warte 5-10 Sekunden, bis der Service vollständig neu gestartet ist.")
                        st.session_state.config_saved = False
                        time.sleep(3)
                        st.rerun()
                    else:
                        st.error(message)
                        st.info("💡 Du kannst den Service auch manuell neu starten: `docker compose restart ml-training`")
    
    # Auto-Reload nach Speichern
    if st.session_state.get("config_just_saved", False):
        st.session_state.config_just_saved = False
        time.sleep(0.5)
        st.rerun()
    
    # Aktuelle Konfiguration anzeigen
    st.subheader("📄 Aktuelle Konfiguration")
    st.json(config)

def tab_logs():
    """Logs Tab"""
    st.title("📋 Service Logs")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        lines = st.number_input("Anzahl Zeilen", min_value=10, max_value=1000, value=100, step=10, key="logs_lines_input")
    
    with col2:
        refresh_logs = st.button("🔄 Logs aktualisieren", key="refresh_logs_button")
        if refresh_logs:
            st.rerun()
    
    logs = get_service_logs(lines=lines)
    
    st.text_area(
        "Service Logs (neueste oben)",
        logs,
        height=600,
        key="logs_display",
        help="Die neuesten Logs stehen oben, die ältesten unten."
    )
    
    if not logs or logs.strip() == "":
        st.warning("⚠️ Keine Logs verfügbar. Prüfe ob der Service läuft.")
    
    auto_refresh = st.checkbox("🔄 Auto-Refresh Logs (10s)", key="auto_refresh_logs")
    if auto_refresh:
        # Verwende st.empty() und st.rerun() ohne time.sleep() - Streamlit wird automatisch neu rendern
        placeholder = st.empty()
        placeholder.info("⏳ Auto-Refresh aktiv...")
        st.rerun()

def tab_metrics():
    """Metriken Tab"""
    st.title("📈 Metriken")
    
    if st.button("🔄 Metriken aktualisieren"):
        st.rerun()
    
    # Prometheus Metrics
    try:
        response = httpx.get(f"{API_BASE_URL}/api/metrics", timeout=5)
        if response.status_code == 200:
            metrics = response.text
            st.subheader("📄 Prometheus Metriken (Raw)")
            st.code(metrics, language="text")
            
            # Parse Metriken
            metrics_dict = {}
            for line in metrics.split('\n'):
                if line and not line.startswith('#'):
                    parts = line.split()
                    if len(parts) >= 2:
                        metric_name = parts[0]
                        try:
                            metric_value = float(parts[1]) if '.' in parts[1] else int(parts[1])
                            metrics_dict[metric_name] = metric_value
                        except:
                            metrics_dict[metric_name] = parts[1]
            
            st.subheader("📊 Metriken als strukturierte Daten")
            st.json(metrics_dict)
        else:
            st.error(f"❌ Fehler beim Abrufen der Metriken: HTTP {response.status_code}")
    except Exception as e:
        st.error(f"❌ Fehler beim Abrufen der Metriken: {str(e)}")
    
    auto_refresh_metrics = st.checkbox("🔄 Auto-Refresh Metriken (5s)", key="auto_refresh_metrics")
    if auto_refresh_metrics:
        # Verwende st.empty() und st.rerun() ohne time.sleep() - Streamlit wird automatisch neu rendern
        placeholder = st.empty()
        placeholder.info("⏳ Auto-Refresh aktiv...")
        st.rerun()

def tab_info():
    """Info Tab - Vollständige Projekt-Informationen"""
    st.title("ℹ️ Projekt-Informationen")
    
    # Projekt-Übersicht
    st.header("📋 Was macht dieses Projekt?")
    st.markdown("""
    **ML Training Service** ist ein Machine-Learning-Service für Kryptowährungs-Datenanalyse.
    
    Das System:
    - ✅ Trainiert ML-Modelle (Random Forest, XGBoost) für Pump-Detection
    - ✅ Unterstützt zeitbasierte Vorhersagen ("Steigt in 10 Min um 30%")
    - ✅ Feature-Engineering für bessere Performance
    - ✅ Verwaltet Trainings-Jobs in einer asynchronen Queue
    - ✅ Testet Modelle auf neuen Daten
    - ✅ Vergleicht Modelle miteinander
    - ✅ Speichert Modelle persistent (Datenbank + .pkl Dateien)
    - ✅ Bietet eine Web-UI für Monitoring und Konfiguration
    - ✅ Exportiert Prometheus-Metriken für Monitoring
    """)
    
    st.divider()
    
    # Datenfluss
    st.header("🔄 Datenfluss")
    st.code("""
    PostgreSQL Datenbank
            ├─ coin_metrics (OHLCV + Metriken)
            ├─ exchange_rates (SOL-Preis, Marktstimmung)
            └─ ref_coin_phases (Phasen-Konfiguration)
            ↓
    FastAPI Service (app/main.py)
            ├─ API-Endpunkte (/api/models/create, /api/models/test, etc.)
            ├─ Job-Queue (ml_jobs Tabelle)
            └─ Asynchroner Worker (verarbeitet Jobs)
            ↓
    Training Engine (app/training/engine.py)
            ├─ Lädt Daten aus coin_metrics
            ├─ Feature-Engineering (optional)
            ├─ Marktstimmung-Enrichment (optional)
            ├─ Label-Erstellung (zeitbasiert oder klassisch)
            ├─ Modell-Training (Random Forest / XGBoost)
            └─ Speichert Modell (.pkl + Metadaten in DB)
            ↓
    ml_models Tabelle
            ├─ Modell-Metadaten
            ├─ Performance-Metriken
            ├─ Feature Importance
            └─ Verweis auf .pkl Datei
    """, language="text")
    
    st.divider()
    
    # Was macht das System genau?
    st.header("🔍 Was macht das System genau?")
    
    st.subheader("1️⃣ Modell-Training")
    st.markdown("""
    **Prozess:**
    1. **Job-Erstellung:** Benutzer erstellt Training-Job über Web-UI oder API
    2. **Job-Queue:** Job wird in `ml_jobs` Tabelle mit Status `PENDING` gespeichert
    3. **Worker-Verarbeitung:** Asynchroner Worker findet Job und startet Training
    4. **Daten-Laden:** System lädt Daten aus `coin_metrics` für den gewählten Zeitraum
    5. **Feature-Engineering:** Optional werden ~40 zusätzliche Features erstellt
    6. **Marktstimmung:** Optional wird SOL-Preis-Kontext aus `exchange_rates` hinzugefügt
    7. **Label-Erstellung:** Labels werden erstellt (zeitbasiert oder klassisch)
    8. **Training:** Modell wird trainiert (Random Forest oder XGBoost)
    9. **Evaluation:** Modell wird auf Test-Set evaluiert
    10. **Speicherung:** Modell wird als .pkl Datei gespeichert + Metadaten in DB
    """)
    
    st.subheader("2️⃣ Modell-Testing")
    st.markdown("""
    **Prozess:**
    1. **Test-Job erstellen:** Benutzer wählt Modell und Test-Zeitraum
    2. **Daten-Laden:** System lädt Test-Daten aus `coin_metrics`
    3. **Vorhersagen:** Modell macht Vorhersagen auf Test-Daten
    4. **Evaluation:** Metriken werden berechnet (Accuracy, Precision, Recall, F1, etc.)
    5. **Speicherung:** Test-Ergebnisse werden in `ml_test_results` gespeichert
    """)
    
    st.subheader("3️⃣ Modell-Vergleich")
    st.markdown("""
    **Prozess:**
    1. **Vergleichs-Job erstellen:** Benutzer wählt 2 Modelle und Test-Zeitraum
    2. **Parallele Tests:** Beide Modelle werden auf denselben Daten getestet
    3. **Metriken-Vergleich:** Alle Metriken werden verglichen
    4. **Speicherung:** Vergleichs-Ergebnisse werden in `ml_comparisons` gespeichert
    """)
    
    st.divider()
    
    # Welche Informationen werden verwendet?
    st.header("📤 Welche Informationen werden verwendet?")
    
    st.subheader("1️⃣ Basis-Daten aus coin_metrics")
    st.markdown("""
    **OHLCV-Daten:**
    - `price_open`, `price_high`, `price_low`, `price_close` - Preis-Daten
    - `volume_sol`, `buy_volume_sol`, `sell_volume_sol` - Volumen-Daten
    - `market_cap_close` - Market Cap
    
    **Rug-Detection-Metriken:**
    - `dev_sold_amount` - ⚠️ **KRITISCH:** Wie viel SOL hat der Dev verkauft?
    - `buy_pressure_ratio` - Verhältnis Buy- zu Sell-Volumen
    - `unique_signer_ratio` - Anteil einzigartiger Trader
    - `whale_buy_volume_sol`, `whale_sell_volume_sol` - Whale-Aktivität
    - `num_whale_buys`, `num_whale_sells` - Anzahl Whale-Transaktionen
    - `net_volume_sol` - Netto-Volumen (Buy - Sell)
    - `volatility_pct` - Volatilität in Prozent
    - `avg_trade_size_sol` - Durchschnittliche Trade-Größe
    
    **Zeitstempel & Phasen:**
    - `timestamp` - Zeitstempel der Metrik
    - `phase_id_at_time` - Phase des Coins zu diesem Zeitpunkt
    """)
    
    st.subheader("2️⃣ Marktstimmung aus exchange_rates")
    st.markdown("""
    **Markt-Kontext (optional, wenn `use_market_context=True`):**
    - `sol_price_usd` - SOL-Preis in USD (der "Wasserstand")
    - `usd_to_eur_rate` - Währungsumrechnung
    
    **Berechnete Features:**
    - `sol_price_change_pct` - SOL-Preis-Änderung in Prozent
    - `sol_price_ma_5` - 5-Perioden Moving Average des SOL-Preises
    - `sol_price_volatility` - Volatilität des SOL-Preises
    
    **Zweck:** Unterscheidung zwischen echten Token-Pumps und allgemeinen Marktbewegungen
    """)
    
    st.subheader("3️⃣ Feature-Engineering (optional)")
    st.markdown("""
    **Wenn `use_engineered_features=True`:**
    
    **Momentum-Features:**
    - Price-Momentum (5, 10, 15 Perioden)
    - Volume-Momentum
    - Rate of Change (ROC)
    
    **Volumen-Patterns:**
    - Volume-MA-Ratio
    - Buy/Sell-Volumen-Ratio
    - Net-Volumen-Trend
    
    **Whale-Aktivität:**
    - Whale-Buy-Rate
    - Whale-Sell-Rate
    - Whale-Aktivitäts-Trend
    
    **Dev-Tracking:**
    - Dev-Sold-Amount-Trend
    - Dev-Sold-Rate
    
    **Volatilität:**
    - Rolling-Volatilität
    - Price-Range-Ratio
    
    **Insgesamt:** ~40 zusätzliche Features werden erstellt
    """)
    
    st.divider()
    
    # Datenbankschema
    st.header("🗄️ Datenbankschema")
    
    st.subheader("Haupttabelle: `ml_models`")
    st.markdown("""
    Speichert alle trainierten Modelle mit Metadaten und Performance-Metriken.
    
    **Wichtige Felder:**
    - `id` - Eindeutige Modell-ID
    - `name` - Modell-Name (eindeutig)
    - `model_type` - "random_forest" oder "xgboost"
    - `status` - "TRAINING", "READY", "FAILED"
    - `target_variable` - Ziel-Variable (z.B. "price_close")
    - `future_minutes` - Bei zeitbasierter Vorhersage: Minuten in die Zukunft
    - `price_change_percent` - Bei zeitbasierter Vorhersage: Mindest-Prozent-Änderung
    - `target_direction` - Bei zeitbasierter Vorhersage: "up" oder "down"
    - `features` - JSONB Array mit Feature-Namen
    - `phases` - JSONB Array mit Coin-Phasen (z.B. [1, 2, 3])
    - `params` - JSONB Object mit Hyperparametern
    - `feature_importance` - JSONB Object mit Feature-Importance-Werten
    - `training_accuracy`, `training_f1`, `training_precision`, `training_recall` - Basis-Metriken
    - `cv_scores` - JSONB Object mit Cross-Validation-Ergebnissen
    - `roc_auc`, `mcc`, `fpr`, `fnr` - Erweiterte Metriken
    - `confusion_matrix` - JSONB Object mit TP, TN, FP, FN
    - `simulated_profit_pct` - Simulierter Profit in Prozent
    - `rug_detection_metrics` - JSONB Object mit Rug-Detection-Metriken
    - `market_context_enabled` - Boolean: Wurde Marktstimmung verwendet?
    - `model_file_path` - Pfad zur .pkl Datei
    """)
    
    st.subheader("Tabelle: `ml_test_results`")
    st.markdown("""
    Speichert Test-Ergebnisse für Modelle auf neuen Daten.
    
    **Wichtige Felder:**
    - `id` - Eindeutige Test-ID
    - `model_id` - Verweis auf ml_models
    - `test_start`, `test_end` - Test-Zeitraum
    - `test_accuracy`, `test_f1`, `test_precision`, `test_recall` - Test-Metriken
    - `confusion_matrix` - JSONB Object mit TP, TN, FP, FN
    - `rug_detection_metrics` - JSONB Object mit Rug-Detection-Metriken
    """)
    
    st.subheader("Tabelle: `ml_comparisons`")
    st.markdown("""
    Speichert Vergleichs-Ergebnisse zwischen 2 Modellen.
    
    **Wichtige Felder:**
    - `id` - Eindeutige Vergleichs-ID
    - `model_a_id`, `model_b_id` - Verweise auf ml_models
    - `test_start`, `test_end` - Test-Zeitraum
    - `winner_model_id` - ID des besseren Modells
    - `comparison_metrics` - JSONB Object mit Vergleichs-Metriken
    """)
    
    st.subheader("Tabelle: `ml_jobs`")
    st.markdown("""
    Verwaltet alle Jobs (Training, Testing, Comparison) in einer Queue.
    
    **Wichtige Felder:**
    - `id` - Eindeutige Job-ID
    - `job_type` - "TRAIN", "TEST", "COMPARE"
    - `status` - "PENDING", "RUNNING", "COMPLETED", "FAILED", "CANCELLED"
    - `progress` - Fortschritt in Prozent (0.0 - 1.0)
    - `progress_msg` - Fortschritts-Nachricht
    - `train_model_type` - Modell-Typ (bei TRAIN-Jobs)
    - `train_features` - JSONB Array mit Features
    - `train_params` - JSONB Object mit Parametern
    - `started_at`, `completed_at` - Zeitstempel
    """)
    
    st.subheader("Referenztabelle: `ref_coin_phases`")
    st.markdown("""
    Definiert Coin-Phasen für Phasen-basiertes Training.
    
    **Phasen:**
    - **Baby Zone** (ID: 1): 0-10 Min, Intervall: 5s
    - **Survival Zone** (ID: 2): 10-60 Min, Intervall: 30s
    - **Mature Zone** (ID: 3): 1-24 Std, Intervall: 60s
    - **Finished** (ID: 99): Ab 24 Std
    - **Graduated** (ID: 100): Graduierte Tokens
    """)
    
    st.divider()
    
    # API Endpoints
    st.header("🔌 API Endpoints")
    
    st.subheader("Modelle")
    st.markdown("""
    | Endpoint | Methode | Beschreibung |
    |----------|---------|-------------|
    | `/api/models/create` | POST | Erstellt einen TRAIN-Job (asynchron) |
    | `/api/models` | GET | Listet alle Modelle (optional: `?status=READY`) |
    | `/api/models/{model_id}` | GET | Gibt Details eines Modells zurück |
    | `/api/models/{model_id}` | PATCH | Aktualisiert Modell (z.B. Beschreibung) |
    | `/api/models/{model_id}` | DELETE | Löscht Modell (soft delete) |
    """)
    
    st.subheader("Testing")
    st.markdown("""
    | Endpoint | Methode | Beschreibung |
    |----------|---------|-------------|
    | `/api/models/test` | POST | Erstellt einen TEST-Job (asynchron) |
    | `/api/test-results` | GET | Listet alle Test-Ergebnisse |
    | `/api/test-results/{test_id}` | GET | Gibt Details eines Tests zurück |
    """)
    
    st.subheader("Vergleich")
    st.markdown("""
    | Endpoint | Methode | Beschreibung |
    |----------|---------|-------------|
    | `/api/models/compare` | POST | Erstellt einen COMPARE-Job (asynchron) |
    | `/api/comparisons` | GET | Listet alle Vergleiche |
    | `/api/comparisons/{comparison_id}` | GET | Gibt Details eines Vergleichs zurück |
    """)
    
    st.subheader("Jobs")
    st.markdown("""
    | Endpoint | Methode | Beschreibung |
    |----------|---------|-------------|
    | `/api/queue` | GET | Listet alle Jobs (optional: `?status=RUNNING`) |
    | `/api/queue/{job_id}` | GET | Gibt Details eines Jobs zurück |
    """)
    
    st.subheader("System")
    st.markdown("""
    | Endpoint | Methode | Beschreibung |
    |----------|---------|-------------|
    | `/api/health` | GET | Health Check (Status, DB-Verbindung, Uptime) |
    | `/api/metrics` | GET | Prometheus-Metriken (Text-Format) |
    | `/api/data-availability` | GET | Verfügbare Daten-Zeiträume |
    | `/api/reload-config` | POST | Lädt Konfiguration neu (ohne Neustart) |
    """)
    
    st.divider()
    
    # Technische Details
    st.header("🔧 Technische Details")
    
    st.subheader("Services")
    st.markdown("""
    - **FastAPI Service** (`app/main.py`): API-Endpunkte, Job-Queue, Health-Checks, Prometheus-Metriken
    - **Streamlit UI** (`app/streamlit_app.py`): Web-Interface für Monitoring und Konfiguration
    - **Job Manager** (`app/queue/job_manager.py`): Asynchroner Worker für Job-Verarbeitung
    - **Training Engine** (`app/training/engine.py`): Modell-Training-Logik
    - **Feature Engineering** (`app/training/feature_engineering.py`): Feature-Erstellung
    - **Model Loader** (`app/training/model_loader.py`): Modell-Laden und Testing
    """)
    
    st.subheader("Ports")
    st.markdown("""
    **Externe Ports (Docker Host):**
    - **API**: Port `8012` (FastAPI)
    - **Web UI**: Port `8502` (Streamlit)
    
    **Interne Ports (Docker Container):**
    - **API**: Port `8000` (FastAPI)
    - **Web UI**: Port `8501` (Streamlit)
    """)
    
    st.subheader("Job-System")
    st.markdown("""
    **Asynchrones Job-System:**
    - Jobs werden in `ml_jobs` Tabelle gespeichert
    - Worker prüft alle 5 Sekunden auf neue `PENDING` Jobs
    - Max. 2 Jobs parallel (konfigurierbar über `MAX_CONCURRENT_JOBS`)
    - Training läuft in separatem Thread (blockiert nicht Event Loop)
    - Progress wird kontinuierlich aktualisiert
    """)
    
    st.subheader("Modell-Typen")
    st.markdown("""
    **Random Forest:**
    - Ensemble-Methode mit mehreren Decision Trees
    - Robust gegen Overfitting
    - Gute Performance auf imbalanced Data
    
    **XGBoost:**
    - Gradient Boosting Framework
    - Sehr gute Performance
    - Unterstützt Feature Importance
    """)
    
    st.divider()
    
    # Zeitbasierte Vorhersagen
    st.header("⏰ Zeitbasierte Vorhersagen")
    
    st.markdown("""
    **Konzept:**
    Statt zu fragen "Ist price_close > 50000?", fragt das System:
    "Steigt price_close in 10 Minuten um mindestens 30%?"
    
    **Parameter:**
    - `future_minutes`: Anzahl Minuten in die Zukunft (z.B. 10)
    - `min_percent_change`: Mindest-Prozent-Änderung (z.B. 30.0 für 30%)
    - `direction`: "up" (steigt) oder "down" (fällt)
    
    **Label-Erstellung:**
    1. Für jeden Zeitpunkt wird der aktuelle Wert (`price_close`) genommen
    2. Der Wert nach `future_minutes` Minuten wird berechnet
    3. Prozent-Änderung wird berechnet: `((future - current) / current) * 100`
    4. Label = 1 wenn Änderung >= `min_percent_change` (bei "up") oder <= -`min_percent_change` (bei "down")
    5. Label = 0 sonst
    
    **Vorteile:**
    - Realistischere Vorhersagen (zeitbasiert)
    - Besser für Trading-Strategien geeignet
    - Berücksichtigt Zeit-Komponente
    """)
    
    st.divider()
    
    # Feature-Engineering
    st.header("🎯 Feature-Engineering")
    
    st.markdown("""
    **Aktivierung:** `use_engineered_features=True`
    
    **Erstellte Features:**
    - **Momentum:** Price-Momentum, Volume-Momentum, ROC
    - **Volumen-Patterns:** Volume-MA-Ratio, Buy/Sell-Ratio, Net-Volumen-Trend
    - **Whale-Aktivität:** Whale-Buy-Rate, Whale-Sell-Rate, Whale-Aktivitäts-Trend
    - **Dev-Tracking:** Dev-Sold-Amount-Trend, Dev-Sold-Rate
    - **Volatilität:** Rolling-Volatilität, Price-Range-Ratio
    
    **Fenstergrößen:** Konfigurierbar über `feature_engineering_windows` (z.B. [5, 10, 15])
    
    **Insgesamt:** ~40 zusätzliche Features werden erstellt
    """)
    
    st.divider()
    
    # Marktstimmung
    st.header("📈 Marktstimmung (Market Context)")
    
    st.markdown("""
    **Aktivierung:** `use_market_context=True`
    
    **Zweck:** Unterscheidung zwischen echten Token-Pumps und allgemeinen Marktbewegungen
    
    **Datenquelle:** `exchange_rates` Tabelle (SOL-Preis in USD)
    
    **Erstellte Features:**
    - `sol_price_change_pct` - SOL-Preis-Änderung in Prozent
    - `sol_price_ma_5` - 5-Perioden Moving Average des SOL-Preises
    - `sol_price_volatility` - Volatilität des SOL-Preises
    
    **Beispiele:**
    - **"Token steigt, während SOL stabil ist"** → Bullish (Echter Pump) ✅
    - **"Token steigt, weil SOL um 5% steigt"** → Neutral (Marktbewegung) ⚠️
    - **"Token ist stabil, während SOL crasht"** → Stärke (Relative Strength) 💪
    """)
    
    st.divider()
    
    # Rug-Detection-Metriken
    st.header("🚨 Rug-Detection-Metriken")
    
    st.markdown("""
    **Spezielle Metriken für Rug-Pull-Erkennung:**
    
    - **Dev-Sold Detection Rate:** Wie oft wurde `dev_sold_amount > 0` korrekt erkannt?
    - **Wash-Trading Detection Rate:** Wie oft wurde Wash-Trading erkannt?
    - **Weighted Cost:** Kosten-Funktion (False Negatives 10x schwerer als False Positives)
    - **Precision at K:** Precision bei Top-K Vorhersagen
    
    **Kritische Features:**
    - `dev_sold_amount` - ⚠️ **KRITISCH:** Wie viel SOL hat der Dev verkauft?
    - `buy_pressure_ratio` - Verhältnis Buy- zu Sell-Volumen
    - `unique_signer_ratio` - Anteil einzigartiger Trader
    """)
    
    st.divider()
    
    # Dokumentation
    st.header("📚 Dokumentation")
    
    st.markdown("""
    **Wichtige Dokumentationen:**
    - **[README.md](../README.md)** - Projekt-Übersicht
    - **[KOMPLETTE_KI_MODELL_ANLEITUNG.md](../docs/KOMPLETTE_KI_MODELL_ANLEITUNG.md)** - Vollständige Anleitung zur Modell-Erstellung
    - **[IMPLEMENTIERUNGS_ANLEITUNG_METRIKEN.md](../docs/IMPLEMENTIERUNGS_ANLEITUNG_METRIKEN.md)** - Metriken-Integration
    - **[LABEL_VALIDIERUNGSBERICHT.md](../docs/LABEL_VALIDIERUNGSBERICHT.md)** - Label-Validierung
    - **[PRODUCTION_READINESS_BERICHT.md](../docs/PRODUCTION_READINESS_BERICHT.md)** - Production-Readiness
    - **[XGBOOST_MODEL_ERSTELLUNG.md](../docs/XGBOOST_MODEL_ERSTELLUNG.md)** - XGBoost-Optimierung
    - **[COOLIFY_DEPLOYMENT_CHECKLIST.md](../docs/COOLIFY_DEPLOYMENT_CHECKLIST.md)** - Coolify Deployment
    - **[complete_schema.sql](../sql/complete_schema.sql)** - Vollständiges Datenbank-Schema
    """)
    
    st.divider()
    
    # Label-Erstellung - DETAILLIERT
    st.header("🏷️ Label-Erstellung - Schritt für Schritt")
    
    st.subheader("1️⃣ Klassische Labels (Schwellwert-basiert)")
    st.markdown("""
    **Wann verwendet:** Wenn `use_time_based_prediction = False`
    
    **Prozess:**
    1. Für jede Zeile wird geprüft: `target_variable operator target_value`
    2. Beispiel: `price_close > 50000`
    3. Label = 1 wenn Bedingung erfüllt, 0 wenn nicht erfüllt
    
    **Operatoren:**
    - `>` - Größer als
    - `<` - Kleiner als
    - `>=` - Größer oder gleich
    - `<=` - Kleiner oder gleich
    - `=` - Gleich
    
    **Beispiel:**
    ```
    Zeile 1: price_close = 45000 → 45000 > 50000? → Nein → Label = 0
    Zeile 2: price_close = 55000 → 55000 > 50000? → Ja → Label = 1
    Zeile 3: price_close = 60000 → 60000 > 50000? → Ja → Label = 1
    ```
    """)
    
    st.subheader("2️⃣ Zeitbasierte Labels (Zukunfts-Vorhersage)")
    st.markdown("""
    **Wann verwendet:** Wenn `use_time_based_prediction = True`
    
    **Konzept:** Statt zu fragen "Ist price_close > 50000?", fragt das System:
    "Steigt price_close in 10 Minuten um mindestens 30%?"
    
    **Schritt-für-Schritt Prozess:**
    
    **Schritt 1: Aktueller Wert**
    ```python
    current_value = data.loc[idx, "price_close"]  # z.B. 100.0
    ```
    
    **Schritt 2: Zukünftiger Wert berechnen**
    - System berechnet, wie viele Zeilen in die Zukunft geschaut werden muss
    - Beispiel: Phase 1 hat `interval_seconds=5` → 10 Minuten = 120 Zeilen
    - Zukünftiger Wert: `future_value = data.loc[idx + 120, "price_close"]`  # z.B. 130.0
    
    **Schritt 3: Prozent-Änderung berechnen**
    ```python
    percent_change = ((future_value - current_value) / current_value) * 100
    # Beispiel: ((130.0 - 100.0) / 100.0) * 100 = 30.0%
    ```
    
    **Schritt 4: Label erstellen**
    ```python
    if direction == "up":
        label = 1 if percent_change >= min_percent_change else 0
        # Beispiel: 30.0% >= 30.0%? → Ja → Label = 1
    else:  # "down"
        label = 1 if percent_change <= -min_percent_change else 0
        # Beispiel: -30.0% <= -30.0%? → Ja → Label = 1
    ```
    
    **Wichtig:**
    - ⚠️ **Data Leakage Prevention:** `target_variable` wird aus Features entfernt!
    - ⚠️ **NaN-Handling:** Zeilen ohne Zukunftswerte werden entfernt (am Ende des Datensatzes)
    - ⚠️ **Division durch Null:** Wird verhindert durch `valid_mask`
    - ⚠️ **Phase-Intervalle:** System verwendet exakte Intervalle pro Phase (genauer als Durchschnitt)
    """)
    
    st.subheader("3️⃣ Label-Validierung")
    st.markdown("""
    **Automatische Prüfungen:**
    - ✅ Mindestens 1 positives Label (sonst: Fehler)
    - ✅ Mindestens 1 negatives Label (sonst: Fehler)
    - ⚠️ Warnung wenn Labels sehr unausgewogen (< 5% oder > 95% positiv)
    
    **Empfehlung:** Labels sollten zwischen 20% und 80% positiv sein für beste Performance
    """)
    
    st.divider()
    
    # Was beim KI-Modell-Erstellen beachten?
    st.header("⚠️ Was beim KI-Modell-Erstellen beachten?")
    
    st.subheader("1️⃣ Datenqualität")
    st.markdown("""
    **Wichtig:**
    - ✅ **Zeitraum wählen:** Mindestens 4-6 Stunden Daten für aussagekräftige Modelle
    - ✅ **Phasen berücksichtigen:** Verschiedene Phasen haben unterschiedliche Verhaltensmuster
    - ✅ **Datenverfügbarkeit prüfen:** Prüfe `/api/data-availability` vor dem Training
    - ⚠️ **Max. 500.000 Zeilen:** System begrenzt automatisch (RAM-Management)
    
    **Empfehlung:**
    - Training: Letzte 4-6 Stunden der verfügbaren Daten
    - Test: Letzte 10-30 Minuten (separat vom Training)
    """)
    
    st.subheader("2️⃣ Feature-Auswahl")
    st.markdown("""
    **Kritische Features (empfohlen):**
    - ✅ `dev_sold_amount` - **KRITISCH** für Rug-Detection
    - ✅ `buy_pressure_ratio` - Buy/Sell-Verhältnis
    - ✅ `unique_signer_ratio` - Anteil einzigartiger Trader
    - ✅ `price_close`, `volume_sol` - Basis-Daten
    
    **Optional (Feature-Engineering):**
    - ✅ `use_engineered_features=True` - Erstellt ~40 zusätzliche Features
    - ✅ `feature_engineering_windows=[5, 10, 15]` - Fenstergrößen für Features
    
    **Feature-Ausschluss:**
    - ⚠️ `exclude_features` - Liste von Features die ausgeschlossen werden sollen
    - Beispiel: `exclude_features=["dev_sold_amount"]` - Wenn Dev-Tracking nicht gewünscht
    """)
    
    st.subheader("3️⃣ Zeitbasierte Vorhersage")
    st.markdown("""
    **Parameter:**
    - `future_minutes`: 5-30 Minuten empfohlen (zu kurz = zu viele False Positives, zu lang = zu wenige Positives)
    - `min_percent_change`: 5-30% empfohlen (zu niedrig = zu viele False Positives, zu hoch = zu wenige Positives)
    - `direction`: "up" (steigt) oder "down" (fällt)
    
    **Empfehlung:**
    - **Konservativ:** 10 Minuten, 5% Änderung
    - **Mittel:** 10 Minuten, 10% Änderung
    - **Aggressiv:** 5 Minuten, 20% Änderung
    
    **Wichtig:**
    - ⚠️ Bei zeitbasierter Vorhersage wird `target_variable` automatisch aus Features entfernt (Data Leakage Prevention)
    - ⚠️ Am Ende des Datensatzes werden Zeilen ohne Zukunftswerte entfernt
    """)
    
    st.subheader("4️⃣ Hyperparameter")
    st.markdown("""
    **Random Forest:**
    - `n_estimators`: 100-300 (mehr = besser, aber langsamer)
    - `max_depth`: 10-20 (zu tief = Overfitting)
    - `min_samples_split`: 5-10 (zu niedrig = Overfitting)
    
    **XGBoost:**
    - `n_estimators`: 200-500 (mehr = besser, aber langsamer)
    - `max_depth`: 6-10 (zu tief = Overfitting)
    - `learning_rate`: 0.05-0.1 (niedriger = besser, aber langsamer)
    
    **Empfehlung:** Verwende Standard-Parameter zuerst, dann optimiere basierend auf Ergebnissen
    """)
    
    st.subheader("5️⃣ Imbalanced Data Handling")
    st.markdown("""
    **Problem:** Wenn Labels sehr unausgewogen sind (z.B. 95% negativ, 5% positiv)
    
    **Lösung:** `use_smote=True`
    - SMOTE (Synthetic Minority Oversampling Technique)
    - Erstellt synthetische positive Beispiele
    - Verbessert Performance bei unausgewogenen Daten
    
    **Wann verwenden:** Wenn Labels < 20% oder > 80% positiv sind
    """)
    
    st.subheader("6️⃣ Cross-Validation")
    st.markdown("""
    **Aktivierung:** `use_timeseries_split=True` (Standard: aktiviert)
    
    **Was passiert:**
    - TimeSeriesSplit respektiert zeitliche Reihenfolge (verhindert Data Leakage)
    - Anzahl Splits: `cv_splits=5` (Standard)
    - Berechnet Train- und Test-Accuracy für jeden Split
    - Overfitting-Gap wird berechnet (Differenz zwischen Train- und Test-Accuracy)
    
    **Wichtig:**
    - ⚠️ TimeSeriesSplit ist wichtig für Zeitreihen-Daten!
    - ⚠️ Normale K-Fold würde Data Leakage verursachen (Zukunft in Training)
    """)
    
    st.subheader("7️⃣ Marktstimmung")
    st.markdown("""
    **Aktivierung:** `use_market_context=True`
    
    **Vorteile:**
    - Unterscheidet echte Token-Pumps von Marktbewegungen
    - Erstellt Features: `sol_price_change_pct`, `sol_price_ma_5`, `sol_price_volatility`
    
    **Wann verwenden:** Immer empfohlen (verbessert Modell-Performance)
    
    **Voraussetzung:** `exchange_rates` Tabelle muss Daten enthalten
    """)
    
    st.divider()
    
    # Kompletter Workflow - Schritt für Schritt
    st.header("🔄 Kompletter Workflow - Schritt für Schritt")
    
    st.subheader("Phase 1: Job-Erstellung")
    st.markdown("""
    **1.1 Web-UI oder API Request**
    - Benutzer erstellt Training-Job über Web-UI oder API
    - Parameter werden eingegeben (Name, Modell-Typ, Features, Zeitraum, etc.)
    
    **1.2 Validierung**
    - Request wird validiert (Pydantic Schema)
    - Zeitbasierte Parameter werden in `train_params._time_based` gespeichert
    - Feature-Engineering Parameter werden in `train_params` gespeichert
    
    **1.3 Job in Datenbank**
    - Job wird in `ml_jobs` Tabelle erstellt mit `status='PENDING'`
    - Modell-Name wird in `progress_msg` temporär gespeichert
    - Response mit `job_id` wird zurückgegeben
    """)
    
    st.subheader("Phase 2: Worker-Verarbeitung")
    st.markdown("""
    **2.1 Job gefunden**
    - Worker prüft alle 5 Sekunden auf `PENDING` Jobs
    - Wenn Job gefunden UND weniger als 2 Jobs aktiv:
      - Status wird auf `RUNNING` gesetzt
      - Job wird asynchron verarbeitet
    
    **2.2 Parameter extrahieren**
    - Modell-Name aus `progress_msg`
    - Features, Phasen, Zeitraum aus Job-Daten
    - Hyperparameter aus `train_params`
    - Zeitbasierte Parameter aus `train_params._time_based`
    """)
    
    st.subheader("Phase 3: Daten-Laden")
    st.markdown("""
    **3.1 SQL Query**
    ```sql
    SELECT timestamp, price_open, price_high, price_low, price_close,
           volume_sol, buy_volume_sol, sell_volume_sol, dev_sold_amount,
           buy_pressure_ratio, unique_signer_ratio, ...
    FROM coin_metrics
    WHERE timestamp >= $1 AND timestamp <= $2
      AND phase_id_at_time = ANY($3)  -- Falls Phasen gefiltert
    ORDER BY timestamp
    LIMIT 500000
    ```
    
    **3.2 Daten-Verarbeitung**
    - Daten werden nach `timestamp` sortiert (wichtig für zeitbasierte Labels!)
    - `timestamp` wird als Index gesetzt
    - Duplikate werden entfernt
    - Decimal-Typen werden zu float konvertiert (PostgreSQL → Pandas)
    - Max. 500.000 Zeilen (RAM-Management)
    
    **3.3 Marktstimmung (optional)**
    - Wenn `use_market_context=True`:
      - SOL-Preis wird aus `exchange_rates` geladen
      - Features werden erstellt: `sol_price_change_pct`, `sol_price_ma_5`, `sol_price_volatility`
    """)
    
    st.subheader("Phase 4: Feature-Vorbereitung")
    st.markdown("""
    **4.1 Data Leakage Prevention**
    ```python
    # Bei zeitbasierter Vorhersage:
    features_for_loading = ["price_open", "volume_sol", "price_close"]  # Enthält target_var
    features_for_training = ["price_open", "volume_sol"]  # target_var ENTFERNT!
    ```
    
    **4.2 Feature-Engineering (optional)**
    - Wenn `use_engineered_features=True`:
      - ~40 zusätzliche Features werden erstellt
      - Momentum, Volumen-Patterns, Whale-Aktivität, etc.
      - Features werden zu `features_for_training` hinzugefügt
    """)
    
    st.subheader("Phase 5: Label-Erstellung")
    st.markdown("""
    **5.1 Zeitbasierte Labels (wenn aktiviert)**
    ```python
    # Für jede Zeile:
    current_value = data.loc[idx, "price_close"]  # z.B. 100.0
    future_value = data.loc[idx + rows_to_shift, "price_close"]  # z.B. 130.0
    percent_change = ((130.0 - 100.0) / 100.0) * 100  # = 30.0%
    
    if direction == "up":
        label = 1 if percent_change >= min_percent_change else 0
    ```
    
    **5.2 Klassische Labels (wenn zeitbasierte Vorhersage deaktiviert)**
    ```python
    # Für jede Zeile:
    value = data.loc[idx, "price_close"]  # z.B. 55000
    label = 1 if value > target_value else 0  # z.B. 55000 > 50000? → Ja → 1
    ```
    
    **5.3 Label-Validierung**
    - Prüft ob mindestens 1 positives und 1 negatives Label vorhanden
    - Warnung wenn Labels sehr unausgewogen (< 5% oder > 95% positiv)
    """)
    
    st.subheader("Phase 6: Training")
    st.markdown("""
    **6.1 Train/Test Split**
    - Standard: 80% Training, 20% Test
    - TimeSeriesSplit respektiert zeitliche Reihenfolge
    
    **6.2 SMOTE (optional)**
    - Wenn `use_smote=True` und Labels unausgewogen:
      - Synthetische positive Beispiele werden erstellt
      - Verbessert Performance bei unausgewogenen Daten
    
    **6.3 Modell-Training**
    - Random Forest oder XGBoost wird trainiert
    - Training läuft in separatem Thread (blockiert nicht Event Loop)
    - Progress wird kontinuierlich aktualisiert (20% → 60% → 80% → 100%)
    
    **6.4 Cross-Validation (optional)**
    - Wenn `use_timeseries_split=True`:
      - TimeSeriesSplit mit 5 Splits
      - Train- und Test-Accuracy für jeden Split
      - Overfitting-Gap wird berechnet
    """)
    
    st.subheader("Phase 7: Evaluation")
    st.markdown("""
    **7.1 Basis-Metriken**
    - Accuracy, Precision, Recall, F1-Score
    - Confusion Matrix (TP, TN, FP, FN)
    
    **7.2 Erweiterte Metriken**
    - ROC-AUC (wenn Modell Wahrscheinlichkeiten unterstützt)
    - MCC (Matthews Correlation Coefficient)
    - FPR (False Positive Rate)
    - FNR (False Negative Rate)
    
    **7.3 Rug-Detection-Metriken**
    - Dev-Sold Detection Rate
    - Wash-Trading Detection Rate
    - Weighted Cost (FN 10x schwerer als FP)
    - Precision at K
    
    **7.4 Feature Importance**
    - Wichtigste Features werden berechnet
    - Gespeichert als JSONB in `ml_models.feature_importance`
    """)
    
    st.subheader("Phase 8: Speicherung")
    st.markdown("""
    **8.1 Modell-Datei**
    - Modell wird als .pkl Datei gespeichert
    - Pfad: `/app/models/{model_id}_{model_name}.pkl`
    
    **8.2 Datenbank-Metadaten**
    - Alle Metriken werden in `ml_models` Tabelle gespeichert
    - Status wird auf `READY` gesetzt
    - Job-Status wird auf `COMPLETED` gesetzt
    """)
    
    st.divider()
    
    # Best Practices
    st.header("💡 Best Practices")
    
    st.subheader("1️⃣ Modell-Erstellung")
    st.markdown("""
    - ✅ **Zeitraum:** Mindestens 4-6 Stunden Trainingsdaten
    - ✅ **Features:** Immer `dev_sold_amount` inkludieren (kritisch für Rug-Detection)
    - ✅ **Marktstimmung:** Immer aktivieren (`use_market_context=True`)
    - ✅ **Feature-Engineering:** Aktivieren für bessere Performance
    - ✅ **Cross-Validation:** Immer aktivieren (verhindert Overfitting)
    - ✅ **SMOTE:** Aktivieren wenn Labels unausgewogen (< 20% oder > 80% positiv)
    """)
    
    st.subheader("2️⃣ Zeitbasierte Vorhersage")
    st.markdown("""
    - ✅ **future_minutes:** 5-30 Minuten (10 Minuten empfohlen)
    - ✅ **min_percent_change:** 5-30% (10% empfohlen)
    - ✅ **direction:** "up" für Pump-Detection, "down" für Crash-Detection
    - ⚠️ **Zu niedrige Werte:** Viele False Positives
    - ⚠️ **Zu hohe Werte:** Zu wenige positive Labels
    """)
    
    st.subheader("3️⃣ Hyperparameter-Tuning")
    st.markdown("""
    - ✅ **Starte mit Standard-Parametern:** System hat gute Defaults
    - ✅ **Iteriere basierend auf Ergebnissen:** Teste verschiedene Kombinationen
    - ✅ **Vergleiche Modelle:** Verwende Vergleichs-Funktion um beste Parameter zu finden
    - ⚠️ **Overfitting vermeiden:** Zu hohe `max_depth` oder zu niedrige `min_samples_split`
    """)
    
    st.subheader("4️⃣ Testing")
    st.markdown("""
    - ✅ **Separater Zeitraum:** Teste auf Daten die NICHT im Training waren
    - ✅ **Realistische Zeiträume:** 10-30 Minuten Test-Daten
    - ✅ **Vergleiche Modelle:** Teste mehrere Modelle auf denselben Daten
    - ⚠️ **Nicht auf Trainingsdaten testen:** Würde unrealistische Metriken zeigen
    """)
    
    st.divider()
    
    # Häufige Fehler und Lösungen
    st.header("🐛 Häufige Fehler und Lösungen")
    
    st.subheader("1️⃣ 'Keine positiven Labels gefunden'")
    st.markdown("""
    **Problem:** Alle Labels sind 0 (Bedingung wird nie erfüllt)
    
    **Ursachen:**
    - `min_percent_change` zu hoch (z.B. 50% bei zeitbasierter Vorhersage)
    - `target_value` zu hoch (z.B. `price_close > 1000000`)
    - Zeitraum zu kurz (zu wenige Daten)
    
    **Lösung:**
    - Reduziere `min_percent_change` oder `target_value`
    - Wähle längeren Zeitraum
    - Prüfe Datenverfügbarkeit vor Training
    """)
    
    st.subheader("2️⃣ 'Keine negativen Labels gefunden'")
    st.markdown("""
    **Problem:** Alle Labels sind 1 (Bedingung wird immer erfüllt)
    
    **Ursachen:**
    - `min_percent_change` zu niedrig (z.B. 0.1%)
    - `target_value` zu niedrig (z.B. `price_close > 0`)
    
    **Lösung:**
    - Erhöhe `min_percent_change` oder `target_value`
    - Prüfe Label-Verteilung vor Training
    """)
    
    st.subheader("3️⃣ 'Nicht genug Daten'")
    st.markdown("""
    **Problem:** Zu wenige Zeilen für Training
    
    **Ursachen:**
    - Zeitraum zu kurz
    - Phasen-Filter zu restriktiv
    - Datenbank enthält keine Daten für Zeitraum
    
    **Lösung:**
    - Wähle längeren Zeitraum
    - Entferne Phasen-Filter oder wähle mehr Phasen
    - Prüfe `/api/data-availability` für verfügbare Daten
    """)
    
    st.subheader("4️⃣ 'Modell-Performance schlecht'")
    st.markdown("""
    **Problem:** Accuracy < 60% oder viele False Positives
    
    **Ursachen:**
    - Zu wenige Features
    - Feature-Engineering nicht aktiviert
    - Labels zu unausgewogen
    - Hyperparameter nicht optimal
    
    **Lösung:**
    - Aktiviere Feature-Engineering
    - Aktiviere Marktstimmung
    - Aktiviere SMOTE wenn Labels unausgewogen
    - Teste verschiedene Hyperparameter
    - Vergleiche mehrere Modelle
    """)
    
    st.subheader("5️⃣ 'Training dauert zu lange'")
    st.markdown("""
    **Problem:** Training dauert > 30 Minuten
    
    **Ursachen:**
    - Zu viele Daten (> 500.000 Zeilen)
    - Zu viele Features (> 50)
    - `n_estimators` zu hoch
    
    **Lösung:**
    - Reduziere Zeitraum (weniger Daten)
    - Reduziere Feature-Anzahl
    - Reduziere `n_estimators`
    - System begrenzt automatisch auf 500.000 Zeilen
    """)
    
    st.divider()
    
    # Performance-Tipps
    st.header("⚡ Performance-Tipps")
    
    st.markdown("""
    **1. Datenmenge:**
    - ✅ 4-6 Stunden Trainingsdaten sind ausreichend
    - ⚠️ Mehr Daten = längeres Training, aber nicht unbedingt bessere Performance
    
    **2. Feature-Anzahl:**
    - ✅ 10-20 Basis-Features + Feature-Engineering (~40 Features) = optimal
    - ⚠️ Zu viele Features (> 100) = Overfitting-Risiko
    
    **3. Hyperparameter:**
    - ✅ Random Forest: `n_estimators=200`, `max_depth=10`
    - ✅ XGBoost: `n_estimators=300`, `max_depth=8`, `learning_rate=0.1`
    
    **4. Parallelisierung:**
    - ✅ System verarbeitet max. 2 Jobs parallel
    - ✅ Training läuft in separatem Thread (blockiert nicht Event Loop)
    
    **5. Caching:**
    - ✅ Modelle werden persistent gespeichert (DB + .pkl)
    - ✅ Kein Re-Training nötig für Tests
    """)
    
    st.divider()
    
    # Zusammenfassung
    st.header("📊 Zusammenfassung")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Modell-Typen", "2", "Random Forest, XGBoost")
    
    with col2:
        st.metric("Vorhersage-Typen", "2", "Klassisch, Zeitbasiert")
    
    with col3:
        st.metric("Job-Typen", "3", "TRAIN, TEST, COMPARE")
    
    st.info("""
    **Wichtig:** 
    - Alle Jobs werden asynchron verarbeitet (nicht sofort)
    - Modell-Training kann mehrere Minuten dauern
    - Progress wird kontinuierlich aktualisiert
    - Modelle werden persistent gespeichert (DB + .pkl Dateien)
    - Prometheus-Metriken werden für Monitoring exportiert
    - Labels werden automatisch validiert (keine Data Leakage)
    - Feature-Engineering erstellt ~40 zusätzliche Features
    - Marktstimmung unterscheidet echte Pumps von Marktbewegungen
    """)

# ============================================================
# Main App
# ============================================================

def main():
    """Hauptfunktion mit Tab-basiertem Layout"""
    st.title("🤖 ML Training Service - Control Panel")
    
    # Tabs Navigation
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10, tab11, tab12 = st.tabs([
        "📊 Dashboard",
        "⚙️ Konfiguration",
        "📋 Logs",
        "📈 Metriken",
        "ℹ️ Info",
        "🏠 Modelle",
        "➕ Training",
        "🧪 Testen",
        "📋 Test-Ergebnisse",
        "⚔️ Vergleichen",
        "⚖️ Vergleichs-Übersicht",
        "📊 Jobs"
    ])
    
    with tab1:
        tab_dashboard()
    
    with tab2:
        tab_configuration()
    
    with tab3:
        tab_logs()
    
    with tab4:
        tab_metrics()
    
    with tab5:
        tab_info()
    
    with tab6:
        page_overview()
    
    with tab7:
        page_train()
    
    with tab8:
        page_test()
    
    with tab9:
        page_test_results()
    
    with tab10:
        page_compare()
    
    with tab11:
        page_comparisons()
    
    with tab12:
        page_jobs()
    
    # Details-Seiten werden weiterhin über Session State gehandhabt
    if st.session_state.get('page') == 'details':
        page_details()
    elif st.session_state.get('page') == 'comparison_details':
        page_comparison_details()
    elif st.session_state.get('page') == 'test_details':
        page_test_details()

if __name__ == "__main__":
    main()

