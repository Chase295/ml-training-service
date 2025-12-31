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
# Konfigurationssystem (wie in vorherigen Services)
# ============================================================

def load_config():
    """Lädt Konfiguration aus YAML-Datei oder .env"""
    config = {}
    
    # SCHRITT 1: Lade aus Environment Variables als Basis
    env_vars = {
        'DB_DSN': os.getenv('DB_DSN'),
        'API_PORT': os.getenv('API_PORT', '8000'),
        'STREAMLIT_PORT': os.getenv('STREAMLIT_PORT', '8501'),
        'MODEL_STORAGE_PATH': os.getenv('MODEL_STORAGE_PATH', '/app/models'),
        'API_BASE_URL': os.getenv('API_BASE_URL', 'http://localhost:8000'),
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
    
    # SCHRITT 2: Lade aus .env Datei
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
    
    # SCHRITT 3: Lade aus YAML-Datei (höchste Priorität)
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
    
    # Speichere YAML
    try:
        with open(CONFIG_FILE, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
    except (OSError, PermissionError) as e:
        raise OSError(f"Config-Datei kann nicht geschrieben werden: {e}")
    
    # Speichere .env
    env_content = f"""# ============================================================================
# ML TRAINING SERVICE - Umgebungsvariablen
# ============================================================================
# Diese Datei wird automatisch von der Streamlit UI verwaltet.
# Änderungen können über den "Konfiguration neu laden" Button übernommen werden.
# ============================================================================

# Datenbank
DB_DSN={config.get('DB_DSN', 'postgresql://user:pass@localhost:5432/crypto')}

# Ports
API_PORT={config.get('API_PORT', 8000)}
STREAMLIT_PORT={config.get('STREAMLIT_PORT', 8501)}

# Modelle
MODEL_STORAGE_PATH={config.get('MODEL_STORAGE_PATH', '/app/models')}

# API Base URL
API_BASE_URL={config.get('API_BASE_URL', 'http://localhost:8000')}

# Job Queue
JOB_POLL_INTERVAL={config.get('JOB_POLL_INTERVAL', 5)}
MAX_CONCURRENT_JOBS={config.get('MAX_CONCURRENT_JOBS', 2)}

# Logging
LOG_LEVEL={config.get('LOG_LEVEL', 'INFO')}
LOG_FORMAT={config.get('LOG_FORMAT', 'text')}
LOG_JSON_INDENT={config.get('LOG_JSON_INDENT', 0)}
"""
    try:
        os.makedirs(os.path.dirname(ENV_FILE), exist_ok=True)
        with open(ENV_FILE, 'w') as f:
            f.write(env_content)
    except (OSError, PermissionError):
        pass  # Optional
    
    return True

def get_default_config():
    """Gibt Standard-Konfiguration zurück"""
    return {
        "DB_DSN": "postgresql://user:pass@localhost:5432/crypto",
        "API_PORT": 8000,
        "STREAMLIT_PORT": 8501,
        "MODEL_STORAGE_PATH": "/app/models",
        "API_BASE_URL": "http://localhost:8000",
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
    """Startet Service neu"""
    if COOLIFY_MODE:
        success, message = reload_config()
        if success:
            return True, f"✅ {message} (ohne Neustart - funktioniert in Coolify!)"
        else:
            return False, f"⚠️ Coolify-Modus: {message}. Bitte Service im Coolify-Dashboard neu starten."
    
    try:
        import docker
        client = docker.from_env()
        container_names = [SERVICE_NAME, "ml-training-service", "ml-training"]
        container = None
        for name in container_names:
            try:
                container = client.containers.get(name)
                break
            except docker.errors.NotFound:
                continue
        
        if not container:
            return False, "Container nicht gefunden"
        
        container.stop(timeout=10)
        container.start()
        return True, "Service erfolgreich neu gestartet!"
    except ImportError:
        try:
            import subprocess
            compose_file = "/app/../docker-compose.yml"
            if not os.path.exists(compose_file):
                compose_file = "/app/docker-compose.yml"
            
            if os.path.exists(compose_file):
                work_dir = os.path.dirname(compose_file)
                result = subprocess.run(
                    ["docker", "compose", "restart", "ml-training"],
                    cwd=work_dir,
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                if result.returncode == 0:
                    return True, "Service neu gestartet (via docker compose)"
                else:
                    return False, f"Docker Compose Fehler: {result.stderr}"
            else:
                return False, "docker-compose.yml nicht gefunden"
        except Exception as e:
            return False, f"Fehler: {str(e)}"
    except Exception as e:
        return False, f"Fehler: {str(e)}"

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
# API Helper Functions
# ============================================================

def api_get(endpoint: str) -> Any:
    """GET Request zur API"""
    try:
        response = httpx.get(f"{API_BASE_URL}{endpoint}", timeout=30.0)
        response.raise_for_status()
        return response.json()
    except httpx.HTTPError as e:
        return [] if 'comparisons' in endpoint or 'models' in endpoint else {}

def api_post(endpoint: str, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """POST Request zur API"""
    try:
        response = httpx.post(f"{API_BASE_URL}{endpoint}", json=data, timeout=30.0)
        response.raise_for_status()
        return response.json()
    except httpx.HTTPError as e:
        st.error(f"❌ API-Fehler: {e}")
        return None

def api_delete(endpoint: str) -> bool:
    """DELETE Request zur API"""
    try:
        response = httpx.delete(f"{API_BASE_URL}{endpoint}", timeout=30.0)
        response.raise_for_status()
        return True
    except httpx.HTTPError:
        return False

def api_patch(endpoint: str, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """PATCH Request zur API"""
    try:
        response = httpx.patch(f"{API_BASE_URL}{endpoint}", json=data, timeout=30.0)
        response.raise_for_status()
        return response.json()
    except httpx.HTTPError as e:
        st.error(f"❌ API-Fehler: {e}")
        return None

# Verfügbare Features
AVAILABLE_FEATURES = [
    "price_open", "price_high", "price_low", "price_close",
    "volume_sol",
    "market_cap_close"
]

AVAILABLE_TARGETS = [
    "market_cap_close", "price_close", "volume_sol"
]

def load_phases() -> List[Dict[str, Any]]:
    """Lade Phasen aus der API"""
    phases_data = api_get("/api/phases")
    if isinstance(phases_data, list):
        return phases_data
    return []

# ============================================================
# Tab-Funktionen (werden später implementiert)
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
        # Hier würde die bestehende Modelle-Ansicht eingefügt werden
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
                    st.rerun()
                else:
                    st.error(message)
    
    with col2:
        if st.button("🔄 Seite aktualisieren"):
            st.rerun()
    
    # Auto-Refresh
    if st.checkbox("🔄 Auto-Refresh (5s)"):
        time.sleep(5)
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
        config["API_BASE_URL"] = st.text_input("API Base URL", value=config.get("API_BASE_URL", "http://localhost:8000"))
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
                        time.sleep(0.5)
                        st.rerun()
                except (OSError, PermissionError) as e:
                    st.error(f"❌ **Fehler beim Speichern:** {e}")
        
        if reset_button:
            default_config = get_default_config()
            if save_config(default_config):
                st.session_state.config_saved = True
                st.success("✅ Konfiguration auf Standard zurückgesetzt!")
                st.warning("⚠️ Bitte Service neu starten oder 'Konfiguration neu laden' Button unten verwenden!")
                time.sleep(0.5)
                st.rerun()
    
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
        time.sleep(10)
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
    
    if st.checkbox("🔄 Auto-Refresh Metriken (5s)"):
        time.sleep(5)
        st.rerun()

def tab_info():
    """Info Tab"""
    st.title("ℹ️ Projekt-Informationen")
    
    st.header("📋 Was macht dieses Projekt?")
    st.markdown("""
    **ML Training Service** ist ein Machine-Learning-Service für Coin-Bot Training.
    
    Das System:
    - ✅ Trainiert ML-Modelle (Random Forest, XGBoost)
    - ✅ Verwaltet Trainings-Jobs in einer Queue
    - ✅ Speichert Modelle persistent
    - ✅ Bietet eine Web-UI für Monitoring und Konfiguration
    """)
    
    st.header("🔧 Technische Details")
    st.markdown("""
    **Services:**
    - **FastAPI Service** (`app/main.py`): API-Endpunkte, Job-Queue, Health-Checks
    - **Streamlit UI** (`app/streamlit_app.py`): Web-Interface für Monitoring und Konfiguration
    
    **Ports:**
    - **API**: Port `8000` (FastAPI)
    - **Web UI**: Port `8501` (Streamlit)
    """)
    
    st.header("📚 Dokumentation")
    st.markdown("""
    - **[README.md](../README.md)** - Projekt-Übersicht
    - **[DEPLOYMENT.md](../docs/DEPLOYMENT.md)** - Deployment-Anleitung
    - **[COOLIFY_DEPLOYMENT.md](../docs/COOLIFY_DEPLOYMENT.md)** - Coolify Deployment
    """)

# ============================================================
# Main App
# ============================================================

def main():
    """Hauptfunktion"""
    st.title("🤖 ML Training Service - Control Panel")
    
    # Tabs Navigation
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Dashboard", 
        "⚙️ Konfiguration", 
        "📋 Logs", 
        "📈 Metriken", 
        "ℹ️ Info",
        "🏠 Modelle"  # Bestehende Funktionalität
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
        st.title("🏠 Modelle")
        st.info("💡 Die bestehende Modelle-Funktionalität wird hier integriert.")
        # Hier würde die bestehende page_overview() Funktion eingefügt werden

if __name__ == "__main__":
    main()

