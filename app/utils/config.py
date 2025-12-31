import os
import json

# Pfad zur Runtime-Konfigurationsdatei
CONFIG_FILE = "/tmp/ml_training_config.json"

# Datenbank (EXTERNE DB!)
DB_DSN = os.getenv("DB_DSN", "postgresql://user:pass@localhost:5432/crypto")
DB_REFRESH_INTERVAL = int(os.getenv("DB_REFRESH_INTERVAL", "10"))  # Sekunden zwischen DB-Abfragen

# Ports
API_PORT = int(os.getenv("API_PORT", "8000"))
STREAMLIT_PORT = int(os.getenv("STREAMLIT_PORT", "8501"))

# Modelle
MODEL_STORAGE_PATH = os.getenv("MODEL_STORAGE_PATH", "/app/models")

# Job Queue
JOB_POLL_INTERVAL = int(os.getenv("JOB_POLL_INTERVAL", "5"))  # Sekunden zwischen Job-Checks
MAX_CONCURRENT_JOBS = int(os.getenv("MAX_CONCURRENT_JOBS", "2"))  # Parallele Jobs

# Logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()  # DEBUG, INFO, WARNING, ERROR, CRITICAL
LOG_FORMAT = os.getenv("LOG_FORMAT", "text")  # "text" oder "json"
LOG_JSON_INDENT = int(os.getenv("LOG_JSON_INDENT", "0"))  # 0 = kompakt, 2+ = formatiert

# ML Training Configuration
MODEL_STORAGE_PATH = os.getenv("MODEL_STORAGE_PATH", "/app/models")
MAX_CONCURRENT_JOBS = int(os.getenv("MAX_CONCURRENT_JOBS", "2"))
JOB_POLL_INTERVAL = int(os.getenv("JOB_POLL_INTERVAL", "5"))

# Training Parameters
DEFAULT_TRAINING_HOURS = int(os.getenv("DEFAULT_TRAINING_HOURS", "24"))
MAX_TRAINING_HOURS = int(os.getenv("MAX_TRAINING_HOURS", "168"))  # 1 week
MIN_TRAINING_HOURS = int(os.getenv("MIN_TRAINING_HOURS", "1"))

# ============================================================
# Runtime Configuration Storage
# ============================================================

def load_runtime_config():
    """Load runtime configuration from file"""
    try:
        if os.path.exists(CONFIG_FILE):
            with open(CONFIG_FILE, 'r') as f:
                return json.load(f)
    except Exception:
        pass
    return {}

def save_runtime_config(config):
    """Save runtime configuration to file"""
    try:
        with open(CONFIG_FILE, 'w') as f:
            json.dump(config, f)
    except Exception:
        pass

# Load initial runtime config
_runtime_config = load_runtime_config()

def get_runtime_config(key: str, default=None):
    """Get runtime configuration value"""
    return _runtime_config.get(key, default)

def set_runtime_config(key: str, value):
    """Set runtime configuration value"""
    global _runtime_config
    _runtime_config[key] = value
    save_runtime_config(_runtime_config)

# ============================================================
# Config Management Functions
# ============================================================

def get_config_dict() -> dict:
    """Get all configuration as dictionary"""
    return {
        # Database
        "db_dsn": get_runtime_config('db_dsn', DB_DSN),
        "db_refresh_interval": get_runtime_config('db_refresh_interval', DB_REFRESH_INTERVAL),

        # ML Training
        "model_storage_path": get_runtime_config('model_storage_path', MODEL_STORAGE_PATH),
        "max_concurrent_jobs": get_runtime_config('max_concurrent_jobs', MAX_CONCURRENT_JOBS),
        "job_poll_interval": get_runtime_config('job_poll_interval', JOB_POLL_INTERVAL),
        "default_training_hours": get_runtime_config('default_training_hours', DEFAULT_TRAINING_HOURS),
        "max_training_hours": get_runtime_config('max_training_hours', MAX_TRAINING_HOURS),
        "min_training_hours": get_runtime_config('min_training_hours', MIN_TRAINING_HOURS),
    }

def update_config_from_dict(config_dict: dict) -> None:
    """Update configuration from dictionary (runtime only - no persistence)"""
    for key, value in config_dict.items():
        if key in ['db_dsn', 'db_refresh_interval', 'model_storage_path', 'max_concurrent_jobs',
                   'job_poll_interval', 'default_training_hours', 'max_training_hours', 'min_training_hours']:
            set_runtime_config(key, value)

def reload_config_from_env() -> None:
    """Reload configuration from environment variables"""
    global DB_DSN, DB_REFRESH_INTERVAL, MODEL_STORAGE_PATH, MAX_CONCURRENT_JOBS, JOB_POLL_INTERVAL
    global DEFAULT_TRAINING_HOURS, MAX_TRAINING_HOURS, MIN_TRAINING_HOURS

    # Reload from environment (simulates reading .env file)
    DB_DSN = os.getenv("DB_DSN", "postgresql://user:pass@localhost:5432/crypto")
    DB_REFRESH_INTERVAL = int(os.getenv("DB_REFRESH_INTERVAL", "10"))
    MODEL_STORAGE_PATH = os.getenv("MODEL_STORAGE_PATH", "/app/models")
    MAX_CONCURRENT_JOBS = int(os.getenv("MAX_CONCURRENT_JOBS", "2"))
    JOB_POLL_INTERVAL = int(os.getenv("JOB_POLL_INTERVAL", "5"))
    DEFAULT_TRAINING_HOURS = int(os.getenv("DEFAULT_TRAINING_HOURS", "24"))
    MAX_TRAINING_HOURS = int(os.getenv("MAX_TRAINING_HOURS", "168"))
    MIN_TRAINING_HOURS = int(os.getenv("MIN_TRAINING_HOURS", "1"))

