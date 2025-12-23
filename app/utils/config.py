import os

# Datenbank (EXTERNE DB!)
DB_DSN = os.getenv("DB_DSN", "postgresql://user:pass@localhost:5432/crypto")

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

