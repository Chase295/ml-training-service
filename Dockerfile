# ML Training Service Dockerfile
# Python 3.11-slim mit FastAPI + Streamlit
FROM python:3.11-slim

WORKDIR /app

# System-Dependencies installieren
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl supervisor && \
    rm -rf /var/lib/apt/lists/*

# Python Dependencies installieren
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App-Code kopieren
COPY app/ ./app/

# Models-Verzeichnis erstellen (wird als Volume gemappt)
RUN mkdir -p /app/models

# Supervisor Config für zwei Prozesse (FastAPI + Streamlit)
RUN printf '[supervisord]\n\
nodaemon=true\n\
\n\
[program:fastapi]\n\
command=uvicorn app.main:app --host 0.0.0.0 --port 8000\n\
directory=/app\n\
autostart=true\n\
autorestart=true\n\
stderr_logfile=/dev/stderr\n\
stderr_logfile_maxbytes=0\n\
stdout_logfile=/dev/stdout\n\
stdout_logfile_maxbytes=0\n\
\n\
[program:streamlit]\n\
command=streamlit run app/streamlit_app.py --server.port 8501 --server.address 0.0.0.0 --server.headless=true\n\
directory=/app\n\
autostart=true\n\
autorestart=true\n\
stderr_logfile=/dev/stderr\n\
stderr_logfile_maxbytes=0\n\
stdout_logfile=/dev/stdout\n\
stdout_logfile_maxbytes=0\n\
' > /etc/supervisor/conf.d/supervisord.conf

# Ports freigeben
EXPOSE 8000 8501

# Health Check (wie in pump-discover)
HEALTHCHECK --interval=10s --timeout=5s --start-period=10s --retries=5 \
    CMD curl -f http://localhost:8000/api/health || exit 1

# Graceful Shutdown
STOPSIGNAL SIGTERM

# Start Supervisor (startet beide Prozesse)
CMD ["/usr/bin/supervisord", "-c", "/etc/supervisor/conf.d/supervisord.conf"]

