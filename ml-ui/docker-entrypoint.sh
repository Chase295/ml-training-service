#!/bin/sh

# Setze Standardwerte für Environment-Variablen
BACKEND_URL=${BACKEND_URL:-"ml-service:8000"}

# Generiere nginx-Konfiguration aus Template
envsubst '${BACKEND_URL}' < /etc/nginx/nginx.conf.template > /etc/nginx/nginx.conf

# Zeige generierte Konfiguration für Debugging
echo "=== Generated nginx.conf ==="
cat /etc/nginx/nginx.conf
echo "==========================="

# Starte nginx
exec nginx -g "daemon off;"
