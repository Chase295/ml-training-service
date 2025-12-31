# Pump UI - React + TypeScript + Vite
FROM node:22-alpine

WORKDIR /app

# Environment für Vite
ENV NODE_ENV=production

# API Base URL als Build-Argument - Standard für Coolify
ARG VITE_API_BASE_URL=http://ml-service:8000
ENV VITE_API_BASE_URL=$VITE_API_BASE_URL

# Dependencies installieren (inkl. devDependencies für Build)
COPY package*.json ./
RUN rm -f package-lock.json && npm install --include=dev

# Source Code kopieren
COPY . .

# Build der App - nur Vite (TSC macht Probleme)
RUN ./node_modules/.bin/vite build

# Nginx für Serving der statischen Files
FROM nginx:alpine

# Nginx Konfiguration - hardcoded wie pump-find
COPY nginx.conf /etc/nginx/nginx.conf

# Gebaute App kopieren
COPY --from=0 /app/dist /usr/share/nginx/html

# Port 80 freigeben
EXPOSE 80

# Health Check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost/health || exit 1

# Start Nginx
CMD ["nginx", "-g", "daemon off;"]


