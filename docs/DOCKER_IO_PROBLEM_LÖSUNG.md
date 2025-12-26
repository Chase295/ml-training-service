# 🔧 Docker I/O-Problem Lösung

**Problem:** `input/output error` beim Zugriff auf Docker-Container

## 🚨 Symptome
- Container zeigt Status "dead" oder "Up" aber Services nicht erreichbar
- Fehler: `write /var/lib/desktop-containerd/daemon/...: input/output error`
- API nicht erreichbar auf Port 8012

## ✅ Lösung

### Schritt 1: Docker Desktop neu starten
1. Docker Desktop komplett beenden
2. Docker Desktop neu starten
3. Warten bis Docker vollständig geladen ist

### Schritt 2: Container neu erstellen
```bash
cd /Users/moritzhaslbeck/Library/Mobile\ Documents/com~apple~CloudDocs/cursor\ ai\ projekte/pump-training/ml-training-service

# Container entfernen
docker compose down

# Container neu erstellen
docker compose up -d --build
```

### Schritt 3: Prüfen ob API erreichbar ist
```bash
curl http://localhost:8012/api/health
```

### Schritt 4: Modelle erstellen
```bash
python3 scripts/create_xgboost_models_and_compare.py
```

## 🔄 Alternative: Docker System Prüfen

Falls das Problem weiterhin besteht:

```bash
# Docker System prüfen
docker system df
docker system prune -a  # ⚠️ Vorsicht: Löscht alle nicht verwendeten Images/Container

# Docker Desktop komplett neu installieren (letzter Ausweg)
```

## 📝 Was wurde bereits behoben

✅ **Decimal-Konvertierung:** Alle numerischen Spalten werden jetzt automatisch von `Decimal` zu `float` konvertiert (in `app/training/feature_engineering.py`)

✅ **Script erstellt:** `scripts/create_xgboost_models_and_compare.py` - Erstellt automatisch 2 XGBoost-Modelle und startet Vergleich

✅ **UI-Fehler behoben:** Checkbox-Handler, Auto-Refresh, API-Funktionen

