# 🚀 Sofort-Lösung: XGBoost-Modelle erstellen

## ⚠️ Aktuelles Problem

Docker-Container ist in einem inkonsistenten Zustand:
- `docker ps` zeigt Container als "Up" und "healthy"
- Aber `docker exec` sagt Container läuft nicht
- API ist nicht erreichbar

## ✅ Lösung (3 Schritte)

### Schritt 1: Docker Desktop neu starten
```bash
# Docker Desktop komplett beenden und neu starten
# Warten bis Docker vollständig geladen ist
```

### Schritt 2: Container neu starten
```bash
cd "/Users/moritzhaslbeck/Library/Mobile Documents/com~apple~CloudDocs/cursor ai projekte/pump-training/ml-training-service"

# Container stoppen und entfernen
docker compose down

# Container neu erstellen
docker compose up -d

# Warten bis Container läuft (ca. 10-15 Sekunden)
sleep 15

# Prüfen ob API erreichbar ist
curl http://localhost:8012/api/health
```

### Schritt 3: Modelle erstellen
```bash
# Script ausführen (erstellt automatisch 2 Modelle und startet Vergleich)
python3 scripts/create_xgboost_models_and_compare.py
```

## 📊 Was das Script macht

1. **Erstellt Modell 1 (Konservativ):**
   - XGBoost mit n_estimators=200, max_depth=6, learning_rate=0.05
   - Alle kritischen Features
   - Feature-Engineering aktiviert
   - SMOTE aktiviert
   - Marktstimmung aktiviert

2. **Erstellt Modell 2 (Aggressiver):**
   - XGBoost mit n_estimators=300, max_depth=8, learning_rate=0.1
   - Gleiche Features wie Modell 1

3. **Wartet auf Training-Abschluss** (beide Modelle)

4. **Startet automatisch Vergleich** mit den letzten 10 Minuten Daten

## ✅ Was bereits behoben wurde

- ✅ Decimal-Konvertierung (PostgreSQL → float)
- ✅ UI-Fehler (Checkbox-Handler, Auto-Refresh)
- ✅ API-Funktionen verbessert
- ✅ Script erstellt für automatische Modell-Erstellung

## 🔍 Falls weiterhin Probleme

1. Docker Desktop komplett neu installieren
2. Oder: Container manuell über Web UI erstellen (http://localhost:8502)

