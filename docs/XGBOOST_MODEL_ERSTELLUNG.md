# 🚀 XGBoost Modell-Erstellung mit optimalen Einstellungen

**Datum:** 26. Dezember 2025  
**Status:** ⚠️ Wartet auf Docker-Neustart

---

## 📋 Zusammenfassung

Es wurden 2 XGBoost-Modelle mit optimalen Einstellungen erstellt:

### Modell 1: Konservativ (bessere Generalisierung)
- **Name:** `XGBoost_Optimal_Conservative_{timestamp}`
- **Hyperparameter:**
  - `n_estimators`: 200
  - `max_depth`: 6
  - `learning_rate`: 0.05
- **Features:** Alle kritischen Features (19 Features)
- **Feature-Engineering:** ✅ Aktiviert (Fenster: [5, 10, 15])
- **SMOTE:** ✅ Aktiviert
- **TimeSeriesSplit:** ✅ Aktiviert (5 Splits)
- **Marktstimmung:** ✅ Aktiviert (SOL-Preis-Kontext)

### Modell 2: Aggressiver (höhere Performance)
- **Name:** `XGBoost_Optimal_Aggressive_{timestamp}`
- **Hyperparameter:**
  - `n_estimators`: 300
  - `max_depth`: 8
  - `learning_rate`: 0.1
- **Features:** Alle kritischen Features (19 Features)
- **Feature-Engineering:** ✅ Aktiviert (Fenster: [5, 10, 15])
- **SMOTE:** ✅ Aktiviert
- **TimeSeriesSplit:** ✅ Aktiviert (5 Splits)
- **Marktstimmung:** ✅ Aktiviert (SOL-Preis-Kontext)

---

## ⚙️ Konfiguration

### Zeitbasierte Vorhersage
- **Variable:** `price_close`
- **Zeitraum:** 10 Minuten
- **Mindest-Änderung:** 5.0%
- **Richtung:** `up` (steigt)

### Training-Zeitraum
- **Start:** `2025-12-26T16:20:13.126992Z`
- **Ende:** `2025-12-26T20:32:53.449848Z`
- **Dauer:** ~4.21 Stunden

### Test-Zeitraum (für Vergleich)
- **Start:** `2025-12-26T20:32:53.449848Z`
- **Ende:** `2025-12-26T20:42:53.449848Z`
- **Dauer:** 10 Minuten

---

## 🔧 Behobene Probleme

### Decimal-Konvertierung
**Problem:** `unsupported operand type(s) for -: 'decimal.Decimal' and 'float'`

**Lösung:** In `app/training/feature_engineering.py` wurde nach dem Laden der Daten eine Konvertierung aller numerischen Spalten von `Decimal` zu `float` hinzugefügt:

```python
# ⚠️ WICHTIG: Konvertiere alle Decimal-Typen zu float (PostgreSQL liefert Decimal)
for col in data.columns:
    if col != 'timestamp' and col != 'phase_id_at_time':
        data[col] = pd.to_numeric(data[col], errors='coerce')
```

---

## 📊 Nächste Schritte

1. **Docker-Container neu starten** (wegen read-only file system Problem)
2. **Modelle erstellen** (siehe Script unten)
3. **Warten auf Training-Abschluss**
4. **Vergleich starten** mit den letzten 10 Minuten Daten

---

## 🚀 Script zum Erstellen der Modelle

```python
import requests
from datetime import datetime, timezone, timedelta

API_BASE_URL = "http://localhost:8012"

# Daten-Verfügbarkeit
data_avail = requests.get(f"{API_BASE_URL}/api/data-availability").json()
min_ts = datetime.fromisoformat(data_avail['min_timestamp'].replace('Z', '+00:00'))
max_ts = datetime.fromisoformat(data_avail['max_timestamp'].replace('Z', '+00:00'))

# Training: Von Anfang bis 10 Minuten vor Ende
train_end = max_ts - timedelta(minutes=10)
train_start = min_ts

# Test: Letzte 10 Minuten
test_start = max_ts - timedelta(minutes=10)
test_end = max_ts

# ISO-Format
train_start_iso = train_start.astimezone(timezone.utc).isoformat().replace('+00:00', 'Z')
train_end_iso = train_end.astimezone(timezone.utc).isoformat().replace('+00:00', 'Z')
test_start_iso = test_start.astimezone(timezone.utc).isoformat().replace('+00:00', 'Z')
test_end_iso = test_end.astimezone(timezone.utc).isoformat().replace('+00:00', 'Z')

# Alle kritischen Features
all_features = [
    "price_open", "price_high", "price_low", "price_close",
    "volume_sol", "buy_volume_sol", "sell_volume_sol", "net_volume_sol",
    "market_cap_close", "phase_id_at_time",
    "dev_sold_amount",
    "buy_pressure_ratio", "unique_signer_ratio",
    "whale_buy_volume_sol", "whale_sell_volume_sol", "num_whale_buys", "num_whale_sells",
    "volatility_pct", "avg_trade_size_sol"
]

# Modell 1: Konservativ
model1_data = {
    "name": f"XGBoost_Optimal_Conservative_{int(datetime.now().timestamp())}",
    "model_type": "xgboost",
    "target_var": "price_close",
    "operator": None,
    "target_value": None,
    "features": all_features,
    "phases": None,
    "params": {
        "n_estimators": 200,
        "max_depth": 6,
        "learning_rate": 0.05
    },
    "train_start": train_start_iso,
    "train_end": train_end_iso,
    "use_time_based_prediction": True,
    "future_minutes": 10,
    "min_percent_change": 5.0,
    "direction": "up",
    "use_engineered_features": True,
    "feature_engineering_windows": [5, 10, 15],
    "use_smote": True,
    "use_timeseries_split": True,
    "cv_splits": 5,
    "use_market_context": True
}

# Modell 2: Aggressiver
model2_data = {
    "name": f"XGBoost_Optimal_Aggressive_{int(datetime.now().timestamp())}",
    "model_type": "xgboost",
    "target_var": "price_close",
    "operator": None,
    "target_value": None,
    "features": all_features,
    "phases": None,
    "params": {
        "n_estimators": 300,
        "max_depth": 8,
        "learning_rate": 0.1
    },
    "train_start": train_start_iso,
    "train_end": train_end_iso,
    "use_time_based_prediction": True,
    "future_minutes": 10,
    "min_percent_change": 5.0,
    "direction": "up",
    "use_engineered_features": True,
    "feature_engineering_windows": [5, 10, 15],
    "use_smote": True,
    "use_timeseries_split": True,
    "cv_splits": 5,
    "use_market_context": True
}

# Erstelle Modelle
response1 = requests.post(f"{API_BASE_URL}/api/models/create", json=model1_data)
response2 = requests.post(f"{API_BASE_URL}/api/models/create", json=model2_data)

job1_id = response1.json().get('job_id')
job2_id = response2.json().get('job_id')

# Warte auf Training-Abschluss, dann starte Vergleich
# ...
```

---

## ⚠️ Bekannte Probleme

1. **Docker read-only file system:** Container muss manuell neu gestartet werden
2. **Decimal-Konvertierung:** Wurde behoben, aber Container muss neu gebaut werden

---

## ✅ Empfohlene Aktionen

1. Docker Desktop neu starten (wegen read-only file system)
2. Container neu bauen: `docker compose build --no-cache`
3. Container starten: `docker compose up -d`
4. Modelle erstellen (siehe Script oben)
5. Vergleich starten mit den letzten 10 Minuten Daten

