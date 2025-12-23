# 🎯 Modell-Erstellung - Vollständige Übersicht

## 📋 Inhaltsverzeichnis
1. [Erstellungs-Methoden](#erstellungs-methoden)
2. [Modell-Typen](#modell-typen)
3. [Vorhersage-Arten](#vorhersage-arten)
4. [Features & Phasen](#features--phasen)
5. [Hyperparameter](#hyperparameter)
6. [Beispiele](#beispiele)

---

## 🚀 Erstellungs-Methoden

### 1. Web UI (Streamlit)
**URL:** http://localhost:8501

**Vorteile:**
- ✅ Einfache Bedienung
- ✅ Visuelle Auswahl aller Optionen
- ✅ Live-Status-Anzeige
- ✅ Automatische Validierung

**Schritte:**
1. Öffne http://localhost:8501
2. Gehe zu "Modell trainieren"
3. Fülle alle Felder aus
4. Klicke "Modell trainieren"

---

### 2. REST API
**Base URL:** http://localhost:8000/api

**Endpoint:** `POST /api/models/create`

**Vorteile:**
- ✅ Automatisierung möglich
- ✅ Integration in andere Systeme
- ✅ Skript-basierte Erstellung

**Beispiel:**
```bash
curl -X POST http://localhost:8000/api/models/create \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Mein_Modell_1",
    "model_type": "random_forest",
    "features": ["price_open", "price_high", "volume_sol"],
    "phases": [1, 2],
    "train_start": "2025-12-01T00:00:00Z",
    "train_end": "2025-12-20T00:00:00Z",
    "target_var": "market_cap_close",
    "operator": ">",
    "target_value": 50000
  }'
```

---

## 🤖 Modell-Typen

**⚠️ WICHTIG:** Aktuell werden nur `random_forest` und `xgboost` in der API/UI unterstützt, obwohl das Schema mehr Typen erlaubt.

### 1. Random Forest (`random_forest`) ✅ VERFÜGBAR
**Beschreibung:** Ensemble-Methode mit mehreren Entscheidungsbäumen

**Standard-Hyperparameter:**
```json
{
  "n_estimators": 100,
  "max_depth": 10,
  "min_samples_split": 2
}
```

**Vorteile:**
- ✅ Robust gegen Overfitting
- ✅ Gute Performance auf strukturierten Daten
- ✅ Schnelles Training

**Wann verwenden:**
- Für allgemeine Klassifikationsaufgaben
- Wenn du schnell Ergebnisse brauchst
- Bei mittelgroßen Datensätzen

---

### 2. XGBoost (`xgboost`) ✅ VERFÜGBAR
**Beschreibung:** Gradient Boosting mit optimierter Implementierung

**Standard-Hyperparameter:**
```json
{
  "n_estimators": 100,
  "max_depth": 6,
  "learning_rate": 0.1
}
```

**Vorteile:**
- ✅ Sehr hohe Genauigkeit
- ✅ Gut für komplexe Muster
- ✅ Feature Importance automatisch

**Wann verwenden:**
- Für beste Performance
- Bei komplexen Datenmustern
- Wenn Genauigkeit wichtiger als Geschwindigkeit ist

---

### 3. Gradient Boosting (`gradient_boosting`) ⚠️ NICHT IN API/UI
**Beschreibung:** Sequentielles Boosting-Verfahren

**Standard-Hyperparameter:**
```json
{
  "n_estimators": 100,
  "max_depth": 3,
  "learning_rate": 0.1
}
```

**Status:** Im Schema definiert, aber noch nicht in API/UI implementiert

---

### 4. Logistic Regression (`logistic_regression`) ⚠️ NICHT IN API/UI
**Beschreibung:** Lineares Modell für Klassifikation

**Standard-Hyperparameter:**
```json
{
  "C": 1.0,
  "max_iter": 100
}
```

**Status:** Im Schema definiert, aber noch nicht in API/UI implementiert

---

### 5. Neural Network (`neural_network`) ⚠️ NICHT IN API/UI
**Beschreibung:** Multi-Layer Perceptron (MLP)

**Standard-Hyperparameter:**
```json
{
  "hidden_layers": [100, 50],
  "activation": "relu",
  "max_iter": 200
}
```

**Status:** Im Schema definiert, aber noch nicht in API/UI implementiert

---

## 🎯 Vorhersage-Arten

### 1. Normale Vorhersage (Klassifikation)
**Was wird vorhergesagt:** Ob eine Variable einen bestimmten Wert erreicht

**Konfiguration:**
- `use_time_based_prediction: false`
- `target_var`: Variable die überwacht wird (z.B. `market_cap_close`)
- `operator`: Vergleichsoperator (`>`, `<`, `>=`, `<=`, `=`)
- `target_value`: Schwellenwert (z.B. `50000`)

**Beispiel:**
```
Vorhersage: Wird market_cap_close > 50000?
```

**Label-Erstellung:**
- `1` wenn `market_cap_close > 50000`
- `0` wenn `market_cap_close <= 50000`

---

### 2. Zeitbasierte Vorhersage ⏰
**Was wird vorhergesagt:** Ob eine Variable innerhalb von X Minuten um X% steigt/fällt

**Konfiguration:**
- `use_time_based_prediction: true`
- `target_var`: Variable die überwacht wird (z.B. `price_close`)
- `future_minutes`: Minuten in die Zukunft (z.B. `10`)
- `min_percent_change`: Mindest-Prozent-Änderung (z.B. `5.0` für 5%)
- `direction`: Richtung (`up` oder `down`)

**Beispiel:**
```
Vorhersage: Steigt price_close innerhalb der nächsten 10 Minuten um mindestens 5%?
```

**Label-Erstellung:**
- `1` wenn `(price_future - price_now) / price_now * 100 >= 5.0`
- `0` sonst

**Wichtig:**
- Berücksichtigt `interval_seconds` pro Phase automatisch
- Verwendet zukünftige Daten für Labels (nur beim Training!)

---

## 📊 Features & Phasen

### Verfügbare Features
Alle Spalten aus der `coin_metrics` Tabelle:

**Preis-Features:**
- `price_open` - Eröffnungspreis
- `price_high` - Höchstpreis
- `price_low` - Tiefstpreis
- `price_close` - Schlusspreis

**Volumen-Features:**
- `volume_sol` - Gesamt-Volumen in SOL
- `buy_volume_sol` - Kauf-Volumen in SOL
- `sell_volume_sol` - Verkaufs-Volumen in SOL

**Market Cap:**
- `market_cap_close` - Market Cap zum Schluss

**Trading-Features:**
- `num_buys` - Anzahl Käufe
- `num_sells` - Anzahl Verkäufe
- `num_micro_trades` - Anzahl Micro-Trades
- `max_single_buy_sol` - Größter Einzelkauf in SOL
- `max_single_sell_sol` - Größter Einzelverkauf in SOL

**Weitere Features:**
- `bonding_curve_pct` - Bonding Curve Prozent
- `virtual_sol_reserves` - Virtuelle SOL-Reserven
- `unique_wallets` - Anzahl einzigartiger Wallets
- `dev_sold_amount` - Entwickler-Verkaufsbetrag
- `is_koth` - King of the Hill Status (boolean)

**Empfehlung:**
- Mindestens 3-5 Features verwenden
- Kombiniere Preis, Volumen und Market Cap
- Beispiel: `["price_open", "price_high", "price_low", "volume_sol", "market_cap_close"]`
- Für Trading: `["price_close", "buy_volume_sol", "sell_volume_sol", "num_buys", "num_sells"]`

---

### Coin-Phasen
Phasen werden dynamisch aus `ref_coin_phases` geladen.

**Verfügbare Phasen (aktuell):**
- **Phase 1: Baby Zone** - Interval: 5s, Max Age: 20min
- **Phase 2: Survival Zone** - Interval: 30s, Max Age: 60min
- **Phase 3: Mature Zone** - Interval: 60s, Max Age: 1440min
- **Phase 99: Finished** - Interval: 0s, Max Age: 999999min

**Jede Phase hat:**
- `interval_seconds` - Zeitintervall zwischen Metriken
- `max_age_minutes` - Maximale Alter der Daten

**Empfehlung:**
- Mehrere Phasen kombinieren für mehr Daten
- Beispiel: `[1, 2]` für Phase 1 und 2 (Baby + Survival)
- Phase 1 hat die meisten Datenpunkte (5s Interval)
- Phase 3 für längerfristige Muster

**Wichtig bei zeitbasierten Vorhersagen:**
- `interval_seconds` wird automatisch berücksichtigt
- Verschiedene Phasen haben verschiedene Intervalle
- Phase 1 (5s) = 12 Datenpunkte pro Minute
- Phase 2 (30s) = 2 Datenpunkte pro Minute
- Phase 3 (60s) = 1 Datenpunkt pro Minute

---

## ⚙️ Hyperparameter

### Wann anpassen?
- ✅ Wenn Standard-Parameter nicht ausreichen
- ✅ Für Hyperparameter-Tuning
- ✅ Bei spezifischen Anforderungen

### Wie anpassen?

**In Web UI:**
1. Aktiviere "Hyperparameter anpassen"
2. Gib JSON ein, z.B.:
```json
{
  "n_estimators": 200,
  "max_depth": 15,
  "min_samples_split": 5
}
```

**Über API:**
```json
{
  "params": {
    "n_estimators": 200,
    "max_depth": 15
  }
}
```

### Wichtige Parameter pro Modell-Typ

**Random Forest:**
- `n_estimators`: Anzahl Bäume (mehr = genauer, aber langsamer)
- `max_depth`: Maximale Tiefe (mehr = komplexer)
- `min_samples_split`: Min. Samples zum Splitten

**XGBoost:**
- `n_estimators`: Anzahl Boosting-Runden
- `max_depth`: Maximale Tiefe
- `learning_rate`: Lernrate (kleiner = langsamer, aber genauer)

**Gradient Boosting:**
- Ähnlich wie XGBoost

**Logistic Regression:**
- `C`: Regularisierungsstärke (größer = weniger Regularisierung)
- `max_iter`: Maximale Iterationen

**Neural Network:**
- `hidden_layers`: Liste der Layer-Größen, z.B. `[100, 50]`
- `activation`: Aktivierungsfunktion (`relu`, `tanh`, `sigmoid`)
- `max_iter`: Maximale Iterationen

---

## 📝 Beispiele

### Beispiel 1: Einfaches Random Forest Modell
**Ziel:** Vorhersage ob Market Cap > 50000

**Web UI:**
- Modell-Typ: `random_forest`
- Features: `["price_open", "price_high", "price_low", "volume_sol", "market_cap_close"]`
- Phasen: `[1, 2]`
- Trainings-Zeitraum: `2025-12-01` bis `2025-12-20`
- Target: `market_cap_close > 50000`
- Zeitbasierte Vorhersage: ❌ Deaktiviert

**API:**
```json
{
  "name": "RF_MarketCap_50000",
  "model_type": "random_forest",
  "features": ["price_open", "price_high", "price_low", "volume_sol", "market_cap_close"],
  "phases": [1, 2],
  "train_start": "2025-12-01T00:00:00Z",
  "train_end": "2025-12-20T23:59:59Z",
  "target_var": "market_cap_close",
  "operator": ">",
  "target_value": 50000
}
```

---

### Beispiel 2: Zeitbasierte XGBoost Vorhersage
**Ziel:** Vorhersage ob Preis in 10 Min um 5% steigt

**Web UI:**
- Modell-Typ: `xgboost`
- Features: `["price_open", "price_high", "price_low", "volume_sol"]`
- Phasen: `[1, 2]`
- Trainings-Zeitraum: `2025-12-01` bis `2025-12-20`
- Zeitbasierte Vorhersage: ✅ Aktiviert
- Zukünftige Minuten: `10`
- Min. Prozentuale Änderung: `5.0`
- Richtung: `up`
- Target-Variable: `price_close` (wird überwacht)

**API:**
```json
{
  "name": "XGB_Price_10min_5pct",
  "model_type": "xgboost",
  "features": ["price_open", "price_high", "price_low", "volume_sol"],
  "phases": [1, 2],
  "train_start": "2025-12-01T00:00:00Z",
  "train_end": "2025-12-20T23:59:59Z",
  "use_time_based_prediction": true,
  "target_var": "price_close",
  "future_minutes": 10,
  "min_percent_change": 5.0,
  "direction": "up"
}
```

---

### Beispiel 3: Mit angepassten Hyperparametern
**Ziel:** Optimiertes Random Forest Modell

**Web UI:**
- Alle Einstellungen wie Beispiel 1
- Hyperparameter anpassen: ✅ Aktiviert
- Hyperparameter JSON:
```json
{
  "n_estimators": 200,
  "max_depth": 15,
  "min_samples_split": 5
}
```

**API:**
```json
{
  "name": "RF_Optimized",
  "model_type": "random_forest",
  "features": ["price_open", "price_high", "price_low", "volume_sol", "market_cap_close"],
  "phases": [1, 2],
  "train_start": "2025-12-01T00:00:00Z",
  "train_end": "2025-12-20T23:59:59Z",
  "target_var": "market_cap_close",
  "operator": ">",
  "target_value": 50000,
  "params": {
    "n_estimators": 200,
    "max_depth": 15,
    "min_samples_split": 5
  }
}
```

---

## 🎓 Best Practices

### 1. Feature-Auswahl
- ✅ Verwende 3-10 Features
- ✅ Kombiniere verschiedene Feature-Typen (Preis, Volumen, Market Cap)
- ✅ Vermeide zu viele Features (Overfitting-Risiko)

### 2. Trainings-Zeitraum
- ✅ Mindestens 1-2 Wochen Daten
- ✅ Vermeide zu kurze Zeiträume (< 1 Tag)
- ✅ Berücksichtige Marktzyklen

### 3. Phasen-Auswahl
- ✅ Mehrere Phasen = mehr Daten
- ✅ Prüfe welche Phasen Daten haben
- ✅ Bei zeitbasierten Vorhersagen: Phasen mit ähnlichen Intervallen kombinieren

### 4. Modell-Typ
- ✅ Starte mit `random_forest` (schnell, robust)
- ✅ Für beste Performance: `xgboost`
- ✅ Für Experimente: Verschiedene Typen testen

### 5. Zeitbasierte Vorhersagen
- ✅ Realistische Zeiträume (5-30 Minuten)
- ✅ Realistische Prozent-Änderungen (1-10%)
- ✅ Teste verschiedene Kombinationen

### 6. Hyperparameter
- ✅ Beginne mit Standard-Werten
- ✅ Passe nur an wenn nötig
- ✅ Teste systematisch (nicht zu viele Parameter gleichzeitig)

---

## 🔍 Nach der Erstellung

### 1. Modell-Status prüfen
- Status: `READY` = erfolgreich
- Status: `FAILED` = Fehler (siehe Logs)
- Status: `TRAINING` = läuft noch

### 2. Metriken ansehen
- `training_accuracy`: Genauigkeit auf Trainingsdaten
- `training_f1`: F1-Score
- `training_precision`: Precision
- `training_recall`: Recall
- `feature_importance`: Wichtigkeit der Features

### 3. Modell testen
- Erstelle Test-Job mit neuen Daten
- Vergleiche mit anderen Modellen
- Prüfe Overlap-Warnungen

---

## 📚 Zusammenfassung

**Du kannst Modelle erstellen mit:**
- ✅ 5 Modell-Typen (Random Forest, XGBoost, Gradient Boosting, Logistic Regression, Neural Network)
- ✅ 2 Vorhersage-Arten (Normal, Zeitbasiert)
- ✅ Viele Features aus `coin_metrics`
- ✅ Dynamische Phasen-Auswahl
- ✅ Anpassbare Hyperparameter
- ✅ 2 Methoden (Web UI, REST API)

**Nächste Schritte:**
1. Erstelle dein erstes Modell über Web UI
2. Teste verschiedene Modell-Typen
3. Experimentiere mit zeitbasierten Vorhersagen
4. Vergleiche Modelle miteinander

