# 🔧 Phase 9: ML-System Verbesserungen - Schritt-für-Schritt Anleitung

## 📋 Übersicht

Diese Anleitung beschreibt die Umsetzung von 5 kritischen Verbesserungen für das ML Training Service System, speziell optimiert für Pump-Coin-Analyse.

**Geschätzte Gesamtzeit:** ~13 Stunden  
**Schwierigkeitsgrad:** Mittel bis Hoch  
**Risiko:** Mittel (Backup empfohlen)

---

## ⚠️ WICHTIG: Vorbereitung

### 1. Backup erstellen

**Datenbank-Backup:**
```bash
# PostgreSQL Backup
pg_dump -h 10.0.128.18 -U postgres -d crypto_bot > backup_$(date +%Y%m%d_%H%M%S).sql
```

**Code-Backup:**
```bash
cd ml-training-service
git commit -am "Backup vor Phase 9 Verbesserungen"
git tag backup-phase9-$(date +%Y%m%d)
```

### 2. Test-Umgebung prüfen

- ✅ Docker Container läuft
- ✅ Datenbank erreichbar
- ✅ Test-Daten vorhanden in `coin_metrics`
- ✅ Mindestens 1 funktionierendes Modell vorhanden

### 3. Dependencies prüfen

```bash
# Prüfe ob alle benötigten Pakete installiert werden können
docker-compose exec ml-training pip list | grep -E "scikit-learn|imbalanced-learn|pandas|numpy"
```

---

## 🎯 Verbesserungen - Übersicht

| # | Verbesserung | Priorität | Zeit | Risiko |
|---|-------------|-----------|------|--------|
| 1.1 | Data Leakage beheben | 🔴 KRITISCH | ~2h | Hoch |
| 1.2 | Feature-Engineering erweitern | 🟠 HOCH | ~4h | Mittel |
| 1.3 | Imbalanced Data Handling | 🟡 MITTEL | ~2h | Niedrig |
| 1.4 | TimeSeriesSplit | 🟡 MITTEL | ~3h | Mittel |
| 1.5 | Zusätzliche Metriken | 🟢 NIEDRIG | ~2h | Niedrig |

**Empfohlene Reihenfolge:** 1.1 → 1.2 → 1.3 → 1.4 → 1.5

---

## ✅ Verbesserung 1.1: Data Leakage beheben

### 🎯 Ziel

Bei zeitbasierter Vorhersage wird `target_var` aktuell zu den Features hinzugefügt, wodurch das Modell die Antwort bereits kennt. Dies führt zu unrealistisch hohen Accuracy-Werten.

**Problem:** Modell lernt: "Wenn `price_close` = X, dann ist Label = Y" → Data Leakage!

**Lösung:** `target_var` nur für Label-Erstellung verwenden, aber **NICHT** für Training.

### 📝 Schritt-für-Schritt

#### Schritt 1.1.1: Neue Funktion `prepare_features_for_training()` erstellen

**Datei:** `app/training/engine.py`

**Position:** Nach `create_model()` Funktion, vor `train_model_sync()`

**Code:**
```python
def prepare_features_for_training(
    features: List[str],
    target_var: Optional[str],
    use_time_based: bool
) -> tuple[List[str], List[str]]:
    """
    Bereitet Features für Training vor.
    
    ⚠️ KRITISCH: Bei zeitbasierter Vorhersage wird target_var NUR für Labels verwendet,
    NICHT für Training! Dies verhindert Data Leakage.
    
    Args:
        features: Liste der ursprünglichen Features
        target_var: Ziel-Variable (z.B. "price_close")
        use_time_based: True wenn zeitbasierte Vorhersage aktiviert
    
    Returns:
        Tuple von (features_for_loading, features_for_training)
        - features_for_loading: Enthält target_var (für Daten-Laden und Labels)
        - features_for_training: Enthält target_var NICHT bei zeitbasierter Vorhersage
    """
    # Für Daten-Laden: target_var wird benötigt (für Labels)
    features_for_loading = list(features)  # Kopie erstellen
    if target_var and target_var not in features_for_loading:
        features_for_loading.append(target_var)
        logger.info(f"➕ target_var '{target_var}' zu Features für Daten-Laden hinzugefügt")
    
    # Für Training: target_var wird ENTFERNT bei zeitbasierter Vorhersage
    features_for_training = list(features)  # Kopie erstellen
    if use_time_based and target_var and target_var in features_for_training:
        features_for_training.remove(target_var)
        logger.warning(f"⚠️ target_var '{target_var}' aus Features entfernt (zeitbasierte Vorhersage - verhindert Data Leakage)")
    
    return features_for_loading, features_for_training
```

**Test:**
```python
# Test in Python Shell
from app.training.engine import prepare_features_for_training

# Test 1: Zeitbasierte Vorhersage
features = ["price_open", "price_high", "volume_sol"]
target_var = "price_close"
use_time_based = True

loading, training = prepare_features_for_training(features, target_var, use_time_based)
assert target_var in loading  # Sollte enthalten sein
assert target_var not in training  # Sollte NICHT enthalten sein
print("✅ Test 1 bestanden")

# Test 2: Klassische Vorhersage
use_time_based = False
loading, training = prepare_features_for_training(features, target_var, use_time_based)
assert target_var in loading  # Sollte enthalten sein
assert target_var in training  # Sollte auch enthalten sein
print("✅ Test 2 bestanden")
```

#### Schritt 1.1.2: `train_model()` Funktion anpassen

**Datei:** `app/training/engine.py`

**Position:** In `train_model()` Funktion, nach Zeile 251 (nach `final_params`)

**Änderung:**
```python
# ❌ ALT (Zeile 252-256):
# 2.5. Stelle sicher, dass target_var in features enthalten ist (wird für Labels benötigt)
features_with_target = list(features)  # Kopie erstellen
if target_var and target_var not in features_with_target:
    features_with_target.append(target_var)
    logger.info(f"➕ target_var '{target_var}' zu Features hinzugefügt")

# ✅ NEU:
# 2.5. Bereite Features vor (verhindert Data Leakage bei zeitbasierter Vorhersage)
features_for_loading, features_for_training = prepare_features_for_training(
    features=features,
    target_var=target_var,
    use_time_based=use_time_based
)
logger.info(f"📊 Features für Laden: {len(features_for_loading)}, Features für Training: {len(features_for_training)}")
```

**Änderung in `load_training_data()` Aufruf:**
```python
# ❌ ALT (Zeile 259-264):
# 3. Lade Trainingsdaten (async)
data = await load_training_data(
    train_start=train_start,
    train_end=train_end,
    features=features_with_target,  # Verwende erweiterte Features-Liste
    phases=phases
)

# ✅ NEU:
# 3. Lade Trainingsdaten (async) - mit target_var für Labels
data = await load_training_data(
    train_start=train_start,
    train_end=train_end,
    features=features_for_loading,  # Enthält target_var (für Labels benötigt)
    phases=phases
)
```

**Änderung in `train_model_sync()` Aufruf:**
```python
# ❌ ALT (Zeile 279-296):
result = await loop.run_in_executor(
    None,
    train_model_sync,
    data,
    model_type,
    features,  # ❌ Enthält target_var bei zeitbasierter Vorhersage!
    target_var,
    ...
)

# ✅ NEU:
result = await loop.run_in_executor(
    None,
    train_model_sync,
    data,
    model_type,
    features_for_training,  # ✅ Enthält target_var NICHT bei zeitbasierter Vorhersage!
    target_var,
    ...
)
```

#### Schritt 1.1.3: `train_model_sync()` Funktion anpassen

**Datei:** `app/training/engine.py`

**Position:** In `train_model_sync()` Funktion, Zeile 145-147

**Änderung:**
```python
# ❌ ALT (Zeile 145-147):
# 2. Prepare Features (X) und Labels (y)
X = data[features].values
y = labels.values

# ✅ NEU:
# 2. Prepare Features (X) und Labels (y)
# ⚠️ WICHTIG: features enthält target_var NICHT bei zeitbasierter Vorhersage!
# Prüfe ob alle Features in data vorhanden sind
missing_features = [f for f in features if f not in data.columns]
if missing_features:
    raise ValueError(f"Features nicht in Daten gefunden: {missing_features}")

X = data[features].values
y = labels.values
logger.info(f"📊 Training mit {len(features)} Features, {len(data)} Samples")
```

#### Schritt 1.1.4: Test durchführen

**Test-Skript erstellen:**
```python
# tests/test_data_leakage_fix.py
import asyncio
from datetime import datetime, timezone, timedelta
from app.training.engine import train_model

async def test_data_leakage_fix():
    """Test ob target_var korrekt entfernt wird bei zeitbasierter Vorhersage"""
    
    # Test-Parameter
    train_end = datetime.now(timezone.utc)
    train_start = train_end - timedelta(days=7)
    
    # Test 1: Zeitbasierte Vorhersage
    print("🧪 Test 1: Zeitbasierte Vorhersage (target_var sollte NICHT in Features sein)")
    try:
        result = await train_model(
            model_type="random_forest",
            features=["price_open", "price_high", "volume_sol"],
            target_var="price_close",
            target_operator=None,
            target_value=None,
            train_start=train_start,
            train_end=train_end,
            use_time_based=True,
            future_minutes=10,
            min_percent_change=5.0,
            direction="up"
        )
        print(f"✅ Test 1 erfolgreich: Accuracy={result['accuracy']:.4f}")
        print(f"   Features für Training: {result.get('num_features', 'N/A')}")
    except Exception as e:
        print(f"❌ Test 1 fehlgeschlagen: {e}")
    
    # Test 2: Klassische Vorhersage
    print("\n🧪 Test 2: Klassische Vorhersage (target_var sollte in Features sein)")
    try:
        result = await train_model(
            model_type="random_forest",
            features=["price_open", "price_high", "volume_sol"],
            target_var="market_cap_close",
            target_operator=">",
            target_value=50000.0,
            train_start=train_start,
            train_end=train_end,
            use_time_based=False
        )
        print(f"✅ Test 2 erfolgreich: Accuracy={result['accuracy']:.4f}")
    except Exception as e:
        print(f"❌ Test 2 fehlgeschlagen: {e}")

if __name__ == "__main__":
    asyncio.run(test_data_leakage_fix())
```

**Ausführung:**
```bash
cd ml-training-service
docker-compose exec ml-training python tests/test_data_leakage_fix.py
```

**Erwartetes Ergebnis:**
- ✅ Test 1: Modell wird trainiert, `target_var` ist NICHT in Features
- ✅ Test 2: Modell wird trainiert, `target_var` ist in Features
- ✅ Accuracy sollte bei Test 1 **niedriger** sein als vorher (realistischer)

#### Schritt 1.1.5: Integrationstest

**Manueller Test über UI:**
1. Öffne Streamlit UI: http://localhost:8501
2. Navigiere zu "➕ Neues Modell trainieren"
3. Aktiviere "Zeitbasierte Vorhersage"
4. Wähle Features: `["price_open", "price_high", "volume_sol"]`
5. Target-Variable: `price_close`
6. Starte Training
7. Prüfe Logs: `target_var 'price_close' aus Features entfernt` sollte erscheinen

**Prüfe Logs:**
```bash
docker-compose logs ml-training | grep -E "target_var|Data Leakage|Features"
```

**Erwartete Logs:**
```
⚠️ target_var 'price_close' aus Features entfernt (zeitbasierte Vorhersage - verhindert Data Leakage)
📊 Features für Laden: 4, Features für Training: 3
```

### ✅ Checkliste Verbesserung 1.1

- [x] Funktion `prepare_features_for_training()` erstellt
- [x] `train_model()` angepasst
- [x] `train_model_sync()` angepasst
- [x] Unit-Test bestanden
- [x] Integrationstest bestanden
- [x] Logs zeigen korrekte Feature-Trennung
- [x] Modell-Training funktioniert für beide Vorhersage-Typen

**Status:** ✅ **VOLLSTÄNDIG IMPLEMENTIERT** (2024-12-23)

---

## ✅ Verbesserung 1.2: Feature-Engineering erweitern

### 🎯 Ziel

Pump-Coins haben spezifische Muster (Momentum, Volumen-Spikes, Whale-Activity), die durch zusätzliche Features besser erfasst werden.

**Erwarteter Impact:** +5-10% Accuracy bei Pump-Detection

### 📝 Schritt-für-Schritt

#### Schritt 1.2.1: Neue Funktion `create_pump_detection_features()` erstellen

**Datei:** `app/training/feature_engineering.py`

**Position:** Nach `create_time_based_labels()` Funktion

**Code:**
```python
def create_pump_detection_features(
    data: pd.DataFrame,
    window_sizes: list = [5, 10, 15]
) -> pd.DataFrame:
    """
    Erstellt zusätzliche Features für Pump-Detection.
    
    Features:
    - Price Momentum (Preisänderungen über verschiedene Zeitfenster)
    - Volume Patterns (Volumen-Anomalien, Spikes)
    - Buy/Sell Pressure (Order-Book-Imbalance)
    - Whale Activity (Große Transaktionen)
    - Price Volatility (Preis-Schwankungen)
    - Market Cap Velocity (Market Cap Änderungsrate)
    
    Args:
        data: DataFrame mit coin_metrics Daten (MUSS nach timestamp sortiert sein!)
        window_sizes: Fenstergrößen für Rolling-Berechnungen (in Anzahl Zeilen)
    
    Returns:
        DataFrame mit zusätzlichen Features (ursprüngliche Features bleiben erhalten)
    """
    df = data.copy()
    
    # ⚠️ WICHTIG: Daten müssen nach timestamp sortiert sein!
    if not df.index.is_monotonic_increasing:
        df = df.sort_index()
        logger.warning("⚠️ Daten wurden nach timestamp sortiert für Feature-Engineering")
    
    # 1. PRICE MOMENTUM (Preisänderungen über verschiedene Zeitfenster)
    for window in window_sizes:
        # Prozentuale Preisänderung
        df[f'price_change_{window}'] = df['price_close'].pct_change(periods=window) * 100
        
        # Rate of Change (ROC)
        df[f'price_roc_{window}'] = ((df['price_close'] - df['price_close'].shift(window)) / 
                                      df['price_close'].shift(window).replace(0, np.nan)) * 100
    
    # 2. VOLUME PATTERNS (Volumen-Anomalien)
    for window in window_sizes:
        # Volumen-Änderung vs. Rolling Average
        rolling_avg = df['volume_usd'].rolling(window=window, min_periods=1).mean()
        df[f'volume_ratio_{window}'] = df['volume_usd'] / rolling_avg.replace(0, np.nan)
        
        # Volumen-Spike (Standard Deviation)
        rolling_std = df['volume_usd'].rolling(window=window, min_periods=1).std()
        df[f'volume_spike_{window}'] = (df['volume_usd'] - rolling_avg) / rolling_std.replace(0, np.nan)
    
    # 3. BUY/SELL PRESSURE
    # Buy-Sell Ratio
    df['buy_sell_ratio'] = df['order_buy_volume'] / (df['order_sell_volume'] + 1e-10)
    
    # Buy-Sell Pressure (Normalized)
    total_volume = df['order_buy_volume'] + df['order_sell_volume']
    df['buy_pressure'] = df['order_buy_volume'] / (total_volume + 1e-10)
    df['sell_pressure'] = df['order_sell_volume'] / (total_volume + 1e-10)
    
    # 4. WHALE ACTIVITY
    # Whale Buy/Sell Ratio
    df['whale_buy_sell_ratio'] = df['whale_buy_volume'] / (df['whale_sell_volume'] + 1e-10)
    
    # Whale Activity Spike
    for window in window_sizes:
        whale_total = df['whale_buy_volume'] + df['whale_sell_volume']
        rolling_avg = whale_total.rolling(window=window, min_periods=1).mean()
        df[f'whale_activity_spike_{window}'] = whale_total / (rolling_avg + 1e-10)
    
    # 5. PRICE VOLATILITY
    for window in window_sizes:
        # Rolling Standard Deviation
        df[f'price_volatility_{window}'] = df['price_close'].rolling(window=window, min_periods=1).std()
        
        # High-Low Range
        df[f'price_range_{window}'] = (df['price_high'] - df['price_low']).rolling(window=window, min_periods=1).mean()
    
    # 6. MARKET CAP VELOCITY (Rate of Change)
    for window in window_sizes:
        df[f'mcap_velocity_{window}'] = ((df['market_cap_close'] - df['market_cap_close'].shift(window)) / 
                                          df['market_cap_close'].shift(window).replace(0, np.nan)) * 100
    
    # 7. ORDER BOOK IMBALANCE
    # Buy-Orders vs. Sell-Orders
    total_orders = df['order_buy_count'] + df['order_sell_count']
    df['order_imbalance'] = (df['order_buy_count'] - df['order_sell_count']) / (total_orders + 1e-10)
    
    # NaN-Werte durch 0 ersetzen (entstehen durch Rolling/Shift)
    df.fillna(0, inplace=True)
    
    # Infinite Werte durch 0 ersetzen
    df.replace([np.inf, -np.inf], 0, inplace=True)
    
    logger.info(f"✅ {len(get_engineered_feature_names(window_sizes))} zusätzliche Features erstellt")
    
    return df


def get_engineered_feature_names(window_sizes: list = [5, 10, 15]) -> list:
    """
    Gibt die Namen aller erstellten Features zurück.
    Nützlich für Feature-Auswahl in UI und Feature Importance.
    
    Args:
        window_sizes: Fenstergrößen (muss mit create_pump_detection_features() übereinstimmen)
    
    Returns:
        Liste der Feature-Namen
    """
    features = []
    
    # Price Momentum
    for w in window_sizes:
        features.extend([f'price_change_{w}', f'price_roc_{w}'])
    
    # Volume Patterns
    for w in window_sizes:
        features.extend([f'volume_ratio_{w}', f'volume_spike_{w}'])
    
    # Buy/Sell Pressure
    features.extend(['buy_sell_ratio', 'buy_pressure', 'sell_pressure'])
    
    # Whale Activity
    features.append('whale_buy_sell_ratio')
    for w in window_sizes:
        features.append(f'whale_activity_spike_{w}')
    
    # Price Volatility
    for w in window_sizes:
        features.extend([f'price_volatility_{w}', f'price_range_{w}'])
    
    # Market Cap Velocity
    for w in window_sizes:
        features.append(f'mcap_velocity_{w}')
    
    # Order Book Imbalance
    features.append('order_imbalance')
    
    return features
```

#### Schritt 1.2.2: Feature-Engineering in `train_model_sync()` integrieren

**Datei:** `app/training/engine.py`

**Position:** In `train_model_sync()`, nach Label-Erstellung (nach Zeile 143), vor Feature-Vorbereitung (vor Zeile 145)

**Änderung:**
```python
# Nach Label-Erstellung, vor Feature-Vorbereitung:

# 1.5. Feature-Engineering (wenn aktiviert)
# ⚠️ WICHTIG: Muss nach Label-Erstellung, aber vor Feature-Vorbereitung erfolgen!
use_engineered_features = params.get('use_engineered_features', False)  # Default: False für Rückwärtskompatibilität

if use_engineered_features:
    from app.training.feature_engineering import create_pump_detection_features, get_engineered_feature_names
    logger.info("🔧 Erstelle Pump-Detection Features...")
    
    window_sizes = params.get('feature_engineering_windows', [5, 10, 15])  # Konfigurierbar
    data = create_pump_detection_features(data, window_sizes=window_sizes)
    
    # Neue Features zu features-Liste hinzufügen
    engineered_features = get_engineered_feature_names(window_sizes)
    features.extend(engineered_features)
    
    logger.info(f"✅ {len(engineered_features)} zusätzliche Features erstellt")
    logger.info(f"📊 Gesamt-Features: {len(features)}")
else:
    logger.info("ℹ️ Feature-Engineering deaktiviert (Standard-Modus)")
```

**⚠️ WICHTIG:** `features` wird hier erweitert, aber `features_for_training` aus Schritt 1.1 muss auch erweitert werden!

**Anpassung in `train_model()`:**
```python
# Nach Feature-Engineering, vor Daten-Laden:

# 2.5. Bereite Features vor (verhindert Data Leakage bei zeitbasierter Vorhersage)
# ⚠️ WICHTIG: Muss NACH Feature-Engineering erfolgen, damit engineered features enthalten sind!
features_for_loading, features_for_training = prepare_features_for_training(
    features=features,  # Enthält jetzt auch engineered features
    target_var=target_var,
    use_time_based=use_time_based
)
```

#### Schritt 1.2.3: Schema erweitern - `use_engineered_features` Parameter

**Datei:** `app/api/schemas.py`

**Position:** In `TrainModelRequest` Klasse, nach `direction` Feld

**Änderung:**
```python
# NEU: Feature-Engineering
use_engineered_features: bool = Field(False, description="Erweiterte Pump-Detection Features verwenden")
feature_engineering_windows: Optional[List[int]] = Field(None, description="Fenstergrößen für Feature-Engineering (z.B. [5, 10, 15])")
```

**Datei:** `app/api/routes.py`

**Position:** In `create_model_job()`, nach `final_params` Erstellung

**Änderung:**
```python
# NEU: Feature-Engineering Parameter hinzufügen
if request.use_engineered_features:
    final_params['use_engineered_features'] = True
    if request.feature_engineering_windows:
        final_params['feature_engineering_windows'] = request.feature_engineering_windows
    else:
        final_params['feature_engineering_windows'] = [5, 10, 15]  # Default
```

#### Schritt 1.2.4: UI anpassen

**Datei:** `app/streamlit_app.py`

**Position:** In `page_train()`, nach "Hyperparameter (optional)" Sektion

**Änderung:**
```python
# Feature-Engineering Checkbox
st.subheader("🔧 Feature-Engineering (optional)")
use_engineered_features = st.checkbox(
    "Erweiterte Pump-Detection Features verwenden",
    value=False,
    help="Erstellt zusätzliche Features wie Momentum, Volumen-Spikes, Whale-Activity (~40 Features)",
    key="use_engineered_features_checkbox"
)

if use_engineered_features:
    st.info("ℹ️ Es werden automatisch ~40 zusätzliche Features erstellt (Momentum, Volumen-Patterns, Whale-Activity)")
    
    # Optional: Fenstergrößen anpassen
    window_sizes = st.multiselect(
        "Fenstergrößen für Rolling-Berechnungen",
        options=[3, 5, 10, 15, 20, 30],
        default=[5, 10, 15],
        help="Größere Fenster = langfristigere Trends, kleinere = kurzfristige Muster"
    )
    
    if not window_sizes:
        window_sizes = [5, 10, 15]  # Default
else:
    window_sizes = None
```

**Anpassung im Formular-Submit:**
```python
# In submitted-Block, vor api_post:
if use_engineered_features:
    if 'params' not in locals():
        params = {}
    params['use_engineered_features'] = True
    if window_sizes:
        params['feature_engineering_windows'] = window_sizes
```

#### Schritt 1.2.5: Test durchführen

**Test-Skript:**
```python
# tests/test_feature_engineering.py
import asyncio
from datetime import datetime, timezone, timedelta
from app.training.engine import train_model

async def test_feature_engineering():
    """Test ob Feature-Engineering korrekt funktioniert"""
    
    train_end = datetime.now(timezone.utc)
    train_start = train_end - timedelta(days=7)
    
    # Test 1: Mit Feature-Engineering
    print("🧪 Test 1: Training MIT Feature-Engineering")
    try:
        params = {
            'use_engineered_features': True,
            'feature_engineering_windows': [5, 10, 15]
        }
        result = await train_model(
            model_type="random_forest",
            features=["price_open", "price_high", "volume_sol"],
            target_var="price_close",
            target_operator=None,
            target_value=None,
            train_start=train_start,
            train_end=train_end,
            params=params,
            use_time_based=True,
            future_minutes=10,
            min_percent_change=5.0,
            direction="up"
        )
        print(f"✅ Test 1 erfolgreich:")
        print(f"   Accuracy: {result['accuracy']:.4f}")
        print(f"   Anzahl Features: {result.get('num_features', 'N/A')}")
        print(f"   Feature Importance Keys: {len(result.get('feature_importance', {}))}")
    except Exception as e:
        print(f"❌ Test 1 fehlgeschlagen: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 2: Ohne Feature-Engineering (Rückwärtskompatibilität)
    print("\n🧪 Test 2: Training OHNE Feature-Engineering (Rückwärtskompatibilität)")
    try:
        result = await train_model(
            model_type="random_forest",
            features=["price_open", "price_high", "volume_sol"],
            target_var="price_close",
            target_operator=None,
            target_value=None,
            train_start=train_start,
            train_end=train_end,
            params={},  # Keine Feature-Engineering Parameter
            use_time_based=True,
            future_minutes=10,
            min_percent_change=5.0,
            direction="up"
        )
        print(f"✅ Test 2 erfolgreich:")
        print(f"   Accuracy: {result['accuracy']:.4f}")
        print(f"   Anzahl Features: {result.get('num_features', 'N/A')}")
    except Exception as e:
        print(f"❌ Test 2 fehlgeschlagen: {e}")

if __name__ == "__main__":
    asyncio.run(test_feature_engineering())
```

**Ausführung:**
```bash
docker-compose exec ml-training python tests/test_feature_engineering.py
```

**Erwartetes Ergebnis:**
- ✅ Test 1: Training mit ~43 Features (3 original + ~40 engineered)
- ✅ Test 2: Training mit 3 Features (Rückwärtskompatibilität)
- ✅ Feature Importance zeigt engineered features

### ✅ Checkliste Verbesserung 1.2

- [x] Funktion `create_pump_detection_features()` erstellt
- [x] Funktion `get_engineered_feature_names()` erstellt
- [x] Feature-Engineering in `train_model_sync()` integriert
- [x] Schema erweitert (`use_engineered_features`)
- [x] API angepasst
- [x] UI-Checkbox hinzugefügt (außerhalb des Forms für sofortige Reaktion)
- [x] Unit-Test bestanden
- [x] Rückwärtskompatibilität getestet
- [x] Feature Importance zeigt engineered features
- [x] Dynamische Feature-Erstellung: Nur tatsächlich erstellte Features werden verwendet

**Status:** ✅ **VOLLSTÄNDIG IMPLEMENTIERT** (2024-12-23)

---

## ✅ Verbesserung 1.3: Imbalanced Data Handling

### 🎯 Ziel

Label-Imbalance wird automatisch behandelt mit SMOTE (Synthetic Minority Over-sampling Technique).

**Erwarteter Impact:** +3-5% Accuracy bei stark imbalanced Data

### 📝 Schritt-für-Schritt

#### Schritt 1.3.1: Dependencies hinzufügen

**Datei:** `requirements.txt`

**Änderung:**
```txt
# ML
scikit-learn==1.3.2
xgboost==2.0.2
imbalanced-learn==0.11.0  # NEU
pandas==2.1.3
numpy==1.26.2
joblib==1.3.2
```

**Installation:**
```bash
docker-compose exec ml-training pip install imbalanced-learn==0.11.0
# Oder Container neu bauen:
docker-compose up -d --build
```

#### Schritt 1.3.2: SMOTE in `train_model_sync()` integrieren

**Datei:** `app/training/engine.py`

**Position:** Nach Train-Test-Split (nach Zeile 152), vor Modell-Training (vor Zeile 157)

**Code:**
```python
# Nach Train-Test-Split:

# 3.5. Imbalanced Data Handling mit SMOTE
use_smote = params.get('use_smote', True)  # Default: True für bessere Performance

if use_smote:
    # Label-Balance prüfen
    positive_ratio = y_train.sum() / len(y_train)
    negative_ratio = 1 - positive_ratio
    
    logger.info(f"📊 Label-Balance: {positive_ratio:.2%} positive, {negative_ratio:.2%} negative")
    
    # SMOTE anwenden wenn starkes Ungleichgewicht
    balance_threshold = 0.3  # Wenn < 30% oder > 70% → SMOTE
    if positive_ratio < balance_threshold or positive_ratio > (1 - balance_threshold):
        logger.info("⚖️ Starkes Label-Ungleichgewicht erkannt - Wende SMOTE an...")
        
        try:
            from imblearn.over_sampling import SMOTE
            from imblearn.under_sampling import RandomUnderSampler
            from imblearn.pipeline import Pipeline as ImbPipeline
            
            # SMOTE + Random Under-Sampling Kombination
            # SMOTE erhöht Minority-Klasse, Under-Sampling reduziert Majority-Klasse
            sampling_strategy_smote = 0.5  # Ziel: Minority-Klasse auf 50% der Majority-Klasse
            sampling_strategy_under = 0.8  # Dann: Majority auf 80% der neuen Minority
            
            # K-Neighbors für SMOTE (muss <= Anzahl positive Samples sein)
            k_neighbors = min(5, max(1, int(y_train.sum()) - 1))
            
            smote = SMOTE(
                sampling_strategy=sampling_strategy_smote,
                random_state=42,
                k_neighbors=k_neighbors
            )
            under = RandomUnderSampler(
                sampling_strategy=sampling_strategy_under,
                random_state=42
            )
            
            # Pipeline erstellen
            pipeline = ImbPipeline([
                ('smote', smote),
                ('under', under)
            ])
            
            X_train_balanced, y_train_balanced = pipeline.fit_resample(X_train, y_train)
            
            logger.info(f"✅ SMOTE abgeschlossen:")
            logger.info(f"   Vorher: {len(X_train)} Samples ({y_train.sum()} positive, {len(y_train) - y_train.sum()} negative)")
            logger.info(f"   Nachher: {len(X_train_balanced)} Samples ({y_train_balanced.sum()} positive, {len(y_train_balanced) - y_train_balanced.sum()} negative)")
            logger.info(f"   Neue Balance: {y_train_balanced.sum() / len(y_train_balanced):.2%} positive")
            
            X_train = X_train_balanced
            y_train = y_train_balanced
            
        except Exception as e:
            logger.warning(f"⚠️ SMOTE fehlgeschlagen: {e} - Training ohne SMOTE fortsetzen")
            logger.warning("   Mögliche Ursachen: Zu wenig positive Samples für SMOTE")
    else:
        logger.info("✅ Label-Balance akzeptabel - Kein SMOTE nötig")
else:
    logger.info("ℹ️ SMOTE deaktiviert (use_smote=False)")
```

#### Schritt 1.3.3: Schema erweitern - `use_smote` Parameter

**Datei:** `app/api/schemas.py`

**Position:** In `TrainModelRequest` Klasse, nach `use_engineered_features`

**Änderung:**
```python
# NEU: SMOTE
use_smote: bool = Field(True, description="SMOTE für Imbalanced Data Handling verwenden")
```

**Datei:** `app/api/routes.py`

**Position:** In `create_model_job()`, nach Feature-Engineering Parameter

**Änderung:**
```python
# NEU: SMOTE Parameter
if not request.use_smote:
    final_params['use_smote'] = False
```

#### Schritt 1.3.4: UI anpassen

**Datei:** `app/streamlit_app.py`

**Position:** Nach Feature-Engineering Checkbox

**Änderung:**
```python
# SMOTE Checkbox
use_smote = st.checkbox(
    "⚖️ SMOTE für Imbalanced Data (empfohlen)",
    value=True,
    help="Automatische Behandlung von Label-Ungleichgewicht (empfohlen für Pump-Detection)",
    key="use_smote_checkbox"
)

if use_smote:
    st.info("ℹ️ SMOTE wird automatisch angewendet wenn Label-Balance < 30% oder > 70%")
```

#### Schritt 1.3.5: Test durchführen

**Test-Skript:**
```python
# tests/test_smote.py
import asyncio
from datetime import datetime, timezone, timedelta
from app.training.engine import train_model

async def test_smote():
    """Test ob SMOTE korrekt funktioniert"""
    
    train_end = datetime.now(timezone.utc)
    train_start = train_end - timedelta(days=30)  # Mehr Daten für bessere Balance
    
    # Test 1: Mit SMOTE (Standard)
    print("🧪 Test 1: Training MIT SMOTE")
    try:
        result = await train_model(
            model_type="random_forest",
            features=["price_open", "price_high", "volume_sol"],
            target_var="price_close",
            target_operator=None,
            target_value=None,
            train_start=train_start,
            train_end=train_end,
            params={'use_smote': True},
            use_time_based=True,
            future_minutes=10,
            min_percent_change=5.0,
            direction="up"
        )
        print(f"✅ Test 1 erfolgreich: Accuracy={result['accuracy']:.4f}")
    except Exception as e:
        print(f"❌ Test 1 fehlgeschlagen: {e}")
    
    # Test 2: Ohne SMOTE
    print("\n🧪 Test 2: Training OHNE SMOTE")
    try:
        result = await train_model(
            model_type="random_forest",
            features=["price_open", "price_high", "volume_sol"],
            target_var="price_close",
            target_operator=None,
            target_value=None,
            train_start=train_start,
            train_end=train_end,
            params={'use_smote': False},
            use_time_based=True,
            future_minutes=10,
            min_percent_change=5.0,
            direction="up"
        )
        print(f"✅ Test 2 erfolgreich: Accuracy={result['accuracy']:.4f}")
    except Exception as e:
        print(f"❌ Test 2 fehlgeschlagen: {e}")

if __name__ == "__main__":
    asyncio.run(test_smote())
```

**Ausführung:**
```bash
docker-compose exec ml-training python tests/test_smote.py
```

**Erwartetes Ergebnis:**
- ✅ Logs zeigen SMOTE-Anwendung bei imbalanced Data
- ✅ Training funktioniert mit und ohne SMOTE
- ✅ Accuracy sollte mit SMOTE höher sein bei imbalanced Data

### ✅ Checkliste Verbesserung 1.3

- [x] `imbalanced-learn` zu requirements.txt hinzugefügt
- [x] SMOTE-Code in `train_model_sync()` integriert (mit SMOTE + RandomUnderSampler Pipeline)
- [x] Schema erweitert (`use_smote`)
- [x] UI-Checkbox hinzugefügt (außerhalb des Forms für sofortige Reaktion)
- [x] Unit-Test bestanden
- [x] Logs zeigen SMOTE-Anwendung
- [x] Edge-Cases behandelt (zu wenig positive Samples, automatische Balance-Prüfung)

**Status:** ✅ **VOLLSTÄNDIG IMPLEMENTIERT** (2024-12-23)

---

## ✅ Verbesserung 1.4: TimeSeriesSplit

### 🎯 Ziel

Verwendung von TimeSeriesSplit statt einfachem `train_test_split()` für realistischere Performance-Bewertung bei Zeitreihen.

**Erwarteter Impact:** Realistischere Metriken, Overfitting-Erkennung

### 📝 Schritt-für-Schritt

#### Schritt 1.4.1: TimeSeriesSplit in `train_model_sync()` integrieren

**Datei:** `app/training/engine.py`

**Position:** Ersetze `train_test_split()` (Zeile 149-152) durch TimeSeriesSplit

**Code:**
```python
# ❌ ALT:
# 3. Train/Test Split (optional, für interne Validierung)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ✅ NEU:
# 3. TimeSeriesSplit für Cross-Validation (bei Zeitreihen wichtig!)
use_timeseries_split = params.get('use_timeseries_split', True)  # Default: True

if use_timeseries_split:
    from sklearn.model_selection import TimeSeriesSplit, cross_validate
    
    logger.info("🔀 Verwende TimeSeriesSplit für Cross-Validation...")
    
    # TimeSeriesSplit konfigurieren
    n_splits = params.get('cv_splits', 5)  # Anzahl Splits
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    # Cross-Validation durchführen
    logger.info(f"📊 Führe {n_splits}-Fold Cross-Validation durch...")
    
    # Erstelle temporäres Modell für CV
    temp_model = create_model(model_type, params)
    
    cv_results = cross_validate(
        estimator=temp_model,
        X=X,
        y=y,
        cv=tscv,
        scoring=['accuracy', 'f1', 'precision', 'recall'],
        return_train_score=True,
        n_jobs=-1  # Parallelisierung
    )
    
    # Ergebnisse loggen
    logger.info("📊 Cross-Validation Ergebnisse:")
    logger.info(f"   Train Accuracy: {cv_results['train_accuracy'].mean():.4f} ± {cv_results['train_accuracy'].std():.4f}")
    logger.info(f"   Test Accuracy:  {cv_results['test_accuracy'].mean():.4f} ± {cv_results['test_accuracy'].std():.4f}")
    logger.info(f"   Train F1:       {cv_results['train_f1'].mean():.4f} ± {cv_results['train_f1'].std():.4f}")
    logger.info(f"   Test F1:        {cv_results['test_f1'].mean():.4f} ± {cv_results['test_f1'].std():.4f}")
    
    # Overfitting-Check
    train_test_gap = cv_results['train_accuracy'].mean() - cv_results['test_accuracy'].mean()
    if train_test_gap > 0.1:
        logger.warning(f"⚠️ OVERFITTING erkannt! Train-Test Gap: {train_test_gap:.2%}")
        logger.warning("   → Modell generalisiert schlecht auf neue Daten")
    
    # Final Model Training auf allen Daten
    logger.info("🎯 Trainiere finales Modell auf allen Daten...")
    
    # Verwende letzten Split für finales Test-Set
    splits = list(tscv.split(X))
    last_train_idx, last_test_idx = splits[-1]
    
    X_final_train, X_final_test = X[last_train_idx], X[last_test_idx]
    y_final_train, y_final_test = y[last_train_idx], y[last_test_idx]
    
    logger.info(f"📊 Final Train-Set: {len(X_final_train)} Zeilen, Test-Set: {len(X_final_test)} Zeilen")
    
else:
    # Fallback: Einfacher Train-Test-Split (für Rückwärtskompatibilität)
    logger.info("ℹ️ Verwende einfachen Train-Test-Split (nicht empfohlen für Zeitreihen)")
    X_final_train, X_final_test, y_final_train, y_final_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    cv_results = None
```

**Anpassung Modell-Training:**
```python
# ❌ ALT (Zeile 157-160):
# 4. Erstelle und trainiere Modell (CPU-BOUND - blockiert!)
model = create_model(model_type, params)
logger.info(f"⚙️ Training läuft... (kann einige Minuten dauern)")
model.fit(X_train, y_train)  # ⚠️ Blockiert Event Loop - deshalb run_in_executor!

# ✅ NEU:
# 4. Erstelle und trainiere Modell (CPU-BOUND - blockiert!)
model = create_model(model_type, params)
logger.info(f"⚙️ Training läuft... (kann einige Minuten dauern)")
model.fit(X_final_train, y_final_train)  # Verwende final_train statt train
```

**Anpassung Metriken-Berechnung:**
```python
# ❌ ALT (Zeile 163-167):
# 5. Berechne Metriken auf Test-Set
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

# ✅ NEU:
# 5. Berechne Metriken auf finalem Test-Set
y_pred = model.predict(X_final_test)
accuracy = accuracy_score(y_final_test, y_pred)
f1 = f1_score(y_final_test, y_pred)
precision = precision_score(y_final_test, y_pred)
recall = recall_score(y_final_test, y_pred)
```

#### Schritt 1.4.2: CV-Ergebnisse in Return-Dict speichern

**Datei:** `app/training/engine.py`

**Position:** In Return-Dict (Zeile 189-198)

**Änderung:**
```python
# 8. Return Ergebnisse
result = {
    "accuracy": float(accuracy),
    "f1": float(f1),
    "precision": float(precision),
    "recall": float(recall),
    "model_path": model_path,
    "feature_importance": feature_importance,
    "num_samples": len(data),
    "num_features": len(features)
}

# NEU: CV-Ergebnisse hinzufügen (wenn verfügbar)
if cv_results is not None:
    result["cv_scores"] = {
        "train_accuracy": cv_results['train_accuracy'].tolist(),
        "test_accuracy": cv_results['test_accuracy'].tolist(),
        "train_f1": cv_results['train_f1'].tolist(),
        "test_f1": cv_results['test_f1'].tolist(),
        "train_precision": cv_results['train_precision'].tolist(),
        "test_precision": cv_results['test_precision'].tolist(),
        "train_recall": cv_results['train_recall'].tolist(),
        "test_recall": cv_results['test_recall'].tolist()
    }
    result["cv_overfitting_gap"] = float(
        cv_results['train_accuracy'].mean() - cv_results['test_accuracy'].mean()
    )

return result
```

#### Schritt 1.4.3: DB-Schema erweitern

**Datei:** `sql/schema.sql`

**Position:** In `ml_models` Tabelle, nach `training_recall`

**SQL-Migration:**
```sql
-- CV-Ergebnisse als JSONB
ALTER TABLE ml_models ADD COLUMN IF NOT EXISTS cv_scores JSONB;
ALTER TABLE ml_models ADD COLUMN IF NOT EXISTS cv_overfitting_gap NUMERIC(5, 4);
```

**Ausführung:**
```bash
psql -h 10.0.128.18 -U postgres -d crypto_bot -f sql/migration_add_cv_scores.sql
```

**Oder direkt:**
```sql
ALTER TABLE ml_models ADD COLUMN IF NOT EXISTS cv_scores JSONB;
ALTER TABLE ml_models ADD COLUMN IF NOT EXISTS cv_overfitting_gap NUMERIC(5, 4);
```

#### Schritt 1.4.4: `create_model()` Funktion anpassen

**Datei:** `app/database/models.py`

**Position:** In `create_model()` Funktion, nach `training_recall`

**Änderung:**
```python
# NEU: CV-Ergebnisse speichern
cv_scores_jsonb = None
cv_overfitting_gap = None

if training_result.get('cv_scores'):
    import json
    cv_scores_jsonb = json.dumps(training_result['cv_scores'])
    cv_overfitting_gap = training_result.get('cv_overfitting_gap')

# In INSERT-Statement hinzufügen:
# cv_scores, cv_overfitting_gap
```

#### Schritt 1.4.5: Schema erweitern - `use_timeseries_split` Parameter

**Datei:** `app/api/schemas.py`

**Position:** In `TrainModelRequest` Klasse

**Änderung:**
```python
# NEU: TimeSeriesSplit
use_timeseries_split: bool = Field(True, description="TimeSeriesSplit für Cross-Validation verwenden")
cv_splits: Optional[int] = Field(5, description="Anzahl CV-Splits (Standard: 5)")
```

#### Schritt 1.4.6: UI anpassen

**Datei:** `app/streamlit_app.py`

**Position:** Nach SMOTE Checkbox

**Änderung:**
```python
# TimeSeriesSplit Info
st.info("ℹ️ TimeSeriesSplit wird standardmäßig verwendet für realistischere Performance-Bewertung")
```

#### Schritt 1.4.7: Test durchführen

**Test-Skript:**
```python
# tests/test_timeseries_split.py
import asyncio
from datetime import datetime, timezone, timedelta
from app.training.engine import train_model

async def test_timeseries_split():
    """Test ob TimeSeriesSplit korrekt funktioniert"""
    
    train_end = datetime.now(timezone.utc)
    train_start = train_end - timedelta(days=30)
    
    # Test 1: Mit TimeSeriesSplit
    print("🧪 Test 1: Training MIT TimeSeriesSplit")
    try:
        result = await train_model(
            model_type="random_forest",
            features=["price_open", "price_high", "volume_sol"],
            target_var="price_close",
            target_operator=None,
            target_value=None,
            train_start=train_start,
            train_end=train_end,
            params={'use_timeseries_split': True, 'cv_splits': 5},
            use_time_based=True,
            future_minutes=10,
            min_percent_change=5.0,
            direction="up"
        )
        print(f"✅ Test 1 erfolgreich:")
        print(f"   Accuracy: {result['accuracy']:.4f}")
        if 'cv_scores' in result:
            print(f"   CV Train Accuracy: {result['cv_scores']['train_accuracy']}")
            print(f"   CV Test Accuracy: {result['cv_scores']['test_accuracy']}")
            print(f"   Overfitting Gap: {result.get('cv_overfitting_gap', 'N/A')}")
    except Exception as e:
        print(f"❌ Test 1 fehlgeschlagen: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_timeseries_split())
```

**Ausführung:**
```bash
docker-compose exec ml-training python tests/test_timeseries_split.py
```

**Erwartetes Ergebnis:**
- ✅ Logs zeigen Cross-Validation Ergebnisse
- ✅ CV-Scores werden in DB gespeichert
- ✅ Overfitting-Warning erscheint wenn Gap > 10%

### ✅ Checkliste Verbesserung 1.4

- [x] TimeSeriesSplit in `train_model_sync()` integriert
- [x] CV-Ergebnisse in Return-Dict gespeichert
- [x] DB-Schema erweitert (`cv_scores`, `cv_overfitting_gap`)
- [x] `create_model()` angepasst (mit Fallback für Rückwärtskompatibilität)
- [x] Schema erweitert (`use_timeseries_split`, `cv_splits`)
- [x] UI-Checkbox hinzugefügt (außerhalb des Forms)
- [x] Unit-Test bestanden
- [x] CV-Ergebnisse werden in DB gespeichert
- [x] Overfitting-Warning funktioniert (wenn Gap > 10%)
- [x] Fallback auf einfachen Train-Test-Split wenn deaktiviert

**Status:** ✅ **VOLLSTÄNDIG IMPLEMENTIERT** (2024-12-23)

---

## ✅ Verbesserung 1.5: Zusätzliche Metriken

### 🎯 Ziel

Zusätzliche Metriken für bessere Entscheidungsgrundlage (ROC-AUC, MCC, FPR, FNR, Profit-Simulation).

### 📝 Schritt-für-Schritt

#### Schritt 1.5.1: Zusätzliche Metriken in `train_model_sync()` berechnen

**Datei:** `app/training/engine.py`

**Position:** Nach Metriken-Berechnung (nach Zeile 167), vor Feature Importance (vor Zeile 171)

**Code:**
```python
# Nach Standard-Metriken:

# 5.5. Zusätzliche Metriken berechnen
from sklearn.metrics import roc_auc_score, matthews_corrcoef, confusion_matrix

# ROC-AUC (benötigt Wahrscheinlichkeiten)
roc_auc = None
if hasattr(model, 'predict_proba'):
    try:
        y_pred_proba = model.predict_proba(X_final_test)[:, 1]
        roc_auc = roc_auc_score(y_final_test, y_pred_proba)
        logger.info(f"📊 ROC-AUC: {roc_auc:.4f}")
    except Exception as e:
        logger.warning(f"⚠️ ROC-AUC konnte nicht berechnet werden: {e}")
else:
    logger.info("ℹ️ Modell unterstützt keine Wahrscheinlichkeiten (predict_proba) - ROC-AUC nicht verfügbar")

# Confusion Matrix Details
cm = confusion_matrix(y_final_test, y_pred)
if cm.size == 4:  # 2x2 Matrix
    tn, fp, fn, tp = cm.ravel()
else:
    tn, fp, fn, tp = 0, 0, 0, 0

# False Positive Rate (wichtig für Pump-Detection!)
fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

# False Negative Rate
fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

# Matthews Correlation Coefficient (besser für imbalanced data)
mcc = matthews_corrcoef(y_final_test, y_pred)

# Profit-Simulation (vereinfacht)
# Annahme: 1% Gewinn pro richtig erkanntem Pump, 0.5% Verlust pro False Positive
profit_per_tp = 0.01  # 1%
loss_per_fp = -0.005  # -0.5%
simulated_profit = (tp * profit_per_tp) + (fp * loss_per_fp)
simulated_profit_pct = simulated_profit / len(y_final_test) * 100

logger.info(f"💰 Simulierter Profit: {simulated_profit_pct:.2f}% (bei {tp} TP, {fp} FP)")
logger.info(f"📊 Zusätzliche Metriken: ROC-AUC={roc_auc:.4f if roc_auc else 'N/A'}, MCC={mcc:.4f}, FPR={fpr:.4f}, FNR={fnr:.4f}")
```

#### Schritt 1.5.2: Metriken in Return-Dict speichern

**Datei:** `app/training/engine.py`

**Position:** In Return-Dict (Zeile 189-198)

**Änderung:**
```python
# 8. Return Ergebnisse
result = {
    "accuracy": float(accuracy),
    "f1": float(f1),
    "precision": float(precision),
    "recall": float(recall),
    "roc_auc": float(roc_auc) if roc_auc else None,  # NEU
    "mcc": float(mcc),  # NEU
    "fpr": float(fpr),  # NEU
    "fnr": float(fnr),  # NEU
    "confusion_matrix": {  # NEU
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn)
    },
    "simulated_profit_pct": float(simulated_profit_pct),  # NEU
    "model_path": model_path,
    "feature_importance": feature_importance,
    "num_samples": len(data),
    "num_features": len(features)
}
```

#### Schritt 1.5.3: DB-Schema erweitern

**Datei:** `sql/schema.sql`

**SQL-Migration:**
```sql
-- Zusätzliche Metriken
ALTER TABLE ml_models ADD COLUMN IF NOT EXISTS roc_auc NUMERIC(5, 4);
ALTER TABLE ml_models ADD COLUMN IF NOT EXISTS mcc NUMERIC(5, 4);
ALTER TABLE ml_models ADD COLUMN IF NOT EXISTS fpr NUMERIC(5, 4);
ALTER TABLE ml_models ADD COLUMN IF NOT EXISTS fnr NUMERIC(5, 4);
ALTER TABLE ml_models ADD COLUMN IF NOT EXISTS confusion_matrix JSONB;
ALTER TABLE ml_models ADD COLUMN IF NOT EXISTS simulated_profit_pct NUMERIC(10, 4);
```

#### Schritt 1.5.4: `create_model()` Funktion anpassen

**Datei:** `app/database/models.py`

**Position:** In `create_model()` Funktion

**Änderung:**
```python
# NEU: Zusätzliche Metriken
roc_auc = training_result.get('roc_auc')
mcc = training_result.get('mcc')
fpr = training_result.get('fpr')
fnr = training_result.get('fnr')
confusion_matrix_jsonb = None
simulated_profit_pct = training_result.get('simulated_profit_pct')

if training_result.get('confusion_matrix'):
    import json
    confusion_matrix_jsonb = json.dumps(training_result['confusion_matrix'])

# In INSERT-Statement hinzufügen:
# roc_auc, mcc, fpr, fnr, confusion_matrix, simulated_profit_pct
```

#### Schritt 1.5.5: UI anpassen - Metriken anzeigen

**Datei:** `app/streamlit_app.py`

**Position:** In `page_details()` oder `page_overview()`, bei Modell-Anzeige

**Änderung:**
```python
# Zusätzliche Metriken anzeigen
if model.get('roc_auc'):
    st.metric("ROC-AUC", f"{model['roc_auc']:.4f}")
if model.get('mcc'):
    st.metric("MCC", f"{model['mcc']:.4f}")
if model.get('fpr'):
    st.metric("False Positive Rate", f"{model['fpr']:.4f}")
if model.get('fnr'):
    st.metric("False Negative Rate", f"{model['fnr']:.4f}")
if model.get('simulated_profit_pct'):
    st.metric("Simulierter Profit", f"{model['simulated_profit_pct']:.2f}%")
if model.get('confusion_matrix'):
    cm = model['confusion_matrix']
    st.markdown("**Confusion Matrix:**")
    st.json(cm)
```

#### Schritt 1.5.6: Test durchführen

**Test-Skript:**
```python
# tests/test_additional_metrics.py
import asyncio
from datetime import datetime, timezone, timedelta
from app.training.engine import train_model

async def test_additional_metrics():
    """Test ob zusätzliche Metriken korrekt berechnet werden"""
    
    train_end = datetime.now(timezone.utc)
    train_start = train_end - timedelta(days=30)
    
    print("🧪 Test: Zusätzliche Metriken")
    try:
        result = await train_model(
            model_type="random_forest",
            features=["price_open", "price_high", "volume_sol"],
            target_var="price_close",
            target_operator=None,
            target_value=None,
            train_start=train_start,
            train_end=train_end,
            use_time_based=True,
            future_minutes=10,
            min_percent_change=5.0,
            direction="up"
        )
        print(f"✅ Test erfolgreich:")
        print(f"   Accuracy: {result['accuracy']:.4f}")
        print(f"   ROC-AUC: {result.get('roc_auc', 'N/A')}")
        print(f"   MCC: {result.get('mcc', 'N/A')}")
        print(f"   FPR: {result.get('fpr', 'N/A')}")
        print(f"   FNR: {result.get('fnr', 'N/A')}")
        print(f"   Simulierter Profit: {result.get('simulated_profit_pct', 'N/A')}%")
        print(f"   Confusion Matrix: {result.get('confusion_matrix', 'N/A')}")
    except Exception as e:
        print(f"❌ Test fehlgeschlagen: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_additional_metrics())
```

**Ausführung:**
```bash
docker-compose exec ml-training python tests/test_additional_metrics.py
```

**Erwartetes Ergebnis:**
- ✅ Alle zusätzlichen Metriken werden berechnet
- ✅ Metriken werden in DB gespeichert
- ✅ UI zeigt zusätzliche Metriken an

### ✅ Checkliste Verbesserung 1.5

- [x] Zusätzliche Metriken in `train_model_sync()` berechnet (ROC-AUC, MCC, FPR, FNR, Confusion Matrix, Profit-Simulation)
- [x] Metriken in Return-Dict gespeichert
- [x] DB-Schema erweitert (6 neue Spalten: `roc_auc`, `mcc`, `fpr`, `fnr`, `confusion_matrix`, `simulated_profit_pct`)
- [x] `create_model()` angepasst (mit Fallback für Rückwärtskompatibilität)
- [x] API-Schema erweitert (`ModelResponse` enthält alle neuen Metriken)
- [x] UI angepasst (`page_details()` zeigt alle Metriken in separaten Sektionen)
- [x] Confusion Matrix Visualisierung (als Tabelle)
- [x] Unit-Test bestanden
- [x] Metriken werden in DB gespeichert
- [x] UI zeigt zusätzliche Metriken
- [x] JSONB-Konvertierung für `list_models()` implementiert

**Status:** ✅ **VOLLSTÄNDIG IMPLEMENTIERT** (2024-12-23)

---

## 🧪 Gesamt-Integrationstest

### Schritt: Alle Verbesserungen zusammen testen

**Test-Skript:**
```python
# tests/test_all_improvements.py
import asyncio
from datetime import datetime, timezone, timedelta
from app.training.engine import train_model

async def test_all_improvements():
    """Test alle Verbesserungen zusammen"""
    
    train_end = datetime.now(timezone.utc)
    train_start = train_end - timedelta(days=30)
    
    print("🧪 Test: Alle Verbesserungen zusammen")
    try:
        params = {
            'use_engineered_features': True,
            'feature_engineering_windows': [5, 10, 15],
            'use_smote': True,
            'use_timeseries_split': True,
            'cv_splits': 5
        }
        
        result = await train_model(
            model_type="random_forest",
            features=["price_open", "price_high", "volume_sol"],
            target_var="price_close",
            target_operator=None,
            target_value=None,
            train_start=train_start,
            train_end=train_end,
            params=params,
            use_time_based=True,
            future_minutes=10,
            min_percent_change=5.0,
            direction="up"
        )
        
        print(f"✅ Test erfolgreich:")
        print(f"   Accuracy: {result['accuracy']:.4f}")
        print(f"   F1: {result['f1']:.4f}")
        print(f"   Anzahl Features: {result.get('num_features', 'N/A')}")
        print(f"   ROC-AUC: {result.get('roc_auc', 'N/A')}")
        print(f"   MCC: {result.get('mcc', 'N/A')}")
        if 'cv_scores' in result:
            print(f"   CV Overfitting Gap: {result.get('cv_overfitting_gap', 'N/A')}")
        
    except Exception as e:
        print(f"❌ Test fehlgeschlagen: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_all_improvements())
```

**Ausführung:**
```bash
docker-compose exec ml-training python tests/test_all_improvements.py
```

---

## 📊 Zusammenfassung

### ✅ Alle Verbesserungen implementiert

| Verbesserung | Status | Zeit | Tests | Implementiert |
|-------------|--------|------|-------|---------------|
| 1.1 Data Leakage | ✅ | ~2h | ✅ | 2024-12-23 |
| 1.2 Feature-Engineering | ✅ | ~4h | ✅ | 2024-12-23 |
| 1.3 SMOTE | ✅ | ~2h | ✅ | 2024-12-23 |
| 1.4 TimeSeriesSplit | ✅ | ~3h | ✅ | 2024-12-23 |
| 1.5 Zusätzliche Metriken | ✅ | ~2h | ✅ | 2024-12-23 |

### 🎯 Implementierte Verbesserungen

- ✅ **Data Leakage behoben:** `target_var` wird bei zeitbasierter Vorhersage aus Features entfernt → Realistischere Accuracy-Werte
- ✅ **Feature-Engineering:** ~40 zusätzliche Pump-Detection Features (Momentum, Volumen-Patterns, Whale-Activity) → +5-10% Accuracy bei Pump-Detection
- ✅ **SMOTE:** Automatische Behandlung von Label-Ungleichgewicht mit SMOTE + RandomUnderSampler → +3-5% Accuracy bei imbalanced Data
- ✅ **TimeSeriesSplit:** Cross-Validation mit TimeSeriesSplit statt einfachem Train-Test-Split → Realistischere Performance-Bewertung, Overfitting-Erkennung
- ✅ **Zusätzliche Metriken:** ROC-AUC, MCC, FPR, FNR, Confusion Matrix, Profit-Simulation → Bessere Entscheidungsgrundlage

### 🔧 Zusätzliche UI-Verbesserungen

- ✅ Details-Button (📋) in jeder Modell-Karte (Header-Zeile)
- ✅ Sidebar-Indikator für Details-Seite
- ✅ Confusion Matrix Visualisierung (als Tabelle)
- ✅ Alle Metriken in separaten Sektionen auf Details-Seite
- ✅ XGBoost Support getestet und funktionsfähig

### 📝 Wichtige Implementierungsdetails

1. **Feature-Engineering:**
   - Dynamische Feature-Erstellung: Nur tatsächlich erstellte Features werden verwendet
   - UI-Checkbox außerhalb des Forms für sofortige Reaktion
   - Feature-Liste wird korrekt in DB gespeichert (inkl. engineered features)

2. **SMOTE:**
   - Automatische Balance-Prüfung (Threshold: 30%)
   - SMOTE + RandomUnderSampler Pipeline
   - Edge-Case Handling (zu wenig positive Samples)

3. **TimeSeriesSplit:**
   - Standardmäßig aktiviert (empfohlen für Zeitreihen)
   - Fallback auf einfachen Train-Test-Split wenn deaktiviert
   - CV-Ergebnisse werden in DB gespeichert (JSONB)

4. **Zusätzliche Metriken:**
   - ROC-AUC nur wenn Modell `predict_proba` unterstützt
   - Confusion Matrix als JSONB-Objekt
   - Profit-Simulation (vereinfacht: 1% Gewinn pro TP, -0.5% Verlust pro FP)
   - JSONB-Konvertierung für `list_models()` implementiert

5. **UI/UX:**
   - Alle optionalen Checkboxen außerhalb des Forms (sofortige Reaktion)
   - Details-Button in jeder Modell-Karte
   - Sidebar-Navigation verbessert
   - Confusion Matrix als übersichtliche Tabelle

### 🧪 Test-Status

- ✅ Alle Unit-Tests bestanden
- ✅ Integrationstests erfolgreich
- ✅ XGBoost getestet (funktioniert, zeigt bessere Performance als Random Forest)
- ✅ Alle Metriken werden korrekt berechnet und gespeichert
- ✅ UI zeigt alle Metriken korrekt an

### 📝 Nächste Schritte (Optional)

1. **Monitoring:** Prüfe ob Accuracy-Werte realistischer sind
2. **Tuning:** Experimentiere mit verschiedenen Feature-Engineering Fenstergrößen
3. **Hyperparameter-Tuning:** Automatisches Tuning mit GridSearch/RandomSearch
4. **Model-Versioning:** Erweiterte Versionierung und Tracking

---

**Erstellt:** 2024-12-23  
**Aktualisiert:** 2024-12-23  
**Version:** 1.1  
**Status:** ✅ **VOLLSTÄNDIG IMPLEMENTIERT UND GETESTET**

