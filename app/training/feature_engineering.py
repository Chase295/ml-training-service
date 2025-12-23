"""
Feature Engineering für ML Training Service
Lädt Daten aus coin_metrics und erstellt Labels
"""
import pandas as pd
import numpy as np
import logging
from typing import List, Optional, Dict, Any
from datetime import datetime, timezone
from app.database.connection import get_pool

logger = logging.getLogger(__name__)

# ⚠️ RAM-Management: Max Anzahl Zeilen
MAX_TRAINING_ROWS = 500000

def _ensure_utc(dt: str | datetime) -> datetime:
    """
    Konvertiert datetime zu UTC (tz-aware).
    
    Hilfsfunktion für konsistente Zeitzone-Behandlung.
    Unterstützt ISO-Format Strings und datetime-Objekte.
    
    Args:
        dt: Datetime als String (ISO-Format) oder datetime-Objekt
        
    Returns:
        datetime-Objekt mit UTC-Zeitzone
        
    Example:
        ```python
        dt1 = _ensure_utc("2024-01-01T00:00:00Z")
        dt2 = _ensure_utc(datetime.now())
        ```
    """
    if isinstance(dt, str):
        # ISO-Format mit Z oder +00:00
        dt = dt.replace('Z', '+00:00')
        dt = datetime.fromisoformat(dt)
    
    if dt.tzinfo is None:
        # Keine Zeitzone → UTC annehmen
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        # Konvertiere zu UTC falls andere Zeitzone
        dt = dt.astimezone(timezone.utc)
    
    return dt

async def load_training_data(
    train_start: str | datetime,
    train_end: str | datetime,
    features: List[str],
    phases: Optional[List[int]] = None
) -> pd.DataFrame:
    """
    Lädt Trainingsdaten aus coin_metrics
    
    ⚠️ KRITISCH: UTC-Zeitzone!
    ⚠️ RAM-Management: LIMIT 500000 Zeilen!
    
    Args:
        train_start: Start-Zeitpunkt (ISO-Format oder datetime, wird zu UTC konvertiert)
        train_end: Ende-Zeitpunkt (ISO-Format oder datetime, wird zu UTC konvertiert)
        features: Liste der Feature-Namen (z.B. ["price_open", "price_high", "volume_sol"])
        phases: Liste der Coin-Phasen (z.B. [1, 2, 3]) oder None für alle
    
    Returns:
        DataFrame mit Trainingsdaten
    """
    pool = await get_pool()
    
    # ⚠️ Konvertiere zu UTC
    train_start_utc = _ensure_utc(train_start)
    train_end_utc = _ensure_utc(train_end)
    
    # Baue SQL Query
    feature_list = ", ".join(features)
    
    # Phase-Filter
    if phases:
        phase_filter = "AND phase_id_at_time = ANY($3)"
        params = [train_start_utc, train_end_utc, phases]
        param_count = 3
    else:
        phase_filter = ""
        params = [train_start_utc, train_end_utc]
        param_count = 2
    
    # ⚠️ RAM-Management: LIMIT für große Datensätze
    query = f"""
        SELECT timestamp, {feature_list}, phase_id_at_time
        FROM coin_metrics
        WHERE timestamp >= $1 AND timestamp <= $2
        {phase_filter}
        ORDER BY timestamp
        LIMIT ${param_count + 1}
    """
    params.append(MAX_TRAINING_ROWS)
    
    logger.info(f"📊 Lade Daten: {train_start_utc} bis {train_end_utc}, Features: {features}, Phasen: {phases}")
    
    # Führe Query aus
    rows = await pool.fetch(query, *params)
    
    if not rows:
        logger.warning("⚠️ Keine Daten gefunden!")
        return pd.DataFrame()
    
    # Konvertiere zu DataFrame
    data = pd.DataFrame([dict(row) for row in rows])
    
    # Setze timestamp als Index
    if 'timestamp' in data.columns:
        # ⚠️ WICHTIG: Entferne doppelte Timestamps (kann bei mehreren Coins passieren)
        # Behalte nur die erste Zeile pro Timestamp
        data = data.drop_duplicates(subset='timestamp', keep='first')
        data.set_index('timestamp', inplace=True)
        # Sortiere nach Index (falls nicht bereits sortiert)
        data = data.sort_index()
    
    # ⚠️ WICHTIG: phase_id_at_time muss als Spalte bleiben (für zeitbasierte Labels)
    # Es wird nicht als Index verwendet, sondern als normale Spalte behalten
    
    logger.info(f"✅ {len(data)} Zeilen geladen (nach Duplikat-Entfernung)")
    
    # Prüfe ob LIMIT erreicht wurde
    if len(data) >= MAX_TRAINING_ROWS:
        logger.warning(f"⚠️ LIMIT erreicht ({MAX_TRAINING_ROWS} Zeilen)! Möglicherweise wurden Daten abgeschnitten.")
    
    return data

def create_labels(
    data: pd.DataFrame,
    target_variable: str,
    target_operator: str,
    target_value: float
) -> pd.Series:
    """
    Erstellt binäre Labels (0/1) basierend auf target_variable/operator/value
    
    Args:
        data: DataFrame mit Trainingsdaten
        target_variable: Ziel-Variable (z.B. "market_cap_close")
        target_operator: Vergleichsoperator (">", "<", ">=", "<=", "=")
        target_value: Schwellwert
    
    Returns:
        Series mit binären Labels (0 oder 1)
    """
    if target_variable not in data.columns:
        raise ValueError(f"Target-Variable '{target_variable}' nicht in Daten gefunden!")
    
    values = data[target_variable]
    
    # Erstelle Labels basierend auf Operator
    if target_operator == ">":
        labels = (values > target_value).astype(int)
    elif target_operator == "<":
        labels = (values < target_value).astype(int)
    elif target_operator == ">=":
        labels = (values >= target_value).astype(int)
    elif target_operator == "<=":
        labels = (values <= target_value).astype(int)
    elif target_operator == "=":
        labels = (values == target_value).astype(int)
    else:
        raise ValueError(f"Unbekannter Operator: {target_operator}")
    
    positive = labels.sum()
    negative = len(labels) - positive
    
    logger.info(f"✅ Labels erstellt: {positive} positive, {negative} negative")
    
    return labels

def create_time_based_labels(
    data: pd.DataFrame,
    target_variable: str,
    future_minutes: int,
    min_percent_change: float,
    direction: str = "up",  # "up" oder "down"
    phase_intervals: Optional[Dict[int, int]] = None  # {phase_id: interval_seconds}
) -> pd.Series:
    """
    Erstellt Labels für zeitbasierte Vorhersagen
    
    Beispiel: "Steigt price_close in 10 Minuten um mindestens 5%?"
    
    ⚠️ WICHTIG: 
    - Daten müssen nach timestamp sortiert sein!
    - Verwendet interval_seconds pro Phase aus ref_coin_phases (genauer als Durchschnitt!)
    
    Args:
        data: DataFrame mit Trainingsdaten (MUSS nach timestamp sortiert sein!)
        target_variable: Variable die überwacht wird (z.B. "price_close")
        future_minutes: Anzahl Minuten in die Zukunft (z.B. 10)
        min_percent_change: Mindest-Prozent-Änderung (z.B. 5.0 für 5%)
        direction: "up" (steigt) oder "down" (fällt)
    
    Returns:
        Series mit binären Labels (1 = Bedingung erfüllt, 0 = nicht erfüllt)
    """
    if target_variable not in data.columns:
        raise ValueError(f"Target-Variable '{target_variable}' nicht in Daten gefunden!")
    
    # ⚠️ WICHTIG: Daten müssen nach timestamp sortiert sein!
    if not data.index.is_monotonic_increasing:
        data = data.sort_index()
        logger.warning("⚠️ Daten wurden nach timestamp sortiert")
    
    # Aktueller Wert
    current_values = data[target_variable]
    
    if len(data) < 2:
        raise ValueError("Nicht genug Daten für zeitbasierte Labels (mindestens 2 Zeilen benötigt)")
    
    # NEU: Verwende interval_seconds pro Phase aus ref_coin_phases (falls übergeben)
    # Prüfe ob phase_id_at_time vorhanden ist
    if 'phase_id_at_time' in data.columns and phase_intervals:
        # Verwende interval_seconds pro Phase (genauer!)
        logger.info(f"✅ Verwende interval_seconds pro Phase aus ref_coin_phases")
        
        # Erstelle Series mit rows_to_shift pro Zeile basierend auf Phase
        def calculate_rows_to_shift(phase_id):
            if pd.isna(phase_id) or phase_id not in phase_intervals:
                # Fallback: Verwende Durchschnitt wenn Phase unbekannt
                time_diffs = data.index.to_series().diff().dropna()
                avg_interval_minutes = time_diffs.mean().total_seconds() / 60.0
                return int(round(future_minutes / avg_interval_minutes)) if avg_interval_minutes > 0 else 0
            
            interval_seconds = phase_intervals[phase_id]
            if interval_seconds <= 0:
                # Fallback für Phase 99 (Finished) mit interval_seconds = 0
                time_diffs = data.index.to_series().diff().dropna()
                avg_interval_minutes = time_diffs.mean().total_seconds() / 60.0
                return int(round(future_minutes / avg_interval_minutes)) if avg_interval_minutes > 0 else 0
            
            interval_minutes = interval_seconds / 60.0
            return int(round(future_minutes / interval_minutes))
        
        # Berechne rows_to_shift pro Zeile
        rows_to_shift_series = data['phase_id_at_time'].apply(calculate_rows_to_shift)
        
        # Berechne zukünftige Werte pro Zeile (verschiedene Shifts je nach Phase)
        future_values = pd.Series(index=data.index, dtype=float)
        for idx in data.index:
            rows_to_shift_val = rows_to_shift_series.loc[idx]
            # Konvertiere zu int falls Series
            if isinstance(rows_to_shift_val, pd.Series):
                rows_to_shift_val = rows_to_shift_val.iloc[0] if len(rows_to_shift_val) > 0 else 0
            rows_to_shift = int(rows_to_shift_val) if not pd.isna(rows_to_shift_val) else 0
            
            if rows_to_shift > 0:
                # Finde Index nach rows_to_shift Zeilen
                try:
                    current_pos = data.index.get_loc(idx)
                    if isinstance(current_pos, slice):
                        current_pos = current_pos.start if current_pos.start is not None else 0
                    future_pos = current_pos + rows_to_shift
                    if future_pos < len(data.index):
                        future_idx = data.index[future_pos]
                        future_values.loc[idx] = data.loc[future_idx, target_variable]
                    else:
                        future_values.loc[idx] = np.nan
                except (IndexError, KeyError, TypeError):
                    # Am Ende des Datensatzes: kein Zukunftswert verfügbar
                    future_values.loc[idx] = np.nan
            else:
                future_values.loc[idx] = np.nan
        
        logger.info(f"📊 Zeitbasierte Labels: {future_minutes} Minuten")
        logger.info(f"   Phase-Intervalle verwendet: {len(phase_intervals)} Phasen geladen")
        
    else:
        # Fallback: Verwende Durchschnitt (wenn phase_id_at_time nicht vorhanden)
        logger.warning("⚠️ phase_id_at_time nicht gefunden, verwende Durchschnitts-Intervall")
        
        time_diffs = data.index.to_series().diff().dropna()
        avg_interval_minutes = time_diffs.mean().total_seconds() / 60.0
        
        if avg_interval_minutes <= 0:
            raise ValueError(f"Ungültiges Zeitintervall: {avg_interval_minutes} Minuten")
        
        rows_to_shift = int(round(future_minutes / avg_interval_minutes))
        
        if rows_to_shift <= 0:
            raise ValueError(f"future_minutes ({future_minutes}) ist kleiner als durchschnittliches Intervall ({avg_interval_minutes:.2f} Minuten)")
        
        logger.info(f"📊 Zeitbasierte Labels: {future_minutes} Minuten = ~{rows_to_shift} Zeilen (Intervall: {avg_interval_minutes:.2f} Min)")
        
        # Zukünftiger Wert (shift nach hinten, da wir in die Zukunft schauen)
        future_values = data[target_variable].shift(-rows_to_shift)
    
    # Berechne prozentuale Änderung
    percent_change = ((future_values - current_values) / current_values) * 100
    
    # Erstelle Labels basierend auf Richtung
    if direction == "up":
        # Steigt um mindestens min_percent_change?
        labels = (percent_change >= min_percent_change).astype(int)
    else:  # "down"
        # Fällt um mindestens min_percent_change?
        labels = (percent_change <= -min_percent_change).astype(int)
    
    # Entferne NaN-Werte (am Ende des Datensatzes, wo keine Zukunftswerte existieren)
    # Setze auf 0 (konservativ: wenn keine Zukunft bekannt, dann "nicht erfüllt")
    labels = labels.fillna(0)
    
    positive = labels.sum()
    negative = len(labels) - positive
    
    logger.info(f"✅ Zeitbasierte Labels erstellt: {positive} positive, {negative} negative")
    logger.info(f"   Zeitraum: {future_minutes} Minuten, Min-Änderung: {min_percent_change}%, Richtung: {direction}")
    
    return labels

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
    
    # Prüfe ob benötigte Spalten vorhanden sind
    required_cols = ['price_close', 'volume_usd', 'order_buy_volume', 'order_sell_volume',
                     'whale_buy_volume', 'whale_sell_volume', 'price_high', 'price_low',
                     'market_cap_close', 'order_buy_count', 'order_sell_count']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        logger.warning(f"⚠️ Fehlende Spalten für Feature-Engineering: {missing_cols}. Überspringe diese Features.")
    
    # 1. PRICE MOMENTUM (Preisänderungen über verschiedene Zeitfenster)
    if 'price_close' in df.columns:
        for window in window_sizes:
            # Prozentuale Preisänderung
            df[f'price_change_{window}'] = df['price_close'].pct_change(periods=window) * 100
            
            # Rate of Change (ROC)
            df[f'price_roc_{window}'] = ((df['price_close'] - df['price_close'].shift(window)) / 
                                          df['price_close'].shift(window).replace(0, np.nan)) * 100
    
    # 2. VOLUME PATTERNS (Volumen-Anomalien)
    if 'volume_usd' in df.columns:
        for window in window_sizes:
            # Volumen-Änderung vs. Rolling Average
            rolling_avg = df['volume_usd'].rolling(window=window, min_periods=1).mean()
            df[f'volume_ratio_{window}'] = df['volume_usd'] / rolling_avg.replace(0, np.nan)
            
            # Volumen-Spike (Standard Deviation)
            rolling_std = df['volume_usd'].rolling(window=window, min_periods=1).std()
            df[f'volume_spike_{window}'] = (df['volume_usd'] - rolling_avg) / rolling_std.replace(0, np.nan)
    
    # 3. BUY/SELL PRESSURE
    if 'order_buy_volume' in df.columns and 'order_sell_volume' in df.columns:
        # Buy-Sell Ratio
        df['buy_sell_ratio'] = df['order_buy_volume'] / (df['order_sell_volume'] + 1e-10)
        
        # Buy-Sell Pressure (Normalized)
        total_volume = df['order_buy_volume'] + df['order_sell_volume']
        df['buy_pressure'] = df['order_buy_volume'] / (total_volume + 1e-10)
        df['sell_pressure'] = df['order_sell_volume'] / (total_volume + 1e-10)
    
    # 4. WHALE ACTIVITY
    if 'whale_buy_volume' in df.columns and 'whale_sell_volume' in df.columns:
        # Whale Buy/Sell Ratio
        df['whale_buy_sell_ratio'] = df['whale_buy_volume'] / (df['whale_sell_volume'] + 1e-10)
        
        # Whale Activity Spike
        for window in window_sizes:
            whale_total = df['whale_buy_volume'] + df['whale_sell_volume']
            rolling_avg = whale_total.rolling(window=window, min_periods=1).mean()
            df[f'whale_activity_spike_{window}'] = whale_total / (rolling_avg + 1e-10)
    
    # 5. PRICE VOLATILITY
    if 'price_close' in df.columns and 'price_high' in df.columns and 'price_low' in df.columns:
        for window in window_sizes:
            # Rolling Standard Deviation
            df[f'price_volatility_{window}'] = df['price_close'].rolling(window=window, min_periods=1).std()
            
            # High-Low Range
            df[f'price_range_{window}'] = (df['price_high'] - df['price_low']).rolling(window=window, min_periods=1).mean()
    
    # 6. MARKET CAP VELOCITY (Rate of Change)
    if 'market_cap_close' in df.columns:
        for window in window_sizes:
            df[f'mcap_velocity_{window}'] = ((df['market_cap_close'] - df['market_cap_close'].shift(window)) / 
                                              df['market_cap_close'].shift(window).replace(0, np.nan)) * 100
    
    # 7. ORDER BOOK IMBALANCE
    if 'order_buy_count' in df.columns and 'order_sell_count' in df.columns:
        # Buy-Orders vs. Sell-Orders
        total_orders = df['order_buy_count'] + df['order_sell_count']
        df['order_imbalance'] = (df['order_buy_count'] - df['order_sell_count']) / (total_orders + 1e-10)
    
    # NaN-Werte durch 0 ersetzen (entstehen durch Rolling/Shift)
    df.fillna(0, inplace=True)
    
    # Infinite Werte durch 0 ersetzen
    df.replace([np.inf, -np.inf], 0, inplace=True)
    
    engineered_count = len(get_engineered_feature_names(window_sizes))
    logger.info(f"✅ {engineered_count} zusätzliche Features erstellt")
    
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


def check_overlap(
    train_start: datetime | str,
    train_end: datetime | str,
    test_start: datetime | str,
    test_end: datetime | str
) -> Dict[str, Any]:
    """
    Prüft ob Test-Zeitraum sich mit Training überschneidet
    
    ⚠️ Wichtig: Gibt Warnung zurück, blockiert aber NICHT den Test!
    
    Args:
        train_start: Training Start-Zeitpunkt
        train_end: Training Ende-Zeitpunkt
        test_start: Test Start-Zeitpunkt
        test_end: Test Ende-Zeitpunkt
    
    Returns:
        Dict mit has_overlap (bool) und overlap_note (str)
    """
    # Konvertiere zu UTC
    train_start_utc = _ensure_utc(train_start)
    train_end_utc = _ensure_utc(train_end)
    test_start_utc = _ensure_utc(test_start)
    test_end_utc = _ensure_utc(test_end)
    
    # Prüfe Überschneidung
    train_duration = (train_end_utc - train_start_utc).total_seconds()
    test_duration = (test_end_utc - test_start_utc).total_seconds()
    
    # Berechne Überschneidung
    overlap_start = max(train_start_utc, test_start_utc)
    overlap_end = min(train_end_utc, test_end_utc)
    
    if overlap_start < overlap_end:
        overlap_duration = (overlap_end - overlap_start).total_seconds()
        overlap_percent = (overlap_duration / test_duration) * 100 if test_duration > 0 else 0
        
        return {
            "has_overlap": True,
            "overlap_note": f"⚠️ {overlap_percent:.1f}% Überschneidung mit Trainingsdaten - Ergebnisse können verfälscht sein"
        }
    else:
        return {
            "has_overlap": False,
            "overlap_note": "✅ Keine Überschneidung mit Trainingsdaten"
        }

