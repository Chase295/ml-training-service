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
    phases: Optional[List[int]] = None,
    include_ath: bool = True  # NEU: ATH-Daten optional laden
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
        include_ath: Wenn True, werden ATH-Daten aus coin_streams geladen (Default: True)
    
    Returns:
        DataFrame mit Trainingsdaten
    """
    pool = await get_pool()
    
    # ⚠️ Konvertiere zu UTC
    train_start_utc = _ensure_utc(train_start)
    train_end_utc = _ensure_utc(train_end)
    
    # ⚠️ WICHTIG: Lade IMMER alle neuen Metriken (auch wenn nicht in Features-Liste)
    # Damit sind sie für Feature-Engineering verfügbar
    base_columns = """
        cm.timestamp, 
        cm.phase_id_at_time,
        cm.mint,
        
        -- Basis OHLC
        cm.price_open, cm.price_high, cm.price_low, cm.price_close,
        
        -- Volumen
        cm.volume_sol, cm.buy_volume_sol, cm.sell_volume_sol, cm.net_volume_sol,
        
        -- Market Cap & Phase
        cm.market_cap_close,
        
        -- ⚠️ KRITISCH: Dev-Tracking (Rug-Pull-Indikator)
        cm.dev_sold_amount,
        
        -- Ratio-Metriken (Bot-Spam vs. echtes Interesse)
        cm.buy_pressure_ratio,
        cm.unique_signer_ratio,
        
        -- Whale-Aktivität
        cm.whale_buy_volume_sol,
        cm.whale_sell_volume_sol,
        cm.num_whale_buys,
        cm.num_whale_sells,
        
        -- Volatilität
        cm.volatility_pct,
        cm.avg_trade_size_sol
    """
    
    # 🆕 ATH-Tracking (vereinfacht - historische Berechnung erfolgt in Python)
    if include_ath:
        base_columns += """,
            -- ATH-Daten aus coin_streams (nur aktuelle Werte für historische Berechnung)
            COALESCE(cs.ath_price_sol, cm.price_high) as current_ath_price_sol,
            cs.ath_timestamp as current_ath_timestamp
        """
    
    # Zusätzliche Features aus Request (falls nicht bereits in base_columns)
    base_column_names = [
        'timestamp', 'phase_id_at_time', 'mint', 'price_open', 'price_high', 'price_low', 'price_close',
        'volume_sol', 'buy_volume_sol', 'sell_volume_sol', 'net_volume_sol',
        'market_cap_close', 'dev_sold_amount', 'buy_pressure_ratio', 'unique_signer_ratio',
        'whale_buy_volume_sol', 'whale_sell_volume_sol', 'num_whale_buys', 'num_whale_sells',
        'volatility_pct', 'avg_trade_size_sol'
    ]
    
    # 🆕 ATH-Features hinzufügen (wenn include_ath aktiviert)
    if include_ath:
        base_column_names.extend([
            'current_ath_price_sol', 'current_ath_timestamp'
        ])
    
    # Filtere ATH-Features aus, da sie erst später berechnet werden
    ath_features_to_remove = ['rolling_ath', 'ath_distance_pct', 'ath_breakout', 'minutes_since_ath', 'ath_age_hours', 'ath_is_recent', 'ath_is_old']
    filtered_features = [f for f in features if f not in ath_features_to_remove]

    additional_features = [f for f in filtered_features if f not in base_column_names]
    
    # Phase-Filter
    if phases:
        phase_filter = "AND phase_id_at_time = ANY($3)"
        params = [train_start_utc, train_end_utc, phases]
        param_count = 3
    else:
        phase_filter = ""
        params = [train_start_utc, train_end_utc]
        param_count = 2
    
    # 🆕 JOIN mit coin_streams für ATH-Daten (wenn aktiviert)
    if include_ath:
        join_clause = """
            LEFT JOIN coin_streams cs ON cm.mint = cs.token_address
        """
    else:
        join_clause = ""
    
    # ⚠️ RAM-Management: LIMIT für große Datensätze
    if additional_features:
        additional_list = ", ".join(additional_features)
        query = f"""
            SELECT {base_columns}, {additional_list}
            FROM coin_metrics cm
            {join_clause}
            WHERE cm.timestamp >= $1 AND cm.timestamp <= $2
            {phase_filter}
            ORDER BY cm.timestamp
            LIMIT ${param_count + 1}
        """
    else:
        query = f"""
            SELECT {base_columns}
            FROM coin_metrics cm
            {join_clause}
            WHERE cm.timestamp >= $1 AND cm.timestamp <= $2
            {phase_filter}
            ORDER BY cm.timestamp
            LIMIT ${param_count + 1}
        """
    params.append(MAX_TRAINING_ROWS)
    
    ath_status = "mit ATH-Daten" if include_ath else "ohne ATH-Daten"
    logger.info(f"📊 Lade Daten: {train_start_utc} bis {train_end_utc}, Features: {features}, Phasen: {phases}, {ath_status}")
    
    # Führe Query aus
    rows = await pool.fetch(query, *params)
    
    if not rows:
        logger.warning("⚠️ Keine Daten gefunden!")
        return pd.DataFrame()
    
    # Konvertiere zu DataFrame
    data = pd.DataFrame([dict(row) for row in rows])

    # 🧹 DATA CLEANING: Entferne "tote" Daten (Garbage In, Garbage Out verhindern)
    # 1. Entferne Zeilen mit NULL-Werten in kritischen Spalten
    # Diese Coins hatten nie Trades oder wurden vom Tracker verpasst
    logger.info(f"🧹 Data Cleaning: {len(data)} Zeilen vor Filter")

    # Kritische Spalten, die nicht NULL sein dürfen
    critical_columns = ['price_close', 'volume_sol']
    if include_ath:
        critical_columns.append('current_ath_price_sol')

    # Entferne Zeilen mit NULL in kritischen Spalten
    data_clean = data.dropna(subset=critical_columns)
    logger.info(f"🧹 Data Cleaning: {len(data_clean)} Zeilen nach NULL-Filter (entfernt: {len(data) - len(data_clean)})")

    # 2. Entferne Coins mit zu wenigen Datenpunkten (Rauschen herausfiltern)
    # Ein Coin mit nur 3 Datenpunkten kann kein sinnvolles Muster zeigen
    if 'mint' in data_clean.columns:
        coin_counts = data_clean['mint'].value_counts()
        valid_mints = coin_counts[coin_counts >= 30].index  # Mindestens 30 Einträge pro Coin
        data_final = data_clean[data_clean['mint'].isin(valid_mints)]
        removed_coins = len(coin_counts) - len(valid_mints)
        logger.info(f"🧹 Data Cleaning: {len(data_final)} Zeilen nach Coin-Filter (entfernt: {removed_coins} Coins mit <30 Datenpunkten)")
    else:
        data_final = data_clean
        logger.info("🧹 Data Cleaning: Kein 'mint' Column gefunden, überspringe Coin-Filter")

    # Verwende die bereinigten Daten weiter
    data = data_final

    # ⚠️ WICHTIG: Konvertiere alle Decimal-Typen zu float (PostgreSQL liefert Decimal)
    # Dies verhindert "unsupported operand type(s) for -: 'decimal.Decimal' and 'float'" Fehler
    for col in data.columns:
        if col != 'timestamp' and col != 'phase_id_at_time':  # Timestamp und Phase-ID bleiben unverändert
            # Konvertiere zu numeric (float), ignoriere Fehler (z.B. bei Strings)
            data[col] = pd.to_numeric(data[col], errors='coerce')

    # 🆕 HISTORISCHE ATH-FEATURES: Data Leakage-frei berechnen
    if include_ath:
        logger.info("🧠 Berechne historische ATH-Features...")
        data = add_ath_features(data)
        logger.info(f"✅ ATH-Features hinzugefügt: rolling_ath, ath_distance_pct, ath_breakout, minutes_since_ath")

        # Erweitere Data Cleaning für die neuen ATH-Features
        ath_critical_columns = ['rolling_ath', 'ath_distance_pct', 'ath_breakout', 'minutes_since_ath']
        data_clean_ath = data.dropna(subset=ath_critical_columns)
        removed_ath_rows = len(data) - len(data_clean_ath)
        if removed_ath_rows > 0:
            logger.info(f"🧹 ATH-Data-Cleaning: {removed_ath_rows} Zeilen mit NULL ATH-Features entfernt")
            data = data_clean_ath

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
    
    logger.info(f"✅ {len(data)} Zeilen final (nach Data Cleaning und Duplikat-Entfernung)")
    
    # Prüfe ob LIMIT erreicht wurde (vor Data Cleaning)
    # Nach Data Cleaning kann die Anzahl geringer sein
    if len(data) == 0:
        logger.warning("⚠️ Alle Daten wurden durch Data Cleaning entfernt! Überprüfe die Datenbank.")
    elif len(data) < 1000:
        logger.warning(f"⚠️ Sehr wenige Daten nach Cleaning: {len(data)} Zeilen. Modell-Training könnte unzuverlässig sein.")
    
    return data

async def enrich_with_market_context(
    data: pd.DataFrame,
    train_start: datetime,
    train_end: datetime
) -> pd.DataFrame:
    """
    Fügt Marktstimmung (SOL-Preis) zu Trainingsdaten hinzu.
    Merge mit Forward-Fill (nimmt letzten bekannten Wert).
    
    Args:
        data: DataFrame mit Trainingsdaten (muss timestamp als Index haben)
        train_start: Start-Zeitpunkt
        train_end: Ende-Zeitpunkt
    
    Returns:
        DataFrame mit zusätzlichen Spalten: sol_price_usd, sol_price_change_pct, sol_price_ma_5, sol_price_volatility
    """
    pool = await get_pool()
    
    # Konvertiere zu UTC
    train_start_utc = _ensure_utc(train_start)
    train_end_utc = _ensure_utc(train_end)
    
    # Lade Exchange Rates
    sql = """
        SELECT 
            created_at as timestamp,
            sol_price_usd,
            usd_to_eur_rate
        FROM exchange_rates
        WHERE created_at >= $1 AND created_at <= $2
        ORDER BY created_at
    """
    
    try:
        rows = await pool.fetch(sql, train_start_utc, train_end_utc)
        
        if not rows:
            logger.warning("⚠️ Keine Exchange Rates gefunden - Marktstimmung wird nicht hinzugefügt")
            return data
        
        # Konvertiere zu DataFrame
        rates_df = pd.DataFrame([dict(row) for row in rows])
        rates_df['timestamp'] = pd.to_datetime(rates_df['timestamp'])
        rates_df.set_index('timestamp', inplace=True)
        
        # Merge mit Forward-Fill (nehme letzten bekannten Wert)
        data = data.merge(rates_df[['sol_price_usd']], left_index=True, right_index=True, how='left')
        data['sol_price_usd'].fillna(method='ffill', inplace=True)
        
        # ✅ NEUE Features berechnen:
        data['sol_price_change_pct'] = data['sol_price_usd'].pct_change() * 100
        data['sol_price_ma_5'] = data['sol_price_usd'].rolling(5, min_periods=1).mean()
        data['sol_price_volatility'] = data['sol_price_usd'].rolling(10, min_periods=1).std()
        
        # NaN-Werte durch 0 ersetzen (am Anfang der Serie)
        data['sol_price_change_pct'].fillna(0, inplace=True)
        data['sol_price_ma_5'].fillna(data['sol_price_usd'], inplace=True)
        data['sol_price_volatility'].fillna(0, inplace=True)
        
        logger.info("✅ Marktstimmung hinzugefügt: sol_price_usd, sol_price_change_pct, sol_price_ma_5, sol_price_volatility")
        
    except Exception as e:
        logger.warning(f"⚠️ Fehler beim Laden der Exchange Rates: {e} - Marktstimmung wird nicht hinzugefügt")
    
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
            # ⚠️ WICHTIG: Verwende ceil() statt round() für konservativere Berechnung
            return max(1, int(np.ceil(future_minutes / interval_minutes)))
        
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
        
        # ⚠️ WICHTIG: Verwende ceil() statt round() für konservativere Berechnung
        # ceil() stellt sicher, dass wir mindestens genug Zeilen nehmen
        rows_to_shift = max(1, int(np.ceil(future_minutes / avg_interval_minutes)))
        
        if rows_to_shift <= 0:
            raise ValueError(f"future_minutes ({future_minutes}) ist kleiner als durchschnittliches Intervall ({avg_interval_minutes:.2f} Minuten)")
        
        logger.info(f"📊 Zeitbasierte Labels: {future_minutes} Minuten = ~{rows_to_shift} Zeilen (Intervall: {avg_interval_minutes:.2f} Min)")
        
        # Zukünftiger Wert (shift nach hinten, da wir in die Zukunft schauen)
        future_values = data[target_variable].shift(-rows_to_shift)
    
    # Berechne prozentuale Änderung
    # ⚠️ WICHTIG: Konvertiere Decimal zu float (Datenbank liefert Decimal, Pandas braucht float)
    current_values = pd.to_numeric(current_values, errors='coerce')
    future_values = pd.to_numeric(future_values, errors='coerce')
    
    # ⚠️ WICHTIG: Vermeide Division durch Null
    # Erstelle Mask für gültige Werte (current_values != 0 und nicht NaN)
    valid_mask = (current_values != 0) & current_values.notna() & future_values.notna()
    
    # Berechne Prozent-Änderung nur für gültige Werte
    percent_change = pd.Series(index=data.index, dtype=float)
    percent_change[valid_mask] = ((future_values[valid_mask] - current_values[valid_mask]) / current_values[valid_mask]) * 100
    
    # Setze ungültige Werte auf NaN (werden später behandelt)
    percent_change[~valid_mask] = np.nan
    
    # Erstelle Labels basierend auf Richtung
    if direction == "up":
        # Steigt um mindestens min_percent_change?
        labels = (percent_change >= min_percent_change).astype(float)  # float für NaN-Handling
    else:  # "down"
        # Fällt um mindestens min_percent_change?
        labels = (percent_change <= -min_percent_change).astype(float)  # float für NaN-Handling
    
    # ⚠️ WICHTIG: Behandle NaN-Werte (am Ende des Datensatzes oder bei Null-Werten)
    nan_count = labels.isna().sum()
    if nan_count > 0:
        logger.warning(f"⚠️ {nan_count} Zeilen ohne gültige Zukunftswerte (werden ausgeschlossen)")
        # Entferne Zeilen mit NaN aus Labels (und später auch aus Daten)
        labels = labels.dropna()
        # Entferne entsprechende Zeilen aus Daten (wichtig für Alignment!)
        data = data.loc[labels.index]
        logger.info(f"📊 Nach NaN-Entfernung: {len(labels)} gültige Labels")
    
    # Konvertiere zu int (nach NaN-Entfernung)
    labels = labels.astype(int)
    
    positive = labels.sum()
    negative = len(labels) - positive
    
    logger.info(f"✅ Zeitbasierte Labels erstellt: {positive} positive, {negative} negative")
    logger.info(f"   Zeitraum: {future_minutes} Minuten, Min-Änderung: {min_percent_change}%, Richtung: {direction}")
    
    # ⚠️ WICHTIG: Wenn Daten gefiltert wurden (NaN entfernt), gib auch gefilterte Daten zurück
    # Prüfe ob data und labels noch aligned sind
    if len(data) != len(labels):
        logger.warning(f"⚠️ Daten und Labels nicht aligned: {len(data)} Daten, {len(labels)} Labels")
        # Filtere Daten auf Labels-Index
        data = data.loc[labels.index]
        logger.info(f"📊 Daten gefiltert: {len(data)} Zeilen")
        return labels, data  # Gib beide zurück
    
    return labels  # Normale Rückgabe (nur Labels)

def add_ath_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    HISTORISCH KORREKTE ATH-FEATURES: Berechnet ATH-Features ohne Data Leakage

    WICHTIG: Diese Funktion berechnet historische ATH-Werte basierend auf den Daten
    im DataFrame, nicht aus statischen Datenbank-Snapshots. Dadurch wird Data Leakage
    verhindert, da die KI nur historische Informationen nutzt.

    Features:
    - rolling_ath: Historisches All-Time-High bis zu jedem Zeitpunkt
    - ath_distance_pct: Wie weit ist der aktuelle Preis vom historischen ATH entfernt?
    - ath_breakout: Ist das aktuelle High höher als das vorherige historische ATH?
    - minutes_since_ath: Wie viele Minuten sind seit dem letzten ATH vergangen?

    Args:
        df: DataFrame mit price_high, price_close, timestamp, mint Spalten

    Returns:
        DataFrame mit zusätzlichen ATH-Features
    """
    logger.info("🚀 Berechne historische ATH-Features (Data Leakage-frei)")

    # Kopie erstellen
    df = df.copy()

    # Sortiere nach mint und timestamp (wichtig für historische Berechnung!)
    df = df.sort_values(['mint', 'timestamp'])

    # 1. ROLLING ATH berechnen (historisches All-Time-High)
    df['rolling_ath'] = df.groupby('mint')['price_high'].cummax()

    # 2. ATH_DISTANCE_PCT: Wie weit entfernt vom ATH?
    # Formel: (ATH - Close) / ATH * 100
    # Negativ = unter ATH, positiv = über ATH
    df['ath_distance_pct'] = ((df['rolling_ath'] - df['price_close']) / df['rolling_ath'].replace(0, np.nan)) * 100

    # 3. ATH_BREAKOUT: Neue ATH-Breaks
    # Ist das aktuelle High höher als das vorherige historische ATH?
    df['prev_rolling_ath'] = df.groupby('mint')['rolling_ath'].shift(1)
    df['ath_breakout'] = (df['price_high'] > df['prev_rolling_ath']).astype(int)

    # 4. MINUTES_SINCE_ATH: Zeit seit letztem ATH
    # Finde den Timestamp des letzten ATH-Breakouts
    def calculate_minutes_since_ath(group):
        # Finde alle Zeiten wo ein neuer ATH erreicht wurde
        ath_timestamps = group.loc[group['ath_breakout'] == 1, 'timestamp']

        if len(ath_timestamps) == 0:
            # Kein ATH-Breakout gefunden, verwende ersten Timestamp als Baseline
            group['minutes_since_ath'] = (group['timestamp'] - group['timestamp'].iloc[0]).dt.total_seconds() / 60
        else:
            # Berechne Zeit seit letztem ATH-Breakout
            group['last_ath_timestamp'] = ath_timestamps.reindex(group.index, method='ffill')
            group['minutes_since_ath'] = (group['timestamp'] - group['last_ath_timestamp']).dt.total_seconds() / 60

        return group

    df = df.groupby('mint', group_keys=False).apply(calculate_minutes_since_ath)

    # 5. Zusätzliche nützliche ATH-Features
    df['ath_age_hours'] = df['minutes_since_ath'] / 60.0
    df['ath_is_recent'] = (df['minutes_since_ath'] < 60).astype(int)  # Innerhalb 1 Stunde
    df['ath_is_old'] = (df['minutes_since_ath'] > 1440).astype(int)  # Älter als 24 Stunden

    # NaN-Werte sinnvoll behandeln
    df['minutes_since_ath'].fillna(0, inplace=True)  # Am Anfang: 0 Minuten
    df['ath_age_hours'].fillna(0, inplace=True)
    df['ath_distance_pct'].fillna(0, inplace=True)  # Wenn kein ATH: 0%

    # Hilfsspalten entfernen
    df.drop(['prev_rolling_ath', 'last_ath_timestamp'], axis=1, inplace=True, errors='ignore')

    logger.info(f"✅ Historische ATH-Features erstellt: {len(df)} Zeilen")
    logger.info(f"   Features: rolling_ath, ath_distance_pct, ath_breakout, minutes_since_ath")

    return df

def create_pump_detection_features(
    data: pd.DataFrame,
    window_sizes: list = [5, 10, 15]
) -> pd.DataFrame:
    """
    MODERNISIERT: Nutzt neue Metriken aus coin_metrics.
    Erstellt zusätzliche Features für Pump-Detection.
    
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
    
    # ✅ 1. Dev-Tracking Features (KRITISCH!)
    if 'dev_sold_amount' in df.columns:
        df['dev_sold_flag'] = (df['dev_sold_amount'] > 0).astype(int)
        df['dev_sold_cumsum'] = df['dev_sold_amount'].cumsum()
        for window in window_sizes:
            df[f'dev_sold_spike_{window}'] = (
                df['dev_sold_amount'].rolling(window, min_periods=1).sum() > 0
            ).astype(int)
    
    # ✅ 2. Ratio-Features (schon berechnet in coin_metrics!)
    if 'buy_pressure_ratio' in df.columns:
        for window in window_sizes:
            df[f'buy_pressure_ma_{window}'] = (
                df['buy_pressure_ratio'].rolling(window, min_periods=1).mean()
            )
            df[f'buy_pressure_trend_{window}'] = (
                df['buy_pressure_ratio'] - df[f'buy_pressure_ma_{window}']
            )
    
    # ✅ 3. Whale-Aktivität Features
    if 'whale_buy_volume_sol' in df.columns and 'whale_sell_volume_sol' in df.columns:
        df['whale_net_volume'] = (
            df['whale_buy_volume_sol'] - df['whale_sell_volume_sol']
        )
        for window in window_sizes:
            df[f'whale_activity_{window}'] = (
                df['whale_buy_volume_sol'].rolling(window, min_periods=1).sum() +
                df['whale_sell_volume_sol'].rolling(window, min_periods=1).sum()
            )
    
    # ✅ 4. Volatilitäts-Features (nutzt neue volatility_pct Spalte!)
    if 'volatility_pct' in df.columns:
        for window in window_sizes:
            df[f'volatility_ma_{window}'] = (
                df['volatility_pct'].rolling(window, min_periods=1).mean()
            )
            df[f'volatility_spike_{window}'] = (
                df['volatility_pct'] > 
                df[f'volatility_ma_{window}'] * 1.5
            ).astype(int)
    
    # ✅ 5. Wash-Trading Detection
    if 'unique_signer_ratio' in df.columns:
        for window in window_sizes:
            df[f'wash_trading_flag_{window}'] = (
                df['unique_signer_ratio'].rolling(window, min_periods=1).mean() < 0.15
            ).astype(int)
    
    # ✅ 6. Net-Volume Features
    if 'net_volume_sol' in df.columns:
        for window in window_sizes:
            df[f'net_volume_ma_{window}'] = (
                df['net_volume_sol'].rolling(window, min_periods=1).mean()
            )
            df[f'volume_flip_{window}'] = (
                (df['net_volume_sol'] > 0).astype(int).diff().abs()
            )
    
    # ✅ 7. Price Momentum (nutzt price_close)
    if 'price_close' in df.columns:
        for window in window_sizes:
            df[f'price_change_{window}'] = df['price_close'].pct_change(periods=window) * 100
            df[f'price_roc_{window}'] = (
                (df['price_close'] - df['price_close'].shift(window)) / 
                df['price_close'].shift(window).replace(0, np.nan)
            ) * 100
    
    # ✅ 8. Volume Patterns (nutzt volume_sol)
    if 'volume_sol' in df.columns:
        for window in window_sizes:
            rolling_avg = df['volume_sol'].rolling(window=window, min_periods=1).mean()
            df[f'volume_ratio_{window}'] = df['volume_sol'] / rolling_avg.replace(0, np.nan)
            rolling_std = df['volume_sol'].rolling(window=window, min_periods=1).std()
            df[f'volume_spike_{window}'] = (
                (df['volume_sol'] - rolling_avg) / rolling_std.replace(0, np.nan)
            )
    
    # ✅ 9. Market Cap Velocity
    if 'market_cap_close' in df.columns:
        for window in window_sizes:
            df[f'mcap_velocity_{window}'] = (
                (df['market_cap_close'] - df['market_cap_close'].shift(window)) / 
                df['market_cap_close'].shift(window).replace(0, np.nan)
            ) * 100
    
    # 🆕 10. ATH-basierte Rolling-Window Features (auf bereits berechneten historischen ATH-Features)
    # Die historischen ATH-Features werden bereits in add_ath_features() berechnet
    if 'ath_distance_pct' in df.columns and 'ath_breakout' in df.columns:
        # Rolling-Windows für ATH-Features (auf historisch korrekten Werten)
        for window in window_sizes:
            # ATH-Trend (nähert sich Preis dem ATH?)
            df[f'ath_distance_trend_{window}'] = (
                df['ath_distance_pct'].rolling(window, min_periods=1).mean()
            )
            df[f'ath_approach_{window}'] = (
                df[f'ath_distance_trend_{window}'].diff() < 0
            ).astype(int)  # Nähert sich dem ATH

            # ATH-Breakout-Häufigkeit
            df[f'ath_breakout_count_{window}'] = (
                df['ath_breakout'].rolling(window, min_periods=1).sum()
            )

            # ATH-Volumen bei Breakouts
            if 'volume_sol' in df.columns:
                ath_breakout_volume = df['ath_breakout'] * df['volume_sol']
                df[f'ath_breakout_volume_ma_{window}'] = (
                    ath_breakout_volume.rolling(window, min_periods=1).mean()
                )

    # 🆕 11. ATH-Zeit-Features Rolling-Windows (auf bereits berechneten historischen Werten)
    if 'minutes_since_ath' in df.columns:
        for window in window_sizes:
            df[f'ath_age_trend_{window}'] = (
                df['minutes_since_ath'].rolling(window, min_periods=1).mean()
            )
    
    # NaN-Werte durch 0 ersetzen (entstehen durch Rolling/Shift)
    df.fillna(0, inplace=True)
    
    # Infinite Werte durch 0 ersetzen
    df.replace([np.inf, -np.inf], 0, inplace=True)
    
    engineered_count = len([c for c in df.columns if c not in data.columns])
    logger.info(f"✅ {engineered_count} zusätzliche Features erstellt")
    
    return df


def get_engineered_feature_names(window_sizes: list = [5, 10, 15]) -> list:
    """
    Gibt die Namen aller erstellten Features zurück.
    Nützlich für Feature-Auswahl in UI und Feature Importance.
    
    ⚠️ WICHTIG: Diese Liste muss mit create_pump_detection_features() übereinstimmen!
    
    Args:
        window_sizes: Fenstergrößen (muss mit create_pump_detection_features() übereinstimmen)
    
    Returns:
        Liste der Feature-Namen
    """
    features = []
    
    # ✅ 1. Dev-Tracking Features
    features.extend(['dev_sold_flag', 'dev_sold_cumsum'])
    for w in window_sizes:
        features.append(f'dev_sold_spike_{w}')
    
    # ✅ 2. Ratio-Features
    for w in window_sizes:
        features.extend([f'buy_pressure_ma_{w}', f'buy_pressure_trend_{w}'])
    
    # ✅ 3. Whale-Aktivität Features
    features.append('whale_net_volume')
    for w in window_sizes:
        features.append(f'whale_activity_{w}')
    
    # ✅ 4. Volatilitäts-Features
    for w in window_sizes:
        features.extend([f'volatility_ma_{w}', f'volatility_spike_{w}'])
    
    # ✅ 5. Wash-Trading Detection
    for w in window_sizes:
        features.append(f'wash_trading_flag_{w}')
    
    # ✅ 6. Net-Volume Features
    for w in window_sizes:
        features.extend([f'net_volume_ma_{w}', f'volume_flip_{w}'])
    
    # ✅ 7. Price Momentum
    for w in window_sizes:
        features.extend([f'price_change_{w}', f'price_roc_{w}'])
    
    # ✅ 8. Volume Patterns
    for w in window_sizes:
        features.extend([f'volume_ratio_{w}', f'volume_spike_{w}'])
    
    # ✅ 9. Market Cap Velocity
    for w in window_sizes:
        features.append(f'mcap_velocity_{w}')
    
    # 🆕 10. ATH-basierte Rolling-Window Features
    # Hinweis: ath_distance_pct, ath_breakout, ath_age_hours, ath_is_recent, ath_is_old
    # werden bereits in add_ath_features() erstellt und sind nicht in dieser Liste

    # ATH Rolling-Window Features
    for w in window_sizes:
        features.extend([
            f'ath_distance_trend_{w}',
            f'ath_approach_{w}',
            f'ath_breakout_count_{w}',
            f'ath_breakout_volume_ma_{w}',
            f'ath_age_trend_{w}'
        ])
    
    return features

# Kritische Features für Rug-Detection
CRITICAL_FEATURES = [
    "dev_sold_amount",  # KRITISCH: Rug-Pull-Indikator
    "buy_pressure_ratio",  # Relatives Buy/Sell-Verhältnis
    "unique_signer_ratio",  # Wash-Trading-Erkennung
    "whale_buy_volume_sol",
    "whale_sell_volume_sol",
    "net_volume_sol",
    "volatility_pct",
    "ath_distance_pct",  # 🆕 KRITISCH: Historische ATH-Distance (Data Leakage-frei)
    "ath_breakout",  # 🆕 KRITISCH: Historische ATH-Breakouts
    "minutes_since_ath"  # 🆕 KRITISCH: Zeit seit letztem ATH
]

def get_available_ath_features(include_ath: bool = True) -> list:
    """
    Gibt die verfügbaren ATH-Features zurück (wenn ATH aktiviert ist).

    Args:
        include_ath: Ob ATH-Features berechnet wurden

    Returns:
        Liste der verfügbaren ATH-Features
    """
    if not include_ath:
        return []

    return [
        'rolling_ath',
        'ath_distance_pct',
        'ath_breakout',
        'minutes_since_ath',
        'ath_age_hours',
        'ath_is_recent',
        'ath_is_old'
    ]

def validate_critical_features(features: List[str]) -> Dict[str, bool]:
    """
    Prüft ob kritische Features verwendet werden.
    
    Args:
        features: Liste der Feature-Namen
    
    Returns:
        Dict mit {feature_name: bool} - True wenn Feature vorhanden
    """
    return {
        feature: feature in features 
        for feature in CRITICAL_FEATURES
    }


async def validate_ath_data_availability(
    train_start: datetime | str,
    train_end: datetime | str
) -> Dict[str, Any]:
    """
    Prüft ob ATH-Daten für den Zeitraum verfügbar sind.
    
    Args:
        train_start: Start-Zeitpunkt
        train_end: Ende-Zeitpunkt
    
    Returns:
        Dict mit:
        - available: bool
        - coins_with_ath: int
        - coins_without_ath: int
        - coverage_pct: float
        - total_coins: int
    """
    pool = await get_pool()
    
    # Konvertiere zu UTC
    train_start_utc = _ensure_utc(train_start)
    train_end_utc = _ensure_utc(train_end)
    
    # Prüfe wie viele Coins ATH-Daten haben
    query = """
        SELECT 
            COUNT(DISTINCT cm.mint) as total_coins,
            COUNT(DISTINCT CASE WHEN COALESCE(cs.ath_price_sol, 0) > 0 THEN cm.mint END) as coins_with_ath,
            COUNT(DISTINCT CASE WHEN COALESCE(cs.ath_price_sol, 0) = 0 OR cs.ath_price_sol IS NULL THEN cm.mint END) as coins_without_ath
        FROM coin_metrics cm
        LEFT JOIN coin_streams cs ON cm.mint = cs.token_address
        WHERE cm.timestamp >= $1 AND cm.timestamp <= $2
    """
    
    try:
        row = await pool.fetchrow(query, train_start_utc, train_end_utc)
        
        total_coins = row['total_coins'] or 0
        coins_with_ath = row['coins_with_ath'] or 0
        coins_without_ath = row['coins_without_ath'] or 0
        
        coverage_pct = (coins_with_ath / total_coins * 100) if total_coins > 0 else 0.0
        
        return {
            "available": coins_with_ath > 0,
            "coins_with_ath": coins_with_ath,
            "coins_without_ath": coins_without_ath,
            "coverage_pct": coverage_pct,
            "total_coins": total_coins
        }
    except Exception as e:
        logger.warning(f"⚠️ Fehler bei ATH-Daten-Validierung: {e}")
        return {
            "available": False,
            "coins_with_ath": 0,
            "coins_without_ath": 0,
            "coverage_pct": 0.0,
            "total_coins": 0,
            "error": str(e)
        }


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

