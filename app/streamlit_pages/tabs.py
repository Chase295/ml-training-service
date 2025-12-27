"""
Tabs Page Module
Extrahierte Seite aus streamlit_app.py
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import httpx
import time
import os
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any, Optional

# Import aus streamlit_utils
from streamlit_utils import (
    api_get, api_post, api_delete, api_patch,
    AVAILABLE_FEATURES, FEATURE_CATEGORIES, CRITICAL_FEATURES,
    API_BASE_URL, load_phases, load_config, save_config,
    get_default_config, validate_url, validate_port,
    reload_config, restart_service, get_service_logs
)


def tab_dashboard():
    """Dashboard Tab"""
    st.title("📊 Dashboard")
    
    # Health Status
    health = api_get("/api/health")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if health:
            status = "🟢 Healthy" if health.get("status") == "healthy" else "🔴 Degraded"
            st.metric("Status", status)
        else:
            st.metric("Status", "❌ Nicht erreichbar")
    
    with col2:
        if health:
            st.metric("Jobs verarbeitet", health.get("total_jobs_processed", 0))
        else:
            st.metric("Jobs verarbeitet", "-")
    
    with col3:
        if health:
            db_status = "✅ Verbunden" if health.get("db_connected") else "❌ Getrennt"
            st.metric("Datenbank", db_status)
        else:
            st.metric("Datenbank", "-")
    
    with col4:
        if health:
            uptime = health.get("uptime_seconds", 0)
            hours = uptime // 3600
            minutes = (uptime % 3600) // 60
            st.metric("Uptime", f"{int(hours)}h {int(minutes)}m")
        else:
            st.metric("Uptime", "-")
    
    # Modelle-Übersicht
    st.subheader("📋 Modelle-Übersicht")
    models = api_get("/api/models")
    if models:
        st.info(f"📊 {len(models)} Modell(e) gefunden")
    else:
        st.info("ℹ️ Keine Modelle gefunden")
    
    # Service-Management
    st.subheader("🔧 Service-Management")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🔄 Konfiguration neu laden", type="primary"):
            with st.spinner("Konfiguration wird neu geladen..."):
                success, message = reload_config()
                if success:
                    st.success(message)
                    time.sleep(2)
                else:
                    st.error(message)
                st.rerun()
    
    with col2:
        if st.button("🔄 Seite aktualisieren"):
            st.rerun()
    
    # Auto-Refresh - ohne time.sleep() um UI nicht zu blockieren
    auto_refresh_enabled = st.checkbox("🔄 Auto-Refresh (5s)", key="auto_refresh_dashboard")
    if auto_refresh_enabled:
        # Verwende st.empty() und st.rerun() ohne time.sleep() - Streamlit wird automatisch neu rendern
        placeholder = st.empty()
        placeholder.info("⏳ Auto-Refresh aktiv...")
        st.rerun()
    


def tab_configuration():
    """Konfiguration Tab"""
    st.title("⚙️ Konfiguration")
    
    try:
        config = load_config()
    except Exception as e:
        st.error(f"⚠️ Fehler beim Laden der Config: {e}")
        config = get_default_config()
    
    using_env_vars = bool(os.getenv('DB_DSN'))
    
    if using_env_vars:
        st.info("🌐 **Coolify-Modus erkannt:** Environment Variables haben Priorität, aber du kannst die Konfiguration trotzdem hier speichern.")
    else:
        st.info("💡 Änderungen werden in der Konfigurationsdatei gespeichert. Nutze den 'Konfiguration neu laden' Button, um Änderungen ohne Neustart zu übernehmen.")
    
    with st.form("config_form"):
        st.subheader("🗄️ Datenbank Einstellungen")
        config["DB_DSN"] = st.text_input("DB DSN", value=config.get("DB_DSN", ""), help="PostgreSQL Connection String")
        if config["DB_DSN"]:
            db_valid, db_error = validate_url(config["DB_DSN"], allow_empty=False)
            if not db_valid:
                st.error(f"❌ {db_error}")
        
        st.subheader("🔌 Port Einstellungen")
        config["API_PORT"] = st.number_input("API Port", min_value=1, max_value=65535, value=int(config.get("API_PORT", 8000)))
        config["STREAMLIT_PORT"] = st.number_input("Streamlit Port", min_value=1, max_value=65535, value=int(config.get("STREAMLIT_PORT", 8501)))
        
        st.subheader("📁 Pfad Einstellungen")
        config["MODEL_STORAGE_PATH"] = st.text_input("Model Storage Path", value=config.get("MODEL_STORAGE_PATH", "/app/models"))
        config["API_BASE_URL"] = st.text_input("API Base URL", value=config.get("API_BASE_URL", "http://localhost:8000"), help="Innerhalb des Containers: localhost:8000, von außen: localhost:8012")
        if config["API_BASE_URL"]:
            api_valid, api_error = validate_url(config["API_BASE_URL"], allow_empty=False)
            if not api_valid:
                st.error(f"❌ {api_error}")
        
        st.subheader("⚙️ Job Queue Einstellungen")
        config["JOB_POLL_INTERVAL"] = st.number_input("Job Poll Interval (Sekunden)", min_value=1, max_value=300, value=int(config.get("JOB_POLL_INTERVAL", 5)))
        config["MAX_CONCURRENT_JOBS"] = st.number_input("Max Concurrent Jobs", min_value=1, max_value=10, value=int(config.get("MAX_CONCURRENT_JOBS", 2)))
        
        st.subheader("📝 Logging Einstellungen")
        config["LOG_LEVEL"] = st.selectbox("Log Level", ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], index=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"].index(config.get("LOG_LEVEL", "INFO")))
        config["LOG_FORMAT"] = st.selectbox("Log Format", ["text", "json"], index=["text", "json"].index(config.get("LOG_FORMAT", "text")))
        config["LOG_JSON_INDENT"] = st.number_input("Log JSON Indent", min_value=0, max_value=4, value=config.get("LOG_JSON_INDENT", 0))
        
        col1, col2 = st.columns(2)
        with col1:
            save_button = st.form_submit_button("💾 Konfiguration speichern", type="primary")
        with col2:
            reset_button = st.form_submit_button("🔄 Auf Standard zurücksetzen")
        
        if save_button:
            errors = []
            
            db_valid, db_error = validate_url(config["DB_DSN"], allow_empty=False)
            if not db_valid:
                errors.append(f"DB DSN: {db_error}")
            
            api_valid, api_error = validate_url(config["API_BASE_URL"], allow_empty=False)
            if not api_valid:
                errors.append(f"API Base URL: {api_error}")
            
            if errors:
                st.error("❌ **Validierungsfehler:**")
                for error in errors:
                    st.error(f"  - {error}")
            else:
                try:
                    result = save_config(config)
                    if result:
                        st.session_state.config_saved = True
                        st.success("✅ Konfiguration gespeichert!")
                        if using_env_vars:
                            st.info("💡 **Tipp:** Nutze den 'Konfiguration neu laden' Button unten, um die Änderungen ohne Neustart zu übernehmen.")
                        else:
                            st.info("💡 **Tipp:** Nutze den 'Konfiguration neu laden' Button unten, um die Änderungen ohne Neustart zu übernehmen.")
                        st.session_state.config_just_saved = True
                except Exception as e:
                    st.error(f"❌ **Fehler beim Speichern:** {e}")
        
        if reset_button:
            try:
                default_config = get_default_config()
                if save_config(default_config):
                    st.session_state.config_saved = True
                    st.success("✅ Konfiguration auf Standard zurückgesetzt!")
                    st.warning("⚠️ Bitte Service neu starten oder 'Konfiguration neu laden' Button unten verwenden!")
                    st.session_state.config_just_saved = True
            except Exception as e:
                st.error(f"❌ **Fehler beim Zurücksetzen:** {e}")
    
    # Reload-Button
    st.divider()
    st.subheader("🔄 Konfiguration neu laden")
    st.caption("Lädt die gespeicherte Konfiguration im Service neu (ohne Neustart)")
    if st.button("🔄 Konfiguration neu laden", type="primary", key="reload_config_button"):
        with st.spinner("Konfiguration wird neu geladen..."):
            success, message = reload_config()
            if success:
                st.success(f"✅ {message}")
                st.info("💡 Die neue Konfiguration ist jetzt aktiv! Kein Neustart nötig.")
            else:
                st.error(f"❌ {message}")
                st.info("💡 Falls der Reload fehlschlägt, starte den Service manuell neu.")
    
    # Neustart-Button
    if st.session_state.get("config_saved", False):
        st.divider()
        st.subheader("🔄 Service-Neustart")
        col1, col2 = st.columns([2, 1])
        with col1:
            st.info("💡 Die Konfiguration wurde gespeichert. Starte den Service neu, damit die neuen Werte geladen werden.")
        with col2:
            if st.button("🔄 Service neu starten", type="primary", use_container_width=True):
                with st.spinner("Service wird neu gestartet..."):
                    success, message = restart_service()
                    if success:
                        st.success(message)
                        st.info("⏳ Bitte warte 5-10 Sekunden, bis der Service vollständig neu gestartet ist.")
                        st.session_state.config_saved = False
                        # Kein automatisches Rerun - User kann manuell aktualisieren
                    else:
                        st.error(message)
                        st.info("💡 Du kannst den Service auch manuell neu starten: `docker compose restart ml-training`")
    
    # Info nach Speichern (ohne Auto-Rerun)
    if st.session_state.get("config_just_saved", False):
        st.session_state.config_just_saved = False
        st.info("💡 Konfiguration wurde gespeichert! Verwende 'Konfiguration neu laden' um Änderungen zu aktivieren.")
    
    # Aktuelle Konfiguration anzeigen
    st.subheader("📄 Aktuelle Konfiguration")
    st.json(config)



def tab_logs():
    """Logs Tab"""
    st.title("📋 Service Logs")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        lines = st.number_input("Anzahl Zeilen", min_value=10, max_value=1000, value=100, step=10, key="logs_lines_input")
    
    with col2:
        refresh_logs = st.button("🔄 Logs aktualisieren", key="refresh_logs_button")
        if refresh_logs:
            st.rerun()
    
    logs = get_service_logs(lines=lines)
    
    st.text_area(
        "Service Logs (neueste oben)",
        logs,
        height=600,
        key="logs_display",
        help="Die neuesten Logs stehen oben, die ältesten unten."
    )
    
    if not logs or logs.strip() == "":
        st.warning("⚠️ Keine Logs verfügbar. Prüfe ob der Service läuft.")
    
    auto_refresh = st.checkbox("🔄 Auto-Refresh Logs (10s)", key="auto_refresh_logs")
    if auto_refresh:
        # Verwende st.empty() und st.rerun() ohne time.sleep() - Streamlit wird automatisch neu rendern
        placeholder = st.empty()
        placeholder.info("⏳ Auto-Refresh aktiv...")
        st.rerun()



def tab_metrics():
    """Metriken Tab - Vollständige Metriken-Übersicht"""
    st.title("📈 Metriken & Monitoring")

    # Metriken-Erklärung
    st.markdown("""
    **📊 ML-Modell Metriken:**
    - **Accuracy:** Anteil korrekter Vorhersagen (0-1)
    - **Precision:** Anteil korrekter positiver Vorhersagen (0-1)
    - **Recall:** Anteil gefundener positiver Fälle (0-1)
    - **F1-Score:** Harmonisches Mittel von Precision und Recall (0-1)
    - **ROC-AUC:** Area under ROC Curve (0-1, >0.5 ist besser als zufällig)

    **🎯 Rug-Detection Metriken:**
    - **Dev-Sold Rate:** Wie oft wurden Dev-Verkäufe korrekt erkannt?
    - **Wash-Trading Rate:** Wie oft wurde Wash-Trading erkannt?
    - **Weighted Cost:** Berücksichtigt, dass False Negatives 10x teurer sind

    **⏰ Zeitbasierte Metriken:**
    - **Labels Balance:** Verhältnis positive/negative Labels
    - **Simulierter Profit:** ROI-basierte Bewertung der Strategie
    """)

    st.divider()

    if st.button("🔄 Metriken aktualisieren"):
        st.rerun()

    # System Health Metriken
    st.subheader("🏥 System Health")
    health = api_get("/api/health")
    if health:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Status", "🟢 Healthy" if health.get("status") == "healthy" else "🔴 Degraded")
        with col2:
            uptime = health.get("uptime_seconds", 0)
            hours = uptime // 3600
            minutes = (uptime % 3600) // 60
            st.metric("Uptime", f"{int(hours)}h {int(minutes)}m")
        with col3:
            st.metric("Jobs verarbeitet", health.get("total_jobs_processed", 0))

    # Prometheus Metrics
    st.subheader("📊 Prometheus Monitoring")
    try:
        response = httpx.get(f"{API_BASE_URL}/api/metrics", timeout=5)
        if response.status_code == 200:
            metrics = response.text

            # Parse Metriken nach Kategorien
            system_metrics = {}
            job_metrics = {}
            api_metrics = {}

            for line in metrics.split('\n'):
                if line and not line.startswith('#'):
                    parts = line.split()
                    if len(parts) >= 2:
                        metric_name = parts[0]
                        try:
                            metric_value = float(parts[1]) if '.' in parts[1] else int(parts[1])

                            # Kategorisiere Metriken
                            if 'job' in metric_name.lower():
                                job_metrics[metric_name] = metric_value
                            elif 'api' in metric_name.lower() or 'http' in metric_name.lower():
                                api_metrics[metric_name] = metric_value
                            else:
                                system_metrics[metric_name] = metric_value

                        except:
                            system_metrics[metric_name] = parts[1]

            # Zeige Metriken nach Kategorien
            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown("**🖥️ System Metriken**")
                if system_metrics:
                    st.json(system_metrics)
                else:
                    st.info("Keine System-Metriken verfügbar")

            with col2:
                st.markdown("**⚙️ Job Metriken**")
                if job_metrics:
                    st.json(job_metrics)
                else:
                    st.info("Keine Job-Metriken verfügbar")

            with col3:
                st.markdown("**🌐 API Metriken**")
                if api_metrics:
                    st.json(api_metrics)
                else:
                    st.info("Keine API-Metriken verfügbar")

            # Raw Metriken (expandable)
            with st.expander("📄 Raw Prometheus Metriken"):
                st.code(metrics, language="text")

        else:
            st.error(f"❌ Fehler beim Abrufen der Metriken: HTTP {response.status_code}")
    except Exception as e:
        st.error(f"❌ Fehler beim Abrufen der Metriken: {str(e)}")

    st.divider()

    # Modell-Metriken Übersicht
    st.subheader("🤖 Aktuelle Modelle")
    models = api_get("/api/models")
    if models:
        # Erstelle Übersicht der Metriken
        metrics_data = []
        for model in models[:10]:  # Zeige nur die letzten 10
            metrics_data.append({
                "Name": model.get("name", "N/A"),
                "Status": model.get("status", "N/A"),
                "Accuracy": f"{model.get('training_accuracy', 0):.3f}",
                "F1-Score": f"{model.get('training_f1', 0):.3f}",
                "Features": len(model.get('features', []))
            })

        if metrics_data:
            st.dataframe(metrics_data, use_container_width=True)
    else:
        st.info("Keine Modelle gefunden")

    auto_refresh_metrics = st.checkbox("🔄 Auto-Refresh Metriken (10s)", key="auto_refresh_metrics")
    if auto_refresh_metrics:
        import time
        time.sleep(10)
        st.rerun()



def tab_info():
    """Info Tab - Vollständige Projekt-Informationen"""
    st.title("ℹ️ Projekt-Informationen")
    
    # Projekt-Übersicht
    st.header("📋 Was macht dieses Projekt?")
    st.markdown("""
    **ML Training Service** ist ein Machine-Learning-Service für Kryptowährungs-Datenanalyse.
    
    Das System:
    - ✅ Trainiert ML-Modelle (Random Forest, XGBoost) für Pump-Detection
    - ✅ Unterstützt zeitbasierte Vorhersagen ("Steigt in 10 Min um 30%")
    - ✅ Feature-Engineering für bessere Performance
    - ✅ Verwaltet Trainings-Jobs in einer asynchronen Queue
    - ✅ Testet Modelle auf neuen Daten
    - ✅ Vergleicht Modelle miteinander
    - ✅ Speichert Modelle persistent (Datenbank + .pkl Dateien)
    - ✅ Bietet eine Web-UI für Monitoring und Konfiguration
    - ✅ Exportiert Prometheus-Metriken für Monitoring
    """)
    
    st.divider()
    
    # Datenfluss
    st.header("🔄 Datenfluss")
    st.code("""
    PostgreSQL Datenbank
            ├─ coin_metrics (OHLCV + Metriken)
            ├─ coin_streams (aktuelle ATH-Werte für historische Berechnung)
            ├─ exchange_rates (SOL-Preis, Marktstimmung)
            └─ ref_coin_phases (Phasen-Konfiguration)
            ↓
    FastAPI Service (app/main.py)
            ├─ API-Endpunkte (/api/models/create, /api/models/test, etc.)
            ├─ Job-Queue (ml_jobs Tabelle)
            └─ Asynchroner Worker (verarbeitet Jobs)
            ↓
    Training Engine (app/training/engine.py)
            ├─ Lädt Daten aus coin_metrics
            ├─ 🆕 Data Cleaning (entfernt "tote" Daten ohne Trades)
            ├─ 🆕 Historische ATH-Berechnung (Data Leakage-frei)
            ├─ Feature-Engineering (optional, ~70+ Features inkl. ATH)
            ├─ Marktstimmung-Enrichment (optional)
            ├─ Label-Erstellung (zeitbasiert oder klassisch)
            ├─ Modell-Training (Random Forest / XGBoost)
            └─ Speichert Modell (.pkl + Metadaten in DB)
            ↓
    ml_models Tabelle
            ├─ Modell-Metadaten
            ├─ Performance-Metriken
            ├─ Feature Importance
            └─ Verweis auf .pkl Datei
    """, language="text")
    
    st.divider()

    # 🆕 Data Cleaning - Garbage In, Garbage Out verhindern
    st.header("🧹 Data Cleaning - Garbage In, Garbage Out verhindern")

    st.markdown("""
    **Problem:** Die Datenbank enthält "tote" Coins ohne Trades oder nur unvollständige Daten.
    Das führt zu KI-Abstürzen oder falschen Mustern.

    **Lösung:** Automatische Filterung beim Daten-Laden:
    """)

    st.subheader("1️⃣ NULL-Wert Filter")
    st.markdown("""
    **Entfernt Zeilen mit fehlenden kritischen Daten:**
    - `price_close` (Preis-Information)
    - `volume_sol` (Handelsvolumen)
    - ATH-Features (falls aktiviert)

    **Warum?** Coins ohne Preisverlauf können kein sinnvolles Muster zeigen.
    """)

    st.subheader("2️⃣ Coin-Alter Filter")
    st.markdown("""
    **Entfernt Coins mit zu wenigen Datenpunkten:**
    - Mindestens 30 Datenpunkte pro Coin
    - Filtert Rauschen und unvollständige Historien heraus

    **Warum?** Ein Coin mit nur 3 Datenpunkten kann keine stabilen Muster zeigen.
    """)

    st.subheader("3️⃣ ATH-Data-Validation")
    st.markdown("""
    **Bei ATH-aktivierten Modellen:**
    - Zusätzliche Validierung der historisch berechneten ATH-Features
    - Entfernt Zeilen mit ungültigen ATH-Berechnungen

    **Effekt:** Saubere, vollständige Daten für bessere KI-Performance.
    """)

    st.divider()

    # Was macht das System genau?
    st.header("🔍 Was macht das System genau?")
    
    st.subheader("1️⃣ Modell-Training")
    st.markdown("""
    **Prozess:**
    1. **Job-Erstellung:** Benutzer erstellt Training-Job über Web-UI oder API
    2. **Job-Queue:** Job wird in `ml_jobs` Tabelle mit Status `PENDING` gespeichert
    3. **Worker-Verarbeitung:** Asynchroner Worker findet Job und startet Training
    4. **Daten-Laden:** System lädt Daten aus `coin_metrics` für den gewählten Zeitraum
    5. **🆕 Data Cleaning:** Automatische Filterung "toter" Daten (NULL-Werte, unvollständige Coins)
    6. **🆕 Historische ATH-Berechnung:** Data Leakage-freie ATH-Features werden erstellt
    7. **Feature-Engineering:** Optional werden ~80+ zusätzliche Features erstellt
    8. **Marktstimmung:** Optional wird SOL-Preis-Kontext aus `exchange_rates` hinzugefügt
    9. **Label-Erstellung:** Labels werden erstellt (zeitbasiert oder klassisch)
    10. **Training:** Modell wird trainiert (Random Forest oder XGBoost)
    11. **Evaluation:** Modell wird auf Test-Set evaluiert
    12. **Speicherung:** Modell wird als .pkl Datei gespeichert + Metadaten in DB
    """)
    
    st.subheader("2️⃣ Modell-Testing")
    st.markdown("""
    **Prozess:**
    1. **Test-Job erstellen:** Benutzer wählt Modell und Test-Zeitraum
    2. **Daten-Laden:** System lädt Test-Daten aus `coin_metrics`
    3. **Vorhersagen:** Modell macht Vorhersagen auf Test-Daten
    4. **Evaluation:** Metriken werden berechnet (Accuracy, Precision, Recall, F1, etc.)
    5. **Speicherung:** Test-Ergebnisse werden in `ml_test_results` gespeichert
    """)
    
    st.subheader("3️⃣ Modell-Vergleich")
    st.markdown("""
    **Prozess:**
    1. **Vergleichs-Job erstellen:** Benutzer wählt 2 Modelle und Test-Zeitraum
    2. **Parallele Tests:** Beide Modelle werden auf denselben Daten getestet
    3. **Metriken-Vergleich:** Alle Metriken werden verglichen
    4. **Speicherung:** Vergleichs-Ergebnisse werden in `ml_comparisons` gespeichert
    """)
    
    st.divider()
    
    # Welche Informationen werden verwendet?
    st.header("📤 Welche Informationen werden verwendet?")
    
    st.subheader("1️⃣ Basis-Daten aus coin_metrics")
    st.markdown("""
    **OHLCV-Daten:**
    - `price_open`, `price_high`, `price_low`, `price_close` - Preis-Daten
    - `volume_sol`, `buy_volume_sol`, `sell_volume_sol` - Volumen-Daten
    - `market_cap_close` - Market Cap
    
    **Rug-Detection-Metriken:**
    - `dev_sold_amount` - ⚠️ **KRITISCH:** Wie viel SOL hat der Dev verkauft?
    - `buy_pressure_ratio` - Verhältnis Buy- zu Sell-Volumen
    - `unique_signer_ratio` - Anteil einzigartiger Trader
    - `whale_buy_volume_sol`, `whale_sell_volume_sol` - Whale-Aktivität
    - `num_whale_buys`, `num_whale_sells` - Anzahl Whale-Transaktionen
    - `net_volume_sol` - Netto-Volumen (Buy - Sell)
    - `volatility_pct` - Volatilität in Prozent
    - `avg_trade_size_sol` - Durchschnittliche Trade-Größe
    
    **🆕 ATH-Tracking (Data Leakage-frei, historisch korrekt):**
    - `rolling_ath` - Historisches All-Time-High bis zu jedem Zeitpunkt
    - `ath_distance_pct` - Wie weit entfernt vom historischen ATH? (-100% = am ATH, +100% = tief gefallen)
    - `ath_breakout` - Neue ATH-Breakouts (1 = historisches ATH erreicht/gebrochen)
    - `minutes_since_ath` - Minuten seit letztem historischen ATH-Breakout
    - `ath_age_hours` - Alter des ATH in Stunden
    - `ath_is_recent` - ATH wurde innerhalb 1 Stunde erreicht
    - `ath_is_old` - ATH ist älter als 24 Stunden
    
    **Zeitstempel & Phasen:**
    - `timestamp` - Zeitstempel der Metrik
    - `phase_id_at_time` - Phase des Coins zu diesem Zeitpunkt
    """)
    
    st.subheader("2️⃣ Marktstimmung aus exchange_rates")
    st.markdown("""
    **Markt-Kontext (optional, wenn `use_market_context=True`):**
    - `sol_price_usd` - SOL-Preis in USD (der "Wasserstand")
    - `usd_to_eur_rate` - Währungsumrechnung
    
    **Berechnete Features:**
    - `sol_price_change_pct` - SOL-Preis-Änderung in Prozent
    - `sol_price_ma_5` - 5-Perioden Moving Average des SOL-Preises
    - `sol_price_volatility` - Volatilität des SOL-Preises
    
    **Zweck:** Unterscheidung zwischen echten Token-Pumps und allgemeinen Marktbewegungen
    """)
    
    st.subheader("3️⃣ Feature-Engineering (optional)")
    st.markdown("""
    **Wenn `use_engineered_features=True`:**

    **🆕 Historische ATH-Features (Data Leakage-frei):**
    - Rolling-ATH-Trends (5, 10, 15 Perioden)
    - ATH-Approach-Indikatoren (nähert sich Preis dem ATH?)
    - ATH-Breakout-Häufigkeit
    - ATH-Breakout-Volumen-Trends

    **Momentum-Features:**
    - Price-Momentum (5, 10, 15 Perioden)
    - Volume-Momentum
    - Rate of Change (ROC)

    **Volumen-Patterns:**
    - Volume-MA-Ratio
    - Buy/Sell-Volumen-Ratio
    - Net-Volumen-Trend
    - Volume-Spikes

    **Whale-Aktivität:**
    - Whale-Buy-Rate
    - Whale-Sell-Rate
    - Whale-Aktivitäts-Trend
    - Whale-Net-Volumen

    **Dev-Tracking (KRITISCH für Rug-Detection):**
    - Dev-Sold-Amount-Trend
    - Dev-Sold-Rate
    - Dev-Sold-Spikes
    - Dev-Sold-Cumsum

    **Volatilität:**
    - Rolling-Volatilität
    - Price-Range-Ratio
    - Volatilität-Spikes

    **Bot-Detection:**
    - Wash-Trading-Flags
    - Unique-Signer-Ratios
    - Buy-Pressure-Trends

    **🆕 Insgesamt:** ~80+ zusätzliche Features werden erstellt
    """)
    
    st.divider()
    
    # Datenbankschema
    st.header("🗄️ Datenbankschema")
    
    st.subheader("Haupttabelle: `ml_models`")
    st.markdown("""
    Speichert alle trainierten Modelle mit Metadaten und Performance-Metriken.
    
    **Wichtige Felder:**
    - `id` - Eindeutige Modell-ID
    - `name` - Modell-Name (eindeutig)
    - `model_type` - "random_forest" oder "xgboost"
    - `status` - "TRAINING", "READY", "FAILED"
    - `target_variable` - Ziel-Variable (z.B. "price_close")
    - `future_minutes` - Bei zeitbasierter Vorhersage: Minuten in die Zukunft
    - `min_percent_change` - Bei zeitbasierter Vorhersage: Mindest-Prozent-Änderung
    - `target_direction` - Bei zeitbasierter Vorhersage: "up" oder "down"
    - `features` - JSONB Array mit Feature-Namen
    - `phases` - JSONB Array mit Coin-Phasen (z.B. [1, 2, 3])
    - `params` - JSONB Object mit Hyperparametern
    - `feature_importance` - JSONB Object mit Feature-Importance-Werten
    - `training_accuracy`, `training_f1`, `training_precision`, `training_recall` - Basis-Metriken
    - `cv_scores` - JSONB Object mit Cross-Validation-Ergebnissen
    - `roc_auc`, `mcc`, `fpr`, `fnr` - Erweiterte Metriken
    - `confusion_matrix` - JSONB Object mit TP, TN, FP, FN
    - `simulated_profit_pct` - Simulierter Profit in Prozent
    - `rug_detection_metrics` - JSONB Object mit Rug-Detection-Metriken
    - `market_context_enabled` - Boolean: Wurde Marktstimmung verwendet?
    - `model_file_path` - Pfad zur .pkl Datei
    """)
    
    st.subheader("Tabelle: `ml_test_results`")
    st.markdown("""
    Speichert Test-Ergebnisse für Modelle auf neuen Daten.
    
    **Wichtige Felder:**
    - `id` - Eindeutige Test-ID
    - `model_id` - Verweis auf ml_models
    - `test_start`, `test_end` - Test-Zeitraum
    - `test_accuracy`, `test_f1`, `test_precision`, `test_recall` - Test-Metriken
    - `confusion_matrix` - JSONB Object mit TP, TN, FP, FN
    - `rug_detection_metrics` - JSONB Object mit Rug-Detection-Metriken
    """)
    
    st.subheader("Tabelle: `ml_comparisons`")
    st.markdown("""
    Speichert Vergleichs-Ergebnisse zwischen 2 Modellen.
    
    **Wichtige Felder:**
    - `id` - Eindeutige Vergleichs-ID
    - `model_a_id`, `model_b_id` - Verweise auf ml_models
    - `test_start`, `test_end` - Test-Zeitraum
    - `winner_model_id` - ID des besseren Modells
    - `comparison_metrics` - JSONB Object mit Vergleichs-Metriken
    """)
    
    st.subheader("Tabelle: `ml_jobs`")
    st.markdown("""
    Verwaltet alle Jobs (Training, Testing, Comparison) in einer Queue.
    
    **Wichtige Felder:**
    - `id` - Eindeutige Job-ID
    - `job_type` - "TRAIN", "TEST", "COMPARE"
    - `status` - "PENDING", "RUNNING", "COMPLETED", "FAILED", "CANCELLED"
    - `progress` - Fortschritt in Prozent (0.0 - 1.0)
    - `progress_msg` - Fortschritts-Nachricht
    - `train_model_type` - Modell-Typ (bei TRAIN-Jobs)
    - `train_features` - JSONB Array mit Features
    - `train_params` - JSONB Object mit Parametern
    - `started_at`, `completed_at` - Zeitstempel
    """)
    
    st.subheader("Referenztabelle: `ref_coin_phases`")
    st.markdown("""
    Definiert Coin-Phasen für Phasen-basiertes Training.
    
    **Phasen:**
    - **Baby Zone** (ID: 1): 0-10 Min, Intervall: 5s
    - **Survival Zone** (ID: 2): 10-60 Min, Intervall: 30s
    - **Mature Zone** (ID: 3): 1-24 Std, Intervall: 60s
    - **Finished** (ID: 99): Ab 24 Std
    - **Graduated** (ID: 100): Graduierte Tokens
    """)
    
    st.divider()
    
    # API Endpoints - DETAILLIERT
    st.header("🔌 API Endpoints - Vollständige Dokumentation")

    st.markdown("""
    **📋 API Übersicht:**
    - **Base URL:** `http://localhost:8012/api`
    - **Authentifizierung:** Keine erforderlich (lokale Entwicklung)
    - **Format:** JSON für Request/Response
    - **Async Processing:** Training-Jobs laufen asynchron
    """)

    st.subheader("🚀 Modell-Training")

    st.markdown("""
    **POST `/api/models/create/simple`**
    *Einfacher Modell-Training-Endpunkt*

    **Request Body:**
    ```json
    {
      "name": "Mein XGBoost Modell",
      "model_type": "xgboost",
      "target": "price_close > 0.05",
      "features": [
        "price_close", "volume_sol", "net_volume_sol",
        "buy_pressure_ratio", "unique_signer_ratio",
        "dev_sold_amount", "volatility_pct",
        "ath_distance_pct", "ath_breakout", "minutes_since_ath"
      ],
      "train_start": "2024-12-01T00:00:00Z",
      "train_end": "2024-12-31T23:59:59Z",
      "description": "Optionale Beschreibung"
    }
    ```

    **Response:**
    ```json
    {
      "job_id": 123,
      "message": "Job erstellt. Modell wird trainiert.",
      "status": "PENDING"
    }
    ```

    **Parameter:**
    - `name` (string): Eindeutiger Modell-Name
    - `model_type` (string): `"random_forest"` oder `"xgboost"`
    - `target` (string): Bedingung wie `"price_close > 0.05"`
    - `features` (array): Liste der Features
    - `train_start/end` (datetime): ISO-Format UTC
    """)

    st.markdown("""
    **POST `/api/models/create`**
    *Erweiterter Modell-Training-Endpunkt*

    **Zusätzliche Parameter:**
    ```json
    {
      "hyperparameters": {"max_depth": 10, "n_estimators": 100},
      "validation_split": 0.2,
      "use_time_based_prediction": false,
      "phases": [1, 2, 3],
      "use_engineered_features": true,
      "use_market_context": true,
      "use_smote": true
    }
    ```

    **🆕 Beispiel: 2. Welle Reversal Hunter**
    *Spezialisiert auf Coins nach ATH-Dip*

    ```json
    {
      "name": "2nd_Wave_Reversal_Hunter_v1",
      "model_type": "xgboost",
      "description": "Spezialisiert auf Reversals nach ATH-Dip (>40%). Ignoriert Baby-Zone.",

      "use_time_based_prediction": true,
      "future_minutes": 15,
      "min_percent_change": 30.0,
      "target_direction": "up",
      "target_var": "price_close",

      "phases": [2, 3],

      "use_engineered_features": true,
      "use_market_context": true,
      "use_smote": true,

      "features": [
        "price_close", "volume_sol", "net_volume_sol",
        "buy_pressure_ratio", "unique_signer_ratio",
        "dev_sold_amount", "volatility_pct", "avg_trade_size_sol",
        "ath_distance_pct", "ath_breakout", "minutes_since_ath"
      ],

      "hyperparameters": {
        "n_estimators": 300,
        "max_depth": 7,
        "learning_rate": 0.05,
        "scale_pos_weight": 5
      }
    }
    ```

    **Warum diese Konfiguration?**
    - **Phasen [2,3]**: Fokussiert auf Survival/Mature Zone (kein Chaos der ersten 10 Min)
    - **15 Min, 30%**: Erkennt echte Reversal-Muster
    - **ATH-Features**: Lernt historische Breakout-Patterns
    - **SMOTE + scale_pos_weight**: Handhabt seltene positive Events
    """)

    st.subheader("📊 Modell-Management")

    st.markdown("""
    **GET `/api/models`**
    *Alle Modelle auflisten*

    **Query Parameter:**
    - `?status=READY` - Nur fertige Modelle
    - `?limit=10` - Begrenze Anzahl

    **Response:**
    ```json
    [
      {
        "id": 1,
        "name": "XGBoost Modell",
        "model_type": "xgboost",
        "status": "READY",
        "accuracy": 0.87,
        "created_at": "2024-12-27T10:00:00Z"
      }
    ]
    ```

    **GET `/api/models/{model_id}`**
    *Modell-Details abrufen*

    **PATCH `/api/models/{model_id}`**
    *Modell aktualisieren*

    ```json
    {"description": "Neue Beschreibung"}
    ```

    **DELETE `/api/models/{model_id}`**
    *Modell löschen (soft delete)*
    """)

    st.subheader("🧪 Modell-Testing")

    st.markdown("""
    **POST `/api/models/test`**
    *Modell testen*

    **Request:**
    ```json
    {
      "model_id": 1,
      "test_start": "2024-12-25T00:00:00Z",
      "test_end": "2024-12-27T00:00:00Z"
    }
    ```

    **GET `/api/test-results`**
    *Alle Test-Ergebnisse*

    **GET `/api/test-results/{test_id}`**
    *Test-Details*
    """)

    st.subheader("⚖️ Modell-Vergleich")

    st.markdown("""
    **POST `/api/models/compare`**
    *Zwei Modelle vergleichen*

    **Request:**
    ```json
    {
      "model_a_id": 1,
      "model_b_id": 2,
      "test_start": "2024-12-25T00:00:00Z",
      "test_end": "2024-12-27T00:00:00Z"
    }
    ```

    **GET `/api/comparisons`**
    *Alle Vergleiche auflisten*

    **GET `/api/comparisons/{comparison_id}`**
    *Vergleichs-Details*
    """)

    st.subheader("⚙️ Job-Management")

    st.markdown("""
    **GET `/api/queue`**
    *Alle Jobs auflisten*

    **Query Parameter:**
    - `?status=RUNNING` - Nur laufende Jobs
    - `?limit=5` - Begrenze Anzahl

    **Response:**
    ```json
    [
      {
        "id": 123,
        "type": "TRAIN",
        "status": "RUNNING",
        "progress": 45.0,
        "created_at": "2024-12-27T10:00:00Z",
        "model_name": "XGBoost Modell"
      }
    ]
    ```

    **GET `/api/queue/{job_id}`**
    *Job-Details mit Logs*
    """)

    st.subheader("🔍 System & Monitoring")

    st.markdown("""
    **GET `/api/health`**
    *Health Check*

    **Response:**
    ```json
    {
      "status": "healthy",
      "db_connected": true,
      "uptime_seconds": 3600,
      "start_time": 1735290000.0,
      "total_jobs_processed": 15,
      "last_error": null
    }
    ```

    **GET `/api/metrics`**
    *Prometheus-Metriken (Text-Format)*

    **Verfügbare Metriken:**
    - `ml_jobs_total` - Gesamtanzahl Jobs
    - `ml_jobs_active` - Aktuell laufende Jobs
    - `ml_jobs_completed` - Abgeschlossene Jobs
    - `ml_jobs_failed` - Fehlgeschlagene Jobs
    - `ml_api_requests_total` - Gesamtanzahl API-Requests
    - `ml_api_requests_duration_seconds` - API-Response-Zeit
    - `ml_models_total` - Gesamtanzahl gespeicherte Modelle
    - `ml_training_duration_seconds` - Training-Dauer

    **GET `/api/data-availability`**
    *Verfügbare Daten-Zeiträume*

    **Response:**
    ```json
    {
      "earliest": "2024-01-01T00:00:00Z",
      "latest": "2024-12-27T12:00:00Z",
      "total_hours": 8736,
      "phases_available": [1, 2, 3, 4, 5]
    }
    ```

    **POST `/api/reload-config`**
    *Konfiguration neu laden (ohne Neustart)*

    **Response:**
    ```json
    {
      "success": true,
      "message": "Konfiguration wurde neu geladen"
    }
    ```
    """)

    st.subheader("📋 API-Workflow")

    st.markdown("""
    **Typischer Ablauf:**

    **1. Datenverfügbarkeit prüfen:**
    ```bash
    curl http://localhost:8012/api/data-availability
    ```

    **2. Modell trainieren:**
    ```bash
    curl -X POST http://localhost:8012/api/models/create/simple \\
      -H "Content-Type: application/json" \\
      -d '{"name": "Test", "model_type": "xgboost", ...}'
    ```

    **3. Job-Status überwachen:**
    ```bash
    curl http://localhost:8012/api/queue/123
    ```

    **4. Modell testen:**
    ```bash
    curl -X POST http://localhost:8012/api/models/test \\
      -H "Content-Type: application/json" \\
      -d '{"model_id": 1, ...}'
    ```

    **5. Modelle vergleichen:**
    ```bash
    curl -X POST http://localhost:8012/api/models/compare \\
      -H "Content-Type: application/json" \\
      -d '{"model_a_id": 1, "model_b_id": 2, ...}'
    ```
    """)

    st.subheader("⚠️ Error Handling")

    st.markdown("""
    **HTTP Status Codes:**
    - `200` - Erfolg
    - `201` - Ressource erstellt (Job)
    - `400` - Bad Request (ungültige Parameter)
    - `404` - Not Found (Modell/Job existiert nicht)
    - `422` - Validation Error (z.B. ungültige Target-Bedingung)
    - `500` - Internal Server Error

    **Error Response:**
    ```json
    {
      "detail": "Beschreibung des Fehlers",
      "error_code": "VALIDATION_ERROR"
    }
    ```

    **Job-Fehler überprüfen:**
    ```bash
    curl http://localhost:8012/api/queue/123
    # Schauen nach "error_message" Feld
    ```
    """)
    
    st.divider()
    
    # Technische Details
    st.header("🔧 Technische Details")
    
    st.subheader("Services")
    st.markdown("""
    - **FastAPI Service** (`app/main.py`): API-Endpunkte, Job-Queue, Health-Checks, Prometheus-Metriken
    - **Streamlit UI** (`app/streamlit_app.py`): Web-Interface für Monitoring und Konfiguration
    - **Job Manager** (`app/queue/job_manager.py`): Asynchroner Worker für Job-Verarbeitung
    - **Training Engine** (`app/training/engine.py`): Modell-Training-Logik
    - **Feature Engineering** (`app/training/feature_engineering.py`): Feature-Erstellung
    - **Model Loader** (`app/training/model_loader.py`): Modell-Laden und Testing
    """)
    
    st.subheader("Ports")
    st.markdown("""
    **Externe Ports (Docker Host):**
    - **API**: Port `8012` (FastAPI)
    - **Web UI**: Port `8502` (Streamlit)
    
    **Interne Ports (Docker Container):**
    - **API**: Port `8000` (FastAPI)
    - **Web UI**: Port `8501` (Streamlit)
    """)
    
    st.subheader("Job-System")
    st.markdown("""
    **Asynchrones Job-System:**
    - Jobs werden in `ml_jobs` Tabelle gespeichert
    - Worker prüft alle 5 Sekunden auf neue `PENDING` Jobs
    - Max. 2 Jobs parallel (konfigurierbar über `MAX_CONCURRENT_JOBS`)
    - Training läuft in separatem Thread (blockiert nicht Event Loop)
    - Progress wird kontinuierlich aktualisiert
    """)
    
    st.subheader("Modell-Typen")
    st.markdown("""
    **Random Forest:**
    - Ensemble-Methode mit mehreren Decision Trees
    - Robust gegen Overfitting
    - Gute Performance auf imbalanced Data

    **XGBoost:**
    - Gradient Boosting Framework
    - Sehr gute Performance
    - Unterstützt Feature Importance
    """)

    st.subheader("🚀 Coolify Deployment")
    st.markdown("""
    **System ist vollständig Coolify-kompatibel:**

    **✅ Datenbank:** PostgreSQL mit allen erforderlichen Tabellen
    **✅ Services:** FastAPI + Streamlit in separaten Containern
    **✅ Konfiguration:** Environment-Variablen für DB-Verbindung
    **✅ Health Checks:** Automatische Überwachung verfügbar
    **✅ Logging:** Vollständige Log-Aggregation
    **✅ API:** RESTful Endpunkte für alle Funktionen

    **Deployment-Checkliste:**
    - PostgreSQL-Datenbank in Coolify einrichten
    - Environment-Variablen konfigurieren
    - Docker-Images deployen
    - Reverse Proxy für API + UI einrichten
    - Health Checks aktivieren

    **Monitoring:**
    - Prometheus-Metriken unter `/api/metrics`
    - Health-Status unter `/api/health`
    - Job-Queue Monitoring über Web-UI
    """)
    
    st.divider()

    # 🆕 Wichtige neue Features
    st.header("🚀 Wichtige neue Features")

    st.subheader("1️⃣ Historische ATH-Berechnung (Data Leakage-frei)")
    st.markdown("""
    **Problem behoben:** Statische ATH-Werte aus Datenbank führten zu Data Leakage.

    **Neue Features:**
    - `rolling_ath`: Historisches All-Time-High bis zu jedem Zeitpunkt
    - `ath_distance_pct`: Prozentuale Entfernung vom historischen ATH
    - `ath_breakout`: Erkennt neue historische ATH-Breakouts
    - `minutes_since_ath`: Zeit seit letztem historischen ATH

    **Vorteil:** KI lernt echte historische Muster statt zukünftiges Wissen zu nutzen.
    """)

    st.subheader("2️⃣ Verbesserte Data Cleaning")
    st.markdown("""
    **Automatische Filterung:**
    - Entfernt Coins ohne Trades (NULL-Werte)
    - Filtert unvollständige Coin-Historien (< 30 Datenpunkte)
    - Validiert ATH-Features

    **Ergebnis:** Saubere Daten = Bessere Modell-Performance.
    """)

    st.subheader("3️⃣ Erweiterte Feature-Engineering")
    st.markdown("""
    **Neue Kategorien:**
    - Rolling-ATH-Trends (5, 10, 15 Perioden)
    - Dev-Sold-Spike-Detection
    - Wash-Trading-Erkennung
    - Volume-Spike-Analysen

    **Gesamt:** ~80+ Features statt ~70.
    """)

    st.subheader("4️⃣ Verbesserte Rug-Detection")
    st.markdown("""
    **Kritische Indikatoren:**
    - `dev_sold_amount`: Wie viel SOL hat der Dev verkauft?
    - `unique_signer_ratio`: Bot vs. echte Trader
    - `buy_pressure_ratio`: Echtes Kaufinteresse

    **Integration:** Alle Features werden automatisch in neue Modelle aufgenommen.
    """)

    st.divider()

    # Zeitbasierte Vorhersagen
    st.header("⏰ Zeitbasierte Vorhersagen")
    
    st.markdown("""
    **Konzept:**
    Statt zu fragen "Ist price_close > 50000?", fragt das System:
    "Steigt price_close in 10 Minuten um mindestens 30%?"
    
    **Parameter:**
    - `future_minutes`: Anzahl Minuten in die Zukunft (z.B. 10)
    - `min_percent_change`: Mindest-Prozent-Änderung (z.B. 30.0 für 30%)
    - `direction`: "up" (steigt) oder "down" (fällt)
    
    **Label-Erstellung:**
    1. Für jeden Zeitpunkt wird der aktuelle Wert (`price_close`) genommen
    2. Der Wert nach `future_minutes` Minuten wird berechnet
    3. Prozent-Änderung wird berechnet: `((future - current) / current) * 100`
    4. Label = 1 wenn Änderung >= `min_percent_change` (bei "up") oder <= -`min_percent_change` (bei "down")
    5. Label = 0 sonst
    
    **Vorteile:**
    - Realistischere Vorhersagen (zeitbasiert)
    - Besser für Trading-Strategien geeignet
    - Berücksichtigt Zeit-Komponente
    """)
    
    st.divider()
    
    # Feature-Engineering
    st.header("🎯 Feature-Engineering")
    
    st.markdown("""
    **Aktivierung:** `use_engineered_features=True`
    
    **Erstellte Features:**
    - **Momentum:** Price-Momentum, Volume-Momentum, ROC
    - **Volumen-Patterns:** Volume-MA-Ratio, Buy/Sell-Ratio, Net-Volumen-Trend
    - **Whale-Aktivität:** Whale-Buy-Rate, Whale-Sell-Rate, Whale-Aktivitäts-Trend
    - **Dev-Tracking:** Dev-Sold-Amount-Trend, Dev-Sold-Rate
    - **Volatilität:** Rolling-Volatilität, Price-Range-Ratio
    - **🆕 ATH-basierte Features:** ~30 zusätzliche Features für Breakout-Erkennung
      - `ath_distance_pct` - Abstand vom ATH
      - `is_near_ath`, `is_at_ath` - Nähe zum ATH
      - `ath_breakout` - Neuer ATH erreicht?
      - `ath_approach_{window}` - Nähert sich dem ATH?
      - `ath_age_hours` - Alter des ATH
      - Und viele mehr...
    
    **Fenstergrößen:** Konfigurierbar über `feature_engineering_windows` (z.B. [5, 10, 15])
    
    **Insgesamt:** ~70 zusätzliche Features werden erstellt (inkl. ATH-Features)
    """)
    
    st.divider()
    
    # Marktstimmung
    st.header("📈 Marktstimmung (Market Context)")
    
    st.markdown("""
    **Aktivierung:** `use_market_context=True`
    
    **Zweck:** Unterscheidung zwischen echten Token-Pumps und allgemeinen Marktbewegungen
    
    **Datenquelle:** `exchange_rates` Tabelle (SOL-Preis in USD)
    
    **Erstellte Features:**
    - `sol_price_change_pct` - SOL-Preis-Änderung in Prozent
    - `sol_price_ma_5` - 5-Perioden Moving Average des SOL-Preises
    - `sol_price_volatility` - Volatilität des SOL-Preises
    
    **Beispiele:**
    - **"Token steigt, während SOL stabil ist"** → Bullish (Echter Pump) ✅
    - **"Token steigt, weil SOL um 5% steigt"** → Neutral (Marktbewegung) ⚠️
    - **"Token ist stabil, während SOL crasht"** → Stärke (Relative Strength) 💪
    """)
    
    st.divider()
    
    # ATH-Tracking
    st.header("📈 ATH-Tracking (All-Time High)")
    
    st.markdown("""
    **🆕 NEU:** ATH-Tracking für Breakout-Erkennung und Preis-Momentum-Analyse
    
    **Datenquelle:** `coin_streams` Tabelle (JOIN mit `coin_metrics`)
    
    **🆕 Historische ATH-Metriken (Data Leakage-frei):**
    - `rolling_ath` - Historisches All-Time-High bis zu jedem Zeitpunkt
    - `ath_distance_pct` - Prozentuale Entfernung vom historischen ATH (-100% = am ATH, +100% = tief gefallen)
    - `ath_breakout` - Neue historische ATH-Breakouts (1 = historisches ATH erreicht/gebrochen)
    - `minutes_since_ath` - Minuten seit letztem historischen ATH-Breakout
    - `ath_age_hours` - Alter des ATH in Stunden
    - `ath_is_recent` - ATH wurde innerhalb 1 Stunde erreicht
    - `ath_is_old` - ATH ist älter als 24 Stunden

    **ATH-Rolling-Window Features:**
    - `ath_distance_trend_{5/10/15}` - ATH-Abstands-Trend über Fenster
    - `ath_approach_{5/10/15}` - Nähert sich Preis dem ATH? (Trend-Indikator)
    - `ath_breakout_count_{5/10/15}` - Anzahl ATH-Breakouts im Zeitfenster
    - `ath_breakout_volume_ma_{5/10/15}` - Durchschnittsvolumen bei ATH-Breakouts
    - `ath_age_trend_{5/10/15}` - Trend des ATH-Alters
    
    **Zweck:**
    - **Breakout-Erkennung:** Erkennt wenn Preis neue ATH erreicht
    - **Momentum-Analyse:** Erkennt wenn Preis sich dem ATH nähert
    - **Resistance-Levels:** Identifiziert ATH als Widerstands-Level
    - **Zeitbasierte Vorhersagen:** Wichtig für Pump-Erkennung
    
    **Performance:**
    - ATH-Daten werden über LEFT JOIN geladen (keine Datenverluste)
    - Performance-Indizes optimieren JOIN-Geschwindigkeit
    - 100% Coverage in aktueller Datenbank (alle Coins haben ATH-Daten)
    
    **Wichtig:** ATH-Features sind in `CRITICAL_FEATURES` enthalten (`price_vs_ath_pct`)
    """)
    
    st.divider()
    
    # Rug-Detection-Metriken
    st.header("🚨 Rug-Detection-Metriken")
    
    st.markdown("""
    **Spezielle Metriken für Rug-Pull-Erkennung:**
    
    - **Dev-Sold Detection Rate:** Wie oft wurde `dev_sold_amount > 0` korrekt erkannt?
    - **Wash-Trading Detection Rate:** Wie oft wurde Wash-Trading erkannt?
    - **Weighted Cost:** Kosten-Funktion (False Negatives 10x schwerer als False Positives)
    - **Precision at K:** Precision bei Top-K Vorhersagen
    
    **Kritische Features:**
    - `dev_sold_amount` - ⚠️ **KRITISCH:** Wie viel SOL hat der Dev verkauft?
    - `buy_pressure_ratio` - Verhältnis Buy- zu Sell-Volumen
    - `unique_signer_ratio` - Anteil einzigartiger Trader
    - `ath_distance_pct` - 🆕 **KRITISCH:** Historische ATH-Distance (Data Leakage-frei)
    - `ath_breakout` - 🆕 **KRITISCH:** Historische ATH-Breakouts
    - `minutes_since_ath` - 🆕 **KRITISCH:** Zeit seit letztem ATH
    """)

    st.divider()

    # Data Cleaning
    st.header("🧹 Data Cleaning - Garbage In, Garbage Out verhindern")

    st.markdown("""
    **Warum Data Cleaning kritisch ist:**

    **Das Problem:**
    - Datenbank enthält "tote" Coins ohne Trades (NULL-Werte)
    - Unvollständige Daten führen zu ML-Abstürzen oder falschen Modellen
    - GIGO (Garbage In, Garbage Out): Schlechte Daten = schlechte KI

    **Automatische Filter (in `load_training_data`):**

    **1️⃣ NULL-Werte Filter:**
    ```python
    # Kritische Spalten müssen Daten haben
    critical_columns = ['price_close', 'volume_sol', 'ath_price_sol']
    data_clean = data.dropna(subset=critical_columns)
    ```
    - Entfernt Coins ohne Preis/Volumen-Daten
    - Entfernt Coins ohne ATH-Informationen

    **2️⃣ Coin-Qualitäts-Filter:**
    ```python
    # Coins mit zu wenig Datenpunkten rausfiltern
    coin_counts = data_clean['mint'].value_counts()
    valid_mints = coin_counts[coin_counts >= 30].index
    data_final = data_clean[data_clean['mint'].isin(valid_mints)]
    ```
    - Mindestens 30 Datenpunkte pro Coin
    - Verhindert Training mit Rauschen (3 Datenpunkte = kein Muster)

    **Ergebnis:**
    - ✅ Keine ML-Abstürze durch NULL/NaN
    - ✅ KI lernt nur aus vollständigen Daten
    - ✅ Besseres Training, bessere Vorhersagen
    - ✅ Automatische Logs zeigen Filter-Effekte

    **Live-Betrieb Vorteile:**
    - Robuste Modelle ohne Halluzinationen
    - Zuverlässige Signale basierend auf echten Mustern
    - Keine falschen Alarme durch unvollständige Daten
    """)

    st.divider()

    # Dokumentation
    st.header("📚 Dokumentation")
    
    st.markdown("""
    **Wichtige Dokumentationen:**
    - **[README.md](../README.md)** - Projekt-Übersicht
    - **[KOMPLETTE_KI_MODELL_ANLEITUNG.md](../docs/KOMPLETTE_KI_MODELL_ANLEITUNG.md)** - Vollständige Anleitung zur Modell-Erstellung
    - **[IMPLEMENTIERUNGS_ANLEITUNG_METRIKEN.md](../docs/IMPLEMENTIERUNGS_ANLEITUNG_METRIKEN.md)** - Metriken-Integration
    - **[LABEL_VALIDIERUNGSBERICHT.md](../docs/LABEL_VALIDIERUNGSBERICHT.md)** - Label-Validierung
    - **[PRODUCTION_READINESS_BERICHT.md](../docs/PRODUCTION_READINESS_BERICHT.md)** - Production-Readiness
    - **[XGBOOST_MODEL_ERSTELLUNG.md](../docs/XGBOOST_MODEL_ERSTELLUNG.md)** - XGBoost-Optimierung
    - **[COOLIFY_DEPLOYMENT_CHECKLIST.md](../docs/COOLIFY_DEPLOYMENT_CHECKLIST.md)** - Coolify Deployment
    - **[ERWEITERUNGSPLAN_ATH_UND_METRIKEN.md](../docs/ERWEITERUNGSPLAN_ATH_UND_METRIKEN.md)** - 🆕 ATH-Integration Plan
    - **[ATH_INTEGRATION_ABGESCHLOSSEN.md](../docs/ATH_INTEGRATION_ABGESCHLOSSEN.md)** - 🆕 ATH-Integration Dokumentation
    - **[TEST_ERGEBNISSE_ATH_INTEGRATION.md](../docs/TEST_ERGEBNISSE_ATH_INTEGRATION.md)** - 🆕 ATH-Test-Ergebnisse
    - **[VOLLSTAENDIGER_TESTBERICHT.md](../docs/VOLLSTAENDIGER_TESTBERICHT.md)** - 🆕 Vollständiger System-Testbericht
    - **[DOCKER_TEST_ANLEITUNG.md](../DOCKER_TEST_ANLEITUNG.md)** - 🆕 Docker Test-Anleitung
    - **[complete_schema.sql](../sql/complete_schema.sql)** - Vollständiges Datenbank-Schema
    """)
    
    st.divider()
    
    # Label-Erstellung - DETAILLIERT
    st.header("🏷️ Label-Erstellung - Schritt für Schritt")
    
    st.subheader("1️⃣ Klassische Labels (Schwellwert-basiert)")
    st.markdown("""
    **Wann verwendet:** Wenn `use_time_based_prediction = False`
    
    **Prozess:**
    1. Für jede Zeile wird geprüft: `target_variable operator target_value`
    2. Beispiel: `price_close > 50000`
    3. Label = 1 wenn Bedingung erfüllt, 0 wenn nicht erfüllt
    
    **Operatoren:**
    - `>` - Größer als
    - `<` - Kleiner als
    - `>=` - Größer oder gleich
    - `<=` - Kleiner oder gleich
    - `=` - Gleich
    
    **Beispiel:**
    ```
    Zeile 1: price_close = 45000 → 45000 > 50000? → Nein → Label = 0
    Zeile 2: price_close = 55000 → 55000 > 50000? → Ja → Label = 1
    Zeile 3: price_close = 60000 → 60000 > 50000? → Ja → Label = 1
    ```
    """)
    
    st.subheader("2️⃣ Zeitbasierte Labels (Zukunfts-Vorhersage)")
    st.markdown("""
    **Wann verwendet:** Wenn `use_time_based_prediction = True`
    
    **Konzept:** Statt zu fragen "Ist price_close > 50000?", fragt das System:
    "Steigt price_close in 10 Minuten um mindestens 30%?"
    
    **Schritt-für-Schritt Prozess:**
    
    **Schritt 1: Aktueller Wert**
    ```python
    current_value = data.loc[idx, "price_close"]  # z.B. 100.0
    ```
    
    **Schritt 2: Zukünftiger Wert berechnen**
    - System berechnet, wie viele Zeilen in die Zukunft geschaut werden muss
    - Beispiel: Phase 1 hat `interval_seconds=5` → 10 Minuten = 120 Zeilen
    - Zukünftiger Wert: `future_value = data.loc[idx + 120, "price_close"]`  # z.B. 130.0
    
    **Schritt 3: Prozent-Änderung berechnen**
    ```python
    percent_change = ((future_value - current_value) / current_value) * 100
    # Beispiel: ((130.0 - 100.0) / 100.0) * 100 = 30.0%
    ```
    
    **Schritt 4: Label erstellen**
    ```python
    if direction == "up":
        label = 1 if percent_change >= min_percent_change else 0
        # Beispiel: 30.0% >= 30.0%? → Ja → Label = 1
    else:  # "down"
        label = 1 if percent_change <= -min_percent_change else 0
        # Beispiel: -30.0% <= -30.0%? → Ja → Label = 1
    ```
    
    **Wichtig:**
    - ⚠️ **Data Leakage Prevention:** `target_variable` wird aus Features entfernt!
    - ⚠️ **NaN-Handling:** Zeilen ohne Zukunftswerte werden entfernt (am Ende des Datensatzes)
    - ⚠️ **Division durch Null:** Wird verhindert durch `valid_mask`
    - ⚠️ **Phase-Intervalle:** System verwendet exakte Intervalle pro Phase (genauer als Durchschnitt)
    """)
    
    st.subheader("3️⃣ Label-Validierung")
    st.markdown("""
    **Automatische Prüfungen:**
    - ✅ Mindestens 1 positives Label (sonst: Fehler)
    - ✅ Mindestens 1 negatives Label (sonst: Fehler)
    - ⚠️ Warnung wenn Labels sehr unausgewogen (< 5% oder > 95% positiv)
    
    **Empfehlung:** Labels sollten zwischen 20% und 80% positiv sein für beste Performance
    """)
    
    st.divider()
    
    # Was beim KI-Modell-Erstellen beachten?
    st.header("⚠️ Was beim KI-Modell-Erstellen beachten?")
    
    st.subheader("1️⃣ Datenqualität")
    st.markdown("""
    **Wichtig:**
    - ✅ **Zeitraum wählen:** Mindestens 4-6 Stunden Daten für aussagekräftige Modelle
    - ✅ **Phasen berücksichtigen:** Verschiedene Phasen haben unterschiedliche Verhaltensmuster
    - ✅ **Datenverfügbarkeit prüfen:** Prüfe `/api/data-availability` vor dem Training
    - ⚠️ **Max. 500.000 Zeilen:** System begrenzt automatisch (RAM-Management)
    
    **Empfehlung:**
    - Training: Letzte 4-6 Stunden der verfügbaren Daten
    - Test: Letzte 10-30 Minuten (separat vom Training)
    """)
    
    st.subheader("2️⃣ Feature-Auswahl")
    st.markdown("""
    **Kritische Features (empfohlen):**
    - ✅ `dev_sold_amount` - **KRITISCH** für Rug-Detection
    - ✅ `buy_pressure_ratio` - Buy/Sell-Verhältnis
    - ✅ `unique_signer_ratio` - Anteil einzigartiger Trader
    - ✅ `price_close`, `volume_sol` - Basis-Daten
    
    **Optional (Feature-Engineering):**
    - ✅ `use_engineered_features=True` - Erstellt ~70 zusätzliche Features (inkl. ATH-Features)
    - ✅ `feature_engineering_windows=[5, 10, 15]` - Fenstergrößen für Features
    
    **Feature-Ausschluss:**
    - ⚠️ `exclude_features` - Liste von Features die ausgeschlossen werden sollen
    - Beispiel: `exclude_features=["dev_sold_amount"]` - Wenn Dev-Tracking nicht gewünscht
    """)
    
    st.subheader("3️⃣ Zeitbasierte Vorhersage")
    st.markdown("""
    **Parameter:**
    - `future_minutes`: 5-30 Minuten empfohlen (zu kurz = zu viele False Positives, zu lang = zu wenige Positives)
    - `min_percent_change`: 5-30% empfohlen (zu niedrig = zu viele False Positives, zu hoch = zu wenige Positives)
    - `direction`: "up" (steigt) oder "down" (fällt)
    
    **Empfehlung:**
    - **Konservativ:** 10 Minuten, 5% Änderung
    - **Mittel:** 10 Minuten, 10% Änderung
    - **Aggressiv:** 5 Minuten, 20% Änderung
    
    **Wichtig:**
    - ⚠️ Bei zeitbasierter Vorhersage wird `target_variable` automatisch aus Features entfernt (Data Leakage Prevention)
    - ⚠️ Am Ende des Datensatzes werden Zeilen ohne Zukunftswerte entfernt
    """)
    
    st.subheader("4️⃣ Hyperparameter")
    st.markdown("""
    **Random Forest:**
    - `n_estimators`: 100-300 (mehr = besser, aber langsamer)
    - `max_depth`: 10-20 (zu tief = Overfitting)
    - `min_samples_split`: 5-10 (zu niedrig = Overfitting)
    
    **XGBoost:**
    - `n_estimators`: 200-500 (mehr = besser, aber langsamer)
    - `max_depth`: 6-10 (zu tief = Overfitting)
    - `learning_rate`: 0.05-0.1 (niedriger = besser, aber langsamer)
    
    **Empfehlung:** Verwende Standard-Parameter zuerst, dann optimiere basierend auf Ergebnissen
    """)
    
    st.subheader("5️⃣ Imbalanced Data Handling")
    st.markdown("""
    **Problem:** Wenn Labels sehr unausgewogen sind (z.B. 95% negativ, 5% positiv)
    
    **Lösung:** `use_smote=True`
    - SMOTE (Synthetic Minority Oversampling Technique)
    - Erstellt synthetische positive Beispiele
    - Verbessert Performance bei unausgewogenen Daten
    
    **Wann verwenden:** Wenn Labels < 20% oder > 80% positiv sind
    """)
    
    st.subheader("6️⃣ Cross-Validation")
    st.markdown("""
    **Aktivierung:** `use_timeseries_split=True` (Standard: aktiviert)
    
    **Was passiert:**
    - TimeSeriesSplit respektiert zeitliche Reihenfolge (verhindert Data Leakage)
    - Anzahl Splits: `cv_splits=5` (Standard)
    - Berechnet Train- und Test-Accuracy für jeden Split
    - Overfitting-Gap wird berechnet (Differenz zwischen Train- und Test-Accuracy)
    
    **Wichtig:**
    - ⚠️ TimeSeriesSplit ist wichtig für Zeitreihen-Daten!
    - ⚠️ Normale K-Fold würde Data Leakage verursachen (Zukunft in Training)
    """)
    
    st.subheader("7️⃣ Marktstimmung")
    st.markdown("""
    **Aktivierung:** `use_market_context=True`
    
    **Vorteile:**
    - Unterscheidet echte Token-Pumps von Marktbewegungen
    - Erstellt Features: `sol_price_change_pct`, `sol_price_ma_5`, `sol_price_volatility`
    
    **Wann verwenden:** Immer empfohlen (verbessert Modell-Performance)
    
    **Voraussetzung:** `exchange_rates` Tabelle muss Daten enthalten
    """)
    
    st.divider()
    
    # Kompletter Workflow - Schritt für Schritt
    st.header("🔄 Kompletter Workflow - Schritt für Schritt")
    
    st.subheader("Phase 1: Job-Erstellung")
    st.markdown("""
    **1.1 Web-UI oder API Request**
    - Benutzer erstellt Training-Job über Web-UI oder API
    - Parameter werden eingegeben (Name, Modell-Typ, Features, Zeitraum, etc.)
    
    **1.2 Validierung**
    - Request wird validiert (Pydantic Schema)
    - Zeitbasierte Parameter werden in `train_params._time_based` gespeichert
    - Feature-Engineering Parameter werden in `train_params` gespeichert
    
    **1.3 Job in Datenbank**
    - Job wird in `ml_jobs` Tabelle erstellt mit `status='PENDING'`
    - Modell-Name wird in `progress_msg` temporär gespeichert
    - Response mit `job_id` wird zurückgegeben
    """)
    
    st.subheader("Phase 2: Worker-Verarbeitung")
    st.markdown("""
    **2.1 Job gefunden**
    - Worker prüft alle 5 Sekunden auf `PENDING` Jobs
    - Wenn Job gefunden UND weniger als 2 Jobs aktiv:
      - Status wird auf `RUNNING` gesetzt
      - Job wird asynchron verarbeitet
    
    **2.2 Parameter extrahieren**
    - Modell-Name aus `progress_msg`
    - Features, Phasen, Zeitraum aus Job-Daten
    - Hyperparameter aus `train_params`
    - Zeitbasierte Parameter aus `train_params._time_based`
    """)
    
    st.subheader("Phase 3: Daten-Laden")
    st.markdown("""
    **3.1 SQL Query**
    ```sql
    SELECT timestamp, price_open, price_high, price_low, price_close,
           volume_sol, buy_volume_sol, sell_volume_sol, dev_sold_amount,
           buy_pressure_ratio, unique_signer_ratio, ...
    FROM coin_metrics
    WHERE timestamp >= $1 AND timestamp <= $2
      AND phase_id_at_time = ANY($3)  -- Falls Phasen gefiltert
    ORDER BY timestamp
    LIMIT 500000
    ```
    
    **3.2 Daten-Verarbeitung**
    - Daten werden nach `timestamp` sortiert (wichtig für zeitbasierte Labels!)
    - `timestamp` wird als Index gesetzt
    - Duplikate werden entfernt
    - Decimal-Typen werden zu float konvertiert (PostgreSQL → Pandas)
    - Max. 500.000 Zeilen (RAM-Management)
    
    **3.3 Marktstimmung (optional)**
    - Wenn `use_market_context=True`:
      - SOL-Preis wird aus `exchange_rates` geladen
      - Features werden erstellt: `sol_price_change_pct`, `sol_price_ma_5`, `sol_price_volatility`
    """)
    
    st.subheader("Phase 4: Feature-Vorbereitung")
    st.markdown("""
    **4.1 Data Leakage Prevention**
    ```python
    # Bei zeitbasierter Vorhersage:
    features_for_loading = ["price_open", "volume_sol", "price_close"]  # Enthält target_var
    features_for_training = ["price_open", "volume_sol"]  # target_var ENTFERNT!
    ```
    
    **4.2 Feature-Engineering (optional)**
    - Wenn `use_engineered_features=True`:
      - ~70 zusätzliche Features werden erstellt (inkl. ATH-Features)
      - Momentum, Volumen-Patterns, Whale-Aktivität, etc.
      - Features werden zu `features_for_training` hinzugefügt
    """)
    
    st.subheader("Phase 5: Label-Erstellung")
    st.markdown("""
    **5.1 Zeitbasierte Labels (wenn aktiviert)**
    ```python
    # Für jede Zeile:
    current_value = data.loc[idx, "price_close"]  # z.B. 100.0
    future_value = data.loc[idx + rows_to_shift, "price_close"]  # z.B. 130.0
    percent_change = ((130.0 - 100.0) / 100.0) * 100  # = 30.0%
    
    if direction == "up":
        label = 1 if percent_change >= min_percent_change else 0
    ```
    
    **5.2 Klassische Labels (wenn zeitbasierte Vorhersage deaktiviert)**
    ```python
    # Für jede Zeile:
    value = data.loc[idx, "price_close"]  # z.B. 55000
    label = 1 if value > target_value else 0  # z.B. 55000 > 50000? → Ja → 1
    ```
    
    **5.3 Label-Validierung**
    - Prüft ob mindestens 1 positives und 1 negatives Label vorhanden
    - Warnung wenn Labels sehr unausgewogen (< 5% oder > 95% positiv)
    """)
    
    st.subheader("Phase 6: Training")
    st.markdown("""
    **6.1 Train/Test Split**
    - Standard: 80% Training, 20% Test
    - TimeSeriesSplit respektiert zeitliche Reihenfolge
    
    **6.2 SMOTE (optional)**
    - Wenn `use_smote=True` und Labels unausgewogen:
      - Synthetische positive Beispiele werden erstellt
      - Verbessert Performance bei unausgewogenen Daten
    
    **6.3 Modell-Training**
    - Random Forest oder XGBoost wird trainiert
    - Training läuft in separatem Thread (blockiert nicht Event Loop)
    - Progress wird kontinuierlich aktualisiert (20% → 60% → 80% → 100%)
    
    **6.4 Cross-Validation (optional)**
    - Wenn `use_timeseries_split=True`:
      - TimeSeriesSplit mit 5 Splits
      - Train- und Test-Accuracy für jeden Split
      - Overfitting-Gap wird berechnet
    """)
    
    st.subheader("Phase 7: Evaluation")
    st.markdown("""
    **7.1 Basis-Metriken**
    - Accuracy, Precision, Recall, F1-Score
    - Confusion Matrix (TP, TN, FP, FN)
    
    **7.2 Erweiterte Metriken**
    - ROC-AUC (wenn Modell Wahrscheinlichkeiten unterstützt)
    - MCC (Matthews Correlation Coefficient)
    - FPR (False Positive Rate)
    - FNR (False Negative Rate)
    
    **7.3 Rug-Detection-Metriken**
    - Dev-Sold Detection Rate
    - Wash-Trading Detection Rate
    - Weighted Cost (FN 10x schwerer als FP)
    - Precision at K

    **7.4 Zeitbasierte Metriken (bei zeitbasierter Vorhersage)**
    - Labels-Balance: Verhältnis positive/negative Labels
    - Zeitfenster-Statistiken: Durchschnittliche Zeit bis zum Ziel
    - Early/Late Prediction Accuracy: Performance zu verschiedenen Zeitpunkten
    - Simulierter Profit (%): ROI-basierte Bewertung

    **7.5 Feature Importance**
    - Wichtigste Features werden berechnet (Random Forest, XGBoost)
    - Gespeichert als JSONB in `ml_models.feature_importance`
    - Unterstützt 80+ Features inkl. ATH- und engineered Features
    """)
    
    st.subheader("Phase 8: Speicherung")
    st.markdown("""
    **8.1 Modell-Datei**
    - Modell wird als .pkl Datei gespeichert
    - Pfad: `/app/models/{model_id}_{model_name}.pkl`
    
    **8.2 Datenbank-Metadaten**
    - Alle Metriken werden in `ml_models` Tabelle gespeichert
    - Status wird auf `READY` gesetzt
    - Job-Status wird auf `COMPLETED` gesetzt
    """)
    
    st.divider()
    
    # Best Practices
    st.header("💡 Best Practices")
    
    st.subheader("1️⃣ Modell-Erstellung")
    st.markdown("""
    - ✅ **Zeitraum:** Mindestens 4-6 Stunden Trainingsdaten
    - ✅ **Features:** Immer `dev_sold_amount` inkludieren (kritisch für Rug-Detection)
    - ✅ **Marktstimmung:** Immer aktivieren (`use_market_context=True`)
    - ✅ **Feature-Engineering:** Aktivieren für bessere Performance
    - ✅ **Cross-Validation:** Immer aktivieren (verhindert Overfitting)
    - ✅ **SMOTE:** Aktivieren wenn Labels unausgewogen (< 20% oder > 80% positiv)
    """)
    
    st.subheader("2️⃣ Zeitbasierte Vorhersage")
    st.markdown("""
    - ✅ **future_minutes:** 5-30 Minuten (10 Minuten empfohlen)
    - ✅ **min_percent_change:** 5-30% (10% empfohlen)
    - ✅ **direction:** "up" für Pump-Detection, "down" für Crash-Detection
    - ⚠️ **Zu niedrige Werte:** Viele False Positives
    - ⚠️ **Zu hohe Werte:** Zu wenige positive Labels
    """)
    
    st.subheader("3️⃣ Hyperparameter-Tuning")
    st.markdown("""
    - ✅ **Starte mit Standard-Parametern:** System hat gute Defaults
    - ✅ **Iteriere basierend auf Ergebnissen:** Teste verschiedene Kombinationen
    - ✅ **Vergleiche Modelle:** Verwende Vergleichs-Funktion um beste Parameter zu finden
    - ⚠️ **Overfitting vermeiden:** Zu hohe `max_depth` oder zu niedrige `min_samples_split`
    """)
    
    st.subheader("4️⃣ Testing")
    st.markdown("""
    - ✅ **Separater Zeitraum:** Teste auf Daten die NICHT im Training waren
    - ✅ **Realistische Zeiträume:** 10-30 Minuten Test-Daten
    - ✅ **Vergleiche Modelle:** Teste mehrere Modelle auf denselben Daten
    - ⚠️ **Nicht auf Trainingsdaten testen:** Würde unrealistische Metriken zeigen
    """)
    
    st.divider()
    
    # Häufige Fehler und Lösungen
    st.header("🐛 Häufige Fehler und Lösungen")
    
    st.subheader("1️⃣ 'Keine positiven Labels gefunden'")
    st.markdown("""
    **Problem:** Alle Labels sind 0 (Bedingung wird nie erfüllt)
    
    **Ursachen:**
    - `min_percent_change` zu hoch (z.B. 50% bei zeitbasierter Vorhersage)
    - `target_value` zu hoch (z.B. `price_close > 1000000`)
    - Zeitraum zu kurz (zu wenige Daten)
    
    **Lösung:**
    - Reduziere `min_percent_change` oder `target_value`
    - Wähle längeren Zeitraum
    - Prüfe Datenverfügbarkeit vor Training
    """)
    
    st.subheader("2️⃣ 'Keine negativen Labels gefunden'")
    st.markdown("""
    **Problem:** Alle Labels sind 1 (Bedingung wird immer erfüllt)
    
    **Ursachen:**
    - `min_percent_change` zu niedrig (z.B. 0.1%)
    - `target_value` zu niedrig (z.B. `price_close > 0`)
    
    **Lösung:**
    - Erhöhe `min_percent_change` oder `target_value`
    - Prüfe Label-Verteilung vor Training
    """)
    
    st.subheader("3️⃣ 'Nicht genug Daten'")
    st.markdown("""
    **Problem:** Zu wenige Zeilen für Training
    
    **Ursachen:**
    - Zeitraum zu kurz
    - Phasen-Filter zu restriktiv
    - Datenbank enthält keine Daten für Zeitraum
    
    **Lösung:**
    - Wähle längeren Zeitraum
    - Entferne Phasen-Filter oder wähle mehr Phasen
    - Prüfe `/api/data-availability` für verfügbare Daten
    """)
    
    st.subheader("4️⃣ 'Modell-Performance schlecht'")
    st.markdown("""
    **Problem:** Accuracy < 60% oder viele False Positives
    
    **Ursachen:**
    - Zu wenige Features
    - Feature-Engineering nicht aktiviert
    - Labels zu unausgewogen
    - Hyperparameter nicht optimal
    
    **Lösung:**
    - Aktiviere Feature-Engineering
    - Aktiviere Marktstimmung
    - Aktiviere SMOTE wenn Labels unausgewogen
    - Teste verschiedene Hyperparameter
    - Vergleiche mehrere Modelle
    """)
    
    st.subheader("5️⃣ 'Training dauert zu lange'")
    st.markdown("""
    **Problem:** Training dauert > 30 Minuten
    
    **Ursachen:**
    - Zu viele Daten (> 500.000 Zeilen)
    - Zu viele Features (> 50)
    - `n_estimators` zu hoch
    
    **Lösung:**
    - Reduziere Zeitraum (weniger Daten)
    - Reduziere Feature-Anzahl
    - Reduziere `n_estimators`
    - System begrenzt automatisch auf 500.000 Zeilen
    """)
    
    st.divider()
    
    # Performance-Tipps
    st.header("⚡ Performance-Tipps")
    
    st.markdown("""
    **1. Datenmenge:**
    - ✅ 4-6 Stunden Trainingsdaten sind ausreichend
    - ⚠️ Mehr Daten = längeres Training, aber nicht unbedingt bessere Performance
    
    **2. Feature-Anzahl:**
    - ✅ 10-20 Basis-Features + Feature-Engineering (~70 Features) = optimal
    - ⚠️ Zu viele Features (> 100) = Overfitting-Risiko
    
    **3. Hyperparameter:**
    - ✅ Random Forest: `n_estimators=200`, `max_depth=10`
    - ✅ XGBoost: `n_estimators=300`, `max_depth=8`, `learning_rate=0.1`
    
    **4. Parallelisierung:**
    - ✅ System verarbeitet max. 2 Jobs parallel
    - ✅ Training läuft in separatem Thread (blockiert nicht Event Loop)
    
    **5. Caching:**
    - ✅ Modelle werden persistent gespeichert (DB + .pkl)
    - ✅ Kein Re-Training nötig für Tests
    """)
    
    st.divider()
    
    # Zusammenfassung
    st.header("📊 Zusammenfassung")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Modell-Typen", "2", "Random Forest, XGBoost")
    
    with col2:
        st.metric("Vorhersage-Typen", "2", "Klassisch, Zeitbasiert")
    
    with col3:
        st.metric("Job-Typen", "3", "TRAIN, TEST, COMPARE")
    
    st.info("""
    **Wichtig:** 
    - Alle Jobs werden asynchron verarbeitet (nicht sofort)
    - Modell-Training kann mehrere Minuten dauern
    - Progress wird kontinuierlich aktualisiert
    - Modelle werden persistent gespeichert (DB + .pkl Dateien)
    - Prometheus-Metriken werden für Monitoring exportiert
    - Labels werden automatisch validiert (keine Data Leakage)
    - Feature-Engineering erstellt ~70 zusätzliche Features (inkl. ATH-Features)
    - Marktstimmung unterscheidet echte Pumps von Marktbewegungen
    - 🆕 ATH-Tracking ermöglicht Breakout-Erkennung und Momentum-Analyse
    """)

# ============================================================
# Main App
# ============================================================


