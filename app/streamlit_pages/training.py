"""
Training Page Module
Extrahierte Seite aus streamlit_app.py
"""
import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any, Optional

# Import aus streamlit_utils
from streamlit_utils import (
    api_get, api_post, api_delete, api_patch,
    AVAILABLE_FEATURES, AVAILABLE_TARGETS, FEATURE_CATEGORIES, CRITICAL_FEATURES,
    API_BASE_URL, load_phases, load_config, save_config,
    get_default_config, validate_url, validate_port,
    reload_config, restart_service, get_service_logs
)


def page_train():
    """Neues Modell trainieren - ÜBERARBEITETE VERSION"""
    st.title("🚀 Neues Modell erstellen")
    
    st.info("""
    **📖 Schnellstart:** 
    Fülle die minimalen Felder aus und klicke auf "Modell trainieren".
    Erweiterte Optionen findest du im ausklappbaren Bereich unten.
    """)
    
    # Initialisiere session_state für Features (einmalig, außerhalb des Forms)
    if 'train_features_initialized' not in st.session_state:
        for category, features_in_category in FEATURE_CATEGORIES.items():
            for feature in features_in_category:
                if f"feature_{feature}" not in st.session_state:
                    st.session_state[f"feature_{feature}"] = True  # Default: aktiviert
        st.session_state['train_features_initialized'] = True
    
    # Lade verfügbare Daten (einmalig, außerhalb des Forms)
    data_availability = api_get("/api/data-availability")
    min_timestamp = data_availability.get('min_timestamp') if data_availability else None
    max_timestamp = data_availability.get('max_timestamp') if data_availability else None
    
    min_date = None
    max_date = None
    min_datetime = None
    max_datetime = None
    
    if min_timestamp and max_timestamp:
        try:
            min_datetime = datetime.fromisoformat(min_timestamp.replace('Z', '+00:00'))
            max_datetime = datetime.fromisoformat(max_timestamp.replace('Z', '+00:00'))
            min_date = min_datetime.date()
            max_date = max_datetime.date()
            st.info(f"📊 **Verfügbare Daten:** Von {min_datetime.strftime('%d.%m.%Y %H:%M')} bis {max_datetime.strftime('%d.%m.%Y %H:%M')}")
        except Exception as e:
            st.warning(f"⚠️ Konnte Datumsbereich nicht parsen: {e}")
    else:
        st.warning("⚠️ Keine Trainingsdaten in der Datenbank gefunden!")
    
    # ============================================================
    # FORMULAR
    # ============================================================
    with st.form("train_model_form", clear_on_submit=False):
        # Basis-Informationen
        st.subheader("📝 Basis-Informationen")
        model_name = st.text_input("Modell-Name *", placeholder="z.B. PumpDetector_v1", key="train_model_name")
        model_type = st.selectbox(
            "Modell-Typ *",
            ["random_forest", "xgboost"],
            index=0,
            help="Random Forest: Robust, schnell. XGBoost: Beste Performance",
            key="train_model_type"
        )
        
        st.divider()
        
        # Training-Zeitraum
        st.subheader("📅 Training-Zeitraum")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**🕐 Start-Zeitpunkt**")
            if min_date:
                default_start_date = min_date
            else:
                default_start_date = datetime.now(timezone.utc).date() - timedelta(days=30)
            
            train_start_date = st.date_input(
                "Start-Datum *",
                value=default_start_date,
                min_value=min_date,
                max_value=max_date,
                key="train_start_date"
            )
            
            if min_datetime and train_start_date == min_date:
                default_start_time = min_datetime.time()
            else:
                default_start_time = datetime.now(timezone.utc).time().replace(hour=0, minute=0, second=0, microsecond=0)
            
            train_start_time = st.time_input(
                "Start-Uhrzeit *",
                value=default_start_time,
                key="train_start_time"
            )
        
        with col2:
            st.markdown("**🕐 Ende-Zeitpunkt**")
            if max_date:
                default_end_date = max_date
            else:
                default_end_date = datetime.now(timezone.utc).date()
            
            train_end_date = st.date_input(
                "Ende-Datum *",
                value=default_end_date,
                min_value=train_start_date if train_start_date else None,
                max_value=max_date,
                key="train_end_date"
            )
            
            if max_datetime and train_end_date == max_date:
                default_end_time = max_datetime.time()
            else:
                default_end_time = datetime.now(timezone.utc).time().replace(hour=23, minute=59, second=59, microsecond=0)
            
            train_end_time = st.time_input(
                "Ende-Uhrzeit *",
                value=default_end_time,
                key="train_end_time"
            )
        
        st.divider()
        
        # Vorhersage-Ziel (zeitbasiert - Standard)
        st.subheader("⏰ Vorhersage-Ziel")
        st.caption("Das Modell lernt: 'Steigt/Fällt die Variable in X Minuten um Y%?'")
        
        time_based_target_var = st.selectbox(
            "Variable überwachen *",
            AVAILABLE_TARGETS, 
            index=0,
            help="Welche Variable soll für die prozentuale Änderung verwendet werden?",
            key="train_target_var"
        )
        
        col1, col2, col3 = st.columns(3)
        with col1:
            future_minutes = st.number_input(
                "Zeitraum (Minuten) *", 
                min_value=1, 
                max_value=60,
                value=10,
                step=1,
                help="In wie vielen Minuten soll die Änderung stattfinden?",
                key="train_future_minutes"
            )
        with col2:
            min_percent_change = st.number_input(
                "Mindest-Änderung (%) *",
                min_value=0.1, 
                max_value=1000.0,
                value=5.0,
                step=0.5,
                help="Mindest-Prozentuale Änderung",
                key="train_min_percent"
            )
        with col3:
            direction = st.selectbox(
                "Richtung *",
                ["up", "down"],
                format_func=lambda x: "Steigt" if x == "up" else "Fällt",
                help="Steigt oder fällt die Variable?",
                key="train_direction"
            )
        
        # Label-Erstellung Transparenz
        st.info(f"""
        📊 **Label-Erstellung:**
        
        Für jede Zeile in den Trainingsdaten wird geprüft:
        
        1. **Aktueller Wert**: `{time_based_target_var}` zum Zeitpunkt T
        2. **Zukünftiger Wert**: `{time_based_target_var}` zum Zeitpunkt T + {future_minutes} Minuten
        3. **Prozentuale Änderung**: `((Zukunft - Aktuell) / Aktuell) * 100`
        
        **Label = 1** wenn:
        - Änderung >= {min_percent_change}% (bei "Steigt")
        - Änderung <= -{min_percent_change}% (bei "Fällt")
        
        **Label = 0** wenn:
        - Bedingung nicht erfüllt
        
        **Beispiel:**
        - Aktuell: 100 SOL
        - Zukunft ({future_minutes} Min): 106 SOL
        - Änderung: +6%
        - **Label = 1** ✅ (weil 6% >= {min_percent_change}%)
        """)
        
        st.divider()
        
        # ============================================================
        # ERWEITERTE OPTIONEN (ausklappbar)
        # ============================================================
        with st.expander("⚙️ Erweiterte Optionen", expanded=False):
            # Feature-Auswahl mit Kategorien
            st.subheader("📊 Features")
            st.caption("Wähle Features aus verschiedenen Kategorien. Kritische Features sind für Rug-Detection empfohlen.")
            
            # Verwende Tabs für Feature-Kategorien
            feature_tabs = st.tabs(list(FEATURE_CATEGORIES.keys()))
            
            for tab_idx, (category, features_in_category) in enumerate(FEATURE_CATEGORIES.items()):
                with feature_tabs[tab_idx]:
                    st.markdown(f"**{category}**")
                    for feature in features_in_category:
                        is_critical = feature in CRITICAL_FEATURES
                        st.checkbox(
                            feature,
                            value=st.session_state.get(f"feature_{feature}", True),
                            key=f"feature_{feature}",
                            help=f"{'⚠️ KRITISCH für Rug-Detection!' if is_critical else ''}"
                        )
            
            # Sammle Features aus ALLEN Kategorien (nach Tabs)
            selected_features = []
            for category, features_in_category in FEATURE_CATEGORIES.items():
                for feature in features_in_category:
                    if st.session_state.get(f"feature_{feature}", False):
                        selected_features.append(feature)
            
            # Fallback: Wenn keine Features ausgewählt wurden, verwende alle
            if not selected_features:
                st.warning("⚠️ Keine Features ausgewählt! Alle Features werden verwendet.")
                selected_features = AVAILABLE_FEATURES.copy()
            else:
                st.info(f"✅ {len(selected_features)} Feature(s) ausgewählt")
            
            st.divider()
            
            # Phasen-Filter
            st.subheader("🪙 Coin-Phasen (optional)")
            phases_list = load_phases()
            phases = None
            
            if phases_list:
                phase_options = {}
                for phase in phases_list:
                    phase_id = phase.get("id")
                    phase_name = phase.get("name", f"Phase {phase_id}")
                    interval_sec = phase.get("interval_seconds", 0)
                    if phase_name and phase_name != f"Phase {phase_id}":
                        display_name = f"Phase {phase_id} - {phase_name} ({interval_sec}s)"
                    else:
                        display_name = f"Phase {phase_id} ({interval_sec}s)"
                    phase_options[phase_id] = display_name
                
                sorted_phases = sorted(phase_options.items())
                phase_labels = [label for _, label in sorted_phases]
                phase_ids = [pid for pid, _ in sorted_phases]
                
                selected_labels = st.multiselect(
                    "Phasen auswählen (optional)",
                    phase_labels,
                    help="Welche Coin-Phasen sollen einbezogen werden? (Leer = alle)",
                    key="train_phases"
                )
                
                phases = [phase_ids[phase_labels.index(label)] for label in selected_labels] if selected_labels else None
            else:
                st.warning("⚠️ Phasen konnten nicht geladen werden.")
            
            st.divider()
            
            # Hyperparameter
            st.subheader("⚙️ Hyperparameter (optional)")
            use_custom_params = st.checkbox("Hyperparameter anpassen", value=False, key="train_use_custom_params")
            params = None
            
            if use_custom_params:
                if model_type == "random_forest":
                    col1, col2 = st.columns(2)
                    with col1:
                        n_estimators = st.number_input("n_estimators", min_value=10, max_value=1000, value=100, step=10, key="train_rf_n_estimators")
                    with col2:
                        max_depth = st.number_input("max_depth", min_value=1, max_value=50, value=10, step=1, key="train_rf_max_depth")
                    params = {"n_estimators": int(n_estimators), "max_depth": int(max_depth)}
                else:  # xgboost
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        n_estimators = st.number_input("n_estimators", min_value=10, max_value=1000, value=100, step=10, key="train_xgb_n_estimators")
                    with col2:
                        max_depth = st.number_input("max_depth", min_value=1, max_value=20, value=6, step=1, key="train_xgb_max_depth")
                    with col3:
                        learning_rate = st.number_input("learning_rate", min_value=0.01, max_value=1.0, value=0.1, step=0.01, key="train_xgb_learning_rate")
                    params = {
                        "n_estimators": int(n_estimators),
                        "max_depth": int(max_depth),
                        "learning_rate": float(learning_rate)
                    }
            
            st.divider()
            
            # Feature-Engineering
            st.subheader("🔧 Feature-Engineering (optional)")
            use_engineered_features = st.checkbox(
                "Erweiterte Pump-Detection Features verwenden",
                value=True,
                help="Erstellt ~70 zusätzliche Features aus den Basis-Features (inkl. ATH-Features)",
                key="train_use_engineered_features"
            )
            feature_engineering_windows = [5, 10, 15] if use_engineered_features else None
            
            st.divider()
            
            # Marktstimmung
            st.subheader("📈 Marktstimmung (optional)")
            use_market_context = st.checkbox(
                "SOL-Preis-Kontext hinzufügen",
                value=False,
                help="Hilft dem Modell zu unterscheiden: 'Token steigt, während SOL stabil ist' vs. 'Token steigt, weil SOL steigt'",
                key="train_use_market_context"
            )
            
            st.divider()
            
            # SMOTE & Cross-Validation
            st.subheader("⚖️ Daten-Handling (optional)")
            use_smote = st.checkbox("SMOTE für Imbalanced Data (empfohlen)", value=True, key="train_use_smote")
            use_timeseries_split = st.checkbox("TimeSeriesSplit für Cross-Validation (empfohlen)", value=True, key="train_use_timeseries_split")
            cv_splits = st.number_input("Anzahl Splits", min_value=3, max_value=10, value=5, step=1, key="train_cv_splits") if use_timeseries_split else 5
        
        # Submit Button
        submitted = st.form_submit_button("🚀 Modell trainieren", type="primary", use_container_width=True)
    
    # ============================================================
    # VERARBEITUNG NACH FORM-SUBMISSION
    # ============================================================
    if submitted:
        # Erstelle datetime-Objekte
        try:
            train_start_dt = datetime.combine(train_start_date, train_start_time).replace(tzinfo=timezone.utc)
            train_end_dt = datetime.combine(train_end_date, train_end_time).replace(tzinfo=timezone.utc)
        except Exception as e:
            st.error(f"❌ Fehler beim Erstellen der Datetime-Objekte: {e}")
            return
        
        # Validierung
        errors = []
        
        if not model_name or not model_name.strip():
            errors.append("❌ Modell-Name ist erforderlich!")
        
        # Sammle Features erneut (sicherstellen, dass alle erfasst werden)
        selected_features = []
        for category, features_in_category in FEATURE_CATEGORIES.items():
            for feature in features_in_category:
                if st.session_state.get(f"feature_{feature}", False):
                    selected_features.append(feature)
        
        if not selected_features:
            selected_features = AVAILABLE_FEATURES.copy()  # Fallback: Alle Features
        
        if train_start_dt >= train_end_dt:
            errors.append("❌ Start-Zeitpunkt muss vor End-Zeitpunkt liegen!")
        
        if errors:
            for error in errors:
                st.error(error)
            return
        
        # API-Call
        with st.spinner("🔄 Erstelle Training-Job..."):
            try:
                train_start_iso = train_start_dt.astimezone(timezone.utc).isoformat().replace('+00:00', 'Z')
                train_end_iso = train_end_dt.astimezone(timezone.utc).isoformat().replace('+00:00', 'Z')
                
                data = {
                    "name": model_name.strip(),
                    "model_type": model_type,
                    "target_var": time_based_target_var,
                    "operator": None,
                    "target_value": None,
                    "features": selected_features,
                    "phases": phases if phases else None,
                    "params": params,
                    "train_start": train_start_iso,
                    "train_end": train_end_iso,
                    "use_time_based_prediction": True,
                    "future_minutes": int(future_minutes),
                    "min_percent_change": float(min_percent_change),
                    "direction": direction,
                    "use_engineered_features": use_engineered_features,
                    "feature_engineering_windows": feature_engineering_windows,
                    "use_smote": use_smote,
                    "use_timeseries_split": use_timeseries_split,
                    "cv_splits": int(cv_splits) if use_timeseries_split else None,
                    "use_market_context": use_market_context
                }
                
                result = api_post("/api/models/create", data)
                
                if result:
                    st.success(f"✅ Job erstellt! Job-ID: {result.get('job_id')}")
                    st.info(f"📊 Status: {result.get('status')}. Das Modell wird jetzt trainiert.")
                    st.balloons()
                    st.session_state['last_created_job_id'] = result.get('job_id')
                else:
                    st.error("❌ Fehler beim Erstellen des Jobs. Bitte prüfe die Logs.")
            except Exception as e:
                st.error(f"❌ Fehler beim Erstellen des Jobs: {str(e)}")
                st.exception(e)
    
    # Weiterleitung zu Jobs-Seite
    if st.session_state.get('last_created_job_id'):
        if st.button("📊 Zu Jobs anzeigen", key="goto_jobs_after_train"):
            st.session_state['page'] = 'jobs'
            st.session_state.pop('last_created_job_id', None)
            st.rerun()


