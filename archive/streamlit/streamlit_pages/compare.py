"""
Compare Page Module
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
    AVAILABLE_FEATURES, FEATURE_CATEGORIES, CRITICAL_FEATURES,
    API_BASE_URL, load_phases, load_config, save_config,
    get_default_config, validate_url, validate_port,
    reload_config, restart_service, get_service_logs
)


def page_compare():
    """Modelle vergleichen"""
    st.title("⚔️ Modelle vergleichen")
    
    # Lade Modelle
    models = api_get("/api/models")
    if not models:
        st.warning("⚠️ Keine Modelle gefunden")
        return
    
    # Filter: Nur READY Modelle
    ready_models = [m for m in models if m.get('status') == 'READY' and not m.get('is_deleted')]
    
    if len(ready_models) < 2:
        st.warning("⚠️ Mindestens 2 fertige Modelle benötigt für Vergleich")
        return
    
    # Modell-Auswahl
    model_options = {m.get('id'): f"{m.get('name')} ({m.get('model_type')})" for m in ready_models}
    
    col1, col2 = st.columns(2)
    with col1:
        model_a_id = st.selectbox(
            "Modell A *",
            options=list(model_options.keys()),
            format_func=lambda x: model_options[x],
            index=0 if st.session_state.get('compare_model_a_id') not in model_options else list(model_options.keys()).index(st.session_state.get('compare_model_a_id'))
        )
    with col2:
        # Filter: Modell B darf nicht Modell A sein
        model_b_options = {k: v for k, v in model_options.items() if k != model_a_id}
        model_b_id = st.selectbox(
            "Modell B *",
            options=list(model_b_options.keys()),
            format_func=lambda x: model_b_options[x]
        )
    
    # Test-Zeitraum
    st.subheader("📅 Vergleichs-Zeitraum")
    
    # Lade verfügbare Daten (nur Min/Max Timestamps)
    data_availability = api_get("/api/data-availability")
    
    min_timestamp = data_availability.get('min_timestamp') if data_availability else None
    max_timestamp = data_availability.get('max_timestamp') if data_availability else None
    
    # Parse Timestamps
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
        st.warning("⚠️ Keine Vergleichsdaten in der Datenbank gefunden!")
    
    st.divider()
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**🕐 Start-Zeitpunkt**")
        
        # Default-Start-Datum: Min-Datum oder heute - 7 Tage
        if min_date:
            default_start_date = min_date
        else:
            default_start_date = datetime.now(timezone.utc).date() - timedelta(days=7)
        
        compare_start_date = st.date_input(
            "Start-Datum *",
            value=default_start_date,
            min_value=min_date,
            max_value=max_date,
            key="compare_start_date",
            help=f"Wähle ein Datum zwischen {min_date.strftime('%d.%m.%Y') if min_date else 'N/A'} und {max_date.strftime('%d.%m.%Y') if max_date else 'N/A'}"
        )
        
        # Default-Start-Uhrzeit: Min-Uhrzeit oder 00:00
        if min_datetime and compare_start_date == min_date:
            default_start_time = min_datetime.time()
        else:
            default_start_time = datetime.now(timezone.utc).time().replace(hour=0, minute=0, second=0, microsecond=0)
        
        compare_start_time = st.time_input(
            "Start-Uhrzeit *",
            value=default_start_time,
            key="compare_start_time",
            help="Wähle eine Uhrzeit"
        )
        
        test_start_dt = datetime.combine(compare_start_date, compare_start_time).replace(tzinfo=timezone.utc)
        
        # Warnung wenn außerhalb des verfügbaren Bereichs
        if min_datetime and test_start_dt < min_datetime:
            st.warning(f"⚠️ Start-Zeitpunkt liegt vor dem ältesten Eintrag ({min_datetime.strftime('%d.%m.%Y %H:%M')})")
        elif max_datetime and test_start_dt > max_datetime:
            st.warning(f"⚠️ Start-Zeitpunkt liegt nach dem neuesten Eintrag ({max_datetime.strftime('%d.%m.%Y %H:%M')})")
    
    with col2:
        st.markdown("**🕐 Ende-Zeitpunkt**")
        
        # Default-End-Datum: Max-Datum oder heute
        if max_date:
            default_end_date = max_date
        else:
            default_end_date = datetime.now(timezone.utc).date()
        
        compare_end_date = st.date_input(
            "Ende-Datum *",
            value=default_end_date,
            min_value=compare_start_date,  # Ende muss nach Start sein
            max_value=max_date,
            key="compare_end_date",
            help=f"Wähle ein Datum nach dem Start-Datum (max. {max_date.strftime('%d.%m.%Y') if max_date else 'N/A'})"
        )
        
        # Default-End-Uhrzeit: Max-Uhrzeit oder 23:59
        if max_datetime and compare_end_date == max_date:
            default_end_time = max_datetime.time()
        else:
            default_end_time = datetime.now(timezone.utc).time().replace(hour=23, minute=59, second=59, microsecond=0)
        
        compare_end_time = st.time_input(
            "Ende-Uhrzeit *",
            value=default_end_time,
            key="compare_end_time",
            help="Wähle eine Uhrzeit"
        )
        
        test_end_dt = datetime.combine(compare_end_date, compare_end_time).replace(tzinfo=timezone.utc)
        
        # Warnung wenn außerhalb des verfügbaren Bereichs
        if min_datetime and test_end_dt < min_datetime:
            st.warning(f"⚠️ Ende-Zeitpunkt liegt vor dem ältesten Eintrag ({min_datetime.strftime('%d.%m.%Y %H:%M')})")
        elif max_datetime and test_end_dt > max_datetime:
            st.warning(f"⚠️ Ende-Zeitpunkt liegt nach dem neuesten Eintrag ({max_datetime.strftime('%d.%m.%Y %H:%M')})")
    
    # Submit
    if st.button("⚔️ Vergleich starten", type="primary", use_container_width=True):
        if test_start_dt >= test_end_dt:
            st.error("❌ Start-Zeitpunkt muss vor End-Zeitpunkt liegen!")
        else:
            with st.spinner("🔄 Erstelle Vergleichs-Job..."):
                # Konvertiere datetime zu UTC ISO-Format
                test_start_iso = test_start_dt.astimezone(timezone.utc).isoformat().replace('+00:00', 'Z')
                test_end_iso = test_end_dt.astimezone(timezone.utc).isoformat().replace('+00:00', 'Z')
                
                data = {
                    "model_a_id": model_a_id,
                    "model_b_id": model_b_id,
                    "test_start": test_start_iso,
                    "test_end": test_end_iso
                }
                
                result = api_post("/api/models/compare", data)
                
                if result:
                    st.success(f"✅ Vergleichs-Job erstellt! Job-ID: {result.get('job_id')}")
                    st.info(f"📊 Status: {result.get('status')}. Der Vergleich wird jetzt ausgeführt.")
                    
                    if st.button("📊 Zu Jobs anzeigen"):
                        st.session_state['page'] = 'jobs'
                        st.rerun()


