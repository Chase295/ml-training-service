"""
Test Page Module
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


def page_test():
    """Modell testen"""
    st.title("🧪 Modell testen")
    
    # Prüfe ob Modell-ID aus Übersicht übergeben wurde
    test_model_id = st.session_state.get('test_model_id')
    
    # Lade Modelle
    models = api_get("/api/models")
    if not models:
        st.warning("⚠️ Keine Modelle gefunden")
        return
    
    # Filter: Nur READY Modelle
    ready_models = [m for m in models if m.get('status') == 'READY' and not m.get('is_deleted')]
    
    if not ready_models:
        st.warning("⚠️ Keine fertigen Modelle zum Testen verfügbar")
        return
    
    # Modell auswählen - verwende test_model_id wenn gesetzt
    model_options = {m.get('id'): f"{m.get('name')} ({m.get('model_type')})" for m in ready_models}
    
    # Bestimme initialen Index basierend auf test_model_id
    initial_index = 0
    if test_model_id and test_model_id in model_options:
        initial_index = list(model_options.keys()).index(test_model_id)
    
    selected_model_id = st.selectbox(
        "Modell auswählen *",
        options=list(model_options.keys()),
        format_func=lambda x: model_options[x],
        index=initial_index,
        key="test_model_selectbox"
    )
    
    # Aktualisiere test_model_id wenn sich die Auswahl ändert
    if selected_model_id != test_model_id:
        st.session_state['test_model_id'] = selected_model_id
    
    if selected_model_id:
        selected_model = next((m for m in ready_models if m.get('id') == selected_model_id), None)
        
        if selected_model:
            st.info(f"📋 Modell: {selected_model.get('name')} ({selected_model.get('model_type')})")
            
            # Test-Zeitraum
            st.subheader("📅 Test-Zeitraum")
            
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
                st.warning("⚠️ Keine Testdaten in der Datenbank gefunden!")
            
            st.divider()
            
            # Datum- und Uhrzeit-Auswahl
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**🕐 Start-Zeitpunkt**")
                
                # Default-Start-Datum: Min-Datum oder heute - 7 Tage
                if min_date:
                    default_start_date = min_date
                else:
                    default_start_date = datetime.now(timezone.utc).date() - timedelta(days=7)
                
                test_start_date = st.date_input(
                    "Start-Datum *",
                    value=default_start_date,
                    min_value=min_date,
                    max_value=max_date,
                    key="test_start_date",
                    help=f"Wähle ein Datum zwischen {min_date.strftime('%d.%m.%Y') if min_date else 'N/A'} und {max_date.strftime('%d.%m.%Y') if max_date else 'N/A'}"
                )
                
                # Default-Start-Uhrzeit: Min-Uhrzeit oder 00:00
                if min_datetime and test_start_date == min_date:
                    default_start_time = min_datetime.time()
                else:
                    default_start_time = datetime.now(timezone.utc).time().replace(hour=0, minute=0, second=0, microsecond=0)
                
                test_start_time = st.time_input(
                    "Start-Uhrzeit *",
                    value=default_start_time,
                    key="test_start_time",
                    help="Wähle eine Uhrzeit"
                )
                
                test_start_dt = datetime.combine(test_start_date, test_start_time).replace(tzinfo=timezone.utc)
                
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
                
                test_end_date = st.date_input(
                    "Ende-Datum *",
                    value=default_end_date,
                    min_value=test_start_date,  # Ende muss nach Start sein
                    max_value=max_date,
                    key="test_end_date",
                    help=f"Wähle ein Datum nach dem Start-Datum (max. {max_date.strftime('%d.%m.%Y') if max_date else 'N/A'})"
                )
                
                # Default-End-Uhrzeit: Max-Uhrzeit oder 23:59
                if max_datetime and test_end_date == max_date:
                    default_end_time = max_datetime.time()
                else:
                    default_end_time = datetime.now(timezone.utc).time().replace(hour=23, minute=59, second=59, microsecond=0)
                
                test_end_time = st.time_input(
                    "Ende-Uhrzeit *",
                    value=default_end_time,
                    key="test_end_time",
                    help="Wähle eine Uhrzeit"
                )
                
                test_end_dt = datetime.combine(test_end_date, test_end_time).replace(tzinfo=timezone.utc)
                
                # Warnung wenn außerhalb des verfügbaren Bereichs
                if min_datetime and test_end_dt < min_datetime:
                    st.warning(f"⚠️ Ende-Zeitpunkt liegt vor dem ältesten Eintrag ({min_datetime.strftime('%d.%m.%Y %H:%M')})")
                elif max_datetime and test_end_dt > max_datetime:
                    st.warning(f"⚠️ Ende-Zeitpunkt liegt nach dem neuesten Eintrag ({max_datetime.strftime('%d.%m.%Y %H:%M')})")
            
            # Overlap-Check Info
            if selected_model.get('train_start') and selected_model.get('train_end'):
                train_start_dt = datetime.fromisoformat(selected_model['train_start'].replace('Z', '+00:00'))
                train_end_dt = datetime.fromisoformat(selected_model['train_end'].replace('Z', '+00:00'))
                
                # Prüfe Overlap
                if test_start_dt < train_end_dt and test_end_dt > train_start_dt:
                    overlap_duration = min(test_end_dt, train_end_dt) - max(test_start_dt, train_start_dt)
                    total_duration = test_end_dt - test_start_dt
                    overlap_pct = (overlap_duration.total_seconds() / total_duration.total_seconds() * 100) if total_duration.total_seconds() > 0 else 0
                    st.warning(f"⚠️ {overlap_pct:.1f}% Überschneidung mit Trainingsdaten (Test wird trotzdem ausgeführt)")
            
            # Submit
            if st.button("🧪 Test starten", type="primary", use_container_width=True):
                if test_start_dt >= test_end_dt:
                    st.error("❌ Start-Zeitpunkt muss vor End-Zeitpunkt liegen!")
                else:
                    with st.spinner("🔄 Erstelle Test-Job..."):
                        # Konvertiere datetime zu UTC ISO-Format
                        test_start_iso = test_start_dt.astimezone(timezone.utc).isoformat().replace('+00:00', 'Z')
                        test_end_iso = test_end_dt.astimezone(timezone.utc).isoformat().replace('+00:00', 'Z')
                        
                        data = {
                            "test_start": test_start_iso,
                            "test_end": test_end_iso
                        }
                        
                        result = api_post(f"/api/models/{selected_model_id}/test", data)
                        
                        if result:
                            st.success(f"✅ Test-Job erstellt! Job-ID: {result.get('job_id')}")
                            st.info(f"📊 Status: {result.get('status')}. Der Test wird jetzt ausgeführt.")
                            
                            if st.button("📊 Zu Jobs anzeigen"):
                                st.session_state['page'] = 'jobs'
                                st.rerun()


