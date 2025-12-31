"""
Test Results Page Module
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


def page_test_results():
    """Test-Ergebnisse Übersicht: Liste aller Modell-Tests"""
    st.title("🧪 Test-Ergebnisse Übersicht")
    
    # Initialisiere selected_tests in session_state
    if 'selected_test_ids' not in st.session_state:
        st.session_state['selected_test_ids'] = []
    
    # Lade Test-Ergebnisse
    test_results = api_get("/api/test-results")
    if not test_results or not isinstance(test_results, list):
        st.warning("⚠️ Keine Test-Ergebnisse gefunden")
        return
    
    st.info(f"📊 {len(test_results)} Test-Ergebnis(se) gefunden")
    
    # Kompakte Karten-Ansicht
    if test_results:
        st.subheader("📋 Test-Ergebnisse")
        
        # Erstelle Karten in einem Grid (2 Spalten)
        cols = st.columns(2)
        
        for idx, test in enumerate(test_results):
            test_id = test.get('id')
            model_id = test.get('model_id')
            created = test.get('created_at', '')[:10] if test.get('created_at') else "N/A"
            test_start = test.get('test_start')
            test_end = test.get('test_end')
            
            # Hole Modell-Name und Trainings-Zeitraum
            model = api_get(f"/api/models/{model_id}")
            model_name = model.get('name', f"ID: {model_id}") if model else f"ID: {model_id}"
            model_train_start = model.get('train_start') if model else None
            model_train_end = model.get('train_end') if model else None
            
            # Metriken
            accuracy = test.get('accuracy', 0)
            f1 = test.get('f1_score', 0)
            mcc = test.get('mcc')
            profit = test.get('simulated_profit_pct')
            
            # Checkbox
            is_selected = test_id in st.session_state.get('selected_test_ids', [])
            checkbox_key = f"test_checkbox_{test_id}"
            
            # Wähle Spalte (abwechselnd)
            col = cols[idx % 2]
            
            with col:
                # Karte mit Border
                card_style = """
                <style>
                .test-card {
                    border: 2px solid #e0e0e0;
                    border-radius: 8px;
                    padding: 12px;
                    margin-bottom: 12px;
                    background: white;
                }
                .test-card.selected {
                    border-color: #1f77b4;
                    background: #f0f8ff;
                }
                </style>
                """
                st.markdown(card_style, unsafe_allow_html=True)
                
                # Header mit Checkbox und Titel
                header_col1, header_col2, header_col3 = st.columns([0.3, 4, 0.6])
                with header_col1:
                    checked = st.checkbox("", value=is_selected, key=checkbox_key, label_visibility="collapsed")
                    # Update session_state ohne st.rerun() - Streamlit rendert automatisch neu
                    if checked and test_id not in st.session_state.get('selected_test_ids', []):
                        if 'selected_test_ids' not in st.session_state:
                            st.session_state['selected_test_ids'] = []
                        st.session_state['selected_test_ids'].append(test_id)
                    elif not checked and test_id in st.session_state.get('selected_test_ids', []):
                        st.session_state['selected_test_ids'].remove(test_id)
                
                with header_col2:
                    st.markdown(f"**{model_name}**")
                
                with header_col3:
                    if st.button("📋", key=f"test_details_{test_id}", help="Details anzeigen", use_container_width=True):
                        st.session_state['test_details_id'] = test_id
                        st.session_state['page'] = 'test_details'
                        placeholder = st.empty()
                        placeholder.empty()
                        st.rerun()
                
                # Metriken kompakt - Erweitert
                metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
                with metric_col1:
                    st.metric("Accuracy", f"{accuracy:.3f}", help="Anteil korrekter Vorhersagen (0-1, höher = besser)", label_visibility="visible")
                with metric_col2:
                    st.metric("F1-Score", f"{f1:.3f}", help="Harmonisches Mittel aus Precision und Recall (0-1, höher = besser)", label_visibility="visible")
                with metric_col3:
                    roc_auc = test.get('roc_auc')
                    if roc_auc:
                        st.metric("ROC-AUC", f"{roc_auc:.3f}", help="Area Under ROC Curve (0-1, >0.7 = gut)", label_visibility="visible")
                    else:
                        st.caption("ROC-AUC: N/A")
                with metric_col4:
                    if mcc:
                        st.metric("MCC", f"{mcc:.3f}", help="Matthews Correlation Coefficient (-1 bis +1, höher = besser)", label_visibility="visible")
                    elif profit:
                        st.metric("Profit", f"{profit:.2f}%", help="Simulierter Profit basierend auf TP/FP", label_visibility="visible")
                    else:
                        st.caption("MCC/Profit: N/A")
                
                # Zusätzliche Infos
                info_row1, info_row2 = st.columns(2)
                
                with info_row1:
                    # Trainings-Zeitraum des Modells (mit Uhrzeit)
                    if model_train_start and model_train_end:
                        try:
                            train_start_dt = model_train_start if isinstance(model_train_start, str) else model_train_start
                            train_end_dt = model_train_end if isinstance(model_train_end, str) else model_train_end
                            if isinstance(train_start_dt, str):
                                train_start_dt = datetime.fromisoformat(train_start_dt.replace('Z', '+00:00'))
                            if isinstance(train_end_dt, str):
                                train_end_dt = datetime.fromisoformat(train_end_dt.replace('Z', '+00:00'))
                            
                            train_start_str = train_start_dt.strftime("%d.%m.%Y %H:%M")
                            train_end_str = train_end_dt.strftime("%d.%m.%Y %H:%M")
                            train_days = (train_end_dt - train_start_dt).total_seconds() / 86400.0
                            st.caption(f"🎓 Training: {train_start_str} → {train_end_str} ({train_days:.1f} Tage)")
                        except:
                            st.caption("🎓 Training: Zeitraum verfügbar")
                    else:
                        st.caption("🎓 Training: Zeitraum nicht verfügbar")
                    
                    # Test-Zeitraum mit Uhrzeit
                    if test_start and test_end:
                        try:
                            start_dt = test_start if isinstance(test_start, str) else test_start
                            end_dt = test_end if isinstance(test_end, str) else test_end
                            if isinstance(start_dt, str):
                                start_dt = datetime.fromisoformat(start_dt.replace('Z', '+00:00'))
                            if isinstance(end_dt, str):
                                end_dt = datetime.fromisoformat(end_dt.replace('Z', '+00:00'))
                            
                            start_str = start_dt.strftime("%d.%m.%Y %H:%M")
                            end_str = end_dt.strftime("%d.%m.%Y %H:%M")
                            days = (end_dt - start_dt).total_seconds() / 86400.0
                            st.caption(f"📅 Test: {start_str} → {end_str} ({days:.1f} Tage)")
                        except:
                            st.caption("📅 Test: Zeitraum verfügbar")
                    else:
                        st.caption("📅 Test: Zeitraum nicht verfügbar")
                    
                    # Overfitting Warnung
                    if test.get('is_overfitted'):
                        st.warning("⚠️ Möglicherweise overfitted")
                    elif test.get('accuracy_degradation'):
                        degradation = test.get('accuracy_degradation')
                        if degradation and degradation > 0.1:
                            st.warning(f"⚠️ Performance-Degradation: {degradation:.2%}")
                
                with info_row2:
                    # Anzahl Samples
                    num_samples = test.get('num_samples')
                    if num_samples:
                        st.caption(f"📊 {num_samples} Test-Samples")
                    
                    # Train vs. Test Vergleich
                    train_accuracy = test.get('train_accuracy')
                    if train_accuracy and accuracy:
                        degradation = test.get('accuracy_degradation', 0)
                        if degradation:
                            quality = "✅ OK" if degradation < 0.1 else "⚠️ Degradation"
                            st.caption(f"📈 Train: {train_accuracy:.3f} → Test: {accuracy:.3f} ({quality})")
                    
                    # Erstellt-Datum mit Uhrzeit
                    created_raw = test.get('created_at', '')
                    if created_raw:
                        try:
                            if isinstance(created_raw, str):
                                created_dt = datetime.fromisoformat(created_raw.replace('Z', '+00:00'))
                            else:
                                created_dt = created_raw
                            created_str = created_dt.strftime("%d.%m.%Y %H:%M")
                            st.caption(f"🕐 Erstellt: {created_str}")
                        except:
                            st.caption(f"🕐 Erstellt: {str(created_raw)[:19] if len(str(created_raw)) > 19 else str(created_raw)}")
                    else:
                        st.caption("🕐 Erstellt: N/A")
                
                # Dünne graue Linie zur Trennung
                if idx < len(test_results) - 1:
                    st.markdown("<hr style='margin: 10px 0; border: none; border-top: 1px solid #e0e0e0;'>", unsafe_allow_html=True)
        
        # Zeige ausgewählte Tests
        selected_test_ids = st.session_state.get('selected_test_ids', [])
        # Filtere nur existierende Tests
        selected_test_ids = [tid for tid in selected_test_ids if any(t.get('id') == tid for t in test_results)]
        # Aktualisiere session_state falls Tests entfernt wurden
        if len(selected_test_ids) != len(st.session_state.get('selected_test_ids', [])):
            st.session_state['selected_test_ids'] = selected_test_ids
        
        selected_count = len(selected_test_ids)
        if selected_count > 0:
            st.divider()
            st.subheader(f"🔧 Aktionen ({selected_count} Test(s) ausgewählt)")
            
            selected_tests = [t for t in test_results if t.get('id') in selected_test_ids]
            
            # Zeige ausgewählte Tests
            if selected_count <= 3:
                selected_names = [f"Test {t.get('id')}" for t in selected_tests]
                st.info(f"📌 Ausgewählt: {', '.join(selected_names)}")
            
            # Aktionen
            col1, col2 = st.columns(2)
            
            with col1:
                if selected_count == 1:
                    test_id = selected_test_ids[0]
                    if st.button("📋 Details anzeigen", key="btn_test_details", use_container_width=True, type="primary"):
                        st.session_state['test_details_id'] = test_id
                        st.session_state['page'] = 'test_details'
                        placeholder = st.empty()
                        placeholder.empty()
                        st.rerun()
            
            with col2:
                if st.button("🗑️ Löschen", key="btn_delete_tests", use_container_width=True, type="secondary"):
                    deleted_count = 0
                    failed_count = 0
                    ids_to_delete = list(selected_test_ids)
                    for test_id in ids_to_delete:
                        if api_delete(f"/api/test-results/{test_id}"):
                            deleted_count += 1
                            if test_id in st.session_state.get('selected_test_ids', []):
                                st.session_state['selected_test_ids'].remove(test_id)
                        else:
                            failed_count += 1
                    
                    if deleted_count > 0:
                        st.success(f"✅ {deleted_count} Test(s) gelöscht")
                    if failed_count > 0:
                        st.error(f"❌ {failed_count} Fehler")
                    
                    if deleted_count > 0:
                        st.rerun()


