"""
Comparisons Page Module
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


def page_comparisons():
    """Vergleichs-Übersicht: Liste aller Modell-Vergleiche"""
    st.title("⚖️ Vergleichs-Übersicht")
    
    # Initialisiere selected_comparisons in session_state
    if 'selected_comparison_ids' not in st.session_state:
        st.session_state['selected_comparison_ids'] = []
    
    # Lade Vergleichs-Ergebnisse
    comparisons = api_get("/api/comparisons")
    if not comparisons:
        st.warning("⚠️ Keine Vergleichs-Ergebnisse gefunden")
        return
    
    st.info(f"📊 {len(comparisons)} Vergleich(e) gefunden")
    
    # Kompakte Karten-Ansicht
    if comparisons:
        st.subheader("📋 Vergleichs-Ergebnisse")
        
        # Erstelle Karten in einem Grid (2 Spalten)
        cols = st.columns(2)
        
        for idx, comp in enumerate(comparisons):
            comp_id = comp.get('id')
            model_a_id = comp.get('model_a_id')
            model_b_id = comp.get('model_b_id')
            winner_id = comp.get('winner_id')
            created = comp.get('created_at', '')[:10] if comp.get('created_at') else "N/A"
            test_start = comp.get('test_start')
            test_end = comp.get('test_end')
            
            # Hole Modell-Namen und Trainings-Zeiträume
            model_a = api_get(f"/api/models/{model_a_id}")
            model_b = api_get(f"/api/models/{model_b_id}")
            model_a_name = model_a.get('name', f"ID: {model_a_id}") if model_a else f"ID: {model_a_id}"
            model_b_name = model_b.get('name', f"ID: {model_b_id}") if model_b else f"ID: {model_b_id}"
            model_a_train_start = model_a.get('train_start') if model_a else None
            model_a_train_end = model_a.get('train_end') if model_a else None
            model_b_train_start = model_b.get('train_start') if model_b else None
            model_b_train_end = model_b.get('train_end') if model_b else None
            
            # Metriken aus Test-Ergebnissen laden (statt aus Vergleich)
            # Extrahiere Metriken direkt aus Vergleichs-Objekt (bereits aus Test-Ergebnissen geladen)
            a_accuracy = comp.get('a_accuracy', 0)
            a_f1 = comp.get('a_f1', 0)
            a_roc_auc = comp.get('a_roc_auc')
            a_mcc = comp.get('a_mcc')
            a_profit = comp.get('a_simulated_profit_pct')

            b_accuracy = comp.get('b_accuracy', 0)
            b_f1 = comp.get('b_f1', 0)
            b_roc_auc = comp.get('b_roc_auc')
            b_mcc = comp.get('b_mcc')
            b_profit = comp.get('b_simulated_profit_pct')
            
            # Checkbox
            is_selected = comp_id in st.session_state.get('selected_comparison_ids', [])
            checkbox_key = f"comp_checkbox_{comp_id}"
            
            # Wähle Spalte (abwechselnd)
            col = cols[idx % 2]
            
            with col:
                # Karte mit Border
                card_style = """
                <style>
                .comparison-card {
                    border: 2px solid #e0e0e0;
                    border-radius: 8px;
                    padding: 12px;
                    margin-bottom: 12px;
                    background: white;
                }
                .comparison-card.selected {
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
                    if checked and comp_id not in st.session_state.get('selected_comparison_ids', []):
                        if 'selected_comparison_ids' not in st.session_state:
                            st.session_state['selected_comparison_ids'] = []
                        st.session_state['selected_comparison_ids'].append(comp_id)
                    elif not checked and comp_id in st.session_state.get('selected_comparison_ids', []):
                        st.session_state['selected_comparison_ids'].remove(comp_id)
                
                with header_col2:
                    st.markdown(f"**{model_a_name} vs {model_b_name}**")
                
                with header_col3:
                    if st.button("📋", key=f"comp_details_{comp_id}", help="Details anzeigen", use_container_width=True):
                        # Sofortige Navigation mit minimalem Zucken
                        st.session_state['comparison_details_id'] = comp_id
                        st.session_state['page'] = 'comparison_details'
                        # Platzhalter für sofortiges Update ohne sichtbares Zucken
                        placeholder = st.empty()
                        placeholder.empty()  # Sofort leeren
                        st.rerun()
                
                # Gewinner
                if winner_id:
                    if winner_id == model_a_id:
                        st.success(f"🏆 Gewinner: {model_a_name}")
                    elif winner_id == model_b_id:
                        st.success(f"🏆 Gewinner: {model_b_name}")
                else:
                    st.info("🤝 Unentschieden")
                
                # Metriken kompakt - Erweitert
                st.markdown("**📊 Metriken:**")
                
                # Modell A Metriken
                st.markdown(f"**Modell A ({model_a_name}):**")
                metric_a_col1, metric_a_col2, metric_a_col3, metric_a_col4 = st.columns(4)
                with metric_a_col1:
                    st.metric("Accuracy", f"{a_accuracy:.3f}", help="Anteil korrekter Vorhersagen (0-1, höher = besser)", label_visibility="visible")
                with metric_a_col2:
                    st.metric("F1-Score", f"{a_f1:.3f}", help="Harmonisches Mittel aus Precision und Recall (0-1, höher = besser)", label_visibility="visible")
                with metric_a_col3:
                    if a_roc_auc:
                        st.metric("ROC-AUC", f"{a_roc_auc:.3f}", help="Area Under ROC Curve (0-1, >0.7 = gut)", label_visibility="visible")
                    else:
                        st.caption("ROC-AUC: N/A")
                with metric_a_col4:
                    if a_mcc:
                        st.metric("MCC", f"{a_mcc:.3f}", help="Matthews Correlation Coefficient (-1 bis +1, höher = besser)", label_visibility="visible")
                    elif a_profit:
                        st.metric("Profit", f"{a_profit:.2f}%", help="Simulierter Profit basierend auf TP/FP", label_visibility="visible")
                    else:
                        st.caption("MCC/Profit: N/A")
                
                # Modell B Metriken
                st.markdown(f"**Modell B ({model_b_name}):**")
                metric_b_col1, metric_b_col2, metric_b_col3, metric_b_col4 = st.columns(4)
                with metric_b_col1:
                    st.metric("Accuracy", f"{b_accuracy:.3f}", help="Anteil korrekter Vorhersagen (0-1, höher = besser)", label_visibility="visible")
                with metric_b_col2:
                    st.metric("F1-Score", f"{b_f1:.3f}", help="Harmonisches Mittel aus Precision und Recall (0-1, höher = besser)", label_visibility="visible")
                with metric_b_col3:
                    if b_roc_auc:
                        st.metric("ROC-AUC", f"{b_roc_auc:.3f}", help="Area Under ROC Curve (0-1, >0.7 = gut)", label_visibility="visible")
                    else:
                        st.caption("ROC-AUC: N/A")
                with metric_b_col4:
                    if b_mcc:
                        st.metric("MCC", f"{b_mcc:.3f}", help="Matthews Correlation Coefficient (-1 bis +1, höher = besser)", label_visibility="visible")
                    elif b_profit:
                        st.metric("Profit", f"{b_profit:.2f}%", help="Simulierter Profit basierend auf TP/FP", label_visibility="visible")
                    else:
                        st.caption("MCC/Profit: N/A")
                
                # Zusätzliche Infos
                info_row1, info_row2 = st.columns(2)
                
                with info_row1:
                    # Trainings-Zeiträume der Modelle
                    if model_a_train_start and model_a_train_end:
                        try:
                            train_start_dt = model_a_train_start if isinstance(model_a_train_start, str) else model_a_train_start
                            train_end_dt = model_a_train_end if isinstance(model_a_train_end, str) else model_a_train_end
                            if isinstance(train_start_dt, str):
                                train_start_dt = datetime.fromisoformat(train_start_dt.replace('Z', '+00:00'))
                            if isinstance(train_end_dt, str):
                                train_end_dt = datetime.fromisoformat(train_end_dt.replace('Z', '+00:00'))
                            
                            train_start_str = train_start_dt.strftime("%d.%m.%Y %H:%M")
                            train_end_str = train_end_dt.strftime("%d.%m.%Y %H:%M")
                            train_days = (train_end_dt - train_start_dt).total_seconds() / 86400.0
                            st.caption(f"🎓 Modell A Training: {train_start_str} → {train_end_str} ({train_days:.1f} Tage)")
                        except:
                            st.caption("🎓 Modell A Training: Zeitraum verfügbar")
                    
                    if model_b_train_start and model_b_train_end:
                        try:
                            train_start_dt = model_b_train_start if isinstance(model_b_train_start, str) else model_b_train_start
                            train_end_dt = model_b_train_end if isinstance(model_b_train_end, str) else model_b_train_end
                            if isinstance(train_start_dt, str):
                                train_start_dt = datetime.fromisoformat(train_start_dt.replace('Z', '+00:00'))
                            if isinstance(train_end_dt, str):
                                train_end_dt = datetime.fromisoformat(train_end_dt.replace('Z', '+00:00'))
                            
                            train_start_str = train_start_dt.strftime("%d.%m.%Y %H:%M")
                            train_end_str = train_end_dt.strftime("%d.%m.%Y %H:%M")
                            train_days = (train_end_dt - train_start_dt).total_seconds() / 86400.0
                            st.caption(f"🎓 Modell B Training: {train_start_str} → {train_end_str} ({train_days:.1f} Tage)")
                        except:
                            st.caption("🎓 Modell B Training: Zeitraum verfügbar")
                
                with info_row2:
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
                    
                    # Erstellt-Datum mit Uhrzeit
                    created_raw = comp.get('created_at', '')
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
                if idx < len(comparisons) - 1:
                    st.markdown("<hr style='margin: 10px 0; border: none; border-top: 1px solid #e0e0e0;'>", unsafe_allow_html=True)
        
        # Zeige ausgewählte Vergleiche
        selected_comparison_ids = st.session_state.get('selected_comparison_ids', [])
        # Filtere nur existierende Vergleiche
        selected_comparison_ids = [cid for cid in selected_comparison_ids if any(c.get('id') == cid for c in comparisons)]
        # Aktualisiere session_state falls Vergleiche entfernt wurden
        if len(selected_comparison_ids) != len(st.session_state.get('selected_comparison_ids', [])):
            st.session_state['selected_comparison_ids'] = selected_comparison_ids
        
        selected_count = len(selected_comparison_ids)
        if selected_count > 0:
            st.divider()
            st.subheader(f"🔧 Aktionen ({selected_count} Vergleich(e) ausgewählt)")
            
            selected_comparisons = [c for c in comparisons if c.get('id') in selected_comparison_ids]
            
            # Zeige ausgewählte Vergleiche
            if selected_count <= 3:
                selected_names = [f"Vergleich {c.get('id')}" for c in selected_comparisons]
                st.info(f"📌 Ausgewählt: {', '.join(selected_names)}")
            
            # Aktionen
            col1, col2 = st.columns(2)
            
            with col1:
                if selected_count == 1:
                    comp_id = selected_comparison_ids[0]
                    if st.button("📋 Details anzeigen", key="btn_comp_details", use_container_width=True, type="primary"):
                        # Sofortige Navigation mit minimalem Zucken
                        st.session_state['comparison_details_id'] = comp_id
                        st.session_state['page'] = 'comparison_details'
                        # Platzhalter für sofortiges Update ohne sichtbares Zucken
                        placeholder = st.empty()
                        placeholder.empty()  # Sofort leeren
                        st.rerun()
            
            with col2:
                if st.button("🗑️ Löschen", key="btn_delete_comparisons", use_container_width=True, type="secondary"):
                    deleted_count = 0
                    failed_count = 0
                    ids_to_delete = list(selected_comparison_ids)
                    for comp_id in ids_to_delete:
                        if api_delete(f"/api/comparisons/{comp_id}"):
                            deleted_count += 1
                            if comp_id in st.session_state.get('selected_comparison_ids', []):
                                st.session_state['selected_comparison_ids'].remove(comp_id)
                        else:
                            failed_count += 1
                    
                    if deleted_count > 0:
                        st.success(f"✅ {deleted_count} Vergleich(e) gelöscht")
                    if failed_count > 0:
                        st.error(f"❌ {failed_count} Fehler")
                    
                    if deleted_count > 0:
                        st.rerun()


