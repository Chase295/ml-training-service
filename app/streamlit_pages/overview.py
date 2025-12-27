"""
Overview Page Module
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


def page_overview():
    """Übersicht: Liste aller Modelle"""
    st.title("🏠 Übersicht - ML Modelle")
    
    # Initialisiere selected_models in session_state
    if 'selected_model_ids' not in st.session_state:
        st.session_state['selected_model_ids'] = []
    
    # Lade Modelle ZUERST (wird für "Alle auswählen" benötigt)
    models = api_get("/api/models")
    if not models:
        st.warning("⚠️ Keine Modelle gefunden oder API-Fehler")
        return
    
    # Filter
    col1, col2, col3 = st.columns(3)
    with col1:
        status_filter = st.selectbox(
            "Status Filter",
            ["Alle", "READY", "TRAINING", "FAILED"],
            key="status_filter"
        )
    with col2:
        model_type_filter = st.selectbox(
            "Modell-Typ Filter",
            ["Alle", "random_forest", "xgboost"],
            key="model_type_filter"
        )
    
    # Filter anwenden
    filtered_models = models
    if status_filter != "Alle":
        filtered_models = [m for m in filtered_models if m.get('status') == status_filter]
    if model_type_filter != "Alle":
        filtered_models = [m for m in filtered_models if m.get('model_type') == model_type_filter]
    
    st.info(f"📊 {len(filtered_models)} Modell(e) gefunden")
    
    # Kompakte Karten-Ansicht
    if filtered_models:
        st.subheader("📋 Modelle")
        
        # Erstelle Karten in einem Grid (2 Spalten)
        cols = st.columns(2)
        
        for idx, model in enumerate(filtered_models):
            model_id = model.get('id')
            model_name = model.get('name', f"ID: {model_id}")
            model_type = model.get('model_type', 'N/A')
            status = model.get('status', 'N/A')
            accuracy = model.get('training_accuracy')
            f1 = model.get('training_f1')
            created_raw = model.get('created_at', '')
            train_start = model.get('train_start')
            train_end = model.get('train_end')
            
            # Checkbox
            is_selected = model_id in st.session_state.get('selected_model_ids', [])
            checkbox_key = f"checkbox_{model_id}"
            
            # Wähle Spalte (abwechselnd)
            col = cols[idx % 2]
            
            with col:
                # Karte mit Border
                card_style = """
                <style>
                .model-card {
                    border: 2px solid #e0e0e0;
                    border-radius: 8px;
                    padding: 12px;
                    margin-bottom: 12px;
                    background: white;
                }
                .model-card.selected {
                    border-color: #1f77b4;
                    background: #f0f8ff;
                }
                </style>
                """
                st.markdown(card_style, unsafe_allow_html=True)
                
                # Header mit Checkbox und Name
                header_col1, header_col2, header_col3, header_col4 = st.columns([0.3, 4, 0.6, 0.6])
                with header_col1:
                    checked = st.checkbox("", value=is_selected, key=checkbox_key, label_visibility="collapsed")
                    # Update session_state ohne st.rerun() - Streamlit rendert automatisch neu
                    if checked and model_id not in st.session_state.get('selected_model_ids', []):
                        if 'selected_model_ids' not in st.session_state:
                            st.session_state['selected_model_ids'] = []
                        st.session_state['selected_model_ids'].append(model_id)
                    elif not checked and model_id in st.session_state.get('selected_model_ids', []):
                        st.session_state['selected_model_ids'].remove(model_id)
                
                with header_col2:
                    # Name mit Umbenennen
                    if st.session_state.get(f'renaming_{model_id}', False):
                        # Popup-ähnliches Verhalten mit Expander
                        with st.expander("✏️ Modell bearbeiten", expanded=True):
                            new_name = st.text_input("Name *", value=model_name, key=f"new_name_{model_id}")
                            new_desc = st.text_area("Beschreibung", value=model.get('description', '') or '', key=f"new_desc_{model_id}", height=80)
                            
                            # Buttons nebeneinander ohne verschachtelte Spalten
                            if st.button("💾 Speichern", key=f"save_{model_id}", use_container_width=True, type="primary"):
                                if new_name and new_name.strip():
                                    data = {"name": new_name.strip()}
                                    if new_desc and new_desc.strip():
                                        data["description"] = new_desc.strip()
                                    result = api_patch(f"/api/models/{model_id}", data)
                                    if result:
                                        st.session_state[f'renaming_{model_id}'] = False
                                        st.success("✅ Modell erfolgreich umbenannt")
                                        st.rerun()
                                else:
                                    st.warning("⚠️ Name darf nicht leer sein")
                            
                            if st.button("❌ Abbrechen", key=f"cancel_{model_id}", use_container_width=True):
                                st.session_state[f'renaming_{model_id}'] = False
                                st.rerun()
                    else:
                        st.markdown(f"**{model_name}**")
                
                with header_col3:
                    if st.button("📋", key=f"details_{model_id}", help="Details anzeigen", use_container_width=True):
                        # Sofortige Navigation mit minimalem Zucken
                        st.session_state['details_model_id'] = model_id
                        st.session_state['page'] = 'details'
                        # Platzhalter für sofortiges Update ohne sichtbares Zucken
                        placeholder = st.empty()
                        placeholder.empty()  # Sofort leeren
                        st.rerun()
                
                with header_col4:
                    if not st.session_state.get(f'renaming_{model_id}', False):
                        if st.button("✏️", key=f"rename_{model_id}", help="Umbenennen", use_container_width=True):
                            st.session_state[f'renaming_{model_id}'] = True
                            st.rerun()
                
                # Kompakte Info-Zeile
                info_col1, info_col2, info_col3 = st.columns(3)
                
                with info_col1:
                    type_emoji = "🌲" if model_type == "random_forest" else "🚀" if model_type == "xgboost" else "🤖"
                    st.caption(f"{type_emoji} {model_type}")
                
                with info_col2:
                    if status == "READY":
                        st.caption("✅ READY")
                    elif status == "TRAINING":
                        st.caption("🔄 TRAINING")
                    else:
                        st.caption(f"❌ {status}")
                
                with info_col3:
                    st.caption(f"#{model_id}")
                
                # Metriken kompakt - Erweitert
                metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
                with metric_col1:
                    if accuracy:
                        st.metric("Accuracy", f"{accuracy:.3f}", help="Anteil korrekter Vorhersagen (0-1, höher = besser)", label_visibility="visible")
                    else:
                        st.caption("Accuracy: N/A")
                with metric_col2:
                    if f1:
                        st.metric("F1-Score", f"{f1:.3f}", help="Harmonisches Mittel aus Precision und Recall (0-1, höher = besser)", label_visibility="visible")
                    else:
                        st.caption("F1-Score: N/A")
                with metric_col3:
                    roc_auc = model.get('roc_auc')
                    if roc_auc:
                        st.metric("ROC-AUC", f"{roc_auc:.3f}", help="Area Under ROC Curve (0-1, >0.7 = gut)", label_visibility="visible")
                    else:
                        st.caption("ROC-AUC: N/A")
                with metric_col4:
                    mcc = model.get('mcc')
                    if mcc:
                        st.metric("MCC", f"{mcc:.3f}", help="Matthews Correlation Coefficient (-1 bis +1, höher = besser)", label_visibility="visible")
                    else:
                        st.caption("MCC: N/A")
                
                # Zusätzliche Infos
                info_row1, info_row2 = st.columns(2)
                
                with info_row1:
                    # Features
                    features_list = model.get('features', [])
                    if features_list:
                        num_features = len(features_list)
                        st.caption(f"📊 {num_features} Features")
                    
                    # Zeitbasierte Vorhersage Info - DEUTLICH SICHTBAR
                    # Prüfe zuerst direkte Felder aus der DB
                    future_minutes = model.get('future_minutes')
                    price_change = model.get('price_change_percent') or model.get('min_percent_change')
                    direction = model.get('target_direction')
                    
                    # Falls nicht in direkten Feldern, prüfe in params
                    if not future_minutes or not price_change:
                        params = model.get('params', {})
                        if isinstance(params, str):
                            import json
                            try:
                                params = json.loads(params)
                            except:
                                params = {}
                        
                        time_based_params = params.get('_time_based', {})
                        if time_based_params and time_based_params.get('enabled'):
                            future_minutes = future_minutes or time_based_params.get('future_minutes')
                            price_change = price_change or time_based_params.get('min_percent_change')
                            direction = direction or time_based_params.get('direction')
                    
                    # Zeige zeitbasierte Vorhersage wenn vorhanden
                    if future_minutes and price_change:
                        direction_emoji = "📈" if direction == "up" else "📉" if direction == "down" else ""
                        direction_text = "steigt" if direction == "up" else "fällt" if direction == "down" else ""
                        # Größere, sichtbarere Anzeige
                        st.markdown(f"**⏰ Zeitbasierte Vorhersage:** {future_minutes}min, {price_change}% {direction_text} {direction_emoji}")
                    else:
                        # Normale Ziel-Variable
                        target_var = model.get('target_variable', 'N/A')
                        target_operator = model.get('target_operator')
                        target_value = model.get('target_value')
                        if target_operator and target_value is not None:
                            st.markdown(f"**🎯 Ziel:** {target_var} {target_operator} {target_value}")
                        else:
                            st.caption("🎯 Ziel: Nicht konfiguriert")
                    
                    # Feature-Engineering Info
                    params = model.get('params', {})
                    if isinstance(params, str):
                        import json
                        try:
                            params = json.loads(params)
                        except:
                            params = {}
                    if params.get('use_engineered_features'):
                        st.caption("🔧 Feature-Engineering: ✅")
                
                with info_row2:
                    # Trainingszeitraum mit Uhrzeit
                    if train_start and train_end:
                        try:
                            start_dt = train_start if isinstance(train_start, str) else train_start
                            end_dt = train_end if isinstance(train_end, str) else train_end
                            if isinstance(start_dt, str):
                                start_dt = datetime.fromisoformat(start_dt.replace('Z', '+00:00'))
                            if isinstance(end_dt, str):
                                end_dt = datetime.fromisoformat(end_dt.replace('Z', '+00:00'))
                            
                            start_str = start_dt.strftime("%d.%m.%Y %H:%M")
                            end_str = end_dt.strftime("%d.%m.%Y %H:%M")
                            days = (end_dt - start_dt).total_seconds() / 86400.0
                            st.caption(f"📅 Training: {start_str} → {end_str} ({days:.1f} Tage)")
                        except:
                            st.caption("📅 Training: Zeitraum verfügbar")
                    else:
                        st.caption("📅 Training: Zeitraum nicht verfügbar")
                    
                    # Erstellt-Datum mit Uhrzeit
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
                if idx < len(filtered_models) - 1:
                    st.markdown("<hr style='margin: 10px 0; border: none; border-top: 1px solid #e0e0e0;'>", unsafe_allow_html=True)
        
        # Zeige ausgewählte Modelle
        selected_model_ids = st.session_state.get('selected_model_ids', [])
        # Filtere nur existierende Modelle
        selected_model_ids = [mid for mid in selected_model_ids if any(m.get('id') == mid for m in filtered_models)]
        # Aktualisiere session_state falls Modelle entfernt wurden
        if len(selected_model_ids) != len(st.session_state.get('selected_model_ids', [])):
            st.session_state['selected_model_ids'] = selected_model_ids
        
        selected_count = len(selected_model_ids)
        if selected_count > 0:
            st.divider()
            st.subheader(f"🔧 Aktionen ({selected_count} Modell(e) ausgewählt)")
            
            selected_models = [m for m in filtered_models if m.get('id') in selected_model_ids]
            
            # Zeige ausgewählte Modelle
            if selected_count <= 3:
                selected_names = [f"{m.get('name')} (ID: {m.get('id')})" for m in selected_models]
                st.info(f"📌 Ausgewählt: {', '.join(selected_names)}")
            
            # Aktionen basierend auf Anzahl
            if selected_count == 1:
                # 1 Modell: Testen, Löschen, Details
                model_id = selected_model_ids[0]
                selected_model = selected_models[0]
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    test_clicked = st.button("🧪 Testen", key="btn_test", use_container_width=True)
                    if test_clicked:
                        st.session_state['test_model_id'] = model_id
                        st.session_state['page'] = 'test'
                        st.rerun()
                
                with col2:
                    details_clicked = st.button("📋 Details", key="btn_details", use_container_width=True)
                    if details_clicked:
                        st.session_state['details_model_id'] = model_id
                        st.session_state['page'] = 'details'
                        placeholder = st.empty()
                        placeholder.empty()
                        st.rerun()
                        st.rerun()
                
                with col3:
                    download_clicked = st.button("📥 Download", key="btn_download", use_container_width=True)
                    if download_clicked:
                        download_url = f"{API_BASE_URL}/api/models/{model_id}/download"
                        st.markdown(f"[⬇️ Modell herunterladen]({download_url})")
                
                with col4:
                    delete_clicked = st.button("🗑️ Löschen", key="btn_delete", use_container_width=True, type="secondary")
                    if delete_clicked:
                        if api_delete(f"/api/models/{model_id}"):
                            st.success("✅ Modell gelöscht")
                            # Entferne aus session_state
                            if model_id in st.session_state.get('selected_model_ids', []):
                                st.session_state['selected_model_ids'].remove(model_id)
                            st.rerun()
                        else:
                            st.error("❌ Fehler beim Löschen des Modells")
            
            elif selected_count == 2:
                # 2 Modelle: Vergleichen, Löschen
                model_a_id = selected_model_ids[0]
                model_b_id = selected_model_ids[1]
                
                col1, col2 = st.columns(2)
                
                with col1:
                    if st.button("⚔️ Vergleichen", key="btn_compare", use_container_width=True, type="primary"):
                        st.session_state['compare_model_a_id'] = model_a_id
                        st.session_state['compare_model_b_id'] = model_b_id
                        st.session_state['page'] = 'compare'
                        st.rerun()
            
                with col2:
                    if st.button("🗑️ Beide löschen", key="btn_delete_both", use_container_width=True, type="secondary"):
                        # Lösche beide Modelle
                        ids_to_delete = list(selected_model_ids)
                        deleted_count = 0
                        failed_count = 0
                        
                        for model_id in ids_to_delete:
                            if api_delete(f"/api/models/{model_id}"):
                                deleted_count += 1
                                if model_id in st.session_state.get('selected_model_ids', []):
                                    st.session_state['selected_model_ids'].remove(model_id)
                            else:
                                failed_count += 1
                        
                        if deleted_count > 0:
                            st.success(f"✅ {deleted_count} Modell(e) gelöscht")
                        if failed_count > 0:
                            st.error(f"❌ {failed_count} Fehler")
                        
                        if deleted_count > 0:
                            st.rerun()
            
            else:
                # Mehr als 2: Nur Löschen
                if st.button("🗑️ Alle ausgewählten löschen", key="btn_delete_all", use_container_width=True, type="secondary"):
                    deleted_count = 0
                    failed_count = 0
                    # Kopiere Liste um während Iteration zu ändern - verwende die tatsächliche Liste aus session_state
                    ids_to_delete = list(st.session_state.get('selected_model_ids', []))
                    for model_id in ids_to_delete:
                        if api_delete(f"/api/models/{model_id}"):
                            deleted_count += 1
                            # Entferne aus session_state
                            if model_id in st.session_state.get('selected_model_ids', []):
                                st.session_state['selected_model_ids'].remove(model_id)
                        else:
                            failed_count += 1
                    
                    # Immer rerun wenn etwas passiert ist
                    if deleted_count > 0 or failed_count > 0:
                        if deleted_count > 0:
                            if failed_count > 0:
                                st.warning(f"⚠️ {deleted_count} Modell(e) gelöscht, {failed_count} Fehler")
                            else:
                                st.success(f"✅ {deleted_count} Modell(e) gelöscht")
                        if failed_count > 0 and deleted_count == 0:
                            st.error(f"❌ Fehler beim Löschen von {failed_count} Modell(en)")
                        st.rerun()
        else:
            st.info("💡 Wähle ein oder mehrere Modelle aus, um Aktionen auszuführen")
    else:
        st.info("ℹ️ Keine Modelle gefunden")


