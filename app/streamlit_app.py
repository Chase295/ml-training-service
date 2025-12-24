"""
Streamlit UI für ML Training Service
Web-Interface für Modell-Management
"""
import streamlit as st
import os
import httpx
import pandas as pd
import plotly.express as px
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any, Optional

# Konfiguration
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

# Page Config
st.set_page_config(
    page_title="ML Training Service",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# Helper Functions
# ============================================================

def api_get(endpoint: str) -> Any:
    """GET Request zur API (kann Dict oder List zurückgeben)"""
    try:
        response = httpx.get(f"{API_BASE_URL}{endpoint}", timeout=30.0)
        response.raise_for_status()
        return response.json()
    except httpx.HTTPError as e:
        st.error(f"❌ API-Fehler: {e}")
        return [] if 'comparisons' in endpoint or 'models' in endpoint else {}

def api_post(endpoint: str, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """POST Request zur API"""
    try:
        response = httpx.post(f"{API_BASE_URL}{endpoint}", json=data, timeout=30.0)
        response.raise_for_status()
        return response.json()
    except httpx.HTTPError as e:
        st.error(f"❌ API-Fehler: {e}")
        if hasattr(e.response, 'text'):
            st.error(f"Details: {e.response.text}")
        return None

def api_delete(endpoint: str) -> bool:
    """DELETE Request zur API"""
    try:
        response = httpx.delete(f"{API_BASE_URL}{endpoint}", timeout=30.0)
        response.raise_for_status()
        return True
    except httpx.HTTPError as e:
        # Fehler wird nicht hier angezeigt, sondern in der aufrufenden Funktion
        # damit mehrere Löschungen nicht zu vielen Fehlermeldungen führen
        return False

def api_patch(endpoint: str, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """PATCH Request zur API"""
    try:
        response = httpx.patch(f"{API_BASE_URL}{endpoint}", json=data, timeout=30.0)
        response.raise_for_status()
        return response.json()
    except httpx.HTTPError as e:
        st.error(f"❌ API-Fehler: {e}")
        if hasattr(e, 'response') and hasattr(e.response, 'text'):
            st.error(f"Details: {e.response.text}")
        return None

# Verfügbare Features (aus coin_metrics)
# ⚠️ WICHTIG: Nur Spalten die tatsächlich in der Datenbank existieren!
# Diese Liste muss mit den tatsächlichen Spalten in coin_metrics übereinstimmen!
AVAILABLE_FEATURES = [
    "price_open", "price_high", "price_low", "price_close",
    "volume_sol",
    "market_cap_open", "market_cap_high", "market_cap_low", "market_cap_close"
    # ⚠️ Folgende Spalten existieren NICHT in der Datenbank:
    # - volume_usd
    # - order_buy_count, order_sell_count
    # - order_buy_volume, order_sell_volume
    # - whale_buy_count, whale_sell_count
    # - whale_buy_volume, whale_sell_volume
    # - buy_volume_sol, sell_volume_sol
]

# Verfügbare Target-Variablen
AVAILABLE_TARGETS = [
    "market_cap_close", "price_close", "volume_sol"  # volume_usd existiert nicht!
]

def load_phases() -> List[Dict[str, Any]]:
    """Lade Phasen aus der API"""
    phases_data = api_get("/api/phases")
    if isinstance(phases_data, list):
        return phases_data
    return []

# ============================================================
# Seiten
# ============================================================

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
                    if checked != is_selected:
                        if checked:
                            if model_id not in st.session_state['selected_model_ids']:
                                st.session_state['selected_model_ids'].append(model_id)
                                st.rerun()
                        else:
                            if model_id in st.session_state['selected_model_ids']:
                                st.session_state['selected_model_ids'].remove(model_id)
                                st.rerun()
                
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
                        st.session_state['details_model_id'] = model_id
                        st.session_state['page'] = 'details'
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

def page_train():
    """Neues Modell trainieren"""
    st.title("➕ Neues Modell trainieren")
    
    # Info-Box am Anfang
    st.info("""
    **📖 Anleitung:** 
    Diese Seite erstellt ein neues Machine-Learning-Modell. Alle Felder mit * sind Pflichtfelder.
    Nutze die ℹ️-Icons für detaillierte Erklärungen zu jedem Parameter.
    """)
    
    # NEU: Zeitbasierte Vorhersagen AUSSERHALB des Forms (damit sofort reagiert wird)
    with st.expander("⏰ Zeitbasierte Vorhersage (optional)", expanded=False):
        st.markdown("""
        **Was ist zeitbasierte Vorhersage?**
        
        Statt zu lernen "Ist price_close > 50000?", lernt das Modell:
        - "Steigt price_close in den nächsten 5 Minuten um 30%?"
        - "Fällt price_close in den nächsten 10 Minuten um 20%?"
        
        **Vorteil:** Das Modell kann zukünftige Preisbewegungen vorhersagen, nicht nur aktuelle Werte.
        """)
        
        use_time_based = st.checkbox(
            "✅ Zeitbasierte Vorhersage aktivieren", 
            value=True,  # Standardmäßig aktiviert
            help="Aktiviert die zeitbasierte Vorhersage. Wenn aktiviert, werden die Ziel-Variablen-Felder ausgeblendet.",
            key="use_time_based_checkbox"
        )
        
        if use_time_based:
            st.success("✅ Zeitbasierte Vorhersage ist aktiviert. Die Ziel-Variablen-Felder werden ausgeblendet.")
    
    # Feature-Engineering Checkbox DIREKT NACH Zeitbasierte Vorhersage (damit sofort reagiert wird)
    with st.expander("🔧 Feature-Engineering (optional)", expanded=True):
        st.markdown("""
        **Was ist Feature-Engineering?**
        
        Feature-Engineering erstellt automatisch zusätzliche Features aus den Basis-Features:
        - **Momentum:** Wie schnell ändert sich der Preis?
        - **Volumen-Patterns:** Gibt es ungewöhnliche Volumen-Spikes?
        - **Whale-Activity:** Große Käufe/Verkäufe erkannt?
        - **Volatilität:** Wie stark schwankt der Preis?
        - **Order-Book-Imbalance:** Verkaufsdruck vs. Kaufdruck
        
        **Auswirkung:** Aus ~6 Basis-Features werden ~40 erweiterte Features erstellt.
        Dies verbessert die Modell-Performance erheblich, besonders bei Pump-Detection.
        """)
        
        use_engineered_features = st.checkbox(
            "✅ Erweiterte Pump-Detection Features verwenden",
            value=True,  # Standardmäßig aktiviert
            help="Aktiviert Feature-Engineering. Erstellt ~40 zusätzliche Features aus den Basis-Features.",
            key="use_engineered_features_checkbox"
        )
        
        feature_engineering_windows = None
        if use_engineered_features:
            st.success("✅ Feature-Engineering ist aktiviert. ~40 zusätzliche Features werden erstellt.")
            
            st.markdown("**Fenstergrößen für Rolling-Berechnungen:**")
            st.caption("""
            Diese Fenster bestimmen, über welchen Zeitraum die Features berechnet werden:
            - **Kleine Fenster (3-5):** Erkennt kurzfristige Muster (Sekunden/Minuten)
            - **Mittlere Fenster (10-15):** Erkennt mittelfristige Trends
            - **Große Fenster (20-30):** Erkennt langfristige Trends
            """)
            
            # Optional: Fenstergrößen anpassen
            window_sizes = st.multiselect(
                "Fenstergrößen auswählen",
                options=[3, 5, 10, 15, 20, 30],
                default=[5, 10, 15],
                help="Wähle die Fenstergrößen für Rolling-Berechnungen. Mehr Fenster = mehr Features, aber langsameres Training.",
                key="feature_engineering_windows_select"
            )
            if window_sizes:
                feature_engineering_windows = window_sizes
            else:
                feature_engineering_windows = [5, 10, 15]  # Default
                st.info("ℹ️ Standard-Fenstergrößen [5, 10, 15] werden verwendet.")
    
    # SMOTE Checkbox
    with st.expander("⚖️ Imbalanced Data Handling (optional)", expanded=True):
        st.markdown("""
        **Was ist Imbalanced Data?**
        
        Bei Pump-Detection haben wir oft viel mehr "normale" Coins als "Pump"-Coins.
        Beispiel: 1000 normale Coins, 10 Pump-Coins → 99% vs. 1%
        
        **Problem:** Das Modell lernt nur "alles ist normal" und erreicht 99% Accuracy, aber erkennt keine Pumps.
        
        **Lösung: SMOTE** (Synthetic Minority Over-sampling Technique)
        - Erstellt künstliche "Pump"-Beispiele
        - Balanciert die Daten aus (z.B. 50% normal, 50% pump)
        - Modell kann beide Klassen lernen
        
        **Auswirkung:** Modell erkennt Pumps viel besser, auch wenn sie selten sind.
        """)
        
        use_smote = st.checkbox(
            "✅ SMOTE für Imbalanced Data aktivieren (empfohlen)",
            value=True,
            help="SMOTE wird automatisch angewendet, wenn Label-Balance < 30% oder > 70%",
            key="use_smote_checkbox"
        )
        
        if use_smote:
            st.success("✅ SMOTE ist aktiviert. Wird automatisch angewendet bei unausgewogenen Daten.")
            st.caption("ℹ️ SMOTE wird nur angewendet, wenn die Daten unausgewogen sind (< 30% oder > 70% einer Klasse).")
    
    # TimeSeriesSplit Checkbox
    with st.expander("🔀 Cross-Validation (optional)", expanded=True):
        st.markdown("""
        **Was ist Cross-Validation?**
        
        Cross-Validation teilt die Daten in mehrere Teile auf und testet das Modell auf jedem Teil.
        Dies gibt eine realistischere Einschätzung der Modell-Performance.
        
        **TimeSeriesSplit vs. normaler Split:**
        - **Normaler Split:** Zufällige Aufteilung → kann Daten aus der Zukunft zum Training verwenden (unrealistisch!)
        - **TimeSeriesSplit:** Respektiert die zeitliche Reihenfolge → Training nur mit vergangenen Daten (realistisch!)
        
        **Auswirkung:** Metriken sind realistischer, da das Modell nur mit vergangenen Daten trainiert wird.
        """)
        
        use_timeseries_split = st.checkbox(
            "✅ TimeSeriesSplit für Cross-Validation verwenden (empfohlen)",
            value=True,
            help="Verwendet TimeSeriesSplit statt einfachem Train-Test-Split für realistischere Metriken",
            key="use_timeseries_split_checkbox"
        )
        
        cv_splits = 5
        if use_timeseries_split:
            st.success("✅ TimeSeriesSplit ist aktiviert. Respektiert die zeitliche Reihenfolge der Daten.")
            
            st.markdown("**Anzahl Splits:**")
            st.caption("""
            Mehr Splits = mehr Validierung, aber langsameres Training:
            - **3 Splits:** Schnell, weniger Validierung
            - **5 Splits:** Ausgewogen (empfohlen)
            - **10 Splits:** Sehr gründlich, aber langsam
            """)
            
            cv_splits = st.number_input(
                "Anzahl Splits für Cross-Validation",
                min_value=3,
                max_value=10,
                value=5,
                step=1,
                help="Mehr Splits = mehr Validierung, aber langsameres Training",
                key="cv_splits_input"
            )
    
    # Hyperparameter-Checkbox auch außerhalb des Forms
    with st.expander("⚙️ Hyperparameter (optional)", expanded=False):
        st.markdown("""
        **Was sind Hyperparameter?**
        
        Hyperparameter steuern, wie das Modell lernt:
        - **n_estimators:** Anzahl der Bäume (mehr = genauer, aber langsamer)
        - **max_depth:** Maximale Tiefe der Bäume (tiefer = komplexer, aber Overfitting-Risiko)
        
        **Standard-Werte:** Funktionieren für die meisten Fälle gut.
        **Anpassung:** Nur nötig, wenn du die Performance optimieren willst.
        """)
        
        use_custom_params = st.checkbox(
            "✅ Hyperparameter anpassen", 
            value=True,  # Standardmäßig aktiviert
            help="Aktiviert die Anpassung der Hyperparameter. Standard-Werte funktionieren meist gut.",
            key="use_custom_params_checkbox"
        )
        
        if use_custom_params:
            st.info("ℹ️ Hyperparameter-Anpassung ist aktiviert. Die Felder werden im Formular angezeigt.")
    
    with st.form("train_model_form"):
        # Basis-Informationen
        st.subheader("📝 Basis-Informationen")
        model_name = st.text_input("Modell-Name *", placeholder="z.B. PumpDetector_v1")
        model_type = st.selectbox(
            "Modell-Typ *",
            ["random_forest", "xgboost"],
            help="Random Forest: Robust, schnell. XGBoost: Beste Performance"
        )
        description = st.text_area("Beschreibung (optional)", placeholder="Kurze Beschreibung des Modells")
        
        st.divider()
        
        # Features
        st.subheader("📊 Features")
        st.markdown("""
        **Was sind Features?**
        
        Features sind die Eingabedaten für das Modell. Das Modell lernt aus diesen Daten, Muster zu erkennen.
        
        **Empfohlene Features:**
        - **price_open, price_high, price_low, price_close:** Preis-Informationen
        - **volume_sol, volume_usd:** Handelsvolumen
        - **buy_volume_sol, sell_volume_sol:** Käufer- vs. Verkäufer-Volumen
        """)
        
        features = st.multiselect(
            "Features auswählen *",
            AVAILABLE_FEATURES,
            default=AVAILABLE_FEATURES,  # Alle Features standardmäßig aktiviert
            help="Welche Spalten aus coin_metrics sollen verwendet werden? Mehr Features = mehr Information, aber langsameres Training.",
            label_visibility="visible"
        )
        
        if not features:
            st.warning("⚠️ Bitte wähle mindestens ein Feature aus!")
        
        st.divider()
        
        # Phasen
        st.subheader("🪙 Coin-Phasen")
        st.markdown("""
        **Was sind Coin-Phasen?**
        
        Coins durchlaufen verschiedene Phasen (z.B. Baby Zone, Survival Zone, Mature Zone).
        Jede Phase hat unterschiedliche Eigenschaften und Intervalle.
        
        **Auswirkung:** Wenn Phasen ausgewählt werden, werden nur Daten aus diesen Phasen verwendet.
        Wenn keine Phasen ausgewählt werden, werden alle Phasen verwendet.
        """)
        
        phases_list = load_phases()
        
        if phases_list:
            # Erstelle Anzeige-Strings mit interval_seconds
            phase_options = {}
            for phase in phases_list:
                phase_id = phase.get("id")
                phase_name = phase.get("name", f"Phase {phase_id}")
                interval_sec = phase.get("interval_seconds", 0)
                # Format: "Phase 1 (60s)" oder "Phase 1 - Name (60s)"
                if phase_name and phase_name != f"Phase {phase_id}":
                    display_name = f"Phase {phase_id} - {phase_name} ({interval_sec}s)"
                else:
                    display_name = f"Phase {phase_id} ({interval_sec}s)"
                phase_options[phase_id] = display_name
            
            # Sortiere nach Phase-ID
            sorted_phases = sorted(phase_options.items())
            phase_labels = [label for _, label in sorted_phases]
            phase_ids = [pid for pid, _ in sorted_phases]
            
            selected_labels = st.multiselect(
                "Phasen auswählen (optional)",
                phase_labels,
                help="Welche Coin-Phasen sollen einbezogen werden? (Leer = alle). Interval in Sekunden wird angezeigt.",
                label_visibility="visible"
            )
            
            if selected_labels:
                st.info(f"ℹ️ {len(selected_labels)} Phase(n) ausgewählt. Nur Daten aus diesen Phasen werden verwendet.")
            else:
                st.info("ℹ️ Keine Phasen ausgewählt. Alle Phasen werden verwendet.")
            
            # Konvertiere Labels zurück zu IDs
            phases = [phase_ids[phase_labels.index(label)] for label in selected_labels] if selected_labels else None
        else:
            # Kein Fallback - nur Warnung wenn Phasen nicht geladen werden können
            st.error("❌ Phasen konnten nicht aus ref_coin_phases geladen werden. Bitte API-Verbindung prüfen.")
            phases = None
        
        # Target - NUR anzeigen wenn zeitbasierte Vorhersage NICHT aktiviert ist
        if not use_time_based:
            st.subheader("🎯 Ziel-Variable")
            target_var = st.selectbox("Target-Variable *", AVAILABLE_TARGETS)
            target_operator = st.selectbox("Operator *", [">", "<", ">=", "<=", "="])
            target_value = st.number_input("Target-Wert *", min_value=0.0, value=50000.0, step=1000.0)
            st.info(f"💡 Ziel: {target_var} {target_operator} {target_value}")
        else:
            # Bei zeitbasierter Vorhersage: target_var wird später aus den zeitbasierten Feldern gesetzt
            pass
            # Wir brauchen es für create_time_based_labels, aber zeigen es nicht an
            target_var = None  # Wird später gesetzt
            target_operator = None
            target_value = None
        
        # Training-Zeitraum
        st.subheader("📅 Training-Zeitraum")
        
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
            st.warning("⚠️ Keine Trainingsdaten in der Datenbank gefunden!")
        
        st.divider()
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**🕐 Start-Zeitpunkt**")
            
            # Default-Start-Datum: Min-Datum oder heute - 30 Tage
            if min_date:
                default_start_date = min_date
            else:
                default_start_date = datetime.now(timezone.utc).date() - timedelta(days=30)
            
            train_start_date = st.date_input(
                "Start-Datum *",
                value=default_start_date,
                min_value=min_date,
                max_value=max_date,
                key="train_start_date",
                help=f"Wähle ein Datum zwischen {min_date.strftime('%d.%m.%Y') if min_date else 'N/A'} und {max_date.strftime('%d.%m.%Y') if max_date else 'N/A'}"
            )
            
            # Default-Start-Uhrzeit: Min-Uhrzeit oder 00:00
            if min_datetime and train_start_date == min_date:
                default_start_time = min_datetime.time()
            else:
                default_start_time = datetime.now(timezone.utc).time().replace(hour=0, minute=0, second=0, microsecond=0)
            
            train_start_time = st.time_input(
                "Start-Uhrzeit *",
                value=default_start_time,
                key="train_start_time",
                help="Wähle eine Uhrzeit"
            )
            
            train_start_dt = datetime.combine(train_start_date, train_start_time).replace(tzinfo=timezone.utc)
            
            # Warnung wenn außerhalb des verfügbaren Bereichs
            if min_datetime and train_start_dt < min_datetime:
                st.warning(f"⚠️ Start-Zeitpunkt liegt vor dem ältesten Eintrag ({min_datetime.strftime('%d.%m.%Y %H:%M')})")
            elif max_datetime and train_start_dt > max_datetime:
                st.warning(f"⚠️ Start-Zeitpunkt liegt nach dem neuesten Eintrag ({max_datetime.strftime('%d.%m.%Y %H:%M')})")
        
        with col2:
            st.markdown("**🕐 Ende-Zeitpunkt**")
            
            # Default-End-Datum: Max-Datum oder heute
            if max_date:
                default_end_date = max_date
            else:
                default_end_date = datetime.now(timezone.utc).date()
            
            train_end_date = st.date_input(
                "Ende-Datum *",
                value=default_end_date,
                min_value=train_start_date,  # Ende muss nach Start sein
                max_value=max_date,
                key="train_end_date",
                help=f"Wähle ein Datum nach dem Start-Datum (max. {max_date.strftime('%d.%m.%Y') if max_date else 'N/A'})"
            )
            
            # Default-End-Uhrzeit: Max-Uhrzeit oder 23:59
            if max_datetime and train_end_date == max_date:
                default_end_time = max_datetime.time()
            else:
                default_end_time = datetime.now(timezone.utc).time().replace(hour=23, minute=59, second=59, microsecond=0)
            
            train_end_time = st.time_input(
                "Ende-Uhrzeit *",
                value=default_end_time,
                key="train_end_time",
                help="Wähle eine Uhrzeit"
            )
            
            train_end_dt = datetime.combine(train_end_date, train_end_time).replace(tzinfo=timezone.utc)
            
            # Warnung wenn außerhalb des verfügbaren Bereichs
            if min_datetime and train_end_dt < min_datetime:
                st.warning(f"⚠️ Ende-Zeitpunkt liegt vor dem ältesten Eintrag ({min_datetime.strftime('%d.%m.%Y %H:%M')})")
            elif max_datetime and train_end_dt > max_datetime:
                st.warning(f"⚠️ Ende-Zeitpunkt liegt nach dem neuesten Eintrag ({max_datetime.strftime('%d.%m.%Y %H:%M')})")
        # Zeitbasierte Vorhersage-Felder (nur wenn aktiviert)
        future_minutes = None
        min_percent_change = None
        direction = "up"
        time_based_target_var = None
        
        if use_time_based:
            st.divider()
            st.subheader("⏰ Zeitbasierte Vorhersage - Konfiguration")
            st.markdown("""
            **Zeitbasierte Vorhersage konfigurieren:**
            
            Definiere, welche Variable überwacht werden soll und welche Bedingungen erfüllt sein müssen.
            Das Modell lernt: "Steigt/Fällt die Variable in X Minuten um X%?"
            """)
            
            # Target-Variable für zeitbasierte Vorhersage
            st.markdown("**Variable überwachen** *")
            st.caption("Welche Variable soll für die prozentuale Änderung verwendet werden? (z.B. price_close)")
            time_based_target_var = st.selectbox(
                "Welche Variable wird überwacht? *", 
                AVAILABLE_TARGETS, 
                help="Diese Variable wird für die prozentuale Änderung verwendet (z.B. 'price_close')",
                key="time_based_target_var",
                label_visibility="collapsed"
            )
            
            st.markdown("**Vorhersage-Parameter:**")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown("**Zeitraum (Minuten)** *")
                st.caption("In wie vielen Minuten soll die Änderung stattfinden? (z.B. 5 = nächste 5 Minuten)")
                future_minutes = st.number_input(
                    "Zeitraum (Minuten) *", 
                    min_value=1, 
                    max_value=60,
                    value=5, 
                    step=1,
                    help="In wie vielen Minuten soll die Änderung stattfinden? Beispiel: 5 = Modell lernt 'Steigt price_close in den nächsten 5 Minuten um X%?'",
                    key="future_minutes_input",
                    label_visibility="collapsed"
                )
                st.caption(f"ℹ️ Vorhersage für die nächsten {future_minutes} Minute(n)")
            with col2:
                st.markdown("**Min. Prozentuale Änderung** *")
                st.caption("Mindest-Prozentuale Änderung (z.B. 30 = 30% Steigerung)")
                min_percent_change = st.number_input(
                    "Min. Prozentuale Änderung *", 
                    min_value=0.1, 
                    max_value=1000.0,
                    value=30.0, 
                    step=1.0,
                    help="Mindest-Prozentuale Änderung. Beispiel: 30 = Modell lernt 'Steigt price_close um mindestens 30%?'",
                    key="min_percent_change_input",
                    label_visibility="collapsed"
                )
                st.caption(f"ℹ️ Mindestens {min_percent_change}% Änderung erforderlich")
            with col3:
                st.markdown("**Richtung** *")
                st.caption("Steigt (up) oder fällt (down) die Variable?")
                direction = st.selectbox(
                    "Richtung *",
                    ["up", "down"],
                    format_func=lambda x: "Steigt" if x == "up" else "Fällt",
                    help="Steigt (up) oder fällt (down) die Variable? Beispiel: 'up' = Modell lernt 'Steigt price_close?'",
                    key="direction_select",
                    label_visibility="collapsed"
                )
                direction_emoji = "📈" if direction == "up" else "📉"
                st.caption(f"ℹ️ Richtung: {direction_emoji} {direction.upper()}")
            
            st.success(f"💡 **Ziel:** {time_based_target_var} soll in {future_minutes} Minuten um {min_percent_change}% {'steigen' if direction == 'up' else 'fallen'}")
        
        # Hyperparameter (nur wenn Checkbox aktiviert)
        params = None
        if use_custom_params:
            st.divider()
            st.subheader("⚙️ Hyperparameter anpassen")
            st.markdown("""
            **Hyperparameter anpassen:**
            
            Diese Parameter steuern, wie das Modell lernt. Standard-Werte funktionieren für die meisten Fälle gut.
            """)
            
            if model_type == "random_forest":
                st.markdown("**Random Forest Parameter:**")
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**n_estimators**")
                    st.caption("Anzahl der Bäume im Modell. Mehr Bäume = genauer, aber langsameres Training.")
                    n_estimators = st.number_input(
                        "n_estimators", 
                        min_value=10,
                        max_value=1000,
                        value=100, 
                        step=10,
                        help="Anzahl der Entscheidungsbäume. Mehr = genauer, aber langsamer. Standard: 100",
                        label_visibility="collapsed"
                    )
                    st.caption(f"ℹ️ {n_estimators} Bäume werden verwendet")
                with col2:
                    st.markdown("**max_depth**")
                    st.caption("Maximale Tiefe der Bäume. Tiefer = komplexer, aber Overfitting-Risiko.")
                    max_depth = st.number_input(
                        "max_depth", 
                        min_value=1,
                        max_value=50,
                        value=10, 
                        step=1,
                        help="Maximale Tiefe der Bäume. Mehr = komplexer, aber Overfitting-Risiko. Standard: 10",
                        label_visibility="collapsed"
                    )
                    st.caption(f"ℹ️ Maximale Tiefe: {max_depth}")
                params = {"n_estimators": int(n_estimators), "max_depth": int(max_depth)}
            else:  # xgboost
                st.markdown("**XGBoost Parameter:**")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown("**n_estimators**")
                    st.caption("Anzahl der Boosting-Runden. Mehr = genauer, aber langsamer.")
                    n_estimators = st.number_input(
                        "n_estimators", 
                        min_value=10,
                        max_value=1000,
                        value=100, 
                        step=10,
                        help="Anzahl der Boosting-Runden. Mehr = genauer, aber langsamer. Standard: 100",
                        key="xgboost_n_estimators",
                        label_visibility="collapsed"
                    )
                    st.caption(f"ℹ️ {n_estimators} Runden")
                with col2:
                    st.markdown("**max_depth**")
                    st.caption("Maximale Tiefe der Bäume. Mehr = komplexer.")
                    max_depth = st.number_input(
                        "max_depth", 
                        min_value=1,
                        max_value=20,
                        value=6, 
                        step=1,
                        help="Maximale Tiefe der Bäume. Mehr = komplexer. Standard: 6",
                        key="xgboost_max_depth",
                        label_visibility="collapsed"
                    )
                    st.caption(f"ℹ️ Tiefe: {max_depth}")
                with col3:
                    st.markdown("**learning_rate**")
                    st.caption("Lernrate. Kleiner = langsamer, aber oft genauer.")
                    learning_rate = st.number_input(
                        "learning_rate", 
                        min_value=0.01, 
                        max_value=1.0, 
                        value=0.1, 
                        step=0.01,
                        help="Lernrate. Kleiner = langsamer, aber oft genauer. Standard: 0.1",
                        key="xgboost_learning_rate",
                        label_visibility="collapsed"
                    )
                    st.caption(f"ℹ️ Rate: {learning_rate}")
                params = {
                    "n_estimators": int(n_estimators),
                    "max_depth": int(max_depth),
                    "learning_rate": float(learning_rate)
                }
        
        st.divider()
        
        # Zusammenfassung vor Submit
        st.markdown("### 📋 Zusammenfassung")
        st.caption("Überprüfe deine Einstellungen vor dem Training:")
        
        summary_col1, summary_col2 = st.columns(2)
        with summary_col1:
            st.markdown(f"**Modell:** {model_name if model_name else '❌ (Name fehlt)'}")
            st.markdown(f"**Typ:** {model_type}")
            st.markdown(f"**Features:** {len(features)} ausgewählt")
            if use_engineered_features:
                st.markdown("**Feature-Engineering:** ✅ Aktiviert")
            if use_smote:
                st.markdown("**SMOTE:** ✅ Aktiviert")
            if use_timeseries_split:
                st.markdown(f"**Cross-Validation:** ✅ TimeSeriesSplit ({cv_splits} Splits)")
        with summary_col2:
            if use_time_based:
                if 'time_based_target_var' in locals() and time_based_target_var:
                    st.markdown(f"**Vorhersage:** {time_based_target_var} in {future_minutes if 'future_minutes' in locals() and future_minutes else 'N/A'}min um {min_percent_change if 'min_percent_change' in locals() and min_percent_change else 'N/A'}% {'↑' if direction == 'up' else '↓'}")
                else:
                    st.markdown("**Vorhersage:** ⏰ Zeitbasiert (konfigurieren)")
            else:
                if 'target_var' in locals() and target_var:
                    st.markdown(f"**Ziel:** {target_var} {target_operator if 'target_operator' in locals() and target_operator else ''} {target_value if 'target_value' in locals() and target_value is not None else ''}")
                else:
                    st.markdown("**Ziel:** ❌ (nicht konfiguriert)")
            if 'train_start_dt' in locals() and 'train_end_dt' in locals():
                if train_start_dt < train_end_dt:
                    duration_days = (train_end_dt - train_start_dt).total_seconds() / 86400.0
                    st.markdown(f"**Zeitraum:** {duration_days:.1f} Tage")
                else:
                    st.markdown("**Zeitraum:** ❌ (Start muss vor Ende liegen)")
            else:
                st.markdown("**Zeitraum:** ❌ (nicht konfiguriert)")
        
        # Submit
        submitted = st.form_submit_button("🚀 Modell trainieren", type="primary", use_container_width=True)
        
        if submitted:
            # Validierung
            if not model_name:
                st.error("❌ Modell-Name ist erforderlich!")
                return
            if not features:
                st.error("❌ Mindestens ein Feature muss ausgewählt werden!")
                return
            # Validierung: Zeitbasierte Vorhersage
            if use_time_based:
                if not time_based_target_var:
                    st.error("❌ Variable ist erforderlich (welche Variable wird überwacht?)!")
                    return
                if not future_minutes or future_minutes <= 0:
                    st.error("❌ Zukunft (Minuten) muss größer als 0 sein!")
                    return
                if not min_percent_change or min_percent_change <= 0:
                    st.error("❌ Min. Prozent-Änderung muss größer als 0 sein!")
                    return
                # Setze target_var für API
                target_var = time_based_target_var
            
            # Validierung: Normale Vorhersage
            if not use_time_based:
                if not target_var:
                    st.error("❌ Target-Variable ist erforderlich!")
                    return
                if not target_operator:
                    st.error("❌ Operator ist erforderlich!")
                    return
                if target_value is None:
                    st.error("❌ Target-Wert ist erforderlich!")
                    return
            
            if train_start_dt >= train_end_dt:
                st.error("❌ Start-Zeitpunkt muss vor End-Zeitpunkt liegen!")
                return
            
            # API-Call
            with st.spinner("🔄 Erstelle Training-Job..."):
                # Konvertiere datetime zu UTC ISO-Format
                train_start_iso = train_start_dt.astimezone(timezone.utc).isoformat().replace('+00:00', 'Z')
                train_end_iso = train_end_dt.astimezone(timezone.utc).isoformat().replace('+00:00', 'Z')
                
                data = {
                    "name": model_name,
                    "model_type": model_type,
                    "target_var": target_var,  # Wird bei zeitbasierter Vorhersage auf time_based_target_var gesetzt
                    "operator": target_operator if not use_time_based else None,
                    "target_value": float(target_value) if not use_time_based and target_value is not None else None,
                    "features": features,
                    "phases": phases if phases else None,
                    "params": params,
                    "train_start": train_start_iso,
                    "train_end": train_end_iso,
                    # NEU: Zeitbasierte Parameter
                    "use_time_based_prediction": use_time_based,
                    "future_minutes": int(future_minutes) if use_time_based and future_minutes else None,
                    "min_percent_change": float(min_percent_change) if use_time_based and min_percent_change else None,
                    "direction": direction if use_time_based else "up",
                    # NEU: Feature-Engineering Parameter
                    "use_engineered_features": use_engineered_features,
                    "feature_engineering_windows": feature_engineering_windows if use_engineered_features and feature_engineering_windows else None,
                    # NEU: SMOTE Parameter
                    "use_smote": use_smote,
                    # NEU: TimeSeriesSplit Parameter
                    "use_timeseries_split": use_timeseries_split,
                    "cv_splits": int(cv_splits) if use_timeseries_split else None
                }
                
                result = api_post("/api/models/create", data)
                
                if result:
                    st.success(f"✅ Job erstellt! Job-ID: {result.get('job_id')}")
                    st.info(f"📊 Status: {result.get('status')}. Das Modell wird jetzt trainiert.")
                    st.balloons()
                    
                    # Weiterleitung zu Jobs-Seite
                    if st.button("📊 Zu Jobs anzeigen"):
                        st.session_state['page'] = 'jobs'
                        st.rerun()

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

def page_jobs():
    """Jobs-Übersicht"""
    st.title("📊 Jobs")
    
    # Lade Jobs
    jobs = api_get("/api/queue")
    if not jobs:
        st.info("ℹ️ Keine Jobs gefunden")
        return
    
    # Status-Filter
    status_filter = st.selectbox("Status Filter", ["Alle", "PENDING", "RUNNING", "COMPLETED", "FAILED"])
    
    # Filter anwenden
    filtered_jobs = jobs
    if status_filter != "Alle":
        filtered_jobs = [j for j in jobs if j.get('status') == status_filter]
    
    st.info(f"📊 {len(filtered_jobs)} Job(s) gefunden")
    
    # Tabelle
    if filtered_jobs:
        df = pd.DataFrame([
            {
                "ID": j.get('id'),
                "Typ": j.get('job_type'),
                "Status": j.get('status'),
                "Progress": f"{j.get('progress', 0) * 100:.1f}%" if j.get('progress') else "N/A",
                "Erstellt": j.get('created_at', '')[:19] if j.get('created_at') else "N/A",
                "Nachricht": j.get('progress_msg', '')[:50] if j.get('progress_msg') else "N/A"
            }
            for j in filtered_jobs
        ])
        st.dataframe(df, use_container_width=True, hide_index=True)
        
        # Details
        selected_job_id = st.selectbox("Job auswählen für Details", options=[j.get('id') for j in filtered_jobs])
        if selected_job_id:
            selected_job = next((j for j in filtered_jobs if j.get('id') == selected_job_id), None)
            if selected_job:
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Status", selected_job.get('status', 'N/A'))
                with col2:
                    st.metric("Progress", f"{selected_job.get('progress', 0) * 100:.1f}%")
                with col3:
                    st.metric("Typ", selected_job.get('job_type', 'N/A'))
                
                # Test-Ergebnisse anzeigen
                if selected_job.get('result_test'):
                    st.subheader("🧪 Test-Ergebnisse")
                    test = selected_job['result_test']
                    
                    # Basis-Metriken
                    st.markdown("**📊 Basis-Metriken**")
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Accuracy", f"{test.get('accuracy', 0):.4f}")
                    with col2:
                        st.metric("F1-Score", f"{test.get('f1_score', 0):.4f}")
                    with col3:
                        st.metric("Precision", f"{test.get('precision_score', 0):.4f}")
                    with col4:
                        st.metric("Recall", f"{test.get('recall', 0):.4f}")
                    
                    # Zusätzliche Metriken (Phase 9)
                    if test.get('roc_auc') is not None or test.get('mcc') is not None:
                        st.markdown("**📈 Zusätzliche Metriken**")
                        col1, col2, col3, col4, col5 = st.columns(5)
                        with col1:
                            if test.get('roc_auc') is not None:
                                st.metric("ROC-AUC", f"{test.get('roc_auc', 0):.4f}")
                        with col2:
                            if test.get('mcc') is not None:
                                st.metric("MCC", f"{test.get('mcc', 0):.4f}")
                        with col3:
                            if test.get('fpr') is not None:
                                st.metric("FPR", f"{test.get('fpr', 0):.4f}")
                        with col4:
                            if test.get('fnr') is not None:
                                st.metric("FNR", f"{test.get('fnr', 0):.4f}")
                        with col5:
                            if test.get('simulated_profit_pct') is not None:
                                st.metric("💰 Profit", f"{test.get('simulated_profit_pct', 0):.2f}%")
                    
                    # Confusion Matrix
                    confusion_matrix = test.get('confusion_matrix')
                    if confusion_matrix:
                        st.markdown("**🔢 Confusion Matrix**")
                        cm_col1, cm_col2, cm_col3, cm_col4 = st.columns(4)
                        with cm_col1:
                            st.metric("True Positive (TP)", confusion_matrix.get('tp', 0))
                        with cm_col2:
                            st.metric("True Negative (TN)", confusion_matrix.get('tn', 0))
                        with cm_col3:
                            st.metric("False Positive (FP)", confusion_matrix.get('fp', 0))
                        with cm_col4:
                            st.metric("False Negative (FN)", confusion_matrix.get('fn', 0))
                        
                        # Visualisierung als Tabelle
                        cm_data = {
                            'Tatsächlich': ['Negativ', 'Positiv'],
                            'Vorhergesagt: Negativ': [confusion_matrix.get('tn', 0), confusion_matrix.get('fn', 0)],
                            'Vorhergesagt: Positiv': [confusion_matrix.get('fp', 0), confusion_matrix.get('tp', 0)]
                        }
                        cm_df = pd.DataFrame(cm_data)
                        st.dataframe(cm_df, use_container_width=True, hide_index=True)
                    elif test.get('tp') is not None:
                        # Fallback: Legacy-Format (einzelne Felder)
                        st.markdown("**🔢 Confusion Matrix**")
                        cm_col1, cm_col2, cm_col3, cm_col4 = st.columns(4)
                        with cm_col1:
                            st.metric("True Positive (TP)", test.get('tp', 0))
                        with cm_col2:
                            st.metric("True Negative (TN)", test.get('tn', 0))
                        with cm_col3:
                            st.metric("False Positive (FP)", test.get('fp', 0))
                        with cm_col4:
                            st.metric("False Negative (FN)", test.get('fn', 0))
                    
                    # Samples
                    st.markdown("**📊 Daten-Info**")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Samples", test.get('num_samples', 0))
                    with col2:
                        st.metric("Positive", test.get('num_positive', 0))
                    with col3:
                        st.metric("Negative", test.get('num_negative', 0))
                    
                    # Train vs. Test Vergleich (Phase 2)
                    if test.get('train_accuracy') is not None:
                        st.markdown("**📊 Train vs. Test Vergleich**")
                        train_acc = test.get('train_accuracy')
                        test_acc = test.get('accuracy', 0)
                        acc_degradation = test.get('accuracy_degradation')
                        is_overfitted = test.get('is_overfitted', False)
                        
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Train Accuracy", f"{train_acc:.4f}")
                        with col2:
                            st.metric("Test Accuracy", f"{test_acc:.4f}")
                        with col3:
                            if acc_degradation is not None:
                                degradation_color = "🔴" if is_overfitted else "🟢"
                                st.metric(f"{degradation_color} Degradation", f"{acc_degradation:.2%}")
                        with col4:
                            if is_overfitted:
                                st.error("⚠️ OVERFITTING!")
                            else:
                                st.success("✅ OK")
                        
                        if is_overfitted:
                            st.warning("⚠️ **Overfitting erkannt!** Das Modell generalisiert schlecht auf neue Daten. Train-Test Gap > 10%.")
                        
                        # F1 Vergleich
                        if test.get('train_f1') is not None:
                            train_f1 = test.get('train_f1')
                            test_f1 = test.get('f1_score', 0)
                            f1_degradation = test.get('f1_degradation')
                            if f1_degradation is not None:
                                st.caption(f"F1: Train={train_f1:.4f}, Test={test_f1:.4f}, Gap={f1_degradation:.2%}")
                    
                    # Test-Zeitraum Info (Phase 2)
                    test_duration_days = test.get('test_duration_days')
                    if test_duration_days is not None:
                        if test_duration_days < 1.0:
                            st.warning(f"⚠️ Test-Zeitraum zu kurz: {test_duration_days:.2f} Tage (empfohlen: mindestens 1 Tag)")
                        else:
                            st.caption(f"📅 Test-Zeitraum: {test_duration_days:.2f} Tage")
                    
                    if test.get('has_overlap'):
                        st.warning(f"⚠️ {test.get('overlap_note', 'Überschneidung mit Trainingsdaten')}")
                
                # Vergleichs-Ergebnisse anzeigen
                if selected_job.get('result_comparison'):
                    st.subheader("⚖️ Vergleichs-Ergebnisse")
                    comp = selected_job['result_comparison']
                    
                    # Gewinner
                    winner_id = comp.get('winner_id')
                    if winner_id:
                        st.success(f"🏆 Gewinner: Modell {winner_id}")
                    else:
                        st.info("🤝 Unentschieden")
                    
                    # Basis-Metriken: Modell A vs. Modell B
                    st.markdown("### 📊 Basis-Metriken")
                    col_a1, col_a2, col_a3, col_a4, col_b1, col_b2, col_b3, col_b4 = st.columns(8)
                    with col_a1:
                        st.markdown("**Modell A**")
                        st.metric("Accuracy", f"{comp.get('a_accuracy', 0):.4f}" if comp.get('a_accuracy') else "N/A")
                    with col_a2:
                        st.markdown("&nbsp;")
                        st.metric("F1-Score", f"{comp.get('a_f1', 0):.4f}" if comp.get('a_f1') else "N/A")
                    with col_a3:
                        st.markdown("&nbsp;")
                        st.metric("Precision", f"{comp.get('a_precision', 0):.4f}" if comp.get('a_precision') else "N/A")
                    with col_a4:
                        st.markdown("&nbsp;")
                        st.metric("Recall", f"{comp.get('a_recall', 0):.4f}" if comp.get('a_recall') else "N/A")
                    with col_b1:
                        st.markdown("**Modell B**")
                        st.metric("Accuracy", f"{comp.get('b_accuracy', 0):.4f}" if comp.get('b_accuracy') else "N/A")
                    with col_b2:
                        st.markdown("&nbsp;")
                        st.metric("F1-Score", f"{comp.get('b_f1', 0):.4f}" if comp.get('b_f1') else "N/A")
                    with col_b3:
                        st.markdown("&nbsp;")
                        st.metric("Precision", f"{comp.get('b_precision', 0):.4f}" if comp.get('b_precision') else "N/A")
                    with col_b4:
                        st.markdown("&nbsp;")
                        st.metric("Recall", f"{comp.get('b_recall', 0):.4f}" if comp.get('b_recall') else "N/A")
                    
                    # Zusätzliche Metriken
                    if comp.get('a_mcc') or comp.get('b_mcc') or comp.get('a_simulated_profit_pct') or comp.get('b_simulated_profit_pct'):
                        st.markdown("### 📈 Zusätzliche Metriken")
                        col_a1, col_a2, col_a3, col_a4, col_b1, col_b2, col_b3, col_b4 = st.columns(8)
                        with col_a1:
                            st.markdown("**Modell A**")
                            st.metric("MCC", f"{comp.get('a_mcc', 0):.4f}" if comp.get('a_mcc') else "N/A")
                        with col_a2:
                            st.markdown("&nbsp;")
                            st.metric("FPR", f"{comp.get('a_fpr', 0):.4f}" if comp.get('a_fpr') else "N/A")
                        with col_a3:
                            st.markdown("&nbsp;")
                            st.metric("FNR", f"{comp.get('a_fnr', 0):.4f}" if comp.get('a_fnr') else "N/A")
                        with col_a4:
                            st.markdown("&nbsp;")
                            st.metric("Profit", f"{comp.get('a_simulated_profit_pct', 0):.4f}%" if comp.get('a_simulated_profit_pct') else "N/A")
                        with col_b1:
                            st.markdown("**Modell B**")
                            st.metric("MCC", f"{comp.get('b_mcc', 0):.4f}" if comp.get('b_mcc') else "N/A")
                        with col_b2:
                            st.markdown("&nbsp;")
                            st.metric("FPR", f"{comp.get('b_fpr', 0):.4f}" if comp.get('b_fpr') else "N/A")
                        with col_b3:
                            st.markdown("&nbsp;")
                            st.metric("FNR", f"{comp.get('b_fnr', 0):.4f}" if comp.get('b_fnr') else "N/A")
                        with col_b4:
                            st.markdown("&nbsp;")
                            st.metric("Profit", f"{comp.get('b_simulated_profit_pct', 0):.4f}%" if comp.get('b_simulated_profit_pct') else "N/A")
                    
                    # Confusion Matrix
                    if comp.get('a_confusion_matrix') or comp.get('b_confusion_matrix'):
                        st.markdown("### 📊 Confusion Matrix")
                        col_a, col_b = st.columns(2)
                        with col_a:
                            st.markdown("**Modell A**")
                            if comp.get('a_confusion_matrix'):
                                cm_a = comp['a_confusion_matrix']
                                st.write(f"TP: {cm_a.get('tp', 0)} | TN: {cm_a.get('tn', 0)}")
                                st.write(f"FP: {cm_a.get('fp', 0)} | FN: {cm_a.get('fn', 0)}")
                            else:
                                st.write("N/A")
                        with col_b:
                            st.markdown("**Modell B**")
                            if comp.get('b_confusion_matrix'):
                                cm_b = comp['b_confusion_matrix']
                                st.write(f"TP: {cm_b.get('tp', 0)} | TN: {cm_b.get('tn', 0)}")
                                st.write(f"FP: {cm_b.get('fp', 0)} | FN: {cm_b.get('fn', 0)}")
                            else:
                                st.write("N/A")
                    
                    # Train vs. Test Vergleich
                    if comp.get('a_train_accuracy') or comp.get('b_train_accuracy'):
                        st.markdown("### 📊 Train vs. Test Vergleich")
                        col_a1, col_a2, col_a3, col_b1, col_b2, col_b3 = st.columns(6)
                        with col_a1:
                            st.markdown("**Modell A**")
                            if comp.get('a_train_accuracy'):
                                st.metric("Train Acc", f"{comp.get('a_train_accuracy'):.4f}")
                            else:
                                st.write("Train Acc: N/A")
                        with col_a2:
                            st.markdown("&nbsp;")
                            if comp.get('a_accuracy'):
                                st.metric("Test Acc", f"{comp.get('a_accuracy'):.4f}")
                            else:
                                st.write("Test Acc: N/A")
                        with col_a3:
                            st.markdown("&nbsp;")
                            if comp.get('a_accuracy_degradation') is not None:
                                degradation = comp.get('a_accuracy_degradation')
                                st.metric("Degradation", f"{degradation:.4f}", 
                                         delta=f"{'⚠️ Overfitting' if degradation > 0.1 else '✅ OK'}" if degradation else None)
                            else:
                                st.write("Degradation: N/A")
                        with col_b1:
                            st.markdown("**Modell B**")
                            if comp.get('b_train_accuracy'):
                                st.metric("Train Acc", f"{comp.get('b_train_accuracy'):.4f}")
                            else:
                                st.write("Train Acc: N/A")
                        with col_b2:
                            st.markdown("&nbsp;")
                            if comp.get('b_accuracy'):
                                st.metric("Test Acc", f"{comp.get('b_accuracy'):.4f}")
                            else:
                                st.write("Test Acc: N/A")
                        with col_b3:
                            st.markdown("&nbsp;")
                            if comp.get('b_accuracy_degradation') is not None:
                                degradation = comp.get('b_accuracy_degradation')
                                st.metric("Degradation", f"{degradation:.4f}",
                                         delta=f"{'⚠️ Overfitting' if degradation > 0.1 else '✅ OK'}" if degradation else None)
                            else:
                                st.write("Degradation: N/A")
                        
                        # Overfitting Warnung
                        if comp.get('a_is_overfitted') or comp.get('b_is_overfitted'):
                            if comp.get('a_is_overfitted'):
                                st.warning(f"⚠️ Modell A ist möglicherweise overfitted!")
                            if comp.get('b_is_overfitted'):
                                st.warning(f"⚠️ Modell B ist möglicherweise overfitted!")
                    
                    # Test-Zeitraum Info
                    if comp.get('a_test_duration_days') or comp.get('b_test_duration_days'):
                        st.markdown("### 📅 Test-Zeitraum")
                        duration_a = comp.get('a_test_duration_days', 0) if comp.get('a_test_duration_days') else 0
                        duration_b = comp.get('b_test_duration_days', 0) if comp.get('b_test_duration_days') else 0
                        st.write(f"Dauer: {duration_a:.2f} Tage (beide Modelle getestet auf demselben Zeitraum)")
                        if duration_a < 1:
                            st.warning("⚠️ Test-Zeitraum zu kurz (empfohlen: mindestens 1 Tag)")
                
                # Vollständige Details als JSON (erweiterbar)
                with st.expander("📋 Vollständige Job-Details (JSON)", expanded=False):
                    st.json(selected_job)
    else:
        st.info("ℹ️ Keine Jobs gefunden")

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
            
            # Metriken
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
                    if checked != is_selected:
                        if checked:
                            if comp_id not in st.session_state['selected_comparison_ids']:
                                st.session_state['selected_comparison_ids'].append(comp_id)
                                st.rerun()
                        else:
                            if comp_id in st.session_state['selected_comparison_ids']:
                                st.session_state['selected_comparison_ids'].remove(comp_id)
                                st.rerun()
                
                with header_col2:
                    st.markdown(f"**{model_a_name} vs {model_b_name}**")
                
                with header_col3:
                    if st.button("📋", key=f"comp_details_{comp_id}", help="Details anzeigen", use_container_width=True):
                        st.session_state['comparison_details_id'] = comp_id
                        st.session_state['page'] = 'comparison_details'
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
                        st.session_state['comparison_details_id'] = comp_id
                        st.session_state['page'] = 'comparison_details'
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
                    if checked != is_selected:
                        if checked:
                            if test_id not in st.session_state['selected_test_ids']:
                                st.session_state['selected_test_ids'].append(test_id)
                                st.rerun()
                        else:
                            if test_id in st.session_state['selected_test_ids']:
                                st.session_state['selected_test_ids'].remove(test_id)
                                st.rerun()
                
                with header_col2:
                    st.markdown(f"**{model_name}**")
                
                with header_col3:
                    if st.button("📋", key=f"test_details_{test_id}", help="Details anzeigen", use_container_width=True):
                        st.session_state['test_details_id'] = test_id
                        st.session_state['page'] = 'test_details'
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

def page_test_details():
    """Details-Seite für ein Test-Ergebnis"""
    test_id = st.session_state.get('test_details_id')
    if not test_id:
        st.error("❌ Keine Test-ID gefunden")
        return
    
    test = api_get(f"/api/test-results/{test_id}")
    if not test:
        st.error("❌ Test-Ergebnis nicht gefunden")
        return
    
    # Hole Modell-Name und Details
    model_id = test.get('model_id')
    model = api_get(f"/api/models/{model_id}")
    model_name = model.get('name', f"ID: {model_id}") if model else f"ID: {model_id}"
    model_train_start = model.get('train_start') if model else None
    model_train_end = model.get('train_end') if model else None
    
    st.title(f"📋 Test-Details: {model_name}")
    
    # Info-Box am Anfang
    st.info("""
    **📖 Anleitung:** 
    Diese Seite zeigt alle Details und Metriken des Test-Ergebnisses. Nutze die ℹ️-Icons für Erklärungen zu jedem Wert.
    """)
    
    # Basis-Informationen
    st.subheader("📝 Basis-Informationen")
    info_col1, info_col2, info_col3, info_col4 = st.columns(4)
    with info_col1:
        st.markdown("**Modell-ID**")
        st.write(f"#{model_id}")
    with info_col2:
        st.markdown("**Modell-Name**")
        st.write(model_name)
    with info_col3:
        st.markdown("**Test-ID**")
        st.write(f"#{test_id}")
    with info_col4:
        st.markdown("**Erstellt**")
        created = test.get('created_at', '')
        if created:
            try:
                if isinstance(created, str):
                    created_dt = datetime.fromisoformat(created.replace('Z', '+00:00'))
                else:
                    created_dt = created
                st.write(created_dt.strftime("%d.%m.%Y %H:%M"))
            except:
                st.write(str(created)[:19] if len(str(created)) > 19 else str(created))
        else:
            st.write("N/A")
    
    st.divider()
    
    # Trainings-Zeitraum des Modells
    st.subheader("🎓 Trainings-Zeitraum des Modells")
    st.markdown("""
    **Was ist der Trainings-Zeitraum?**
    
    Der Trainings-Zeitraum zeigt, mit welchen historischen Daten das Modell trainiert wurde.
    Diese Daten wurden verwendet, um das Modell zu erstellen.
    """)
    
    if model_train_start and model_train_end:
        try:
            if isinstance(model_train_start, str):
                train_start_dt = datetime.fromisoformat(model_train_start.replace('Z', '+00:00'))
            else:
                train_start_dt = model_train_start
            if isinstance(model_train_end, str):
                train_end_dt = datetime.fromisoformat(model_train_end.replace('Z', '+00:00'))
            else:
                train_end_dt = model_train_end
            
            train_start_str = train_start_dt.strftime("%d.%m.%Y %H:%M")
            train_end_str = train_end_dt.strftime("%d.%m.%Y %H:%M")
            train_days = (train_end_dt - train_start_dt).total_seconds() / 86400.0
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.write(f"**Start:** {train_start_str}")
            with col2:
                st.write(f"**Ende:** {train_end_str}")
            with col3:
                st.write(f"**Dauer:** {train_days:.1f} Tage")
        except Exception as e:
            st.write(f"Start: {model_train_start}")
            st.write(f"Ende: {model_train_end}")
    else:
        st.write("Trainings-Zeitraum nicht verfügbar")
    
    st.divider()
    
    # Standard-Metriken mit Erklärungen
    st.subheader("📊 Standard-Metriken")
    st.markdown("""
    **Was bedeuten diese Metriken?**
    
    - **Accuracy:** Anteil korrekter Vorhersagen auf den Test-Daten (0-1). Beispiel: 0.85 = 85% der Vorhersagen sind richtig.
    - **F1-Score:** Harmonisches Mittel aus Precision und Recall (0-1). Gut für unausgewogene Daten.
    - **Precision:** Von allen "Positiv"-Vorhersagen, wie viele waren wirklich positiv? (0-1, höher = weniger False Positives)
    - **Recall:** Von allen echten Positiven, wie viele hat das Modell gefunden? (0-1, höher = weniger False Negatives)
    """)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        accuracy = test.get('accuracy')
        if accuracy:
            st.metric("Accuracy", f"{accuracy:.4f}", help="Anteil korrekter Vorhersagen auf Test-Daten (0-1, höher = besser)")
        else:
            st.caption("Accuracy: N/A")
    with col2:
        f1 = test.get('f1_score')
        if f1:
            st.metric("F1-Score", f"{f1:.4f}", help="Harmonisches Mittel aus Precision und Recall (0-1, höher = besser)")
        else:
            st.caption("F1-Score: N/A")
    with col3:
        precision = test.get('precision_score')
        if precision:
            st.metric("Precision", f"{precision:.4f}", help="Von allen 'Positiv'-Vorhersagen, wie viele waren wirklich positiv? (0-1, höher = weniger False Positives)")
        else:
            st.caption("Precision: N/A")
    with col4:
        recall = test.get('recall')
        if recall:
            st.metric("Recall", f"{recall:.4f}", help="Von allen echten Positiven, wie viele hat das Modell gefunden? (0-1, höher = weniger False Negatives)")
        else:
            st.caption("Recall: N/A")
    
    st.divider()
    
    # Zusätzliche Metriken mit Erklärungen
    if test.get('roc_auc') or test.get('mcc') or test.get('fpr') or test.get('fnr') or test.get('simulated_profit_pct'):
        st.subheader("📈 Erweiterte Metriken")
        st.markdown("""
        **Was bedeuten diese Metriken?**
        
        - **ROC-AUC:** Area Under ROC Curve (0-1). Misst die Fähigkeit, zwischen Positiv und Negativ zu unterscheiden. >0.7 = gut, >0.9 = sehr gut.
        - **MCC:** Matthews Correlation Coefficient (-1 bis +1). Berücksichtigt alle 4 Confusion-Matrix-Werte. 0 = zufällig, +1 = perfekt, -1 = perfekt falsch.
        - **FPR (False Positive Rate):** Wie viele Negatives wurden fälschlicherweise als Positiv klassifiziert? (0-1, niedriger = besser)
        - **FNR (False Negative Rate):** Wie viele Positives wurden fälschlicherweise als Negativ klassifiziert? (0-1, niedriger = besser)
        """)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            roc_auc = test.get('roc_auc')
            if roc_auc:
                quality = "Sehr gut" if roc_auc > 0.9 else "Gut" if roc_auc > 0.7 else "Mäßig" if roc_auc > 0.5 else "Schlecht"
                st.metric("ROC-AUC", f"{roc_auc:.4f}", help=f"Area Under ROC Curve (0-1). {quality} (>0.7 = gut, >0.9 = sehr gut)")
            else:
                st.caption("ROC-AUC: N/A")
        with col2:
            mcc = test.get('mcc')
            if mcc:
                quality = "Sehr gut" if mcc > 0.5 else "Gut" if mcc > 0.3 else "Mäßig" if mcc > 0 else "Schlecht"
                st.metric("MCC", f"{mcc:.4f}", help=f"Matthews Correlation Coefficient (-1 bis +1). {quality} (0 = zufällig, +1 = perfekt)")
            else:
                st.caption("MCC: N/A")
        with col3:
            fpr = test.get('fpr')
            if fpr is not None:
                quality = "Gut" if fpr < 0.1 else "Mäßig" if fpr < 0.3 else "Schlecht"
                st.metric("False Positive Rate", f"{fpr:.4f}", help=f"Wie viele Negatives wurden fälschlicherweise als Positiv klassifiziert? {quality} (niedriger = besser)")
            else:
                st.caption("FPR: N/A")
        with col4:
            fnr = test.get('fnr')
            if fnr is not None:
                quality = "Gut" if fnr < 0.1 else "Mäßig" if fnr < 0.3 else "Schlecht"
                st.metric("False Negative Rate", f"{fnr:.4f}", help=f"Wie viele Positives wurden fälschlicherweise als Negativ klassifiziert? {quality} (niedriger = besser)")
            else:
                st.caption("FNR: N/A")
        
        # Profit-Simulation
        simulated_profit = test.get('simulated_profit_pct')
        if simulated_profit is not None:
            st.divider()
            st.markdown("**💰 Profit-Simulation:**")
            st.markdown("""
            **Was ist Profit-Simulation?**
            
            Simuliert den Profit, den das Modell erzielt hätte:
            - **True Positive (TP):** +1% Gewinn (korrekt erkannte Pumps)
            - **False Positive (FP):** -0.5% Verlust (fälschlicherweise als Pump erkannt)
            - **True Negative (TN):** 0% (korrekt als "kein Pump" erkannt)
            - **False Negative (FN):** 0% (verpasste Pumps)
            
            **Auswirkung:** Zeigt, wie profitabel das Modell in der Praxis wäre.
            """)
            
            col1, col2 = st.columns(2)
            with col1:
                profit_quality = "Sehr profitabel" if simulated_profit > 5 else "Profitabel" if simulated_profit > 0 else "Verlust"
                st.metric("💰 Simulierter Profit", f"{simulated_profit:.2f}%", help=f"Simulierter Profit basierend auf TP/FP. {profit_quality}")
            with col2:
                st.caption("**Berechnung:** 1% Gewinn pro TP, -0.5% Verlust pro FP")
    
    st.divider()
    
    # Confusion Matrix mit Erklärung
    confusion_matrix = test.get('confusion_matrix')
    if confusion_matrix:
        st.subheader("🔢 Confusion Matrix")
        st.markdown("""
        **Was ist eine Confusion Matrix?**
        
        Zeigt, wie viele Vorhersagen korrekt und falsch waren:
        - **TP (True Positive):** ✅ Korrekt als "Positiv" erkannt (z.B. Pump erkannt, war wirklich Pump)
        - **TN (True Negative):** ✅ Korrekt als "Negativ" erkannt (z.B. kein Pump erkannt, war wirklich kein Pump)
        - **FP (False Positive):** ❌ Fälschlicherweise als "Positiv" erkannt (z.B. Pump erkannt, war aber kein Pump) → Verluste!
        - **FN (False Negative):** ❌ Fälschlicherweise als "Negativ" erkannt (z.B. kein Pump erkannt, war aber Pump) → Verpasste Chancen!
        
        **Auswirkung:** 
        - Viele TP = Modell erkennt Pumps gut
        - Viele FP = Modell ist zu optimistisch (viele Fehlalarme)
        - Viele FN = Modell verpasst viele Pumps
        """)
        
        cm = confusion_matrix
        cm_col1, cm_col2, cm_col3, cm_col4 = st.columns(4)
        with cm_col1:
            tp = cm.get('tp', 0)
            st.metric("True Positive (TP)", tp, help="✅ Korrekt als 'Positiv' erkannt (z.B. Pump erkannt, war wirklich Pump)")
        with cm_col2:
            tn = cm.get('tn', 0)
            st.metric("True Negative (TN)", tn, help="✅ Korrekt als 'Negativ' erkannt (z.B. kein Pump erkannt, war wirklich kein Pump)")
        with cm_col3:
            fp = cm.get('fp', 0)
            st.metric("False Positive (FP)", fp, help="❌ Fälschlicherweise als 'Positiv' erkannt (z.B. Pump erkannt, war aber kein Pump) → Verluste!")
        with cm_col4:
            fn = cm.get('fn', 0)
            st.metric("False Negative (FN)", fn, help="❌ Fälschlicherweise als 'Negativ' erkannt (z.B. kein Pump erkannt, war aber Pump) → Verpasste Chancen!")
        
        # Visualisierung als Tabelle
        st.markdown("**Confusion Matrix Tabelle:**")
        cm_data = {
            'Tatsächlich': ['Negativ', 'Positiv'],
            'Vorhergesagt: Negativ': [tn, fn],
            'Vorhergesagt: Positiv': [fp, tp]
        }
        cm_df = pd.DataFrame(cm_data)
        st.dataframe(cm_df, use_container_width=True, hide_index=True)
        
        # Interpretation
        total = tp + tn + fp + fn
        if total > 0:
            tp_rate = (tp / total) * 100
            fp_rate = (fp / total) * 100
            fn_rate = (fn / total) * 100
            st.caption(f"ℹ️ Verteilung: {tp_rate:.1f}% TP, {tn/total*100:.1f}% TN, {fp_rate:.1f}% FP, {fn_rate:.1f}% FN")
    
    st.divider()
    
    # Trainings-Zeitraum des Modells (wenn noch nicht angezeigt)
    if model_train_start and model_train_end:
        st.subheader("🎓 Trainings-Zeitraum des Modells")
        st.markdown("""
        **Was ist der Trainings-Zeitraum?**
        
        Der Trainings-Zeitraum zeigt, mit welchen historischen Daten das Modell trainiert wurde.
        Diese Daten wurden verwendet, um das Modell zu erstellen.
        """)
        
        try:
            if isinstance(model_train_start, str):
                train_start_dt = datetime.fromisoformat(model_train_start.replace('Z', '+00:00'))
            else:
                train_start_dt = model_train_start
            if isinstance(model_train_end, str):
                train_end_dt = datetime.fromisoformat(model_train_end.replace('Z', '+00:00'))
            else:
                train_end_dt = model_train_end
            
            train_start_str = train_start_dt.strftime("%d.%m.%Y %H:%M")
            train_end_str = train_end_dt.strftime("%d.%m.%Y %H:%M")
            train_days = (train_end_dt - train_start_dt).total_seconds() / 86400.0
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.write(f"**Start:** {train_start_str}")
            with col2:
                st.write(f"**Ende:** {train_end_str}")
            with col3:
                st.write(f"**Dauer:** {train_days:.1f} Tage")
        except Exception as e:
            st.write(f"Start: {model_train_start}")
            st.write(f"Ende: {model_train_end}")
        
        st.divider()
    
    # Train vs. Test Vergleich mit Erklärung
    if test.get('train_accuracy') is not None:
        st.subheader("📊 Train vs. Test Vergleich")
        st.markdown("""
        **Was bedeutet Train vs. Test Vergleich?**
        
        Vergleicht die Performance auf Trainings- und Test-Daten:
        - **Train Accuracy:** Performance auf den Daten, mit denen das Modell trainiert wurde
        - **Test Accuracy:** Performance auf neuen, ungesehenen Daten
        - **Degradation:** Unterschied zwischen Train- und Test-Accuracy
        
        **Auswirkung:** 
        - Große Degradation (>10%) = Modell ist möglicherweise overfitted (lernt zu spezifisch)
        - Kleine Degradation (<10%) = Modell generalisiert gut auf neue Daten
        """)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            train_acc = test.get('train_accuracy')
            if train_acc:
                st.metric("Train Accuracy", f"{train_acc:.4f}", help="Performance auf Trainingsdaten (0-1, höher = besser)")
            else:
                st.caption("Train Accuracy: N/A")
        with col2:
            test_acc = test.get('accuracy')
            if test_acc:
                st.metric("Test Accuracy", f"{test_acc:.4f}", help="Performance auf Test-Daten (0-1, höher = besser)")
            else:
                st.caption("Test Accuracy: N/A")
        with col3:
            degradation = test.get('accuracy_degradation')
            if degradation is not None:
                quality = "✅ OK" if degradation < 0.1 else "⚠️ Overfitting-Risiko"
                st.metric("Degradation", f"{degradation:.4f}", 
                         delta=quality,
                         help=f"Unterschied zwischen Train- und Test-Accuracy. {quality} (niedriger = besser)")
            else:
                st.caption("Degradation: N/A")
        
        if test.get('is_overfitted'):
            st.warning("⚠️ Modell ist möglicherweise overfitted! Die Performance auf Test-Daten ist deutlich schlechter als auf Trainingsdaten.")
    
    st.divider()
    
    # Test-Zeitraum mit Erklärung
    st.subheader("📅 Test-Zeitraum")
    st.markdown("""
    **Was ist der Test-Zeitraum?**
    
    Der Test-Zeitraum definiert, welche Daten zum Testen des Modells verwendet wurden.
    Diese Daten wurden **nicht** zum Training verwendet.
    
    **Empfehlung:** Mindestens 1 Tag Test-Daten für realistische Ergebnisse.
    """)
    
    test_start = test.get('test_start')
    test_end = test.get('test_end')
    test_duration_days = test.get('test_duration_days')
    
    if test_start and test_end:
        try:
            if isinstance(test_start, str):
                start_dt = datetime.fromisoformat(test_start.replace('Z', '+00:00'))
            else:
                start_dt = test_start
            if isinstance(test_end, str):
                end_dt = datetime.fromisoformat(test_end.replace('Z', '+00:00'))
            else:
                end_dt = test_end
            
            start_str = start_dt.strftime("%d.%m.%Y %H:%M")
            end_str = end_dt.strftime("%d.%m.%Y %H:%M")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.write(f"**Start:** {start_str}")
            with col2:
                st.write(f"**Ende:** {end_str}")
            with col3:
                if test_duration_days:
                    st.write(f"**Dauer:** {test_duration_days:.2f} Tage")
                    if test_duration_days < 1:
                        st.warning("⚠️ Test-Zeitraum zu kurz (empfohlen: mindestens 1 Tag)")
                else:
                    days = (end_dt - start_dt).total_seconds() / 86400.0
                    st.write(f"**Dauer:** {days:.2f} Tage")
                    if days < 1:
                        st.warning("⚠️ Test-Zeitraum zu kurz (empfohlen: mindestens 1 Tag)")
        except Exception as e:
            st.write(f"Start: {test_start}")
            st.write(f"Ende: {test_end}")
    else:
        st.write("Test-Zeitraum nicht verfügbar")
    
    # Anzahl Samples
    num_samples = test.get('num_samples')
    num_positive = test.get('num_positive')
    num_negative = test.get('num_negative')
    if num_samples:
        st.markdown("**Test-Daten:**")
        st.write(f"**Anzahl Samples:** {num_samples}")
        if num_positive is not None and num_negative is not None:
            positive_pct = (num_positive / num_samples) * 100 if num_samples > 0 else 0
            st.write(f"**Positiv:** {num_positive} ({positive_pct:.1f}%)")
            st.write(f"**Negativ:** {num_negative} ({100-positive_pct:.1f}%)")
    
    st.divider()
    
    # Overlap Warnung
    if test.get('has_overlap'):
        st.warning(f"⚠️ **Überschneidung mit Trainingsdaten:** {test.get('overlap_note', 'Die Test-Daten überschneiden sich mit den Trainingsdaten. Dies kann zu unrealistisch guten Ergebnissen führen!')}")
    
    st.divider()
    
    # Vollständige Details
    with st.expander("📋 Vollständige Details (JSON)", expanded=False):
        st.json(test)
    
    # Zurück-Button
    col1, col2 = st.columns([1, 3])
    with col1:
        if st.button("← Zurück zur Test-Übersicht", use_container_width=True):
            st.session_state['page'] = 'test_results'
            st.session_state.pop('test_details_id', None)
            st.rerun()

def page_comparison_details():
    """Details-Seite für einen Vergleich"""
    comparison_id = st.session_state.get('comparison_details_id')
    if not comparison_id:
        st.error("❌ Keine Vergleichs-ID gefunden")
        return
    
    comparison = api_get(f"/api/comparisons/{comparison_id}")
    if not comparison:
        st.error("❌ Vergleich nicht gefunden")
        return
    
    # Hole Modell-Namen und Details
    model_a_id = comparison.get('model_a_id')
    model_b_id = comparison.get('model_b_id')
    model_a = api_get(f"/api/models/{model_a_id}")
    model_b = api_get(f"/api/models/{model_b_id}")
    model_a_name = model_a.get('name', f"ID: {model_a_id}") if model_a else f"ID: {model_a_id}"
    model_b_name = model_b.get('name', f"ID: {model_b_id}") if model_b else f"ID: {model_b_id}"
    model_a_train_start = model_a.get('train_start') if model_a else None
    model_a_train_end = model_a.get('train_end') if model_a else None
    model_b_train_start = model_b.get('train_start') if model_b else None
    model_b_train_end = model_b.get('train_end') if model_b else None
    
    st.title(f"📋 Vergleichs-Details: {model_a_name} vs {model_b_name}")
    
    # Info-Box am Anfang
    st.info("""
    **📖 Anleitung:** 
    Diese Seite zeigt alle Details und Metriken des Modell-Vergleichs. Nutze die ℹ️-Icons für Erklärungen zu jedem Wert.
    """)
    
    # Basis-Informationen
    st.subheader("📝 Basis-Informationen")
    info_col1, info_col2, info_col3, info_col4 = st.columns(4)
    with info_col1:
        st.markdown("**Vergleichs-ID**")
        st.write(f"#{comparison_id}")
    with info_col2:
        st.markdown("**Modell A**")
        st.write(f"{model_a_name} (#{model_a_id})")
    with info_col3:
        st.markdown("**Modell B**")
        st.write(f"{model_b_name} (#{model_b_id})")
    with info_col4:
        st.markdown("**Erstellt**")
        created = comparison.get('created_at', '')
        if created:
            try:
                if isinstance(created, str):
                    created_dt = datetime.fromisoformat(created.replace('Z', '+00:00'))
                else:
                    created_dt = created
                st.write(created_dt.strftime("%d.%m.%Y %H:%M"))
            except:
                st.write(str(created)[:19] if len(str(created)) > 19 else str(created))
        else:
            st.write("N/A")
    
    st.divider()
    
    # Gewinner
    winner_id = comparison.get('winner_id')
    if winner_id:
        if winner_id == model_a_id:
            st.success(f"🏆 **Gewinner: {model_a_name}** - Dieses Modell hat die bessere Performance auf den Test-Daten erzielt.")
        elif winner_id == model_b_id:
            st.success(f"🏆 **Gewinner: {model_b_name}** - Dieses Modell hat die bessere Performance auf den Test-Daten erzielt.")
    else:
        st.info("🤝 **Unentschieden** - Beide Modelle haben eine ähnliche Performance erzielt.")
    
    st.divider()
    
    # Trainings-Zeiträume beider Modelle
    st.subheader("🎓 Trainings-Zeiträume der Modelle")
    st.markdown("""
    **Was sind die Trainings-Zeiträume?**
    
    Die Trainings-Zeiträume zeigen, mit welchen historischen Daten die Modelle trainiert wurden.
    Diese Daten wurden verwendet, um die Modelle zu erstellen.
    """)
    
    train_col1, train_col2 = st.columns(2)
    
    with train_col1:
        st.markdown(f"**Modell A ({model_a_name}):**")
        if model_a_train_start and model_a_train_end:
            try:
                if isinstance(model_a_train_start, str):
                    train_start_dt = datetime.fromisoformat(model_a_train_start.replace('Z', '+00:00'))
                else:
                    train_start_dt = model_a_train_start
                if isinstance(model_a_train_end, str):
                    train_end_dt = datetime.fromisoformat(model_a_train_end.replace('Z', '+00:00'))
                else:
                    train_end_dt = model_a_train_end
                
                train_start_str = train_start_dt.strftime("%d.%m.%Y %H:%M")
                train_end_str = train_end_dt.strftime("%d.%m.%Y %H:%M")
                train_days = (train_end_dt - train_start_dt).total_seconds() / 86400.0
                
                st.write(f"**Start:** {train_start_str}")
                st.write(f"**Ende:** {train_end_str}")
                st.write(f"**Dauer:** {train_days:.1f} Tage")
            except:
                st.write(f"Start: {model_a_train_start}")
                st.write(f"Ende: {model_a_train_end}")
        else:
            st.write("Trainings-Zeitraum nicht verfügbar")
    
    with train_col2:
        st.markdown(f"**Modell B ({model_b_name}):**")
        if model_b_train_start and model_b_train_end:
            try:
                if isinstance(model_b_train_start, str):
                    train_start_dt = datetime.fromisoformat(model_b_train_start.replace('Z', '+00:00'))
                else:
                    train_start_dt = model_b_train_start
                if isinstance(model_b_train_end, str):
                    train_end_dt = datetime.fromisoformat(model_b_train_end.replace('Z', '+00:00'))
                else:
                    train_end_dt = model_b_train_end
                
                train_start_str = train_start_dt.strftime("%d.%m.%Y %H:%M")
                train_end_str = train_end_dt.strftime("%d.%m.%Y %H:%M")
                train_days = (train_end_dt - train_start_dt).total_seconds() / 86400.0
                
                st.write(f"**Start:** {train_start_str}")
                st.write(f"**Ende:** {train_end_str}")
                st.write(f"**Dauer:** {train_days:.1f} Tage")
            except:
                st.write(f"Start: {model_b_train_start}")
                st.write(f"Ende: {model_b_train_end}")
        else:
            st.write("Trainings-Zeitraum nicht verfügbar")
    
    st.divider()
    
    # Standard-Metriken mit Erklärungen
    st.subheader("📊 Standard-Metriken")
    st.markdown("""
    **Was bedeuten diese Metriken?**
    
    - **Accuracy:** Anteil korrekter Vorhersagen auf den Test-Daten (0-1). Beispiel: 0.85 = 85% der Vorhersagen sind richtig.
    - **F1-Score:** Harmonisches Mittel aus Precision und Recall (0-1). Gut für unausgewogene Daten.
    - **Precision:** Von allen "Positiv"-Vorhersagen, wie viele waren wirklich positiv? (0-1, höher = weniger False Positives)
    - **Recall:** Von allen echten Positiven, wie viele hat das Modell gefunden? (0-1, höher = weniger False Negatives)
    """)
    
    # Modell A Metriken
    st.markdown("**Modell A:**")
    col_a1, col_a2, col_a3, col_a4 = st.columns(4)
    with col_a1:
        a_acc = comparison.get('a_accuracy')
        if a_acc:
            st.metric("Accuracy", f"{a_acc:.4f}", help="Anteil korrekter Vorhersagen auf Test-Daten (0-1, höher = besser)")
        else:
            st.caption("Accuracy: N/A")
    with col_a2:
        a_f1 = comparison.get('a_f1')
        if a_f1:
            st.metric("F1-Score", f"{a_f1:.4f}", help="Harmonisches Mittel aus Precision und Recall (0-1, höher = besser)")
        else:
            st.caption("F1-Score: N/A")
    with col_a3:
        a_prec = comparison.get('a_precision')
        if a_prec:
            st.metric("Precision", f"{a_prec:.4f}", help="Von allen 'Positiv'-Vorhersagen, wie viele waren wirklich positiv? (0-1, höher = weniger False Positives)")
        else:
            st.caption("Precision: N/A")
    with col_a4:
        a_rec = comparison.get('a_recall')
        if a_rec:
            st.metric("Recall", f"{a_rec:.4f}", help="Von allen echten Positiven, wie viele hat das Modell gefunden? (0-1, höher = weniger False Negatives)")
        else:
            st.caption("Recall: N/A")
    
    st.divider()
    
    # Modell B Metriken
    st.markdown("**Modell B:**")
    col_b1, col_b2, col_b3, col_b4 = st.columns(4)
    with col_b1:
        b_acc = comparison.get('b_accuracy')
        if b_acc:
            st.metric("Accuracy", f"{b_acc:.4f}", help="Anteil korrekter Vorhersagen auf Test-Daten (0-1, höher = besser)")
        else:
            st.caption("Accuracy: N/A")
    with col_b2:
        b_f1 = comparison.get('b_f1')
        if b_f1:
            st.metric("F1-Score", f"{b_f1:.4f}", help="Harmonisches Mittel aus Precision und Recall (0-1, höher = besser)")
        else:
            st.caption("F1-Score: N/A")
    with col_b3:
        b_prec = comparison.get('b_precision')
        if b_prec:
            st.metric("Precision", f"{b_prec:.4f}", help="Von allen 'Positiv'-Vorhersagen, wie viele waren wirklich positiv? (0-1, höher = weniger False Positives)")
        else:
            st.caption("Precision: N/A")
    with col_b4:
        b_rec = comparison.get('b_recall')
        if b_rec:
            st.metric("Recall", f"{b_rec:.4f}", help="Von allen echten Positiven, wie viele hat das Modell gefunden? (0-1, höher = weniger False Negatives)")
        else:
            st.caption("Recall: N/A")
    
    st.divider()
    
    # Zusätzliche Metriken mit Erklärungen
    if comparison.get('a_roc_auc') or comparison.get('a_mcc') or comparison.get('a_fpr') or comparison.get('a_fnr') or comparison.get('a_simulated_profit_pct') or comparison.get('b_roc_auc') or comparison.get('b_mcc') or comparison.get('b_fpr') or comparison.get('b_fnr') or comparison.get('b_simulated_profit_pct'):
        st.subheader("📈 Erweiterte Metriken")
        st.markdown("""
        **Was bedeuten diese Metriken?**
        
        - **ROC-AUC:** Area Under ROC Curve (0-1). Misst die Fähigkeit, zwischen Positiv und Negativ zu unterscheiden. >0.7 = gut, >0.9 = sehr gut.
        - **MCC:** Matthews Correlation Coefficient (-1 bis +1). Berücksichtigt alle 4 Confusion-Matrix-Werte. 0 = zufällig, +1 = perfekt, -1 = perfekt falsch.
        - **FPR (False Positive Rate):** Wie viele Negatives wurden fälschlicherweise als Positiv klassifiziert? (0-1, niedriger = besser)
        - **FNR (False Negative Rate):** Wie viele Positives wurden fälschlicherweise als Negativ klassifiziert? (0-1, niedriger = besser)
        """)
        
        # Modell A erweiterte Metriken
        st.markdown("**Modell A:**")
        col_a1, col_a2, col_a3, col_a4 = st.columns(4)
        with col_a1:
            a_roc_auc = comparison.get('a_roc_auc')
            if a_roc_auc:
                quality = "Sehr gut" if a_roc_auc > 0.9 else "Gut" if a_roc_auc > 0.7 else "Mäßig" if a_roc_auc > 0.5 else "Schlecht"
                st.metric("ROC-AUC", f"{a_roc_auc:.4f}", help=f"Area Under ROC Curve (0-1). {quality} (>0.7 = gut, >0.9 = sehr gut)")
            else:
                st.caption("ROC-AUC: N/A")
        with col_a2:
            a_mcc = comparison.get('a_mcc')
            if a_mcc:
                quality = "Sehr gut" if a_mcc > 0.5 else "Gut" if a_mcc > 0.3 else "Mäßig" if a_mcc > 0 else "Schlecht"
                st.metric("MCC", f"{a_mcc:.4f}", help=f"Matthews Correlation Coefficient (-1 bis +1). {quality} (0 = zufällig, +1 = perfekt)")
            else:
                st.caption("MCC: N/A")
        with col_a3:
            a_fpr = comparison.get('a_fpr')
            if a_fpr is not None:
                quality = "Gut" if a_fpr < 0.1 else "Mäßig" if a_fpr < 0.3 else "Schlecht"
                st.metric("False Positive Rate", f"{a_fpr:.4f}", help=f"Wie viele Negatives wurden fälschlicherweise als Positiv klassifiziert? {quality} (niedriger = besser)")
            else:
                st.caption("FPR: N/A")
        with col_a4:
            a_fnr = comparison.get('a_fnr')
            if a_fnr is not None:
                quality = "Gut" if a_fnr < 0.1 else "Mäßig" if a_fnr < 0.3 else "Schlecht"
                st.metric("False Negative Rate", f"{a_fnr:.4f}", help=f"Wie viele Positives wurden fälschlicherweise als Negativ klassifiziert? {quality} (niedriger = besser)")
            else:
                st.caption("FNR: N/A")
        
        # Profit für Modell A
        a_profit = comparison.get('a_simulated_profit_pct')
        if a_profit is not None:
            profit_quality = "Sehr profitabel" if a_profit > 5 else "Profitabel" if a_profit > 0 else "Verlust"
            st.caption(f"💰 Simulierter Profit Modell A: {a_profit:.2f}% ({profit_quality})")
        
        st.divider()
        
        # Modell B erweiterte Metriken
        st.markdown("**Modell B:**")
        col_b1, col_b2, col_b3, col_b4 = st.columns(4)
        with col_b1:
            b_roc_auc = comparison.get('b_roc_auc')
            if b_roc_auc:
                quality = "Sehr gut" if b_roc_auc > 0.9 else "Gut" if b_roc_auc > 0.7 else "Mäßig" if b_roc_auc > 0.5 else "Schlecht"
                st.metric("ROC-AUC", f"{b_roc_auc:.4f}", help=f"Area Under ROC Curve (0-1). {quality} (>0.7 = gut, >0.9 = sehr gut)")
            else:
                st.caption("ROC-AUC: N/A")
        with col_b2:
            b_mcc = comparison.get('b_mcc')
            if b_mcc:
                quality = "Sehr gut" if b_mcc > 0.5 else "Gut" if b_mcc > 0.3 else "Mäßig" if b_mcc > 0 else "Schlecht"
                st.metric("MCC", f"{b_mcc:.4f}", help=f"Matthews Correlation Coefficient (-1 bis +1). {quality} (0 = zufällig, +1 = perfekt)")
            else:
                st.caption("MCC: N/A")
        with col_b3:
            b_fpr = comparison.get('b_fpr')
            if b_fpr is not None:
                quality = "Gut" if b_fpr < 0.1 else "Mäßig" if b_fpr < 0.3 else "Schlecht"
                st.metric("False Positive Rate", f"{b_fpr:.4f}", help=f"Wie viele Negatives wurden fälschlicherweise als Positiv klassifiziert? {quality} (niedriger = besser)")
            else:
                st.caption("FPR: N/A")
        with col_b4:
            b_fnr = comparison.get('b_fnr')
            if b_fnr is not None:
                quality = "Gut" if b_fnr < 0.1 else "Mäßig" if b_fnr < 0.3 else "Schlecht"
                st.metric("False Negative Rate", f"{b_fnr:.4f}", help=f"Wie viele Positives wurden fälschlicherweise als Negativ klassifiziert? {quality} (niedriger = besser)")
            else:
                st.caption("FNR: N/A")
        
        # Profit für Modell B
        b_profit = comparison.get('b_simulated_profit_pct')
        if b_profit is not None:
            profit_quality = "Sehr profitabel" if b_profit > 5 else "Profitabel" if b_profit > 0 else "Verlust"
            st.caption(f"💰 Simulierter Profit Modell B: {b_profit:.2f}% ({profit_quality})")
        
        # Profit-Vergleich
        if a_profit is not None and b_profit is not None:
            st.divider()
            st.markdown("**💰 Profit-Vergleich:**")
            st.markdown("""
            **Was ist Profit-Simulation?**
            
            Simuliert den Profit, den das Modell erzielt hätte:
            - **True Positive (TP):** +1% Gewinn (korrekt erkannte Pumps)
            - **False Positive (FP):** -0.5% Verlust (fälschlicherweise als Pump erkannt)
            - **True Negative (TN):** 0% (korrekt als "kein Pump" erkannt)
            - **False Negative (FN):** 0% (verpasste Pumps)
            
            **Auswirkung:** Zeigt, welches Modell in der Praxis profitabler wäre.
            """)
            
            profit_diff = a_profit - b_profit
            if profit_diff > 0:
                st.success(f"📈 **Modell A ist profitabler:** {profit_diff:.2f}% mehr Profit als Modell B")
            elif profit_diff < 0:
                st.success(f"📈 **Modell B ist profitabler:** {abs(profit_diff):.2f}% mehr Profit als Modell A")
            else:
                st.info("🤝 **Gleich profitabel:** Beide Modelle hätten den gleichen Profit erzielt")
    
    st.divider()
    
    # Confusion Matrix mit Erklärung
    if comparison.get('a_confusion_matrix') or comparison.get('b_confusion_matrix'):
        st.subheader("🔢 Confusion Matrix")
        st.markdown("""
        **Was ist eine Confusion Matrix?**
        
        Zeigt, wie viele Vorhersagen korrekt und falsch waren:
        - **TP (True Positive):** ✅ Korrekt als "Positiv" erkannt (z.B. Pump erkannt, war wirklich Pump)
        - **TN (True Negative):** ✅ Korrekt als "Negativ" erkannt (z.B. kein Pump erkannt, war wirklich kein Pump)
        - **FP (False Positive):** ❌ Fälschlicherweise als "Positiv" erkannt (z.B. Pump erkannt, war aber kein Pump) → Verluste!
        - **FN (False Negative):** ❌ Fälschlicherweise als "Negativ" erkannt (z.B. kein Pump erkannt, war aber Pump) → Verpasste Chancen!
        
        **Auswirkung:** 
        - Viele TP = Modell erkennt Pumps gut
        - Viele FP = Modell ist zu optimistisch (viele Fehlalarme)
        - Viele FN = Modell verpasst viele Pumps
        """)
        
        col_a, col_b = st.columns(2)
        
        with col_a:
            st.markdown(f"**Modell A ({model_a_name}):**")
            if comparison.get('a_confusion_matrix'):
                cm_a = comparison['a_confusion_matrix']
                cm_a_col1, cm_a_col2, cm_a_col3, cm_a_col4 = st.columns(4)
                with cm_a_col1:
                    tp_a = cm_a.get('tp', 0)
                    st.metric("TP", tp_a, help="✅ Korrekt als 'Positiv' erkannt")
                with cm_a_col2:
                    tn_a = cm_a.get('tn', 0)
                    st.metric("TN", tn_a, help="✅ Korrekt als 'Negativ' erkannt")
                with cm_a_col3:
                    fp_a = cm_a.get('fp', 0)
                    st.metric("FP", fp_a, help="❌ Fälschlicherweise als 'Positiv' erkannt → Verluste!")
                with cm_a_col4:
                    fn_a = cm_a.get('fn', 0)
                    st.metric("FN", fn_a, help="❌ Fälschlicherweise als 'Negativ' erkannt → Verpasste Chancen!")
                
                # Tabelle
                cm_a_data = {
                    'Tatsächlich': ['Negativ', 'Positiv'],
                    'Vorhergesagt: Negativ': [tn_a, fn_a],
                    'Vorhergesagt: Positiv': [fp_a, tp_a]
                }
                cm_a_df = pd.DataFrame(cm_a_data)
                st.dataframe(cm_a_df, use_container_width=True, hide_index=True)
            else:
                st.write("N/A")
        
        with col_b:
            st.markdown(f"**Modell B ({model_b_name}):**")
            if comparison.get('b_confusion_matrix'):
                cm_b = comparison['b_confusion_matrix']
                cm_b_col1, cm_b_col2, cm_b_col3, cm_b_col4 = st.columns(4)
                with cm_b_col1:
                    tp_b = cm_b.get('tp', 0)
                    st.metric("TP", tp_b, help="✅ Korrekt als 'Positiv' erkannt")
                with cm_b_col2:
                    tn_b = cm_b.get('tn', 0)
                    st.metric("TN", tn_b, help="✅ Korrekt als 'Negativ' erkannt")
                with cm_b_col3:
                    fp_b = cm_b.get('fp', 0)
                    st.metric("FP", fp_b, help="❌ Fälschlicherweise als 'Positiv' erkannt → Verluste!")
                with cm_b_col4:
                    fn_b = cm_b.get('fn', 0)
                    st.metric("FN", fn_b, help="❌ Fälschlicherweise als 'Negativ' erkannt → Verpasste Chancen!")
                
                # Tabelle
                cm_b_data = {
                    'Tatsächlich': ['Negativ', 'Positiv'],
                    'Vorhergesagt: Negativ': [tn_b, fn_b],
                    'Vorhergesagt: Positiv': [fp_b, tp_b]
                }
                cm_b_df = pd.DataFrame(cm_b_data)
                st.dataframe(cm_b_df, use_container_width=True, hide_index=True)
            else:
                st.write("N/A")
    
    st.divider()
    
    # Train vs. Test Vergleich mit Erklärung
    if comparison.get('a_train_accuracy') or comparison.get('b_train_accuracy'):
        st.subheader("📊 Train vs. Test Vergleich")
        st.markdown("""
        **Was bedeutet Train vs. Test Vergleich?**
        
        Vergleicht die Performance auf Trainings- und Test-Daten:
        - **Train Accuracy:** Performance auf den Daten, mit denen das Modell trainiert wurde
        - **Test Accuracy:** Performance auf neuen, ungesehenen Daten
        - **Degradation:** Unterschied zwischen Train- und Test-Accuracy
        
        **Auswirkung:** 
        - Große Degradation (>10%) = Modell ist möglicherweise overfitted (lernt zu spezifisch)
        - Kleine Degradation (<10%) = Modell generalisiert gut auf neue Daten
        """)
        
        # Modell A
        st.markdown("**Modell A:**")
        col_a1, col_a2, col_a3 = st.columns(3)
        with col_a1:
            a_train_acc = comparison.get('a_train_accuracy')
            if a_train_acc:
                st.metric("Train Accuracy", f"{a_train_acc:.4f}", help="Performance auf Trainingsdaten (0-1, höher = besser)")
            else:
                st.caption("Train Accuracy: N/A")
        with col_a2:
            a_test_acc = comparison.get('a_accuracy')
            if a_test_acc:
                st.metric("Test Accuracy", f"{a_test_acc:.4f}", help="Performance auf Test-Daten (0-1, höher = besser)")
            else:
                st.caption("Test Accuracy: N/A")
        with col_a3:
            a_degradation = comparison.get('a_accuracy_degradation')
            if a_degradation is not None:
                quality = "✅ OK" if a_degradation < 0.1 else "⚠️ Overfitting-Risiko"
                st.metric("Degradation", f"{a_degradation:.4f}", 
                         delta=quality,
                         help=f"Unterschied zwischen Train- und Test-Accuracy. {quality} (niedriger = besser)")
            else:
                st.caption("Degradation: N/A")
        
        st.divider()
        
        # Modell B
        st.markdown("**Modell B:**")
        col_b1, col_b2, col_b3 = st.columns(3)
        with col_b1:
            b_train_acc = comparison.get('b_train_accuracy')
            if b_train_acc:
                st.metric("Train Accuracy", f"{b_train_acc:.4f}", help="Performance auf Trainingsdaten (0-1, höher = besser)")
            else:
                st.caption("Train Accuracy: N/A")
        with col_b2:
            b_test_acc = comparison.get('b_accuracy')
            if b_test_acc:
                st.metric("Test Accuracy", f"{b_test_acc:.4f}", help="Performance auf Test-Daten (0-1, höher = besser)")
            else:
                st.caption("Test Accuracy: N/A")
        with col_b3:
            b_degradation = comparison.get('b_accuracy_degradation')
            if b_degradation is not None:
                quality = "✅ OK" if b_degradation < 0.1 else "⚠️ Overfitting-Risiko"
                st.metric("Degradation", f"{b_degradation:.4f}", 
                         delta=quality,
                         help=f"Unterschied zwischen Train- und Test-Accuracy. {quality} (niedriger = besser)")
            else:
                st.caption("Degradation: N/A")
        
        # Overfitting-Warnungen
        if a_degradation and a_degradation > 0.1:
            st.warning(f"⚠️ **Modell A ist möglicherweise overfitted!** Die Performance auf Test-Daten ist deutlich schlechter als auf Trainingsdaten (Degradation: {a_degradation:.2%})")
        if b_degradation and b_degradation > 0.1:
            st.warning(f"⚠️ **Modell B ist möglicherweise overfitted!** Die Performance auf Test-Daten ist deutlich schlechter als auf Trainingsdaten (Degradation: {b_degradation:.2%})")
    
    st.divider()
    
    # Test-Zeitraum mit Erklärung
    st.subheader("📅 Test-Zeitraum")
    st.markdown("""
    **Was ist der Test-Zeitraum?**
    
    Der Test-Zeitraum definiert, welche Daten zum Testen beider Modelle verwendet wurden.
    Diese Daten wurden **nicht** zum Training verwendet.
    
    **Empfehlung:** Mindestens 1 Tag Test-Daten für realistische Ergebnisse.
    """)
    
    test_start = comparison.get('test_start')
    test_end = comparison.get('test_end')
    test_duration_days = comparison.get('a_test_duration_days') or comparison.get('test_duration_days')
    
    if test_start and test_end:
        try:
            if isinstance(test_start, str):
                start_dt = datetime.fromisoformat(test_start.replace('Z', '+00:00'))
            else:
                start_dt = test_start
            if isinstance(test_end, str):
                end_dt = datetime.fromisoformat(test_end.replace('Z', '+00:00'))
            else:
                end_dt = test_end
            
            start_str = start_dt.strftime("%d.%m.%Y %H:%M")
            end_str = end_dt.strftime("%d.%m.%Y %H:%M")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.write(f"**Start:** {start_str}")
            with col2:
                st.write(f"**Ende:** {end_str}")
            with col3:
                if test_duration_days:
                    st.write(f"**Dauer:** {test_duration_days:.2f} Tage")
                    if test_duration_days < 1:
                        st.warning("⚠️ Test-Zeitraum zu kurz (empfohlen: mindestens 1 Tag)")
                else:
                    days = (end_dt - start_dt).total_seconds() / 86400.0
                    st.write(f"**Dauer:** {days:.2f} Tage")
                    if days < 1:
                        st.warning("⚠️ Test-Zeitraum zu kurz (empfohlen: mindestens 1 Tag)")
        except Exception as e:
            st.write(f"Start: {test_start}")
            st.write(f"Ende: {test_end}")
    else:
        st.write("Test-Zeitraum nicht verfügbar")
    
    st.divider()
    
    # Vollständige Details
    with st.expander("📋 Vollständige Details (JSON)", expanded=False):
        st.json(comparison)
    
    # Zurück-Button
    col1, col2 = st.columns([1, 3])
    with col1:
        if st.button("← Zurück zur Vergleichs-Übersicht", use_container_width=True):
            st.session_state['page'] = 'comparisons'
            st.session_state.pop('comparison_details_id', None)
            st.rerun()

def page_details():
    """Modell-Details"""
    model_id = st.session_state.get('details_model_id')
    if not model_id:
        st.warning("⚠️ Kein Modell ausgewählt")
        return
    
    model = api_get(f"/api/models/{model_id}")
    if not model:
        st.error("❌ Modell nicht gefunden")
        return
    
    st.title(f"📋 Modell-Details: {model.get('name')}")
    
    # Info-Box am Anfang
    st.info("""
    **📖 Anleitung:** 
    Diese Seite zeigt alle Details und Metriken des Modells. Nutze die ℹ️-Icons für Erklärungen zu jedem Wert.
    """)
    
    # Basis-Informationen
    st.subheader("📝 Basis-Informationen")
    info_col1, info_col2, info_col3, info_col4 = st.columns(4)
    with info_col1:
        st.markdown("**Modell-Typ**")
        model_type = model.get('model_type', 'N/A')
        type_emoji = "🌲" if model_type == "random_forest" else "🚀" if model_type == "xgboost" else "🤖"
        st.write(f"{type_emoji} {model_type}")
    with info_col2:
        st.markdown("**Status**")
        status = model.get('status', 'N/A')
        if status == "READY":
            st.success("✅ READY")
        elif status == "TRAINING":
            st.info("🔄 TRAINING")
        else:
            st.error(f"❌ {status}")
    with info_col3:
        st.markdown("**Modell-ID**")
        st.write(f"#{model_id}")
    with info_col4:
        st.markdown("**Erstellt**")
        created = model.get('created_at', '')
        if created:
            try:
                if isinstance(created, str):
                    created_dt = datetime.fromisoformat(created.replace('Z', '+00:00'))
                else:
                    created_dt = created
                st.write(created_dt.strftime("%d.%m.%Y %H:%M"))
            except:
                st.write(str(created)[:19] if len(str(created)) > 19 else str(created))
        else:
            st.write("N/A")
    
    # Beschreibung
    description = model.get('description')
    if description:
        st.markdown("**Beschreibung**")
        st.info(description)
    
    st.divider()
    
    # Standard-Metriken mit Erklärungen
    st.subheader("📊 Standard-Metriken")
    st.markdown("""
    **Was bedeuten diese Metriken?**
    
    - **Accuracy:** Anteil korrekter Vorhersagen (0-1). Beispiel: 0.85 = 85% der Vorhersagen sind richtig.
    - **F1-Score:** Harmonisches Mittel aus Precision und Recall (0-1). Gut für unausgewogene Daten.
    - **Precision:** Von allen "Positiv"-Vorhersagen, wie viele waren wirklich positiv? (0-1, höher = weniger False Positives)
    - **Recall:** Von allen echten Positiven, wie viele hat das Modell gefunden? (0-1, höher = weniger False Negatives)
    """)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        accuracy = model.get('training_accuracy')
        if accuracy:
            st.metric("Accuracy", f"{accuracy:.4f}", help="Anteil korrekter Vorhersagen (0-1, höher = besser)")
        else:
            st.caption("Accuracy: N/A")
    with col2:
        f1 = model.get('training_f1')
        if f1:
            st.metric("F1-Score", f"{f1:.4f}", help="Harmonisches Mittel aus Precision und Recall (0-1, höher = besser)")
        else:
            st.caption("F1-Score: N/A")
    with col3:
        precision = model.get('training_precision')
        if precision:
            st.metric("Precision", f"{precision:.4f}", help="Von allen 'Positiv'-Vorhersagen, wie viele waren wirklich positiv? (0-1, höher = weniger False Positives)")
        else:
            st.caption("Precision: N/A")
    with col4:
        recall = model.get('training_recall')
        if recall:
            st.metric("Recall", f"{recall:.4f}", help="Von allen echten Positiven, wie viele hat das Modell gefunden? (0-1, höher = weniger False Negatives)")
        else:
            st.caption("Recall: N/A")
    
    st.divider()
    
    # Zusätzliche Metriken mit Erklärungen
    st.subheader("📈 Erweiterte Metriken")
    st.markdown("""
    **Was bedeuten diese Metriken?**
    
    - **ROC-AUC:** Area Under ROC Curve (0-1). Misst die Fähigkeit, zwischen Positiv und Negativ zu unterscheiden. >0.7 = gut, >0.9 = sehr gut.
    - **MCC:** Matthews Correlation Coefficient (-1 bis +1). Berücksichtigt alle 4 Confusion-Matrix-Werte. 0 = zufällig, +1 = perfekt, -1 = perfekt falsch.
    - **FPR (False Positive Rate):** Wie viele Negatives wurden fälschlicherweise als Positiv klassifiziert? (0-1, niedriger = besser)
    - **FNR (False Negative Rate):** Wie viele Positives wurden fälschlicherweise als Negativ klassifiziert? (0-1, niedriger = besser)
    """)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        roc_auc = model.get('roc_auc')
        if roc_auc:
            quality = "Sehr gut" if roc_auc > 0.9 else "Gut" if roc_auc > 0.7 else "Mäßig" if roc_auc > 0.5 else "Schlecht"
            st.metric("ROC-AUC", f"{roc_auc:.4f}", help=f"Area Under ROC Curve (0-1). {quality} (>0.7 = gut, >0.9 = sehr gut)")
        else:
            st.caption("ROC-AUC: N/A")
    with col2:
        mcc = model.get('mcc')
        if mcc:
            quality = "Sehr gut" if mcc > 0.5 else "Gut" if mcc > 0.3 else "Mäßig" if mcc > 0 else "Schlecht"
            st.metric("MCC", f"{mcc:.4f}", help=f"Matthews Correlation Coefficient (-1 bis +1). {quality} (0 = zufällig, +1 = perfekt)")
        else:
            st.caption("MCC: N/A")
    with col3:
        fpr = model.get('fpr')
        if fpr is not None:
            quality = "Gut" if fpr < 0.1 else "Mäßig" if fpr < 0.3 else "Schlecht"
            st.metric("False Positive Rate", f"{fpr:.4f}", help=f"Wie viele Negatives wurden fälschlicherweise als Positiv klassifiziert? {quality} (niedriger = besser)")
        else:
            st.caption("FPR: N/A")
    with col4:
        fnr = model.get('fnr')
        if fnr is not None:
            quality = "Gut" if fnr < 0.1 else "Mäßig" if fnr < 0.3 else "Schlecht"
            st.metric("False Negative Rate", f"{fnr:.4f}", help=f"Wie viele Positives wurden fälschlicherweise als Negativ klassifiziert? {quality} (niedriger = besser)")
        else:
            st.caption("FNR: N/A")
    
    st.divider()
    
    # Profit-Simulation mit Erklärung
    simulated_profit = model.get('simulated_profit_pct')
    if simulated_profit is not None:
        st.subheader("💰 Profit-Simulation")
        st.markdown("""
        **Was ist Profit-Simulation?**
        
        Simuliert den Profit, den das Modell erzielt hätte:
        - **True Positive (TP):** +1% Gewinn (korrekt erkannte Pumps)
        - **False Positive (FP):** -0.5% Verlust (fälschlicherweise als Pump erkannt)
        - **True Negative (TN):** 0% (korrekt als "kein Pump" erkannt)
        - **False Negative (FN):** 0% (verpasste Pumps)
        
        **Auswirkung:** Zeigt, wie profitabel das Modell in der Praxis wäre.
        """)
        
        col1, col2 = st.columns(2)
        with col1:
            profit_quality = "Sehr profitabel" if simulated_profit > 5 else "Profitabel" if simulated_profit > 0 else "Verlust"
            st.metric("💰 Simulierter Profit", f"{simulated_profit:.2f}%", help=f"Simulierter Profit basierend auf TP/FP. {profit_quality}")
        with col2:
            st.caption("**Berechnung:** 1% Gewinn pro TP, -0.5% Verlust pro FP")
    
    st.divider()
    
    # Confusion Matrix mit Erklärung
    confusion_matrix = model.get('confusion_matrix')
    if confusion_matrix:
        st.subheader("🔢 Confusion Matrix")
        st.markdown("""
        **Was ist eine Confusion Matrix?**
        
        Zeigt, wie viele Vorhersagen korrekt und falsch waren:
        - **TP (True Positive):** ✅ Korrekt als "Positiv" erkannt (z.B. Pump erkannt, war wirklich Pump)
        - **TN (True Negative):** ✅ Korrekt als "Negativ" erkannt (z.B. kein Pump erkannt, war wirklich kein Pump)
        - **FP (False Positive):** ❌ Fälschlicherweise als "Positiv" erkannt (z.B. Pump erkannt, war aber kein Pump) → Verluste!
        - **FN (False Negative):** ❌ Fälschlicherweise als "Negativ" erkannt (z.B. kein Pump erkannt, war aber Pump) → Verpasste Chancen!
        
        **Auswirkung:** 
        - Viele TP = Modell erkennt Pumps gut
        - Viele FP = Modell ist zu optimistisch (viele Fehlalarme)
        - Viele FN = Modell verpasst viele Pumps
        """)
        
        cm_col1, cm_col2, cm_col3, cm_col4 = st.columns(4)
        with cm_col1:
            tp = confusion_matrix.get('tp', 0)
            st.metric("True Positive (TP)", tp, help="✅ Korrekt als 'Positiv' erkannt (z.B. Pump erkannt, war wirklich Pump)")
        with cm_col2:
            tn = confusion_matrix.get('tn', 0)
            st.metric("True Negative (TN)", tn, help="✅ Korrekt als 'Negativ' erkannt (z.B. kein Pump erkannt, war wirklich kein Pump)")
        with cm_col3:
            fp = confusion_matrix.get('fp', 0)
            st.metric("False Positive (FP)", fp, help="❌ Fälschlicherweise als 'Positiv' erkannt (z.B. Pump erkannt, war aber kein Pump) → Verluste!")
        with cm_col4:
            fn = confusion_matrix.get('fn', 0)
            st.metric("False Negative (FN)", fn, help="❌ Fälschlicherweise als 'Negativ' erkannt (z.B. kein Pump erkannt, war aber Pump) → Verpasste Chancen!")
        
        # Visualisierung als Tabelle
        st.markdown("**Confusion Matrix Tabelle:**")
        cm_data = {
            'Tatsächlich': ['Negativ', 'Positiv'],
            'Vorhergesagt: Negativ': [tn, fn],
            'Vorhergesagt: Positiv': [fp, tp]
        }
        cm_df = pd.DataFrame(cm_data)
        st.dataframe(cm_df, use_container_width=True, hide_index=True)
        
        # Interpretation
        total = tp + tn + fp + fn
        if total > 0:
            tp_rate = (tp / total) * 100
            fp_rate = (fp / total) * 100
            fn_rate = (fn / total) * 100
            st.caption(f"ℹ️ Verteilung: {tp_rate:.1f}% TP, {tn/total*100:.1f}% TN, {fp_rate:.1f}% FP, {fn_rate:.1f}% FN")
    
    st.divider()
    
    # Feature Importance Chart mit Erklärung
    if model.get('feature_importance'):
        st.subheader("🎯 Feature Importance")
        st.markdown("""
        **Was ist Feature Importance?**
        
        Zeigt, welche Features am wichtigsten für die Vorhersage sind:
        - **Höhere Werte** = Feature ist wichtiger für die Vorhersage
        - **Niedrigere Werte** = Feature ist weniger wichtig
        
        **Auswirkung:** 
        - Wichtige Features sollten beibehalten werden
        - Unwichtige Features könnten entfernt werden (Feature Selection)
        - Hilft zu verstehen, worauf das Modell basiert
        """)
        
        fi = model['feature_importance']
        if isinstance(fi, dict):
            df_fi = pd.DataFrame(list(fi.items()), columns=['Feature', 'Importance'])
            df_fi = df_fi.sort_values('Importance', ascending=False)
            
            # Zeige Top 10 Features
            st.markdown("**Top 10 wichtigste Features:**")
            st.dataframe(df_fi.head(10), use_container_width=True, hide_index=True)
            
            # Visualisierung
            fig = px.bar(df_fi.head(15), x='Feature', y='Importance', title="Feature Importance (Top 15)")
            fig.update_xaxes(tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
    
    st.divider()
    
    # Modell-Konfiguration
    st.subheader("⚙️ Modell-Konfiguration")
    
    config_col1, config_col2 = st.columns(2)
    
    with config_col1:
        st.markdown("**Training-Zeitraum:**")
        train_start = model.get('train_start')
        train_end = model.get('train_end')
        if train_start and train_end:
            try:
                if isinstance(train_start, str):
                    start_dt = datetime.fromisoformat(train_start.replace('Z', '+00:00'))
                else:
                    start_dt = train_start
                if isinstance(train_end, str):
                    end_dt = datetime.fromisoformat(train_end.replace('Z', '+00:00'))
                else:
                    end_dt = train_end
                
                start_str = start_dt.strftime("%d.%m.%Y %H:%M")
                end_str = end_dt.strftime("%d.%m.%Y %H:%M")
                duration_days = (end_dt - start_dt).total_seconds() / 86400.0
                st.write(f"**Start:** {start_str}")
                st.write(f"**Ende:** {end_str}")
                st.write(f"**Dauer:** {duration_days:.1f} Tage")
            except Exception as e:
                st.write(f"Start: {train_start}")
                st.write(f"Ende: {train_end}")
        else:
            st.write("Zeitraum nicht verfügbar")
        
        st.markdown("**Features:**")
        features_list = model.get('features', [])
        if features_list:
            st.write(f"{len(features_list)} Features ausgewählt")
            with st.expander("Alle Features anzeigen"):
                for feat in features_list:
                    st.write(f"- {feat}")
        else:
            st.write("Keine Features verfügbar")
        
        st.markdown("**Phasen:**")
        phases_list = model.get('phases')
        if phases_list:
            st.write(f"{len(phases_list)} Phase(n) ausgewählt")
            with st.expander("Phasen anzeigen"):
                for phase_id in phases_list:
                    st.write(f"- Phase {phase_id}")
        else:
            st.write("Alle Phasen verwendet")
    
    with config_col2:
        st.markdown("**Ziel-Variable:**")
        target_var = model.get('target_variable', 'N/A')
        target_operator = model.get('target_operator')
        target_value = model.get('target_value')
        
        # Zeitbasierte Vorhersage?
        future_minutes = model.get('future_minutes')
        price_change = model.get('price_change_percent')
        direction = model.get('target_direction')
        
        if future_minutes and price_change:
            st.write(f"**Typ:** ⏰ Zeitbasierte Vorhersage")
            st.write(f"**Variable:** {target_var}")
            st.write(f"**Zeitraum:** {future_minutes} Minuten")
            st.write(f"**Min. Änderung:** {price_change}%")
            direction_text = "📈 Steigt" if direction == "up" else "📉 Fällt" if direction == "down" else "N/A"
            st.write(f"**Richtung:** {direction_text}")
            st.write(f"**Ziel:** {target_var} {direction_text.lower()} in {future_minutes} Minuten um mindestens {price_change}%")
        else:
            st.write(f"**Variable:** {target_var}")
            if target_operator and target_value is not None:
                st.write(f"**Bedingung:** {target_var} {target_operator} {target_value}")
            else:
                st.write("**Bedingung:** Nicht konfiguriert")
        
        st.markdown("**Parameter:**")
        params = model.get('params', {})
        if isinstance(params, str):
            import json
            try:
                params = json.loads(params)
            except:
                params = {}
        
        if params:
            # Zeige wichtige Parameter
            if params.get('use_engineered_features'):
                st.write("🔧 Feature-Engineering: ✅ Aktiviert")
                windows = params.get('feature_engineering_windows', [])
                if windows:
                    st.write(f"   Fenster: {windows}")
            
            if params.get('_time_based', {}).get('enabled'):
                time_based_params = params.get('_time_based', {})
                tb_future_minutes = time_based_params.get('future_minutes') or future_minutes
                tb_min_percent = time_based_params.get('min_percent_change') or price_change
                tb_direction = time_based_params.get('direction') or direction
                if tb_future_minutes and tb_min_percent:
                    direction_text = "steigt" if tb_direction == "up" else "fällt" if tb_direction == "down" else ""
                    st.write(f"⏰ Zeitbasierte Vorhersage: ✅ Aktiviert ({tb_future_minutes}min, {tb_min_percent}% {direction_text})")
                else:
                    st.write("⏰ Zeitbasierte Vorhersage: ✅ Aktiviert")
            
            if params.get('use_smote') is False:
                st.write("⚖️ SMOTE: ❌ Deaktiviert")
            else:
                st.write("⚖️ SMOTE: ✅ Aktiviert (oder automatisch)")
            
            if params.get('use_timeseries_split') is False:
                st.write("🔀 TimeSeriesSplit: ❌ Deaktiviert")
            else:
                st.write("🔀 TimeSeriesSplit: ✅ Aktiviert (oder automatisch)")
            
            cv_splits = params.get('cv_splits')
            if cv_splits:
                st.write(f"🔀 Cross-Validation: {cv_splits} Splits")
            
            # Hyperparameter
            n_estimators = params.get('n_estimators')
            max_depth = params.get('max_depth')
            if n_estimators or max_depth:
                st.write("⚙️ Hyperparameter:")
                if n_estimators:
                    st.write(f"   n_estimators: {n_estimators}")
                if max_depth:
                    st.write(f"   max_depth: {max_depth}")
                learning_rate = params.get('learning_rate')
                if learning_rate:
                    st.write(f"   learning_rate: {learning_rate}")
        else:
            st.write("Keine Parameter verfügbar")
    
    st.divider()
    
    # CV-Scores
    cv_scores = model.get('cv_scores')
    if cv_scores:
        st.subheader("🔀 Cross-Validation Ergebnisse")
        st.markdown("""
        **Was sind CV-Scores?**
        
        Cross-Validation teilt die Daten in mehrere Teile auf und testet das Modell auf jedem Teil.
        Dies gibt eine realistischere Einschätzung der Modell-Performance.
        
        **Auswirkung:** 
        - Zeigt, wie konsistent das Modell über verschiedene Daten-Teile ist
        - Niedrige Standardabweichung = Modell ist stabil
        - Hohe Standardabweichung = Modell ist instabil
        """)
        
        if isinstance(cv_scores, dict):
            cv_col1, cv_col2, cv_col3 = st.columns(3)
            with cv_col1:
                mean_score = cv_scores.get('mean_score')
                if mean_score is not None:
                    st.metric("Durchschnittlicher Score", f"{mean_score:.4f}", help="Durchschnittliche Performance über alle CV-Splits")
            with cv_col2:
                std_score = cv_scores.get('std_score')
                if std_score is not None:
                    st.metric("Standardabweichung", f"{std_score:.4f}", help="Wie stark variiert die Performance? (niedriger = stabiler)")
            with cv_col3:
                cv_overfitting = model.get('cv_overfitting_gap')
                if cv_overfitting is not None:
                    quality = "OK" if cv_overfitting < 0.1 else "⚠️ Overfitting-Risiko"
                    st.metric("Overfitting-Gap", f"{cv_overfitting:.4f}", help=f"Unterschied zwischen Train- und CV-Score. {quality} (niedriger = besser)")
            
            # Einzelne Scores
            individual_scores = cv_scores.get('scores', [])
            if individual_scores:
                st.markdown("**Einzelne CV-Scores:**")
                st.write(f"Scores: {[f'{s:.4f}' for s in individual_scores]}")
    
    st.divider()
    
    # Vollständige Details
    with st.expander("📋 Vollständige Details (JSON)", expanded=False):
        st.json(model)
    
    # Zurück-Button (auch in Sidebar verfügbar)
    col1, col2 = st.columns([1, 3])
    with col1:
        if st.button("← Zurück zur Übersicht", use_container_width=True):
            st.session_state['page'] = 'overview'
            st.session_state.pop('details_model_id', None)
            st.rerun()

# ============================================================
# Main App
# ============================================================

def main():
    """Hauptfunktion"""
    # Sidebar Navigation
    st.sidebar.title("🤖 ML Training Service")
    
    # Seiten-Auswahl
    pages = {
        "🏠 Übersicht": "overview",
        "➕ Neues Modell trainieren": "train",
        "🧪 Modell testen": "test",
        "📋 Test-Ergebnisse": "test_results",
        "⚔️ Modelle vergleichen": "compare",
        "⚖️ Vergleichs-Übersicht": "comparisons",
        "📊 Jobs": "jobs"
    }
    
    # Initialisiere Session State
    if 'page' not in st.session_state:
        st.session_state['page'] = 'overview'
    
    # Bestimme aktuelle page
    current_page_value = st.session_state.get('page', 'overview')
    
    # Wenn page nicht in Sidebar ist (z.B. 'details', 'comparison_details', 'test_details'), zeige entsprechende Seite in Sidebar
    sidebar_page_value = current_page_value
    if current_page_value == 'details':
        sidebar_page_value = 'overview'
    elif current_page_value == 'comparison_details':
        sidebar_page_value = 'comparisons'
    elif current_page_value == 'test_details':
        sidebar_page_value = 'test_results'
    elif current_page_value not in pages.values():
        sidebar_page_value = 'overview'
    
    # Navigation mit Buttons statt Radio - zuverlässiger
    st.sidebar.markdown("**Navigation**")
    for page_key, page_value in pages.items():
        # Markiere aktuelle Seite
        is_active = (page_value == sidebar_page_value)
        button_type = "primary" if is_active else "secondary"
        
        if st.sidebar.button(page_key, key=f"nav_{page_value}", use_container_width=True, type=button_type):
            if page_value != current_page_value:
                st.session_state['page'] = page_value
                st.rerun()
    
    # Details-Seite Indikator (wenn aktiv)
    if current_page_value == 'details':
        model_id = st.session_state.get('details_model_id')
        if model_id:
            st.sidebar.markdown("---")
            st.sidebar.markdown("**📋 Modell-Details**")
            st.sidebar.caption(f"Modell ID: {model_id}")
            if st.sidebar.button("← Zurück zur Übersicht", key="back_to_overview", use_container_width=True):
                st.session_state['page'] = 'overview'
                st.session_state.pop('details_model_id', None)
                st.rerun()
    
    # Vergleichs-Details-Seite Indikator (wenn aktiv)
    if current_page_value == 'comparison_details':
        comparison_id = st.session_state.get('comparison_details_id')
        if comparison_id:
            st.sidebar.markdown("---")
            st.sidebar.markdown("**📋 Vergleichs-Details**")
            st.sidebar.caption(f"Vergleich ID: {comparison_id}")
            if st.sidebar.button("← Zurück zur Vergleichs-Übersicht", key="back_to_comparisons", use_container_width=True):
                st.session_state['page'] = 'comparisons'
                st.session_state.pop('comparison_details_id', None)
                st.rerun()
    
    # Test-Details-Seite Indikator (wenn aktiv)
    if current_page_value == 'test_details':
        test_id = st.session_state.get('test_details_id')
        if test_id:
            st.sidebar.markdown("---")
            st.sidebar.markdown("**📋 Test-Details**")
            st.sidebar.caption(f"Test ID: {test_id}")
            if st.sidebar.button("← Zurück zur Test-Übersicht", key="back_to_test_results", use_container_width=True):
                st.session_state['page'] = 'test_results'
                st.session_state.pop('test_details_id', None)
                st.rerun()
    
    # Health Check
    health = api_get("/api/health")
    if health:
        status_emoji = "✅" if health.get('status') == 'healthy' else "⚠️"
        st.sidebar.markdown(f"**Status:** {status_emoji} {health.get('status', 'unknown')}")
        st.sidebar.markdown(f"**DB:** {'✅' if health.get('db_connected') else '❌'}")
    
    # Seiten rendern
    if st.session_state['page'] == 'overview':
        page_overview()
    elif st.session_state['page'] == 'train':
        page_train()
    elif st.session_state['page'] == 'test':
        page_test()
    elif st.session_state['page'] == 'test_results':
        page_test_results()
    elif st.session_state['page'] == 'compare':
        page_compare()
    elif st.session_state['page'] == 'comparisons':
        page_comparisons()
    elif st.session_state['page'] == 'jobs':
        page_jobs()
    elif st.session_state['page'] == 'details':
        page_details()
    elif st.session_state['page'] == 'comparison_details':
        page_comparison_details()
    elif st.session_state['page'] == 'test_details':
        page_test_details()
    else:
        page_overview()

if __name__ == "__main__":
    main()

