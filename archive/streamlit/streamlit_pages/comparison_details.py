"""
Comparison Details Page Module
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


def page_comparison_details():
    """Details-Seite für einen Vergleich - Überarbeitete Version mit besserer Übersichtlichkeit"""
    
    st.divider()
    
    comparison_id = st.session_state.get('comparison_details_id')
    if not comparison_id:
        st.error("❌ Keine Vergleichs-ID gefunden")
        return
    
    comparison = api_get(f"/api/comparisons/{comparison_id}")
    if not comparison:
        st.error("❌ Vergleich nicht gefunden")
        return
    
    # Hole Modell-Namen und Details (optimiert)
    model_a_id = comparison.get('model_a_id')
    model_b_id = comparison.get('model_b_id')

    # Modell-Namen aus Vergleichsdaten falls verfügbar, sonst API-Call
    model_a_name = comparison.get('model_a_name', f"Modell {model_a_id}")
    model_b_name = comparison.get('model_b_name', f"Modell {model_b_id}")

    # Nur API-Calls machen wenn Namen nicht in Vergleichsdaten
    model_a = None
    model_b = None

    if model_a_name.startswith("Modell "):  # Fallback wurde verwendet
        model_a = api_get(f"/api/models/{model_a_id}")
        if model_a:
            model_a_name = model_a.get('name', f"ID: {model_a_id}")

    if model_b_name.startswith("Modell "):  # Fallback wurde verwendet
        model_b = api_get(f"/api/models/{model_b_id}")
        if model_b:
            model_b_name = model_b.get('name', f"ID: {model_b_id}")

    # Trainingszeiträume laden (nur wenn nicht schon verfügbar)
    model_a_train_start = comparison.get('model_a_train_start')
    model_a_train_end = comparison.get('model_a_train_end')
    model_b_train_start = comparison.get('model_b_train_start')
    model_b_train_end = comparison.get('model_b_train_end')

    # Fallback zu API wenn nicht in Vergleichsdaten
    if model_a_train_start is None and model_a:
        model_a_train_start = model_a.get('train_start')
        model_a_train_end = model_a.get('train_end')

    if model_b_train_start is None and model_b:
        model_b_train_start = model_b.get('train_start')
        model_b_train_end = model_b.get('train_end')

    # Gewinner-Info
    winner_id = comparison.get('winner_id')
    winner_name = None
    if winner_id:
        if winner_id == model_a_id:
            winner_name = model_a_name
        elif winner_id == model_b_id:
            winner_name = model_b_name

    # Metriken aus Vergleich direkt laden (schneller und zuverlässiger)
    a_accuracy = comparison.get('a_accuracy', 0)
    b_accuracy = comparison.get('b_accuracy', 0)
    a_f1 = comparison.get('a_f1', 0)
    b_f1 = comparison.get('b_f1', 0)

    # Header mit wichtigen Infos
    header_col1, header_col2, header_col3 = st.columns([4, 1, 1])
    with header_col1:
        st.title(f"⚖️ {model_a_name} vs {model_b_name}")

        # Gewinner-Status
        if winner_name:
            st.success(f"🏆 **Gewinner: {winner_name}**")
        else:
            st.info("🤝 **Unentschieden**")
    with header_col2:
        st.metric("Accuracy A", f"{a_accuracy:.3f}")
        st.metric("Accuracy B", f"{b_accuracy:.3f}")
    with header_col3:
        st.metric("F1-Score A", f"{a_f1:.3f}")
        st.metric("F1-Score B", f"{b_f1:.3f}")
    
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
    
    # Trainings-Zeiträume beider Modelle - in Expander für Performance
    with st.expander("🎓 Trainings-Zeiträume der Modelle", expanded=False):
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
    
    # Detaillierte Metriken im Vergleich
    st.subheader("📊 Detaillierte Metriken")

    # Erstelle eine kompakte Vergleichstabelle
    import pandas as pd

    # Basis-Metriken immer anzeigen
    basic_metrics = {
        'Metrik': ['Accuracy', 'F1-Score', 'Precision', 'Recall'],
        f'{model_a_name}': [
            f"{comparison.get('a_accuracy', 'N/A')}",
            f"{comparison.get('a_f1', 'N/A')}",
            f"{comparison.get('a_precision', 'N/A')}",
            f"{comparison.get('a_recall', 'N/A')}"
        ],
        f'{model_b_name}': [
            f"{comparison.get('b_accuracy', 'N/A')}",
            f"{comparison.get('b_f1', 'N/A')}",
            f"{comparison.get('b_precision', 'N/A')}",
            f"{comparison.get('b_recall', 'N/A')}"
        ]
    }

    basic_df = pd.DataFrame(basic_metrics)
    st.dataframe(basic_df, use_container_width=True, hide_index=True)

    # Erweiterte Metriken in Expander (für bessere Performance)
    with st.expander("🔍 Erweiterte Metriken anzeigen", expanded=False):
        extended_metrics = {
            'Metrik': ['ROC-AUC', 'MCC', 'False Positive Rate', 'False Negative Rate', 'Simulierter Profit'],
            f'{model_a_name}': [
                f"{comparison.get('a_roc_auc', 'N/A')}",
                f"{comparison.get('a_mcc', 'N/A')}",
                f"{comparison.get('a_fpr', 'N/A')}",
                f"{comparison.get('a_fnr', 'N/A')}",
                f"{comparison.get('a_simulated_profit_pct', 'N/A')}%"
            ],
            f'{model_b_name}': [
                f"{comparison.get('b_roc_auc', 'N/A')}",
                f"{comparison.get('b_mcc', 'N/A')}",
                f"{comparison.get('b_fpr', 'N/A')}",
                f"{comparison.get('b_fnr', 'N/A')}",
                f"{comparison.get('b_simulated_profit_pct', 'N/A')}%"
            ]
        }

        extended_df = pd.DataFrame(extended_metrics)
        st.dataframe(extended_df, use_container_width=True, hide_index=True)
    
        # Profit-Vergleich nur in Expander
        a_profit = comparison.get('a_simulated_profit_pct')
        b_profit = comparison.get('b_simulated_profit_pct')
        if a_profit is not None and b_profit is not None:
            profit_diff = a_profit - b_profit
            if profit_diff > 0:
                st.success(f"💰 **{model_a_name} profitabler:** {profit_diff:.2f}% mehr Profit")
            elif profit_diff < 0:
                st.success(f"💰 **{model_b_name} profitabler:** {abs(profit_diff):.2f}% mehr Profit")
        else:
                st.info("💰 **Gleich profitabel**")

    # Erklärungen
    with st.expander("📖 Metriken-Erklärungen", expanded=False):
        st.markdown("""
        - **Accuracy:** Anteil korrekter Vorhersagen (0-1, höher = besser)
        - **F1-Score:** Harmonisches Mittel aus Precision und Recall (0-1, höher = besser)
        - **Precision:** Von allen "Positiv"-Vorhersagen, wie viele waren richtig? (höher = weniger False Positives)
        - **Recall:** Von allen echten Positiven, wie viele wurden gefunden? (höher = weniger False Negatives)
        - **ROC-AUC:** Area Under ROC Curve (0-1, >0.7 = gut, >0.9 = sehr gut)
        - **MCC:** Matthews Correlation Coefficient (-1 bis +1, höher = besser)
        """)
    
    st.divider()
    
    # Erweiterte Metriken (falls verfügbar)
    extended_metrics = ['a_roc_auc', 'b_roc_auc', 'a_mcc', 'b_mcc', 'a_fpr', 'b_fpr', 'a_fnr', 'b_fnr', 'a_simulated_profit_pct', 'b_simulated_profit_pct']
    has_extended = any(comparison.get(key) is not None for key in extended_metrics)

    if has_extended:
        st.subheader("📈 Erweiterte Metriken")

        # Erweiterte Metriken Tabelle
        extended_data = {
            'Metrik': ['ROC-AUC', 'MCC', 'False Positive Rate', 'False Negative Rate', 'Simulierter Profit'],
            f'{model_a_name}': [
                f"{comparison.get('a_roc_auc', 'N/A')}",
                f"{comparison.get('a_mcc', 'N/A')}",
                f"{comparison.get('a_fpr', 'N/A')}",
                f"{comparison.get('a_fnr', 'N/A')}",
                f"{comparison.get('a_simulated_profit_pct', 'N/A')}%"
            ],
            f'{model_b_name}': [
                f"{comparison.get('b_roc_auc', 'N/A')}",
                f"{comparison.get('b_mcc', 'N/A')}",
                f"{comparison.get('b_fpr', 'N/A')}",
                f"{comparison.get('b_fnr', 'N/A')}",
                f"{comparison.get('b_simulated_profit_pct', 'N/A')}%"
            ]
        }

        extended_df = pd.DataFrame(extended_data)
        st.dataframe(extended_df, use_container_width=True, hide_index=True)
        
        # Profit-Vergleich (falls verfügbar)
        a_profit = comparison.get('a_simulated_profit_pct')
        b_profit = comparison.get('b_simulated_profit_pct')
        if a_profit is not None and b_profit is not None:
            st.divider()
            profit_diff = a_profit - b_profit
            if profit_diff > 0:
                st.success(f"💰 **Modell A profitabler:** {profit_diff:.2f}% mehr Profit")
            elif profit_diff < 0:
                st.success(f"💰 **Modell B profitabler:** {abs(profit_diff):.2f}% mehr Profit")
            else:
                st.info("💰 **Gleich profitabel**")
    
    st.divider()
    
    # Confusion Matrix (falls verfügbar) - in Expander für Performance
    if comparison.get('a_confusion_matrix') or comparison.get('b_confusion_matrix'):
        with st.expander("🔢 Confusion Matrix", expanded=False):
            # Vereinfachte Darstellung
            cm_data = {
                'Modell': [model_a_name, model_b_name],
                'TP (True Positive)': [
                    comparison.get('a_confusion_matrix', {}).get('tp', 'N/A'),
                    comparison.get('b_confusion_matrix', {}).get('tp', 'N/A')
                ],
                'TN (True Negative)': [
                    comparison.get('a_confusion_matrix', {}).get('tn', 'N/A'),
                    comparison.get('b_confusion_matrix', {}).get('tn', 'N/A')
                ],
                'FP (False Positive)': [
                    comparison.get('a_confusion_matrix', {}).get('fp', 'N/A'),
                    comparison.get('b_confusion_matrix', {}).get('fp', 'N/A')
                ],
                'FN (False Negative)': [
                    comparison.get('a_confusion_matrix', {}).get('fn', 'N/A'),
                    comparison.get('b_confusion_matrix', {}).get('fn', 'N/A')
                ]
                }

            cm_df = pd.DataFrame(cm_data)
            st.dataframe(cm_df, use_container_width=True, hide_index=True)
    
    st.divider()
    
    # Train vs. Test Vergleich (falls verfügbar) - in Expander für Performance
    if comparison.get('a_train_accuracy') or comparison.get('b_train_accuracy'):
        with st.expander("📊 Train vs. Test Vergleich", expanded=False):
            train_test_data = {
                'Modell': [model_a_name, model_b_name],
                'Train Accuracy': [
                    f"{comparison.get('a_train_accuracy', 'N/A')}",
                    f"{comparison.get('b_train_accuracy', 'N/A')}"
                ],
                'Test Accuracy': [
                    f"{comparison.get('a_accuracy', 'N/A')}",
                    f"{comparison.get('b_accuracy', 'N/A')}"
                ],
                'Degradation': [
                    f"{comparison.get('a_accuracy_degradation', 'N/A')}",
                    f"{comparison.get('b_accuracy_degradation', 'N/A')}"
                ]
            }

            train_test_df = pd.DataFrame(train_test_data)
            st.dataframe(train_test_df, use_container_width=True, hide_index=True)
        
        # Overfitting-Warnungen
            a_deg = comparison.get('a_accuracy_degradation')
            b_deg = comparison.get('b_accuracy_degradation')
            if a_deg and a_deg > 0.1:
                st.warning(f"⚠️ **{model_a_name}** Overfitting-Risiko (Degradation: {a_deg:.1%})")
            if b_deg and b_deg > 0.1:
                st.warning(f"⚠️ **{model_b_name}** Overfitting-Risiko (Degradation: {b_deg:.1%})")
    
    st.divider()
    
    # Test-Zeitraum mit Erklärung - in Expander für Performance
    with st.expander("📅 Test-Zeitraum", expanded=False):
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
    
    # Vollständige Details (nur bei Bedarf laden)
    with st.expander("📋 Vollständige Details (JSON)", expanded=False):
        # JSON erst beim Öffnen des Expanders laden für bessere Performance
        st.json(comparison)
    
    # Zurück-Button
    col1, col2 = st.columns([1, 3])
    with col1:
        if st.button("← Zurück zur Vergleichs-Übersicht", key="back_to_comparisons_details", use_container_width=True):
            st.session_state.pop('page', None)
            st.session_state.pop('comparison_details_id', None)


