"""
Test Details Page Module
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


def page_test_details():
    """Details-Seite für ein Test-Ergebnis"""
    # Zurück-Button
    col1, col2 = st.columns([1, 3])
    with col1:
        if st.button("← Zurück zu Test-Ergebnissen", key="back_to_test_results_details", use_container_width=True):
            st.session_state.pop('page', None)
            st.session_state.pop('test_details_id', None)
    
    st.divider()
    
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
    
    # Header mit wichtigsten Infos
    test_accuracy = test.get('accuracy', 0)
    test_f1 = test.get('f1_score', 0)
    status = test.get('status', 'N/A')

    # Header mit Status-Badge
    header_col1, header_col2 = st.columns([3, 1])
    with header_col1:
        st.title(f"📋 Test-Details: {model_name}")
    with header_col2:
        if status == "COMPLETED":
            st.success("✅ Abgeschlossen")
        elif status == "FAILED":
            st.error("❌ Fehlgeschlagen")
        elif status == "RUNNING":
            st.warning("🔄 Läuft...")
        else:
            st.info(f"📋 {status}")

    # Info-Box am Anfang
    st.info("""
    **📖 Anleitung:**
    Diese Seite zeigt alle Details und Metriken des Test-Ergebnisses. Nutze die Tabs unten für verschiedene Bereiche.
    """)

    # Tabs für bessere Organisation
    tab_overview, tab_performance, tab_details = st.tabs([
        "📊 Übersicht",
        "📈 Performance",
        "📋 Vollständige Daten"
    ])

    # TAB 1: Übersicht
    with tab_overview:
        # Basis-Informationen
        st.subheader("ℹ️ Basis-Informationen")
        info_col1, info_col2, info_col3, info_col4 = st.columns(4)
        with info_col1:
            st.markdown("**Test-ID**")
            st.write(f"#{test_id}")

        with info_col2:
            st.markdown("**Modell-ID**")
            st.write(f"#{model_id}")

        with info_col3:
            st.markdown("**Modell-Name**")
            st.write(model_name)

        with info_col4:
            st.markdown("**Status**")
            if status == "COMPLETED":
                st.success("✅ Abgeschlossen")
            elif status == "FAILED":
                st.error("❌ Fehlgeschlagen")
            elif status == "RUNNING":
                st.warning("🔄 Läuft...")
            else:
                st.info(status)

        # Erstellungszeit
        st.markdown("**Erstellt am**")
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

        # Schnell-Übersicht der wichtigsten Metriken
        st.subheader("📊 Wichtigste Metriken")
        metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)

        with metric_col1:
            accuracy = test.get('accuracy')
            if accuracy is not None:
                quality = "🟢 Sehr gut" if accuracy > 0.8 else "🟡 Gut" if accuracy > 0.7 else "🔴 Verbesserung nötig"
                st.metric("Accuracy", f"{accuracy:.4f}", help="Anteil korrekter Vorhersagen (0-1, höher = besser)")
                st.caption(quality)
            else:
                st.metric("Accuracy", "N/A")

        with metric_col2:
            f1 = test.get('f1_score')
            if f1 is not None:
                quality = "🟢 Sehr gut" if f1 > 0.8 else "🟡 Gut" if f1 > 0.7 else "🔴 Verbesserung nötig"
                st.metric("F1-Score", f"{f1:.4f}", help="Harmonisches Mittel aus Precision und Recall")
                st.caption(quality)
            else:
                st.metric("F1-Score", "N/A")

        with metric_col3:
            precision = test.get('precision')
            if precision is not None:
                quality = "🟢 Gut" if precision > 0.8 else "🟡 OK" if precision > 0.6 else "🔴 Verbesserung nötig"
                st.metric("Precision", f"{precision:.4f}", help="Von allen Positiv-Vorhersagen, wie viele waren richtig?")
                st.caption(quality)
            else:
                st.metric("Precision", "N/A")

        with metric_col4:
            recall = test.get('recall')
            if recall is not None:
                quality = "🟢 Gut" if recall > 0.8 else "🟡 OK" if recall > 0.6 else "🔴 Verbesserung nötig"
                st.metric("Recall", f"{recall:.4f}", help="Von allen echten Positiven, wie viele wurden gefunden?")
                st.caption(quality)
            else:
                st.metric("Recall", "N/A")

    # TAB 2: Performance - Detaillierte Analyse
    with tab_performance:
        st.subheader("📈 Detaillierte Performance-Analyse")

        # Erklärung der Metriken
        with st.expander("ℹ️ Was bedeuten diese Metriken?", expanded=False):
            st.markdown("""
            ### Grundlegende Metriken:
            - **Accuracy:** Anteil korrekter Vorhersagen auf den Test-Daten (0-1). Beispiel: 0.85 = 85% der Vorhersagen sind richtig.
            - **F1-Score:** Harmonisches Mittel aus Precision und Recall (0-1). Besonders gut für unausgewogene Daten.
            - **Precision:** Von allen "Positiv"-Vorhersagen, wie viele waren wirklich positiv? (0-1, höher = weniger False Positives)
            - **Recall:** Von allen echten Positiven, wie viele hat das Modell gefunden? (0-1, höher = weniger False Negatives)

            ### Erweiterte Metriken:
            - **ROC-AUC:** Fähigkeit zur Unterscheidung zwischen Klassen (0.5 = zufällig, 1.0 = perfekt)
            - **MCC:** Matthews Correlation Coefficient - Ausgewogen für alle vier Confusion-Matrix Werte
            - **FPR/FNR:** False Positive/False Negative Rate - Fehlerquoten
            """)

        st.divider()

        # Standard-Metriken
        st.subheader("📊 Standard-Metriken")

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

        # Erweiterte Metriken
        if test.get('roc_auc') or test.get('mcc') or test.get('fpr') or test.get('fnr'):
            st.subheader("📈 Erweiterte Metriken")

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                roc_auc = test.get('roc_auc')
                if roc_auc:
                    quality = "🟢 Sehr gut" if roc_auc > 0.9 else "🟡 Gut" if roc_auc > 0.7 else "🔴 Mäßig" if roc_auc > 0.5 else "🔴 Schlecht"
                    st.metric("ROC-AUC", f"{roc_auc:.4f}", help="Area Under ROC Curve (0-1). Fähigkeit zur Klassenunterscheidung")
                    st.caption(quality)
                else:
                    st.metric("ROC-AUC", "N/A")

            with col2:
                mcc = test.get('mcc')
                if mcc is not None:
                    quality = "🟢 Sehr gut" if mcc > 0.5 else "🟡 Gut" if mcc > 0.3 else "🟠 Mäßig" if mcc > 0 else "🔴 Schlecht"
                    st.metric("MCC", f"{mcc:.4f}", help="Matthews Correlation Coefficient (-1 bis +1)")
                    st.caption(quality)
                else:
                    st.metric("MCC", "N/A")

            with col3:
                fpr = test.get('fpr')
                if fpr is not None:
                    quality = "🟢 Gut" if fpr < 0.1 else "🟡 Mäßig" if fpr < 0.3 else "🔴 Schlecht"
                    st.metric("False Positive Rate", f"{fpr:.4f}", help="Falsch-Positiv-Rate (niedriger = besser)")
                    st.caption(quality)
                else:
                    st.metric("FPR", "N/A")

            with col4:
                fnr = test.get('fnr')
                if fnr is not None:
                    quality = "🟢 Gut" if fnr < 0.1 else "🟡 Mäßig" if fnr < 0.3 else "🔴 Schlecht"
                    st.metric("False Negative Rate", f"{fnr:.4f}", help="Falsch-Negativ-Rate (niedriger = besser)")
                    st.caption(quality)
                else:
                    st.metric("FNR", "N/A")
        
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
    
    # Zurück-Button am Ende
    st.divider()
    # TAB 3: Vollständige Daten
    with tab_details:
        st.subheader("📋 Vollständige Test-Daten")
        st.caption("Alle verfügbaren Daten des Test-Ergebnisses im JSON-Format")
        st.json(test)

    # Zurück-Button außerhalb der Tabs
    st.divider()
    col1, col2 = st.columns([1, 3])
    with col1:
        if st.button("← Zurück zur Test-Übersicht", key="back_to_test_results_details_bottom", use_container_width=True):
            st.session_state.pop('page', None)
            st.session_state.pop('test_details_id', None)


