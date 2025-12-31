"""
Jobs Page Module
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


