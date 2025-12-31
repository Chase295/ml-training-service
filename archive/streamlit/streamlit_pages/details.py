"""
Details Page Module
Modell-Details Seite
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
    API_BASE_URL
)

def page_details():
    """Modell-Details - Überarbeitete Version mit besserer Übersichtlichkeit"""
    # Zurück-Button
    col1, col2 = st.columns([1, 3])
    with col1:
        if st.button("← Zurück zur Übersicht", key="back_to_overview_top", use_container_width=True):
            st.session_state.pop('page', None)
            st.session_state.pop('details_model_id', None)

    st.divider()

    model_id = st.session_state.get('details_model_id')
    if not model_id:
        st.warning("⚠️ Kein Modell ausgewählt")
        return
    
    model = api_get(f"/api/models/{model_id}")
    if not model:
        st.error("❌ Modell nicht gefunden")
        return
    
    # Header mit wichtigsten Infos
    model_name = model.get('name', 'Unbenannt')
    model_type = model.get('model_type', 'N/A')
    type_emoji = "🌲" if model_type == "random_forest" else "🚀" if model_type == "xgboost" else "🤖"
    status = model.get('status', 'N/A')
    
    # Header mit Status-Badge
    header_col1, header_col2 = st.columns([3, 1])
    with header_col1:
        st.title(f"{type_emoji} {model_name}")
    with header_col2:
        if status == "READY":
            st.success("✅ READY")
        elif status == "TRAINING":
            st.info("🔄 TRAINING")
        else:
            st.error(f"❌ {status}")
    
    # Quick Info Cards
    st.subheader("📊 Quick Overview")
    
    # Erklärung zu den Quick Overview Metriken
    with st.expander("ℹ️ Was bedeuten diese Metriken?", expanded=False):
        st.markdown("""
        **Accuracy (Genauigkeit):**
        - Zeigt an, wie viele Vorhersagen insgesamt korrekt waren
        - Formel: (TP + TN) / (TP + TN + FP + FN)
        - **Beispiel:** 85% Accuracy bedeutet: Von 100 Vorhersagen waren 85 korrekt
        - ⚠️ **Achtung:** Bei unausgewogenen Daten kann Accuracy irreführend sein!
        
        **F1-Score:**
        - Harmonisches Mittel aus Precision und Recall
        - Formel: 2 × (Precision × Recall) / (Precision + Recall)
        - **Bedeutung:** Gibt einen ausgewogenen Wert für beide Metriken
        - **Ideal:** Nahe bei 1.0 (100%)
        - **Praktisch:** >0.7 ist gut, >0.8 ist sehr gut
        
        **Precision (Präzision):**
        - Von allen "Positiv"-Vorhersagen, wie viele waren wirklich positiv?
        - Formel: TP / (TP + FP)
        - **Beispiel:** 90% Precision = Von 100 "Pump"-Vorhersagen waren 90 wirklich Pumps
        - **Wichtig für:** Minimierung von Fehlkäufen (False Positives)
        
        **Recall (Sensitivität):**
        - Von allen echten Pumps, wie viele hat das Modell gefunden?
        - Formel: TP / (TP + FN)
        - **Beispiel:** 80% Recall = Von 100 echten Pumps wurden 80 erkannt
        - **Wichtig für:** Keine echten Pumps verpassen (False Negatives)
        
        **ROC-AUC (Receiver Operating Characteristic - Area Under Curve):**
        - Misst die Fähigkeit, zwischen Positiv und Negativ zu unterscheiden
        - **Werte:** 0.5 = zufällig, 0.7-0.8 = akzeptabel, 0.8-0.9 = gut, >0.9 = sehr gut
        - **Bedeutung:** Je höher, desto besser kann das Modell unterscheiden
        """)
    
    quick_col1, quick_col2, quick_col3, quick_col4, quick_col5 = st.columns(5)
    
    with quick_col1:
        accuracy = model.get('training_accuracy')
        if accuracy:
            st.metric("Accuracy", f"{accuracy:.2%}", help="Anteil korrekter Vorhersagen")
        else:
            st.metric("Accuracy", "N/A")
    
    with quick_col2:
        f1 = model.get('training_f1')
        if f1:
            st.metric("F1-Score", f"{f1:.2%}", help="Harmonisches Mittel aus Precision und Recall")
        else:
            st.metric("F1-Score", "N/A")
    
    with quick_col3:
        precision = model.get('training_precision')
        if precision:
            st.metric("Precision", f"{precision:.2%}", help="Anteil korrekter Positiv-Vorhersagen")
        else:
            st.metric("Precision", "N/A")
    
    with quick_col4:
        recall = model.get('training_recall')
        if recall:
            st.metric("Recall", f"{recall:.2%}", help="Anteil gefundener Positiver")
        else:
            st.metric("Recall", "N/A")
    
    with quick_col5:
        roc_auc = model.get('roc_auc')
        if roc_auc:
            quality = "🟢" if roc_auc > 0.9 else "🟡" if roc_auc > 0.7 else "🔴"
            st.metric("ROC-AUC", f"{quality} {roc_auc:.3f}", help="Area Under ROC Curve")
        else:
            st.metric("ROC-AUC", "N/A")
    
    st.divider()
    
    # Tabs für bessere Organisation
    tab_overview, tab_performance, tab_config, tab_features, tab_details = st.tabs([
        "📊 Übersicht",
        "📈 Performance",
        "⚙️ Konfiguration",
        "🎯 Features",
        "📋 Details"
    ])
    
    # TAB 1: Übersicht
    with tab_overview:
        # Basis-Informationen
        st.subheader("ℹ️ Basis-Informationen")
        info_col1, info_col2, info_col3, info_col4 = st.columns(4)
        
        with info_col1:
            st.markdown("**Modell-Typ**")
            st.write(f"{type_emoji} {model_type}")
        
        with info_col2:
            st.markdown("**Modell-ID**")
            st.code(f"#{model_id}", language=None)
        
        with info_col3:
            st.markdown("**Erstellt am**")
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
        
        with info_col4:
            st.markdown("**Status**")
            if status == "READY":
                st.success("✅ READY")
            elif status == "TRAINING":
                st.info("🔄 TRAINING")
            else:
                st.error(f"❌ {status}")
    
    # Beschreibung
    description = model.get('description')
    if description:
        st.subheader("📝 Beschreibung")
        st.info(description)
    
        # Confusion Matrix (kompakt)
        confusion_matrix = model.get('confusion_matrix')
        if confusion_matrix:
            st.subheader("🔢 Confusion Matrix")
            
        # Ausführliche Erklärung
            with st.expander("ℹ️ Was ist eine Confusion Matrix?", expanded=False):
                st.markdown("""
                Die **Confusion Matrix** zeigt, wie gut das Modell Vorhersagen macht:
                
                **✅ True Positive (TP):**
                - Das Modell sagt "Pump" vorher und es ist wirklich ein Pump
                - **Gut!** → Du kaufst und es steigt tatsächlich
                - **Bedeutung:** Erfolgreiche Vorhersagen
                
                **✅ True Negative (TN):**
                - Das Modell sagt "Kein Pump" vorher und es ist wirklich kein Pump
                - **Gut!** → Du kaufst nicht und verpasst nichts
                - **Bedeutung:** Korrekte Ablehnungen
                
                **❌ False Positive (FP):**
                - Das Modell sagt "Pump" vorher, aber es ist KEIN Pump
                - **Schlecht!** → Du kaufst, aber der Preis steigt nicht
                - **Bedeutung:** Fehlkäufe, Geldverlust
                - **Ziel:** So niedrig wie möglich halten
                
                **❌ False Negative (FN):**
                - Das Modell sagt "Kein Pump" vorher, aber es IST ein Pump
                - **Schlecht!** → Du verpasst eine echte Chance
                - **Bedeutung:** Verpasste Gewinne
                - **Ziel:** So niedrig wie möglich halten
                
                **💡 Praktische Interpretation:**
                - **Hohe TP + niedrige FP** = Viele richtige Pump-Erkennungen, wenige Fehlkäufe
                - **Hohe TN + niedrige FN** = Viele richtige Ablehnungen, wenige verpasste Chancen
                - **Ideal:** Hohe TP und TN, niedrige FP und FN
                """)
            
            cm_col1, cm_col2, cm_col3, cm_col4 = st.columns(4)
            
            with cm_col1:
                tp = confusion_matrix.get('tp', 0)
                st.metric("✅ TP", tp, help="True Positive: Korrekt als Positiv erkannt")
                st.caption("Erfolgreiche Pump-Erkennungen")
            
            with cm_col2:
                tn = confusion_matrix.get('tn', 0)
                st.metric("✅ TN", tn, help="True Negative: Korrekt als Negativ erkannt")
                st.caption("Korrekte Ablehnungen")
            
            with cm_col3:
                fp = confusion_matrix.get('fp', 0)
                st.metric("❌ FP", fp, delta=f"-{fp}", delta_color="inverse", help="False Positive: Falsch als Positiv erkannt")
                st.caption("Fehlkäufe (Geldverlust)")
            
            with cm_col4:
                fn = confusion_matrix.get('fn', 0)
                st.metric("❌ FN", fn, delta=f"-{fn}", delta_color="inverse", help="False Negative: Falsch als Negativ erkannt")
                st.caption("Verpasste Chancen")
            
            # Visualisierung als Tabelle
            st.markdown("**📊 Matrix-Darstellung:**")
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
                accuracy_calc = (tp + tn) / total
                precision_calc = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall_calc = tp / (tp + fn) if (tp + fn) > 0 else 0
                
                st.info(f"""
                **📈 Interpretation:**
                - **Gesamt Vorhersagen:** {total}
                - **Korrekt:** {tp + tn} ({accuracy_calc:.1%})
                - **Fehler:** {fp + fn} ({(fp + fn)/total:.1%})
                - **Precision (aus Matrix):** {precision_calc:.1%} - Von {tp + fp} "Pump"-Vorhersagen waren {tp} richtig
                - **Recall (aus Matrix):** {recall_calc:.1%} - Von {tp + fn} echten Pumps wurden {tp} erkannt
                """)
        
        # Profit-Simulation
        simulated_profit = model.get('simulated_profit_pct')
        if simulated_profit is not None:
            st.subheader("💰 Profit-Simulation")
            
            # Erklärung zur Profit-Simulation
            with st.expander("ℹ️ Wie funktioniert die Profit-Simulation?", expanded=False):
                st.markdown("""
                Die **Profit-Simulation** berechnet einen theoretischen Gewinn/Verlust basierend auf den Vorhersagen:
                
                **📊 Berechnungsformel:**
                - **+1% Profit** für jeden True Positive (TP)
                  - Du kaufst basierend auf der Vorhersage → Preis steigt tatsächlich → +1% Gewinn
                
                - **-0.5% Verlust** für jeden False Positive (FP)
                  - Du kaufst basierend auf der Vorhersage → Preis steigt NICHT → -0.5% Verlust
                
                - **0%** für True Negative (TN) und False Negative (FN)
                  - TN: Du kaufst nicht → kein Gewinn, aber auch kein Verlust
                  - FN: Du verpasst eine Chance → kein Gewinn, aber auch kein Verlust
                
                **💡 Beispiel:**
                - 100 TP → +100% Gewinn
                - 20 FP → -10% Verlust
                - **Gesamt:** +90% Profit
                
                **⚠️ Wichtig:**
                - Dies ist eine **vereinfachte Simulation**
                - Echte Gewinne hängen von vielen Faktoren ab (Timing, Volatilität, etc.)
                - Die Simulation zeigt die **relative Performance** verschiedener Modelle
                """)
            
            profit_col1, profit_col2 = st.columns([1, 2])
            with profit_col1:
                profit_quality = "🟢 Sehr profitabel" if simulated_profit > 5 else "🟡 Profitabel" if simulated_profit > 0 else "🔴 Verlust"
                st.metric("💰 Simulierter Profit", f"{simulated_profit:.2f}%", help="Simulierter Profit basierend auf TP/FP")
                st.caption(profit_quality)
            
            with profit_col2:
                tp = confusion_matrix.get('tp', 0) if confusion_matrix else 0
                fp = confusion_matrix.get('fp', 0) if confusion_matrix else 0
                profit_from_tp = tp * 1.0
                loss_from_fp = fp * 0.5
                st.info(f"""
                **📊 Detaillierte Berechnung:**
                - {tp} TP × 1% = +{profit_from_tp:.2f}%
                - {fp} FP × 0.5% = -{loss_from_fp:.2f}%
                - **Gesamt:** {simulated_profit:.2f}%
                """)
    
    # TAB 2: Performance
    with tab_performance:
        st.subheader("📊 Standard-Metriken")
        
        # Erklärung zu Standard-Metriken
        with st.expander("ℹ️ Detaillierte Erklärung der Standard-Metriken", expanded=False):
            st.markdown("""
            ### Accuracy (Genauigkeit)
            **Was es misst:** Anteil aller korrekten Vorhersagen (sowohl positive als auch negative)
            
            **Formel:** (TP + TN) / (TP + TN + FP + FN)
            
            **Beispiel:**
            - 100 Vorhersagen insgesamt
            - 85 waren korrekt (TP + TN)
            - **Accuracy = 85%**
            
            **⚠️ Wichtig:** Bei unausgewogenen Daten (z.B. 90% negative, 10% positive) kann Accuracy irreführend sein!
            Ein Modell, das immer "negativ" sagt, hätte 90% Accuracy, ist aber nutzlos.
            
            ---
            
            ### Precision (Präzision)
            **Was es misst:** Von allen "Pump"-Vorhersagen, wie viele waren wirklich Pumps?
            
            **Formel:** TP / (TP + FP)
            
            **Beispiel:**
            - Modell sagt 100x "Pump" vorher
            - 90 davon waren wirklich Pumps (TP)
            - 10 waren keine Pumps (FP)
            - **Precision = 90%**
            
            **Praktische Bedeutung:** 
            - Hohe Precision = Wenige Fehlkäufe
            - **Wichtig für:** Minimierung von Geldverlusten durch falsche Käufe
            
            ---
            
            ### Recall (Sensitivität / Trefferquote)
            **Was es misst:** Von allen echten Pumps, wie viele hat das Modell gefunden?
            
            **Formel:** TP / (TP + FN)
            
            **Beispiel:**
            - Es gab 100 echte Pumps
            - Modell hat 80 davon erkannt (TP)
            - 20 wurden verpasst (FN)
            - **Recall = 80%**
            
            **Praktische Bedeutung:**
            - Hoher Recall = Wenige verpasste Chancen
            - **Wichtig für:** Maximierung von Gewinnmöglichkeiten
            
            ---
            
            ### F1-Score
            **Was es misst:** Ausgewogenes Maß zwischen Precision und Recall
            
            **Formel:** 2 × (Precision × Recall) / (Precision + Recall)
            
            **Beispiel:**
            - Precision = 90%, Recall = 80%
            - **F1-Score = 2 × (0.9 × 0.8) / (0.9 + 0.8) = 0.847 (84.7%)**
            
            **Praktische Bedeutung:**
            - Gibt einen einzigen Wert, der beide Metriken berücksichtigt
            - **Ideal:** Nahe bei 1.0 (100%)
            - **Gut:** >0.7, **Sehr gut:** >0.8
            
            **💡 Trade-off:**
            - Hohe Precision → Niedrige Recall (wenige Fehlkäufe, aber viele verpasste Chancen)
            - Hohe Recall → Niedrige Precision (viele erkannte Pumps, aber auch viele Fehlkäufe)
            - F1-Score hilft, den optimalen Balance-Punkt zu finden
            """)
            
            perf_col1, perf_col2, perf_col3, perf_col4 = st.columns(4)
            with perf_col1:
                accuracy = model.get('training_accuracy')
                if accuracy:
                    quality = "🟢 Sehr gut" if accuracy > 0.9 else "🟡 Gut" if accuracy > 0.7 else "🔴 Verbesserung nötig"
                    st.metric("Accuracy", f"{accuracy:.4f}", help="Anteil korrekter Vorhersagen (0-1, höher = besser)")
                    st.caption(quality)
                else:
                    st.write("Accuracy: N/A")
            
            with perf_col2:
                f1 = model.get('training_f1')
                if f1:
                    quality = "🟢 Sehr gut" if f1 > 0.8 else "🟡 Gut" if f1 > 0.7 else "🔴 Verbesserung nötig"
                    st.metric("F1-Score", f"{f1:.4f}", help="Harmonisches Mittel aus Precision und Recall (0-1, höher = besser)")
                    st.caption(quality)
                else:
                    st.write("F1-Score: N/A")
            
            with perf_col3:
                precision = model.get('training_precision')
                if precision:
                    quality = "🟢 Sehr gut" if precision > 0.8 else "🟡 Gut" if precision > 0.7 else "🔴 Verbesserung nötig"
                    st.metric("Precision", f"{precision:.4f}", help="Von allen 'Positiv'-Vorhersagen, wie viele waren wirklich positiv?")
                    st.caption(quality)
                else:
                    st.write("Precision: N/A")
            
            with perf_col4:
                recall = model.get('training_recall')
                if recall:
                    quality = "🟢 Sehr gut" if recall > 0.8 else "🟡 Gut" if recall > 0.7 else "🔴 Verbesserung nötig"
                    st.metric("Recall", f"{recall:.4f}", help="Von allen echten Positiven, wie viele hat das Modell gefunden?")
                    st.caption(quality)
                else:
                    st.write("Recall: N/A")
    
    st.divider()
    
    st.subheader("📈 Erweiterte Metriken")
    
    # Erklärung zu erweiterten Metriken
    with st.expander("ℹ️ Detaillierte Erklärung der erweiterten Metriken", expanded=False):
        st.markdown("""
            ### ROC-AUC (Receiver Operating Characteristic - Area Under Curve)
            **Was es misst:** Die Fähigkeit des Modells, zwischen positiven und negativen Fällen zu unterscheiden
            
            **Werte-Bereich:** 0.0 bis 1.0
            
            **Interpretation:**
            - **0.5** = Zufällig (wie Münzwurf) - Modell ist nutzlos
            - **0.7-0.8** = Akzeptabel - Modell kann unterscheiden
            - **0.8-0.9** = Gut - Gute Unterscheidungsfähigkeit
            - **>0.9** = Sehr gut - Exzellente Unterscheidungsfähigkeit
            - **1.0** = Perfekt (in der Praxis nicht erreichbar)
            
            **Praktische Bedeutung:**
            - Höherer ROC-AUC = Modell kann besser zwischen Pump und Nicht-Pump unterscheiden
            - **Wichtig für:** Vergleich verschiedener Modelle
            
            ---
            
            ### MCC (Matthews Correlation Coefficient)
            **Was es misst:** Ausgewogene Metrik, die alle vier Werte der Confusion Matrix berücksichtigt
            
            **Formel:** (TP × TN - FP × FN) / √((TP + FP) × (TP + FN) × (TN + FP) × (TN + FN))
            
            **Werte-Bereich:** -1.0 bis +1.0
            
            **Interpretation:**
            - **+1.0** = Perfekte Vorhersage
            - **0.0** = Zufällig
            - **-1.0** = Perfekte umgekehrte Vorhersage (Modell ist komplett falsch)
            
            **Vorteil:** Funktioniert auch bei unausgewogenen Daten besser als Accuracy
            
            ---
            
            ### False Positive Rate (FPR)
            **Was es misst:** Anteil der negativen Fälle, die fälschlicherweise als positiv klassifiziert wurden
            
            **Formel:** FP / (FP + TN)
            
            **Beispiel:**
            - 100 echte "Nicht-Pumps"
            - 10 wurden fälschlicherweise als "Pump" erkannt (FP)
            - **FPR = 10%**
            
            **Praktische Bedeutung:**
            - Niedrige FPR = Wenige Fehlkäufe
            - **Ziel:** <10% ist gut, <5% ist sehr gut
            
            ---
            
            ### False Negative Rate (FNR)
            **Was es misst:** Anteil der positiven Fälle, die fälschlicherweise als negativ klassifiziert wurden
            
            **Formel:** FN / (FN + TP)
            
            **Beispiel:**
            - 100 echte Pumps
            - 15 wurden verpasst (FN)
            - **FNR = 15%**
            
            **Praktische Bedeutung:**
            - Niedrige FNR = Wenige verpasste Chancen
            - **Ziel:** <10% ist gut, <5% ist sehr gut
            """)
            
        adv_col1, adv_col2, adv_col3, adv_col4 = st.columns(4)
        with adv_col1:
            roc_auc = model.get('roc_auc')
            if roc_auc:
                quality = "🟢 Sehr gut" if roc_auc > 0.9 else "🟡 Gut" if roc_auc > 0.7 else "🔴 Verbesserung nötig"
                st.metric("ROC-AUC", f"{roc_auc:.4f}", help=f"Area Under ROC Curve. {quality} (>0.7 = gut)")
                st.caption(quality)
            else:
                st.write("ROC-AUC: N/A")
        
        with adv_col2:
            mcc = model.get('mcc')
            if mcc:
                quality = "🟢 Sehr gut" if mcc > 0.5 else "🟡 Gut" if mcc > 0.3 else "🔴 Verbesserung nötig"
                st.metric("MCC", f"{mcc:.4f}", help=f"Matthews Correlation Coefficient. {quality}")
                st.caption(quality)
            else:
                st.write("MCC: N/A")
        
        with adv_col3:
            fpr = model.get('fpr')
            if fpr is not None:
                quality = "🟢 Gut" if fpr < 0.1 else "🟡 Mäßig" if fpr < 0.3 else "🔴 Verbesserung nötig"
                st.metric("False Positive Rate", f"{fpr:.4f}", help=f"Falsch-Positiv-Rate. {quality} (niedriger = besser)")
                st.caption(quality)
            else:
                st.write("FPR: N/A")
        
        with adv_col4:
            fnr = model.get('fnr')
            if fnr is not None:
                quality = "🟢 Gut" if fnr < 0.1 else "🟡 Mäßig" if fnr < 0.3 else "🔴 Verbesserung nötig"
                st.metric("False Negative Rate", f"{fnr:.4f}", help=f"Falsch-Negativ-Rate. {quality} (niedriger = besser)")
                st.caption(quality)
            else:
                st.write("FNR: N/A")

    # Cross-Validation - außerhalb des erweiterten Metriken Expanders
    cv_scores = model.get('cv_scores')
    if cv_scores:
        st.divider()
        st.subheader("🔀 Cross-Validation")

        # Erklärung zu Cross-Validation
        st.info("**ℹ️ Was ist Cross-Validation?** Cross-Validation testet die Generalisierungsfähigkeit eines Modells durch mehrfaches Training mit verschiedenen Daten-Teilen.")
        with st.expander("📖 Detaillierte Erklärung zu Cross-Validation", expanded=False):
                st.markdown("""
                **Cross-Validation (CV)** ist eine Methode, um die Generalisierungsfähigkeit eines Modells zu testen.
                
                **Wie es funktioniert:**
                1. Die Trainingsdaten werden in mehrere "Folds" (z.B. 5) aufgeteilt
                2. Das Modell wird 5x trainiert:
                   - Jedes Mal wird ein anderer Fold als Test-Set verwendet
                   - Die anderen 4 Folds werden zum Training verwendet
                3. Für jeden Fold wird die Performance gemessen
                4. Am Ende erhält man 5 verschiedene Scores
                
                **Vorteile:**
                - **Robustheit:** Zeigt, ob das Modell konsistent gut performt
                - **Overfitting-Erkennung:** Große Unterschiede zwischen Train- und CV-Score deuten auf Overfitting hin
                - **Bessere Schätzung:** Der Durchschnittswert ist eine bessere Schätzung der echten Performance
                
                **Interpretation:**
                - **Durchschnittlicher Score:** Durchschnittliche Performance über alle Folds
                  - Sollte nahe am Trainings-Score sein
                
                - **Standardabweichung:** Wie stark variieren die Scores?
                  - **Niedrig (<0.05):** Modell ist stabil und konsistent
                  - **Hoch (>0.1):** Modell ist instabil, Performance variiert stark
                
                - **Overfitting-Gap:** Unterschied zwischen Train- und CV-Score
                  - **<0.1:** OK, Modell generalisiert gut
                  - **>0.1:** ⚠️ Overfitting-Risiko - Modell lernt zu spezifisch
                
                **💡 Praktische Bedeutung:**
                - Ein Modell mit hohem CV-Score und niedriger Standardabweichung ist zuverlässiger
                - Ein großes Overfitting-Gap bedeutet: Modell funktioniert gut auf Trainingsdaten, aber schlecht auf neuen Daten
                """)

        if isinstance(cv_scores, dict):
            cv_col1, cv_col2, cv_col3 = st.columns(3)
            with cv_col1:
                mean_score = cv_scores.get('mean_score')
                if mean_score is not None:
                    quality = "🟢 Sehr gut" if mean_score > 0.8 else "🟡 Gut" if mean_score > 0.7 else "🔴 Verbesserung nötig"
                    st.metric("Durchschnittlicher Score", f"{mean_score:.4f}", help="Durchschnittliche Performance über alle CV-Splits")
                    st.caption(quality)

            with cv_col2:
                std_score = cv_scores.get('std_score')
                if std_score is not None:
                    quality = "🟢 Stabil" if std_score < 0.05 else "🟡 Mäßig" if std_score < 0.1 else "🔴 Instabil"
                    st.metric("Standardabweichung", f"{std_score:.4f}", help="Wie stark variiert die Performance? (niedriger = stabiler)")
                    st.caption(quality)

            with cv_col3:
                cv_overfitting = model.get('cv_overfitting_gap')
                if cv_overfitting is not None:
                    quality = "🟢 OK" if cv_overfitting < 0.1 else "🟡 ⚠️ Overfitting-Risiko"
                    st.metric("Overfitting-Gap", f"{cv_overfitting:.4f}", help=f"Unterschied zwischen Train- und CV-Score. {quality}")
                    st.caption(quality)

            # Einzelne Scores
            individual_scores = cv_scores.get('scores', [])
            if individual_scores:
                with st.expander("📊 Einzelne CV-Scores anzeigen"):
                    st.write(f"**Scores pro Fold:** {[f'{s:.4f}' for s in individual_scores]}")
                    st.caption(f"Anzahl Folds: {len(individual_scores)}")
    
    # TAB 3: Konfiguration
    with tab_config:
        # Erklärung zur Konfiguration
        with st.expander("ℹ️ Was bedeuten diese Konfigurations-Parameter?", expanded=False):
            st.markdown("""
            ### Training-Zeitraum
            **Was es ist:** Der Zeitraum, aus dem die Trainingsdaten stammen
            
            **Bedeutung:**
            - **Längerer Zeitraum** = Mehr Daten, aber möglicherweise veraltete Muster
            - **Kürzerer Zeitraum** = Aktuellere Daten, aber weniger Beispiele
            - **Empfehlung:** 1-4 Wochen für aktuelle Marktbedingungen
            
            ---
            
            ### Ziel-Konfiguration
            **Was es ist:** Definiert, was das Modell vorhersagen soll
            
            **Zeitbasierte Vorhersage:**
            - **Beispiel:** "Preis wird in 5 Minuten um mindestens 3% steigen"
            - **Variable:** Welche Variable wird beobachtet (z.B. `price_close`)
            - **Zeitraum:** Wie viele Minuten in die Zukunft?
            - **Min. Änderung:** Mindest-Prozent-Änderung für "Positiv"
            - **Richtung:** Steigt (up) oder fällt (down)?
            
            **Klassische Vorhersage:**
            - **Beispiel:** "price_close > 0.001"
            - **Variable:** Welche Variable wird geprüft
            - **Operator:** Vergleichsoperator (>, <, >=, <=, ==)
            - **Wert:** Vergleichswert
            
            ---
            
            ### Feature-Engineering
            **Was es ist:** Automatische Erstellung zusätzlicher Features aus Basis-Daten
            
            **Beispiele:**
            - `price_change_5` - Preisänderung in den letzten 5 Minuten
            - `volume_ratio_10` - Verhältnis von Buy- zu Sell-Volumen (10 Min)
            - `ath_distance_pct` - Abstand zum All-Time High
            - `whale_activity_5` - Whale-Aktivität in den letzten 5 Minuten
            
            **Vorteile:**
            - Modell kann komplexere Muster erkennen
            - Bessere Performance bei Pump-Erkennung
            - **Nachteil:** Längere Trainingszeit
            
            ---
            
            ### SMOTE (Synthetic Minority Oversampling Technique)
            **Was es ist:** Technik zur Behandlung unausgewogener Daten
            
            **Problem:** Wenn es viel mehr "Nicht-Pumps" als "Pumps" gibt, lernt das Modell hauptsächlich "Nicht-Pump" zu sagen
            
            **Lösung:** SMOTE erstellt künstliche "Pump"-Beispiele, um das Verhältnis auszugleichen
            
            **Empfehlung:** Aktiviert lassen, wenn Daten unausgewogen sind
            
            ---
            
            ### Cross-Validation Splits
            **Was es ist:** Anzahl der Folds für Cross-Validation
            
            **Typische Werte:** 5 oder 10
            - **5 Folds:** Schneller, weniger robust
            - **10 Folds:** Langsamer, robuster
            
            **Empfehlung:** 5 für schnelle Tests, 10 für finale Modelle
            """)
    
        config_col1, config_col2 = st.columns(2)
        
        with config_col1:
            st.subheader("📅 Training-Zeitraum")
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
            
            st.subheader("🎯 Ziel-Konfiguration")
            target_var = model.get('target_variable', 'N/A')
            target_operator = model.get('target_operator')
            target_value = model.get('target_value')
            
            # Zeitbasierte Vorhersage?
            future_minutes = model.get('future_minutes')
            price_change = model.get('price_change_percent')
            direction = model.get('target_direction')
            
            if future_minutes and price_change:
                st.write(f"**Typ:** ⏰ Zeitbasierte Vorhersage")
                st.write(f"**Variable:** `{target_var}`")
                st.write(f"**Zeitraum:** {future_minutes} Minuten")
                st.write(f"**Min. Änderung:** {price_change}%")
                direction_text = "📈 Steigt" if direction == "up" else "📉 Fällt" if direction == "down" else "N/A"
                st.write(f"**Richtung:** {direction_text}")
            else:
                st.write(f"**Typ:** 🎯 Klassische Bedingung")
                st.write(f"**Variable:** `{target_var}`")
                if target_operator and target_value is not None:
                    st.write(f"**Bedingung:** `{target_var} {target_operator} {target_value}`")
                else:
                    st.write("**Bedingung:** Nicht konfiguriert")
            
            st.subheader("📊 Daten-Konfiguration")
            features_list = model.get('features', [])
            if features_list:
                st.write(f"**Features:** {len(features_list)} ausgewählt")
                with st.expander("📋 Alle Features anzeigen"):
                    for feat in features_list:
                        st.write(f"- `{feat}`")
            else:
                st.write("Keine Features verfügbar")
            
            phases_list = model.get('phases')
            if phases_list:
                st.write(f"**Phasen:** {len(phases_list)} Phase(n)")
                with st.expander("📋 Phasen anzeigen"):
                    for phase_id in phases_list:
                        st.write(f"- Phase {phase_id}")
            else:
                st.write("**Phasen:** Alle Phasen verwendet")
        
        with config_col2:
            st.subheader("⚙️ Modell-Parameter")
            params = model.get('params', {})
            if isinstance(params, str):
                import json
                try:
                    params = json.loads(params)
                except:
                    params = {}
            
            if params:
                # Feature-Engineering
                if params.get('use_engineered_features'):
                    st.success("🔧 Feature-Engineering: ✅ Aktiviert")
                    windows = params.get('feature_engineering_windows', [])
                    if windows:
                        st.caption(f"   Fenster: {windows}")
                else:
                    st.info("🔧 Feature-Engineering: ❌ Deaktiviert")
                
                # Zeitbasierte Vorhersage
                if params.get('_time_based', {}).get('enabled'):
                    time_based_params = params.get('_time_based', {})
                    tb_future_minutes = time_based_params.get('future_minutes') or future_minutes
                    tb_min_percent = time_based_params.get('min_percent_change') or price_change
                    tb_direction = time_based_params.get('direction') or direction
                    if tb_future_minutes and tb_min_percent:
                        direction_text = "steigt" if tb_direction == "up" else "fällt" if tb_direction == "down" else ""
                        st.success(f"⏰ Zeitbasierte Vorhersage: ✅ ({tb_future_minutes}min, {tb_min_percent}% {direction_text})")
                    else:
                        st.success("⏰ Zeitbasierte Vorhersage: ✅ Aktiviert")
                
                # SMOTE
                if params.get('use_smote') is False:
                    st.info("⚖️ SMOTE: ❌ Deaktiviert")
                else:
                    st.success("⚖️ SMOTE: ✅ Aktiviert")
                
                # TimeSeriesSplit
                if params.get('use_timeseries_split') is False:
                    st.info("🔀 TimeSeriesSplit: ❌ Deaktiviert")
                else:
                    st.success("🔀 TimeSeriesSplit: ✅ Aktiviert")
                
                # CV-Splits
                cv_splits = params.get('cv_splits')
                if cv_splits:
                    st.write(f"🔀 Cross-Validation: {cv_splits} Splits")
                
                # Hyperparameter
                st.subheader("🎛️ Hyperparameter")
                hyperparams = []
                if params.get('n_estimators'):
                    hyperparams.append(f"n_estimators: {params['n_estimators']}")
                if params.get('max_depth'):
                    hyperparams.append(f"max_depth: {params['max_depth']}")
                if params.get('learning_rate'):
                    hyperparams.append(f"learning_rate: {params['learning_rate']}")
                if params.get('min_samples_split'):
                    hyperparams.append(f"min_samples_split: {params['min_samples_split']}")
                if params.get('min_samples_leaf'):
                    hyperparams.append(f"min_samples_leaf: {params['min_samples_leaf']}")
                
                if hyperparams:
                    for hp in hyperparams:
                        st.code(hp, language=None)
                else:
                    st.caption("Standard-Hyperparameter verwendet")
            else:
                st.write("Keine Parameter verfügbar")
    
    # TAB 4: Features
    with tab_features:
        # Feature Importance Chart
        if model.get('feature_importance'):
            st.subheader("🎯 Feature Importance")
            
            # Erklärung zu Feature Importance
            with st.expander("ℹ️ Was ist Feature Importance?", expanded=False):
                st.markdown("""
                **Feature Importance** zeigt, welche Features am wichtigsten für die Vorhersagen des Modells sind.
                
                **Wie wird es berechnet?**
                - **Random Forest / XGBoost:** Misst, wie oft ein Feature zur Verbesserung der Vorhersage beiträgt
                - **Höhere Werte** = Feature ist wichtiger für die Vorhersage
                - **Niedrigere Werte** = Feature hat weniger Einfluss
                
                **Was bedeutet das praktisch?**
                - **Top Features** sind die wichtigsten Indikatoren für Pump-Erkennung
                - Features mit hoher Importance sollten bei der Datenanalyse priorisiert werden
                - Features mit sehr niedriger Importance könnten möglicherweise entfernt werden
                
                **Beispiele für wichtige Features:**
                - `dev_sold_amount` - Wichtigster Rug-Pull-Indikator
                - `price_vs_ath_pct` - Wie nah am All-Time High?
                - `buy_pressure_ratio` - Verhältnis von Käufen zu Verkäufen
                - `volume_sol` - Handelsvolumen
                - `whale_buy_volume_sol` - Große Käufe (Whales)
                
                **💡 Interpretation:**
                - Wenn `dev_sold_amount` sehr hoch ist → Modell erkennt Rug-Pulls gut
                - Wenn `price_vs_ath_pct` wichtig ist → Modell nutzt ATH-Tracking effektiv
                - Wenn viele engineered Features wichtig sind → Feature-Engineering war erfolgreich
                """)
            
            fi = model['feature_importance']
            if isinstance(fi, dict):
                df_fi = pd.DataFrame(list(fi.items()), columns=['Feature', 'Importance'])
                df_fi = df_fi.sort_values('Importance', ascending=False)
                
                # Top 20 Features
                st.write("**Top 20 wichtigste Features:**")
                st.dataframe(df_fi.head(20), use_container_width=True, hide_index=True)
                
                # Visualisierung
                fig = px.bar(df_fi.head(20), x='Feature', y='Importance', title="Feature Importance (Top 20)")
                fig.update_xaxes(tickangle=-45)
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)
                
                # Interpretation der Top Features
                top_5_features = df_fi.head(5)
                st.info(f"""
                **🔍 Top 5 wichtigste Features:**
                {chr(10).join([f"1. **{row['Feature']}** ({row['Importance']:.4f})" for idx, row in top_5_features.iterrows()])}
                
                Diese Features tragen am meisten zur Pump-Erkennung bei.
                """)
                
                # Statistiken
                st.subheader("📊 Feature-Statistiken")
                stat_col1, stat_col2, stat_col3 = st.columns(3)
                with stat_col1:
                    st.metric("Gesamt Features", len(df_fi))
                with stat_col2:
                    st.metric("Durchschnittliche Importance", f"{df_fi['Importance'].mean():.4f}")
                with stat_col3:
                    st.metric("Max Importance", f"{df_fi['Importance'].max():.4f}")
        else:
            st.info("ℹ️ Keine Feature Importance Daten verfügbar")
    
    # TAB 5: Details (JSON)
    with tab_details:
        st.subheader("📋 Vollständige Modell-Daten")
        st.caption("Alle verfügbaren Daten des Modells im JSON-Format")
        st.json(model)
    
    # Zurück-Button (auch in Sidebar verfügbar)
    col1, col2 = st.columns([1, 3])
    with col1:
        if st.button("← Zurück zur Übersicht", key="back_to_overview_bottom", use_container_width=True):
            st.session_state.pop('page', None)
            st.session_state.pop('details_model_id', None)