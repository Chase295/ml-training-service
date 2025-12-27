"""
Streamlit UI für ML Training Service
Web-Interface für Modell-Management mit Tab-basiertem Layout
REFACTORED VERSION - Aufgeteilt in Module
"""
import streamlit as st
import os

# Konfiguration
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

# Page Config
st.set_page_config(
    page_title="ML Training Service - Control Panel",
    page_icon="🤖",
    layout="wide"
)

# ============================================================
# Imports aus streamlit_pages
# ============================================================

from app.streamlit_pages.overview import page_overview
from app.streamlit_pages.details import page_details
from app.streamlit_pages.test_results import page_test_results
from app.streamlit_pages.test_details import page_test_details
from app.streamlit_pages.training import page_train
from app.streamlit_pages.test import page_test
from app.streamlit_pages.compare import page_compare
from app.streamlit_pages.comparisons import page_comparisons
from app.streamlit_pages.comparison_details import page_comparison_details
from app.streamlit_pages.jobs import page_jobs
from app.streamlit_pages.tabs import (
    tab_dashboard,
    tab_configuration,
    tab_logs,
    tab_metrics,
    tab_info
)

# ============================================================
# Main App
# ============================================================

def main():
    """Hauptfunktion mit Tab-basiertem Layout"""
    st.title("🤖 ML Training Service - Control Panel")
    
    # Tabs Navigation
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10, tab11, tab12 = st.tabs([
        "📊 Dashboard",
        "⚙️ Konfiguration",
        "📋 Logs",
        "📈 Metriken",
        "ℹ️ Info",
        "🏠 Modelle",
        "➕ Training",
        "🧪 Testen",
        "📋 Test-Ergebnisse",
        "⚔️ Vergleichen",
        "⚖️ Vergleichs-Übersicht",
        "📊 Jobs"
    ])
    
    with tab1:
        tab_dashboard()
    
    with tab2:
        tab_configuration()
    
    with tab3:
        tab_logs()
    
    with tab4:
        tab_metrics()
    
    with tab5:
        tab_info()
    
    with tab6:
        # Prüfe ob Details-Seite angezeigt werden soll
        if st.session_state.get('page') == 'details':
            page_details()
        else:
            page_overview()
    
    with tab7:
        page_train()
    
    with tab8:
        page_test()
    
    with tab9:
        # Prüfe ob Test-Details angezeigt werden soll
        if st.session_state.get('page') == 'test_details' and st.session_state.get('test_details_id'):
            page_test_details()
        else:
            # Setze page zurück wenn keine test_details_id vorhanden ist
            if st.session_state.get('page') == 'test_details':
                st.session_state.pop('page', None)
            page_test_results()
    
    with tab10:
        # Prüfe ob Vergleichs-Details angezeigt werden soll
        if st.session_state.get('page') == 'comparison_details':
            page_comparison_details()
        else:
            page_compare()
    
    with tab11:
        page_comparisons()
    
    with tab12:
        page_jobs()

if __name__ == "__main__":
    main()


