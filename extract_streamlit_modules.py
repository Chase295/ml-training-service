#!/usr/bin/env python3
"""
Script zum Extrahieren von Streamlit-Seiten-Modulen aus streamlit_app.py
"""
import re
import os
from pathlib import Path

# Pfade
SOURCE_FILE = "app/streamlit_app.py"
PAGES_DIR = "app/streamlit_pages"
UTILS_FILE = "app/streamlit_utils.py"

# Imports die alle Module brauchen
COMMON_IMPORTS = """import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any, Optional

# Import aus streamlit_utils
from app.streamlit_utils import (
    api_get, api_post, api_delete, api_patch,
    AVAILABLE_FEATURES, FEATURE_CATEGORIES, CRITICAL_FEATURES,
    API_BASE_URL, load_phases, load_config, save_config,
    get_default_config, validate_url, validate_port,
    reload_config, restart_service, get_service_logs
)
"""

def extract_function(source_code, func_name):
    """Extrahiert eine Funktion aus dem Source-Code"""
    pattern = rf'^def {func_name}\(.*?\):(.*?)(?=^def |\Z)'
    match = re.search(pattern, source_code, re.MULTILINE | re.DOTALL)
    if match:
        return match.group(0)
    return None

def read_source_file():
    """Liest die Source-Datei"""
    with open(SOURCE_FILE, 'r', encoding='utf-8') as f:
        return f.read()

def create_module_file(module_name, function_code, imports=COMMON_IMPORTS):
    """Erstellt eine Modul-Datei"""
    file_path = Path(PAGES_DIR) / f"{module_name}.py"
    
    content = f'''"""
{module_name.replace('_', ' ').title()} Page Module
Extrahierte Seite aus streamlit_app.py
"""
{imports}

{function_code}
'''
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"✅ Erstellt: {file_path}")

def main():
    """Hauptfunktion"""
    print("🔧 Extrahiere Streamlit-Module...")
    
    # Erstelle Pages-Verzeichnis
    Path(PAGES_DIR).mkdir(exist_ok=True)
    
    # Lese Source-Datei
    source_code = read_source_file()
    
    # Liste der zu extrahierenden Funktionen
    functions_to_extract = [
        ('overview', 'page_overview'),
        ('test_results', 'page_test_results'),
        ('test_details', 'page_test_details'),
        ('details', 'page_details'),
        ('training', 'page_train'),
        ('test', 'page_test'),
        ('compare', 'page_compare'),
        ('comparisons', 'page_comparisons'),
        ('comparison_details', 'page_comparison_details'),
        ('jobs', 'page_jobs'),
        ('tabs', 'tab_dashboard', 'tab_configuration', 'tab_logs', 'tab_metrics', 'tab_info'),
    ]
    
    for item in functions_to_extract:
        if isinstance(item, tuple) and len(item) > 1:
            module_name = item[0]
            func_names = item[1:]
            
            # Extrahiere alle Funktionen für dieses Modul
            func_codes = []
            for func_name in func_names:
                func_code = extract_function(source_code, func_name)
                if func_code:
                    func_codes.append(func_code)
                else:
                    print(f"⚠️ Funktion {func_name} nicht gefunden")
            
            if func_codes:
                combined_code = '\n\n'.join(func_codes)
                create_module_file(module_name, combined_code)
            else:
                print(f"❌ Keine Funktionen für {module_name} gefunden")
    
    print("\n✅ Extraktion abgeschlossen!")

if __name__ == "__main__":
    main()


