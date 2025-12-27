# Streamlit Refactoring Plan

## Problem
Die `streamlit_app.py` Datei hat **5790 Zeilen** und **30 Funktionen**. Das macht sie:
- ❌ Schwer wartbar
- ❌ Fehleranfällig (Indentation-Fehler)
- ❌ Unübersichtlich
- ❌ Schwer zu testen

## Vorschlag: Aufteilung in Module

### Neue Struktur

```
app/
├── streamlit_app.py          # Hauptdatei (nur main() und Tab-Navigation, ~200 Zeilen)
├── streamlit_pages/           # Seiten-Module
│   ├── __init__.py
│   ├── dashboard.py          # Dashboard Tab
│   ├── config.py             # Konfiguration Tab
│   ├── logs.py               # Logs Tab
│   ├── metrics.py            # Metriken Tab
│   ├── info.py               # Info Tab
│   ├── overview.py           # Modell-Übersicht
│   ├── details.py            # Modell-Details (sehr groß, ~1200 Zeilen)
│   ├── training.py           # Training-Seite
│   ├── test.py               # Test-Seite
│   ├── test_results.py       # Test-Ergebnisse Übersicht
│   ├── test_details.py       # Test-Details
│   ├── compare.py            # Vergleich-Seite
│   ├── comparisons.py        # Vergleichs-Übersicht
│   └── comparison_details.py # Vergleichs-Details
├── streamlit_utils.py        # Hilfsfunktionen (API-Calls, etc.)
└── streamlit_components.py   # Wiederverwendbare UI-Komponenten
```

### Vorteile

1. **Bessere Wartbarkeit**
   - Jede Seite in eigener Datei
   - Klare Verantwortlichkeiten
   - Einfacher zu finden und zu ändern

2. **Weniger Fehler**
   - Kleinere Dateien = weniger Indentation-Probleme
   - Einfacher zu debuggen
   - Bessere Code-Organisation

3. **Bessere Testbarkeit**
   - Einzelne Seiten können isoliert getestet werden
   - Wiederverwendbare Komponenten

4. **Teamarbeit**
   - Mehrere Entwickler können parallel arbeiten
   - Weniger Merge-Konflikte

### Migration-Plan

1. **Phase 1: Vorbereitung**
   - ✅ Aktuelle Indentation-Fehler beheben
   - ✅ Backup erstellen

2. **Phase 2: Hilfsfunktionen extrahieren**
   - `streamlit_utils.py` erstellen
   - API-Funktionen (`api_get`, `api_post`, etc.) verschieben
   - Gemeinsame Hilfsfunktionen extrahieren

3. **Phase 3: Seiten-Module erstellen**
   - `streamlit_pages/` Ordner erstellen
   - Jede `page_*()` Funktion in eigene Datei verschieben
   - Jede `tab_*()` Funktion in eigene Datei verschieben

4. **Phase 4: Hauptdatei refactoren**
   - `streamlit_app.py` aufräumen
   - Nur `main()` und Tab-Navigation behalten
   - Imports hinzufügen

5. **Phase 5: Testing**
   - Alle Seiten testen
   - Navigation testen
   - Docker neu bauen

### Geschätzter Aufwand
- **Phase 1-2:** 1-2 Stunden
- **Phase 3:** 2-3 Stunden
- **Phase 4-5:** 1 Stunde
- **Gesamt:** 4-6 Stunden

### Risiken
- ⚠️ Mögliche Import-Fehler
- ⚠️ Session-State muss korrekt geteilt werden
- ⚠️ Streamlit-spezifische Besonderheiten beachten

### Empfehlung
**JA, wir sollten die Datei aufteilen!** Die Vorteile überwiegen deutlich.


