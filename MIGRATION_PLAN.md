# MIGRATION_PLAN.md - ML Training Service: Streamlit → React Migration

## Executive Summary

**Ziel:** Vollständige Migration der Streamlit-basierten ML Training Service UI zu einer modernen React/Vite-Architektur, die exakt dasselbe Look & Feel wie die bestehende `pump-find` React-App hat.

**Quelle der Wahrheit:**
- **Design & Architektur:** `@pump-find` (Referenz-App)
- **Funktionalität:** `@ml-training-service` (Streamlit-App)

---

## Schritt 1: Analyse der Referenz-Architektur (pump-find) ✅

### Tech Stack
- **Framework:** React 18 mit TypeScript
- **Build Tool:** Vite 5.4.8
- **UI Library:** Material-UI (MUI) v7.3.6 mit Emotion
- **State Management:** Zustand 5.0.9
- **Routing:** React Router DOM 7.11.0
- **HTTP Client:** Axios 1.13.2
- **Charts:** Recharts 3.6.0
- **Styling:** MUI Theme System (kein Tailwind)

### Konfiguration (vite.config.ts)
```typescript
server: {
  host: '0.0.0.0',
  port: 5173,
  allowedHosts: ['test.local.chase295.de', 'localhost', '127.0.0.1'],
  proxy: {
    '/api': {
      target: 'http://localhost:8000',
      changeOrigin: true,
      // KEIN rewrite - Pfad bleibt /api erhalten
    }
  }
}
```

### Theme & Styling
- **Mode:** Dark Theme
- **Primary Color:** `#00d4ff` (Cyan)
- **Secondary Color:** `#ff4081` (Pink)
- **Background:** Gradient von `#1a1a2e` → `#16213e` → `#0f0f23`
- **Paper Background:** `rgba(255, 255, 255, 0.05)`
- **Backdrop Filter:** `blur(10px)` für Glaseffekt

### Ordnerstruktur
```
pump-ui/
├── src/
│   ├── App.tsx              # Hauptlayout + Routing
│   ├── main.tsx             # Entry Point
│   ├── pages/               # Seitenkomponenten
│   │   ├── Dashboard.tsx
│   │   ├── Config.tsx
│   │   ├── Logs.tsx
│   │   ├── Metrics.tsx
│   │   └── Info.tsx
│   ├── services/
│   │   └── api.ts           # API Service mit Axios
│   ├── stores/
│   │   └── pumpStore.ts     # Zustand Store
│   ├── types/
│   │   └── api.ts           # TypeScript Typen
│   └── style.css            # Global Styles
```

### Layout-System
- **Sidebar Navigation:** 250px Breite, responsive (mobile: Drawer)
- **Top Bar:** App Bar mit Titel und Version
- **Responsive Design:** Mobile-first mit MUI Breakpoints
- **Navigation Items:** Icon + Label, aktive States mit Cyan Highlight

---

## Schritt 2: Analyse der Funktionalität (ml-training-service) ✅

### Feature Inventory (12 Tabs)

#### 1. 📊 Dashboard
- Service-Status (Health Check)
- System-Metriken Übersicht
- Keine direkte Entsprechung in pump-find → eigenes Dashboard

#### 2. ⚙️ Konfiguration
- Service-Konfiguration anzeigen/bearbeiten
- API-Endpunkt: `GET/PUT /api/config`

#### 3. 📋 Logs
- Service-Logs anzeigen
- Wahrscheinlich API-Endpunkt benötigt

#### 4. 📈 Metriken
- Prometheus-Metriken darstellen
- Charts mit Recharts (LineChart, PieChart)
- API-Endpunkt: `GET /api/metrics`

#### 5. ℹ️ Info
- Statische Informationen
- Entspricht pump-find Info-Seite

#### 6. 🏠 Modelle (Übersicht)
- Modell-Liste mit Filtern (Status, Typ)
- Karten-Layout mit Checkbox-Auswahl
- Aktionen: Details, Umbenennen, Löschen, Download
- Bulk-Aktionen für mehrere Modelle
- API-Endpunkt: `GET /api/models`

#### 7. ➕ Training (Neues Modell)
- Komplexes Formular für Modell-Training
- Feature-Auswahl (kategorisiert)
- Zeitbereich-Auswahl
- Hyperparameter-Konfiguration
- API-Endpunkt: `POST /api/models/create/*`

#### 8. 🧪 Testen (Einzeltest)
- Modell-Test mit Daten
- Test-Konfiguration
- API-Endpunkt: `POST /api/models/{id}/test`

#### 9. 📋 Test-Ergebnisse
- Test-Ergebnisse auflisten
- Detail-Ansicht einzelner Tests
- API-Endpunkt: `GET /api/test-results`

#### 10. ⚔️ Vergleichen (Modellvergleich)
- Zwei Modelle vergleichen
- API-Endpunkt: `POST /api/models/compare`

#### 11. ⚖️ Vergleichs-Übersicht
- Vergleichs-Historie
- API-Endpunkt: `GET /api/comparisons`

#### 12. 📊 Jobs
- Hintergrund-Jobs überwachen
- Training/Test-Jobs Status
- API-Endpunkt: `GET /api/queue`

### API Endpoints Mapping

#### Modelle Management
- `GET /api/models` - Liste aller Modelle
- `GET /api/models/{id}` - Einzelnes Modell
- `POST /api/models/create/simple` - Einfaches Training
- `POST /api/models/create/time-based` - Zeitbasiertes Training
- `POST /api/models/create` - Vollständiges Training
- `DELETE /api/models/{id}` - Modell löschen
- `GET /api/models/{id}/download` - Modell herunterladen

#### Testing & Comparison
- `POST /api/models/{id}/test` - Modell testen
- `POST /api/models/compare` - Modelle vergleichen
- `GET /api/test-results` - Test-Ergebnisse
- `GET /api/comparisons` - Vergleichs-Ergebnisse

#### Jobs & Monitoring
- `GET /api/queue` - Job-Liste
- `GET /api/queue/{id}` - Einzelner Job
- `GET /api/health` - Health Status
- `GET /api/metrics` - Prometheus Metriken

#### Configuration
- `GET /api/config` - Konfiguration laden
- `PUT /api/config` - Konfiguration speichern
- `POST /api/reload-config` - Config neu laden

#### Data & Utils
- `GET /api/phases` - Coin-Phasen
- `GET /api/data-availability` - Datenverfügbarkeit

### UI Komponenten Mapping

#### Streamlit → React/MUI
- `st.title()` → `<Typography variant="h4">`
- `st.subheader()` → `<Typography variant="h6">`
- `st.text_input()` → `<TextField>`
- `st.selectbox()` → `<Select>` oder `<Autocomplete>`
- `st.date_input()` → MUI DatePicker
- `st.time_input()` → MUI TimePicker
- `st.checkbox()` → `<Checkbox>`
- `st.button()` → `<Button>`
- `st.form()` → `<form>` mit MUI FormControl
- `st.columns()` → MUI Grid oder Box mit flex
- `st.dataframe()` → MUI DataGrid oder Table
- `st.plotly_chart()` → Recharts Komponenten
- `st.tabs()` → MUI Tabs
- `st.sidebar` → MUI Drawer (wie in pump-find)
- `st.expander()` → MUI Accordion

---

## Schritt 3: Phasen-Plan

### Phase 1: Setup & Scaffolding (1-2 Tage)

#### 1.1 Projekt-Initialisierung
- Neues Vite-Projekt mit React + TypeScript erstellen
- `package.json` von pump-find kopieren und anpassen
- Dependencies installieren (MUI, Zustand, Axios, Recharts, etc.)

#### 1.2 Konfiguration kopieren
- `vite.config.ts` exakt von pump-find kopieren
- `tsconfig.json` kopieren
- Proxy-Konfiguration für `/api` → `localhost:8000` einrichten

#### 1.3 Basis-Ordnerstruktur
```
ml-training-ui/
├── src/
│   ├── App.tsx              # Layout + Routing
│   ├── main.tsx
│   ├── pages/               # Alle 12 Seiten
│   ├── services/
│   │   └── api.ts           # API Service
│   ├── stores/
│   │   └── mlStore.ts       # Zustand Store
│   ├── types/
│   │   └── api.ts           # API Types
│   ├── components/          # Shared Components
│   └── style.css
```

#### 1.4 Theme & Layout Setup
- Theme exakt von pump-find kopieren (Farben, Typography)
- App-Shell erstellen (Sidebar, TopBar, Layout)
- Navigation mit allen 12 Tabs einrichten

### Phase 2: Komponenten-Bibliothek & Basis-Layout (2-3 Tage)

#### 2.1 Shared Components erstellen
- `ModelCard.tsx` - Wiederverwendbare Modell-Karte
- `DataTable.tsx` - Generische Daten-Tabelle
- `StatusChip.tsx` - Status-Anzeigen
- `MetricCard.tsx` - Metrik-Karten
- `FormComponents.tsx` - Formular-Elemente

#### 2.2 API Service aufbauen
- Axios Setup mit Base-URL
- Alle API-Endpoints typisieren
- Error Handling implementieren
- Interceptors für Logging

#### 2.3 Zustand Store erstellen
- `mlStore.ts` mit allen States:
  - Modelle, Jobs, TestResults, Comparisons
  - Loading States, Errors
  - UI States (selected Models, etc.)
- Actions für alle CRUD-Operationen

#### 2.4 Layout-System fertigstellen
- Responsive Sidebar mit allen 12 Navigation-Items
- Mobile Drawer für kleine Bildschirme
- TopBar mit Titel und Version

### Phase 3: Feature-Implementierung (8-10 Tage)

#### 3.1 Einfache Seiten (2 Tage)
- **Info-Seite:** Statischer Content
- **Health Dashboard:** Service-Status anzeigen
- **Konfiguration:** Config Formular
- **Metriken:** Charts mit Recharts

#### 3.2 Modelle-Übersicht (3 Tage)
- Modell-Liste mit Filtern
- Karten-Layout implementieren
- Selektion und Bulk-Aktionen
- Details-Modal/Dialog

#### 3.3 Training-Formular (3 Tage)
- Komplexes Training-Formular
- Feature-Auswahl mit Kategorien
- Zeitbereich-Picker
- Form-Validierung

#### 3.4 Testing & Comparison (2 Tage)
- Test-Interface für einzelne Modelle
- Vergleichs-Interface für zwei Modelle
- Ergebnisse anzeigen

#### 3.5 Jobs & Monitoring (1 Tag)
- Job-Status Übersicht
- Live-Updates für laufende Jobs

### Phase 4: Validierung & Polishing (2-3 Tage)

#### 4.1 Look & Feel Validierung
- **Visueller Abgleich:** Jedes Element mit pump-find vergleichen
- **Responsive Testing:** Mobile, Tablet, Desktop
- **Theme Consistency:** Farben, Typography, Spacing

#### 4.2 Funktionale Validierung
- **API Integration:** Alle Endpoints testen
- **Error Handling:** Fehlerszenarien abdecken
- **Loading States:** UX für alle Async-Operationen

#### 4.3 Performance & UX
- **Loading Performance:** Lazy Loading für große Listen
- **User Experience:** Intuitive Navigation und Workflows
- **Accessibility:** Keyboard Navigation, Screen Reader

#### 4.4 Cross-Browser Testing
- Chrome, Firefox, Safari
- Mobile Browser Testing

### Phase 5: Deployment & Testing (1-2 Tage)

#### 5.1 Build & Deployment
- Production Build testen
- Docker-Integration
- Nginx-Konfiguration

#### 5.2 Integration Testing
- End-to-End Tests mit Playwright
- API-Integration Tests
- Performance Tests

---

## Technische Herausforderungen

### 1. Komplexe Formulare
- Training-Formular hat 20+ Felder mit komplexer Validierung
- Feature-Auswahl mit 50+ Checkboxen in Kategorien
- Zeitbereich-Handling mit DateTime-Pickern

### 2. State Management
- Mehrere abhängige States (selectedModels, currentPage, etc.)
- Real-time Updates für Jobs und Health-Status
- Form-State Persistierung

### 3. Datenvisualisierung
- Umstellung von Plotly zu Recharts
- Komplexe Charts für Metriken und Vergleiche
- Performance bei großen Datensätzen

### 4. Responsive Design
- Streamlit war nicht responsive
- Alle Komponenten müssen mobile-first sein
- Komplexe Tabellen auf kleinen Bildschirmen

### 5. API Integration
- Alle 20+ Endpoints implementieren
- Error Handling für alle API-Calls
- Loading States für UX

---

## Erfolgs-Kriterien

### ✅ Funktionale Parität
- Alle 12 Tabs/Seiten implementiert
- Alle API-Integrationen funktionieren
- Alle User-Workflows möglich

### ✅ Visuelle Identität
- Exakt gleiches Look & Feel wie pump-find
- Responsive auf allen Geräten
- Konsistente Theme-Anwendung

### ✅ Performance
- < 3s Initial Load
- < 1s für Seitenwechsel
- Smooth Animations und Transitions

### ✅ Code Quality
- TypeScript für Type Safety
- Modularer, wartbarer Code
- Gute Testabdeckung

---

## Ressourcen & Zeitplan

### Geschätzter Zeitaufwand: 18-22 Tage
- Phase 1: 2 Tage
- Phase 2: 3 Tage
- Phase 3: 10 Tage
- Phase 4: 3 Tage
- Phase 5: 2 Tage

### Team Setup
- 1 Senior Fullstack Developer (React/TypeScript)
- 1 UX/UI Developer für Polishing
- 1 Backend Developer für API-Support

### Risiken & Mitigation
- **API Changes:** Regelmäßige Abstimmung mit Backend-Team
- **Scope Creep:** Strenger Feature-Freeze nach Phase 3
- **Performance Issues:** Early Performance Testing in Phase 2

---

## Nächste Schritte

1. **Plan-Genehmigung:** Bitte bestätigen Sie "Plan genehmigt"
2. **Phase 1 Start:** Projekt-Setup beginnen
3. **Weekly Check-ins:** Fortschritts-Updates jede Woche
4. **Milestone Reviews:** Nach jeder Phase Review-Termin

**Bereit für Phase 1?** 🚀
