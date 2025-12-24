# 📊 Schema-Verbesserung - Klare Hierarchie

**Datum:** 2025-12-24  
**Problem:** Duplikate, keine klare Verknüpfung zwischen Tests und Vergleichen

---

## 🎯 Neue Struktur

### 1. **ml_models** - Modelle
- Jedes Modell mit Trainings-Infos und Ergebnissen
- **UNVERÄNDERT**

### 2. **ml_test_results** - Test-Ergebnisse
- **✅ NEU:** UNIQUE Constraint `(model_id, test_start, test_end)`
- **Verhindert Duplikate:** Ein Modell kann nur EINMAL mit dem gleichen Zeitraum getestet werden
- Jeder Test hat Verweis auf Modell (`model_id`)

### 3. **ml_comparisons** - Modell-Vergleiche
- **✅ NEU:** Verweise auf Test-Ergebnisse (`test_a_id`, `test_b_id`)
- **Statt:** Alle Metriken nochmal zu speichern
- **Jetzt:** Verweis auf existierende Test-Ergebnisse
- Metriken werden per JOIN aus Test-Ergebnissen geholt

---

## 🔗 Verknüpfungen

```
ml_models (1) ──┐
                ├──> ml_test_results (N) ──┐
ml_models (1) ──┘                          ├──> ml_comparisons (1)
                                          │
ml_models (1) ──┐                         │
                ├──> ml_test_results (N) ──┘
ml_models (1) ──┘
```

**Beispiel:**
- Modell A (ID: 1) → Test-Ergebnis A (ID: 10, Zeitraum: 2024-01-01 bis 2024-01-07)
- Modell B (ID: 2) → Test-Ergebnis B (ID: 11, Zeitraum: 2024-01-01 bis 2024-01-07)
- Vergleich (ID: 5) → test_a_id=10, test_b_id=11

---

## ✅ Vorteile

1. **Keine Duplikate mehr:**
   - UNIQUE Constraint verhindert mehrfache Tests
   - Ein Vergleich verwendet existierende Tests

2. **Klare Hierarchie:**
   - Modelle → Tests → Vergleiche
   - Alles baut aufeinander auf

3. **Datenintegrität:**
   - Foreign Keys garantieren Konsistenz
   - Tests können nicht "verloren" gehen

4. **Weniger Speicher:**
   - Metriken werden nicht mehrfach gespeichert
   - Nur Verweise in `ml_comparisons`

---

## 🔧 Migration

**Datei:** `sql/migration_schema_verbessert.sql`

**Was wird gemacht:**
1. UNIQUE Constraint auf `ml_test_results(model_id, test_start, test_end)`
2. Spalten `test_a_id` und `test_b_id` in `ml_comparisons`
3. Foreign Keys zu `ml_test_results`
4. Constraint: `test_a_id != test_b_id`

**Ausführung:**
```sql
\i sql/migration_schema_verbessert.sql
```

---

## 💻 Code-Änderungen

### `get_or_create_test_result()`
- Prüft zuerst, ob Test bereits existiert
- Gibt existierende ID zurück (verhindert Duplikate)
- Erstellt nur, wenn nicht vorhanden

### `create_comparison()`
- **Neue Parameter:** `test_a_id`, `test_b_id`
- **Neue Struktur:** Speichert nur Verweise (weniger Daten)
- **Alte Struktur:** Fallback für Rückwärtskompatibilität

### `process_compare_job()`
- Erstellt/findet Test-Ergebnisse für beide Modelle
- Verwendet Test-IDs für Vergleich
- **Keine Duplikate mehr möglich!**

---

## 📋 Beispiel-Workflow

**1. Modell erstellen:**
```
Job TRAIN → ml_models (ID: 1)
```

**2. Modell testen:**
```
Job TEST → ml_test_results (ID: 10, model_id: 1, test_start: ..., test_end: ...)
```

**3. Vergleich erstellen:**
```
Job COMPARE:
  - Findet/erstellt Test A (ID: 10)
  - Findet/erstellt Test B (ID: 11)
  - Erstellt Vergleich (ID: 5, test_a_id: 10, test_b_id: 11)
```

**Ergebnis:**
- ✅ Nur 1 Test-Ergebnis pro Modell + Zeitraum
- ✅ Vergleich verweist auf existierende Tests
- ✅ Keine Duplikate!

---

**Erstellt:** 2025-12-24  
**Status:** ✅ Implementiert

