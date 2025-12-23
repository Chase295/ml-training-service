# Datenbank-Schema Analyse & Verbesserungsvorschläge

**Stand: 2024-12-23**

## ✅ Was ist gut:

1. **JSONB für flexible Daten** - Perfekt für features, params, confusion_matrix
2. **Foreign Keys vorhanden** - Datenintegrität gewährleistet
3. **Basis-Indizes** - status, created_at sind indexiert
4. **Constraints** - CHECK Constraints für Enum-Werte
5. **Soft Delete für Modelle** - Historische Daten bleiben erhalten

## ⚠️ Verbesserungsvorschläge:

### 1. **Fehlende Indizes (Performance)**

**Problem:** Häufig gefilterte Felder haben keinen Index

**Empfehlung:**
```sql
-- Für UI-Filter (model_type wird häufig gefiltert)
CREATE INDEX idx_models_type_status ON ml_models(model_type, status) WHERE is_deleted = FALSE;

-- Für Test-Ergebnisse (häufig nach model_id + created_at sortiert)
CREATE INDEX idx_test_results_model_created ON ml_test_results(model_id, created_at DESC);

-- Für Vergleiche (Gewinner-Suche)
CREATE INDEX idx_comparisons_winner ON ml_comparisons(winner_id) WHERE winner_id IS NOT NULL;

-- Für Jobs (häufig nach job_type gefiltert)
CREATE INDEX idx_jobs_type_status ON ml_jobs(job_type, status, created_at DESC);
```

**Impact:** Mittel - Verbessert Query-Performance bei Filtern

---

### 2. **Fehlende CHECK Constraints (Datenintegrität)**

**Problem:** Keine Validierung für numerische Werte

**Empfehlung:**
```sql
-- In ml_models
ALTER TABLE ml_models ADD CONSTRAINT chk_future_minutes CHECK (future_minutes IS NULL OR future_minutes > 0);
ALTER TABLE ml_models ADD CONSTRAINT chk_price_change_percent CHECK (price_change_percent IS NULL OR price_change_percent > 0);

-- In ml_test_results
ALTER TABLE ml_test_results ADD CONSTRAINT chk_test_dates CHECK (test_start < test_end);
ALTER TABLE ml_test_results ADD CONSTRAINT chk_test_duration CHECK (test_duration_days IS NULL OR test_duration_days >= 0);

-- In ml_comparisons
ALTER TABLE ml_comparisons ADD CONSTRAINT chk_compare_dates CHECK (test_start < test_end);
```

**Impact:** Hoch - Verhindert ungültige Daten

---

### 3. **Redundanz: Confusion Matrix**

**Problem:** confusion_matrix existiert sowohl als JSONB als auch als separate Spalten (tp, tn, fp, fn)

**Aktuell:**
- JSONB: `confusion_matrix JSONB`
- Separate: `tp INT, tn INT, fp INT, fn INT`

**Optionen:**
- **Option A (Empfohlen):** Beides behalten - JSONB für Flexibilität, separate Spalten für einfache Queries
- **Option B:** Nur JSONB - Weniger Redundanz, aber komplexere Queries

**Impact:** Niedrig - Aktuell funktioniert es gut

---

### 4. **Inkonsistente Timestamps**

**Problem:** Nicht alle Tabellen haben `updated_at`

**Aktuell:**
- `ml_models`: ✅ created_at, updated_at
- `ml_test_results`: ❌ nur created_at
- `ml_comparisons`: ❌ nur created_at
- `ml_jobs`: ✅ created_at, started_at, completed_at

**Empfehlung:** 
- Für Test-Ergebnisse und Vergleiche ist `updated_at` nicht kritisch (sind historische Snapshots)
- **Kann so bleiben** - macht Sinn

**Impact:** Niedrig - Nicht kritisch

---

### 5. **Fehlende Versionierung**

**Problem:** Keine Versionsnummer für Modelle

**Empfehlung:**
```sql
ALTER TABLE ml_models ADD COLUMN version INT DEFAULT 1;
CREATE INDEX idx_models_version ON ml_models(version);
```

**Impact:** Mittel - Nützlich für Model-Versionierung, aber nicht kritisch

---

### 6. **Performance: JSONB Indizes**

**Problem:** JSONB-Felder werden nicht indexiert

**Empfehlung (optional):**
```sql
-- Für häufige Filter auf params
CREATE INDEX idx_models_params_gin ON ml_models USING GIN (params);

-- Für häufige Filter auf features
CREATE INDEX idx_models_features_gin ON ml_models USING GIN (features);
```

**Impact:** Niedrig - Nur wenn sehr viele Modelle und häufige JSONB-Filter

---

## 🎯 Priorisierte Empfehlungen:

### **MUST HAVE (Sofort umsetzen):**
1. ✅ **CHECK Constraints** - Verhindert ungültige Daten
2. ✅ **Zusätzliche Indizes** - Verbessert Performance deutlich

### **SHOULD HAVE (Bald umsetzen):**
3. ⚠️ **Versionierung** - Wenn Model-Versionierung geplant ist

### **NICE TO HAVE (Optional):**
4. ℹ️ **JSONB GIN Indizes** - Nur bei sehr großen Datenmengen
5. ℹ️ **Redundanz reduzieren** - Nur wenn Wartbarkeit wichtiger als Performance

---

## 📊 Zusammenfassung:

**Aktueller Zustand:** **Gut** ✅
- Solide Basis-Struktur
- Gute Verwendung von JSONB
- Foreign Keys vorhanden

**Verbesserungspotenzial:** **Mittel** ⚠️
- Indizes für bessere Performance
- CHECK Constraints für Datenintegrität
- Optional: Versionierung

**Empfehlung:** 
- **CHECK Constraints und Indizes hinzufügen** (geringer Aufwand, hoher Nutzen)
- **Rest kann so bleiben** (funktioniert gut)

---

## 🔧 Konkrete SQL-Ergänzungen:

Siehe `sql/schema_improvements.sql` für ausführbare SQL-Statements.

