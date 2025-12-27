# ✅ Schema-Prüfung Ergebnis

## 📊 Aktueller Status

### ✅ ml_models Tabelle
**Vorhanden:**
- Alle Basis-Spalten ✅
- Zeitbasierte Vorhersage-Parameter (`future_minutes`, `price_change_percent`, `target_direction`) ✅
- Cross-Validation Metriken (`cv_scores`, `cv_overfitting_gap`) ✅
- Zusätzliche Metriken (`roc_auc`, `mcc`, `fpr`, `fnr`, `confusion_matrix`, `simulated_profit_pct`) ✅

**FEHLT (wird durch Migration hinzugefügt):**
- ❌ `rug_detection_metrics` (JSONB) - **Wird durch `migration_add_rug_metrics.sql` hinzugefügt**
- ❌ `market_context_enabled` (BOOLEAN) - **Wird durch `migration_add_rug_metrics.sql` hinzugefügt**

### ✅ ml_test_results Tabelle
**Vorhanden:**
- Alle Basis-Metriken ✅
- Zusätzliche Metriken (`mcc`, `fpr`, `fnr`, `simulated_profit_pct`) ✅
- Train vs. Test Vergleich ✅

**FEHLT (wird durch Migration hinzugefügt):**
- ❌ `rug_detection_metrics` (JSONB) - **Wird durch `migration_add_rug_metrics.sql` hinzugefügt**

### ✅ coin_metrics Tabelle
**Erwartete Spalten (aus pump-metric Service):**
- ✅ `dev_sold_amount` (NUMERIC) - **KRITISCH für Rug-Detection**
- ✅ `buy_pressure_ratio` (NUMERIC) - **Wichtig für Bot-Spam-Erkennung**
- ✅ `unique_signer_ratio` (NUMERIC) - **Wichtig für Wash-Trading-Erkennung**
- ✅ `whale_buy_volume_sol` (NUMERIC) - **Whale-Aktivität**
- ✅ `whale_sell_volume_sol` (NUMERIC) - **Whale-Aktivität**
- ✅ `num_whale_buys` (INTEGER) - **Whale-Aktivität**
- ✅ `num_whale_sells` (INTEGER) - **Whale-Aktivität**
- ✅ `net_volume_sol` (NUMERIC) - **Netto-Volumen**
- ✅ `volatility_pct` (NUMERIC) - **Volatilität**
- ✅ `avg_trade_size_sol` (NUMERIC) - **Durchschnittliche Trade-Größe**
- ✅ `mint` (VARCHAR) - **Token-Adresse (für ATH-JOIN benötigt)**

**⚠️ WICHTIG:** Diese Spalten müssen in der `coin_metrics` Tabelle vorhanden sein, bevor das Training startet!

### ✅ coin_streams Tabelle (für ATH-Tracking)
**Erwartete Spalten (aus pump-metric Service):**
- ✅ `token_address` (VARCHAR) - **Token-Adresse (für JOIN mit coin_metrics)**
- ✅ `ath_price_sol` (NUMERIC) - **🆕 All-Time High Preis**
- ✅ `ath_timestamp` (TIMESTAMPTZ) - **🆕 Timestamp des letzten ATH-Updates**
- ✅ `is_active` (BOOLEAN) - **Aktiver Coin (für Filterung)**

**⚠️ WICHTIG:** ATH-Daten werden über LEFT JOIN aus `coin_streams` geladen. Falls keine ATH-Daten verfügbar sind, werden NULL-Werte durch 0 ersetzt.

**Prüfung:**
```sql
-- Prüfe ob alle benötigten Spalten vorhanden sind
SELECT column_name, data_type 
FROM information_schema.columns 
WHERE table_name = 'coin_metrics' 
AND column_name IN (
    'dev_sold_amount',
    'buy_pressure_ratio',
    'unique_signer_ratio',
    'whale_buy_volume_sol',
    'whale_sell_volume_sol',
    'net_volume_sol',
    'volatility_pct',
    'avg_trade_size_sol',
    'num_whale_buys',
    'num_whale_sells'
)
ORDER BY column_name;
```

### ❓ exchange_rates Tabelle
**Status:** Unbekannt - muss geprüft werden

**Erwartete Struktur:**
- `id` (SERIAL PRIMARY KEY)
- `created_at` (TIMESTAMPTZ)
- `sol_price_usd` (NUMERIC) - **KRITISCH für Marktstimmung**
- `usd_to_eur_rate` (NUMERIC)
- `native_currency_price_usd` (NUMERIC)
- `blockchain_id` (INTEGER)
- `source` (VARCHAR)

**Prüfung:**
```sql
-- Prüfe ob exchange_rates Tabelle existiert
SELECT EXISTS (
    SELECT FROM information_schema.tables 
    WHERE table_schema = 'public' 
    AND table_name = 'exchange_rates'
);
```

**Falls nicht vorhanden:** Wird durch `migration_create_exchange_rates.sql` erstellt.

---

## 🔧 Nötige Migrationen

### 1. migration_add_rug_metrics.sql
**Zweck:** Fügt `rug_detection_metrics` und `market_context_enabled` hinzu

**Betroffene Tabellen:**
- `ml_models` → +2 Spalten
- `ml_test_results` → +1 Spalte

**Status:** ✅ Datei erstellt (`sql/migration_add_rug_metrics.sql`)

**Ausführen:**
```bash
psql -h localhost -U postgres -d ml_training -f sql/migration_add_rug_metrics.sql
```

### 2. migration_create_exchange_rates.sql
**Zweck:** Erstellt `exchange_rates` Tabelle (falls nicht vorhanden)

**Betroffene Tabellen:**
- `exchange_rates` → NEU

**Status:** ✅ Datei erstellt (`sql/migration_create_exchange_rates.sql`)

**Ausführen (nur wenn Tabelle nicht existiert):**
```bash
psql -h localhost -U postgres -d ml_training -f sql/migration_create_exchange_rates.sql
```

---

## ✅ Schema nach Migrationen

### ml_models (nach Migration)
```sql
CREATE TABLE ml_models (
    -- ... bestehende Spalten ...
    
    -- NEU (durch Migration):
    rug_detection_metrics JSONB,           -- Rug-Pull-spezifische Metriken
    market_context_enabled BOOLEAN DEFAULT FALSE  -- Marktstimmung aktiviert
);
```

### ml_test_results (nach Migration)
```sql
CREATE TABLE ml_test_results (
    -- ... bestehende Spalten ...
    
    -- NEU (durch Migration):
    rug_detection_metrics JSONB  -- Rug-Pull-spezifische Metriken
);
```

### exchange_rates (nach Migration, falls erstellt)
```sql
CREATE TABLE exchange_rates (
    id SERIAL PRIMARY KEY,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    sol_price_usd NUMERIC(20, 6) NOT NULL,
    usd_to_eur_rate NUMERIC(10, 6),
    native_currency_price_usd NUMERIC(20, 6),
    blockchain_id INTEGER DEFAULT 1,
    source VARCHAR(50)
);
```

---

## 📋 Checkliste vor Implementierung

### Datenbank-Prüfung:
- [ ] `coin_metrics` hat alle neuen Spalten (`dev_sold_amount`, `buy_pressure_ratio`, etc.)
- [ ] `exchange_rates` Tabelle existiert (oder wird erstellt)
- [ ] Migration `migration_add_rug_metrics.sql` ausgeführt
- [ ] Migration `migration_create_exchange_rates.sql` ausgeführt (falls nötig)
- [ ] Spalten `rug_detection_metrics` und `market_context_enabled` in `ml_models` vorhanden
- [ ] Spalte `rug_detection_metrics` in `ml_test_results` vorhanden

### Code-Prüfung:
- [x] `load_training_data()` lädt alle neuen Metriken (inkl. ATH-Daten)
- [ ] `enrich_with_market_context()` kann Exchange Rates laden
- [x] `create_pump_detection_features()` nutzt neue Metriken (inkl. ATH-Features)
- [x] `validate_ath_data_availability()` prüft ATH-Daten-Verfügbarkeit
- [x] Performance-Indizes für ATH-JOIN erstellt (`migration_add_ath_indexes.sql`)

---

## 🚨 Wichtige Hinweise

1. **coin_metrics Schema:** Die `coin_metrics` Tabelle wird vom `pump-metric` Service verwaltet. Stelle sicher, dass alle neuen Spalten vorhanden sind, bevor du trainierst.

2. **exchange_rates Daten:** Die `exchange_rates` Tabelle wird vom `pump-discover` Service befüllt. Falls leer, wird Marktstimmung übersprungen (nur Warnung).

3. **Rückwärtskompatibilität:** Alle neuen Spalten sind optional (NULL erlaubt). Alte Modelle funktionieren weiterhin.

4. **Performance:** Der GIN-Index auf `rug_detection_metrics` verbessert Query-Performance für JSONB-Abfragen.

---

## ✅ Fazit

**Schema-Status:** ✅ **BEREIT FÜR MIGRATION**

**Nächste Schritte:**
1. Führe `migration_add_rug_metrics.sql` aus
2. Prüfe ob `exchange_rates` existiert, falls nicht: `migration_create_exchange_rates.sql` ausführen
3. Prüfe ob `coin_metrics` alle neuen Spalten hat
4. Beginne mit Backend-Implementierung (Phase 2)

---

**Erstellt:** 2024-12-XX
**Status:** ✅ Schema-Prüfung abgeschlossen

