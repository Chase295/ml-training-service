-- ============================================================
-- MIGRATION: Schema-Verbesserung - Klare Hierarchie
-- Datum: 2025-12-24
-- 
-- Problem: 
-- - ml_comparisons hat keine Verweise auf ml_test_results
-- - Tests werden mehrfach erstellt (Duplikate)
-- - Keine eindeutige Verknüpfung
-- 
-- Lösung:
-- 1. UNIQUE Constraint auf ml_test_results (model_id + test_start + test_end)
-- 2. ml_comparisons bekommt test_a_id und test_b_id (Foreign Keys)
-- 3. Metriken werden aus Test-Ergebnissen übernommen (JOIN)
-- ============================================================

-- ============================================================
-- 1. UNIQUE Constraint für ml_test_results
-- ============================================================
-- Verhindert, dass ein Modell mehrfach mit dem gleichen Zeitraum getestet wird
DO $$
BEGIN
    -- Prüfe ob Constraint bereits existiert
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint 
        WHERE conname = 'unique_test_per_model_timeframe'
    ) THEN
        ALTER TABLE ml_test_results 
            ADD CONSTRAINT unique_test_per_model_timeframe 
            UNIQUE (model_id, test_start, test_end);
        RAISE NOTICE 'UNIQUE Constraint unique_test_per_model_timeframe hinzugefügt';
    ELSE
        RAISE NOTICE 'UNIQUE Constraint unique_test_per_model_timeframe existiert bereits';
    END IF;
END $$;

-- Index für schnelle Suche
CREATE INDEX IF NOT EXISTS idx_test_results_model_timeframe 
    ON ml_test_results(model_id, test_start, test_end);

-- ============================================================
-- 2. ml_comparisons: Füge test_a_id und test_b_id hinzu
-- ============================================================
-- Neue Spalten für Verweise auf Test-Ergebnisse
ALTER TABLE ml_comparisons 
    ADD COLUMN IF NOT EXISTS test_a_id BIGINT REFERENCES ml_test_results(id) ON DELETE CASCADE,
    ADD COLUMN IF NOT EXISTS test_b_id BIGINT REFERENCES ml_test_results(id) ON DELETE CASCADE;

-- Constraint: Beide Tests müssen unterschiedlich sein
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint 
        WHERE conname = 'chk_different_tests'
    ) THEN
        ALTER TABLE ml_comparisons 
            ADD CONSTRAINT chk_different_tests CHECK (test_a_id != test_b_id);
        RAISE NOTICE 'Constraint chk_different_tests hinzugefügt';
    ELSE
        RAISE NOTICE 'Constraint chk_different_tests existiert bereits';
    END IF;
END $$;

-- Index für schnelle Suche
CREATE INDEX IF NOT EXISTS idx_comparisons_tests 
    ON ml_comparisons(test_a_id, test_b_id);

-- ============================================================
-- 3. Kommentare hinzufügen
-- ============================================================
COMMENT ON COLUMN ml_test_results.model_id IS 'Foreign Key zu ml_models - jedes Test-Ergebnis gehört zu einem Modell';
COMMENT ON COLUMN ml_test_results.test_start IS 'Test-Zeitraum Start (zusammen mit test_end UNIQUE pro Modell)';
COMMENT ON COLUMN ml_test_results.test_end IS 'Test-Zeitraum Ende (zusammen mit test_start UNIQUE pro Modell)';

COMMENT ON COLUMN ml_comparisons.test_a_id IS 'Foreign Key zu ml_test_results - Test-Ergebnis für Modell A';
COMMENT ON COLUMN ml_comparisons.test_b_id IS 'Foreign Key zu ml_test_results - Test-Ergebnis für Modell B';
COMMENT ON COLUMN ml_comparisons.model_a_id IS 'Foreign Key zu ml_models - Modell A (kann aus test_a_id.model_id geholt werden)';
COMMENT ON COLUMN ml_comparisons.model_b_id IS 'Foreign Key zu ml_models - Modell B (kann aus test_b_id.model_id geholt werden)';

-- ============================================================
-- 4. Status-Anzeige
-- ============================================================
SELECT 
    'ml_test_results' as tabelle,
    COUNT(*) as anzahl_tests,
    COUNT(DISTINCT model_id) as anzahl_modelle
FROM ml_test_results
UNION ALL
SELECT 
    'ml_comparisons' as tabelle,
    COUNT(*) as anzahl_vergleiche,
    COUNT(DISTINCT test_a_id) + COUNT(DISTINCT test_b_id) as anzahl_tests_verknuepft
FROM ml_comparisons;

