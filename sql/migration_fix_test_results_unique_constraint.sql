-- ============================================================================
-- MIGRATION: Fix UNIQUE Constraint für ml_test_results
-- Datum: 2025-12-27
-- 
-- Problem: 
-- ON CONFLICT (model_id, test_start, test_end) schlägt fehl weil kein UNIQUE Constraint existiert
-- 
-- Lösung:
-- Füge UNIQUE Constraint hinzu falls nicht vorhanden
-- ============================================================================

-- Prüfe und füge UNIQUE Constraint hinzu
DO $$
BEGIN
    -- Prüfe ob Constraint bereits existiert
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint 
        WHERE conname = 'unique_test_per_model_timeframe'
        AND conrelid = 'ml_test_results'::regclass
    ) THEN
        ALTER TABLE ml_test_results 
            ADD CONSTRAINT unique_test_per_model_timeframe 
            UNIQUE (model_id, test_start, test_end);
        RAISE NOTICE '✅ UNIQUE Constraint unique_test_per_model_timeframe hinzugefügt';
    ELSE
        RAISE NOTICE 'ℹ️ UNIQUE Constraint unique_test_per_model_timeframe existiert bereits';
    END IF;
END $$;

-- Index für schnelle Suche (falls nicht vorhanden)
CREATE INDEX IF NOT EXISTS idx_test_results_model_timeframe 
    ON ml_test_results(model_id, test_start, test_end);

-- Kommentar
COMMENT ON CONSTRAINT unique_test_per_model_timeframe ON ml_test_results IS 
    'Verhindert, dass ein Modell mehrfach mit dem gleichen Zeitraum getestet wird';


