-- ============================================================
-- Schema-Verbesserungen (Optional, aber empfohlen)
-- Stand: 2024-12-23
-- 
-- Diese Verbesserungen erhöhen Performance und Datenintegrität
-- ============================================================

-- ============================================================
-- 1. ZUSÄTZLICHE INDIZES (Performance)
-- ============================================================

-- Für UI-Filter (model_type wird häufig gefiltert)
CREATE INDEX IF NOT EXISTS idx_models_type_status 
    ON ml_models(model_type, status) 
    WHERE is_deleted = FALSE;

-- Für Test-Ergebnisse (häufig nach model_id + created_at sortiert)
CREATE INDEX IF NOT EXISTS idx_test_results_model_created 
    ON ml_test_results(model_id, created_at DESC);

-- Für Vergleiche (Gewinner-Suche)
CREATE INDEX IF NOT EXISTS idx_comparisons_winner 
    ON ml_comparisons(winner_id) 
    WHERE winner_id IS NOT NULL;

-- Für Jobs (häufig nach job_type gefiltert)
CREATE INDEX IF NOT EXISTS idx_jobs_type_status 
    ON ml_jobs(job_type, status, created_at DESC);

-- ============================================================
-- 2. CHECK CONSTRAINTS (Datenintegrität)
-- ============================================================

-- ml_models: Zeitbasierte Vorhersage-Parameter
ALTER TABLE ml_models 
    DROP CONSTRAINT IF EXISTS chk_future_minutes,
    ADD CONSTRAINT chk_future_minutes 
        CHECK (future_minutes IS NULL OR future_minutes > 0);

ALTER TABLE ml_models 
    DROP CONSTRAINT IF EXISTS chk_price_change_percent,
    ADD CONSTRAINT chk_price_change_percent 
        CHECK (price_change_percent IS NULL OR price_change_percent > 0);

-- ml_test_results: Datum-Validierung
ALTER TABLE ml_test_results 
    DROP CONSTRAINT IF EXISTS chk_test_dates,
    ADD CONSTRAINT chk_test_dates 
        CHECK (test_start < test_end);

ALTER TABLE ml_test_results 
    DROP CONSTRAINT IF EXISTS chk_test_duration,
    ADD CONSTRAINT chk_test_duration 
        CHECK (test_duration_days IS NULL OR test_duration_days >= 0);

-- ml_comparisons: Datum-Validierung
ALTER TABLE ml_comparisons 
    DROP CONSTRAINT IF EXISTS chk_compare_dates,
    ADD CONSTRAINT chk_compare_dates 
        CHECK (test_start < test_end);

-- ml_jobs: Datum-Validierung
ALTER TABLE ml_jobs 
    DROP CONSTRAINT IF EXISTS chk_train_dates,
    ADD CONSTRAINT chk_train_dates 
        CHECK (train_start IS NULL OR train_end IS NULL OR train_start < train_end);

ALTER TABLE ml_jobs 
    DROP CONSTRAINT IF EXISTS chk_test_job_dates,
    ADD CONSTRAINT chk_test_job_dates 
        CHECK (test_start IS NULL OR test_end IS NULL OR test_start < test_end);

ALTER TABLE ml_jobs 
    DROP CONSTRAINT IF EXISTS chk_compare_job_dates,
    ADD CONSTRAINT chk_compare_job_dates 
        CHECK (compare_start IS NULL OR compare_end IS NULL OR compare_start < compare_end);

-- ml_jobs: Zeitbasierte Vorhersage-Parameter
ALTER TABLE ml_jobs 
    DROP CONSTRAINT IF EXISTS chk_train_future_minutes,
    ADD CONSTRAINT chk_train_future_minutes 
        CHECK (train_future_minutes IS NULL OR train_future_minutes > 0);

ALTER TABLE ml_jobs 
    DROP CONSTRAINT IF EXISTS chk_train_price_change_percent,
    ADD CONSTRAINT chk_train_price_change_percent 
        CHECK (train_price_change_percent IS NULL OR train_price_change_percent > 0);

-- ============================================================
-- 3. OPTIONAL: VERSIONIERUNG (Wenn benötigt)
-- ============================================================

-- ALTER TABLE ml_models ADD COLUMN IF NOT EXISTS version INT DEFAULT 1;
-- CREATE INDEX IF NOT EXISTS idx_models_version ON ml_models(version);

-- ============================================================
-- 4. OPTIONAL: JSONB GIN INDIZES (Nur bei sehr großen Datenmengen)
-- ============================================================

-- Für häufige Filter auf params (z.B. WHERE params @> '{"n_estimators": 100}')
-- CREATE INDEX IF NOT EXISTS idx_models_params_gin ON ml_models USING GIN (params);

-- Für häufige Filter auf features (z.B. WHERE features @> '["price_open"]')
-- CREATE INDEX IF NOT EXISTS idx_models_features_gin ON ml_models USING GIN (features);

-- ============================================================
-- KOMMENTARE
-- ============================================================

COMMENT ON INDEX idx_models_type_status IS 'Index für häufige Filter auf model_type und status in der UI';
COMMENT ON INDEX idx_test_results_model_created IS 'Index für Sortierung nach model_id und created_at';
COMMENT ON INDEX idx_comparisons_winner IS 'Index für Suche nach Gewinner-Modellen';
COMMENT ON INDEX idx_jobs_type_status IS 'Index für Filterung nach job_type und status';

