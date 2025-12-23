-- Migration: CV-Ergebnisse zu ml_models hinzufügen
-- Verbesserung 1.4: TimeSeriesSplit

ALTER TABLE ml_models ADD COLUMN IF NOT EXISTS cv_scores JSONB;
ALTER TABLE ml_models ADD COLUMN IF NOT EXISTS cv_overfitting_gap NUMERIC(5, 4);

COMMENT ON COLUMN ml_models.cv_scores IS 'JSONB Object: Cross-Validation Ergebnisse {"train_accuracy": [...], "test_accuracy": [...], ...}';
COMMENT ON COLUMN ml_models.cv_overfitting_gap IS 'Differenz zwischen Train- und Test-Accuracy (Overfitting-Indikator)';

