-- Migration: Erweitere ml_test_results um Train vs. Test Vergleich (Phase 2)
-- Datum: 2024-12-23

-- Train vs. Test Vergleich
ALTER TABLE ml_test_results ADD COLUMN IF NOT EXISTS train_accuracy NUMERIC(5, 4);
ALTER TABLE ml_test_results ADD COLUMN IF NOT EXISTS train_f1 NUMERIC(5, 4);
ALTER TABLE ml_test_results ADD COLUMN IF NOT EXISTS train_precision NUMERIC(5, 4);
ALTER TABLE ml_test_results ADD COLUMN IF NOT EXISTS train_recall NUMERIC(5, 4);
ALTER TABLE ml_test_results ADD COLUMN IF NOT EXISTS accuracy_degradation NUMERIC(5, 4);
ALTER TABLE ml_test_results ADD COLUMN IF NOT EXISTS f1_degradation NUMERIC(5, 4);
ALTER TABLE ml_test_results ADD COLUMN IF NOT EXISTS is_overfitted BOOLEAN;

-- Test-Zeitraum Info
ALTER TABLE ml_test_results ADD COLUMN IF NOT EXISTS test_duration_days NUMERIC(10, 2);

COMMENT ON COLUMN ml_test_results.train_accuracy IS 'Training Accuracy (aus ml_models) für Vergleich';
COMMENT ON COLUMN ml_test_results.train_f1 IS 'Training F1 (aus ml_models) für Vergleich';
COMMENT ON COLUMN ml_test_results.accuracy_degradation IS 'Train - Test Accuracy (Overfitting-Indikator, > 0.1 = Overfitting)';
COMMENT ON COLUMN ml_test_results.is_overfitted IS 'True wenn accuracy_degradation > 0.1 (Modell generalisiert schlecht)';
COMMENT ON COLUMN ml_test_results.test_duration_days IS 'Test-Zeitraum in Tagen (für Validierung)';

