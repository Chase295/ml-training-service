-- Migration: Erweitere ml_test_results um zusätzliche Metriken (Phase 9)
-- Datum: 2024-12-23

-- Zusätzliche Metriken hinzufügen
ALTER TABLE ml_test_results ADD COLUMN IF NOT EXISTS mcc NUMERIC(5, 4);
ALTER TABLE ml_test_results ADD COLUMN IF NOT EXISTS fpr NUMERIC(5, 4);
ALTER TABLE ml_test_results ADD COLUMN IF NOT EXISTS fnr NUMERIC(5, 4);
ALTER TABLE ml_test_results ADD COLUMN IF NOT EXISTS simulated_profit_pct NUMERIC(10, 4);

-- Confusion Matrix als JSONB hinzufügen
ALTER TABLE ml_test_results ADD COLUMN IF NOT EXISTS confusion_matrix JSONB;

COMMENT ON COLUMN ml_test_results.mcc IS 'Matthews Correlation Coefficient (besser für imbalanced data)';
COMMENT ON COLUMN ml_test_results.fpr IS 'False Positive Rate (wichtig für Pump-Detection)';
COMMENT ON COLUMN ml_test_results.fnr IS 'False Negative Rate';
COMMENT ON COLUMN ml_test_results.simulated_profit_pct IS 'Simulierter Profit in Prozent (1% Gewinn pro TP, -0.5% Verlust pro FP)';
COMMENT ON COLUMN ml_test_results.confusion_matrix IS 'Confusion Matrix als JSONB: {"tp": 1234, "tn": 5678, "fp": 890, "fn": 198}';

