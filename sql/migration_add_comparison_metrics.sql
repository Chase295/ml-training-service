-- Migration: Erweitere ml_comparisons um zusätzliche Metriken (Phase 9)
-- Datum: 2024-12-23
-- 
-- Diese Migration fügt alle neuen Metriken hinzu, die auch bei Test und Training verwendet werden:
-- - MCC (Matthews Correlation Coefficient)
-- - FPR (False Positive Rate)
-- - FNR (False Negative Rate)
-- - Simulated Profit
-- - Confusion Matrix (JSONB)
-- - Train vs. Test Vergleich (Accuracy Degradation, Overfitting)
-- - Test Duration

-- Zusätzliche Metriken für Modell A
ALTER TABLE ml_comparisons ADD COLUMN IF NOT EXISTS a_mcc NUMERIC(5, 4);
ALTER TABLE ml_comparisons ADD COLUMN IF NOT EXISTS a_fpr NUMERIC(5, 4);
ALTER TABLE ml_comparisons ADD COLUMN IF NOT EXISTS a_fnr NUMERIC(5, 4);
ALTER TABLE ml_comparisons ADD COLUMN IF NOT EXISTS a_simulated_profit_pct NUMERIC(10, 4);
ALTER TABLE ml_comparisons ADD COLUMN IF NOT EXISTS a_confusion_matrix JSONB;
ALTER TABLE ml_comparisons ADD COLUMN IF NOT EXISTS a_train_accuracy NUMERIC(5, 4);
ALTER TABLE ml_comparisons ADD COLUMN IF NOT EXISTS a_train_f1 NUMERIC(5, 4);
ALTER TABLE ml_comparisons ADD COLUMN IF NOT EXISTS a_accuracy_degradation NUMERIC(5, 4);
ALTER TABLE ml_comparisons ADD COLUMN IF NOT EXISTS a_f1_degradation NUMERIC(5, 4);
ALTER TABLE ml_comparisons ADD COLUMN IF NOT EXISTS a_is_overfitted BOOLEAN;
ALTER TABLE ml_comparisons ADD COLUMN IF NOT EXISTS a_test_duration_days NUMERIC(10, 2);

-- Zusätzliche Metriken für Modell B
ALTER TABLE ml_comparisons ADD COLUMN IF NOT EXISTS b_mcc NUMERIC(5, 4);
ALTER TABLE ml_comparisons ADD COLUMN IF NOT EXISTS b_fpr NUMERIC(5, 4);
ALTER TABLE ml_comparisons ADD COLUMN IF NOT EXISTS b_fnr NUMERIC(5, 4);
ALTER TABLE ml_comparisons ADD COLUMN IF NOT EXISTS b_simulated_profit_pct NUMERIC(10, 4);
ALTER TABLE ml_comparisons ADD COLUMN IF NOT EXISTS b_confusion_matrix JSONB;
ALTER TABLE ml_comparisons ADD COLUMN IF NOT EXISTS b_train_accuracy NUMERIC(5, 4);
ALTER TABLE ml_comparisons ADD COLUMN IF NOT EXISTS b_train_f1 NUMERIC(5, 4);
ALTER TABLE ml_comparisons ADD COLUMN IF NOT EXISTS b_accuracy_degradation NUMERIC(5, 4);
ALTER TABLE ml_comparisons ADD COLUMN IF NOT EXISTS b_f1_degradation NUMERIC(5, 4);
ALTER TABLE ml_comparisons ADD COLUMN IF NOT EXISTS b_is_overfitted BOOLEAN;
ALTER TABLE ml_comparisons ADD COLUMN IF NOT EXISTS b_test_duration_days NUMERIC(10, 2);

-- Kommentare hinzufügen
COMMENT ON COLUMN ml_comparisons.a_mcc IS 'Matthews Correlation Coefficient für Modell A';
COMMENT ON COLUMN ml_comparisons.a_fpr IS 'False Positive Rate für Modell A';
COMMENT ON COLUMN ml_comparisons.a_fnr IS 'False Negative Rate für Modell A';
COMMENT ON COLUMN ml_comparisons.a_simulated_profit_pct IS 'Simulierter Profit in Prozent für Modell A';
COMMENT ON COLUMN ml_comparisons.a_confusion_matrix IS 'Confusion Matrix als JSONB für Modell A: {"tp": 1234, "tn": 5678, "fp": 890, "fn": 198}';
COMMENT ON COLUMN ml_comparisons.a_train_accuracy IS 'Train Accuracy für Modell A (für Train vs. Test Vergleich)';
COMMENT ON COLUMN ml_comparisons.a_train_f1 IS 'Train F1 für Modell A (für Train vs. Test Vergleich)';
COMMENT ON COLUMN ml_comparisons.a_accuracy_degradation IS 'Accuracy Degradation (Test - Train) für Modell A';
COMMENT ON COLUMN ml_comparisons.a_f1_degradation IS 'F1 Degradation (Test - Train) für Modell A';
COMMENT ON COLUMN ml_comparisons.a_is_overfitted IS 'Ist Modell A overfitted? (basierend auf accuracy_degradation > 0.1)';
COMMENT ON COLUMN ml_comparisons.a_test_duration_days IS 'Test-Zeitraum Dauer in Tagen für Modell A';

COMMENT ON COLUMN ml_comparisons.b_mcc IS 'Matthews Correlation Coefficient für Modell B';
COMMENT ON COLUMN ml_comparisons.b_fpr IS 'False Positive Rate für Modell B';
COMMENT ON COLUMN ml_comparisons.b_fnr IS 'False Negative Rate für Modell B';
COMMENT ON COLUMN ml_comparisons.b_simulated_profit_pct IS 'Simulierter Profit in Prozent für Modell B';
COMMENT ON COLUMN ml_comparisons.b_confusion_matrix IS 'Confusion Matrix als JSONB für Modell B: {"tp": 1234, "tn": 5678, "fp": 890, "fn": 198}';
COMMENT ON COLUMN ml_comparisons.b_train_accuracy IS 'Train Accuracy für Modell B (für Train vs. Test Vergleich)';
COMMENT ON COLUMN ml_comparisons.b_train_f1 IS 'Train F1 für Modell B (für Train vs. Test Vergleich)';
COMMENT ON COLUMN ml_comparisons.b_accuracy_degradation IS 'Accuracy Degradation (Test - Train) für Modell B';
COMMENT ON COLUMN ml_comparisons.b_f1_degradation IS 'F1 Degradation (Test - Train) für Modell B';
COMMENT ON COLUMN ml_comparisons.b_is_overfitted IS 'Ist Modell B overfitted? (basierend auf accuracy_degradation > 0.1)';
COMMENT ON COLUMN ml_comparisons.b_test_duration_days IS 'Test-Zeitraum Dauer in Tagen für Modell B';

