-- Migration: Zusätzliche Metriken zu ml_models hinzufügen
-- Verbesserung 1.5: Zusätzliche Metriken

ALTER TABLE ml_models ADD COLUMN IF NOT EXISTS roc_auc NUMERIC(5, 4);
ALTER TABLE ml_models ADD COLUMN IF NOT EXISTS mcc NUMERIC(5, 4);
ALTER TABLE ml_models ADD COLUMN IF NOT EXISTS fpr NUMERIC(5, 4);
ALTER TABLE ml_models ADD COLUMN IF NOT EXISTS fnr NUMERIC(5, 4);
ALTER TABLE ml_models ADD COLUMN IF NOT EXISTS confusion_matrix JSONB;
ALTER TABLE ml_models ADD COLUMN IF NOT EXISTS simulated_profit_pct NUMERIC(10, 4);

COMMENT ON COLUMN ml_models.roc_auc IS 'ROC-AUC Score (Receiver Operating Characteristic - Area Under Curve)';
COMMENT ON COLUMN ml_models.mcc IS 'Matthews Correlation Coefficient (besser für imbalanced data)';
COMMENT ON COLUMN ml_models.fpr IS 'False Positive Rate (wichtig für Pump-Detection)';
COMMENT ON COLUMN ml_models.fnr IS 'False Negative Rate';
COMMENT ON COLUMN ml_models.confusion_matrix IS 'JSONB Object: {"tp": int, "tn": int, "fp": int, "fn": int}';
COMMENT ON COLUMN ml_models.simulated_profit_pct IS 'Simulierter Profit in Prozent (vereinfachte Simulation)';

