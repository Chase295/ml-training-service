-- ============================================================
-- Migration: CV-Scores und zusätzliche Metriken hinzufügen
-- Datum: 2025-12-24
-- Beschreibung: Fügt fehlende Spalten für CV-Scores und zusätzliche Metriken hinzu
-- 
-- Kombiniert:
-- - migration_add_cv_scores.sql (cv_scores, cv_overfitting_gap)
-- - migration_add_additional_metrics.sql (roc_auc, mcc, fpr, fnr, confusion_matrix, simulated_profit_pct)
-- ============================================================

-- Cross-Validation Metriken
ALTER TABLE ml_models ADD COLUMN IF NOT EXISTS cv_scores JSONB;
ALTER TABLE ml_models ADD COLUMN IF NOT EXISTS cv_overfitting_gap NUMERIC(5, 4);

COMMENT ON COLUMN ml_models.cv_scores IS 'JSONB Object: Cross-Validation Ergebnisse {"train_accuracy": [...], "test_accuracy": [...], ...}';
COMMENT ON COLUMN ml_models.cv_overfitting_gap IS 'Differenz zwischen Train- und Test-Accuracy (Overfitting-Indikator)';

-- Zusätzliche Metriken
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

-- Zeige Status (optional - zur Verifizierung)
SELECT 
    column_name, 
    CASE 
        WHEN data_type = 'jsonb' THEN 'JSONB'
        WHEN data_type = 'numeric' THEN data_type || '(' || numeric_precision || ',' || numeric_scale || ')'
        ELSE data_type
    END as full_type
FROM information_schema.columns
WHERE table_name = 'ml_models' 
    AND column_name IN (
        'cv_scores', 'cv_overfitting_gap', 
        'roc_auc', 'mcc', 'fpr', 'fnr', 
        'confusion_matrix', 'simulated_profit_pct'
    )
ORDER BY column_name;

