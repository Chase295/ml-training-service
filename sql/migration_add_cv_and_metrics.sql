-- ============================================================
-- Migration: CV-Scores und zusätzliche Metriken hinzufügen
-- Datum: 2025-12-24
-- Beschreibung: Fügt fehlende Spalten für CV-Scores und zusätzliche Metriken hinzu
-- ============================================================

-- Prüfe ob Spalten bereits existieren, bevor sie hinzugefügt werden
DO $$
BEGIN
    -- Cross-Validation Metriken
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'ml_models' AND column_name = 'cv_scores'
    ) THEN
        ALTER TABLE ml_models ADD COLUMN cv_scores JSONB;
        RAISE NOTICE 'Spalte cv_scores hinzugefügt';
    END IF;

    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'ml_models' AND column_name = 'cv_overfitting_gap'
    ) THEN
        ALTER TABLE ml_models ADD COLUMN cv_overfitting_gap NUMERIC(5, 4);
        RAISE NOTICE 'Spalte cv_overfitting_gap hinzugefügt';
    END IF;

    -- Zusätzliche Metriken
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'ml_models' AND column_name = 'roc_auc'
    ) THEN
        ALTER TABLE ml_models ADD COLUMN roc_auc NUMERIC(5, 4);
        RAISE NOTICE 'Spalte roc_auc hinzugefügt';
    END IF;

    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'ml_models' AND column_name = 'mcc'
    ) THEN
        ALTER TABLE ml_models ADD COLUMN mcc NUMERIC(5, 4);
        RAISE NOTICE 'Spalte mcc hinzugefügt';
    END IF;

    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'ml_models' AND column_name = 'fpr'
    ) THEN
        ALTER TABLE ml_models ADD COLUMN fpr NUMERIC(5, 4);
        RAISE NOTICE 'Spalte fpr hinzugefügt';
    END IF;

    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'ml_models' AND column_name = 'fnr'
    ) THEN
        ALTER TABLE ml_models ADD COLUMN fnr NUMERIC(5, 4);
        RAISE NOTICE 'Spalte fnr hinzugefügt';
    END IF;

    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'ml_models' AND column_name = 'confusion_matrix'
    ) THEN
        ALTER TABLE ml_models ADD COLUMN confusion_matrix JSONB;
        RAISE NOTICE 'Spalte confusion_matrix hinzugefügt';
    END IF;

    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'ml_models' AND column_name = 'simulated_profit_pct'
    ) THEN
        ALTER TABLE ml_models ADD COLUMN simulated_profit_pct NUMERIC(10, 4);
        RAISE NOTICE 'Spalte simulated_profit_pct hinzugefügt';
    END IF;
END $$;

-- Zeige Status
SELECT 
    column_name, 
    data_type,
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

