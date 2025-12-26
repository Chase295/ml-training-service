-- ============================================================================
-- Migration: Rug-Detection-Metriken & Marktstimmung
-- Datum: 2024-12-XX
-- Beschreibung: Fügt rug_detection_metrics und market_context_enabled hinzu
-- ============================================================================

-- Erweitere ml_models Tabelle
ALTER TABLE ml_models 
ADD COLUMN IF NOT EXISTS rug_detection_metrics JSONB,
ADD COLUMN IF NOT EXISTS market_context_enabled BOOLEAN DEFAULT FALSE;

-- Erweitere ml_test_results Tabelle
ALTER TABLE ml_test_results
ADD COLUMN IF NOT EXISTS rug_detection_metrics JSONB;

-- Index für schnellere Queries
CREATE INDEX IF NOT EXISTS idx_ml_models_rug_metrics 
ON ml_models USING GIN (rug_detection_metrics);

-- Kommentare hinzufügen
COMMENT ON COLUMN ml_models.rug_detection_metrics IS 'JSONB Object: Rug-Pull-spezifische Metriken {"dev_sold_detection_rate": 0.85, "wash_trading_detection_rate": 0.72, "weighted_cost": 123.45, "precision_at_10": 0.90, ...}';
COMMENT ON COLUMN ml_models.market_context_enabled IS 'True wenn Marktstimmung (SOL-Preis-Kontext) aktiviert wurde';
COMMENT ON COLUMN ml_test_results.rug_detection_metrics IS 'JSONB Object: Rug-Pull-spezifische Metriken für Test-Ergebnisse';

