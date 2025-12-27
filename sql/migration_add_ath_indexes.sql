-- ============================================================================
-- Migration: Performance-Indizes für ATH-Tracking
-- ============================================================================
-- Zweck: Optimiert JOIN zwischen coin_metrics und coin_streams für ATH-Daten
-- Datum: 2025-01-XX
-- ============================================================================

-- Index für schnellen JOIN zwischen coin_metrics und coin_streams
-- Wichtig: mint-Spalte in coin_metrics muss vorhanden sein!
CREATE INDEX IF NOT EXISTS idx_coin_metrics_mint 
ON coin_metrics(mint);

-- Index für coin_streams.token_address (für JOIN)
CREATE INDEX IF NOT EXISTS idx_coin_streams_token_address 
ON coin_streams(token_address);

-- Composite Index für ATH-Abfragen (nur aktive Coins)
-- Verbessert Performance bei ATH-Daten-Abfragen
CREATE INDEX IF NOT EXISTS idx_coin_streams_ath 
ON coin_streams(token_address, ath_price_sol, ath_timestamp) 
WHERE is_active = TRUE;

-- Index für coin_metrics.timestamp (falls noch nicht vorhanden)
-- Wichtig für Zeitraum-Abfragen
CREATE INDEX IF NOT EXISTS idx_coin_metrics_timestamp 
ON coin_metrics(timestamp);

-- Composite Index für mint + timestamp (für optimierte ATH-JOINs)
CREATE INDEX IF NOT EXISTS idx_coin_metrics_mint_timestamp 
ON coin_metrics(mint, timestamp);

-- ============================================================================
-- Kommentare
-- ============================================================================

COMMENT ON INDEX idx_coin_metrics_mint IS 'Optimiert JOIN zwischen coin_metrics und coin_streams für ATH-Daten';
COMMENT ON INDEX idx_coin_streams_token_address IS 'Optimiert JOIN zwischen coin_streams und coin_metrics';
COMMENT ON INDEX idx_coin_streams_ath IS 'Composite Index für ATH-Abfragen (nur aktive Coins)';
COMMENT ON INDEX idx_coin_metrics_timestamp IS 'Optimiert Zeitraum-Abfragen in coin_metrics';
COMMENT ON INDEX idx_coin_metrics_mint_timestamp IS 'Composite Index für optimierte ATH-JOINs mit Zeitraum-Filter';

-- ============================================================================
-- Prüfung
-- ============================================================================

-- Prüfe ob Indizes erstellt wurden:
-- SELECT indexname, indexdef 
-- FROM pg_indexes 
-- WHERE tablename IN ('coin_metrics', 'coin_streams')
-- AND indexname LIKE '%ath%' OR indexname LIKE '%mint%'
-- ORDER BY tablename, indexname;


