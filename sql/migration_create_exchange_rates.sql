-- ============================================================================
-- Migration: Exchange Rates Tabelle erstellen
-- Datum: 2024-12-XX
-- Beschreibung: Erstellt exchange_rates Tabelle für Marktstimmung (SOL-Preis)
-- ============================================================================

CREATE TABLE IF NOT EXISTS exchange_rates (
    id SERIAL PRIMARY KEY,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    sol_price_usd NUMERIC(20, 6) NOT NULL,
    usd_to_eur_rate NUMERIC(10, 6),
    native_currency_price_usd NUMERIC(20, 6),
    blockchain_id INTEGER DEFAULT 1,
    source VARCHAR(50)
);

-- Indizes
CREATE INDEX IF NOT EXISTS idx_exchange_rates_created_at ON exchange_rates(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_exchange_rates_blockchain_id ON exchange_rates(blockchain_id);

-- Kommentare
COMMENT ON TABLE exchange_rates IS 'Marktstimmung ("Wasserstand") - Referenztabelle für KI-Training zur Unterscheidung von echten Token-Pumps vs. allgemeinen Marktbewegungen';
COMMENT ON COLUMN exchange_rates.sol_price_usd IS 'WICHTIG: Der "Wasserstand" - Aktueller SOL-Preis in USD (z.B. 145.50)';
COMMENT ON COLUMN exchange_rates.created_at IS 'Zeitstempel des Snapshots';

