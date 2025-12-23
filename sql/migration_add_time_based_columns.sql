-- Migration: Zeitbasierte Vorhersage-Spalten hinzufügen
-- Erstellt: 2024-12-23
-- Beschreibung: Fügt Spalten für zeitbasierte Vorhersagen zu ml_models und ml_jobs hinzu

-- ============================================================
-- ml_models: Zeitbasierte Vorhersage-Parameter
-- ============================================================
ALTER TABLE ml_models
ADD COLUMN IF NOT EXISTS future_minutes INTEGER,
ADD COLUMN IF NOT EXISTS price_change_percent NUMERIC(10, 4),
ADD COLUMN IF NOT EXISTS target_direction VARCHAR(10);

-- Kommentare
COMMENT ON COLUMN ml_models.future_minutes IS 'Anzahl Minuten in die Zukunft für zeitbasierte Vorhersage (z.B. 10)';
COMMENT ON COLUMN ml_models.price_change_percent IS 'Mindest-Prozent-Änderung für zeitbasierte Vorhersage (z.B. 5.0)';
COMMENT ON COLUMN ml_models.target_direction IS 'Richtung der Vorhersage: "up" oder "down"';

-- ============================================================
-- ml_jobs: Zeitbasierte Vorhersage-Parameter für Training
-- ============================================================
ALTER TABLE ml_jobs
ADD COLUMN IF NOT EXISTS train_future_minutes INTEGER,
ADD COLUMN IF NOT EXISTS train_price_change_percent NUMERIC(10, 4),
ADD COLUMN IF NOT EXISTS train_target_direction VARCHAR(10);

-- Kommentare
COMMENT ON COLUMN ml_jobs.train_future_minutes IS 'Anzahl Minuten in die Zukunft für zeitbasierte Vorhersage (Training)';
COMMENT ON COLUMN ml_jobs.train_price_change_percent IS 'Mindest-Prozent-Änderung für zeitbasierte Vorhersage (Training)';
COMMENT ON COLUMN ml_jobs.train_target_direction IS 'Richtung der Vorhersage: "up" oder "down" (Training)';

-- ============================================================
-- Index für bessere Performance (optional)
-- ============================================================
-- Keine Indizes nötig, da diese Spalten selten gefiltert werden

