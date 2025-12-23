-- ============================================================
-- ML-Tabellen komplett leeren
-- ============================================================
-- ⚠️ WICHTIG: Dieses Script löscht ALLE Daten aus den ML-Tabellen!
-- Verwende nur wenn du sicher bist, dass du alle Daten löschen möchtest.
-- 
-- Verwendung:
-- psql -h <HOST> -U postgres -d crypto -f sql/clear_ml_tables.sql
-- ============================================================

-- Deaktiviere Foreign Key Constraints temporär (für TRUNCATE)
SET session_replication_role = 'replica';

-- Lösche alle Daten aus ML-Tabellen
-- TRUNCATE ist schneller als DELETE und setzt auch Sequenzen zurück
TRUNCATE TABLE ml_comparisons CASCADE;
TRUNCATE TABLE ml_test_results CASCADE;
TRUNCATE TABLE ml_jobs CASCADE;
TRUNCATE TABLE ml_models CASCADE;

-- Reaktiviere Foreign Key Constraints
SET session_replication_role = 'origin';

-- Zeige Bestätigung
SELECT 
    'ml_models' as tabelle, COUNT(*) as anzahl FROM ml_models
UNION ALL
SELECT 
    'ml_jobs' as tabelle, COUNT(*) as anzahl FROM ml_jobs
UNION ALL
SELECT 
    'ml_test_results' as tabelle, COUNT(*) as anzahl FROM ml_test_results
UNION ALL
SELECT 
    'ml_comparisons' as tabelle, COUNT(*) as anzahl FROM ml_comparisons;

-- Erwartete Ausgabe: Alle Tabellen sollten 0 Zeilen haben
-- ✅ Alle ML-Tabellen wurden erfolgreich geleert!

