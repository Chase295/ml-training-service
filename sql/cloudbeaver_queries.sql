-- ============================================================
-- CloudBeaver SQL-Queries für ML-Tabellen
-- ============================================================

-- 1. Alle ML-Tabellen auflisten
SELECT table_name 
FROM information_schema.tables 
WHERE table_schema = 'public' 
AND table_name LIKE 'ml_%'
ORDER BY table_name;

-- 2. ml_models Daten anzeigen (erste 10)
SELECT 
    id, 
    name, 
    model_type, 
    status, 
    target_variable,
    created_at 
FROM ml_models 
ORDER BY id DESC 
LIMIT 10;

-- 3. ml_jobs Daten anzeigen (erste 10)
SELECT 
    id, 
    job_type, 
    status, 
    progress,
    progress_msg,
    created_at 
FROM ml_jobs 
ORDER BY id DESC 
LIMIT 10;

-- 4. ml_test_results Daten anzeigen (erste 10)
SELECT 
    id, 
    model_id, 
    accuracy, 
    f1_score, 
    precision_score,
    recall,
    num_samples,
    created_at 
FROM ml_test_results 
ORDER BY id DESC 
LIMIT 10;

-- 5. ml_comparisons Daten anzeigen (alle)
SELECT 
    id, 
    model_a_id, 
    model_b_id, 
    winner_id, 
    a_accuracy,
    a_f1,
    b_accuracy,
    b_f1,
    created_at 
FROM ml_comparisons 
ORDER BY id DESC;

-- 6. Anzahl Zeilen pro Tabelle
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

