#!/usr/bin/env python3
"""Prüfe Job-Status"""
import asyncio
import sys
import time
sys.path.insert(0, '/app')
from app.database.connection import get_pool

async def check_job(job_id, max_wait=600):
    pool = await get_pool()
    waited = 0
    
    while waited < max_wait:
        job = await pool.fetchrow('''
            SELECT status, progress, result_model_id, error_msg, progress_msg
            FROM ml_jobs WHERE id = $1
        ''', job_id)
        
        if job:
            status = job['status']
            progress = job.get('progress', 0) * 100
            msg = job.get('progress_msg', '')[:50] if job.get('progress_msg') else ''
            print(f"[{waited}s] Status: {status}, Progress: {progress:.1f}% - {msg}")
            
            if status == 'COMPLETED':
                model_id = job.get('result_model_id')
                print(f"\n✅ Training abgeschlossen! Modell-ID: {model_id}")
                await pool.close()
                return model_id
            elif status == 'FAILED':
                error = job.get('error_msg', 'Unbekannter Fehler')
                print(f"\n❌ Training fehlgeschlagen: {error}")
                await pool.close()
                return None
        
        await pool.close()
        time.sleep(5)
        waited += 5
        pool = await get_pool()
    
    print(f"\n⚠️ Timeout nach {max_wait} Sekunden")
    await pool.close()
    return None

if __name__ == '__main__':
    job_id = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    result = asyncio.run(check_job(job_id))
    if result:
        print(f"MODEL_ID={result}")

