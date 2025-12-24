#!/usr/bin/env python3
"""
Prüft auf hängende Jobs (RUNNING Status > 10 Minuten)
"""

import requests
from datetime import datetime, timezone, timedelta

API_BASE_URL = "http://100.76.209.59:8005/api"

def main():
    print("\n" + "="*60)
    print("🔍 Prüfe auf hängende Jobs")
    print("="*60)
    
    try:
        response = requests.get(f"{API_BASE_URL}/queue", timeout=10)
        if response.status_code == 200:
            jobs = response.json()
            train_jobs = [j for j in jobs if j.get('job_type') == 'TRAIN']
            
            stuck_jobs = []
            now = datetime.now(timezone.utc)
            
            for job in train_jobs:
                if job.get('status') == 'RUNNING':
                    started_at = job.get('started_at')
                    if started_at:
                        try:
                            if isinstance(started_at, str):
                                started_dt = datetime.fromisoformat(started_at.replace('Z', '+00:00'))
                            else:
                                started_dt = started_at
                            
                            if started_dt.tzinfo is None:
                                started_dt = started_dt.replace(tzinfo=timezone.utc)
                            
                            duration = now - started_dt
                            if duration > timedelta(minutes=10):
                                stuck_jobs.append({
                                    'job': job,
                                    'duration': duration
                                })
                        except Exception as e:
                            print(f"⚠️  Fehler beim Parsen von started_at für Job {job.get('id')}: {e}")
            
            if stuck_jobs:
                print(f"\n⚠️  {len(stuck_jobs)} hängende Job(s) gefunden:\n")
                for item in stuck_jobs:
                    job = item['job']
                    duration = item['duration']
                    print(f"   Job ID: {job.get('id')}")
                    print(f"   Status: {job.get('status')}")
                    print(f"   Progress: {job.get('progress', 0)}%")
                    print(f"   Progress Msg: {job.get('progress_msg')}")
                    print(f"   Gestartet: {job.get('started_at')}")
                    print(f"   Hängt seit: {duration}")
                    print(f"   Modell-Typ: {job.get('train_model_type')}")
                    print()
            else:
                print("\n✅ Keine hängenden Jobs gefunden")
            
            # Zeige alle RUNNING Jobs
            running_jobs = [j for j in train_jobs if j.get('status') == 'RUNNING']
            if running_jobs:
                print(f"\n📊 {len(running_jobs)} RUNNING Job(s):\n")
                for job in running_jobs:
                    started_at = job.get('started_at')
                    if started_at:
                        try:
                            if isinstance(started_at, str):
                                started_dt = datetime.fromisoformat(started_at.replace('Z', '+00:00'))
                            else:
                                started_dt = started_at
                            
                            if started_dt.tzinfo is None:
                                started_dt = started_dt.replace(tzinfo=timezone.utc)
                            
                            duration = now - started_dt
                            print(f"   Job {job.get('id')}: {duration} (Progress: {job.get('progress', 0)}%)")
                        except:
                            print(f"   Job {job.get('id')}: Progress: {job.get('progress', 0)}%")
        else:
            print(f"❌ Fehler: {response.status_code}")
    except Exception as e:
        print(f"❌ Fehler: {e}")

if __name__ == "__main__":
    main()

