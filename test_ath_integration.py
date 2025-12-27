"""
Test-Skript für ATH-Integration
Prüft Datenbank-Verbindung, Tabellen-Struktur und ATH-Daten-Verfügbarkeit
"""
import asyncio
import sys
from datetime import datetime, timedelta
import os

# Füge app-Verzeichnis zum Path hinzu
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Setze DB_DSN als Umgebungsvariable VOR dem Import
DB_DSN = "postgresql://postgres:9HVxi6hN6j7xpmqUx84o@100.118.155.75:5432/beta"
os.environ['DB_DSN'] = DB_DSN

from app.database.connection import get_pool, close_pool
from app.training.feature_engineering import (
    load_training_data,
    validate_ath_data_availability,
    create_pump_detection_features
)
import pandas as pd

async def test_database_connection():
    """Test 1: Prüfe Datenbank-Verbindung"""
    print("=" * 60)
    print("TEST 1: Datenbank-Verbindung")
    print("=" * 60)
    
    try:
        pool = await get_pool()
        print("✅ Datenbank-Verbindung erfolgreich!")
        
        # Test-Query
        result = await pool.fetchval("SELECT version()")
        print(f"📊 PostgreSQL Version: {result[:50]}...")
        
        return True
    except Exception as e:
        print(f"❌ Fehler bei Datenbank-Verbindung: {e}")
        return False

async def test_table_structure():
    """Test 2: Prüfe Tabellen-Struktur"""
    print("\n" + "=" * 60)
    print("TEST 2: Tabellen-Struktur")
    print("=" * 60)
    
    try:
        pool = await get_pool()
        
        # Prüfe coin_metrics
        print("\n📊 coin_metrics Tabelle:")
        columns = await pool.fetch("""
            SELECT column_name, data_type 
            FROM information_schema.columns 
            WHERE table_name = 'coin_metrics' 
            AND column_name IN (
                'mint', 'timestamp', 'price_close', 'ath_price_sol',
                'dev_sold_amount', 'buy_pressure_ratio', 'volatility_pct'
            )
            ORDER BY column_name
        """)
        
        required_columns = ['mint', 'timestamp', 'price_close']
        found_columns = [row['column_name'] for row in columns]
        
        for col in required_columns:
            if col in found_columns:
                print(f"  ✅ {col}")
            else:
                print(f"  ❌ {col} FEHLT!")
        
        # Prüfe coin_streams
        print("\n📊 coin_streams Tabelle:")
        columns = await pool.fetch("""
            SELECT column_name, data_type 
            FROM information_schema.columns 
            WHERE table_name = 'coin_streams' 
            AND column_name IN (
                'token_address', 'ath_price_sol', 'ath_timestamp', 'is_active'
            )
            ORDER BY column_name
        """)
        
        required_columns = ['token_address', 'ath_price_sol', 'ath_timestamp']
        found_columns = [row['column_name'] for row in columns]
        
        for col in required_columns:
            if col in found_columns:
                print(f"  ✅ {col}")
            else:
                print(f"  ❌ {col} FEHLT!")
        
        return True
    except Exception as e:
        print(f"❌ Fehler bei Tabellen-Prüfung: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_ath_data_availability():
    """Test 3: Prüfe ATH-Daten-Verfügbarkeit"""
    print("\n" + "=" * 60)
    print("TEST 3: ATH-Daten-Verfügbarkeit")
    print("=" * 60)
    
    try:
        # Letzte 7 Tage
        train_end = datetime.now()
        train_start = train_end - timedelta(days=7)
        
        result = await validate_ath_data_availability(train_start, train_end)
        
        print(f"\n📊 Ergebnisse:")
        print(f"  Verfügbar: {result['available']}")
        print(f"  Coins mit ATH: {result['coins_with_ath']}")
        print(f"  Coins ohne ATH: {result['coins_without_ath']}")
        print(f"  Coverage: {result['coverage_pct']:.1f}%")
        print(f"  Gesamt Coins: {result['total_coins']}")
        
        if result['available']:
            print("\n✅ ATH-Daten verfügbar!")
        else:
            print("\n⚠️ Keine ATH-Daten verfügbar!")
        
        return result['available']
    except Exception as e:
        print(f"❌ Fehler bei ATH-Daten-Prüfung: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_load_training_data_with_ath():
    """Test 4: Lade Trainingsdaten mit ATH"""
    print("\n" + "=" * 60)
    print("TEST 4: Trainingsdaten mit ATH laden")
    print("=" * 60)
    
    try:
        # Letzte 7 Tage
        train_end = datetime.now()
        train_start = train_end - timedelta(days=7)
        
        print(f"\n📅 Zeitraum: {train_start} bis {train_end}")
        
        # Lade Daten mit ATH
        data = await load_training_data(
            train_start=train_start,
            train_end=train_end,
            features=['price_close', 'volume_sol'],
            include_ath=True
        )
        
        print(f"\n📊 Geladene Daten:")
        print(f"  Zeilen: {len(data)}")
        print(f"  Spalten: {list(data.columns)}")
        
        # Prüfe ATH-Spalten
        ath_columns = ['ath_price_sol', 'price_vs_ath_pct', 'minutes_since_ath']
        for col in ath_columns:
            if col in data.columns:
                non_null = data[col].notna().sum()
                print(f"  ✅ {col}: {non_null}/{len(data)} nicht-NULL Werte")
            else:
                print(f"  ❌ {col} FEHLT!")
        
        if len(data) > 0:
            print(f"\n📊 Beispiel-Daten (erste 5 Zeilen):")
            print(data[['price_close', 'ath_price_sol', 'price_vs_ath_pct']].head())
        
        return len(data) > 0
    except Exception as e:
        print(f"❌ Fehler beim Laden der Daten: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_ath_feature_engineering():
    """Test 5: ATH-Feature-Engineering"""
    print("\n" + "=" * 60)
    print("TEST 5: ATH-Feature-Engineering")
    print("=" * 60)
    
    try:
        # Erstelle Test-Daten
        test_data = pd.DataFrame({
            'price_close': [0.001, 0.0015, 0.002, 0.0025],
            'ath_price_sol': [0.002, 0.002, 0.0025, 0.0025],
            'volume_sol': [10, 20, 30, 40],
            'minutes_since_ath': [100, 50, 10, 5]
        })
        test_data.index = pd.date_range(start='2024-01-01', periods=4, freq='5min')
        
        print(f"\n📊 Test-Daten: {len(test_data)} Zeilen")
        
        # Erstelle Features
        df = create_pump_detection_features(test_data, window_sizes=[5])
        
        # Prüfe ATH-Features
        ath_features = [col for col in df.columns if 'ath' in col.lower()]
        print(f"\n📊 Erstellte ATH-Features: {len(ath_features)}")
        for feature in sorted(ath_features):
            print(f"  ✅ {feature}")
        
        if len(ath_features) > 0:
            print("\n✅ ATH-Feature-Engineering erfolgreich!")
        else:
            print("\n⚠️ Keine ATH-Features erstellt!")
        
        return len(ath_features) > 0
    except Exception as e:
        print(f"❌ Fehler bei Feature-Engineering: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Hauptfunktion"""
    print("\n" + "=" * 60)
    print("ATH-INTEGRATION TEST")
    print("=" * 60)
    
    results = {}
    
    # Test 1: Datenbank-Verbindung
    results['connection'] = await test_database_connection()
    
    if not results['connection']:
        print("\n❌ Datenbank-Verbindung fehlgeschlagen! Abbruch.")
        await close_pool()
        return
    
    # Test 2: Tabellen-Struktur
    results['structure'] = await test_table_structure()
    
    # Test 3: ATH-Daten-Verfügbarkeit
    results['ath_availability'] = await test_ath_data_availability()
    
    # Test 4: Trainingsdaten laden
    results['load_data'] = await test_load_training_data_with_ath()
    
    # Test 5: Feature-Engineering
    results['feature_engineering'] = await test_ath_feature_engineering()
    
    # Zusammenfassung
    print("\n" + "=" * 60)
    print("ZUSAMMENFASSUNG")
    print("=" * 60)
    
    for test_name, result in results.items():
        status = "✅" if result else "❌"
        print(f"{status} {test_name}: {result}")
    
    all_passed = all(results.values())
    
    if all_passed:
        print("\n✅ Alle Tests erfolgreich!")
    else:
        print("\n⚠️ Einige Tests fehlgeschlagen!")
    
    await close_pool()

if __name__ == "__main__":
    asyncio.run(main())

