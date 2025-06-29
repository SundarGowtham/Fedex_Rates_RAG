"""
Master script to populate all data tables with real API data.
Runs weather, fuel, and traffic data fetchers.
"""

import asyncio
import logging
import sys
sys.path.append('.')

from src.utils.weather_data_fetcher import fetch_and_populate_weather_data
from src.utils.fuel_data_fetcher import fetch_and_populate_fuel_data
from src.utils.traffic_data_fetcher import fetch_and_populate_traffic_data
from src.core.database import get_database_manager

logger = logging.getLogger(__name__)

async def populate_all_data():
    """Populate all data tables with real API data."""
    print("🚀 Starting data population for all tables...")
    
    results = {}
    
    # Populate weather data
    print("\n🌤️  Fetching weather data...")
    try:
        results['weather'] = await fetch_and_populate_weather_data()
        print(f"✅ Weather data: {'Success' if results['weather'] else 'Failed'}")
    except Exception as e:
        print(f"❌ Weather data failed: {e}")
        results['weather'] = False
    
    # Populate fuel price data
    print("\n⛽ Fetching fuel price data...")
    try:
        results['fuel'] = await fetch_and_populate_fuel_data()
        print(f"✅ Fuel data: {'Success' if results['fuel'] else 'Failed'}")
    except Exception as e:
        print(f"❌ Fuel data failed: {e}")
        results['fuel'] = False
    
    # Populate traffic data
    print("\n🚗 Fetching traffic data...")
    try:
        results['traffic'] = await fetch_and_populate_traffic_data()
        print(f"✅ Traffic data: {'Success' if results['traffic'] else 'Failed'}")
    except Exception as e:
        print(f"❌ Traffic data failed: {e}")
        results['traffic'] = False
    
    # Show summary
    print("\n📊 Data Population Summary:")
    for data_type, success in results.items():
        status = "✅ Success" if success else "❌ Failed"
        print(f"  {data_type.title()}: {status}")
    
    # Show final table counts
    print("\n📈 Final Table Counts:")
    try:
        db = await get_database_manager()
        tables = ['weather_data', 'fuel_prices', 'traffic_data']
        
        for table in tables:
            result = await db.execute_query(f"SELECT COUNT(*) as count FROM {table}")
            count = result[0]['count']
            print(f"  {table}: {count} rows")
            
    except Exception as e:
        print(f"❌ Error getting table counts: {e}")
    
    return results

if __name__ == "__main__":
    asyncio.run(populate_all_data()) 