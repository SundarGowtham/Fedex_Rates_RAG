"""
Script to populate only weather data from OpenWeatherMap API.
This avoids hitting API limits for fuel and traffic data.
"""

import asyncio
import logging
import sys
sys.path.append('.')

from src.utils.weather_data_fetcher import fetch_and_populate_weather_data
from src.core.database import get_database_manager

logger = logging.getLogger(__name__)

async def populate_weather_only():
    """Populate only weather data table."""
    print("🌤️  Starting weather data population...")
    
    try:
        # Populate weather data
        success = await fetch_and_populate_weather_data()
        
        if success:
            print("✅ Weather data populated successfully!")
            
            # Show the final count
            db = await get_database_manager()
            result = await db.execute_query("SELECT COUNT(*) as count FROM weather_data")
            count = result[0]['count']
            print(f"📊 Weather data table now contains {count} records")
            
            # Show a sample of the data
            print("\n📋 Sample weather data:")
            sample = await db.execute_query("SELECT * FROM weather_data LIMIT 5")
            for row in sample:
                print(f"  {row['location']}: {row['temperature']}°F, {row['weather_condition']}")
                
        else:
            print("❌ Weather data population failed!")
            
    except Exception as e:
        print(f"❌ Error populating weather data: {e}")
        return False
    
    return success

if __name__ == "__main__":
    asyncio.run(populate_weather_only()) 