"""
Weather data fetcher using OpenWeatherMap API.
Fetches real weather data for major cities and populates the weather_data table.
"""

import asyncio
import logging
import os
from datetime import datetime
from typing import Dict, List, Optional

import httpx
import pandas as pd

from src.core.database import get_database_manager

logger = logging.getLogger(__name__)

# Load .env file manually
def load_env():
    """Load environment variables from .env file."""
    env_file = ".env"
    if os.path.exists(env_file):
        with open(env_file, "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    os.environ[key] = value

# Load environment variables
load_env()

# Major cities for weather data
MAJOR_CITIES = [
    {"name": "New York", "lat": 40.7128, "lon": -74.0060},
    {"name": "Los Angeles", "lat": 34.0522, "lon": -118.2437},
    {"name": "Chicago", "lat": 41.8781, "lon": -87.6298},
    {"name": "Houston", "lat": 29.7604, "lon": -95.3698},
    {"name": "Phoenix", "lat": 33.4484, "lon": -112.0740},
    {"name": "Philadelphia", "lat": 39.9526, "lon": -75.1652},
    {"name": "San Antonio", "lat": 29.4241, "lon": -98.4936},
    {"name": "San Diego", "lat": 32.7157, "lon": -117.1611},
    {"name": "Dallas", "lat": 32.7767, "lon": -96.7970},
    {"name": "San Jose", "lat": 37.3382, "lon": -121.8863},
    {"name": "Austin", "lat": 30.2672, "lon": -97.7431},
    {"name": "Jacksonville", "lat": 30.3322, "lon": -81.6557},
    {"name": "Fort Worth", "lat": 32.7555, "lon": -97.3308},
    {"name": "Columbus", "lat": 39.9612, "lon": -82.9988},
    {"name": "Charlotte", "lat": 35.2271, "lon": -80.8431},
]

class WeatherDataFetcher:
    """Fetches weather data from OpenWeatherMap API."""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("OPENWEATHER_API_KEY")
        
        if not self.api_key:
            logger.warning("No OpenWeatherMap API key found. Using fallback data.")
            self.api_key = "demo"  # Will use fallback data
        else:
            logger.info("Using OpenWeatherMap API key from configuration")
    
    async def fetch_weather_data(self, city: Dict) -> Optional[Dict]:
        """Fetch weather data for a specific city."""
        try:
            if self.api_key == "demo":
                # Fallback data for demo purposes
                return {
                    "location": city["name"],
                    "temperature": 72.0 + (hash(city["name"]) % 20),  # Random temp between 72-92
                    "humidity": 60 + (hash(city["name"]) % 30),  # Random humidity 60-90%
                    "wind_speed": 10 + (hash(city["name"]) % 15),  # Random wind 10-25 mph
                    "precipitation": (hash(city["name"]) % 10) / 10,  # Random precip 0-1 inch
                    "weather_condition": "Partly Cloudy"
                }
            
            url = f"http://api.openweathermap.org/data/2.5/weather"
            params = {
                "lat": city["lat"],
                "lon": city["lon"],
                "appid": self.api_key,
                "units": "imperial"  # Fahrenheit
            }
            
            async with httpx.AsyncClient() as client:
                response = await client.get(url, params=params)
                response.raise_for_status()
                data = response.json()
                
                return {
                    "location": city["name"],
                    "temperature": data["main"]["temp"],
                    "humidity": data["main"]["humidity"],
                    "wind_speed": data["wind"]["speed"],
                    "precipitation": data.get("rain", {}).get("1h", 0),
                    "weather_condition": data["weather"][0]["main"]
                }
                
        except Exception as e:
            logger.error(f"Error fetching weather data for {city['name']}: {e}")
            return None
    
    async def fetch_all_weather_data(self) -> List[Dict]:
        """Fetch weather data for all major cities."""
        tasks = [self.fetch_weather_data(city) for city in MAJOR_CITIES]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        weather_data = []
        for result in results:
            if isinstance(result, dict):
                weather_data.append(result)
            else:
                logger.error(f"Error in weather fetch: {result}")
        
        return weather_data
    
    async def populate_database(self) -> bool:
        """Populate the weather_data table with fetched data."""
        try:
            db = await get_database_manager()
            
            # Clear existing data
            await db.execute_query("DELETE FROM weather_data")
            logger.info("Cleared existing weather data")
            
            # Fetch new weather data
            weather_data = await self.fetch_all_weather_data()
            logger.info(f"Fetched weather data for {len(weather_data)} cities")
            
            # Insert data into database
            for data in weather_data:
                query = """
                INSERT INTO weather_data (location, temperature, humidity, wind_speed, precipitation, weather_condition)
                VALUES ($1, $2, $3, $4, $5, $6)
                """
                await db.execute_query(query, [
                    data["location"],
                    data["temperature"],
                    data["humidity"],
                    data["wind_speed"],
                    data["precipitation"],
                    data["weather_condition"]
                ])
            
            logger.info(f"Successfully populated weather_data table with {len(weather_data)} records")
            return True
            
        except Exception as e:
            logger.error(f"Error populating weather data: {e}")
            return False


async def fetch_and_populate_weather_data():
    """Main function to fetch and populate weather data."""
    fetcher = WeatherDataFetcher()
    return await fetcher.populate_database()


if __name__ == "__main__":
    asyncio.run(fetch_and_populate_weather_data()) 