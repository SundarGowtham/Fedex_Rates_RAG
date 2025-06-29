"""
Fuel price data fetcher using EIA API.
Fetches real fuel price data and populates the fuel_prices table.
"""

import asyncio
import logging
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any

import httpx
import pandas as pd
from sqlalchemy import select, delete, text

from src.core.database import get_database_manager, FuelPrices

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

# Fuel types and regions
FUEL_TYPES = ["regular", "midgrade", "premium", "diesel"]
REGIONS = [
    "East Coast", "New England", "Central Atlantic", "Lower Atlantic",
    "Midwest", "Gulf Coast", "Rocky Mountain", "West Coast"
]

class FuelDataFetcher:
    """Fetches fuel price data from EIA API."""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("EIA_API_KEY")
        
        if not self.api_key:
            logger.warning("No EIA API key found. Using fallback data.")
            self.api_key = "demo"  # Will use fallback data
        else:
            logger.info("Using EIA API key from configuration")
    
    def _get_fallback_data(self):
        """Return fallback fuel price data when API is unavailable."""
        return [
            {"fuel_type": "regular", "price_per_gallon": 3.50, "region": "National Average"},
            {"fuel_type": "midgrade", "price_per_gallon": 4.02, "region": "National Average"},
            {"fuel_type": "premium", "price_per_gallon": 4.55, "region": "National Average"},
            {"fuel_type": "diesel", "price_per_gallon": 4.20, "region": "National Average"}
        ]
    
    async def fetch_fuel_prices(self) -> List[Dict[str, Any]]:
        """Fetch current fuel prices from EIA API"""
        if not self.api_key:
            print("No EIA API key provided, using fallback data")
            return self._get_fallback_data()
        
        try:
            # Try the working endpoint format but with parameters to get price values
            url = "https://api.eia.gov/v2/petroleum/pri/spt/data/"
            params = {
                "api_key": self.api_key,
                "frequency": "weekly",
                "data[]": "value",
                "offset": 0,
                "length": 5  # Get a few recent entries
            }
            
            async with httpx.AsyncClient() as client:
                response = await client.get(url, params=params, timeout=10.0)
                response.raise_for_status()
                data = response.json()
                
                print(f"EIA API response: {data}")
                
                if data["response"]["data"] and len(data["response"]["data"]) > 0:
                    # Look for any entry with a value field
                    entries_with_values = [item for item in data["response"]["data"] if item.get("value")]
                    
                    if entries_with_values:
                        # Use the first entry with a value
                        latest_entry = entries_with_values[0]
                        latest_price = float(latest_entry["value"])  # Convert string to float
                        
                        print(f"Found price data: {latest_entry}")
                        print(f"Using price: ${latest_price} per gallon")
                        
                        # Create fuel data with estimated prices for other types
                        fuel_data = [
                            {
                                "fuel_type": "regular",
                                "price_per_gallon": round(latest_price * 0.85, 2),  # Gasoline typically cheaper than diesel
                                "region": "National Average"
                            },
                            {
                                "fuel_type": "midgrade", 
                                "price_per_gallon": round(latest_price * 0.98, 2),  # Close to diesel price
                                "region": "National Average"
                            },
                            {
                                "fuel_type": "premium",
                                "price_per_gallon": round(latest_price * 1.10, 2),  # Premium more than diesel
                                "region": "National Average"
                            },
                            {
                                "fuel_type": "diesel",
                                "price_per_gallon": latest_price,  # Use the actual diesel price
                                "region": "National Average"
                            }
                        ]
                    else:
                        print("No entries with price values found, using fallback data")
                        fuel_data = self._get_fallback_data()
                else:
                    print("No price data available from EIA API, using fallback data")
                    fuel_data = self._get_fallback_data()
            
            return fuel_data
            
        except Exception as e:
            print(f"Error fetching fuel price data: {e}")
            return self._get_fallback_data()
    
    async def populate_database(self) -> bool:
        """Populate the fuel_prices table with fetched data."""
        try:
            db = await get_database_manager()
            
            if db.async_session_factory is None:
                await db.initialize_async()
            
            # Clear existing data
            async with db.async_session_factory() as session:
                await session.execute("DELETE FROM fuel_prices")
                logger.info("Cleared existing fuel price data")
            
            # Fetch new fuel price data
            fuel_data = await self.fetch_fuel_prices()
            logger.info(f"Fetched fuel price data for {len(fuel_data)} records")
            
            # Insert data into database
            async with db.async_session_factory() as session:
                for data in fuel_data:
                    query = """
                    INSERT INTO fuel_prices (fuel_type, price_per_gallon, region)
                    VALUES (:fuel_type, :price_per_gallon, :region)
                    """
                    await session.execute(text(query), {
                        "fuel_type": data["fuel_type"],
                        "price_per_gallon": data["price_per_gallon"],
                        "region": data["region"]
                    })
            
            logger.info(f"Successfully populated fuel_prices table with {len(fuel_data)} records")
            return True
            
        except Exception as e:
            logger.error(f"Error populating fuel price data: {e}")
            return False


async def fetch_and_populate_fuel_data():
    """Fetch fuel prices and populate the database, only clearing if table has 4 sample rows."""
    db = await get_database_manager()
    if db.async_session_factory is None:
        await db.initialize_async()
    
    # Check current rows
    async with db.async_session_factory() as session:
        result = await session.execute(select(FuelPrices))
        rows = result.scalars().all()
        sample_set = [
            {"fuel_type": "regular", "price_per_gallon": 3.50, "region": "National Average"},
            {"fuel_type": "midgrade", "price_per_gallon": 4.02, "region": "National Average"},
            {"fuel_type": "premium", "price_per_gallon": 4.55, "region": "National Average"},
            {"fuel_type": "diesel", "price_per_gallon": 4.20, "region": "National Average"}
        ]
        def is_sample_row(row):
            for s in sample_set:
                if (
                    row.fuel_type == s["fuel_type"] and
                    abs(row.price_per_gallon - s["price_per_gallon"]) < 0.01 and
                    row.region == s["region"]
                ):
                    return True
            return False
        if len(rows) == 4 and all(is_sample_row(row) for row in rows):
            await session.execute(text("DELETE FROM fuel_prices"))
            await session.commit()
            print("Cleared sample data from fuel_prices table.")
        else:
            print(f"fuel_prices table has {len(rows)} rows; not clearing.")
    
    # Now fetch and insert new data using the working raw SQL approach
    fetcher = FuelDataFetcher()
    fuel_data = await fetcher.fetch_fuel_prices()
    
    async with db.async_session_factory() as session:
        for data in fuel_data:
            query = """
            INSERT INTO fuel_prices (fuel_type, price_per_gallon, region)
            VALUES (:fuel_type, :price_per_gallon, :region)
            """
            await session.execute(text(query), {
                "fuel_type": data["fuel_type"],
                "price_per_gallon": data["price_per_gallon"],
                "region": data["region"]
            })
        await session.commit()
    
    print(f"Inserted {len(fuel_data)} new fuel price records.")
    return True


if __name__ == "__main__":
    asyncio.run(fetch_and_populate_fuel_data()) 