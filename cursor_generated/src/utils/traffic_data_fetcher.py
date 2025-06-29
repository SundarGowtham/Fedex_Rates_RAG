"""
Traffic data fetcher using Google Maps API.
Fetches real traffic data for major routes and populates the traffic_data table.
"""

import asyncio
import logging
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import httpx
import pandas as pd

from src.core.database import get_database_manager

logger = logging.getLogger(__name__)

# Major routes for traffic data
MAJOR_ROUTES = [
    {
        "route": "I-95 New York to Boston",
        "origin": "New York, NY",
        "destination": "Boston, MA",
        "distance_miles": 215
    },
    {
        "route": "I-5 Los Angeles to San Francisco", 
        "origin": "Los Angeles, CA",
        "destination": "San Francisco, CA",
        "distance_miles": 382
    },
    {
        "route": "I-90 Chicago to Cleveland",
        "origin": "Chicago, IL", 
        "destination": "Cleveland, OH",
        "distance_miles": 344
    },
    {
        "route": "I-10 Houston to New Orleans",
        "origin": "Houston, TX",
        "destination": "New Orleans, LA", 
        "distance_miles": 350
    },
    {
        "route": "I-85 Atlanta to Charlotte",
        "origin": "Atlanta, GA",
        "destination": "Charlotte, NC",
        "distance_miles": 245
    },
    {
        "route": "I-80 San Francisco to Sacramento",
        "origin": "San Francisco, CA",
        "destination": "Sacramento, CA",
        "distance_miles": 88
    },
    {
        "route": "I-35 Dallas to Austin",
        "origin": "Dallas, TX",
        "destination": "Austin, TX",
        "distance_miles": 195
    },
    {
        "route": "I-75 Atlanta to Tampa",
        "origin": "Atlanta, GA",
        "destination": "Tampa, FL",
        "distance_miles": 456
    }
]

class TrafficDataFetcher:
    """Fetches traffic data from Google Maps API."""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("GOOGLE_MAPS_API_KEY")
        if not self.api_key:
            logger.warning("No Google Maps API key found. Using fallback data.")
            self.api_key = "demo"  # Will use fallback data
    
    async def fetch_traffic_data(self, route: Dict) -> Optional[Dict]:
        """Fetch traffic data for a specific route."""
        try:
            if self.api_key == "demo":
                # Fallback data for demo purposes
                base_speed = 65  # Base speed in mph
                route_factor = hash(route["route"]) % 20  # Route-specific variation
                time_factor = datetime.now().hour % 24  # Time-based variation
                
                # Simulate traffic patterns
                if 7 <= time_factor <= 9 or 16 <= time_factor <= 18:  # Rush hour
                    speed = base_speed - 25 - route_factor
                    congestion = "High"
                    delay = 15 + route_factor
                elif 10 <= time_factor <= 15:  # Mid-day
                    speed = base_speed - 10 - route_factor
                    congestion = "Medium"
                    delay = 5 + route_factor
                else:  # Off-peak
                    speed = base_speed - 5 - route_factor
                    congestion = "Low"
                    delay = 2 + route_factor
                
                return {
                    "route": route["route"],
                    "congestion_level": congestion,
                    "average_speed": max(speed, 20),  # Minimum 20 mph
                    "delay_minutes": delay
                }
            
            # Google Maps Directions API
            url = "https://maps.googleapis.com/maps/api/directions/json"
            params = {
                "origin": route["origin"],
                "destination": route["destination"],
                "key": self.api_key,
                "departure_time": "now",
                "traffic_model": "best_guess"
            }
            
            async with httpx.AsyncClient() as client:
                response = await client.get(url, params=params)
                response.raise_for_status()
                data = response.json()
                
                if data["status"] == "OK" and data["routes"]:
                    route_data = data["routes"][0]
                    leg = route_data["legs"][0]
                    
                    # Calculate traffic metrics
                    duration_in_traffic = leg.get("duration_in_traffic", leg["duration"])
                    duration_no_traffic = leg["duration"]
                    
                    # Convert to minutes
                    delay_minutes = (duration_in_traffic["value"] - duration_no_traffic["value"]) // 60
                    
                    # Calculate average speed
                    distance_meters = leg["distance"]["value"]
                    duration_hours = duration_in_traffic["value"] / 3600
                    average_speed = distance_meters / 1609.34 / duration_hours  # Convert to mph
                    
                    # Determine congestion level
                    if delay_minutes > 15:
                        congestion_level = "High"
                    elif delay_minutes > 5:
                        congestion_level = "Medium"
                    else:
                        congestion_level = "Low"
                    
                    return {
                        "route": route["route"],
                        "congestion_level": congestion_level,
                        "average_speed": round(average_speed, 1),
                        "delay_minutes": delay_minutes
                    }
                
                return None
                
        except Exception as e:
            logger.error(f"Error fetching traffic data for {route['route']}: {e}")
            return None
    
    async def fetch_all_traffic_data(self) -> List[Dict]:
        """Fetch traffic data for all major routes."""
        tasks = [self.fetch_traffic_data(route) for route in MAJOR_ROUTES]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        traffic_data = []
        for result in results:
            if isinstance(result, dict):
                traffic_data.append(result)
            else:
                logger.error(f"Error in traffic fetch: {result}")
        
        return traffic_data
    
    async def populate_database(self) -> bool:
        """Populate the traffic_data table with fetched data."""
        try:
            db = await get_database_manager()
            
            # Clear existing data
            await db.execute_query("DELETE FROM traffic_data")
            logger.info("Cleared existing traffic data")
            
            # Fetch new traffic data
            traffic_data = await self.fetch_all_traffic_data()
            logger.info(f"Fetched traffic data for {len(traffic_data)} routes")
            
            # Insert data into database
            for data in traffic_data:
                query = """
                INSERT INTO traffic_data (route, congestion_level, average_speed, delay_minutes)
                VALUES ($1, $2, $3, $4)
                """
                await db.execute_query(query, [
                    data["route"],
                    data["congestion_level"],
                    data["average_speed"],
                    data["delay_minutes"]
                ])
            
            logger.info(f"Successfully populated traffic_data table with {len(traffic_data)} records")
            return True
            
        except Exception as e:
            logger.error(f"Error populating traffic data: {e}")
            return False


async def fetch_and_populate_traffic_data():
    """Main function to fetch and populate traffic data."""
    fetcher = TrafficDataFetcher()
    return await fetcher.populate_database()


if __name__ == "__main__":
    asyncio.run(fetch_and_populate_traffic_data()) 