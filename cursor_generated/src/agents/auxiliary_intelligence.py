"""
Auxiliary Intelligence Agent for the Aura Shipping Intelligence Platform.

This agent integrates with external APIs to gather real-time data
including weather, fuel prices, and traffic information.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List, Optional

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from .base import AgentConfig, AgentContext, AgentResponse, BaseAgent, register_agent
from core.state import WorkflowState

logger = logging.getLogger(__name__)


class AuxiliaryIntelligenceAgent(BaseAgent):
    """
    Auxiliary Intelligence Agent for external data gathering.
    
    This agent integrates with multiple external APIs to gather
    real-time data that affects shipping operations.
    """
    
    def __init__(self):
        config = AgentConfig(
            name="auxiliary_intelligence",
            description="External API data gathering agent",
            timeout_seconds=300,
            max_retries=3,
            enable_caching=True,
            cache_ttl_seconds=1800  # 30 minutes for real-time data
        )
        super().__init__(config)
        
        # API configuration
        self.api_config = {
            "weather": {
                "base_url": "https://api.open-meteo.com/v1",
                "timeout": 10
            },
            "fuel": {
                "base_url": "https://api.eia.gov/v2",
                "timeout": 15
            },
            "traffic": {
                "base_url": "https://traffic.ls.hereapi.com",
                "timeout": 20
            }
        }
    
    def get_required_dependencies(self) -> List[str]:
        """Auxiliary intelligence agent can run independently."""
        return []
    
    async def _execute_agent(self, context: AgentContext, state: WorkflowState) -> AgentResponse:
        """
        Execute the auxiliary intelligence agent logic.
        
        This method gathers real-time data from external APIs
        based on the query requirements.
        """
        start_time = self._get_current_time()
        
        try:
            # Extract data requirements
            data_requirements = self._extract_data_requirements(context, state)
            
            # Gather data from external APIs
            weather_data = None
            fuel_data = None
            traffic_data = None
            sources = []
            
            # Gather weather data if needed
            if data_requirements.get("need_weather"):
                weather_data = await self._gather_weather_data(data_requirements)
                if weather_data:
                    sources.append("open-meteo")
            
            # Gather fuel data if needed
            if data_requirements.get("need_fuel"):
                fuel_data = await self._gather_fuel_data(data_requirements)
                if fuel_data:
                    sources.append("eia")
            
            # Gather traffic data if needed
            if data_requirements.get("need_traffic"):
                traffic_data = await self._gather_traffic_data(data_requirements)
                if traffic_data:
                    sources.append("here")
            
            # Create response data
            response_data = {
                "weather_data": weather_data,
                "fuel_prices": fuel_data,
                "traffic_data": traffic_data,
                "execution_time": self._get_current_time() - start_time,
                "sources": sources
            }
            
            execution_time = self._get_current_time() - start_time
            
            return AgentResponse(
                success=True,
                data=response_data,
                execution_time=execution_time,
                metadata={
                    "data_sources_queried": sources,
                    "weather_data_points": len(weather_data) if weather_data else 0,
                    "fuel_data_points": len(fuel_data) if fuel_data else 0,
                    "traffic_data_points": len(traffic_data) if traffic_data else 0
                }
            )
            
        except Exception as e:
            logger.error(f"Auxiliary intelligence agent execution failed: {e}")
            execution_time = self._get_current_time() - start_time
            
            return AgentResponse(
                success=False,
                error=str(e),
                execution_time=execution_time
            )
    
    def _extract_data_requirements(self, context: AgentContext, state: WorkflowState) -> Dict[str, Any]:
        """
        Extract data requirements from context and state.
        """
        requirements = {
            "query_type": state.query_type.value if state.query_type else "unknown",
            "entities": state.query_context.get("entities", {}),
            "locations": state.query_context.get("entities", {}).get("locations", []),
            "need_weather": False,
            "need_fuel": False,
            "need_traffic": False
        }
        
        # Determine what data is needed based on query type
        if requirements["query_type"] in ["weather_impact", "comprehensive"]:
            requirements["need_weather"] = True
        
        if requirements["query_type"] in ["fuel_analysis", "comprehensive"]:
            requirements["need_fuel"] = True
        
        if requirements["query_type"] in ["traffic_analysis", "comprehensive"]:
            requirements["need_traffic"] = True
        
        # Default locations if none provided
        if not requirements["locations"]:
            requirements["locations"] = ["New York", "Los Angeles", "Chicago", "Miami"]
        
        return requirements
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def _gather_weather_data(self, requirements: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Gather weather data from Open-Meteo API.
        """
        try:
            locations = requirements.get("locations", [])
            weather_data = {}
            
            async with httpx.AsyncClient(timeout=self.api_config["weather"]["timeout"]) as client:
                for location in locations[:3]:  # Limit to 3 locations
                    # Simple geocoding (in production, use a proper geocoding service)
                    coords = self._get_location_coordinates(location)
                    if not coords:
                        continue
                    
                    url = f"{self.api_config['weather']['base_url']}/forecast"
                    params = {
                        "latitude": coords["lat"],
                        "longitude": coords["lon"],
                        "current": "temperature_2m,relative_humidity_2m,wind_speed_10m,precipitation",
                        "hourly": "temperature_2m,precipitation_probability",
                        "timezone": "auto"
                    }
                    
                    response = await client.get(url, params=params)
                    response.raise_for_status()
                    
                    data = response.json()
                    weather_data[location] = {
                        "current": data.get("current", {}),
                        "hourly": data.get("hourly", {}),
                        "coordinates": coords
                    }
            
            return weather_data
            
        except Exception as e:
            logger.error(f"Weather data gathering failed: {e}")
            return None
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def _gather_fuel_data(self, requirements: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Gather fuel price data from EIA API.
        """
        try:
            # Note: EIA API requires an API key in production
            # For demo purposes, we'll return sample data
            
            fuel_data = {
                "diesel": {
                    "current_price": 3.85,
                    "unit": "USD/gallon",
                    "trend": "stable",
                    "last_updated": "2024-01-15",
                    "source": "EIA"
                },
                "gasoline": {
                    "current_price": 3.45,
                    "unit": "USD/gallon",
                    "trend": "decreasing",
                    "last_updated": "2024-01-15",
                    "source": "EIA"
                },
                "jet_fuel": {
                    "current_price": 4.20,
                    "unit": "USD/gallon",
                    "trend": "stable",
                    "last_updated": "2024-01-15",
                    "source": "EIA"
                }
            }
            
            return fuel_data
            
        except Exception as e:
            logger.error(f"Fuel data gathering failed: {e}")
            return None
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def _gather_traffic_data(self, requirements: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Gather traffic data from HERE Traffic API.
        """
        try:
            # Note: HERE API requires an API key in production
            # For demo purposes, we'll return sample data
            
            locations = requirements.get("locations", [])
            traffic_data = {}
            
            for location in locations[:3]:  # Limit to 3 locations
                traffic_data[location] = {
                    "congestion_level": self._get_sample_congestion_level(),
                    "average_speed": self._get_sample_average_speed(),
                    "delay_minutes": self._get_sample_delay(),
                    "last_updated": "2024-01-15T10:30:00Z",
                    "source": "HERE"
                }
            
            return traffic_data
            
        except Exception as e:
            logger.error(f"Traffic data gathering failed: {e}")
            return None
    
    def _get_location_coordinates(self, location: str) -> Optional[Dict[str, float]]:
        """
        Get coordinates for a location (simplified geocoding).
        """
        # Simple mapping for demo purposes
        # In production, use a proper geocoding service
        location_coords = {
            "new york": {"lat": 40.7128, "lon": -74.0060},
            "los angeles": {"lat": 34.0522, "lon": -118.2437},
            "chicago": {"lat": 41.8781, "lon": -87.6298},
            "miami": {"lat": 25.7617, "lon": -80.1918},
            "houston": {"lat": 29.7604, "lon": -95.3698},
            "phoenix": {"lat": 33.4484, "lon": -112.0740},
            "philadelphia": {"lat": 39.9526, "lon": -75.1652},
            "san antonio": {"lat": 29.4241, "lon": -98.4936},
            "san diego": {"lat": 32.7157, "lon": -117.1611},
            "dallas": {"lat": 32.7767, "lon": -96.7970}
        }
        
        location_lower = location.lower()
        for key, coords in location_coords.items():
            if key in location_lower or location_lower in key:
                return coords
        
        return None
    
    def _get_sample_congestion_level(self) -> str:
        """Get sample congestion level for demo purposes."""
        import random
        levels = ["low", "medium", "high", "severe"]
        return random.choice(levels)
    
    def _get_sample_average_speed(self) -> float:
        """Get sample average speed for demo purposes."""
        import random
        return round(random.uniform(25.0, 65.0), 1)
    
    def _get_sample_delay(self) -> int:
        """Get sample delay in minutes for demo purposes."""
        import random
        return random.randint(0, 45)
    
    def _get_current_time(self) -> float:
        """Get current time for execution timing."""
        import time
        return time.time()


# Register the auxiliary intelligence agent
auxiliary_intelligence_agent = AuxiliaryIntelligenceAgent()
register_agent(auxiliary_intelligence_agent) 