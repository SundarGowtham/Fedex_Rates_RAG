"""
Synthesis & Visualization Agent for the Aura Shipping Intelligence Platform.

This agent combines results from all other agents and generates
comprehensive insights, recommendations, and interactive visualizations.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

import plotly.graph_objects as go
import plotly.express as px
from langchain.schema import HumanMessage, SystemMessage
from langchain_ollama import OllamaLLM

from .base import AgentConfig, AgentContext, AgentResponse, BaseAgent, register_agent
from core.state import WorkflowState

logger = logging.getLogger(__name__)


class SynthesisVisualizationAgent(BaseAgent):
    """
    Synthesis & Visualization Agent for insights and visualizations.
    
    This agent combines data from all other agents to generate
    comprehensive insights, recommendations, and interactive charts.
    """
    
    def __init__(self):
        config = AgentConfig(
            name="synthesis_visualization",
            description="Insight synthesis and visualization generation agent",
            timeout_seconds=300,
            max_retries=3,
            enable_caching=True,
            cache_ttl_seconds=3600
        )
        super().__init__(config)
        
        # Initialize Ollama LLM
        self.llm = OllamaLLM(
            model="llama3:8b",
            temperature=0.1,
            base_url="http://localhost:11434"
        )
    
    def get_required_dependencies(self) -> List[str]:
        """Synthesis agent depends on other agents completing first."""
        return ["supervisor"]
    
    async def _execute_agent(self, context: AgentContext, state: WorkflowState) -> AgentResponse:
        """
        Execute the synthesis and visualization agent logic.
        
        This method combines all agent results and generates
        comprehensive insights and visualizations.
        """
        start_time = self._get_current_time()
        
        try:
            # Collect data from all agents
            agent_data = self._collect_agent_data(state)
            
            # Generate insights using LLM
            insights = await self._generate_insights(context, agent_data)
            
            # Generate recommendations
            recommendations = await self._generate_recommendations(context, agent_data)
            
            # Create visualizations
            visualizations = self._create_visualizations(agent_data)
            
            # Create chart data for frontend
            chart_data = self._prepare_chart_data(agent_data)
            
            # Create response data
            response_data = {
                "insights": insights,
                "visualizations": visualizations,
                "recommendations": recommendations,
                "execution_time": self._get_current_time() - start_time,
                "chart_data": chart_data
            }
            
            execution_time = self._get_current_time() - start_time
            
            return AgentResponse(
                success=True,
                data=response_data,
                execution_time=execution_time,
                metadata={
                    "insights_count": len(insights),
                    "recommendations_count": len(recommendations),
                    "visualizations_count": len(visualizations)
                }
            )
            
        except Exception as e:
            logger.error(f"Synthesis and visualization agent execution failed: {e}")
            execution_time = self._get_current_time() - start_time
            
            return AgentResponse(
                success=False,
                error=str(e),
                execution_time=execution_time
            )
    
    def _collect_agent_data(self, state: WorkflowState) -> Dict[str, Any]:
        """
        Collect data from all completed agents.
        """
        agent_data = {
            "user_query": state.user_query,
            "query_type": state.query_type.value if state.query_type else "unknown",
            "structured_data": None,
            "vector_search": None,
            "auxiliary_data": None
        }
        
        # Get structured data results
        if state.structured_data:
            agent_data["structured_data"] = {
                "sql_query": state.structured_data.sql_query,
                "query_results": state.structured_data.query_results,
                "row_count": state.structured_data.row_count
            }
        
        # Get vector search results
        if state.vector_search:
            agent_data["vector_search"] = {
                "query": state.vector_search.query,
                "documents": state.vector_search.documents,
                "scores": state.vector_search.scores
            }
        
        # Get auxiliary data results
        if state.auxiliary_data:
            agent_data["auxiliary_data"] = {
                "weather_data": state.auxiliary_data.weather_data,
                "fuel_prices": state.auxiliary_data.fuel_prices,
                "traffic_data": state.auxiliary_data.traffic_data,
                "sources": state.auxiliary_data.sources
            }
        
        return agent_data
    
    async def _generate_insights(self, context: AgentContext, agent_data: Dict[str, Any]) -> List[str]:
        """
        Generate insights using LLM based on collected data.
        """
        try:
            # Prepare data summary for LLM
            data_summary = self._prepare_data_summary(agent_data)
            
            system_prompt = """
            You are an expert shipping logistics analyst. Based on the provided data, generate 3-5 key insights about shipping operations, costs, and logistics. Focus on actionable insights that would be valuable for shipping decisions.
            
            Format your response as a JSON array of insight strings:
            ["insight 1", "insight 2", "insight 3"]
            """
            
            user_prompt = f"""
            User Query: {context.user_query}
            
            Data Summary:
            {data_summary}
            
            Generate insights based on this data.
            """
            
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]
            
            response = await self.llm.ainvoke(messages)
            
            # Parse JSON response
            try:
                insights = json.loads(response.content)
                if isinstance(insights, list):
                    return insights
            except json.JSONDecodeError:
                pass
            
            # Fallback insights
            return self._generate_fallback_insights(agent_data)
            
        except Exception as e:
            logger.error(f"LLM insight generation failed: {e}")
            return self._generate_fallback_insights(agent_data)
    
    async def _generate_recommendations(self, context: AgentContext, agent_data: Dict[str, Any]) -> List[str]:
        """
        Generate recommendations using LLM based on collected data.
        """
        try:
            # Prepare data summary for LLM
            data_summary = self._prepare_data_summary(agent_data)
            
            system_prompt = """
            You are an expert shipping logistics consultant. Based on the provided data, generate 3-5 actionable recommendations for optimizing shipping operations, reducing costs, or improving efficiency.
            
            Format your response as a JSON array of recommendation strings:
            ["recommendation 1", "recommendation 2", "recommendation 3"]
            """
            
            user_prompt = f"""
            User Query: {context.user_query}
            
            Data Summary:
            {data_summary}
            
            Generate actionable recommendations based on this data.
            """
            
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]
            
            response = await self.llm.ainvoke(messages)
            
            # Parse JSON response
            try:
                recommendations = json.loads(response.content)
                if isinstance(recommendations, list):
                    return recommendations
            except json.JSONDecodeError:
                pass
            
            # Fallback recommendations
            return self._generate_fallback_recommendations(agent_data)
            
        except Exception as e:
            logger.error(f"LLM recommendation generation failed: {e}")
            return self._generate_fallback_recommendations(agent_data)
    
    def _create_visualizations(self, agent_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Create interactive visualizations based on collected data.
        """
        visualizations = []
        
        # Pricing visualization
        if agent_data.get("structured_data") and agent_data["structured_data"].get("query_results"):
            pricing_chart = self._create_pricing_chart(agent_data["structured_data"]["query_results"])
            if pricing_chart:
                visualizations.append(pricing_chart)
        
        # Zone distance visualization
        if agent_data.get("structured_data") and agent_data["structured_data"].get("query_results"):
            zone_chart = self._create_zone_chart(agent_data["structured_data"]["query_results"])
            if zone_chart:
                visualizations.append(zone_chart)
        
        # Weather impact visualization
        if agent_data.get("auxiliary_data") and agent_data["auxiliary_data"].get("weather_data"):
            weather_chart = self._create_weather_chart(agent_data["auxiliary_data"]["weather_data"])
            if weather_chart:
                visualizations.append(weather_chart)
        
        # Fuel price visualization
        if agent_data.get("auxiliary_data") and agent_data["auxiliary_data"].get("fuel_prices"):
            fuel_chart = self._create_fuel_chart(agent_data["auxiliary_data"]["fuel_prices"])
            if fuel_chart:
                visualizations.append(fuel_chart)
        
        return visualizations
    
    def _create_pricing_chart(self, query_results: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """
        Create pricing comparison chart.
        """
        try:
            # Extract pricing data
            pricing_data = []
            for result in query_results:
                if "weight" in result and "price" in result:
                    pricing_data.append({
                        "weight": result["weight"],
                        "price": result["price"],
                        "service_type": result.get("service_type", "Unknown"),
                        "zone": result.get("zone", "Unknown")
                    })
            
            if not pricing_data:
                return None
            
            # Create Plotly chart
            fig = px.scatter(
                pricing_data,
                x="weight",
                y="price",
                color="service_type",
                size="weight",
                hover_data=["zone"],
                title="Shipping Pricing by Weight and Service Type",
                labels={"weight": "Weight (lbs)", "price": "Price ($)"}
            )
            
            return {
                "type": "pricing_chart",
                "title": "Shipping Pricing Analysis",
                "chart": fig.to_dict(),
                "description": "Interactive chart showing shipping costs by weight and service type"
            }
            
        except Exception as e:
            logger.error(f"Failed to create pricing chart: {e}")
            return None
    
    def _create_zone_chart(self, query_results: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """
        Create zone distance chart.
        """
        try:
            # Extract zone data
            zone_data = []
            for result in query_results:
                if "zone" in result and "min_distance" in result and "max_distance" in result:
                    zone_data.append({
                        "zone": result["zone"],
                        "min_distance": result["min_distance"],
                        "max_distance": result["max_distance"],
                        "avg_distance": result.get("avg_distance", (result["min_distance"] + result["max_distance"]) / 2)
                    })
            
            if not zone_data:
                return None
            
            # Create Plotly chart
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                x=[d["zone"] for d in zone_data],
                y=[d["avg_distance"] for d in zone_data],
                name="Average Distance",
                marker_color="lightblue"
            ))
            
            fig.update_layout(
                title="Shipping Zones by Distance",
                xaxis_title="Zone",
                yaxis_title="Distance (miles)",
                showlegend=True
            )
            
            return {
                "type": "zone_chart",
                "title": "Zone Distance Analysis",
                "chart": fig.to_dict(),
                "description": "Chart showing distance ranges for different shipping zones"
            }
            
        except Exception as e:
            logger.error(f"Failed to create zone chart: {e}")
            return None
    
    def _create_weather_chart(self, weather_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Create weather impact chart.
        """
        try:
            if not weather_data:
                return None
            
            # Extract temperature data
            locations = []
            temperatures = []
            
            for location, data in weather_data.items():
                if isinstance(data, dict) and "current" in data:
                    current = data["current"]
                    if "temperature_2m" in current:
                        locations.append(location)
                        temperatures.append(current["temperature_2m"])
            
            if not locations:
                return None
            
            # Create Plotly chart
            fig = px.bar(
                x=locations,
                y=temperatures,
                title="Current Temperature by Location",
                labels={"x": "Location", "y": "Temperature (°F)"},
                color=temperatures,
                color_continuous_scale="RdYlBu_r"
            )
            
            return {
                "type": "weather_chart",
                "title": "Weather Impact Analysis",
                "chart": fig.to_dict(),
                "description": "Current temperature conditions across shipping locations"
            }
            
        except Exception as e:
            logger.error(f"Failed to create weather chart: {e}")
            return None
    
    def _create_fuel_chart(self, fuel_prices: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Create fuel price chart.
        """
        try:
            if not fuel_prices:
                return None
            
            # Extract fuel data
            fuel_types = []
            prices = []
            
            for fuel_type, data in fuel_prices.items():
                if isinstance(data, dict) and "current_price" in data:
                    fuel_types.append(fuel_type.title())
                    prices.append(data["current_price"])
            
            if not fuel_types:
                return None
            
            # Create Plotly chart
            fig = px.pie(
                values=prices,
                names=fuel_types,
                title="Current Fuel Prices",
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            
            return {
                "type": "fuel_chart",
                "title": "Fuel Price Analysis",
                "chart": fig.to_dict(),
                "description": "Current fuel prices by type"
            }
            
        except Exception as e:
            logger.error(f"Failed to create fuel chart: {e}")
            return None
    
    def _prepare_data_summary(self, agent_data: Dict[str, Any]) -> str:
        """
        Prepare a summary of collected data for LLM processing.
        """
        summary_parts = []
        
        # Structured data summary
        if agent_data.get("structured_data"):
            structured = agent_data["structured_data"]
            summary_parts.append(f"Database Results: {structured.get('row_count', 0)} records retrieved")
            if structured.get("query_results"):
                summary_parts.append(f"Key data points: {len(structured['query_results'])} pricing/zone records")
        
        # Vector search summary
        if agent_data.get("vector_search"):
            vector = agent_data["vector_search"]
            summary_parts.append(f"Knowledge Base: {len(vector.get('documents', []))} relevant documents found")
        
        # Auxiliary data summary
        if agent_data.get("auxiliary_data"):
            auxiliary = agent_data["auxiliary_data"]
            if auxiliary.get("weather_data"):
                summary_parts.append(f"Weather Data: {len(auxiliary['weather_data'])} locations")
            if auxiliary.get("fuel_prices"):
                summary_parts.append(f"Fuel Prices: {len(auxiliary['fuel_prices'])} fuel types")
            if auxiliary.get("traffic_data"):
                summary_parts.append(f"Traffic Data: {len(auxiliary['traffic_data'])} locations")
        
        return "\n".join(summary_parts)
    
    def _prepare_chart_data(self, agent_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare chart data for frontend consumption.
        """
        chart_data = {
            "pricing_data": [],
            "zone_data": [],
            "weather_data": [],
            "fuel_data": []
        }
        
        # Extract pricing data
        if agent_data.get("structured_data") and agent_data["structured_data"].get("query_results"):
            for result in agent_data["structured_data"]["query_results"]:
                if "weight" in result and "price" in result:
                    chart_data["pricing_data"].append({
                        "weight": result["weight"],
                        "price": result["price"],
                        "service_type": result.get("service_type", "Unknown"),
                        "zone": result.get("zone", "Unknown")
                    })
        
        # Extract zone data
        if agent_data.get("structured_data") and agent_data["structured_data"].get("query_results"):
            for result in agent_data["structured_data"]["query_results"]:
                if "zone" in result and "min_distance" in result:
                    chart_data["zone_data"].append({
                        "zone": result["zone"],
                        "min_distance": result["min_distance"],
                        "max_distance": result["max_distance"],
                        "avg_distance": result.get("avg_distance", 0)
                    })
        
        # Extract weather data
        if agent_data.get("auxiliary_data") and agent_data["auxiliary_data"].get("weather_data"):
            for location, data in agent_data["auxiliary_data"]["weather_data"].items():
                if isinstance(data, dict) and "current" in data:
                    current = data["current"]
                    chart_data["weather_data"].append({
                        "location": location,
                        "temperature": current.get("temperature_2m", 0),
                        "humidity": current.get("relative_humidity_2m", 0),
                        "wind_speed": current.get("wind_speed_10m", 0)
                    })
        
        # Extract fuel data
        if agent_data.get("auxiliary_data") and agent_data["auxiliary_data"].get("fuel_prices"):
            for fuel_type, data in agent_data["auxiliary_data"]["fuel_prices"].items():
                if isinstance(data, dict) and "current_price" in data:
                    chart_data["fuel_data"].append({
                        "fuel_type": fuel_type,
                        "price": data["current_price"],
                        "trend": data.get("trend", "stable")
                    })
        
        return chart_data
    
    def _generate_fallback_insights(self, agent_data: Dict[str, Any]) -> List[str]:
        """
        Generate fallback insights when LLM fails.
        """
        insights = []
        
        if agent_data.get("structured_data"):
            insights.append("Pricing data available for analysis and comparison")
        
        if agent_data.get("vector_search"):
            insights.append("Knowledge base contains relevant shipping information")
        
        if agent_data.get("auxiliary_data"):
            if agent_data["auxiliary_data"].get("weather_data"):
                insights.append("Weather conditions may impact delivery times")
            if agent_data["auxiliary_data"].get("fuel_prices"):
                insights.append("Fuel prices affect overall shipping costs")
        
        if not insights:
            insights.append("Data analysis completed successfully")
        
        return insights
    
    def _generate_fallback_recommendations(self, agent_data: Dict[str, Any]) -> List[str]:
        """
        Generate fallback recommendations when LLM fails.
        """
        recommendations = []
        
        if agent_data.get("structured_data"):
            recommendations.append("Compare pricing across different service types and zones")
        
        if agent_data.get("auxiliary_data"):
            if agent_data["auxiliary_data"].get("weather_data"):
                recommendations.append("Monitor weather conditions for delivery planning")
            if agent_data["auxiliary_data"].get("fuel_prices"):
                recommendations.append("Consider fuel surcharges in cost calculations")
        
        if not recommendations:
            recommendations.append("Review all available data for optimal shipping decisions")
        
        return recommendations
    
    def _get_current_time(self) -> float:
        """Get current time for execution timing."""
        import time
        return time.time()


# Register the synthesis and visualization agent
synthesis_visualization_agent = SynthesisVisualizationAgent()
register_agent(synthesis_visualization_agent) 