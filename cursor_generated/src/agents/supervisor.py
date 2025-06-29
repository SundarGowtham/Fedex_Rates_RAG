"""
Supervisor Agent for the Aura Shipping Intelligence Platform.

This agent is responsible for query deconstruction, task routing,
and overall workflow coordination.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional

from langchain.schema import HumanMessage, SystemMessage
from langchain_ollama import OllamaLLM

from .base import AgentConfig, AgentContext, AgentResponse, BaseAgent, register_agent
from src.core.state import QueryType, WorkflowState

logger = logging.getLogger(__name__)


class SupervisorAgent(BaseAgent):
    """
    Supervisor Agent for query analysis and task routing.
    
    This agent analyzes user queries to determine:
    - Query type and intent
    - Required data sources
    - Agent execution order
    - Context enrichment
    """
    
    def __init__(self):
        config = AgentConfig(
            name="supervisor",
            description="Query deconstruction and task routing agent",
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
        """Supervisor has no dependencies - it's the entry point."""
        return []
    
    async def _execute_agent(self, context: AgentContext, state: WorkflowState) -> AgentResponse:
        """
        Execute the supervisor agent logic.
        
        This method analyzes the user query and determines the appropriate
        workflow routing and context enrichment.
        """
        start_time = self._get_current_time()
        
        try:
            # Analyze the query
            query_analysis = await self._analyze_query(context.user_query)
            
            # Extract entities and context
            entities = await self._extract_entities(context.user_query)
            
            # Determine query type
            query_type = self._determine_query_type(context.user_query, query_analysis)
            
            # Update state with analysis results
            state.query_type = query_type
            state.query_context.update({
                "analysis": query_analysis,
                "entities": entities,
                "intent": query_analysis.get("intent"),
                "complexity": query_analysis.get("complexity")
            })
            
            # Create response data
            response_data = {
                "query_type": query_type.value,
                "analysis": query_analysis,
                "entities": entities,
                "routing_plan": self._create_routing_plan(query_type, query_analysis),
                "context_enrichment": self._enrich_context(entities, query_analysis)
            }
            
            execution_time = self._get_current_time() - start_time
            
            return AgentResponse(
                success=True,
                data=response_data,
                execution_time=execution_time,
                metadata={
                    "query_type": query_type.value,
                    "entities_count": len(entities),
                    "analysis_confidence": query_analysis.get("confidence", 0.0)
                }
            )
            
        except Exception as e:
            logger.error(f"Supervisor agent execution failed: {e}")
            execution_time = self._get_current_time() - start_time
            
            return AgentResponse(
                success=False,
                error=str(e),
                execution_time=execution_time
            )
    
    async def _analyze_query(self, user_query: str) -> Dict[str, Any]:
        """
        Analyze the user query to understand intent and complexity.
        
        This method uses the LLM to perform semantic analysis of the query
        and extract key information for routing decisions.
        """
        system_prompt = """
You are an expert shipping logistics analyst. Analyze the given query to understand:
1. The primary intent (pricing, routing, weather impact, fuel analysis, traffic analysis, or comprehensive)
2. The complexity level (simple, moderate, complex)
3. Required data sources
4. Confidence level in your analysis (0.0 to 1.0)

You MUST respond with ONLY a valid JSON object in this exact format:
{
    "intent": "primary_intent",
    "complexity": "complexity_level", 
    "data_sources": ["source1", "source2"],
    "confidence": 0.85,
    "keywords": ["keyword1", "keyword2"],
    "summary": "brief_summary"
}

Do not include any other text, only the JSON object.
"""
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Analyze this shipping query: {user_query}")
        ]
        
        try:
            response = await self.llm.ainvoke(messages)
            # Parse the response (OllamaLLM returns string directly)
            import json
            if hasattr(response, 'content'):
                # Handle LangChain message objects
                response_text = response.content
            else:
                # Handle direct string responses from OllamaLLM
                response_text = str(response)
            
            # Clean up the response text
            response_text = response_text.strip()
            
            # Try to extract JSON from the response
            try:
                analysis = json.loads(response_text)
            except json.JSONDecodeError:
                # Try to find JSON in the response
                import re
                json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                if json_match:
                    try:
                        analysis = json.loads(json_match.group(0))
                    except json.JSONDecodeError:
                        logger.warning(f"Could not parse JSON from response: {response_text}")
                        return self._fallback_analysis(user_query)
                else:
                    logger.warning(f"No JSON found in response: {response_text}")
                    return self._fallback_analysis(user_query)
            
            # Validate the analysis structure
            required_keys = ["intent", "complexity", "data_sources", "confidence", "keywords", "summary"]
            if not all(key in analysis for key in required_keys):
                logger.warning(f"Missing required keys in analysis: {analysis}")
                return self._fallback_analysis(user_query)
            
            return analysis
        except Exception as e:
            logger.warning(f"LLM analysis failed, using fallback: {e}")
            return self._fallback_analysis(user_query)
    
    def _fallback_analysis(self, user_query: str) -> Dict[str, Any]:
        """
        Fallback analysis using rule-based approach when LLM fails.
        """
        query_lower = user_query.lower()
        
        # Determine intent based on keywords
        intent = "comprehensive"
        if any(word in query_lower for word in ["rate", "price", "cost", "fee"]):
            intent = "pricing"
        elif any(word in query_lower for word in ["route", "path", "way", "optimize"]):
            intent = "route_optimization"
        elif any(word in query_lower for word in ["weather", "climate", "storm", "rain"]):
            intent = "weather_impact"
        elif any(word in query_lower for word in ["fuel", "gas", "diesel", "energy"]):
            intent = "fuel_analysis"
        elif any(word in query_lower for word in ["traffic", "congestion", "delay"]):
            intent = "traffic_analysis"
        
        # Determine complexity
        complexity = "simple"
        if len(query_lower.split()) > 15 or any(word in query_lower for word in ["comprehensive", "analysis", "detailed"]):
            complexity = "complex"
        elif len(query_lower.split()) > 8:
            complexity = "moderate"
        
        return {
            "intent": intent,
            "complexity": complexity,
            "data_sources": ["fedex_pricing", "weather", "fuel", "traffic"],
            "confidence": 0.7,
            "keywords": self._extract_keywords(query_lower),
            "summary": f"Query about {intent} with {complexity} complexity"
        }
    
    async def _extract_entities(self, user_query: str) -> Dict[str, Any]:
        """
        Extract entities from the user query.
        
        This method identifies key entities like locations, weights,
        service types, and other relevant information.
        """
        entities = {
            "locations": [],
            "weights": [],
            "service_types": [],
            "carriers": [],
            "dates": [],
            "zones": []
        }
        
        # Extract weights (e.g., "5-pound", "10 lb", "2kg")
        weight_patterns = [
            r'(\d+(?:\.\d+)?)\s*(?:pound|lb|kg|gram|g)s?',
            r'(\d+(?:\.\d+)?)\s*(?:pound|lb|kg|gram|g)'
        ]
        
        for pattern in weight_patterns:
            matches = re.findall(pattern, user_query, re.IGNORECASE)
            entities["weights"].extend([float(match) for match in matches])
        
        # Extract service types
        service_keywords = ["ground", "express", "priority", "overnight", "standard"]
        for service in service_keywords:
            if service in user_query.lower():
                entities["service_types"].append(service)
        
        # Extract carriers
        carrier_keywords = ["fedex", "ups", "usps", "dhl"]
        for carrier in carrier_keywords:
            if carrier in user_query.lower():
                entities["carriers"].append(carrier)
        
        # Extract locations (simple pattern matching)
        location_patterns = [
            r'from\s+([A-Za-z\s]+?)\s+to\s+([A-Za-z\s]+)',
            r'between\s+([A-Za-z\s]+?)\s+and\s+([A-Za-z\s]+)',
            r'([A-Za-z\s]+?)\s+to\s+([A-Za-z\s]+)'
        ]
        
        for pattern in location_patterns:
            matches = re.findall(pattern, user_query, re.IGNORECASE)
            for match in matches:
                if isinstance(match, tuple):
                    entities["locations"].extend([loc.strip() for loc in match])
                else:
                    entities["locations"].append(match.strip())
        
        # Extract zones
        zone_pattern = r'zone\s*(\d+)'
        zone_matches = re.findall(zone_pattern, user_query, re.IGNORECASE)
        entities["zones"].extend(zone_matches)
        
        return entities
    
    def _determine_query_type(self, user_query: str, analysis: Dict[str, Any]) -> QueryType:
        """
        Determine the query type based on analysis and keywords.
        """
        intent = analysis.get("intent", "").lower()
        
        # Map intent to query type
        intent_mapping = {
            "pricing": QueryType.PRICING,
            "route_optimization": QueryType.ROUTE_OPTIMIZATION,
            "weather_impact": QueryType.WEATHER_IMPACT,
            "fuel_analysis": QueryType.FUEL_ANALYSIS,
            "traffic_analysis": QueryType.TRAFFIC_ANALYSIS,
            "comprehensive": QueryType.COMPREHENSIVE
        }
        
        return intent_mapping.get(intent, QueryType.COMPREHENSIVE)
    
    def _create_routing_plan(self, query_type: QueryType, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a routing plan for agent execution.
        """
        routing_plan = {
            "required_agents": [],
            "optional_agents": [],
            "execution_order": [],
            "parallel_execution": []
        }
        
        # Determine required agents based on query type
        if query_type in [QueryType.PRICING, QueryType.COMPREHENSIVE]:
            routing_plan["required_agents"].append("structured_data")
        
        if query_type in [QueryType.ROUTE_OPTIMIZATION, QueryType.COMPREHENSIVE]:
            routing_plan["required_agents"].append("vector_search")
        
        if query_type in [QueryType.WEATHER_IMPACT, QueryType.FUEL_ANALYSIS, 
                         QueryType.TRAFFIC_ANALYSIS, QueryType.COMPREHENSIVE]:
            routing_plan["required_agents"].append("auxiliary_intelligence")
        
        # Synthesis is always required
        routing_plan["required_agents"].append("synthesis_visualization")
        
        # Determine execution order
        routing_plan["execution_order"] = [
            "supervisor",
            *routing_plan["required_agents"]
        ]
        
        # Determine parallel execution opportunities
        parallel_groups = []
        if "structured_data" in routing_plan["required_agents"] and "vector_search" in routing_plan["required_agents"]:
            parallel_groups.append(["structured_data", "vector_search"])
        
        if "auxiliary_intelligence" in routing_plan["required_agents"]:
            if parallel_groups:
                parallel_groups[0].append("auxiliary_intelligence")
            else:
                parallel_groups.append(["auxiliary_intelligence"])
        
        routing_plan["parallel_execution"] = parallel_groups
        
        return routing_plan
    
    def _enrich_context(self, entities: Dict[str, Any], analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enrich the context with extracted information.
        """
        context = {
            "entities": entities,
            "analysis": analysis,
            "data_requirements": self._determine_data_requirements(entities, analysis),
            "priority": self._determine_priority(analysis),
            "estimated_complexity": analysis.get("complexity", "simple")
        }
        
        return context
    
    def _determine_data_requirements(self, entities: Dict[str, Any], analysis: Dict[str, Any]) -> List[str]:
        """
        Determine what data sources are required.
        """
        requirements = []
        
        # Always need Fedex pricing for shipping queries
        if entities.get("weights") or entities.get("service_types"):
            requirements.append("fedex_pricing")
        
        # Need zone mapping for location-based queries
        if entities.get("locations") or entities.get("zones"):
            requirements.append("zone_distance_mapping")
        
        # Weather data for weather impact queries
        if analysis.get("intent") == "weather_impact":
            requirements.append("weather_data")
        
        # Fuel data for fuel analysis
        if analysis.get("intent") == "fuel_analysis":
            requirements.append("fuel_prices")
        
        # Traffic data for traffic analysis
        if analysis.get("intent") == "traffic_analysis":
            requirements.append("traffic_data")
        
        return requirements
    
    def _determine_priority(self, analysis: Dict[str, Any]) -> str:
        """
        Determine the priority level of the query.
        """
        complexity = analysis.get("complexity", "simple")
        confidence = analysis.get("confidence", 0.5)
        
        if complexity == "complex" and confidence > 0.8:
            return "high"
        elif complexity == "moderate" or confidence > 0.6:
            return "medium"
        else:
            return "low"
    
    def _extract_keywords(self, query: str) -> List[str]:
        """
        Extract keywords from the query.
        """
        # Simple keyword extraction
        keywords = []
        shipping_keywords = [
            "rate", "price", "cost", "shipping", "delivery", "package",
            "weight", "zone", "service", "ground", "express", "priority",
            "weather", "fuel", "traffic", "route", "optimization"
        ]
        
        for keyword in shipping_keywords:
            if keyword in query:
                keywords.append(keyword)
        
        return keywords
    
    def _get_current_time(self) -> float:
        """Get current time for execution timing."""
        import time
        return time.time()


# Register the supervisor agent
supervisor_agent = SupervisorAgent()
register_agent(supervisor_agent) 