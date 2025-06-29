"""
Vector Search Agent for the Aura Shipping Intelligence Platform.

This agent uses SentenceTransformers and Qdrant for semantic search
of shipping documents and knowledge base.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from sentence_transformers import SentenceTransformer

from .base import AgentConfig, AgentContext, AgentResponse, BaseAgent, register_agent
from src.core.state import WorkflowState

logger = logging.getLogger(__name__)


class VectorSearchAgent(BaseAgent):
    """
    Vector Search Agent for semantic search operations.
    
    This agent uses SentenceTransformers to generate embeddings and
    Qdrant vector database for semantic search of shipping-related
    documents and knowledge.
    """
    
    def __init__(self):
        config = AgentConfig(
            name="vector_search",
            description="Semantic search and vector operations agent",
            timeout_seconds=300,
            max_retries=3,
            enable_caching=True,
            cache_ttl_seconds=3600
        )
        super().__init__(config)
        
        # Initialize SentenceTransformer
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Initialize Qdrant client
        self.qdrant_client = QdrantClient(
            host="localhost",
            port=6333
        )
        
        # Collection name for shipping documents
        self.collection_name = "aura_shipping_documents"
        
        # Initialize collection if it doesn't exist
        self._initialize_collection()
    
    def get_required_dependencies(self) -> List[str]:
        """Vector search agent can run independently."""
        return []
    
    async def _execute_agent(self, context: AgentContext, state: WorkflowState) -> AgentResponse:
        """
        Execute the vector search agent logic.
        
        This method performs semantic search on shipping documents
        and knowledge base to find relevant information.
        """
        start_time = self._get_current_time()
        
        try:
            # Extract search requirements
            search_requirements = self._extract_search_requirements(context, state)
            
            # Generate search queries
            search_queries = self._generate_search_queries(search_requirements)
            
            # Perform vector search
            search_results = []
            for query in search_queries:
                results = await self._perform_vector_search(query, search_requirements)
                search_results.extend(results)
            
            # Rank and filter results
            ranked_results = self._rank_results(search_results, search_requirements)
            
            # Create response data
            response_data = {
                "query": search_requirements.get("primary_query", context.user_query),
                "documents": ranked_results,
                "scores": [doc.get("score", 0.0) for doc in ranked_results],
                "metadata": {
                    "total_queries": len(search_queries),
                    "total_results": len(search_results),
                    "filtered_results": len(ranked_results),
                    "search_requirements": search_requirements
                },
                "execution_time": self._get_current_time() - start_time
            }
            
            execution_time = self._get_current_time() - start_time
            
            return AgentResponse(
                success=True,
                data=response_data,
                execution_time=execution_time,
                metadata={
                    "search_queries_count": len(search_queries),
                    "total_documents_found": len(search_results),
                    "relevant_documents": len(ranked_results)
                }
            )
            
        except Exception as e:
            logger.error(f"Vector search agent execution failed: {e}")
            execution_time = self._get_current_time() - start_time
            
            return AgentResponse(
                success=False,
                error=str(e),
                execution_time=execution_time
            )
    
    def _extract_search_requirements(self, context: AgentContext, state: WorkflowState) -> Dict[str, Any]:
        """
        Extract search requirements from context and state.
        """
        requirements = {
            "primary_query": context.user_query,
            "query_type": state.query_type.value if state.query_type else "unknown",
            "entities": state.query_context.get("entities", {}),
            "max_results": 20,
            "min_score": 0.3
        }
        
        # Extract specific search requirements based on query type
        if requirements["query_type"] == "route_optimization":
            requirements.update({
                "search_topics": ["route optimization", "shipping routes", "logistics planning"],
                "focus_areas": ["distance", "time", "efficiency"]
            })
        
        elif requirements["query_type"] == "pricing":
            requirements.update({
                "search_topics": ["pricing", "rates", "costs", "fees"],
                "focus_areas": ["pricing structure", "rate calculation"]
            })
        
        elif requirements["query_type"] == "weather_impact":
            requirements.update({
                "search_topics": ["weather impact", "climate effects", "delivery delays"],
                "focus_areas": ["weather conditions", "delivery times"]
            })
        
        elif requirements["query_type"] == "comprehensive":
            requirements.update({
                "search_topics": ["shipping", "logistics", "delivery", "transportation"],
                "focus_areas": ["comprehensive analysis", "multi-factor analysis"]
            })
        
        return requirements
    
    def _generate_search_queries(self, requirements: Dict[str, Any]) -> List[str]:
        """
        Generate search queries based on requirements.
        """
        queries = [requirements["primary_query"]]
        
        # Add topic-specific queries
        search_topics = requirements.get("search_topics", [])
        for topic in search_topics:
            queries.append(f"{requirements['primary_query']} {topic}")
        
        # Add entity-based queries
        entities = requirements.get("entities", {})
        
        if entities.get("weights"):
            for weight in entities["weights"]:
                queries.append(f"shipping rates for {weight} pound package")
        
        if entities.get("service_types"):
            for service in entities["service_types"]:
                queries.append(f"{service} shipping service information")
        
        if entities.get("locations"):
            for location in entities["locations"]:
                queries.append(f"shipping to {location}")
        
        # Remove duplicates and limit
        unique_queries = list(set(queries))
        return unique_queries[:5]  # Limit to 5 queries
    
    async def _perform_vector_search(self, query: str, requirements: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Perform vector search for a given query.
        """
        try:
            # Generate query embedding
            query_embedding = self.embedding_model.encode(query)
            
            # Search in Qdrant
            search_results = self.qdrant_client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding.tolist(),
                limit=requirements.get("max_results", 20),
                score_threshold=requirements.get("min_score", 0.3)
            )
            
            # Convert to standard format
            results = []
            for result in search_results:
                results.append({
                    "id": result.id,
                    "score": result.score,
                    "payload": result.payload,
                    "query": query
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Vector search failed for query '{query}': {e}")
            return []
    
    def _rank_results(self, search_results: List[Dict[str, Any]], requirements: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Rank and filter search results.
        """
        if not search_results:
            return []
        
        # Sort by score (descending)
        ranked_results = sorted(search_results, key=lambda x: x.get("score", 0.0), reverse=True)
        
        # Apply additional filtering based on requirements
        filtered_results = []
        focus_areas = requirements.get("focus_areas", [])
        
        for result in ranked_results:
            payload = result.get("payload", {})
            content = payload.get("content", "").lower()
            
            # Check if result matches focus areas
            if focus_areas:
                matches_focus = any(area.lower() in content for area in focus_areas)
                if matches_focus:
                    filtered_results.append(result)
            else:
                filtered_results.append(result)
        
        # If no focus area matches, return top results
        if not filtered_results and ranked_results:
            filtered_results = ranked_results[:10]
        
        return filtered_results
    
    def _initialize_collection(self) -> None:
        """
        Initialize the Qdrant collection for shipping documents.
        """
        try:
            # Check if collection exists
            collections = self.qdrant_client.get_collections()
            collection_names = [col.name for col in collections.collections]
            
            if self.collection_name not in collection_names:
                # Create collection
                self.qdrant_client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=384,  # all-MiniLM-L6-v2 embedding size
                        distance=Distance.COSINE
                    )
                )
                
                # Add sample documents
                self._add_sample_documents()
                
                logger.info(f"Created Qdrant collection: {self.collection_name}")
            else:
                logger.info(f"Using existing Qdrant collection: {self.collection_name}")
                
        except Exception as e:
            logger.error(f"Failed to initialize Qdrant collection: {e}")
    
    def _add_sample_documents(self) -> None:
        """
        Add sample shipping documents to the vector database.
        """
        sample_documents = [
            {
                "content": "Fedex Ground shipping is the most cost-effective option for packages under 70 pounds. Delivery typically takes 1-7 business days depending on the distance and zone.",
                "metadata": {
                    "type": "shipping_info",
                    "service": "ground",
                    "topic": "pricing"
                }
            },
            {
                "content": "Fedex Express provides overnight and 2-day delivery options. Express Saver offers 3-day delivery at a lower cost than standard Express.",
                "metadata": {
                    "type": "shipping_info",
                    "service": "express",
                    "topic": "service_types"
                }
            },
            {
                "content": "Shipping zones are determined by distance from origin to destination. Zone 1 covers 0-150 miles, Zone 2 covers 151-300 miles, and so on.",
                "metadata": {
                    "type": "shipping_info",
                    "topic": "zones",
                    "category": "distance"
                }
            },
            {
                "content": "Weather conditions can significantly impact delivery times. Severe weather events may cause delays of 1-3 business days.",
                "metadata": {
                    "type": "shipping_info",
                    "topic": "weather_impact",
                    "category": "delays"
                }
            },
            {
                "content": "Fuel prices affect shipping costs through fuel surcharges. These surcharges are typically updated weekly based on current fuel prices.",
                "metadata": {
                    "type": "shipping_info",
                    "topic": "fuel_analysis",
                    "category": "costs"
                }
            },
            {
                "content": "Route optimization considers factors like distance, traffic conditions, weather, and delivery time windows to find the most efficient path.",
                "metadata": {
                    "type": "shipping_info",
                    "topic": "route_optimization",
                    "category": "efficiency"
                }
            },
            {
                "content": "Package weight and dimensions determine shipping rates. Heavier packages cost more to ship, and dimensional weight may apply for large, lightweight packages.",
                "metadata": {
                    "type": "shipping_info",
                    "topic": "pricing",
                    "category": "weight"
                }
            },
            {
                "content": "Traffic congestion can add 1-2 hours to delivery times in urban areas. Real-time traffic data helps optimize delivery routes.",
                "metadata": {
                    "type": "shipping_info",
                    "topic": "traffic_analysis",
                    "category": "delays"
                }
            }
        ]
        
        try:
            # Generate embeddings and add to collection
            documents = []
            for i, doc in enumerate(sample_documents):
                embedding = self.embedding_model.encode(doc["content"])
                documents.append({
                    "id": i + 1,
                    "vector": embedding.tolist(),
                    "payload": {
                        "content": doc["content"],
                        "metadata": doc["metadata"]
                    }
                })
            
            # Add documents to collection
            self.qdrant_client.upsert(
                collection_name=self.collection_name,
                points=documents
            )
            
            logger.info(f"Added {len(sample_documents)} sample documents to collection")
            
        except Exception as e:
            logger.error(f"Failed to add sample documents: {e}")
    
    def _get_current_time(self) -> float:
        """Get current time for execution timing."""
        import time
        return time.time()


# Register the vector search agent
vector_search_agent = VectorSearchAgent()
register_agent(vector_search_agent) 