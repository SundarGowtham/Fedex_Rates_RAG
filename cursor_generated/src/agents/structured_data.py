"""
Structured Data Agent for the Aura Shipping Intelligence Platform.

This agent uses Vanna.ai and PostgreSQL to handle SQL operations
and query structured data sources like Fedex pricing information.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

# Vanna AI imports - using Qdrant with OpenAI
from vanna.openai import OpenAI_Chat
from vanna.qdrant import Qdrant_VectorStore
from qdrant_client import QdrantClient

from langchain.schema import HumanMessage, SystemMessage

from .base import AgentConfig, AgentContext, AgentResponse, BaseAgent, register_agent
from core.database import get_database_manager
from core.state import WorkflowState

logger = logging.getLogger(__name__)

DEFAULT_QDRANT_URL = "http://localhost:6333"
def initialize_qdrant_client(url: str = DEFAULT_QDRANT_URL) -> QdrantClient:
    """Initialize and return Qdrant client."""
    logger.debug(f"Initializing Qdrant client with URL: {url}")
    return QdrantClient(url=url)


class MyVanna(Qdrant_VectorStore, OpenAI_Chat):
    """Custom Vanna class combining Qdrant vector store with OpenAI chat."""
    def __init__(self, config=None):
        OpenAI_Chat.__init__(self, config=config)
        Qdrant_VectorStore.__init__(self, config=config)

    def log(self, message: str, title: str = "Info"):
        """Override the log method to do nothing."""
        pass


class StructuredDataAgent(BaseAgent):
    """
    Structured Data Agent for SQL operations and data querying.
    
    This agent uses Vanna.ai to convert natural language queries to SQL
    and executes them against the PostgreSQL database to retrieve
    structured shipping data.
    """
    
    def __init__(self):
        config = AgentConfig(
            name="structured_data",
            description="SQL operations and structured data querying agent",
            timeout_seconds=300,
            max_retries=3,
            enable_caching=True,
            cache_ttl_seconds=3600
        )
        super().__init__(config)
        
        # Initialize database connection
        self.db_manager = None
        # Initialize Vanna instance
        self.vanna_instance = None
        self._setup_vanna()
    
    def _setup_vanna(self):
        """Initialize Vanna instance with proper configuration."""
        # Check for required API key
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError(
                "OpenAI API key is required for Vanna AI. "
                "Please set the OPENAI_API_KEY environment variable."
            )
        
        try:
            # Initialize custom Vanna instance
            vanna_config = {
                'api_key': api_key,
                'model': 'gpt-3.5-turbo'  # or 'gpt-4' if you have access
            }
            
            self.vanna_instance = MyVanna(config=vanna_config)
            logger.info("Successfully initialized Vanna AI with Qdrant and OpenAI")
            
        except Exception as e:
            logger.error(f"Failed to initialize Vanna AI: {e}")
            raise RuntimeError(f"Vanna AI initialization failed: {e}")
    
    def get_required_dependencies(self) -> List[str]:
        """Structured data agent requires Vanna AI to be properly initialized."""
        return ["vanna", "qdrant", "qdrant-client", "openai"]
    
    async def _get_table_descriptions(self):
        """Get table descriptions from the database."""
        try:
            sql = """
                SELECT table_name, column_name, description
                FROM table_descriptions
                ORDER BY table_name, column_name
            """
            rows = await self.db_manager.execute_query(sql)
            table_docs = {}
            for row in rows:
                table = row['table_name']
                if table not in table_docs:
                    table_docs[table] = []
                table_docs[table].append(f"- {row['column_name']}: {row['description']}")
            
            doc_strings = []
            for table, columns in table_docs.items():
                doc_strings.append(f"\nThe {table} table contains the following columns:")
                doc_strings.extend(columns)
            return "\n".join(doc_strings)
            
        except Exception as e:
            logger.error(f"Failed to get table descriptions: {e}")
            # Use hardcoded schema as last resort
            return self._get_hardcoded_schema()
    
    def _get_hardcoded_schema(self) -> str:
        """Return hardcoded schema information."""
        return """
        Database Schema Information:
        
        Table: datasource_fedex_pricing
        - weight (REAL): Package weight in pounds
        - transportation_type (TEXT): Type of transportation (ground, air, etc.)
        - zone (TEXT): Shipping zone code  
        - service_type (TEXT): Fedex service type
        - price (REAL): Shipping price in USD
        
        Table: datasource_fedex_zone_distance_mapping  
        - zone (TEXT): Zone code
        - min_distance (REAL): Minimum distance in miles for this zone
        - max_distance (REAL): Maximum distance in miles for this zone
        
        Additional Context:
        - Prices are in USD
        - Weights are in pounds
        - Distances are in miles
        - Zone codes are alphanumeric (e.g., 'Zone2', 'Zone5')
        - Transportation types include 'GROUND', 'AIR', etc.
        - Service types include 'STANDARD', 'EXPRESS', 'OVERNIGHT', etc.
        """
    
    async def _train_vanna_model(self):
        """Train the Vanna model with database schema and sample queries."""
        try:
            # Get table descriptions
            table_doc = await self._get_table_descriptions()
            
            # Train with documentation
            self.vanna_instance.train(documentation=table_doc)
            
            # Add some example SQL queries for better training
            training_data = [
                {
                    "question": "What are the cheapest shipping options?",
                    "sql": "SELECT service_type, transportation_type, zone, weight, price FROM datasource_fedex_pricing ORDER BY price ASC LIMIT 10;"
                },
                {
                    "question": "Show me pricing for different weights",
                    "sql": "SELECT weight, AVG(price) as avg_price, MIN(price) as min_price, MAX(price) as max_price FROM datasource_fedex_pricing GROUP BY weight ORDER BY weight;"
                },
                {
                    "question": "What zones are available and their distances?",
                    "sql": "SELECT zone, min_distance, max_distance FROM datasource_fedex_zone_distance_mapping ORDER BY min_distance;"
                },
                {
                    "question": "Compare prices by service type",
                    "sql": "SELECT service_type, COUNT(*) as count, AVG(price) as avg_price, MIN(price) as min_price, MAX(price) as max_price FROM datasource_fedex_pricing GROUP BY service_type ORDER BY avg_price;"
                }
            ]
            
            # Train with examples
            for example in training_data:
                self.vanna_instance.train(question=example["question"], sql=example["sql"])
            
            logger.info("Successfully trained Vanna model with schema and examples")
            
        except Exception as e:
            logger.error(f"Failed to train Vanna model: {e}")
            raise RuntimeError(f"Vanna model training failed: {e}")
    
    async def _execute_agent(self, context: AgentContext, state: WorkflowState) -> AgentResponse:
        """
        Execute the structured data agent logic using Vanna AI exclusively.
        """
        start_time = self._get_current_time()
        
        if not self.vanna_instance:
            raise RuntimeError("Vanna AI is not properly initialized")
        
        try:
            # Initialize database connection
            if not self.db_manager:
                self.db_manager = await get_database_manager()
            
            # Train the Vanna model
            await self._train_vanna_model()
            
            user_query = context.user_query
            logger.info(f"Processing query with Vanna AI: {user_query}")
            
            # Generate SQL using Vanna AI
            sql_query = self.vanna_instance.generate_sql(user_query)
            logger.info(f"Vanna generated SQL: {sql_query}")
            
            # Validate the generated SQL (optional safety check)
            if not self._validate_sql_query(sql_query):
                raise ValueError(f"Generated SQL query failed validation: {sql_query}")
            
            # Execute the generated SQL
            result = await self._execute_sql_query(sql_query)
            logger.info(f"SQL execution result: Query returned {result.get('row_count', 0)} rows")
            
            # Aggregate and format results
            aggregated_results = self._aggregate_results([result])
            
            # Generate natural language explanation of results (optional)
            explanation = self._generate_explanation(user_query, sql_query, aggregated_results)
            
            # Create response data
            response_data = {
                "sql_query": sql_query,
                "query_results": aggregated_results,
                "explanation": explanation,
                "metadata": {
                    "total_queries": 1,
                    "total_rows": len(aggregated_results),
                    "data_sources": self._get_data_sources([result]),
                    "query_requirements": user_query,
                    "vanna_model_used": True
                },
                "execution_time": self._get_current_time() - start_time,
                "row_count": len(aggregated_results)
            }
            
            execution_time = self._get_current_time() - start_time
            
            return AgentResponse(
                success=True,
                data=response_data,
                execution_time=execution_time,
                metadata={
                    "sql_queries_count": 1,
                    "total_rows_retrieved": len(aggregated_results),
                    "data_sources_queried": self._get_data_sources([result]),
                    "vanna_powered": True
                }
            )
            
        except Exception as e:
            logger.error(f"Structured data agent execution failed: {e}")
            execution_time = self._get_current_time() - start_time
            
            return AgentResponse(
                success=False,
                error=f"Vanna AI execution failed: {str(e)}",
                execution_time=execution_time
            )
    
    def _validate_sql_query(self, sql_query: str) -> bool:
        """
        Validate the generated SQL query for basic safety and correctness.
        """
        if not sql_query or not sql_query.strip():
            return False
        
        sql_lower = sql_query.lower().strip()
        
        # Check for dangerous operations
        dangerous_keywords = ['drop', 'delete', 'truncate', 'alter', 'create', 'insert', 'update']
        if any(keyword in sql_lower for keyword in dangerous_keywords):
            logger.warning(f"SQL query contains potentially dangerous keyword: {sql_query}")
            return False
        
        # Ensure it's a SELECT query
        if not sql_lower.startswith('select'):
            logger.warning(f"SQL query is not a SELECT statement: {sql_query}")
            return False
        
        # Check for required table names
        required_tables = ['datasource_fedex_pricing', 'datasource_fedex_zone_distance_mapping']
        if not any(table in sql_lower for table in required_tables):
            logger.warning(f"SQL query doesn't reference expected tables: {sql_query}")
            return False
        
        return True
    
    def _generate_explanation(self, user_query: str, sql_query: str, results: List[Dict[str, Any]]) -> str:
        """
        Generate a natural language explanation of the query results.
        """
        try:
            if not results:
                return f"No results found for the query: '{user_query}'"
            
            result_count = len(results)
            explanation = f"Found {result_count} result(s) for your query: '{user_query}'. "
            
            # Add more specific explanations based on the SQL query
            sql_lower = sql_query.lower()
            if 'order by price' in sql_lower:
                if 'asc' in sql_lower:
                    explanation += "Results are sorted by price from lowest to highest."
                else:
                    explanation += "Results are sorted by price from highest to lowest."
            elif 'group by' in sql_lower:
                explanation += "Results are grouped and aggregated for analysis."
            elif 'avg(' in sql_lower:
                explanation += "Results include average calculations."
            
            return explanation
            
        except Exception as e:
            logger.error(f"Failed to generate explanation: {e}")
            return f"Query executed successfully with {len(results)} results."
    
    async def _execute_sql_query(self, sql_query: str) -> Dict[str, Any]:
        """
        Execute a SQL query and return results.
        """
        try:
            # Execute query using database manager
            results = await self.db_manager.execute_query(sql_query)
            
            return {
                "query": sql_query,
                "results": results,
                "success": True,
                "row_count": len(results) if results else 0
            }
            
        except Exception as e:
            logger.error(f"SQL query execution failed: {e}")
            raise RuntimeError(f"Database query execution failed: {e}")
    
    def _aggregate_results(self, query_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Aggregate results from multiple queries.
        """
        aggregated = []
        
        for result in query_results:
            if result.get("success") and result.get("results"):
                aggregated.extend(result["results"])
        
        return aggregated
    
    def _get_data_sources(self, query_results: List[Dict[str, Any]]) -> List[str]:
        """
        Extract data sources from query results.
        """
        sources = set()
        
        for result in query_results:
            query = result.get("query", "").lower()
            
            if "datasource_fedex_pricing" in query:
                sources.add("fedex_pricing")
            
            if "datasource_fedex_zone_distance_mapping" in query:
                sources.add("zone_distance_mapping")
        
        return list(sources)
    
    def _get_current_time(self) -> float:
        """Get current time for execution timing."""
        import time
        return time.time()


# Register the structured data agent
structured_data_agent = StructuredDataAgent()