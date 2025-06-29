"""
Structured Data Agent for the Aura Shipping Intelligence Platform.

This agent uses Vanna.ai and PostgreSQL to handle SQL operations
and query structured data sources like Fedex pricing information.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional
import re

# Vanna AI imports - using Qdrant with Ollama
from vanna.qdrant import Qdrant_VectorStore
from qdrant_client import QdrantClient
from langchain_ollama import OllamaLLM

from .base import AgentConfig, AgentContext, AgentResponse, BaseAgent, register_agent
from src.core.database import get_database_manager
from src.core.state import WorkflowState
import psycopg2
from src.utils.get_distance_between_cities import get_distance_between_cities

logger = logging.getLogger(__name__)

DEFAULT_QDRANT_URL = "http://localhost:6333"
DEFAULT_COLLECTION_NAME = "aura_shipping_vanna"
OLLAMA_MODEL = "llama3:8b"

def initialize_qdrant_client(url: str = DEFAULT_QDRANT_URL) -> QdrantClient:
    """Initialize and return Qdrant client."""
    logger.debug(f"Initializing Qdrant client with URL: {url}")
    return QdrantClient(url=url)


class OllamaVanna:
    """Custom Vanna class using Ollama LLM with Qdrant vector store."""
    
    def __init__(self, config=None):
        self.config = config or {}
        self.llm = OllamaLLM(
            model=OLLAMA_MODEL,
            temperature=0.1,
            base_url="http://localhost:11434"
        )
        
        # Initialize Qdrant client
        self.qdrant_client = QdrantClient(
            host=self.config.get('host', 'localhost'),
            port=self.config.get('port', 6333)
        )
        
        self.collection_name = self.config.get('collection_name', DEFAULT_COLLECTION_NAME)
        self._initialize_collection()
        
        # Training data storage
        self.training_data = []
    
    def _initialize_collection(self):
        """Initialize Qdrant collection if it doesn't exist."""
        try:
            collections = self.qdrant_client.get_collections()
            collection_names = [col.name for col in collections.collections]
            
            if self.collection_name not in collection_names:
                from qdrant_client.models import Distance, VectorParams
                self.qdrant_client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=384,  # Standard embedding size
                        distance=Distance.COSINE
                    )
                )
                logger.info(f"Created Qdrant collection: {self.collection_name}")
            else:
                logger.info(f"Using existing Qdrant collection: {self.collection_name}")
                
        except Exception as e:
            logger.error(f"Failed to initialize Qdrant collection: {e}")
    
    def train(self, documentation=None, question=None, sql=None, ddl=None):
        """Train the model with various types of data."""
        if documentation:
            self.training_data.append({
                "type": "documentation",
                "content": documentation
            })
            logger.info("Added documentation to training data")
        
        if question and sql:
            self.training_data.append({
                "type": "question_sql",
                "question": question,
                "sql": sql
            })
            logger.info("Added question-SQL pair to training data")
        
        if ddl:
            self.training_data.append({
                "type": "ddl",
                "content": ddl
            })
            logger.info("Added DDL to training data")
    
    def validate_sql_syntax(self, sql_query: str) -> bool:
        """Validate SQL syntax using Postgres PREPARE statement."""
        try:
            conn = psycopg2.connect(
                host=self.config.get('host', 'localhost'),
                port=self.config.get('port', 5432),
                dbname=self.config.get('database', 'aura_db'),
                user=self.config.get('user', 'aura_user'),
                password=self.config.get('password', 'aura_password'),
            )
            cur = conn.cursor()
            try:
                cur.execute(f"PREPARE validate_stmt AS {sql_query}")
                cur.execute("DEALLOCATE validate_stmt")
            finally:
                cur.close()
                conn.close()
            return True
        except Exception as e:
            logger.warning(f"SQL validation error: {e}")
            return False

    def generate_sql(self, question: str, zone: str = None) -> str:
        """Generate SQL from natural language question using Ollama. Optionally inject a FedEx zone."""
        try:
            # Create context from training data
            context = self._build_context(question)
            
            if zone:
                zone_hint = f"""
<<CRITICAL ZONE REQUIREMENT>>
The FedEx zone for this query is: {zone}
You MUST include this zone in your WHERE clause: WHERE zone = '{zone}'
This is MANDATORY - do not generate SQL without the zone filter.
<</CRITICAL ZONE REQUIREMENT>>
"""
            else:
                zone_hint = ""
            
            prompt = f"""
<<ROLE>>
You are a SQL expert. Based on the following context and question, generate a valid SQL query.

<<CONTEXT>>
{context}
{zone_hint}
<<GUARDRAILS>>
- You are ONLY allowed to use data from the following tables: datasource_fedex_pricing, datasource_fedex_zone_distance_mapping, shipping_rates, shipping_zones, weather_data, fuel_prices, traffic_data.
- Do NOT mention, speculate about, or reference any shipping carrier other than FedEx. This includes USPS, UPS, DHL, or any other company.
- If the answer cannot be found in the provided data, respond with: "The requested information is not available in the current dataset."
- Do NOT use any prior knowledge, training data, or assumptions about shipping, pricing, or carriers. Only use the facts present in the database.
- Do NOT invent, guess, or hallucinate any information.
- If asked about unavailable data, clearly state its absence.
<</GUARDRAILS>>

<<QUESTION>>
{question}
<</QUESTION>>

<<MAIN ANSWER TABLE FORMAT>>
Present all relevant available services in a markdown table. Include "N/A" for unavailable combinations.

| Service | Price | Delivery Time | Transport Type | Notes |
| :---------------------- | :-------- | :-------------------------------- | :------------- | :------------------ |
| FedEx First Overnight | $XXX.XX | Next business day by 8:00 a.m. | Air | Premium overnight |
| FedEx Priority Overnight | $XXX.XX | Next business day by 10:30 a.m. | Air | Standard overnight |
| FedEx Standard Overnight | $XXX.XX | Next business day by 5:00 p.m. | Air | Economy overnight |
| FedEx 2Day A.M. | $XXX.XX | 2 business days by 10:30 a.m. | Air | Early 2-day |
| FedEx 2Day | $XXX.XX | 2 business days by 5:00 p.m. | Air | Standard 2-day |
| FedEx Express Saver | $XXX.XX | 3 business days by 5:00 p.m. | Air | Economy air |
| FedEx Ground | $XXX.XX | 1-5 business days | Ground | Standard ground |
| FedEx Home Delivery | $XXX.XX | 4-7 business days (incl. Sat/Sun) | Ground | Residential delivery |

**3. Recommendations** (When applicable)
Offer tailored recommendations based on implied user needs:
-   **Budget-conscious**: Suggest the lowest-cost option.
-   **Time-sensitive**: Recommend the fastest available option.
-   **Balanced Value**: Propose an option that combines reasonable cost and speed.

<</MAIN ANSWER TABLE FORMAT>>


<<DELIVERY TIME MAPPING>>
-   FedEx First Overnight → "Next business day by 8:00 a.m."
-   FedEx Priority Overnight → "Next business day by 10:30 a.m."
-   FedEx Standard Overnight → "Next business day by 5:00 p.m."
-   FedEx 2Day A.M. → "2 business days by 10:30 a.m."
-   FedEx 2Day → "2 business days by 5:00 p.m."
-   FedEx Express Saver → "3 business days by 5:00 p.m."
-   FedEx Ground → "1-5 business days"
-   FedEx Home Delivery → "Roughly 4-7 business days (including Saturday & Sunday)"
-   FedEx Home Delivery Truck driver speed: 500 miles/day (use this to calculate delivery time)
<</DELIVERY TIME MAPPING>>


<<ERROR HANDLING AND CLARIFICATION>>
-   **Missing Data**: If a service, weight, or zone combination is not found, state "N/A" for the price.
-   **Weight Approximation**: Clearly state if an exact weight was not found and an approximation was used.
-   **Zone Ambiguity**: If origin/destination is unclear, ask for clarification or provide options for plausible zones.
-   **Out-of-Scope Requests**: Politely inform the user if the request is outside the supported domestic USA (Zones 2-8) or if it's an international shipment. Explain limitations.
<</ERROR HANDLING AND CLARIFICATION>>

<<OUTPUT FORMAT REQUIREMENTS>>
- Generate SQL that returns results in a clear, tabular format with columns for service_type, transportation_type, zone, weight, price, and any other relevant cost factors.
- The results should enable a direct cost-benefit analysis for the user, making it easy to compare different FedEx shipping options for the given weight and zone.
- If possible, include columns for estimated delivery time or other differentiators present in the data.
- The SQL should be suitable for rendering as a table in the UI, with each row representing a unique shipping option.
- Do NOT include explanations, markdown, or formatting—just the raw SQL query ending with a semicolon.
<</OUTPUT FORMAT REQUIREMENTS>>

<<EXAMPLE>>
SELECT service_type, transportation_type, zone, weight, price FROM datasource_fedex_pricing WHERE weight = 5 AND zone = '8' ORDER BY price ASC;
<</EXAMPLE>>

<<END>>
"""
            
            # Get response from Ollama
            response = self.llm.invoke(prompt)
            
            # Extract SQL from response
            sql = self._extract_sql_from_response(response)
            
            logger.info(f"Generated SQL: {sql}")
            return sql
        except Exception as e:
            logger.error(f"Failed to generate SQL: {e}")
            # Return a safe fallback query
            return "SELECT * FROM datasource_fedex_pricing LIMIT 5;"
    
    def _build_context(self, question: str) -> str:
        """Build context from training data for the given question."""
        context_parts = []
        
        # Add enhanced schema information
        schema_info = """
Database Schema:
- datasource_fedex_pricing: weight (REAL), transportation_type (TEXT), zone (TEXT), service_type (TEXT), price (REAL)
- datasource_fedex_zone_distance_mapping: zone (TEXT), min_distance (REAL), max_distance (REAL)

IMPORTANT: Always use datasource_fedex_pricing for shipping pricing queries.
Use datasource_fedex_zone_distance_mapping only for zone distance information.
"""
        context_parts.append(schema_info)
        
        # Add relevant training examples
        for item in self.training_data:
            if item["type"] == "question_sql":
                context_parts.append(f"Example: Q: {item['question']} A: {item['sql']}")
            elif item["type"] == "documentation":
                context_parts.append(f"Documentation: {item['content']}")
        
        return "\n".join(context_parts)
    
    def _extract_sql_from_response(self, response: str) -> str:
        """Extract SQL query from Ollama response."""
        if not isinstance(response, str):
            response = str(response)
        
        # Clean up the response
        response = response.strip()
        
        # First, try to find a complete SQL query with semicolon
        import re
        
        # Look for SQL starting with SELECT and ending with semicolon
        sql_match = re.search(r'SELECT\s+.*?;', response, re.IGNORECASE | re.DOTALL)
        if sql_match:
            sql = sql_match.group(0).strip()
            # Validate it has a FROM clause
            if 'FROM' in sql.upper():
                return sql
        
        # If no semicolon, look for SQL ending with FROM clause
        sql_match = re.search(r'SELECT\s+.*?FROM\s+\w+', response, re.IGNORECASE | re.DOTALL)
        if sql_match:
            sql = sql_match.group(0).strip()
            # Add semicolon if missing
            if not sql.endswith(';'):
                sql += ';'
            return sql
        
        # Try to extract SQL from code blocks
        code_block_match = re.search(r'```sql\s*(.*?)\s*```', response, re.DOTALL)
        if code_block_match:
            sql = code_block_match.group(1).strip()
            if sql.upper().startswith('SELECT') and 'FROM' in sql.upper():
                if not sql.endswith(';'):
                    sql += ';'
                return sql
        
        # Try to find SQL in lines
        lines = response.split('\n')
        sql_lines = []
        in_sql = False
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Check if this line starts SQL
            if line.upper().startswith('SELECT'):
                in_sql = True
                sql_lines = [line]
            elif in_sql:
                sql_lines.append(line)
                # Check if we've reached the end
                if line.endswith(';') or 'FROM' in line.upper():
                    break
        
        if sql_lines:
            sql = ' '.join(sql_lines)
            # Ensure it has FROM clause
            if 'FROM' in sql.upper():
                if not sql.endswith(';'):
                    sql += ';'
                return sql
        
        # Final fallback - create a basic query
        logger.warning(f"Could not extract SQL from response: {response[:200]}...")
        return "SELECT * FROM datasource_fedex_pricing LIMIT 5;"
    
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
        """Initialize Vanna instance with Ollama configuration."""
        try:
            # Initialize custom Vanna instance with Ollama
            vanna_config = {
                'host': 'localhost',
                'port': 6333,
                'collection_name': DEFAULT_COLLECTION_NAME
            }
            
            self.vanna_instance = OllamaVanna(config=vanna_config)
            logger.info("Successfully initialized Vanna AI with Qdrant and Ollama")
            
        except Exception as e:
            logger.error(f"Failed to initialize Vanna AI: {e}")
            raise RuntimeError(f"Vanna AI initialization failed: {e}")
    
    def get_required_dependencies(self) -> List[str]:
        """Structured data agent has no agent dependencies - it only requires package dependencies."""
        return []
    
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
        """Train the Vanna model with database schema and real data examples."""
        try:
            # Get table descriptions
            table_doc = await self._get_table_descriptions()
            
            # Get real data statistics for better training
            stats_query = """
            SELECT 
                COUNT(*) as total_records,
                COUNT(DISTINCT transportation_type) as unique_transport_types,
                COUNT(DISTINCT zone) as unique_zones,
                COUNT(DISTINCT service_type) as unique_service_types,
                MIN(weight) as min_weight,
                MAX(weight) as max_weight,
                MIN(price) as min_price,
                MAX(price) as max_price
            FROM datasource_fedex_pricing;
            """
            
            stats = await self.db_manager.execute_query(stats_query)
            stats_row = stats[0] if stats else {}
            
            # Get unique values for training
            transport_query = "SELECT DISTINCT transportation_type FROM datasource_fedex_pricing ORDER BY transportation_type;"
            transport_types = await self.db_manager.execute_query(transport_query)
            transport_list = [row['transportation_type'] for row in transport_types]
            
            zones_query = "SELECT DISTINCT zone FROM datasource_fedex_pricing ORDER BY zone;"
            zones = await self.db_manager.execute_query(zones_query)
            zone_list = [row['zone'] for row in zones]
            
            services_query = "SELECT DISTINCT service_type FROM datasource_fedex_pricing ORDER BY service_type;"
            services = await self.db_manager.execute_query(services_query)
            service_list = [row['service_type'] for row in services]
            
            # Enhanced schema information with real data
            enhanced_schema = f"""
{table_doc}

Real Data Statistics:
- Total records: {stats_row.get('total_records', 'Unknown')}
- Transportation types: {', '.join(transport_list)}
- Zones: {', '.join(zone_list)}
- Service types: {', '.join(service_list)}
- Weight range: {stats_row.get('min_weight', 'Unknown')} - {stats_row.get('max_weight', 'Unknown')} lbs
- Price range: ${stats_row.get('min_price', 0):.2f} - ${stats_row.get('max_price', 0):.2f}

Data Patterns:
- Weight is stored as numeric values (e.g., 1, 2, 3...)
- Price is stored as decimal values in USD
- Zone values are strings (e.g., '2', '3', '4', '5', '6', '7', '8')
- Transportation types include: {', '.join(transport_list)}
- Service types include: {', '.join(service_list)}
"""
            
            # Train with enhanced documentation
            self.vanna_instance.train(documentation=enhanced_schema)
            logger.info("Trained with enhanced schema and real data statistics")
            
            # Add comprehensive real-world training examples
            real_training_data = [
                # Price analysis examples
                {
                    "question": "What are the cheapest shipping options under $50?",
                    "sql": "SELECT service_type, transportation_type, zone, weight, price FROM datasource_fedex_pricing WHERE price < 50 ORDER BY price ASC LIMIT 10;"
                },
                {
                    "question": "Show me the most expensive shipping options",
                    "sql": "SELECT service_type, transportation_type, zone, weight, price FROM datasource_fedex_pricing ORDER BY price DESC LIMIT 10;"
                },
                {
                    "question": "What is the average price by service type?",
                    "sql": "SELECT service_type, AVG(price) as avg_price, COUNT(*) as count FROM datasource_fedex_pricing GROUP BY service_type ORDER BY avg_price;"
                },
                # Weight-based examples
                {
                    "question": "Show me pricing for packages weighing 10 pounds",
                    "sql": "SELECT service_type, transportation_type, zone, weight, price FROM datasource_fedex_pricing WHERE weight = 10 ORDER BY price;"
                },
                {
                    "question": "What are the prices for packages between 5 and 10 pounds?",
                    "sql": "SELECT service_type, transportation_type, zone, weight, price FROM datasource_fedex_pricing WHERE weight BETWEEN 5 AND 10 ORDER BY weight, price;"
                },
                # Zone-based examples
                {
                    "question": "Show me all pricing for zone 2",
                    "sql": "SELECT service_type, transportation_type, zone, weight, price FROM datasource_fedex_pricing WHERE zone = '2' ORDER BY weight, price;"
                },
                {
                    "question": "Compare prices between zones 2 and 3",
                    "sql": "SELECT zone, service_type, AVG(price) as avg_price FROM datasource_fedex_pricing WHERE zone IN ('2', '3') GROUP BY zone, service_type ORDER BY zone, avg_price;"
                },
                # Weight AND Zone combination examples (CRITICAL for city-based queries)
                {
                    "question": "What are the shipping options for a 5 pound package in zone 2?",
                    "sql": "SELECT service_type, transportation_type, zone, weight, price FROM datasource_fedex_pricing WHERE weight = 5 AND zone = '2' ORDER BY price;"
                },
                {
                    "question": "Show me pricing for 10lb packages in zone 3",
                    "sql": "SELECT service_type, transportation_type, zone, weight, price FROM datasource_fedex_pricing WHERE weight = 10 AND zone = '3' ORDER BY price;"
                },
                {
                    "question": "What does it cost to ship a 2 pound package in zone 1?",
                    "sql": "SELECT service_type, transportation_type, zone, weight, price FROM datasource_fedex_pricing WHERE weight = 2 AND zone = '1' ORDER BY price;"
                },
                {
                    "question": "I need to send a 5lb package from Fremont, CA to New York, NY",
                    "sql": "SELECT service_type, transportation_type, zone, weight, price FROM datasource_fedex_pricing WHERE weight = 5 AND zone = '8' ORDER BY price;"
                },
                {
                    "question": "Shipping a 3 pound package from Los Angeles, CA to Chicago, IL",
                    "sql": "SELECT service_type, transportation_type, zone, weight, price FROM datasource_fedex_pricing WHERE weight = 3 AND zone = '6' ORDER BY price;"
                },
                # Transportation type examples
                {
                    "question": "Show me all air shipping options",
                    "sql": "SELECT service_type, zone, weight, price FROM datasource_fedex_pricing WHERE transportation_type = 'air' ORDER BY price;"
                },
                {
                    "question": "Compare air vs ground shipping prices",
                    "sql": "SELECT transportation_type, service_type, AVG(price) as avg_price FROM datasource_fedex_pricing WHERE transportation_type IN ('air', 'ground') GROUP BY transportation_type, service_type ORDER BY transportation_type, avg_price;"
                },
                # Service type examples
                {
                    "question": "Show me all FedEx First Overnight options",
                    "sql": "SELECT transportation_type, zone, weight, price FROM datasource_fedex_pricing WHERE service_type = 'FedEx First Overnight' ORDER BY price;"
                },
                # Complex analysis examples
                {
                    "question": "What is the price difference between zones for the same weight?",
                    "sql": "SELECT zone, weight, AVG(price) as avg_price FROM datasource_fedex_pricing GROUP BY zone, weight ORDER BY weight, zone;"
                },
                {
                    "question": "Find the best value shipping options (lowest price per pound)",
                    "sql": "SELECT service_type, transportation_type, zone, weight, price, (price/weight) as price_per_pound FROM datasource_fedex_pricing ORDER BY price_per_pound ASC LIMIT 10;"
                },
                # Additional examples based on actual data
                {
                    "question": "What are the cheapest shipping options?",
                    "sql": "SELECT service_type, transportation_type, zone, weight, price FROM datasource_fedex_pricing ORDER BY price ASC LIMIT 10;"
                },
                {
                    "question": "Show me pricing for different weights",
                    "sql": "SELECT weight, AVG(price) as avg_price, MIN(price) as min_price, MAX(price) as max_price FROM datasource_fedex_pricing GROUP BY weight ORDER BY weight;"
                },
                {
                    "question": "Compare prices by service type",
                    "sql": "SELECT service_type, COUNT(*) as count, AVG(price) as avg_price, MIN(price) as min_price, MAX(price) as max_price FROM datasource_fedex_pricing GROUP BY service_type ORDER BY avg_price;"
                }
            ]
            
            # Train with real examples
            for example in real_training_data:
                self.vanna_instance.train(question=example["question"], sql=example["sql"])
            
            logger.info(f"Successfully trained Vanna model with {len(real_training_data)} real-world examples and enhanced schema")
            
        except Exception as e:
            logger.error(f"Failed to train Vanna model: {e}")
            raise RuntimeError(f"Vanna model training failed: {e}")
    
    async def _execute_agent(self, context: AgentContext, state: WorkflowState) -> AgentResponse:
        """
        Execute the structured data agent logic using Vanna AI with Ollama.
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

            # --- New: Extract cities and zone ---
            origin_city, dest_city = None, None
            zone = None
            # Try to extract cities using regex (simple heuristic)
            city_pattern = re.compile(r'from ([^,]+, [A-Z]{2}) to ([^,]+, [A-Z]{2})', re.IGNORECASE)
            match = city_pattern.search(user_query)
            if match:
                origin_city, dest_city = match.group(1).strip(), match.group(2).strip()
                try:
                    distance_km = get_distance_between_cities(origin_city, dest_city)
                    distance_miles = distance_km * 0.621371

                    print(f"Distance between {origin_city} and {dest_city} is {distance_km:.2f} km")
                    print(f"Distance between {origin_city} and {dest_city} is {distance_miles:.2f} miles")
                    # Query the zone mapping table
                    sql = """
                        SELECT zone FROM datasource_fedex_zone_distance_mapping
                        WHERE min_distance <= $1 AND max_distance >= $2
                        LIMIT 1;
                    """
                    rows = await self.db_manager.execute_query(sql, [distance_miles, distance_miles])
                    if rows and 'zone' in rows[0]:
                        zone = rows[0]['zone']
                        logger.info(f"Identified FedEx zone {zone} for distance {distance_miles:.2f} miles between {origin_city} and {dest_city}")
                except Exception as e:
                    logger.warning(f"Failed to get zone for cities: {e}")
            # --- End new logic ---

            # Generate SQL using Vanna AI, injecting the zone if found
            sql_query = self.vanna_instance.generate_sql(user_query, zone=zone)
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
                    "vanna_model_used": True,
                    "llm_model": OLLAMA_MODEL
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
                    "llm_model": OLLAMA_MODEL
                }
            )
        except Exception as e:
            logger.error(f"Structured data agent execution failed: {e}")
            execution_time = self._get_current_time() - start_time
            return AgentResponse(
                success=False,
                error=str(e),
                execution_time=execution_time
            )
    
    def _validate_sql_query(self, sql_query: str) -> bool:
        """
        Basic validation of generated SQL query.
        """
        if not sql_query:
            return False
        
        sql_upper = sql_query.upper().strip()
        
        # Must start with SELECT
        if not sql_upper.startswith('SELECT'):
            return False
        
        # Must contain FROM
        if 'FROM' not in sql_upper:
            return False
        
        # Must end with semicolon
        if not sql_query.strip().endswith(';'):
            return False
        
        # Basic security check - no dangerous keywords
        dangerous_keywords = ['DROP', 'DELETE', 'UPDATE', 'INSERT', 'CREATE', 'ALTER', 'TRUNCATE']
        for keyword in dangerous_keywords:
            if keyword in sql_upper:
                return False
        
        return True
    
    def _generate_explanation(self, user_query: str, sql_query: str, results: List[Dict[str, Any]]) -> str:
        """
        Generate a natural language explanation of the SQL results.
        """
        try:
            if not results:
                return "No results found for your query."
            
            # Simple explanation based on result count
            if len(results) == 1:
                return f"Found 1 result for your query about {user_query.lower()}."
            else:
                return f"Found {len(results)} results for your query about {user_query.lower()}."
                
        except Exception as e:
            logger.error(f"Failed to generate explanation: {e}")
            return "Results retrieved successfully."
    
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
            return {
                "query": sql_query,
                "results": [],
                "success": False,
                "error": str(e),
                "row_count": 0
            }
    
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
register_agent(structured_data_agent)