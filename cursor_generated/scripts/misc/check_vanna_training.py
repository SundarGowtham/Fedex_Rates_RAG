#!/usr/bin/env python3
"""
Script to check Vanna AI training status and enhance training with real data.
"""

import asyncio
import logging
import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from src.agents.structured_data import OllamaVanna, StructuredDataAgent
from src.core.database import get_database_manager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def check_vanna_training():
    """Check and enhance Vanna training with real data."""
    
    print("🔍 Checking Vanna AI Training Status...")
    print("=" * 50)
    
    try:
        # Initialize database manager
        db_manager = await get_database_manager()
        
        # Check what data is in the table
        print("📊 Checking datasource_fedex_pricing table...")
        
        # Get sample data
        sample_query = "SELECT * FROM datasource_fedex_pricing LIMIT 10;"
        sample_data = await db_manager.execute_query(sample_query)
        
        print(f"✅ Found {len(sample_data)} sample records")
        print("Sample data:")
        for i, row in enumerate(sample_data[:5]):
            print(f"  {i+1}. {row}")
        
        # Get data statistics
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
        
        stats = await db_manager.execute_query(stats_query)
        if stats:
            stats_row = stats[0]
            print(f"\n📈 Data Statistics:")
            print(f"  Total records: {stats_row['total_records']}")
            print(f"  Transportation types: {stats_row['unique_transport_types']}")
            print(f"  Zones: {stats_row['unique_zones']}")
            print(f"  Service types: {stats_row['unique_service_types']}")
            print(f"  Weight range: {stats_row['min_weight']} - {stats_row['max_weight']} lbs")
            print(f"  Price range: ${stats_row['min_price']:.2f} - ${stats_row['max_price']:.2f}")
        
        # Get unique values for training
        print(f"\n🔍 Getting unique values for training...")
        
        # Get unique transportation types
        transport_query = "SELECT DISTINCT transportation_type FROM datasource_fedex_pricing ORDER BY transportation_type;"
        transport_types = await db_manager.execute_query(transport_query)
        transport_list = [row['transportation_type'] for row in transport_types]
        print(f"  Transportation types: {transport_list}")
        
        # Get unique zones
        zones_query = "SELECT DISTINCT zone FROM datasource_fedex_pricing ORDER BY zone;"
        zones = await db_manager.execute_query(zones_query)
        zone_list = [row['zone'] for row in zones]
        print(f"  Zones: {zone_list}")
        
        # Get unique service types
        services_query = "SELECT DISTINCT service_type FROM datasource_fedex_pricing ORDER BY service_type;"
        services = await db_manager.execute_query(services_query)
        service_list = [row['service_type'] for row in services]
        print(f"  Service types: {service_list}")
        
        # Initialize Vanna
        print(f"\n🤖 Initializing Vanna AI...")
        vanna = OllamaVanna()
        
        # Enhanced training with real data
        print(f"\n📚 Training Vanna with real data samples...")
        
        # Train with schema information
        schema_info = f"""
Database Schema for Fedex Pricing:
- datasource_fedex_pricing table contains shipping pricing data
- Columns: weight (REAL), transportation_type (TEXT), zone (TEXT), service_type (TEXT), price (REAL)
- Weight is in pounds (lbs)
- Price is in USD
- Transportation types available: {', '.join(transport_list)}
- Zones available: {', '.join(zone_list)}
- Service types available: {', '.join(service_list)}
- Total records: {stats_row['total_records'] if stats else 'Unknown'}
- Price range: ${stats_row['min_price']:.2f} - ${stats_row['max_price']:.2f} if stats else 'Unknown'
"""
        vanna.train(documentation=schema_info)
        print("  ✅ Trained with schema information")
        
        # Train with real data examples
        real_training_examples = [
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
            # Transportation type examples
            {
                "question": "Show me all air shipping options",
                "sql": "SELECT service_type, zone, weight, price FROM datasource_fedex_pricing WHERE transportation_type = 'air' ORDER BY price;"
            },
            {
                "question": "Compare air vs ground shipping prices",
                "sql": "SELECT transportation_type, service_type, AVG(price) as avg_price FROM datasource_fedex_pricing GROUP BY transportation_type, service_type ORDER BY transportation_type, avg_price;"
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
            }
        ]
        
        for i, example in enumerate(real_training_examples):
            vanna.train(question=example["question"], sql=example["sql"])
            print(f"  ✅ Trained with example {i+1}: {example['question'][:50]}...")
        
        print(f"\n🎯 Training completed! Vanna now has knowledge of:")
        print(f"  - {len(real_training_examples)} real-world query examples")
        print(f"  - Complete schema information")
        print(f"  - Data statistics and ranges")
        print(f"  - All available transportation types, zones, and service types")
        
        # Test the training
        print(f"\n🧪 Testing Vanna with sample queries...")
        
        test_queries = [
            "What are the cheapest shipping options?",
            "Show me pricing for 5 pound packages",
            "Compare air vs ground shipping",
            "What zones are available?"
        ]
        
        for query in test_queries:
            try:
                sql = vanna.generate_sql(query)
                print(f"  Q: {query}")
                print(f"  A: {sql}")
                print()
            except Exception as e:
                print(f"  Q: {query}")
                print(f"  Error: {e}")
                print()
        
        print("✅ Vanna training check completed!")
        
    except Exception as e:
        logger.error(f"Error checking Vanna training: {e}")
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    asyncio.run(check_vanna_training()) 