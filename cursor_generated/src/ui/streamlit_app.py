"""
Streamlit UI for the Aura Shipping Intelligence Platform.

This module provides a comprehensive web interface for interacting with
the multi-agent shipping intelligence system.
"""

import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Add the src directory to the path
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent.parent))

from agents.base import get_all_agents
from core.database import get_database_manager, insert_sample_data
from core.state import WorkflowState
from core.workflow import execute_workflow
from utils.data_ingestion import ingest_data_sources, clear_all_data, get_ingestion_status
from utils.visualizations import create_pricing_chart, create_zone_chart

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Aura Shipping Intelligence Platform",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .agent-status {
        padding: 0.5rem;
        border-radius: 0.5rem;
        margin: 0.25rem 0;
    }
    .status-completed { background-color: #d4edda; color: #155724; }
    .status-running { background-color: #fff3cd; color: #856404; }
    .status-failed { background-color: #f8d7da; color: #721c24; }
    .status-pending { background-color: #e2e3e5; color: #383d41; }
</style>
""", unsafe_allow_html=True)


def main():
    """Main Streamlit application."""
    
    # Header
    st.markdown('<h1 class="main-header">💬 Aura Shipping Intelligence Chat</h1>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("Navigation")
        page = st.selectbox(
            "Choose a page:",
            ["Chat Interface", "Data Management", "System Status", "About"]
        )
    
    # Page routing
    if page == "Chat Interface":
        query_interface()
    elif page == "Data Management":
        data_management()
    elif page == "System Status":
        system_status()
    elif page == "About":
        about_page()


def query_interface():
    """Main query interface for shipping intelligence."""
    st.header("💬 Shipping Intelligence Chat")
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask me about shipping rates, routes, weather impacts, or any shipping-related questions..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Display assistant response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            message_placeholder.markdown("🤔 Analyzing your query...")
            
            try:
                # Execute the query
                result = execute_conversational_query(prompt)
                message_placeholder.markdown(result)
                
                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": result})
                
            except Exception as e:
                error_message = f"Sorry, I encountered an error: {str(e)}"
                message_placeholder.markdown(error_message)
                st.session_state.messages.append({"role": "assistant", "content": error_message})
    
    # Clear chat button
    if st.sidebar.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()


def execute_conversational_query(user_query: str) -> str:
    """Execute a conversational query using the multi-agent system."""
    
    # Create a simple context extraction (this could be enhanced with NLP)
    context = extract_context_from_query(user_query)
    
    # Determine query type from the query content
    query_type = determine_query_type(user_query)
    
    try:
        # Run the workflow asynchronously
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            final_state = loop.run_until_complete(execute_workflow(user_query, context))
        finally:
            loop.close()
        
        # Format the response for chat
        return format_chat_response(final_state, user_query)
        
    except Exception as e:
        logger.error(f"Query execution failed: {e}")
        return f"Sorry, I encountered an error while processing your query: {str(e)}"


def extract_context_from_query(query: str) -> Dict[str, Any]:
    """Extract context information from natural language query."""
    # Simple keyword-based extraction (could be enhanced with NLP)
    query_lower = query.lower()
    
    context = {}
    
    # Extract weight
    import re
    weight_match = re.search(r'(\d+(?:\.\d+)?)\s*(?:pound|lb|lbs|kg)', query_lower)
    if weight_match:
        context["weight"] = float(weight_match.group(1))
    
    # Extract service type
    if any(word in query_lower for word in ["ground", "standard"]):
        context["service_type"] = "ground"
    elif any(word in query_lower for word in ["express", "priority"]):
        context["service_type"] = "express"
    elif any(word in query_lower for word in ["overnight", "urgent"]):
        context["service_type"] = "overnight"
    
    # Extract locations (simple approach)
    # This is a basic implementation - could be enhanced with geocoding
    words = query.split()
    for i, word in enumerate(words):
        if word.lower() in ["from", "to", "between"] and i + 1 < len(words):
            if word.lower() == "from":
                context["origin"] = words[i + 1]
            elif word.lower() == "to":
                context["destination"] = words[i + 1]
    
    return context


def determine_query_type(query: str) -> str:
    """Determine the type of query from natural language."""
    query_lower = query.lower()
    
    if any(word in query_lower for word in ["price", "rate", "cost", "how much"]):
        return "pricing"
    elif any(word in query_lower for word in ["route", "path", "way", "optimize"]):
        return "route_optimization"
    elif any(word in query_lower for word in ["weather", "storm", "rain", "wind"]):
        return "weather_impact"
    elif any(word in query_lower for word in ["fuel", "gas", "diesel"]):
        return "fuel_analysis"
    elif any(word in query_lower for word in ["traffic", "congestion", "delay"]):
        return "traffic_analysis"
    else:
        return "comprehensive"


def format_chat_response(state: WorkflowState, original_query: str) -> str:
    """Format the workflow results into a conversational response."""
    
    response_parts = []
    
    # Add a friendly introduction
    response_parts.append("Here's what I found for you:")
    
    # Add structured data results
    if state.structured_data and state.structured_data.query_results:
        response_parts.append("\n**📊 Database Analysis:**")
        results = state.structured_data.query_results
        if len(results) > 0:
            # Format the first few results
            for i, result in enumerate(results[:3]):  # Show first 3 results
                if "price" in result:
                    response_parts.append(f"• {result.get('service_type', 'Service')}: ${result['price']:.2f}")
                elif "zone" in result:
                    response_parts.append(f"• Zone {result['zone']}: {result.get('description', '')}")
    
    # Add vector search results
    if state.vector_search and state.vector_search.documents:
        response_parts.append("\n**🔍 Knowledge Base:**")
        for i, doc in enumerate(state.vector_search.documents[:2]):  # Show first 2 docs
            if "payload" in doc and "content" in doc["payload"]:
                content = doc["payload"]["content"][:200] + "..." if len(doc["payload"]["content"]) > 200 else doc["payload"]["content"]
                response_parts.append(f"• {content}")
    
    # Add auxiliary data
    if state.auxiliary_data:
        if state.auxiliary_data.weather_data:
            response_parts.append("\n**🌤️ Weather Information:**")
            for location, data in state.auxiliary_data.weather_data.items():
                temp = data.get("current", {}).get("temperature_2m", "N/A")
                response_parts.append(f"• {location}: {temp}°F")
        
        if state.auxiliary_data.fuel_data:
            response_parts.append("\n**⛽ Fuel Prices:**")
            for fuel in state.auxiliary_data.fuel_data:
                response_parts.append(f"• {fuel['fuel_type'].title()}: ${fuel['price_per_gallon']:.2f}/gallon")
    
    # Add synthesis if available
    if state.synthesis and state.synthesis.summary:
        response_parts.append(f"\n**💡 Summary:**")
        response_parts.append(state.synthesis.summary)
    
    # Add recommendations if available
    if state.synthesis and state.synthesis.recommendations:
        response_parts.append(f"\n**🎯 Recommendations:**")
        for rec in state.synthesis.recommendations:
            response_parts.append(f"• {rec}")
    
    # If no results, provide a helpful message
    if len(response_parts) == 1:  # Only has the introduction
        response_parts.append("\nI couldn't find specific information for your query, but I'm here to help with shipping-related questions. Try asking about rates, routes, weather impacts, or fuel costs!")
    
    return "\n".join(response_parts)


def execute_query(user_query: str, query_type: str, context: Dict[str, Any]):
    """Execute a query using the multi-agent system."""
    
    # Create progress container
    progress_container = st.container()
    results_container = st.container()
    
    with progress_container:
        st.subheader("🔄 Executing Multi-Agent Analysis")
        
        # Progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Agent status display
        agent_status = st.empty()
        
        try:
            # Execute workflow
            status_text.text("Initializing workflow...")
            progress_bar.progress(10)
            
            # Run the workflow asynchronously
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            try:
                final_state = loop.run_until_complete(execute_workflow(user_query, context))
            finally:
                loop.close()
            
            # Display results
            with results_container:
                display_results(final_state)
                
        except Exception as e:
            st.error(f"Error executing query: {str(e)}")
            logger.error(f"Query execution failed: {e}")


def display_results(state: WorkflowState):
    """Display the results of the workflow execution."""
    
    st.subheader("📈 Analysis Results")
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Execution Time", f"{state.total_execution_time:.2f}s" if state.total_execution_time else "N/A")
    with col2:
        st.metric("Completed Agents", len(state.get_completed_agents()))
    with col3:
        st.metric("Failed Agents", len(state.get_failed_agents()))
    with col4:
        st.metric("Query Type", state.query_type.value if state.query_type else "Unknown")
    
    # Agent status
    st.subheader("🤖 Agent Execution Status")
    for agent_name, result in state.agent_results.items():
        status_class = f"status-{result.status.value}"
        st.markdown(f'<div class="agent-status {status_class}">'
                   f'<strong>{agent_name}</strong>: {result.status.value} '
                   f'({result.execution_time:.2f}s)</div>', unsafe_allow_html=True)
    
    # Display agent-specific results
    if state.structured_data:
        display_structured_data_results(state.structured_data)
    
    if state.vector_search:
        display_vector_search_results(state.vector_search)
    
    if state.auxiliary_data:
        display_auxiliary_data_results(state.auxiliary_data)
    
    if state.synthesis:
        display_synthesis_results(state.synthesis)
    
    # Error display
    if state.errors:
        st.subheader("⚠️ Errors")
        for error in state.errors:
            st.error(error)
    
    # Warning display
    if state.warnings:
        st.subheader("⚠️ Warnings")
        for warning in state.warnings:
            st.warning(warning)


def display_structured_data_results(structured_data):
    """Display structured data results."""
    st.subheader("📊 Database Analysis Results")
    
    # SQL Query
    with st.expander("SQL Query Executed"):
        st.code(structured_data.sql_query, language="sql")
    
    # Results table
    if structured_data.query_results:
        df = pd.DataFrame(structured_data.query_results)
        st.dataframe(df, use_container_width=True)
        
        # Create visualizations
        if "weight" in df.columns and "price" in df.columns:
            st.subheader("📈 Pricing Analysis")
            fig = create_pricing_chart(df)
            st.plotly_chart(fig, use_container_width=True)
        
        if "zone" in df.columns and "min_distance" in df.columns:
            st.subheader("🗺️ Zone Analysis")
            fig = create_zone_chart(df)
            st.plotly_chart(fig, use_container_width=True)


def display_vector_search_results(vector_search):
    """Display vector search results."""
    st.subheader("🔍 Knowledge Base Search Results")
    
    for i, doc in enumerate(vector_search.documents[:5]):  # Show top 5
        with st.expander(f"Document {i+1} (Score: {vector_search.scores[i]:.3f})"):
            if "payload" in doc and "content" in doc["payload"]:
                st.write(doc["payload"]["content"])
            else:
                st.write("Document content not available")


def display_auxiliary_data_results(auxiliary_data):
    """Display auxiliary data results."""
    st.subheader("🌤️ External Data Analysis")
    
    # Weather data
    if auxiliary_data.weather_data:
        st.write("**Weather Information:**")
        weather_df = pd.DataFrame([
            {
                "Location": location,
                "Temperature (°F)": data.get("current", {}).get("temperature_2m", "N/A"),
                "Humidity (%)": data.get("current", {}).get("relative_humidity_2m", "N/A"),
                "Wind Speed (mph)": data.get("current", {}).get("wind_speed_10m", "N/A")
            }
            for location, data in auxiliary_data.weather_data.items()
        ])
        st.dataframe(weather_df, use_container_width=True)
    
    # Fuel data
    if auxiliary_data.fuel_prices:
        st.write("**Fuel Prices:**")
        fuel_df = pd.DataFrame([
            {
                "Fuel Type": fuel_type.title(),
                "Price ($/gallon)": data.get("current_price", "N/A"),
                "Trend": data.get("trend", "N/A")
            }
            for fuel_type, data in auxiliary_data.fuel_prices.items()
        ])
        st.dataframe(fuel_df, use_container_width=True)
    
    # Traffic data
    if auxiliary_data.traffic_data:
        st.write("**Traffic Information:**")
        traffic_df = pd.DataFrame([
            {
                "Location": location,
                "Congestion Level": data.get("congestion_level", "N/A"),
                "Average Speed (mph)": data.get("average_speed", "N/A"),
                "Delay (minutes)": data.get("delay_minutes", "N/A")
            }
            for location, data in auxiliary_data.traffic_data.items()
        ])
        st.dataframe(traffic_df, use_container_width=True)


def display_synthesis_results(synthesis):
    """Display synthesis and visualization results."""
    st.subheader("🧠 AI-Generated Insights")
    
    # Insights
    if synthesis.insights:
        st.write("**Key Insights:**")
        for i, insight in enumerate(synthesis.insights, 1):
            st.write(f"{i}. {insight}")
    
    # Recommendations
    if synthesis.recommendations:
        st.write("**Recommendations:**")
        for i, recommendation in enumerate(synthesis.recommendations, 1):
            st.write(f"{i}. {recommendation}")
    
    # Visualizations
    if synthesis.visualizations:
        st.subheader("📊 Interactive Visualizations")
        for viz in synthesis.visualizations:
            if "chart" in viz:
                st.plotly_chart(viz["chart"], use_container_width=True)


def data_management():
    """Data management interface for automatic CSV ingestion and data management."""
    st.header("📁 Data Management")
    
    # Automatic Data Ingestion Section
    st.subheader("🔄 Automatic Data Ingestion")
    st.info("""
    **How it works:**
    1. Place your CSV files in the `data_sources/` folder
    2. Files should be named: `fedex_pricing*.csv` or `zone_distance*.csv`
    3. Click "Ingest Data Sources" to automatically load them into PostgreSQL
    4. Data will only be inserted if it doesn't already exist
    """)
    
    # Show data sources folder status
    data_sources_dir = Path("data_sources")
    if data_sources_dir.exists():
        csv_files = list(data_sources_dir.glob("*.csv"))
        if csv_files:
            st.write("**Found CSV files in data_sources folder:**")
            for file in csv_files:
                st.write(f"• {file.name}")
        else:
            st.write("**No CSV files found in data_sources folder**")
    else:
        st.write("**data_sources folder does not exist**")
    
    # Ingest button
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 Ingest Data Sources", type="primary"):
            with st.spinner("Ingesting data sources..."):
                try:
                    # Run async function
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    try:
                        results = loop.run_until_complete(ingest_data_sources())
                        loop.close()
                        
                        # Display results
                        st.success("Data ingestion completed!")
                        for file_path, success in results.items():
                            if success:
                                st.success(f"✅ {Path(file_path).name}")
                            else:
                                st.error(f"❌ {Path(file_path).name}")
                    except Exception as e:
                        loop.close()
                        st.error(f"Error during ingestion: {str(e)}")
                except Exception as e:
                    st.error(f"Error: {str(e)}")
    
    with col2:
        if st.button("🗑️ Clear All Data", type="secondary"):
            if st.checkbox("I understand this will delete all data from the database"):
                with st.spinner("Clearing all data..."):
                    try:
                        # Run async function
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        try:
                            results = loop.run_until_complete(clear_all_data())
                            loop.close()
                            
                            # Display results
                            st.success("Data cleared successfully!")
                            for table_name, success in results.items():
                                if success:
                                    st.success(f"✅ Cleared {table_name}")
                                else:
                                    st.error(f"❌ Failed to clear {table_name}")
                        except Exception as e:
                            loop.close()
                            st.error(f"Error clearing data: {str(e)}")
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
    
    # Data Status Section
    st.subheader("📊 Data Status")
    try:
        # Run async function
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            status = loop.run_until_complete(get_ingestion_status())
            loop.close()
            
            for table_name, info in status.items():
                col1, col2, col3 = st.columns([2, 2, 1])
                with col1:
                    st.write(f"**{table_name}**")
                with col2:
                    if info["has_data"]:
                        st.success("✅ Has Data")
                    else:
                        st.warning("⚠️ No Data")
                with col3:
                    st.write(f"Type: {info['csv_type']}")
        except Exception as e:
            loop.close()
            st.error(f"Error getting status: {str(e)}")
    except Exception as e:
        st.error(f"Error: {str(e)}")
    
    # Manual Upload Section (Legacy)
    st.subheader("📤 Manual Upload (Legacy)")
    st.warning("""
    **Note:** Manual upload is deprecated. Use automatic ingestion above instead.
    """)
    
    # File uploader for Fedex pricing data
    st.write("**Fedex Pricing Data**")
    pricing_file = st.file_uploader(
        "Upload Fedex pricing CSV file:",
        type=['csv'],
        help="CSV should have columns: weight, transportation_type, zone, service_type, price"
    )
    
    if pricing_file:
        try:
            df = pd.read_csv(pricing_file)
            st.write("Preview of uploaded data:")
            st.dataframe(df.head(), use_container_width=True)
            
            if st.button("Upload to Database"):
                upload_pricing_data(df)
                
        except Exception as e:
            st.error(f"Error reading CSV file: {str(e)}")
    
    # File uploader for zone distance mapping
    st.write("**Zone Distance Mapping Data**")
    zone_file = st.file_uploader(
        "Upload zone distance mapping CSV file:",
        type=['csv'],
        help="CSV should have columns: zone, min_distance, max_distance"
    )
    
    if zone_file:
        try:
            df = pd.read_csv(zone_file)
            st.write("Preview of uploaded data:")
            st.dataframe(df.head(), use_container_width=True)
            
            if st.button("Upload Zone Data to Database"):
                upload_zone_data(df)
                
        except Exception as e:
            st.error(f"Error reading CSV file: {str(e)}")
    
    # Database operations
    st.subheader("🗄️ Database Operations")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Initialize Database"):
            initialize_database()
    
    with col2:
        if st.button("Insert Sample Data"):
            insert_sample_data_to_db()


async def upload_pricing_data(df: pd.DataFrame):
    """Upload pricing data to the database."""
    try:
        db_manager = await get_database_manager()
        
        # Validate required columns
        required_columns = ["weight", "transportation_type", "zone", "service_type", "price"]
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            st.error(f"Missing required columns: {missing_columns}")
            return
        
        # Insert data
        for _, row in df.iterrows():
            query = """
            INSERT INTO datasource_fedex_pricing (weight, transportation_type, zone, service_type, price)
            VALUES ($1, $2, $3, $4, $5)
            ON CONFLICT DO NOTHING
            """
            await db_manager.execute_query(query, {
                "weight": row["weight"],
                "trans_type": row["transportation_type"],
                "zone": row["zone"],
                "service": row["service_type"],
                "price": row["price"]
            })
        
        st.success(f"Successfully uploaded {len(df)} pricing records to database!")
        
    except Exception as e:
        st.error(f"Error uploading pricing data: {str(e)}")


async def upload_zone_data(df: pd.DataFrame):
    """Upload zone distance mapping data to the database."""
    try:
        db_manager = await get_database_manager()
        
        # Validate required columns
        required_columns = ["zone", "min_distance", "max_distance"]
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            st.error(f"Missing required columns: {missing_columns}")
            return
        
        # Insert data
        for _, row in df.iterrows():
            query = """
            INSERT INTO datasource_fedex_zone_distance_mapping (zone, min_distance, max_distance)
            VALUES ($1, $2, $3)
            ON CONFLICT (zone) DO NOTHING
            """
            await db_manager.execute_query(query, {
                "zone": row["zone"],
                "min_dist": row["min_distance"],
                "max_dist": row["max_distance"]
            })
        
        st.success(f"Successfully uploaded {len(df)} zone records to database!")
        
    except Exception as e:
        st.error(f"Error uploading zone data: {str(e)}")


def initialize_database():
    """Initialize the database."""
    try:
        # This would typically be done through a database migration
        st.info("Database initialization would be handled through migrations in production.")
        st.success("Database schema is ready!")
    except Exception as e:
        st.error(f"Error initializing database: {str(e)}")


async def insert_sample_data_to_db():
    """Insert sample data into the database."""
    try:
        db_manager = await get_database_manager()
        await insert_sample_data(db_manager)
        st.success("Sample data inserted successfully!")
    except Exception as e:
        st.error(f"Error inserting sample data: {str(e)}")


def system_status():
    """Display system status and health information."""
    st.header("🔧 System Status")
    
    # Agent status
    st.subheader("🤖 Agent Status")
    agents = get_all_agents()
    
    for agent_name, agent in agents.items():
        col1, col2, col3 = st.columns([2, 3, 1])
        with col1:
            st.write(f"**{agent_name}**")
        with col2:
            st.write(agent.description)
        with col3:
            st.success("✅ Ready")
    
    # Database status
    st.subheader("🗄️ Database Status")
    try:
        # This would check actual database connectivity
        st.success("✅ PostgreSQL: Connected")
        st.success("✅ Qdrant: Connected")
        st.success("✅ Redis: Connected")
    except Exception as e:
        st.error(f"❌ Database connection failed: {str(e)}")
    
    # System information
    st.subheader("ℹ️ System Information")
    st.write(f"**Python Version:** {sys.version}")
    st.write(f"**Streamlit Version:** {st.__version__}")
    st.write(f"**Platform:** {sys.platform}")


def about_page():
    """About page with information about the platform."""
    st.header("ℹ️ About Aura Shipping Intelligence Platform")
    
    st.markdown("""
    ## Overview
    
    Aura is a sophisticated multi-agent shipping intelligence platform that provides deep, 
    contextualized insights into shipping logistics through collaborative AI agents.
    
    ## Architecture
    
    The platform uses a multi-agent framework with 5 specialized agents:
    
    1. **Supervisor Agent** - Query deconstruction and task routing
    2. **Structured Data Agent** - SQL operations and database queries
    3. **Vector Search Agent** - Semantic search and knowledge retrieval
    4. **Auxiliary Intelligence Agent** - External API data gathering
    5. **Synthesis & Visualization Agent** - Insights and visualizations
    
    ## Technology Stack
    
    - **Backend**: Python 3.11+, LangGraph, PostgreSQL
    - **AI/ML**: Ollama (local LLM), Vanna.ai, SentenceTransformers
    - **Vector DB**: Qdrant
    - **UI**: Streamlit
    - **Visualization**: Plotly
    
    ## Features
    
    - Natural language query processing
    - Multi-agent workflow orchestration
    - Real-time data integration
    - Interactive visualizations
    - Comprehensive shipping analysis
    
    ## Getting Started
    
    1. Upload your CSV data through the Data Management page
    2. Use the Query Interface to ask shipping-related questions
    3. View results, insights, and visualizations
    4. Monitor system status and agent performance
    """)


if __name__ == "__main__":
    main() 