"""
Pytest configuration and fixtures for Aura Shipping Intelligence Platform tests.
"""

import asyncio
import os
import sys
from pathlib import Path
from typing import AsyncGenerator, Dict, Any

import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, MagicMock

# Add the src directory to the path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from core.database import DatabaseManager
from core.state import WorkflowState, AgentStatus, QueryType
from agents.base import BaseAgent, AgentResult


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def sample_workflow_state():
    """Create a sample workflow state for testing."""
    return WorkflowState(
        query="What are the Fedex rates for a 5-pound package?",
        query_type=QueryType.PRICING,
        context={"weight": 5.0, "origin": "NYC", "destination": "LA"}
    )


@pytest.fixture
def sample_agent_result():
    """Create a sample agent result for testing."""
    return AgentResult(
        status=AgentStatus.COMPLETED,
        execution_time=1.5,
        data={"result": "test data"},
        error=None
    )


@pytest.fixture
def mock_database_manager():
    """Create a mock database manager for testing."""
    mock_db = AsyncMock(spec=DatabaseManager)
    
    # Mock successful query execution
    mock_db.execute_query.return_value = [
        {"weight": 5.0, "price": 12.50, "zone": 2},
        {"weight": 5.0, "price": 15.75, "zone": 3}
    ]
    
    # Mock connection methods
    mock_db.connect.return_value = None
    mock_db.disconnect.return_value = None
    
    return mock_db


@pytest.fixture
def mock_ollama_client():
    """Create a mock Ollama client for testing."""
    mock_client = AsyncMock()
    mock_client.chat.return_value = {
        "message": {
            "content": "This is a test response from the LLM."
        }
    }
    return mock_client


@pytest.fixture
def mock_qdrant_client():
    """Create a mock Qdrant client for testing."""
    mock_client = MagicMock()
    mock_client.search.return_value = [
        {
            "id": "doc1",
            "score": 0.85,
            "payload": {"content": "Test document content"}
        }
    ]
    return mock_client


@pytest.fixture
def sample_fedex_pricing_data():
    """Sample Fedex pricing data for testing."""
    return [
        {"weight": 1.0, "transportation_type": "ground", "zone": 1, "service_type": "ground", "price": 8.50},
        {"weight": 2.0, "transportation_type": "ground", "zone": 1, "service_type": "ground", "price": 9.25},
        {"weight": 5.0, "transportation_type": "ground", "zone": 2, "service_type": "ground", "price": 12.75},
        {"weight": 10.0, "transportation_type": "express", "zone": 3, "service_type": "express", "price": 25.50},
    ]


@pytest.fixture
def sample_zone_distance_data():
    """Sample zone distance mapping data for testing."""
    return [
        {"zone": 1, "min_distance": 0, "max_distance": 150},
        {"zone": 2, "min_distance": 151, "max_distance": 300},
        {"zone": 3, "min_distance": 301, "max_distance": 600},
        {"zone": 4, "min_distance": 601, "max_distance": 1000},
        {"zone": 5, "min_distance": 1001, "max_distance": 1400},
    ]


@pytest.fixture
def sample_weather_data():
    """Sample weather data for testing."""
    return {
        "NYC": {
            "current": {
                "temperature_2m": 72.5,
                "relative_humidity_2m": 65,
                "wind_speed_10m": 12.3
            }
        },
        "LA": {
            "current": {
                "temperature_2m": 78.2,
                "relative_humidity_2m": 45,
                "wind_speed_10m": 8.7
            }
        }
    }


@pytest.fixture
def sample_fuel_data():
    """Sample fuel price data for testing."""
    return {
        "diesel": {
            "current_price": 3.85,
            "trend": "increasing"
        },
        "gasoline": {
            "current_price": 3.25,
            "trend": "stable"
        }
    }


@pytest.fixture
def sample_traffic_data():
    """Sample traffic data for testing."""
    return {
        "NYC": {
            "congestion_level": "moderate",
            "average_speed": 25.5,
            "delay_minutes": 8.2
        },
        "LA": {
            "congestion_level": "high",
            "average_speed": 18.3,
            "delay_minutes": 15.7
        }
    }


@pytest.fixture
def test_env_vars():
    """Set up test environment variables."""
    original_env = os.environ.copy()
    
    # Set test environment variables
    test_env = {
        "DATABASE_URL": "postgresql://test:test@localhost:5432/test_db",
        "QDRANT_URL": "http://localhost:6333",
        "REDIS_URL": "redis://localhost:6379",
        "OLLAMA_BASE_URL": "http://localhost:11434",
        "OLLAMA_MODEL": "llama3.2",
        "OPENWEATHER_API_KEY": "test_weather_key",
        "FUEL_API_KEY": "test_fuel_key",
        "TRAFFIC_API_KEY": "test_traffic_key",
        "LOG_LEVEL": "DEBUG"
    }
    
    os.environ.update(test_env)
    
    yield test_env
    
    # Restore original environment
    os.environ.clear()
    os.environ.update(original_env)


@pytest.fixture
def mock_http_client():
    """Create a mock HTTP client for testing external APIs."""
    mock_client = AsyncMock()
    
    # Mock weather API response
    mock_client.get.return_value = AsyncMock(
        status_code=200,
        json=AsyncMock(return_value={
            "current": {
                "temperature_2m": 72.5,
                "relative_humidity_2m": 65,
                "wind_speed_10m": 12.3
            }
        })
    )
    
    return mock_client


@pytest.fixture
def mock_vanna_client():
    """Create a mock Vanna client for testing SQL generation."""
    mock_client = MagicMock()
    mock_client.generate_sql.return_value = "SELECT * FROM datasource_fedex_pricing WHERE weight = 5.0"
    mock_client.run.return_value = [
        {"weight": 5.0, "price": 12.50, "zone": 2},
        {"weight": 5.0, "price": 15.75, "zone": 3}
    ]
    return mock_client


@pytest.fixture
def mock_sentence_transformer():
    """Create a mock SentenceTransformer for testing embeddings."""
    mock_model = MagicMock()
    mock_model.encode.return_value = [[0.1, 0.2, 0.3, 0.4, 0.5]] * 10  # 10-dimensional embeddings
    return mock_model 