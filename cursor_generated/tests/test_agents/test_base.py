"""
Tests for the base agent functionality.
"""

import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Dict, Any

from agents.base import BaseAgent, AgentResult, AgentStatus, get_all_agents


class TestBaseAgent:
    """Test cases for BaseAgent class."""
    
    def test_base_agent_creation(self):
        """Test creating a new BaseAgent."""
        agent = BaseAgent(
            name="test_agent",
            description="Test agent for unit testing"
        )
        
        assert agent.name == "test_agent"
        assert agent.description == "Test agent for unit testing"
        assert agent.max_retries == 3
        assert agent.timeout == 300
    
    def test_base_agent_with_custom_params(self):
        """Test creating BaseAgent with custom parameters."""
        agent = BaseAgent(
            name="custom_agent",
            description="Custom test agent",
            max_retries=5,
            timeout=600
        )
        
        assert agent.name == "custom_agent"
        assert agent.description == "Custom test agent"
        assert agent.max_retries == 5
        assert agent.timeout == 600
    
    @pytest.mark.asyncio
    async def test_execute_success(self):
        """Test successful agent execution."""
        agent = BaseAgent(
            name="test_agent",
            description="Test agent"
        )
        
        # Mock the _execute method
        agent._execute = AsyncMock(return_value={"result": "success"})
        
        result = await agent.execute({"test": "data"})
        
        assert result.status == AgentStatus.COMPLETED
        assert result.data == {"result": "success"}
        assert result.error is None
        assert result.execution_time > 0
    
    @pytest.mark.asyncio
    async def test_execute_with_error(self):
        """Test agent execution with error."""
        agent = BaseAgent(
            name="test_agent",
            description="Test agent"
        )
        
        # Mock the _execute method to raise an exception
        agent._execute = AsyncMock(side_effect=Exception("Test error"))
        
        result = await agent.execute({"test": "data"})
        
        assert result.status == AgentStatus.FAILED
        assert result.data == {}
        assert result.error == "Test error"
        assert result.execution_time > 0
    
    @pytest.mark.asyncio
    async def test_execute_with_retries(self):
        """Test agent execution with retries."""
        agent = BaseAgent(
            name="test_agent",
            description="Test agent",
            max_retries=2
        )
        
        # Mock the _execute method to fail twice then succeed
        mock_execute = AsyncMock()
        mock_execute.side_effect = [Exception("Error 1"), Exception("Error 2"), {"result": "success"}]
        agent._execute = mock_execute
        
        result = await agent.execute({"test": "data"})
        
        assert result.status == AgentStatus.COMPLETED
        assert result.data == {"result": "success"}
        assert result.error is None
        assert mock_execute.call_count == 3
    
    @pytest.mark.asyncio
    async def test_execute_max_retries_exceeded(self):
        """Test agent execution when max retries are exceeded."""
        agent = BaseAgent(
            name="test_agent",
            description="Test agent",
            max_retries=2
        )
        
        # Mock the _execute method to always fail
        agent._execute = AsyncMock(side_effect=Exception("Persistent error"))
        
        result = await agent.execute({"test": "data"})
        
        assert result.status == AgentStatus.FAILED
        assert result.data == {}
        assert "Persistent error" in result.error
        assert agent._execute.call_count == 3  # Initial + 2 retries
    
    @pytest.mark.asyncio
    async def test_execute_timeout(self):
        """Test agent execution timeout."""
        agent = BaseAgent(
            name="test_agent",
            description="Test agent",
            timeout=0.1  # Very short timeout
        )
        
        # Mock the _execute method to take longer than timeout
        async def slow_execute(context):
            import asyncio
            await asyncio.sleep(0.2)  # Longer than timeout
            return {"result": "success"}
        
        agent._execute = slow_execute
        
        result = await agent.execute({"test": "data"})
        
        assert result.status == AgentStatus.FAILED
        assert "timeout" in result.error.lower()
    
    def test_get_cache_key(self):
        """Test cache key generation."""
        agent = BaseAgent(
            name="test_agent",
            description="Test agent"
        )
        
        context = {"weight": 5.0, "origin": "NYC", "destination": "LA"}
        cache_key = agent.get_cache_key(context)
        
        assert "test_agent" in cache_key
        assert "weight" in cache_key
        assert "origin" in cache_key
        assert "destination" in cache_key
    
    def test_get_cache_key_empty_context(self):
        """Test cache key generation with empty context."""
        agent = BaseAgent(
            name="test_agent",
            description="Test agent"
        )
        
        cache_key = agent.get_cache_key({})
        assert "test_agent" in cache_key
    
    @pytest.mark.asyncio
    async def test_execute_with_cache_hit(self):
        """Test agent execution with cache hit."""
        agent = BaseAgent(
            name="test_agent",
            description="Test agent"
        )
        
        # Mock cache to return a cached result
        cached_result = AgentResult(
            status=AgentStatus.COMPLETED,
            execution_time=0.1,
            data={"cached": "data"}
        )
        
        with patch.object(agent, 'get_cached_result', return_value=cached_result):
            result = await agent.execute({"test": "data"})
            
            assert result == cached_result
            assert result.data == {"cached": "data"}
    
    @pytest.mark.asyncio
    async def test_execute_with_cache_miss(self):
        """Test agent execution with cache miss."""
        agent = BaseAgent(
            name="test_agent",
            description="Test agent"
        )
        
        # Mock cache to return None (cache miss)
        with patch.object(agent, 'get_cached_result', return_value=None):
            # Mock the _execute method
            agent._execute = AsyncMock(return_value={"result": "fresh"})
            
            # Mock cache storage
            with patch.object(agent, 'cache_result'):
                result = await agent.execute({"test": "data"})
                
                assert result.status == AgentStatus.COMPLETED
                assert result.data == {"result": "fresh"}
                agent.cache_result.assert_called_once()
    
    def test_to_dict(self):
        """Test converting agent to dictionary."""
        agent = BaseAgent(
            name="test_agent",
            description="Test agent description",
            max_retries=3,
            timeout=300
        )
        
        agent_dict = agent.to_dict()
        
        assert agent_dict["name"] == "test_agent"
        assert agent_dict["description"] == "Test agent description"
        assert agent_dict["max_retries"] == 3
        assert agent_dict["timeout"] == 300


class TestAgentResult:
    """Test cases for AgentResult class."""
    
    def test_agent_result_creation(self):
        """Test creating a new AgentResult."""
        result = AgentResult(
            status=AgentStatus.COMPLETED,
            execution_time=1.5,
            data={"test": "data"}
        )
        
        assert result.status == AgentStatus.COMPLETED
        assert result.execution_time == 1.5
        assert result.data == {"test": "data"}
        assert result.error is None
    
    def test_agent_result_with_error(self):
        """Test creating AgentResult with error."""
        result = AgentResult(
            status=AgentStatus.FAILED,
            execution_time=0.5,
            data={},
            error="Test error message"
        )
        
        assert result.status == AgentStatus.FAILED
        assert result.error == "Test error message"
    
    def test_agent_result_to_dict(self):
        """Test converting AgentResult to dictionary."""
        result = AgentResult(
            status=AgentStatus.COMPLETED,
            execution_time=1.5,
            data={"price": 12.50},
            error=None
        )
        
        result_dict = result.to_dict()
        
        assert result_dict["status"] == "completed"
        assert result_dict["execution_time"] == 1.5
        assert result_dict["data"] == {"price": 12.50}
        assert result_dict["error"] is None
    
    def test_agent_result_to_dict_with_error(self):
        """Test converting AgentResult with error to dictionary."""
        result = AgentResult(
            status=AgentStatus.FAILED,
            execution_time=0.5,
            data={},
            error="Database connection failed"
        )
        
        result_dict = result.to_dict()
        
        assert result_dict["status"] == "failed"
        assert result_dict["execution_time"] == 0.5
        assert result_dict["data"] == {}
        assert result_dict["error"] == "Database connection failed"


class TestAgentFunctions:
    """Test cases for agent utility functions."""
    
    def test_get_all_agents(self):
        """Test getting all available agents."""
        agents = get_all_agents()
        
        # Should return a dictionary of agent instances
        assert isinstance(agents, dict)
        assert len(agents) > 0
        
        # Check that all agents are instances of BaseAgent
        for agent_name, agent in agents.items():
            assert isinstance(agent, BaseAgent)
            assert agent.name == agent_name
        
        # Check for expected agent names
        expected_agents = [
            "supervisor",
            "structured_data", 
            "vector_search",
            "auxiliary_intelligence",
            "synthesis_visualization"
        ]
        
        for expected_agent in expected_agents:
            assert expected_agent in agents


class TestAgentCaching:
    """Test cases for agent caching functionality."""
    
    @pytest.mark.asyncio
    async def test_cache_operations(self):
        """Test cache get and set operations."""
        agent = BaseAgent(
            name="test_agent",
            description="Test agent"
        )
        
        context = {"weight": 5.0, "origin": "NYC"}
        cache_key = agent.get_cache_key(context)
        
        # Test cache miss
        cached_result = agent.get_cached_result(cache_key)
        assert cached_result is None
        
        # Test cache set and get
        test_result = AgentResult(
            status=AgentStatus.COMPLETED,
            execution_time=1.0,
            data={"cached": "data"}
        )
        
        agent.cache_result(cache_key, test_result)
        
        # Retrieve from cache
        retrieved_result = agent.get_cached_result(cache_key)
        assert retrieved_result is not None
        assert retrieved_result.data == {"cached": "data"}
    
    def test_cache_key_consistency(self):
        """Test that cache keys are consistent for same input."""
        agent = BaseAgent(
            name="test_agent",
            description="Test agent"
        )
        
        context = {"weight": 5.0, "origin": "NYC", "destination": "LA"}
        
        key1 = agent.get_cache_key(context)
        key2 = agent.get_cache_key(context)
        
        assert key1 == key2
    
    def test_cache_key_uniqueness(self):
        """Test that cache keys are unique for different inputs."""
        agent = BaseAgent(
            name="test_agent",
            description="Test agent"
        )
        
        context1 = {"weight": 5.0, "origin": "NYC"}
        context2 = {"weight": 10.0, "origin": "NYC"}
        
        key1 = agent.get_cache_key(context1)
        key2 = agent.get_cache_key(context2)
        
        assert key1 != key2


class TestAgentErrorHandling:
    """Test cases for agent error handling."""
    
    @pytest.mark.asyncio
    async def test_handle_execution_error(self):
        """Test handling of execution errors."""
        agent = BaseAgent(
            name="test_agent",
            description="Test agent"
        )
        
        # Test with a simple exception
        try:
            raise ValueError("Test value error")
        except Exception as e:
            result = agent.handle_execution_error(e, 1.5)
            
            assert result.status == AgentStatus.FAILED
            assert result.execution_time == 1.5
            assert result.data == {}
            assert "Test value error" in result.error
    
    @pytest.mark.asyncio
    async def test_handle_timeout_error(self):
        """Test handling of timeout errors."""
        agent = BaseAgent(
            name="test_agent",
            description="Test agent"
        )
        
        # Test with timeout exception
        try:
            import asyncio
            raise asyncio.TimeoutError("Operation timed out")
        except Exception as e:
            result = agent.handle_execution_error(e, 0.5)
            
            assert result.status == AgentStatus.FAILED
            assert result.execution_time == 0.5
            assert "timeout" in result.error.lower()
    
    @pytest.mark.asyncio
    async def test_handle_connection_error(self):
        """Test handling of connection errors."""
        agent = BaseAgent(
            name="test_agent",
            description="Test agent"
        )
        
        # Test with connection exception
        try:
            raise ConnectionError("Connection refused")
        except Exception as e:
            result = agent.handle_execution_error(e, 2.0)
            
            assert result.status == AgentStatus.FAILED
            assert result.execution_time == 2.0
            assert "connection" in result.error.lower() 