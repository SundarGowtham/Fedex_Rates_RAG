"""
Tests for the WorkflowState and state management classes.
"""

import pytest
from datetime import datetime
from typing import Dict, Any

from core.state import WorkflowState, AgentStatus, QueryType, AgentResult


class TestWorkflowState:
    """Test cases for WorkflowState class."""
    
    def test_workflow_state_creation(self):
        """Test creating a new WorkflowState."""
        state = WorkflowState(
            query="Test query",
            query_type=QueryType.PRICING,
            context={"weight": 5.0}
        )
        
        assert state.query == "Test query"
        assert state.query_type == QueryType.PRICING
        assert state.context == {"weight": 5.0}
        assert state.agent_results == {}
        assert state.errors == []
        assert state.warnings == []
        assert state.total_execution_time is None
    
    def test_workflow_state_with_optional_params(self):
        """Test creating WorkflowState with optional parameters."""
        state = WorkflowState(
            query="Test query",
            query_type=QueryType.COMPREHENSIVE,
            context={"origin": "NYC", "destination": "LA"},
            session_id="test-session-123"
        )
        
        assert state.session_id == "test-session-123"
        assert state.query_type == QueryType.COMPREHENSIVE
    
    def test_add_agent_result(self):
        """Test adding agent results to the state."""
        state = WorkflowState(
            query="Test query",
            query_type=QueryType.PRICING
        )
        
        result = AgentResult(
            status=AgentStatus.COMPLETED,
            execution_time=1.5,
            data={"price": 12.50}
        )
        
        state.add_agent_result("test_agent", result)
        
        assert "test_agent" in state.agent_results
        assert state.agent_results["test_agent"] == result
    
    def test_add_error(self):
        """Test adding errors to the state."""
        state = WorkflowState(
            query="Test query",
            query_type=QueryType.PRICING
        )
        
        state.add_error("Database connection failed")
        state.add_error("API timeout")
        
        assert len(state.errors) == 2
        assert "Database connection failed" in state.errors
        assert "API timeout" in state.errors
    
    def test_add_warning(self):
        """Test adding warnings to the state."""
        state = WorkflowState(
            query="Test query",
            query_type=QueryType.PRICING
        )
        
        state.add_warning("Slow response time")
        state.add_warning("Partial data available")
        
        assert len(state.warnings) == 2
        assert "Slow response time" in state.warnings
        assert "Partial data available" in state.warnings
    
    def test_get_completed_agents(self):
        """Test getting list of completed agents."""
        state = WorkflowState(
            query="Test query",
            query_type=QueryType.PRICING
        )
        
        # Add different agent results
        state.add_agent_result("agent1", AgentResult(
            status=AgentStatus.COMPLETED,
            execution_time=1.0,
            data={}
        ))
        state.add_agent_result("agent2", AgentResult(
            status=AgentStatus.FAILED,
            execution_time=0.5,
            data={}
        ))
        state.add_agent_result("agent3", AgentResult(
            status=AgentStatus.COMPLETED,
            execution_time=2.0,
            data={}
        ))
        
        completed = state.get_completed_agents()
        assert len(completed) == 2
        assert "agent1" in completed
        assert "agent3" in completed
        assert "agent2" not in completed
    
    def test_get_failed_agents(self):
        """Test getting list of failed agents."""
        state = WorkflowState(
            query="Test query",
            query_type=QueryType.PRICING
        )
        
        # Add different agent results
        state.add_agent_result("agent1", AgentResult(
            status=AgentStatus.COMPLETED,
            execution_time=1.0,
            data={}
        ))
        state.add_agent_result("agent2", AgentResult(
            status=AgentStatus.FAILED,
            execution_time=0.5,
            data={}
        ))
        state.add_agent_result("agent3", AgentResult(
            status=AgentStatus.FAILED,
            execution_time=2.0,
            data={}
        ))
        
        failed = state.get_failed_agents()
        assert len(failed) == 2
        assert "agent2" in failed
        assert "agent3" in failed
        assert "agent1" not in failed
    
    def test_calculate_total_execution_time(self):
        """Test calculating total execution time."""
        state = WorkflowState(
            query="Test query",
            query_type=QueryType.PRICING
        )
        
        # Add agent results with execution times
        state.add_agent_result("agent1", AgentResult(
            status=AgentStatus.COMPLETED,
            execution_time=1.5,
            data={}
        ))
        state.add_agent_result("agent2", AgentResult(
            status=AgentStatus.COMPLETED,
            execution_time=2.3,
            data={}
        ))
        state.add_agent_result("agent3", AgentResult(
            status=AgentStatus.COMPLETED,
            execution_time=0.8,
            data={}
        ))
        
        total_time = state.calculate_total_execution_time()
        assert total_time == 4.6
    
    def test_to_dict(self):
        """Test converting state to dictionary."""
        state = WorkflowState(
            query="Test query",
            query_type=QueryType.PRICING,
            context={"weight": 5.0},
            session_id="test-session"
        )
        
        state.add_agent_result("test_agent", AgentResult(
            status=AgentStatus.COMPLETED,
            execution_time=1.5,
            data={"result": "test"}
        ))
        state.add_error("Test error")
        state.add_warning("Test warning")
        
        state_dict = state.to_dict()
        
        assert state_dict["query"] == "Test query"
        assert state_dict["query_type"] == "pricing"
        assert state_dict["context"] == {"weight": 5.0}
        assert state_dict["session_id"] == "test-session"
        assert "test_agent" in state_dict["agent_results"]
        assert "Test error" in state_dict["errors"]
        assert "Test warning" in state_dict["warnings"]


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
            error="Connection timeout"
        )
        
        assert result.status == AgentStatus.FAILED
        assert result.error == "Connection timeout"
    
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


class TestQueryType:
    """Test cases for QueryType enum."""
    
    def test_query_type_values(self):
        """Test that all expected query types exist."""
        assert QueryType.PRICING == "pricing"
        assert QueryType.ROUTE_OPTIMIZATION == "route_optimization"
        assert QueryType.WEATHER_IMPACT == "weather_impact"
        assert QueryType.FUEL_ANALYSIS == "fuel_analysis"
        assert QueryType.TRAFFIC_ANALYSIS == "traffic_analysis"
        assert QueryType.COMPREHENSIVE == "comprehensive"
    
    def test_query_type_list(self):
        """Test getting list of all query types."""
        query_types = list(QueryType)
        expected_types = [
            "pricing",
            "route_optimization", 
            "weather_impact",
            "fuel_analysis",
            "traffic_analysis",
            "comprehensive"
        ]
        
        assert len(query_types) == len(expected_types)
        for query_type in expected_types:
            assert query_type in query_types


class TestAgentStatus:
    """Test cases for AgentStatus enum."""
    
    def test_agent_status_values(self):
        """Test that all expected agent statuses exist."""
        assert AgentStatus.PENDING == "pending"
        assert AgentStatus.RUNNING == "running"
        assert AgentStatus.COMPLETED == "completed"
        assert AgentStatus.FAILED == "failed"
    
    def test_agent_status_list(self):
        """Test getting list of all agent statuses."""
        statuses = list(AgentStatus)
        expected_statuses = ["pending", "running", "completed", "failed"]
        
        assert len(statuses) == len(expected_statuses)
        for status in expected_statuses:
            assert status in statuses 