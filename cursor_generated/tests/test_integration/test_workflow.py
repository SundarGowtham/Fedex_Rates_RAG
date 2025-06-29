"""
Integration tests for the workflow execution.
"""

import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Dict, Any

from core.workflow import execute_workflow
from core.state import WorkflowState, QueryType, AgentStatus
from agents.base import AgentResult


class TestWorkflowExecution:
    """Test cases for workflow execution."""
    
    @pytest.mark.asyncio
    async def test_workflow_execution_success(self, sample_workflow_state):
        """Test successful workflow execution."""
        # Mock all agents to return successful results
        with patch('core.workflow.get_all_agents') as mock_get_agents:
            mock_agents = {
                'supervisor': AsyncMock(),
                'structured_data': AsyncMock(),
                'vector_search': AsyncMock(),
                'auxiliary_intelligence': AsyncMock(),
                'synthesis_visualization': AsyncMock()
            }
            
            # Set up successful agent results
            for agent_name, mock_agent in mock_agents.items():
                mock_agent.execute.return_value = AgentResult(
                    status=AgentStatus.COMPLETED,
                    execution_time=1.0,
                    data={f"{agent_name}_result": "success"}
                )
            
            mock_get_agents.return_value = mock_agents
            
            # Execute workflow
            final_state = await execute_workflow(
                "What are the Fedex rates for a 5-pound package?",
                {"weight": 5.0, "origin": "NYC", "destination": "LA"}
            )
            
            # Verify final state
            assert isinstance(final_state, WorkflowState)
            assert final_state.query == "What are the Fedex rates for a 5-pound package?"
            assert final_state.query_type == QueryType.PRICING
            assert len(final_state.agent_results) == 5
            
            # Verify all agents were executed
            for agent_name in mock_agents:
                assert agent_name in final_state.agent_results
                assert final_state.agent_results[agent_name].status == AgentStatus.COMPLETED
    
    @pytest.mark.asyncio
    async def test_workflow_execution_with_agent_failure(self, sample_workflow_state):
        """Test workflow execution when some agents fail."""
        with patch('core.workflow.get_all_agents') as mock_get_agents:
            mock_agents = {
                'supervisor': AsyncMock(),
                'structured_data': AsyncMock(),
                'vector_search': AsyncMock(),
                'auxiliary_intelligence': AsyncMock(),
                'synthesis_visualization': AsyncMock()
            }
            
            # Set up mixed results (some success, some failure)
            mock_agents['supervisor'].execute.return_value = AgentResult(
                status=AgentStatus.COMPLETED,
                execution_time=1.0,
                data={"supervisor_result": "success"}
            )
            
            mock_agents['structured_data'].execute.return_value = AgentResult(
                status=AgentStatus.FAILED,
                execution_time=0.5,
                data={},
                error="Database connection failed"
            )
            
            mock_agents['vector_search'].execute.return_value = AgentResult(
                status=AgentStatus.COMPLETED,
                execution_time=1.2,
                data={"vector_search_result": "success"}
            )
            
            mock_agents['auxiliary_intelligence'].execute.return_value = AgentResult(
                status=AgentStatus.FAILED,
                execution_time=0.3,
                data={},
                error="API timeout"
            )
            
            mock_agents['synthesis_visualization'].execute.return_value = AgentResult(
                status=AgentStatus.COMPLETED,
                execution_time=1.5,
                data={"synthesis_result": "success"}
            )
            
            mock_get_agents.return_value = mock_agents
            
            # Execute workflow
            final_state = await execute_workflow(
                "What are the Fedex rates for a 5-pound package?",
                {"weight": 5.0}
            )
            
            # Verify final state
            assert isinstance(final_state, WorkflowState)
            assert len(final_state.agent_results) == 5
            
            # Check completed agents
            completed_agents = final_state.get_completed_agents()
            assert len(completed_agents) == 3
            assert "supervisor" in completed_agents
            assert "vector_search" in completed_agents
            assert "synthesis_visualization" in completed_agents
            
            # Check failed agents
            failed_agents = final_state.get_failed_agents()
            assert len(failed_agents) == 2
            assert "structured_data" in failed_agents
            assert "auxiliary_intelligence" in failed_agents
    
    @pytest.mark.asyncio
    async def test_workflow_execution_timeout(self, sample_workflow_state):
        """Test workflow execution with timeout."""
        with patch('core.workflow.get_all_agents') as mock_get_agents:
            mock_agents = {
                'supervisor': AsyncMock(),
                'structured_data': AsyncMock(),
                'vector_search': AsyncMock(),
                'auxiliary_intelligence': AsyncMock(),
                'synthesis_visualization': AsyncMock()
            }
            
            # Set up agents to take too long
            async def slow_execute(context):
                import asyncio
                await asyncio.sleep(2.0)  # Longer than timeout
                return {"result": "success"}
            
            for agent_name, mock_agent in mock_agents.items():
                mock_agent.execute.side_effect = slow_execute
            
            mock_get_agents.return_value = mock_agents
            
            # Execute workflow with short timeout
            with patch('core.workflow.WORKFLOW_TIMEOUT', 1.0):
                final_state = await execute_workflow(
                    "What are the Fedex rates for a 5-pound package?",
                    {"weight": 5.0}
                )
            
            # Verify that some agents failed due to timeout
            failed_agents = final_state.get_failed_agents()
            assert len(failed_agents) > 0
    
    @pytest.mark.asyncio
    async def test_workflow_with_different_query_types(self):
        """Test workflow execution with different query types."""
        query_tests = [
            ("What are the Fedex rates for a 5-pound package?", QueryType.PRICING),
            ("What's the best route from NYC to LA?", QueryType.ROUTE_OPTIMIZATION),
            ("How does weather affect shipping from Chicago?", QueryType.WEATHER_IMPACT),
            ("What are current fuel prices and trends?", QueryType.FUEL_ANALYSIS),
            ("What's the traffic situation in Miami?", QueryType.TRAFFIC_ANALYSIS),
            ("Give me a comprehensive shipping analysis", QueryType.COMPREHENSIVE)
        ]
        
        with patch('core.workflow.get_all_agents') as mock_get_agents:
            mock_agents = {
                'supervisor': AsyncMock(),
                'structured_data': AsyncMock(),
                'vector_search': AsyncMock(),
                'auxiliary_intelligence': AsyncMock(),
                'synthesis_visualization': AsyncMock()
            }
            
            # Set up successful agent results
            for agent_name, mock_agent in mock_agents.items():
                mock_agent.execute.return_value = AgentResult(
                    status=AgentStatus.COMPLETED,
                    execution_time=1.0,
                    data={f"{agent_name}_result": "success"}
                )
            
            mock_get_agents.return_value = mock_agents
            
            for query, expected_type in query_tests:
                final_state = await execute_workflow(query, {})
                
                assert final_state.query == query
                assert final_state.query_type == expected_type
    
    @pytest.mark.asyncio
    async def test_workflow_error_handling(self, sample_workflow_state):
        """Test workflow error handling."""
        with patch('core.workflow.get_all_agents') as mock_get_agents:
            mock_agents = {
                'supervisor': AsyncMock(),
                'structured_data': AsyncMock(),
                'vector_search': AsyncMock(),
                'auxiliary_intelligence': AsyncMock(),
                'synthesis_visualization': AsyncMock()
            }
            
            # Set up agents to raise exceptions
            mock_agents['supervisor'].execute.side_effect = Exception("Supervisor error")
            mock_agents['structured_data'].execute.side_effect = ValueError("Database error")
            mock_agents['vector_search'].execute.side_effect = ConnectionError("Network error")
            
            # Set up successful agents
            mock_agents['auxiliary_intelligence'].execute.return_value = AgentResult(
                status=AgentStatus.COMPLETED,
                execution_time=1.0,
                data={"auxiliary_result": "success"}
            )
            mock_agents['synthesis_visualization'].execute.return_value = AgentResult(
                status=AgentStatus.COMPLETED,
                execution_time=1.0,
                data={"synthesis_result": "success"}
            )
            
            mock_get_agents.return_value = mock_agents
            
            # Execute workflow
            final_state = await execute_workflow(
                "What are the Fedex rates for a 5-pound package?",
                {"weight": 5.0}
            )
            
            # Verify error handling
            assert len(final_state.errors) > 0
            assert len(final_state.get_failed_agents()) == 3
            assert len(final_state.get_completed_agents()) == 2
    
    @pytest.mark.asyncio
    async def test_workflow_state_persistence(self, sample_workflow_state):
        """Test that workflow state persists throughout execution."""
        with patch('core.workflow.get_all_agents') as mock_get_agents:
            mock_agents = {
                'supervisor': AsyncMock(),
                'structured_data': AsyncMock(),
                'vector_search': AsyncMock(),
                'auxiliary_intelligence': AsyncMock(),
                'synthesis_visualization': AsyncMock()
            }
            
            # Set up successful agent results
            for agent_name, mock_agent in mock_agents.items():
                mock_agent.execute.return_value = AgentResult(
                    status=AgentStatus.COMPLETED,
                    execution_time=1.0,
                    data={f"{agent_name}_result": "success"}
                )
            
            mock_get_agents.return_value = mock_agents
            
            # Execute workflow
            final_state = await execute_workflow(
                "What are the Fedex rates for a 5-pound package?",
                {"weight": 5.0, "origin": "NYC", "destination": "LA"}
            )
            
            # Verify state persistence
            assert final_state.query == "What are the Fedex rates for a 5-pound package?"
            assert final_state.context == {"weight": 5.0, "origin": "NYC", "destination": "LA"}
            assert final_state.session_id is not None
            assert final_state.total_execution_time is not None
            assert final_state.total_execution_time > 0
    
    @pytest.mark.asyncio
    async def test_workflow_parallel_execution(self, sample_workflow_state):
        """Test that agents can execute in parallel."""
        import asyncio
        import time
        
        with patch('core.workflow.get_all_agents') as mock_get_agents:
            mock_agents = {
                'supervisor': AsyncMock(),
                'structured_data': AsyncMock(),
                'vector_search': AsyncMock(),
                'auxiliary_intelligence': AsyncMock(),
                'synthesis_visualization': AsyncMock()
            }
            
            # Set up agents to simulate work with delays
            async def delayed_execute(context, delay=0.1):
                await asyncio.sleep(delay)
                return {"result": "success"}
            
            for agent_name, mock_agent in mock_agents.items():
                mock_agent.execute.side_effect = lambda context, name=agent_name: delayed_execute(context, 0.1)
            
            mock_get_agents.return_value = mock_agents
            
            # Execute workflow and measure time
            start_time = time.time()
            final_state = await execute_workflow(
                "What are the Fedex rates for a 5-pound package?",
                {"weight": 5.0}
            )
            end_time = time.time()
            
            execution_time = end_time - start_time
            
            # If agents run in parallel, total time should be less than sum of individual times
            # (5 agents * 0.1s each = 0.5s, but parallel execution should be ~0.1s)
            assert execution_time < 0.5
            assert execution_time > 0.05  # Should take some time
    
    @pytest.mark.asyncio
    async def test_workflow_with_empty_context(self, sample_workflow_state):
        """Test workflow execution with empty context."""
        with patch('core.workflow.get_all_agents') as mock_get_agents:
            mock_agents = {
                'supervisor': AsyncMock(),
                'structured_data': AsyncMock(),
                'vector_search': AsyncMock(),
                'auxiliary_intelligence': AsyncMock(),
                'synthesis_visualization': AsyncMock()
            }
            
            # Set up successful agent results
            for agent_name, mock_agent in mock_agents.items():
                mock_agent.execute.return_value = AgentResult(
                    status=AgentStatus.COMPLETED,
                    execution_time=1.0,
                    data={f"{agent_name}_result": "success"}
                )
            
            mock_get_agents.return_value = mock_agents
            
            # Execute workflow with empty context
            final_state = await execute_workflow(
                "What are the Fedex rates?",
                {}
            )
            
            # Verify workflow still executes
            assert isinstance(final_state, WorkflowState)
            assert final_state.context == {}
            assert len(final_state.agent_results) == 5
    
    @pytest.mark.asyncio
    async def test_workflow_result_validation(self, sample_workflow_state):
        """Test that workflow results are properly validated."""
        with patch('core.workflow.get_all_agents') as mock_get_agents:
            mock_agents = {
                'supervisor': AsyncMock(),
                'structured_data': AsyncMock(),
                'vector_search': AsyncMock(),
                'auxiliary_intelligence': AsyncMock(),
                'synthesis_visualization': AsyncMock()
            }
            
            # Set up agents with different result types
            mock_agents['supervisor'].execute.return_value = AgentResult(
                status=AgentStatus.COMPLETED,
                execution_time=1.0,
                data={"query_type": "pricing", "intent": "rate_inquiry"}
            )
            
            mock_agents['structured_data'].execute.return_value = AgentResult(
                status=AgentStatus.COMPLETED,
                execution_time=1.5,
                data={"sql_query": "SELECT * FROM pricing", "results": [{"price": 12.50}]}
            )
            
            mock_agents['vector_search'].execute.return_value = AgentResult(
                status=AgentStatus.COMPLETED,
                execution_time=0.8,
                data={"documents": [{"content": "Fedex pricing guide"}], "scores": [0.85]}
            )
            
            mock_agents['auxiliary_intelligence'].execute.return_value = AgentResult(
                status=AgentStatus.COMPLETED,
                execution_time=2.0,
                data={"weather": {"NYC": {"temp": 72}}, "fuel": {"diesel": 3.85}}
            )
            
            mock_agents['synthesis_visualization'].execute.return_value = AgentResult(
                status=AgentStatus.COMPLETED,
                execution_time=1.2,
                data={"insights": ["Ground shipping is cheapest"], "recommendations": ["Use ground service"]}
            )
            
            mock_get_agents.return_value = mock_agents
            
            # Execute workflow
            final_state = await execute_workflow(
                "What are the Fedex rates for a 5-pound package?",
                {"weight": 5.0}
            )
            
            # Validate results
            assert final_state.structured_data is not None
            assert "sql_query" in final_state.structured_data.data
            assert "results" in final_state.structured_data.data
            
            assert final_state.vector_search is not None
            assert "documents" in final_state.vector_search.data
            assert "scores" in final_state.vector_search.data
            
            assert final_state.auxiliary_data is not None
            assert "weather" in final_state.auxiliary_data.data
            assert "fuel" in final_state.auxiliary_data.data
            
            assert final_state.synthesis is not None
            assert "insights" in final_state.synthesis.data
            assert "recommendations" in final_state.synthesis.data 