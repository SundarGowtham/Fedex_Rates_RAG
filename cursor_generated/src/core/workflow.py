"""
Workflow management for the Aura Shipping Intelligence Platform.

This module defines the LangGraph workflow that orchestrates the
multi-agent execution with proper state management and error handling.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from langchain.schema import BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph

from agents.base import AgentContext, AgentConfig, BaseAgent, get_agent
from core.state import AgentStatus, WorkflowState

logger = logging.getLogger(__name__)


class WorkflowManager:
    """
    Manager for the multi-agent workflow execution.
    
    This class handles the orchestration of agents using LangGraph,
    including state management, error handling, and result aggregation.
    """
    
    def __init__(self):
        self.graph: Optional[StateGraph] = None
        self.workflow_config: Dict[str, Any] = {}
    
    def build_workflow(self) -> StateGraph:
        """
        Build the LangGraph workflow with all agents.
        
        This method creates the workflow graph with proper node
        connections and conditional routing.
        """
        # Create the state graph
        workflow = StateGraph(WorkflowState)
        
        # Add nodes for each agent
        workflow.add_node("supervisor", self._supervisor_node)
        workflow.add_node("structured_data", self._structured_data_node)
        workflow.add_node("vector_search", self._vector_search_node)
        workflow.add_node("auxiliary_intelligence", self._auxiliary_intelligence_node)
        workflow.add_node("synthesis_visualization", self._synthesis_visualization_node)
        
        # Define the workflow edges
        workflow.set_entry_point("supervisor")
        
        # Supervisor routes to other agents based on query type
        workflow.add_conditional_edges(
            "supervisor",
            self._route_from_supervisor,
            {
                "structured_data": "structured_data",
                "vector_search": "vector_search",
                "auxiliary_intelligence": "auxiliary_intelligence",
                "synthesis_visualization": "synthesis_visualization",
                "end": END
            }
        )
        
        # Structured data can run in parallel with vector search
        workflow.add_edge("structured_data", "synthesis_visualization")
        
        # Vector search can run in parallel with structured data
        workflow.add_edge("vector_search", "synthesis_visualization")
        
        # Auxiliary intelligence can run in parallel
        workflow.add_edge("auxiliary_intelligence", "synthesis_visualization")
        
        # Synthesis is the final step
        workflow.add_edge("synthesis_visualization", END)
        
        self.graph = workflow.compile()
        logger.info("Workflow graph built successfully")
        
        return self.graph
    
    async def _supervisor_node(self, state: WorkflowState) -> WorkflowState:
        """Execute the supervisor agent."""
        return await self._execute_agent_node("supervisor", state)
    
    async def _structured_data_node(self, state: WorkflowState) -> WorkflowState:
        """Execute the structured data agent."""
        return await self._execute_agent_node("structured_data", state)
    
    async def _vector_search_node(self, state: WorkflowState) -> WorkflowState:
        """Execute the vector search agent."""
        return await self._execute_agent_node("vector_search", state)
    
    async def _auxiliary_intelligence_node(self, state: WorkflowState) -> WorkflowState:
        """Execute the auxiliary intelligence agent."""
        return await self._execute_agent_node("auxiliary_intelligence", state)
    
    async def _synthesis_visualization_node(self, state: WorkflowState) -> WorkflowState:
        """Execute the synthesis and visualization agent."""
        return await self._execute_agent_node("synthesis_visualization", state)
    
    async def _execute_agent_node(self, agent_name: str, state: WorkflowState) -> WorkflowState:
        """Execute a specific agent node."""
        try:
            # Mark workflow as running
            if state.workflow_status == AgentStatus.PENDING:
                state.workflow_status = AgentStatus.RUNNING
                state.execution_start = datetime.utcnow()
            
            # Get the agent
            agent = get_agent(agent_name)
            if not agent:
                error_msg = f"Agent {agent_name} not found"
                state.add_error(error_msg)
                logger.error(error_msg)
                return state
            
            # Create agent context
            context = AgentContext(
                session_id=state.session_id,
                query_id=state.query_id,
                user_query=state.user_query,
                query_context=state.query_context,
                agent_config=agent.config,
                dependencies=agent.get_required_dependencies()
            )
            
            # Update current agent
            state.current_agent = agent_name
            
            # Execute the agent
            result = await agent.execute(context, state)
            
            # Add result to state
            state.add_agent_result(agent_name, result)
            
            # Update state with agent-specific results
            await self._update_state_with_result(state, agent_name, result)
            
            logger.info(f"Agent {agent_name} completed with status: {result.status}")
            
        except Exception as e:
            error_msg = f"Error executing agent {agent_name}: {str(e)}"
            state.add_error(error_msg)
            logger.error(error_msg)
            
            # Mark agent as failed
            failed_result = AgentResult(
                agent_name=agent_name,
                status=AgentStatus.FAILED,
                error=str(e),
                execution_time=0.0,
                timestamp=datetime.utcnow()
            )
            state.add_agent_result(agent_name, failed_result)
        
        return state
    
    async def _update_state_with_result(self, state: WorkflowState, agent_name: str, result: AgentResult) -> None:
        """Update state with agent-specific results."""
        if result.status != AgentStatus.COMPLETED or not result.data:
            return
        
        if agent_name == "structured_data":
            from core.state import StructuredDataResult
            state.structured_data = StructuredDataResult(**result.data)
        
        elif agent_name == "vector_search":
            from core.state import VectorSearchResult
            state.vector_search = VectorSearchResult(**result.data)
        
        elif agent_name == "auxiliary_intelligence":
            from core.state import AuxiliaryDataResult
            state.auxiliary_data = AuxiliaryDataResult(**result.data)
        
        elif agent_name == "synthesis_visualization":
            from core.state import VisualizationResult
            state.synthesis = VisualizationResult(**result.data)
    
    def _route_from_supervisor(self, state: WorkflowState) -> str:
        """
        Route from supervisor based on query type and state.
        
        This method determines which agents should be executed next
        based on the query type and current state.
        """
        # If supervisor hasn't completed, continue with supervisor
        if not state.is_agent_completed("supervisor"):
            return "supervisor"
        
        # Get supervisor result to determine routing
        supervisor_result = state.get_agent_result("supervisor")
        if not supervisor_result or supervisor_result.status != AgentStatus.COMPLETED:
            return "end"
        
        # Determine which agents to run based on query type
        query_type = state.query_type
        
        if query_type.value in ["pricing", "comprehensive"]:
            # For pricing queries, we need structured data
            if not state.is_agent_completed("structured_data"):
                return "structured_data"
        
        if query_type.value in ["route_optimization", "comprehensive"]:
            # For route optimization, we need vector search
            if not state.is_agent_completed("vector_search"):
                return "vector_search"
        
        if query_type.value in ["weather_impact", "fuel_analysis", "traffic_analysis", "comprehensive"]:
            # For environmental queries, we need auxiliary intelligence
            if not state.is_agent_completed("auxiliary_intelligence"):
                return "auxiliary_intelligence"
        
        # If all required agents are complete, proceed to synthesis
        if not state.is_agent_completed("synthesis_visualization"):
            return "synthesis_visualization"
        
        return "end"
    
    async def execute_workflow(self, user_query: str, query_context: Optional[Dict[str, Any]] = None) -> WorkflowState:
        """
        Execute the complete workflow for a user query.
        
        This method orchestrates the entire multi-agent workflow
        and returns the final state with all results.
        """
        # Create initial state
        state = WorkflowState(
            user_query=user_query,
            query_context=query_context or {}
        )
        
        # Build workflow if not already built
        if not self.graph:
            self.build_workflow()
        
        try:
            # Execute the workflow
            logger.info(f"Starting workflow execution for session: {state.session_id}")
            
            # Run the workflow
            final_state = await self.graph.ainvoke(
                state
            )
            
            # Calculate total execution time
            if final_state.execution_start:
                final_state.total_execution_time = (
                    datetime.utcnow() - final_state.execution_start
                ).total_seconds()
            
            # Mark workflow as completed
            final_state.workflow_status = AgentStatus.COMPLETED
            final_state.execution_end = datetime.utcnow()
            
            logger.info(f"Workflow completed for session: {state.session_id}")
            
            return final_state
            
        except Exception as e:
            error_msg = f"Workflow execution failed: {str(e)}"
            state.add_error(error_msg)
            state.workflow_status = AgentStatus.FAILED
            state.execution_end = datetime.utcnow()
            
            logger.error(error_msg)
            return state
    
    async def execute_parallel_agents(self, state: WorkflowState, agent_names: List[str]) -> WorkflowState:
        """
        Execute multiple agents in parallel.
        
        This method allows for parallel execution of independent agents
        to improve performance.
        """
        if not agent_names:
            return state
        
        # Create tasks for parallel execution
        tasks = []
        for agent_name in agent_names:
            agent = get_agent(agent_name)
            if agent and not state.is_agent_completed(agent_name):
                context = AgentContext(
                    session_id=state.session_id,
                    query_id=state.query_id,
                    user_query=state.user_query,
                    query_context=state.query_context,
                    agent_config=agent.config,
                    dependencies=agent.get_required_dependencies()
                )
                task = self._execute_agent_node(agent_name, state)
                tasks.append(task)
        
        # Execute tasks in parallel
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Update state with results
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"Parallel agent execution failed: {result}")
                else:
                    state = result
        
        return state
    
    def get_workflow_status(self, state: WorkflowState) -> Dict[str, Any]:
        """Get a summary of the workflow status."""
        return {
            "session_id": state.session_id,
            "query_id": state.query_id,
            "workflow_status": state.workflow_status.value,
            "current_agent": state.current_agent,
            "completed_agents": state.get_completed_agents(),
            "failed_agents": state.get_failed_agents(),
            "total_execution_time": state.total_execution_time,
            "error_count": len(state.errors),
            "warning_count": len(state.warnings),
            "created_at": state.created_at.isoformat(),
            "updated_at": state.updated_at.isoformat(),
        }
    
    def validate_workflow_state(self, state: WorkflowState) -> Tuple[bool, List[str]]:
        """Validate the workflow state."""
        errors = []
        
        # Check required fields
        if not state.user_query:
            errors.append("User query is required")
        
        if not state.session_id:
            errors.append("Session ID is required")
        
        # Check agent dependencies
        all_agents = get_all_agents()
        for agent_name, agent in all_agents.items():
            dependencies = agent.get_required_dependencies()
            for dependency in dependencies:
                if dependency not in all_agents:
                    errors.append(f"Agent {agent_name} depends on missing agent: {dependency}")
        
        return len(errors) == 0, errors


# Global workflow manager instance
workflow_manager = WorkflowManager()


async def execute_workflow(user_query: str, query_context: Optional[Dict[str, Any]] = None) -> WorkflowState:
    """Execute the workflow for a user query."""
    return await workflow_manager.execute_workflow(user_query, query_context)


def get_workflow_manager() -> WorkflowManager:
    """Get the global workflow manager instance."""
    return workflow_manager 