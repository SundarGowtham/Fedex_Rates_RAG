"""
Core state management for the Aura Shipping Intelligence Platform.

This module defines the state objects and management system used across
all agents in the multi-agent workflow.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, validator


class QueryType(str, Enum):
    """Types of queries that can be processed by the system."""
    
    PRICING = "pricing"
    ROUTE_OPTIMIZATION = "route_optimization"
    WEATHER_IMPACT = "weather_impact"
    FUEL_ANALYSIS = "fuel_analysis"
    TRAFFIC_ANALYSIS = "traffic_analysis"
    COMPREHENSIVE = "comprehensive"
    UNKNOWN = "unknown"


class AgentStatus(str, Enum):
    """Status of agent execution."""
    
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class AgentResult(BaseModel):
    """Result from a single agent execution."""
    
    agent_name: str
    status: AgentStatus
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    execution_time: Optional[float] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class StructuredDataResult(BaseModel):
    """Result from structured data agent."""
    
    sql_query: str
    query_results: List[Dict[str, Any]]
    metadata: Dict[str, Any] = Field(default_factory=dict)
    execution_time: float
    row_count: int


class VectorSearchResult(BaseModel):
    """Result from vector search agent."""
    
    query: str
    documents: List[Dict[str, Any]]
    scores: List[float]
    metadata: Dict[str, Any] = Field(default_factory=dict)
    execution_time: float


class AuxiliaryDataResult(BaseModel):
    """Result from auxiliary intelligence agent."""
    
    weather_data: Optional[Dict[str, Any]] = None
    fuel_prices: Optional[Dict[str, Any]] = None
    traffic_data: Optional[Dict[str, Any]] = None
    execution_time: float
    sources: List[str] = Field(default_factory=list)


class VisualizationResult(BaseModel):
    """Result from synthesis and visualization agent."""
    
    insights: List[str]
    visualizations: List[Dict[str, Any]]
    recommendations: List[str]
    execution_time: float
    chart_data: Optional[Dict[str, Any]] = None


class WorkflowState(BaseModel):
    """
    Main state object for the multi-agent workflow.
    
    This object is passed between agents and contains all necessary
    information for the workflow execution.
    """
    
    # Core identifiers
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    query_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    
    # User query and context
    user_query: str
    query_type: QueryType = QueryType.UNKNOWN
    query_context: Dict[str, Any] = Field(default_factory=dict)
    
    # Agent execution tracking
    agent_results: Dict[str, AgentResult] = Field(default_factory=dict)
    current_agent: Optional[str] = None
    workflow_status: AgentStatus = AgentStatus.PENDING
    
    # Structured data results
    structured_data: Optional[StructuredDataResult] = None
    
    # Vector search results
    vector_search: Optional[VectorSearchResult] = None
    
    # Auxiliary intelligence results
    auxiliary_data: Optional[AuxiliaryDataResult] = None
    
    # Synthesis and visualization results
    synthesis: Optional[VisualizationResult] = None
    
    # Metadata
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    execution_start: Optional[datetime] = None
    execution_end: Optional[datetime] = None
    
    # Error handling
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    
    # Performance metrics
    total_execution_time: Optional[float] = None
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
    
    @validator('updated_at', pre=True, always=True)
    def update_timestamp(cls, v):
        """Update timestamp whenever state is modified."""
        return datetime.utcnow()
    
    def add_agent_result(self, agent_name: str, result: AgentResult) -> None:
        """Add a result from an agent execution."""
        self.agent_results[agent_name] = result
        self.updated_at = datetime.utcnow()
    
    def get_agent_result(self, agent_name: str) -> Optional[AgentResult]:
        """Get result from a specific agent."""
        return self.agent_results.get(agent_name)
    
    def is_agent_completed(self, agent_name: str) -> bool:
        """Check if a specific agent has completed successfully."""
        result = self.get_agent_result(agent_name)
        return result is not None and result.status == AgentStatus.COMPLETED
    
    def get_failed_agents(self) -> List[str]:
        """Get list of agents that failed."""
        return [
            name for name, result in self.agent_results.items()
            if result.status == AgentStatus.FAILED
        ]
    
    def get_completed_agents(self) -> List[str]:
        """Get list of successfully completed agents."""
        return [
            name for name, result in self.agent_results.items()
            if result.status == AgentStatus.COMPLETED
        ]
    
    def add_error(self, error: str) -> None:
        """Add an error message to the state."""
        self.errors.append(error)
        self.updated_at = datetime.utcnow()
    
    def add_warning(self, warning: str) -> None:
        """Add a warning message to the state."""
        self.warnings.append(warning)
        self.updated_at = datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert state to dictionary for serialization."""
        return json.loads(self.json())
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> WorkflowState:
        """Create state from dictionary."""
        return cls(**data)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the workflow state."""
        return {
            "session_id": self.session_id,
            "query_id": self.query_id,
            "user_query": self.user_query,
            "query_type": self.query_type.value,
            "workflow_status": self.workflow_status.value,
            "completed_agents": self.get_completed_agents(),
            "failed_agents": self.get_failed_agents(),
            "total_execution_time": self.total_execution_time,
            "error_count": len(self.errors),
            "warning_count": len(self.warnings),
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }


class StateManager:
    """Manager for workflow state operations."""
    
    def __init__(self):
        self.active_states: Dict[str, WorkflowState] = {}
    
    def create_state(self, user_query: str, query_context: Optional[Dict[str, Any]] = None) -> WorkflowState:
        """Create a new workflow state."""
        state = WorkflowState(
            user_query=user_query,
            query_context=query_context or {}
        )
        self.active_states[state.session_id] = state
        return state
    
    def get_state(self, session_id: str) -> Optional[WorkflowState]:
        """Get state by session ID."""
        return self.active_states.get(session_id)
    
    def update_state(self, state: WorkflowState) -> None:
        """Update an existing state."""
        self.active_states[state.session_id] = state
    
    def remove_state(self, session_id: str) -> None:
        """Remove a state from active tracking."""
        self.active_states.pop(session_id, None)
    
    def get_all_states(self) -> Dict[str, WorkflowState]:
        """Get all active states."""
        return self.active_states.copy()
    
    def cleanup_old_states(self, max_age_hours: int = 24) -> None:
        """Clean up states older than specified hours."""
        cutoff_time = datetime.utcnow().timestamp() - (max_age_hours * 3600)
        
        states_to_remove = [
            session_id for session_id, state in self.active_states.items()
            if state.created_at.timestamp() < cutoff_time
        ]
        
        for session_id in states_to_remove:
            self.remove_state(session_id) 