"""
Base agent class for the Aura Shipping Intelligence Platform.

This module defines the base agent class that all specialized agents
inherit from, providing common functionality and interfaces.
"""

from __future__ import annotations

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from langchain_core.messages import BaseMessage
from pydantic import BaseModel, Field

from core.state import AgentResult, AgentStatus, WorkflowState

logger = logging.getLogger(__name__)


class AgentConfig(BaseModel):
    """Configuration for agent execution."""
    
    name: str
    description: str
    timeout_seconds: int = 300
    max_retries: int = 3
    retry_delay_seconds: int = 5
    enable_caching: bool = True
    cache_ttl_seconds: int = 3600
    
    class Config:
        frozen = True


class AgentContext(BaseModel):
    """Context information passed to agents during execution."""
    
    session_id: str
    query_id: str
    user_query: str
    query_context: Dict[str, Any] = Field(default_factory=dict)
    agent_config: AgentConfig
    dependencies: List[str] = Field(default_factory=list)
    
    class Config:
        frozen = True


class AgentResponse(BaseModel):
    """Response from agent execution."""
    
    success: bool
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    execution_time: float
    metadata: Dict[str, Any] = Field(default_factory=dict)
    messages: List[BaseMessage] = Field(default_factory=list)


class BaseAgent(ABC):
    """
    Base class for all agents in the Aura platform.
    
    This class provides common functionality for agent execution,
    error handling, logging, and state management.
    """
    
    def __init__(self, config: AgentConfig):
        self.config = config
        self.logger = logging.getLogger(f"agent.{config.name}")
        self._cache: Dict[str, Any] = {}
    
    @property
    def name(self) -> str:
        """Get the agent name."""
        return self.config.name
    
    @property
    def description(self) -> str:
        """Get the agent description."""
        return self.config.description
    
    def get_cache_key(self, context: AgentContext, **kwargs) -> str:
        """Generate a cache key for the given context and parameters."""
        import hashlib
        
        # Create a unique key based on context and parameters
        key_data = {
            "session_id": context.session_id,
            "query_id": context.query_id,
            "user_query": context.user_query,
            "agent_name": self.name,
            **kwargs
        }
        
        key_string = str(sorted(key_data.items()))
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def get_cached_result(self, cache_key: str) -> Optional[AgentResponse]:
        """Get cached result if available and not expired."""
        if not self.config.enable_caching:
            return None
        
        if cache_key in self._cache:
            cached_data = self._cache[cache_key]
            if time.time() - cached_data["timestamp"] < self.config.cache_ttl_seconds:
                self.logger.info(f"Using cached result for key: {cache_key}")
                return cached_data["response"]
            else:
                # Remove expired cache entry
                del self._cache[cache_key]
        
        return None
    
    def cache_result(self, cache_key: str, response: AgentResponse) -> None:
        """Cache the result for future use."""
        if not self.config.enable_caching:
            return
        
        self._cache[cache_key] = {
            "response": response,
            "timestamp": time.time()
        }
        self.logger.info(f"Cached result for key: {cache_key}")
    
    async def execute(self, context: AgentContext, state: WorkflowState) -> AgentResult:
        """
        Execute the agent with the given context and state.
        
        This is the main entry point for agent execution. It handles
        caching, retries, error handling, and result formatting.
        """
        start_time = time.time()
        cache_key = self.get_cache_key(context)
        
        # Check cache first
        cached_result = self.get_cached_result(cache_key)
        if cached_result:
            return AgentResult(
                agent_name=self.name,
                status=AgentStatus.COMPLETED,
                data=cached_result.data,
                execution_time=cached_result.execution_time,
                timestamp=state.updated_at
            )
        
        # Execute with retries
        for attempt in range(self.config.max_retries + 1):
            try:
                self.logger.info(f"Executing agent {self.name} (attempt {attempt + 1})")
                
                # Validate dependencies
                if not self._check_dependencies(state, context.dependencies):
                    return AgentResult(
                        agent_name=self.name,
                        status=AgentStatus.SKIPPED,
                        error="Dependencies not met",
                        execution_time=0.0,
                        timestamp=state.updated_at
                    )
                
                # Execute the agent
                response = await self._execute_agent(context, state)
                
                # Cache successful result
                if response.success:
                    self.cache_result(cache_key, response)
                
                execution_time = time.time() - start_time
                
                return AgentResult(
                    agent_name=self.name,
                    status=AgentStatus.COMPLETED if response.success else AgentStatus.FAILED,
                    data=response.data,
                    error=response.error,
                    execution_time=execution_time,
                    timestamp=state.updated_at
                )
                
            except Exception as e:
                self.logger.error(f"Agent {self.name} execution failed (attempt {attempt + 1}): {e}")
                
                if attempt < self.config.max_retries:
                    await asyncio.sleep(self.config.retry_delay_seconds)
                    continue
                else:
                    execution_time = time.time() - start_time
                    return AgentResult(
                        agent_name=self.name,
                        status=AgentStatus.FAILED,
                        error=str(e),
                        execution_time=execution_time,
                        timestamp=state.updated_at
                    )
    
    def _check_dependencies(self, state: WorkflowState, dependencies: List[str]) -> bool:
        """Check if all dependencies are satisfied."""
        for dependency in dependencies:
            if not state.is_agent_completed(dependency):
                self.logger.warning(f"Dependency {dependency} not completed for agent {self.name}")
                return False
        return True
    
    @abstractmethod
    async def _execute_agent(self, context: AgentContext, state: WorkflowState) -> AgentResponse:
        """
        Execute the specific agent logic.
        
        This method must be implemented by each agent subclass.
        It should contain the actual agent-specific logic.
        """
        pass
    
    async def preprocess(self, context: AgentContext, state: WorkflowState) -> None:
        """
        Preprocess the context and state before execution.
        
        This method can be overridden by subclasses to perform
        any necessary preprocessing.
        """
        pass
    
    async def postprocess(self, context: AgentContext, state: WorkflowState, result: AgentResult) -> None:
        """
        Postprocess the result after execution.
        
        This method can be overridden by subclasses to perform
        any necessary postprocessing.
        """
        pass
    
    def validate_input(self, context: AgentContext, state: WorkflowState) -> bool:
        """
        Validate the input context and state.
        
        This method can be overridden by subclasses to perform
        input validation.
        """
        return True
    
    def get_required_dependencies(self) -> List[str]:
        """
        Get the list of agent dependencies.
        
        This method can be overridden by subclasses to specify
        which agents must complete before this agent can run.
        """
        return []
    
    def get_optional_dependencies(self) -> List[str]:
        """
        Get the list of optional agent dependencies.
        
        This method can be overridden by subclasses to specify
        which agents are optional dependencies.
        """
        return []
    
    def cleanup(self) -> None:
        """Clean up any resources used by the agent."""
        self._cache.clear()
    
    def get_agent_info(self) -> Dict[str, Any]:
        """Get information about the agent."""
        return {
            "name": self.name,
            "description": self.description,
            "dependencies": self.get_required_dependencies(),
            "optional_dependencies": self.get_optional_dependencies(),
            "config": self.config.dict()
        }


class AgentRegistry:
    """Registry for managing all available agents."""
    
    def __init__(self):
        self._agents: Dict[str, BaseAgent] = {}
    
    def register(self, agent: BaseAgent) -> None:
        """Register an agent in the registry."""
        self._agents[agent.name] = agent
        logger.info(f"Registered agent: {agent.name}")
    
    def get_agent(self, name: str) -> Optional[BaseAgent]:
        """Get an agent by name."""
        return self._agents.get(name)
    
    def get_all_agents(self) -> Dict[str, BaseAgent]:
        """Get all registered agents."""
        return self._agents.copy()
    
    def get_agent_names(self) -> List[str]:
        """Get names of all registered agents."""
        return list(self._agents.keys())
    
    def unregister(self, name: str) -> None:
        """Unregister an agent from the registry."""
        if name in self._agents:
            agent = self._agents.pop(name)
            agent.cleanup()
            logger.info(f"Unregistered agent: {name}")
    
    def clear(self) -> None:
        """Clear all registered agents."""
        for agent in self._agents.values():
            agent.cleanup()
        self._agents.clear()
        logger.info("Cleared all agents from registry")


# Global agent registry
agent_registry = AgentRegistry()


def register_agent(agent: BaseAgent) -> None:
    """Register an agent in the global registry."""
    agent_registry.register(agent)


def get_agent(name: str) -> Optional[BaseAgent]:
    """Get an agent from the global registry."""
    return agent_registry.get_agent(name)


def get_all_agents() -> Dict[str, BaseAgent]:
    """Get all agents from the global registry."""
    return agent_registry.get_all_agents() 