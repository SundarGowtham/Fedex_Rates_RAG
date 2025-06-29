"""
Configuration settings for the Aura Shipping Intelligence Platform.

This module manages all configuration settings including environment
variables, API keys, and system parameters.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from pydantic import Field, validator
from pydantic_settings import BaseSettings


class DatabaseSettings(BaseSettings):
    """Database configuration settings."""
    
    host: str = Field(default="localhost", env="DATABASE_HOST")
    port: int = Field(default=5432, env="DATABASE_PORT")
    database: str = Field(default="aura_db", env="DATABASE_NAME")
    user: str = Field(default="aura_user", env="DATABASE_USER")
    password: str = Field(default="aura_password", env="DATABASE_PASSWORD")
    min_size: int = Field(default=5, env="DATABASE_MIN_SIZE")
    max_size: int = Field(default=20, env="DATABASE_MAX_SIZE")
    
    @property
    def connection_string(self) -> str:
        """Get the database connection string."""
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"
    
    @property
    def async_connection_string(self) -> str:
        """Get the async database connection string."""
        return f"postgresql+asyncpg://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"
    
    class Config:
        env_prefix = "DATABASE_"


class QdrantSettings(BaseSettings):
    """Qdrant vector database configuration settings."""
    
    host: str = Field(default="localhost", env="QDRANT_HOST")
    port: int = Field(default=6333, env="QDRANT_PORT")
    api_key: Optional[str] = Field(default=None, env="QDRANT_API_KEY")
    collection_name: str = Field(default="aura_documents", env="QDRANT_COLLECTION")
    
    @property
    def url(self) -> str:
        """Get the Qdrant URL."""
        return f"http://{self.host}:{self.port}"
    
    class Config:
        env_prefix = "QDRANT_"


class RedisSettings(BaseSettings):
    """Redis cache configuration settings."""
    
    host: str = Field(default="localhost", env="REDIS_HOST")
    port: int = Field(default=6379, env="REDIS_PORT")
    password: Optional[str] = Field(default=None, env="REDIS_PASSWORD")
    database: int = Field(default=0, env="REDIS_DATABASE")
    
    @property
    def url(self) -> str:
        """Get the Redis URL."""
        if self.password:
            return f"redis://:{self.password}@{self.host}:{self.port}/{self.database}"
        return f"redis://{self.host}:{self.port}/{self.database}"
    
    class Config:
        env_prefix = "REDIS_"


class OllamaSettings(BaseSettings):
    """Ollama LLM configuration settings."""
    
    host: str = Field(default="localhost", env="OLLAMA_HOST")
    port: int = Field(default=11434, env="OLLAMA_PORT")
    model: str = Field(default="llama2", env="OLLAMA_MODEL")
    temperature: float = Field(default=0.1, env="OLLAMA_TEMPERATURE")
    max_tokens: int = Field(default=4096, env="OLLAMA_MAX_TOKENS")
    
    @property
    def url(self) -> str:
        """Get the Ollama URL."""
        return f"http://{self.host}:{self.port}"
    
    class Config:
        env_prefix = "OLLAMA_"


class APISettings(BaseSettings):
    """External API configuration settings."""
    
    # EIA (Energy Information Administration) API
    eia_api_key: Optional[str] = Field(default=None, env="EIA_API_KEY")
    eia_base_url: str = Field(default="https://api.eia.gov/v2", env="EIA_BASE_URL")
    
    # OpenWeatherMap API
    openweather_api_key: Optional[str] = Field(default=None, env="OPENWEATHER_API_KEY")
    
    # Open-Meteo Weather API
    openmeteo_base_url: str = Field(default="https://api.open-meteo.com/v1", env="OPENMETEO_BASE_URL")
    
    # Azure Maps API
    azure_maps_key: Optional[str] = Field(default=None, env="AZURE_MAPS_KEY")
    azure_maps_base_url: str = Field(default="https://atlas.microsoft.com", env="AZURE_MAPS_BASE_URL")
    
    # HERE Traffic API
    here_api_key: Optional[str] = Field(default=None, env="HERE_API_KEY")
    here_base_url: str = Field(default="https://traffic.ls.hereapi.com", env="HERE_BASE_URL")
    
    class Config:
        env_prefix = "API_"


class LoggingSettings(BaseSettings):
    """Logging configuration settings."""
    
    level: str = Field(default="INFO", env="LOG_LEVEL")
    format: str = Field(default="%(asctime)s - %(name)s - %(levelname)s - %(message)s", env="LOG_FORMAT")
    file_path: Optional[str] = Field(default=None, env="LOG_FILE_PATH")
    enable_console: bool = Field(default=True, env="LOG_ENABLE_CONSOLE")
    enable_file: bool = Field(default=False, env="LOG_ENABLE_FILE")
    
    @validator("level")
    def validate_log_level(cls, v: str) -> str:
        """Validate log level."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f"Invalid log level: {v}. Must be one of {valid_levels}")
        return v.upper()
    
    class Config:
        env_prefix = "LOG_"


class AgentSettings(BaseSettings):
    """Agent configuration settings."""
    
    # Supervisor Agent
    supervisor_timeout: int = Field(default=300, env="SUPERVISOR_TIMEOUT")
    supervisor_max_retries: int = Field(default=3, env="SUPERVISOR_MAX_RETRIES")
    
    # Structured Data Agent
    structured_data_timeout: int = Field(default=300, env="STRUCTURED_DATA_TIMEOUT")
    structured_data_max_retries: int = Field(default=3, env="STRUCTURED_DATA_MAX_RETRIES")
    
    # Vector Search Agent
    vector_search_timeout: int = Field(default=300, env="VECTOR_SEARCH_TIMEOUT")
    vector_search_max_retries: int = Field(default=3, env="VECTOR_SEARCH_MAX_RETRIES")
    embedding_model: str = Field(default="all-MiniLM-L6-v2", env="EMBEDDING_MODEL")
    
    # Auxiliary Intelligence Agent
    auxiliary_timeout: int = Field(default=300, env="AUXILIARY_TIMEOUT")
    auxiliary_max_retries: int = Field(default=3, env="AUXILIARY_MAX_RETRIES")
    
    # Synthesis & Visualization Agent
    synthesis_timeout: int = Field(default=300, env="SYNTHESIS_TIMEOUT")
    synthesis_max_retries: int = Field(default=3, env="SYNTHESIS_MAX_RETRIES")
    
    class Config:
        env_prefix = "AGENT_"


class WebSettings(BaseSettings):
    """Web application configuration settings."""
    
    # FastAPI settings
    host: str = Field(default="0.0.0.0", env="WEB_HOST")
    port: int = Field(default=8000, env="WEB_PORT")
    debug: bool = Field(default=False, env="WEB_DEBUG")
    reload: bool = Field(default=False, env="WEB_RELOAD")
    
    # Streamlit settings
    streamlit_port: int = Field(default=8501, env="STREAMLIT_PORT")
    streamlit_host: str = Field(default="0.0.0.0", env="STREAMLIT_HOST")
    
    # CORS settings
    cors_origins: List[str] = Field(default=["*"], env="CORS_ORIGINS")
    cors_methods: List[str] = Field(default=["GET", "POST", "PUT", "DELETE"], env="CORS_METHODS")
    cors_headers: List[str] = Field(default=["*"], env="CORS_HEADERS")
    
    class Config:
        env_prefix = "WEB_"


class Settings(BaseSettings):
    """Main application settings."""
    
    # Application metadata
    app_name: str = Field(default="Aura Shipping Intelligence Platform", env="APP_NAME")
    app_version: str = Field(default="1.0.0", env="APP_VERSION")
    environment: str = Field(default="development", env="ENVIRONMENT")
    
    # Component settings
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    qdrant: QdrantSettings = Field(default_factory=QdrantSettings)
    redis: RedisSettings = Field(default_factory=RedisSettings)
    ollama: OllamaSettings = Field(default_factory=OllamaSettings)
    api: APISettings = Field(default_factory=APISettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)
    agent: AgentSettings = Field(default_factory=AgentSettings)
    web: WebSettings = Field(default_factory=WebSettings)
    
    # Feature flags
    enable_caching: bool = Field(default=True, env="ENABLE_CACHING")
    enable_monitoring: bool = Field(default=True, env="ENABLE_MONITORING")
    enable_rate_limiting: bool = Field(default=True, env="ENABLE_RATE_LIMITING")
    
    @validator("environment")
    def validate_environment(cls, v: str) -> str:
        """Validate environment setting."""
        valid_environments = ["development", "staging", "production"]
        if v.lower() not in valid_environments:
            raise ValueError(f"Invalid environment: {v}. Must be one of {valid_environments}")
        return v.lower()
    
    def is_development(self) -> bool:
        """Check if running in development mode."""
        return self.environment == "development"
    
    def is_production(self) -> bool:
        """Check if running in production mode."""
        return self.environment == "production"
    
    def get_agent_config(self, agent_name: str) -> Dict[str, Any]:
        """Get configuration for a specific agent."""
        config_mapping = {
            "supervisor": {
                "timeout_seconds": self.agent.supervisor_timeout,
                "max_retries": self.agent.supervisor_max_retries,
            },
            "structured_data": {
                "timeout_seconds": self.agent.structured_data_timeout,
                "max_retries": self.agent.structured_data_max_retries,
            },
            "vector_search": {
                "timeout_seconds": self.agent.vector_search_timeout,
                "max_retries": self.agent.vector_search_max_retries,
                "embedding_model": self.agent.embedding_model,
            },
            "auxiliary_intelligence": {
                "timeout_seconds": self.agent.auxiliary_timeout,
                "max_retries": self.agent.auxiliary_max_retries,
            },
            "synthesis_visualization": {
                "timeout_seconds": self.agent.synthesis_timeout,
                "max_retries": self.agent.synthesis_max_retries,
            },
        }
        return config_mapping.get(agent_name, {})
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


# Global settings instance
settings = Settings()


def get_settings() -> Settings:
    """Get the global settings instance."""
    return settings


def load_env_file(env_file: str = ".env") -> None:
    """Load environment variables from a file."""
    if os.path.exists(env_file):
        with open(env_file, "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    key, value = line.split("=", 1)
                    os.environ[key] = value


# Load environment variables on module import
load_env_file() 