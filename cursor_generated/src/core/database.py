"""
Database configuration and schema for the Aura Shipping Intelligence Platform.

This module handles database connections, schema management, and provides
utilities for database operations.
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional, Union

import asyncpg
from sqlalchemy import (
    Column, DateTime, Float, Integer, MetaData, String, Text, create_engine,
    text
)
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.sql import text as sql_text

logger = logging.getLogger(__name__)

# Database configuration
DATABASE_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "database": "aura_db",
    "user": "aura_user",
    "password": "aura_password",
    "min_size": 5,
    "max_size": 20,
}

# Create base class for SQLAlchemy models
Base = declarative_base()

# Metadata for schema management
metadata = MetaData()


class TableDescriptions(Base):
    """Table for storing metadata about database tables and columns."""
    
    __tablename__ = "table_descriptions"
    
    table_name = Column(String(255), primary_key=True)
    column_name = Column(String(255), primary_key=True)
    description = Column(Text)


class FedexPricing(Base):
    """Table for storing Fedex pricing data."""
    
    __tablename__ = "datasource_fedex_pricing"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    weight = Column(Float, nullable=False)
    transportation_type = Column(String(100), nullable=False)
    zone = Column(String(10), nullable=False)
    service_type = Column(String(100), nullable=False)
    price = Column(Float, nullable=False)


class FedexZoneDistanceMapping(Base):
    """Table for storing Fedex zone distance mapping."""
    
    __tablename__ = "datasource_fedex_zone_distance_mapping"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    zone = Column(String(10), primary_key=True)
    min_distance = Column(Float, nullable=False)
    max_distance = Column(Float, nullable=False)


class ShippingRates(Base):
    """Table for storing general shipping rates."""
    
    __tablename__ = "shipping_rates"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    origin_zip = Column(String(10), nullable=False)
    destination_zip = Column(String(10), nullable=False)
    weight = Column(Float, nullable=False)
    service_type = Column(String(100), nullable=False)
    carrier = Column(String(50), nullable=False)
    price = Column(Float, nullable=False)
    transit_days = Column(Integer)
    created_at = Column(DateTime, server_default=text('CURRENT_TIMESTAMP'))
    updated_at = Column(DateTime, server_default=text('CURRENT_TIMESTAMP'))


class ShippingZones(Base):
    """Table for storing shipping zones information."""
    
    __tablename__ = "shipping_zones"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    zone_code = Column(String(10), unique=True, nullable=False)
    zone_name = Column(String(255), nullable=False)
    description = Column(Text)
    created_at = Column(DateTime, server_default=text('CURRENT_TIMESTAMP'))


class WeatherData(Base):
    """Table for storing weather data."""
    
    __tablename__ = "weather_data"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    location = Column(String(255), nullable=False)
    temperature = Column(Float)
    humidity = Column(Float)
    wind_speed = Column(Float)
    precipitation = Column(Float)
    weather_condition = Column(String(100))
    timestamp = Column(DateTime, server_default=text('CURRENT_TIMESTAMP'))


class FuelPrices(Base):
    """Table for storing fuel price data."""
    
    __tablename__ = "fuel_prices"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    fuel_type = Column(String(50), nullable=False)
    price_per_gallon = Column(Float, nullable=False)
    region = Column(String(100))
    timestamp = Column(DateTime, server_default=text('CURRENT_TIMESTAMP'))


class TrafficData(Base):
    """Table for storing traffic data."""
    
    __tablename__ = "traffic_data"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    route = Column(String(255), nullable=False)
    congestion_level = Column(String(50))
    average_speed = Column(Float)
    delay_minutes = Column(Integer)
    timestamp = Column(DateTime, server_default=text('CURRENT_TIMESTAMP'))


class DatabaseManager:
    """Manager for database operations and connections."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or DATABASE_CONFIG
        self.engine = None
        self.async_engine = None
        self.session_factory = None
        self.async_session_factory = None
        self.pool = None
    
    def get_sync_connection_string(self) -> str:
        """Get synchronous database connection string."""
        return (
            f"postgresql://{self.config['user']}:{self.config['password']}"
            f"@{self.config['host']}:{self.config['port']}/{self.config['database']}"
        )
    
    def get_async_connection_string(self) -> str:
        """Get asynchronous database connection string."""
        return (
            f"postgresql+asyncpg://{self.config['user']}:{self.config['password']}"
            f"@{self.config['host']}:{self.config['port']}/{self.config['database']}"
        )
    
    def initialize_sync(self) -> None:
        """Initialize synchronous database connection."""
        connection_string = self.get_sync_connection_string()
        self.engine = create_engine(
            connection_string,
            pool_size=10,
            max_overflow=20,
            pool_pre_ping=True,
            echo=False
        )
        self.session_factory = sessionmaker(
            bind=self.engine,
            autocommit=False,
            autoflush=False
        )
        logger.info("Synchronous database connection initialized")
    
    async def initialize_async(self) -> None:
        """Initialize asynchronous database connection."""
        connection_string = self.get_async_connection_string()
        self.async_engine = create_async_engine(
            connection_string,
            pool_size=10,
            max_overflow=20,
            pool_pre_ping=True,
            echo=False
        )
        self.async_session_factory = async_sessionmaker(
            bind=self.async_engine,
            autocommit=False,
            autoflush=False
        )
        logger.info("Asynchronous database connection initialized")
    
    async def initialize_pool(self) -> None:
        """Initialize connection pool for raw SQL operations."""
        self.pool = await asyncpg.create_pool(
            host=self.config["host"],
            port=self.config["port"],
            database=self.config["database"],
            user=self.config["user"],
            password=self.config["password"],
            min_size=self.config["min_size"],
            max_size=self.config["max_size"],
        )
        logger.info("Database connection pool initialized")
    
    def create_tables(self) -> None:
        """Create all database tables."""
        if not self.engine:
            self.initialize_sync()
        
        Base.metadata.create_all(bind=self.engine)
        logger.info("Database tables created successfully")
    
    async def create_tables_async(self) -> None:
        """Create all database tables asynchronously."""
        if not self.async_engine:
            await self.initialize_async()
        
        async with self.async_engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        logger.info("Database tables created successfully (async)")
    
    def get_session(self):
        """Get a synchronous database session."""
        if not self.session_factory:
            self.initialize_sync()
        return self.session_factory()
    
    @asynccontextmanager
    async def get_async_session(self) -> AsyncSession:
        """Get an asynchronous database session."""
        if not self.async_session_factory:
            await self.initialize_async()
        
        async with self.async_session_factory() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise
            finally:
                await session.close()
    
    async def execute_query(self, query: str, params: Optional[Union[Dict[str, Any], List[Any]]] = None) -> List[Dict[str, Any]]:
        """Execute a raw SQL query and return results."""
        if not self.pool:
            await self.initialize_pool()
        
        async with self.pool.acquire() as conn:
            if params:
                if isinstance(params, dict):
                    # Named parameters
                    result = await conn.fetch(query, **params)
                else:
                    # Positional parameters
                    result = await conn.fetch(query, *params)
            else:
                result = await conn.fetch(query)
            
            return [dict(row) for row in result]
    
    async def execute_transaction(self, queries: List[str], params: Optional[List[Dict[str, Any]]] = None) -> None:
        """Execute multiple queries in a transaction."""
        if not self.pool:
            await self.initialize_pool()
        
        async with self.pool.acquire() as conn:
            async with conn.transaction():
                for i, query in enumerate(queries):
                    if params and i < len(params):
                        await conn.execute(query, **params[i])
                    else:
                        await conn.execute(query)
    
    async def health_check(self) -> bool:
        """Check database health."""
        try:
            if not self.pool:
                await self.initialize_pool()
            
            async with self.pool.acquire() as conn:
                await conn.fetchval("SELECT 1")
            return True
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return False
    
    async def close(self) -> None:
        """Close all database connections."""
        if self.pool:
            await self.pool.close()
        if self.async_engine:
            await self.async_engine.dispose()
        if self.engine:
            self.engine.dispose()
        logger.info("Database connections closed")


# Global database manager instance
db_manager = DatabaseManager()


async def get_database_manager() -> DatabaseManager:
    """Get the global database manager instance."""
    return db_manager


async def initialize_database() -> None:
    """Initialize the database with all components."""
    manager = await get_database_manager()
    await manager.initialize_async()
    await manager.initialize_pool()
    await manager.create_tables_async()
    
    # Insert initial table descriptions
    await insert_initial_table_descriptions(manager)


async def insert_initial_table_descriptions(manager: DatabaseManager) -> None:
    """Insert initial table descriptions for metadata."""
    descriptions = [
        ("datasource_fedex_pricing", "weight", "Package weight in pounds"),
        ("datasource_fedex_pricing", "transportation_type", "Type of transportation (ground, air, etc.)"),
        ("datasource_fedex_pricing", "zone", "Shipping zone code"),
        ("datasource_fedex_pricing", "service_type", "Fedex service type"),
        ("datasource_fedex_pricing", "price", "Shipping price in USD"),
        ("datasource_fedex_zone_distance_mapping", "zone", "Zone code"),
        ("datasource_fedex_zone_distance_mapping", "min_distance", "Minimum distance in miles for this zone"),
        ("datasource_fedex_zone_distance_mapping", "max_distance", "Maximum distance in miles for this zone"),
        ("shipping_rates", "origin_zip", "Origin ZIP code"),
        ("shipping_rates", "destination_zip", "Destination ZIP code"),
        ("shipping_rates", "weight", "Package weight in pounds"),
        ("shipping_rates", "service_type", "Shipping service type"),
        ("shipping_rates", "carrier", "Shipping carrier name"),
        ("shipping_rates", "price", "Shipping price in USD"),
        ("shipping_rates", "transit_days", "Estimated transit time in days"),
        ("weather_data", "location", "Geographic location"),
        ("weather_data", "temperature", "Temperature in Fahrenheit"),
        ("weather_data", "humidity", "Humidity percentage"),
        ("weather_data", "wind_speed", "Wind speed in mph"),
        ("weather_data", "precipitation", "Precipitation amount"),
        ("fuel_prices", "fuel_type", "Type of fuel"),
        ("fuel_prices", "price_per_gallon", "Price per gallon in USD"),
        ("fuel_prices", "region", "Geographic region"),
        ("traffic_data", "route", "Route description"),
        ("traffic_data", "congestion_level", "Traffic congestion level"),
        ("traffic_data", "average_speed", "Average speed in mph"),
        ("traffic_data", "delay_minutes", "Delay in minutes"),
    ]
    
    for table_name, column_name, description in descriptions:
        query = """
        INSERT INTO table_descriptions (table_name, column_name, description)
        VALUES ($1, $2, $3)
        ON CONFLICT (table_name, column_name) DO NOTHING
        """
        await manager.execute_query(query, {
            "table_name": table_name,
            "column_name": column_name,
            "description": description
        })
    
    logger.info("Initial table descriptions inserted")


async def insert_sample_data(manager: DatabaseManager) -> None:
    """Insert sample data for testing and development."""
    
    # Sample Fedex pricing data
    fedex_pricing_data = [
        (1.0, "ground", "1", "Ground", 8.50),
        (2.0, "ground", "1", "Ground", 9.25),
        (5.0, "ground", "1", "Ground", 12.75),
        (10.0, "ground", "1", "Ground", 18.50),
        (1.0, "ground", "2", "Ground", 9.25),
        (2.0, "ground", "2", "Ground", 10.50),
        (5.0, "ground", "2", "Ground", 15.25),
        (10.0, "ground", "2", "Ground", 22.75),
        (1.0, "air", "1", "Express", 25.50),
        (2.0, "air", "1", "Express", 28.75),
        (5.0, "air", "1", "Express", 35.25),
        (10.0, "air", "1", "Express", 45.50),
    ]
    
    for weight, trans_type, zone, service, price in fedex_pricing_data:
        query = """
        INSERT INTO datasource_fedex_pricing (weight, transportation_type, zone, service_type, price)
        VALUES ($1, $2, $3, $4, $5)
        ON CONFLICT DO NOTHING
        """
        await manager.execute_query(query, {
            "weight": weight,
            "trans_type": trans_type,
            "zone": zone,
            "service": service,
            "price": price
        })
    
    # Sample zone distance mapping
    zone_distance_data = [
        ("1", 0, 150),
        ("2", 151, 300),
        ("3", 301, 600),
        ("4", 601, 1000),
        ("5", 1001, 1400),
        ("6", 1401, 1800),
        ("7", 1801, 2200),
        ("8", 2201, 2600),
    ]
    
    for zone, min_dist, max_dist in zone_distance_data:
        query = """
        INSERT INTO datasource_fedex_zone_distance_mapping (zone, min_distance, max_distance)
        VALUES ($1, $2, $3)
        ON CONFLICT (zone) DO NOTHING
        """
        await manager.execute_query(query, {
            "zone": zone,
            "min_dist": min_dist,
            "max_dist": max_dist
        })
    
    logger.info("Sample data inserted successfully") 