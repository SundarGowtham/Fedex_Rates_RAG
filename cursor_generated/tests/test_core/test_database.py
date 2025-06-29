"""
Tests for the database management functionality.
"""

import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from typing import List, Dict, Any

from core.database import DatabaseManager, get_database_manager, insert_sample_data


class TestDatabaseManager:
    """Test cases for DatabaseManager class."""
    
    @pytest.mark.asyncio
    async def test_database_manager_creation(self):
        """Test creating a new DatabaseManager."""
        with patch('core.database.asyncpg.create_pool') as mock_create_pool:
            mock_pool = AsyncMock()
            mock_create_pool.return_value = mock_pool
            
            db_manager = DatabaseManager("postgresql://test:test@localhost:5432/test")
            
            assert db_manager.database_url == "postgresql://test:test@localhost:5432/test"
            assert db_manager.pool is None
    
    @pytest.mark.asyncio
    async def test_connect_success(self):
        """Test successful database connection."""
        with patch('core.database.asyncpg.create_pool') as mock_create_pool:
            mock_pool = AsyncMock()
            mock_create_pool.return_value = mock_pool
            
            db_manager = DatabaseManager("postgresql://test:test@localhost:5432/test")
            await db_manager.connect()
            
            mock_create_pool.assert_called_once()
            assert db_manager.pool == mock_pool
    
    @pytest.mark.asyncio
    async def test_connect_failure(self):
        """Test database connection failure."""
        with patch('core.database.asyncpg.create_pool') as mock_create_pool:
            mock_create_pool.side_effect = Exception("Connection failed")
            
            db_manager = DatabaseManager("postgresql://test:test@localhost:5432/test")
            
            with pytest.raises(Exception, match="Connection failed"):
                await db_manager.connect()
    
    @pytest.mark.asyncio
    async def test_disconnect(self):
        """Test database disconnection."""
        with patch('core.database.asyncpg.create_pool') as mock_create_pool:
            mock_pool = AsyncMock()
            mock_create_pool.return_value = mock_pool
            
            db_manager = DatabaseManager("postgresql://test:test@localhost:5432/test")
            await db_manager.connect()
            await db_manager.disconnect()
            
            mock_pool.close.assert_called_once()
            await mock_pool.wait_closed.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_execute_query_success(self):
        """Test successful query execution."""
        with patch('core.database.asyncpg.create_pool') as mock_create_pool:
            mock_pool = AsyncMock()
            mock_conn = AsyncMock()
            mock_pool.acquire.return_value.__aenter__.return_value = mock_conn
            
            mock_create_pool.return_value = mock_pool
            
            db_manager = DatabaseManager("postgresql://test:test@localhost:5432/test")
            await db_manager.connect()
            
            expected_result = [{"id": 1, "name": "test"}]
            mock_conn.fetch.return_value = expected_result
            
            result = await db_manager.execute_query(
                "SELECT * FROM test_table WHERE id = $1",
                {"id": 1}
            )
            
            assert result == expected_result
            mock_conn.fetch.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_execute_query_without_connection(self):
        """Test query execution without connection."""
        db_manager = DatabaseManager("postgresql://test:test@localhost:5432/test")
        
        with pytest.raises(RuntimeError, match="Database not connected"):
            await db_manager.execute_query("SELECT * FROM test_table")
    
    @pytest.mark.asyncio
    async def test_execute_query_failure(self):
        """Test query execution failure."""
        with patch('core.database.asyncpg.create_pool') as mock_create_pool:
            mock_pool = AsyncMock()
            mock_conn = AsyncMock()
            mock_pool.acquire.return_value.__aenter__.return_value = mock_conn
            
            mock_create_pool.return_value = mock_pool
            
            db_manager = DatabaseManager("postgresql://test:test@localhost:5432/test")
            await db_manager.connect()
            
            mock_conn.fetch.side_effect = Exception("Query failed")
            
            with pytest.raises(Exception, match="Query failed"):
                await db_manager.execute_query("SELECT * FROM test_table")
    
    @pytest.mark.asyncio
    async def test_execute_transaction_success(self):
        """Test successful transaction execution."""
        with patch('core.database.asyncpg.create_pool') as mock_create_pool:
            mock_pool = AsyncMock()
            mock_conn = AsyncMock()
            mock_pool.acquire.return_value.__aenter__.return_value = mock_conn
            
            mock_create_pool.return_value = mock_pool
            
            db_manager = DatabaseManager("postgresql://test:test@localhost:5432/test")
            await db_manager.connect()
            
            async def transaction_func(conn):
                await conn.execute("INSERT INTO test_table (name) VALUES ($1)", "test")
                return "success"
            
            result = await db_manager.execute_transaction(transaction_func)
            
            assert result == "success"
            mock_conn.execute.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_execute_transaction_failure(self):
        """Test transaction execution failure."""
        with patch('core.database.asyncpg.create_pool') as mock_create_pool:
            mock_pool = AsyncMock()
            mock_conn = AsyncMock()
            mock_pool.acquire.return_value.__aenter__.return_value = mock_conn
            
            mock_create_pool.return_value = mock_pool
            
            db_manager = DatabaseManager("postgresql://test:test@localhost:5432/test")
            await db_manager.connect()
            
            async def transaction_func(conn):
                raise Exception("Transaction failed")
            
            with pytest.raises(Exception, match="Transaction failed"):
                await db_manager.execute_transaction(transaction_func)
    
    @pytest.mark.asyncio
    async def test_context_manager(self):
        """Test DatabaseManager as context manager."""
        with patch('core.database.asyncpg.create_pool') as mock_create_pool:
            mock_pool = AsyncMock()
            mock_create_pool.return_value = mock_pool
            
            async with DatabaseManager("postgresql://test:test@localhost:5432/test") as db_manager:
                assert db_manager.pool == mock_pool
            
            mock_pool.close.assert_called_once()
            await mock_pool.wait_closed.assert_called_once()


class TestDatabaseFunctions:
    """Test cases for database utility functions."""
    
    @pytest.mark.asyncio
    async def test_get_database_manager(self):
        """Test getting database manager instance."""
        with patch('core.database.DatabaseManager') as mock_db_manager_class:
            mock_db_manager = AsyncMock()
            mock_db_manager_class.return_value = mock_db_manager
            
            result = await get_database_manager()
            
            assert result == mock_db_manager
            mock_db_manager_class.assert_called_once()
            mock_db_manager.connect.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_insert_sample_data(self, mock_database_manager):
        """Test inserting sample data."""
        # Mock the database manager
        mock_db = mock_database_manager
        
        await insert_sample_data(mock_db)
        
        # Verify that sample data insertion was attempted
        assert mock_db.execute_query.call_count > 0
        
        # Check that the correct queries were called
        calls = mock_db.execute_query.call_args_list
        query_strings = [call[0][0] for call in calls]
        
        # Should have calls for both pricing and zone data
        pricing_calls = [q for q in query_strings if "datasource_fedex_pricing" in q]
        zone_calls = [q for q in query_strings if "datasource_fedex_zone_distance_mapping" in q]
        
        assert len(pricing_calls) > 0
        assert len(zone_calls) > 0


class TestDatabaseSchema:
    """Test cases for database schema operations."""
    
    @pytest.mark.asyncio
    async def test_create_tables(self):
        """Test creating database tables."""
        with patch('core.database.asyncpg.create_pool') as mock_create_pool:
            mock_pool = AsyncMock()
            mock_conn = AsyncMock()
            mock_pool.acquire.return_value.__aenter__.return_value = mock_conn
            
            mock_create_pool.return_value = mock_pool
            
            db_manager = DatabaseManager("postgresql://test:test@localhost:5432/test")
            await db_manager.connect()
            
            # Mock the table creation queries
            create_tables_query = """
            CREATE TABLE IF NOT EXISTS datasource_fedex_pricing (
                id SERIAL PRIMARY KEY,
                weight DECIMAL(8,2) NOT NULL,
                transportation_type VARCHAR(50) NOT NULL,
                zone INTEGER NOT NULL,
                service_type VARCHAR(50) NOT NULL,
                price DECIMAL(10,2) NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """
            
            await db_manager.execute_query(create_tables_query)
            
            mock_conn.execute.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_insert_pricing_data(self):
        """Test inserting Fedex pricing data."""
        with patch('core.database.asyncpg.create_pool') as mock_create_pool:
            mock_pool = AsyncMock()
            mock_conn = AsyncMock()
            mock_pool.acquire.return_value.__aenter__.return_value = mock_conn
            
            mock_create_pool.return_value = mock_pool
            
            db_manager = DatabaseManager("postgresql://test:test@localhost:5432/test")
            await db_manager.connect()
            
            sample_data = [
                {"weight": 1.0, "transportation_type": "ground", "zone": 1, "service_type": "ground", "price": 8.50},
                {"weight": 2.0, "transportation_type": "ground", "zone": 1, "service_type": "ground", "price": 9.25}
            ]
            
            for data in sample_data:
                await db_manager.execute_query(
                    """
                    INSERT INTO datasource_fedex_pricing (weight, transportation_type, zone, service_type, price)
                    VALUES ($1, $2, $3, $4, $5)
                    ON CONFLICT DO NOTHING
                    """,
                    {
                        "weight": data["weight"],
                        "trans_type": data["transportation_type"],
                        "zone": data["zone"],
                        "service": data["service_type"],
                        "price": data["price"]
                    }
                )
            
            assert mock_conn.execute.call_count == 2
    
    @pytest.mark.asyncio
    async def test_insert_zone_data(self):
        """Test inserting zone distance mapping data."""
        with patch('core.database.asyncpg.create_pool') as mock_create_pool:
            mock_pool = AsyncMock()
            mock_conn = AsyncMock()
            mock_pool.acquire.return_value.__aenter__.return_value = mock_conn
            
            mock_create_pool.return_value = mock_pool
            
            db_manager = DatabaseManager("postgresql://test:test@localhost:5432/test")
            await db_manager.connect()
            
            sample_data = [
                {"zone": 1, "min_distance": 0, "max_distance": 150},
                {"zone": 2, "min_distance": 151, "max_distance": 300}
            ]
            
            for data in sample_data:
                await db_manager.execute_query(
                    """
                    INSERT INTO datasource_fedex_zone_distance_mapping (zone, min_distance, max_distance)
                    VALUES ($1, $2, $3)
                    ON CONFLICT (zone) DO NOTHING
                    """,
                    {
                        "zone": data["zone"],
                        "min_dist": data["min_distance"],
                        "max_dist": data["max_distance"]
                    }
                )
            
            assert mock_conn.execute.call_count == 2


class TestDatabaseQueries:
    """Test cases for specific database queries."""
    
    @pytest.mark.asyncio
    async def test_query_fedex_pricing(self):
        """Test querying Fedex pricing data."""
        with patch('core.database.asyncpg.create_pool') as mock_create_pool:
            mock_pool = AsyncMock()
            mock_conn = AsyncMock()
            mock_pool.acquire.return_value.__aenter__.return_value = mock_conn
            
            mock_create_pool.return_value = mock_pool
            
            db_manager = DatabaseManager("postgresql://test:test@localhost:5432/test")
            await db_manager.connect()
            
            expected_result = [
                {"weight": 5.0, "price": 12.50, "zone": 2, "service_type": "ground"},
                {"weight": 5.0, "price": 15.75, "zone": 3, "service_type": "ground"}
            ]
            mock_conn.fetch.return_value = expected_result
            
            result = await db_manager.execute_query(
                """
                SELECT weight, price, zone, service_type 
                FROM datasource_fedex_pricing 
                WHERE weight = $1 AND service_type = $2
                ORDER BY price
                """,
                {"weight": 5.0, "service_type": "ground"}
            )
            
            assert result == expected_result
    
    @pytest.mark.asyncio
    async def test_query_zone_distance(self):
        """Test querying zone distance mapping data."""
        with patch('core.database.asyncpg.create_pool') as mock_create_pool:
            mock_pool = AsyncMock()
            mock_conn = AsyncMock()
            mock_pool.acquire.return_value.__aenter__.return_value = mock_conn
            
            mock_create_pool.return_value = mock_pool
            
            db_manager = DatabaseManager("postgresql://test:test@localhost:5432/test")
            await db_manager.connect()
            
            expected_result = [
                {"zone": 2, "min_distance": 151, "max_distance": 300}
            ]
            mock_conn.fetch.return_value = expected_result
            
            result = await db_manager.execute_query(
                """
                SELECT zone, min_distance, max_distance 
                FROM datasource_fedex_zone_distance_mapping 
                WHERE zone = $1
                """,
                {"zone": 2}
            )
            
            assert result == expected_result 