#!/usr/bin/env python3
"""
Script to clear all caches for Vanna, Qdrant, and LangGraph agents.
"""

import asyncio
import logging
import os
import sys
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from src.core.database import get_database_manager
from src.core.state import WorkflowState

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEFAULT_QDRANT_URL = "http://localhost:6333"
DEFAULT_COLLECTION_NAME = "aura_shipping_vanna"

async def clear_qdrant_cache():
    """Clear Qdrant vector database cache."""
    try:
        logger.info("Clearing Qdrant cache...")
        client = QdrantClient(url=DEFAULT_QDRANT_URL)
        
        # Get all collections
        collections = client.get_collections()
        collection_names = [col.name for col in collections.collections]
        
        # Delete the Vanna collection if it exists
        if DEFAULT_COLLECTION_NAME in collection_names:
            client.delete_collection(collection_name=DEFAULT_COLLECTION_NAME)
            logger.info(f"Deleted Qdrant collection: {DEFAULT_COLLECTION_NAME}")
        else:
            logger.info(f"Qdrant collection {DEFAULT_COLLECTION_NAME} not found")
        
        # Recreate the collection
        client.create_collection(
            collection_name=DEFAULT_COLLECTION_NAME,
            vectors_config=VectorParams(
                size=384,  # Standard embedding size
                distance=Distance.COSINE
            )
        )
        logger.info(f"Recreated Qdrant collection: {DEFAULT_COLLECTION_NAME}")
        
    except Exception as e:
        logger.error(f"Failed to clear Qdrant cache: {e}")

async def clear_database_cache():
    """Clear any cached database connections."""
    try:
        logger.info("Clearing database cache...")
        db_manager = await get_database_manager()
        
        # Close existing connections
        await db_manager.close()
        logger.info("Database connections closed")
        
    except Exception as e:
        logger.error(f"Failed to clear database cache: {e}")

def clear_python_cache():
    """Clear Python cache files."""
    try:
        logger.info("Clearing Python cache files...")
        
        # Find and remove __pycache__ directories
        src_dir = Path(__file__).parent.parent.parent / "src"
        cache_dirs = list(src_dir.rglob("__pycache__"))
        
        for cache_dir in cache_dirs:
            import shutil
            shutil.rmtree(cache_dir)
            logger.info(f"Removed cache directory: {cache_dir}")
        
        # Remove .pyc files
        pyc_files = list(src_dir.rglob("*.pyc"))
        for pyc_file in pyc_files:
            pyc_file.unlink()
            logger.info(f"Removed cache file: {pyc_file}")
            
    except Exception as e:
        logger.error(f"Failed to clear Python cache: {e}")

def clear_redis_cache():
    """Clear Redis cache if available."""
    try:
        logger.info("Clearing Redis cache...")
        import redis
        
        # Try to connect to Redis
        r = redis.Redis(host='localhost', port=6379, db=0)
        r.flushdb()
        logger.info("Redis cache cleared")
        
    except Exception as e:
        logger.warning(f"Could not clear Redis cache (Redis may not be running): {e}")

async def clear_all_caches():
    """Clear all caches."""
    logger.info("Starting cache clearing process...")
    
    # Clear Python cache
    clear_python_cache()
    
    # Clear Redis cache
    clear_redis_cache()
    
    # Clear Qdrant cache
    await clear_qdrant_cache()
    
    # Clear database cache
    await clear_database_cache()
    
    logger.info("All caches cleared successfully!")

if __name__ == "__main__":
    asyncio.run(clear_all_caches()) 