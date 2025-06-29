"""
Data ingestion utilities for automatically loading CSV files into PostgreSQL.

This module provides functionality to:
- Monitor a data_sources folder for CSV files
- Automatically ingest CSV data into PostgreSQL
- Check if data already exists to avoid duplicates
- Clear existing data for testing purposes
"""

import asyncio
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import re
import numpy as np

from src.core.database import get_database_manager

logger = logging.getLogger(__name__)

# Configuration
DATA_SOURCES_DIR = Path("data_sources")
SUPPORTED_CSV_TYPES = {
    "fedex_pricing": {
        "filename_pattern": "fedex_pricing*.csv",
        "table_name": "datasource_fedex_pricing",
        "required_columns": ["weight", "transportation_type", "zone", "service_type", "price"],
        "column_mapping": {
            "weight": "weight",
            "transportation_type": "transportation_type", 
            "zone": "zone",
            "service_type": "service_type",
            "price": "price"
        }
    },
    "zone_distance_mapping": {
        "filename_pattern": "zone_distance*.csv",
        "table_name": "datasource_fedex_zone_distance_mapping",
        "required_columns": ["zone", "min_distance", "max_distance"],
        "column_mapping": {
            "zone": "zone",
            "min_distance": "min_distance",
            "max_distance": "max_distance"
        }
    }
}


class DataIngestionManager:
    """Manager for automatic CSV data ingestion into PostgreSQL."""
    
    def __init__(self, data_sources_dir: Path = DATA_SOURCES_DIR):
        self.data_sources_dir = data_sources_dir
        self.data_sources_dir.mkdir(exist_ok=True)
        self.db_manager = None
    
    async def initialize(self):
        """Initialize the database manager."""
        self.db_manager = await get_database_manager()
        await self.db_manager.initialize_async()
        await self.db_manager.initialize_pool()
        # Create tables if they don't exist
        await self.db_manager.create_tables_async()
    
    async def check_data_exists(self, table_name: str) -> bool:
        """Check if data already exists in the specified table."""
        try:
            query = f"SELECT COUNT(*) as count FROM {table_name}"
            result = await self.db_manager.execute_query(query)
            count = result[0]["count"] if result else 0
            return count > 0
        except Exception as e:
            logger.error(f"Error checking data existence in {table_name}: {e}")
            return False
    
    async def clear_table_data(self, table_name: str) -> bool:
        """Clear all data from the specified table."""
        try:
            query = f"DELETE FROM {table_name}"
            await self.db_manager.execute_query(query)
            logger.info(f"Cleared all data from {table_name}")
            return True
        except Exception as e:
            logger.error(f"Error clearing data from {table_name}: {e}")
            return False
    
    async def clear_all_data(self) -> Dict[str, bool]:
        """Clear all data from all supported tables."""
        results = {}
        for csv_type, config in SUPPORTED_CSV_TYPES.items():
            table_name = config["table_name"]
            results[table_name] = await self.clear_table_data(table_name)
        return results
    
    def find_csv_files(self) -> List[Tuple[str, Path]]:
        """Find all CSV files in the data_sources directory."""
        csv_files = []
        
        if not self.data_sources_dir.exists():
            logger.warning(f"Data sources directory {self.data_sources_dir} does not exist")
            return csv_files
        
        for file_path in self.data_sources_dir.glob("*.csv"):
            # Determine CSV type based on filename
            csv_type = self._determine_csv_type(file_path.name)
            if csv_type:
                csv_files.append((csv_type, file_path))
            else:
                logger.warning(f"Unknown CSV type for file: {file_path.name}")
        
        return csv_files
    
    def _determine_csv_type(self, filename: str) -> Optional[str]:
        """Determine the type of CSV based on filename patterns."""
        for csv_type, config in SUPPORTED_CSV_TYPES.items():
            pattern = config["filename_pattern"]
            if self._matches_pattern(filename, pattern):
                return csv_type
        return None
    
    def _matches_pattern(self, filename: str, pattern: str) -> bool:
        """Check if filename matches the pattern."""
        # Simple pattern matching - can be enhanced with regex if needed
        if "*" in pattern:
            prefix = pattern.split("*")[0]
            suffix = pattern.split("*")[1] if len(pattern.split("*")) > 1 else ""
            return filename.startswith(prefix) and filename.endswith(suffix)
        return filename == pattern
    
    def validate_csv_structure(self, file_path: Path, csv_type: str) -> Tuple[bool, List[str]]:
        """Validate that CSV has the required columns."""
        try:
            df = pd.read_csv(file_path)
            required_columns = SUPPORTED_CSV_TYPES[csv_type]["required_columns"]
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                return False, missing_columns
            
            return True, []
        except Exception as e:
            logger.error(f"Error validating CSV structure for {file_path}: {e}")
            return False, [str(e)]
    
    def _preprocess_dataframe(self, df: pd.DataFrame, csv_type: str) -> pd.DataFrame:
        """Preprocess DataFrame columns to match expected SQL types."""
        config = SUPPORTED_CSV_TYPES[csv_type]
        column_mapping = config["column_mapping"]
        
        # Preprocess weight
        if "weight" in column_mapping.values():
            def parse_weight(val):
                try:
                    val = str(val).strip().lower().replace('"', '').replace(",", "")
                    if "oz" in val:
                        num = float(re.findall(r"[\d.]+", val)[0])
                        return round(num / 16, 4)  # 16 oz = 1 lb
                    elif "lb" in val or "lbs" in val:
                        num = float(re.findall(r"[\d.]+", val)[0])
                        return num
                    else:
                        return float(val)
                except Exception as e:
                    logger.warning(f"Could not parse weight '{val}': {e}")
                    return np.nan
            weight_col = [k for k, v in column_mapping.items() if v == "weight"][0]
            df[weight_col] = df[weight_col].apply(parse_weight)
        
        # Preprocess price
        if "price" in column_mapping.values():
            def parse_price(val):
                try:
                    val = str(val).replace("$", "").replace(",", "").strip()
                    return float(val)
                except Exception as e:
                    logger.warning(f"Could not parse price '{val}': {e}")
                    return np.nan
            price_col = [k for k, v in column_mapping.items() if v == "price"][0]
            df[price_col] = df[price_col].apply(parse_price)
        
        # Ensure zone column is string type
        if "zone" in column_mapping.values():
            zone_col = [k for k, v in column_mapping.items() if v == "zone"][0]
            df[zone_col] = df[zone_col].astype(str)
        
        # Drop rows with NaN in any required column
        required_cols = [k for k, v in column_mapping.items()]
        df = df.dropna(subset=required_cols)
        return df

    async def ingest_csv_file(self, csv_type: str, file_path: Path) -> bool:
        """Ingest a single CSV file into the database."""
        try:
            # Validate CSV structure
            is_valid, missing_columns = self.validate_csv_structure(file_path, csv_type)
            if not is_valid:
                logger.error(f"Invalid CSV structure for {file_path}: missing columns {missing_columns}")
                return False
            
            # Check if data already exists
            table_name = SUPPORTED_CSV_TYPES[csv_type]["table_name"]
            if await self.check_data_exists(table_name):
                logger.info(f"Data already exists in {table_name}, skipping ingestion")
                return True
            
            # Read CSV data
            df = pd.read_csv(file_path)
            logger.info(f"Loading {len(df)} rows from {file_path}")
            
            # Preprocess DataFrame
            df = self._preprocess_dataframe(df, csv_type)
            logger.info(f"After preprocessing: {len(df)} rows remain for ingestion")
            
            # Insert data into database
            await self._insert_dataframe(df, csv_type, table_name)
            
            logger.info(f"Successfully ingested {len(df)} rows into {table_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error ingesting CSV file {file_path}: {e}")
            return False
    
    async def _insert_dataframe(self, df: pd.DataFrame, csv_type: str, table_name: str):
        """Insert DataFrame into the specified table."""
        config = SUPPORTED_CSV_TYPES[csv_type]
        column_mapping = config["column_mapping"]
        
        # Prepare data for insertion
        for _, row in df.iterrows():
            # Map DataFrame columns to database columns
            insert_data = {}
            for db_col, csv_col in column_mapping.items():
                if csv_col in row:
                    insert_data[db_col] = row[csv_col]
            
            # Build INSERT query with positional parameters
            columns = list(insert_data.keys())
            placeholders = [f"${i+1}" for i in range(len(columns))]
            query = f"""
                INSERT INTO {table_name} ({', '.join(columns)})
                VALUES ({', '.join(placeholders)})
                ON CONFLICT DO NOTHING
            """
            
            # Execute insertion with positional parameters
            values = list(insert_data.values())
            await self.db_manager.execute_query(query, values)
    
    async def ingest_all_csv_files(self) -> Dict[str, bool]:
        """Ingest all CSV files found in the data_sources directory."""
        if not self.db_manager:
            await self.initialize()
        
        csv_files = self.find_csv_files()
        results = {}
        
        for csv_type, file_path in csv_files:
            logger.info(f"Processing {csv_type} file: {file_path}")
            results[str(file_path)] = await self.ingest_csv_file(csv_type, file_path)
        
        return results
    
    async def get_ingestion_status(self) -> Dict[str, Dict]:
        """Get status of data ingestion for all supported tables."""
        status = {}
        
        for csv_type, config in SUPPORTED_CSV_TYPES.items():
            table_name = config["table_name"]
            has_data = await self.check_data_exists(table_name)
            
            status[table_name] = {
                "csv_type": csv_type,
                "has_data": has_data,
                "required_columns": config["required_columns"],
                "filename_pattern": config["filename_pattern"]
            }
        
        return status


# Global instance
data_ingestion_manager = DataIngestionManager()


async def initialize_data_ingestion():
    """Initialize the data ingestion system."""
    await data_ingestion_manager.initialize()


async def ingest_data_sources():
    """Ingest all CSV files from the data_sources directory."""
    return await data_ingestion_manager.ingest_all_csv_files()


async def clear_all_data():
    """Clear all data from all tables."""
    return await data_ingestion_manager.clear_all_data()


async def get_ingestion_status():
    """Get the status of data ingestion."""
    return await data_ingestion_manager.get_ingestion_status() 