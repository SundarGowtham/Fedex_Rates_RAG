#!/usr/bin/env python3
"""
Script to check and clear fuel data from the database
"""

import asyncio
import sys
sys.path.append('.')

from src.core.database import get_database_manager
from sqlalchemy import text

async def check_and_clear_fuel_data():
    """Check current fuel data and clear the table"""
    db = await get_database_manager()
    
    # Check current data
    result = await db.execute_query('SELECT COUNT(*) as count FROM fuel_prices')
    count = result[0]["count"]
    print(f'Current fuel_prices table has {count} rows')
    
    if count > 0:
        result = await db.execute_query('SELECT * FROM fuel_prices LIMIT 5')
        print('Sample data:')
        for row in result:
            print(row)
        
        # Clear the table
        await db.execute_query('DELETE FROM fuel_prices')
        print(f'Cleared {count} rows from fuel_prices table')
    else:
        print('Table is already empty')

if __name__ == "__main__":
    asyncio.run(check_and_clear_fuel_data()) 