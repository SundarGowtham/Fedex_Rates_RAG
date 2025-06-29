import asyncio
import sys
sys.path.append('.')

from src.core.database import get_database_manager

async def check_data():
    db = await get_database_manager()
    
    # Check count
    result = await db.execute_query('SELECT COUNT(*) as count FROM datasource_fedex_pricing')
    print(f'Total rows: {result[0]["count"]}')
    
    # Check sample data
    result = await db.execute_query('SELECT * FROM datasource_fedex_pricing LIMIT 5')
    print('Sample data:')
    for row in result:
        print(row)

if __name__ == "__main__":
    asyncio.run(check_data()) 