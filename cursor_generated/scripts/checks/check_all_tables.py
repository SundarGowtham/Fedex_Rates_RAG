import asyncio
import sys
sys.path.append('.')

from src.core.database import get_database_manager

async def check_all_tables():
    db = await get_database_manager()
    
    # Get list of all tables
    result = await db.execute_query("""
        SELECT table_name 
        FROM information_schema.tables 
        WHERE table_schema = 'public' 
        ORDER BY table_name
    """)
    
    print("All tables in database:")
    for row in result:
        table_name = row['table_name']
        
        # Get row count for each table
        count_result = await db.execute_query(f"SELECT COUNT(*) as count FROM {table_name}")
        count = count_result[0]['count']
        
        print(f"  {table_name}: {count} rows")

if __name__ == "__main__":
    asyncio.run(check_all_tables()) 