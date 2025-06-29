#!/usr/bin/env python3
"""
Simple script to run the Streamlit application directly.
"""

import asyncio
import os
import sys
import subprocess
from pathlib import Path

# Add the src directory to Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

async def clear_caches():
    """Clear all caches for fresh start."""
    print("🧹 Clearing caches for fresh start...")
    
    try:
        # Run the cache clearing script
        result = subprocess.run([sys.executable, "scripts/misc/clear_caches.py"], 
                              capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            print("✅ Caches cleared successfully")
            return True
        else:
            print(f"⚠️  Cache clearing completed with warnings: {result.stderr}")
            return True  # Continue anyway
    except subprocess.TimeoutExpired:
        print("⚠️  Cache clearing timed out, continuing...")
        return True
    except Exception as e:
        print(f"⚠️  Error clearing caches: {e}, continuing...")
        return True  # Continue anyway

async def initialize_data():
    """Initialize data ingestion on startup."""
    try:
        from utils.data_ingestion import initialize_data_ingestion, ingest_data_sources
        
        print("🔄 Initializing data ingestion...")
        await initialize_data_ingestion()
        
        print("📁 Checking for CSV files in data_sources folder...")
        results = await ingest_data_sources()
        
        if results:
            print("✅ Data ingestion completed:")
            for file_path, success in results.items():
                if success:
                    print(f"   ✅ {Path(file_path).name}")
                else:
                    print(f"   ❌ {Path(file_path).name}")
        else:
            print("ℹ️  No CSV files found in data_sources folder")
            
    except Exception as e:
        print(f"⚠️  Warning: Data ingestion initialization failed: {e}")

if __name__ == "__main__":
    print("🚢 Starting Aura Shipping Intelligence Platform...")
    
    # Clear caches first
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(clear_caches())
        loop.close()
    except Exception as e:
        print(f"⚠️  Warning: Could not clear caches: {e}")
    
    # Initialize data ingestion
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(initialize_data())
        loop.close()
    except Exception as e:
        print(f"⚠️  Warning: Could not initialize data: {e}")
    
    # Get the path to the Streamlit app
    streamlit_app_path = src_path / "ui" / "streamlit_app.py"
    
    if not streamlit_app_path.exists():
        print(f"Error: Streamlit app not found at {streamlit_app_path}")
        sys.exit(1)
    
    print("🌐 Access the application at: http://localhost:8501")
    print("⏹️  Press Ctrl+C to stop the application")
    print("-" * 60)
    
    try:
        # Run Streamlit
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            str(streamlit_app_path)
        ])
    except KeyboardInterrupt:
        print("\n🛑 Application stopped by user")
    except Exception as e:
        print(f"❌ Error starting application: {e}")
        sys.exit(1) 