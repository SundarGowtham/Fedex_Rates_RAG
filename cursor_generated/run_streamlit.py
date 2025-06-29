#!/usr/bin/env python3
"""
Simple script to run the Streamlit application directly.
"""

import asyncio
import os
import sys
from pathlib import Path

# Add the src directory to Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

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
    import subprocess
    
    # Get the path to the Streamlit app
    streamlit_app_path = src_path / "ui" / "streamlit_app.py"
    
    if not streamlit_app_path.exists():
        print(f"Error: Streamlit app not found at {streamlit_app_path}")
        sys.exit(1)
    
    print("🚢 Starting Aura Shipping Intelligence Platform...")
    
    # Initialize data ingestion
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(initialize_data())
        loop.close()
    except Exception as e:
        print(f"⚠️  Warning: Could not initialize data: {e}")
    
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