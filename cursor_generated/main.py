#!/usr/bin/env python3
"""
Main entry point for the Aura Shipping Intelligence Platform.

This script starts the Streamlit application with proper configuration
and environment setup.
"""

import os
import sys
import asyncio
import subprocess
from pathlib import Path

# Add the src directory to Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

# Set environment variables for Streamlit
os.environ["STREAMLIT_SERVER_PORT"] = "8501"
os.environ["STREAMLIT_SERVER_ADDRESS"] = "0.0.0.0"
os.environ["STREAMLIT_SERVER_HEADLESS"] = "true"
os.environ["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"

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
    
    # Get the path to the Streamlit app
    streamlit_app_path = src_path / "ui" / "streamlit_app.py"
    
    if not streamlit_app_path.exists():
        print(f"Error: Streamlit app not found at {streamlit_app_path}")
        sys.exit(1)
    
    print(f"📱 Streamlit app location: {streamlit_app_path}")
    print("🌐 Access the application at: http://localhost:8501")
    print("⏹️  Press Ctrl+C to stop the application")
    print("-" * 60)
    
    try:
        # Run Streamlit
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            str(streamlit_app_path),
            "--server.port", "8501",
            "--server.address", "0.0.0.0",
            "--server.headless", "true",
            "--browser.gatherUsageStats", "false"
        ])
    except KeyboardInterrupt:
        print("\n🛑 Application stopped by user")
    except Exception as e:
        print(f"❌ Error starting application: {e}")
        sys.exit(1)
