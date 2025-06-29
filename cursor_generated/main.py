#!/usr/bin/env python3
"""
Main entry point for the Aura Shipping Intelligence Platform.

This script starts the Streamlit application with proper configuration
and environment setup.
"""

import os
import sys
from pathlib import Path

# Add the src directory to Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

# Set environment variables for Streamlit
os.environ["STREAMLIT_SERVER_PORT"] = "8501"
os.environ["STREAMLIT_SERVER_ADDRESS"] = "0.0.0.0"
os.environ["STREAMLIT_SERVER_HEADLESS"] = "true"
os.environ["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"

if __name__ == "__main__":
    import subprocess
    
    # Get the path to the Streamlit app
    streamlit_app_path = src_path / "ui" / "streamlit_app.py"
    
    if not streamlit_app_path.exists():
        print(f"Error: Streamlit app not found at {streamlit_app_path}")
        sys.exit(1)
    
    print("🚢 Starting Aura Shipping Intelligence Platform...")
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
