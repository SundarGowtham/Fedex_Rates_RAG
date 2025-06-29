#!/usr/bin/env python3
"""
Aura Shipping Intelligence Platform - Quick Start Script

This script helps you get started with the Aura platform by:
1. Checking prerequisites
2. Setting up the environment
3. Starting the infrastructure
4. Clearing caches for fresh start
5. Launching the Streamlit application
"""

import os
import sys
import subprocess
import time
import asyncio
from pathlib import Path

def print_banner():
    """Print the Aura banner."""
    print("=" * 60)
    print("🚢 Aura Shipping Intelligence Platform")
    print("=" * 60)
    print()

def clear_caches():
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

def check_prerequisites():
    """Check if all prerequisites are installed."""
    print("🔍 Checking prerequisites...")
    
    # Check Python version
    if sys.version_info < (3, 11):
        print("❌ Python 3.11+ is required")
        return False
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
    
    # Check Ollama
    try:
        result = subprocess.run(["ollama", "--version"], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Ollama is installed")
        else:
            print("❌ Ollama is not installed")
            return False
    except FileNotFoundError:
        print("❌ Ollama is not installed")
        return False
    
    print()
    return True

def setup_environment():
    """Set up the environment file."""
    print("⚙️  Setting up environment...")
    
    env_file = Path(".env")
    env_template = Path("config/env_template.txt")
    
    if env_file.exists():
        print("✅ Environment file already exists")
        return True
    
    if not env_template.exists():
        print("❌ Environment template not found")
        return False
    
    # Copy template to .env
    try:
        with open(env_template, 'r') as f:
            template_content = f.read()
        
        with open(env_file, 'w') as f:
            f.write(template_content)
        
        print("✅ Environment file created from template")
        print("📝 Please edit .env file with your configuration")
        return True
    except Exception as e:
        print(f"❌ Error creating environment file: {e}")
        return False

def start_infrastructure():
    """Check if local infrastructure services are running."""
    print("🏗️  Checking local infrastructure services...")
    
    services_status = {}
    
    # Check PostgreSQL
    try:
        result = subprocess.run(["pg_isready", "-h", "localhost", "-p", "5432"], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print("✅ PostgreSQL is running on localhost:5432")
            services_status["postgresql"] = True
        else:
            print("❌ PostgreSQL is not running on localhost:5432")
            services_status["postgresql"] = False
    except (subprocess.TimeoutExpired, FileNotFoundError, Exception) as e:
        print(f"❌ Error checking PostgreSQL: {e}")
        services_status["postgresql"] = False
    
    # Check Redis
    try:
        # Use netcat or telnet to check if port is open
        result = subprocess.run(["nc", "-z", "localhost", "6379"], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print("✅ Redis is running on localhost:6379")
            services_status["redis"] = True
        else:
            print("❌ Redis is not running on localhost:6379")
            services_status["redis"] = False
    except (subprocess.TimeoutExpired, FileNotFoundError, Exception) as e:
        print(f"❌ Error checking Redis: {e}")
        services_status["redis"] = False
    
    # Check Qdrant
    try:
        result = subprocess.run(["curl", "-s", "http://localhost:6333/collections"], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print("✅ Qdrant is running on localhost:6333")
            services_status["qdrant"] = True
        else:
            print("❌ Qdrant is not running on localhost:6333")
            services_status["qdrant"] = False
    except (subprocess.TimeoutExpired, FileNotFoundError, Exception) as e:
        print(f"❌ Error checking Qdrant: {e}")
        services_status["qdrant"] = False
    
    # Check Ollama
    try:
        result = subprocess.run(["curl", "-s", "http://localhost:11434/api/tags"], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print("✅ Ollama is running on localhost:11434")
            services_status["ollama"] = True
        else:
            print("❌ Ollama is not running on localhost:11434")
            services_status["ollama"] = False
    except (subprocess.TimeoutExpired, FileNotFoundError, Exception) as e:
        print(f"❌ Error checking Ollama: {e}")
        services_status["ollama"] = False
    
    # Summary
    all_running = all(services_status.values())
    if all_running:
        print("✅ All infrastructure services are running")
        return True
    else:
        print("❌ Some infrastructure services are not running:")
        for service, status in services_status.items():
            if not status:
                print(f"   - {service}")
        print("\n📖 Please start the missing services before continuing.")
        print("   See README.md for instructions on starting local services.")
        return False

def install_dependencies():
    """Install Python dependencies."""
    print("📦 Installing dependencies...")
    
    # Check if uv is available
    try:
        result = subprocess.run(["uv", "--version"], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Using uv for dependency management")
            result = subprocess.run(["uv", "sync"], capture_output=True, text=True)
            if result.returncode == 0:
                print("✅ Dependencies installed with uv")
                return True
            else:
                print(f"❌ Error installing dependencies with uv: {result.stderr}")
                return False
    except FileNotFoundError:
        print("⚠️  uv not found, trying pip...")
    
    # Fallback to pip
    try:
        result = subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Dependencies installed with pip")
            return True
        else:
            print(f"❌ Error installing dependencies with pip: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Error installing dependencies: {e}")
        return False

def launch_application():
    """Launch the Streamlit application."""
    print("🚀 Launching Aura application...")
    print("🌐 The application will be available at: http://localhost:8501")
    print("⏹️  Press Ctrl+C to stop the application")
    print("-" * 60)
    
    try:
        # Run the Streamlit app
        subprocess.run([sys.executable, "run_streamlit.py"])
    except KeyboardInterrupt:
        print("\n🛑 Application stopped by user")
    except Exception as e:
        print(f"❌ Error launching application: {e}")

def main():
    """Main function."""
    print_banner()
    
    # Step 1: Check prerequisites
    if not check_prerequisites():
        print("\n❌ Prerequisites check failed. Please install the required software.")
        print("📖 See the README.md file for installation instructions.")
        return
    
    # Step 2: Setup environment
    if not setup_environment():
        print("\n❌ Environment setup failed.")
        return
    
    # Step 3: Start infrastructure
    if not start_infrastructure():
        print("\n❌ Infrastructure startup failed.")
        return
    
    # Step 4: Install dependencies
    if not install_dependencies():
        print("\n❌ Dependency installation failed.")
        return
    
    # Step 5: Clear caches
    if not clear_caches():
        print("\n❌ Cache clearing failed.")
        return
    
    # Step 6: Launch application
    print("\n🎉 Setup complete! Launching Aura...")
    launch_application()

if __name__ == "__main__":
    main() 