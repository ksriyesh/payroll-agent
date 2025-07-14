#!/usr/bin/env python3
"""
Payroll Agent Streamlit App Launcher
====================================

This script launches the Streamlit app for the Payroll Agent.
Make sure you have set up your .env file with OPENAI_API_KEY before running.
"""

import os
import subprocess
import sys
from pathlib import Path

def check_environment():
    """Check if environment is properly set up."""
    env_file = Path('.env')
    if not env_file.exists():
        print("❌ Error: .env file not found!")
        print("Please create a .env file with your OPENAI_API_KEY")
        print("Example:")
        print("OPENAI_API_KEY=your_api_key_here")
        return False
    
    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv()
    
    if not os.getenv('OPENAI_API_KEY'):
        print("❌ Error: OPENAI_API_KEY not found in .env file!")
        print("Please add your OpenAI API key to the .env file")
        return False
    
    return True

def install_requirements():
    """Install required packages."""
    try:
        import streamlit
        import langchain
        import langgraph
        print("✅ All required packages are installed")
        return True
    except ImportError:
        print("📦 Installing required packages...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-e", "."])
        return True

def main():
    """Main function to launch the app."""
    print("🚀 Payroll Agent Streamlit App Launcher")
    print("=" * 50)
    
    # Check environment
    if not check_environment():
        return
    
    # Install requirements
    if not install_requirements():
        return
    
    print("🎉 Environment is ready!")
    print("🌐 Launching Streamlit app...")
    print("📱 The app will open in your browser automatically")
    print("🛑 Press Ctrl+C to stop the app")
    print("-" * 50)
    
    # Launch Streamlit app
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", "streamlit_app.py"])
    except KeyboardInterrupt:
        print("\n👋 App stopped by user")
    except Exception as e:
        print(f"❌ Error launching app: {e}")

if __name__ == "__main__":
    main() 