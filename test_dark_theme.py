#!/usr/bin/env python3
"""
Test script to preview the dark theme Streamlit app.
Run this to see the new dark mode styling.
"""

import subprocess
import sys
import os

def main():
    """Launch the Streamlit app with dark theme."""
    print("🌙 Launching Payroll Agent with Dark Theme...")
    print("📊 Features:")
    print("  • Dark background with cyan accents")
    print("  • Gradient buttons and cards")
    print("  • Smooth hover animations")
    print("  • LangSmith tracing indicators")
    print("  • Professional dark UI")
    print("\n🚀 Starting Streamlit app...")
    
    try:
        # Run the streamlit app
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "streamlit_app.py",
            "--theme.base", "dark"
        ], check=True)
    except KeyboardInterrupt:
        print("\n👋 App stopped by user")
    except Exception as e:
        print(f"❌ Error running app: {e}")
        print("💡 Try running: streamlit run streamlit_app.py")

if __name__ == "__main__":
    main() 