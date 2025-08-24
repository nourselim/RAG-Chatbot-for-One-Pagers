#!/usr/bin/env python3
"""
Simple script to run the AI Chat Assistant
"""

import os
import sys
import subprocess

def run_streamlit():
    """Run the Streamlit application"""
    try:
        # Check if streamlit is installed
        import streamlit
    except ImportError:
        print("Streamlit not found. Installing dependencies...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    
    # Run the app
    subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"])

if __name__ == "__main__":
    # Change to the script directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    run_streamlit()
