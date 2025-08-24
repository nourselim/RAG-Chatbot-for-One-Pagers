#!/usr/bin/env python3
"""
Complete pipeline runner for DeBotte AI Employee Skills Finder

This script runs the complete pipeline:
1. Extract data from PPTX files using docling
2. Build the RAG system with FAISS
3. Start the frontend

Usage:
    python run_pipeline.py
"""

import subprocess
import sys
from pathlib import Path
import os

def run_command(command, description, cwd=None):
    """Run a shell command and print its output."""
    print(f"\n{'='*60}")
    print(f"🚀 {description}")
    print(f"{'='*60}")
    print(f"Running: {command}")
    if cwd:
        print(f"Working directory: {cwd}")
    
    try:
        result = subprocess.run(
            command, 
            shell=True, 
            capture_output=True, 
            text=True, 
            cwd=cwd
        )
        
        if result.returncode == 0:
            print(f"✅ {description} completed successfully")
            if result.stdout:
                print("Output:", result.stdout)
        else:
            print(f"❌ Error running: {command}")
            print(f"Error output: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Exception running command: {e}")
        return False
    
    return True

def check_file_exists(file_path):
    """Check if a file exists and is not empty."""
    return file_path.exists() and file_path.stat().st_size > 0

def main():
    print("🎯 DeBotte AI Employee Skills Finder - Complete Pipeline")
    print("=" * 60)
    
    # Check if we're in the right directory
    if not Path("docling-one-pagers").exists():
        print("❌ Error: docling-one-pagers folder not found.")
        print("Please run this script from the project root directory.")
        sys.exit(1)
    
    # Check for OpenAI API key
    from dotenv import load_dotenv
    load_dotenv()
    
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Error: OPENAI_API_KEY not found in environment.")
        print("Please set your OpenAI API key in a .env file or environment variable.")
        sys.exit(1)
    
    print("✅ OpenAI API key found")
    
    # Step 1: Check if all_employees.json exists
    all_employees_path = Path("docling-one-pagers/json_output/all_employees.json")
    if not check_file_exists(all_employees_path):
        print("📦 Step 1: Extracting data from PPTX files...")
        if not run_command(
            "python employee_rag_extractor.py input_dir --out json_output", 
            "Extracting employee data from PPTX files",
            cwd="docling-one-pagers"
        ):
            print("❌ Failed to extract employee data. Please check the docling-one-pagers folder.")
            sys.exit(1)
    else:
        print("✅ Step 1: Employee data already exists")
    
    # Step 2: Build RAG system
    print("🧠 Step 2: Building RAG system...")
    if not run_command(
        "python main.py auto", 
        "Building RAG system with FAISS",
        cwd="rag"
    ):
        print("❌ Failed to build RAG system. Please check the rag folder.")
        sys.exit(1)
    
    # Step 3: Start frontend
    print("🌐 Step 3: Starting frontend...")
    print("\n🎉 Pipeline completed successfully!")
    print("\nStarting Streamlit frontend...")
    print("The application will open in your browser.")
    print("Press Ctrl+C to stop the frontend.")
    
    # Start the frontend
    try:
        run_command(
            "streamlit run app.py", 
            "Starting Streamlit frontend",
            cwd="frontend"
        )
    except KeyboardInterrupt:
        print("\n👋 Frontend stopped by user")
    except Exception as e:
        print(f"\n❌ Error starting frontend: {e}")

if __name__ == "__main__":
    main()
