#!/usr/bin/env python3
"""
Test script for DeBotte AI RAG system
This script tests the basic functionality of the RAG components
"""

import sys
from pathlib import Path
import os

def test_imports():
    """Test if all required modules can be imported."""
    print("🧪 Testing module imports...")
    
    try:
        # Add rag folder to path
        rag_path = Path(__file__).parent / "rag"
        sys.path.append(str(rag_path))
        
        # Test imports
        import faiss_service
        print("✅ faiss_service imported successfully")
        
        import embed_only
        print("✅ embed_only imported successfully")
        
        import convert_json_to_chunks
        print("✅ convert_json_to_chunks imported successfully")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def test_data_files():
    """Test if required data files exist."""
    print("\n📁 Testing data files...")
    
    # Check if all_employees.json exists
    all_employees_path = Path("docling-one-pagers/json_output/all_employees.json")
    if all_employees_path.exists():
        print(f"✅ {all_employees_path} exists")
        file_size = all_employees_path.stat().st_size
        print(f"   File size: {file_size:,} bytes")
    else:
        print(f"❌ {all_employees_path} not found")
        return False
    
    # Check if PPTX files exist
    pptx_dir = Path("docling-one-pagers/input_dir")
    if pptx_dir.exists():
        pptx_files = list(pptx_dir.glob("*.pptx"))
        if pptx_files:
            print(f"✅ Found {len(pptx_files)} PPTX files")
            for pptx in pptx_files:
                print(f"   - {pptx.name}")
        else:
            print("❌ No PPTX files found in input_dir")
            return False
    else:
        print("❌ input_dir not found")
        return False
    
    return True

def test_environment():
    """Test environment configuration."""
    print("\n🔧 Testing environment...")
    
    # Check Python version
    python_version = sys.version_info
    print(f"✅ Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    if python_version < (3, 8):
        print("❌ Python 3.8+ required")
        return False
    
    # Check for .env file
    env_file = Path(".env")
    if env_file.exists():
        print("✅ .env file found")
        
        # Check for OpenAI API key
        from dotenv import load_dotenv
        load_dotenv()
        
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            print("✅ OPENAI_API_KEY found")
            # Mask the key for security
            masked_key = api_key[:8] + "..." + api_key[-4:] if len(api_key) > 12 else "***"
            print(f"   API Key: {masked_key}")
        else:
            print("❌ OPENAI_API_KEY not found in .env")
            return False
    else:
        print("❌ .env file not found")
        print("   Please create .env file with your OPENAI_API_KEY")
        return False
    
    return True

def test_dependencies():
    """Test if required packages are installed."""
    print("\n📦 Testing dependencies...")
    
    required_packages = [
        "numpy", "ujson", "tqdm", "python-dotenv", 
        "openai", "faiss-cpu", "streamlit"
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace("-", "_"))
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package}")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠️  Missing packages: {', '.join(missing_packages)}")
        print("   Run: pip install " + " ".join(missing_packages))
        return False
    
    return True

def main():
    """Run all tests."""
    print("🎯 DeBotte AI RAG System - System Test")
    print("=" * 50)
    
    tests = [
        ("Environment", test_environment),
        ("Dependencies", test_dependencies),
        ("Data Files", test_data_files),
        ("Module Imports", test_imports),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} test failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Results Summary")
    print("=" * 50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:20} {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! System is ready to use.")
        print("\nNext steps:")
        print("1. Run: python run_pipeline.py")
        print("2. Or use: run_pipeline.bat (Windows)")
        print("3. Or use: ./run_pipeline.ps1 (PowerShell)")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please fix issues before proceeding.")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
