@echo off
echo 🎯 DeBotte AI Employee Skills Finder - Complete Pipeline
echo ============================================================

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Error: Python is not installed or not in PATH
    pause
    exit /b 1
)

REM Check if .env file exists
if not exist ".env" (
    echo ⚠️  Warning: .env file not found
    echo Please create a .env file with your OPENAI_API_KEY
    echo See env_template.txt for reference
    echo.
)

REM Step 1: Extract data from PPTX files
echo 📦 Step 1: Extracting data from PPTX files...
cd docling-one-pagers
python employee_rag_extractor.py input_dir --out json_output
if errorlevel 1 (
    echo ❌ Failed to extract employee data
    pause
    exit /b 1
)
cd ..

REM Step 2: Build RAG system
echo 🧠 Step 2: Building RAG system...
cd rag
python main.py auto
if errorlevel 1 (
    echo ❌ Failed to build RAG system
    pause
    exit /b 1
)
cd ..

REM Step 3: Start frontend
echo 🌐 Step 3: Starting frontend...
echo.
echo 🎉 Pipeline completed successfully!
echo Starting Streamlit frontend...
echo The application will open in your browser.
echo Press Ctrl+C to stop the frontend.
echo.
cd frontend
streamlit run app.py

pause
