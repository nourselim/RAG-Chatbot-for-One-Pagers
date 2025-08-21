@echo off
echo 🛠️  Installing Dependencies for DeBotte AI
echo ===========================================

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Error: Python is not installed or not in PATH
    echo Please install Python 3.8+ from https://python.org
    pause
    exit /b 1
)

echo ✅ Python found
echo.

REM Install frontend dependencies
echo 📦 Installing frontend dependencies...
pip install -r frontend/requirements.txt
if errorlevel 1 (
    echo ❌ Failed to install frontend dependencies
    pause
    exit /b 1
)

REM Install additional RAG dependencies
echo 🧠 Installing RAG dependencies...
pip install numpy ujson tqdm python-dotenv openai faiss-cpu
if errorlevel 1 (
    echo ❌ Failed to install RAG dependencies
    pause
    exit /b 1
)

echo.
echo ✅ All dependencies installed successfully!
echo.
echo Next steps:
echo 1. Create a .env file with your OPENAI_API_KEY
echo 2. Run run_pipeline.bat to start the system
echo.
pause
