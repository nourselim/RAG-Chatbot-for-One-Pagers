# DeBotte AI Employee Skills Finder - Complete Pipeline
# PowerShell Script

Write-Host "🎯 DeBotte AI Employee Skills Finder - Complete Pipeline" -ForegroundColor Green
Write-Host "===========================================================" -ForegroundColor Green

# Check if Python is available
try {
    $pythonVersion = python --version 2>&1
    Write-Host "✅ Python found: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "❌ Error: Python is not installed or not in PATH" -ForegroundColor Red
    Write-Host "Please install Python 3.8+ from https://python.org" -ForegroundColor Yellow
    Read-Host "Press Enter to exit"
    exit 1
}

# Check if .env file exists
if (-not (Test-Path ".env")) {
    Write-Host "⚠️  Warning: .env file not found" -ForegroundColor Yellow
    Write-Host "Please create a .env file with your OPENAI_API_KEY" -ForegroundColor Yellow
    Write-Host "See env_template.txt for reference" -ForegroundColor Yellow
    Write-Host ""
}

# Step 1: Extract data from PPTX files
Write-Host "📦 Step 1: Extracting data from PPTX files..." -ForegroundColor Cyan
Set-Location "docling-one-pagers"
try {
    python employee_rag_extractor.py
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to extract employee data"
    }
} catch {
    Write-Host "❌ Failed to extract employee data: $_" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}
Set-Location ".."

# Step 2: Build RAG system
Write-Host "🧠 Step 2: Building RAG system..." -ForegroundColor Cyan
Set-Location "rag"
try {
    python main.py auto
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to build RAG system"
    }
} catch {
    Write-Host "❌ Failed to build RAG system: $_" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}
Set-Location ".."

# Step 3: Start frontend
Write-Host "🌐 Step 3: Starting frontend..." -ForegroundColor Cyan
Write-Host ""
Write-Host "🎉 Pipeline completed successfully!" -ForegroundColor Green
Write-Host "Starting Streamlit frontend..." -ForegroundColor Green
Write-Host "The application will open in your browser." -ForegroundColor Green
Write-Host "Press Ctrl+C to stop the frontend." -ForegroundColor Yellow
Write-Host ""

Set-Location "frontend"
try {
    streamlit run app.py
} catch {
    Write-Host "❌ Error starting frontend: $_" -ForegroundColor Red
    Read-Host "Press Enter to exit"
}
