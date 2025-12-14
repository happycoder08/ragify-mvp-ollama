# RAGify Startup Script
# Starts all required services for the RAGify application

param(
    [switch]$WithPostgres = $false,
    [string]$Mode = "demo"
)

Write-Host "╔════════════════════════════════════════════════════════════════╗"
Write-Host "║           RAGify MVP - Startup Script                          ║"
Write-Host "║           Starting all required services...                    ║"
Write-Host "╚════════════════════════════════════════════════════════════════╝"
Write-Host ""

# Get script directory
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

# Colors for output
$successColor = "Green"
$errorColor = "Red"
$infoColor = "Cyan"
$warningColor = "Yellow"

# Check if running as administrator
$isAdmin = [bool]([System.Security.Principal.WindowsIdentity]::GetCurrent().Groups -match "S-1-5-32-544")
if (-not $isAdmin) {
    Write-Host "⚠️  Warning: Not running as administrator. Some services may fail to start." -ForegroundColor $warningColor
}

# ============================================================================
# 1. Check Docker
# ============================================================================
Write-Host "`n[1/4] Checking Docker status..." -ForegroundColor $infoColor

try {
    $dockerStatus = docker ps 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ Docker is running" -ForegroundColor $successColor
    } else {
        Write-Host "❌ Docker is not responding" -ForegroundColor $warningColor
    }
} catch {
    Write-Host "❌ Docker is not installed or not available" -ForegroundColor $warningColor
}

# ============================================================================
# 2. Start PostgreSQL (Optional)
# ============================================================================
if ($WithPostgres) {
    Write-Host "`n[2/4] Starting PostgreSQL database..." -ForegroundColor $infoColor
    
    $pgContainer = docker ps -a --filter "name=ragify-postgres" --format "{{.Names}}"
    
    if ($pgContainer -eq "ragify-postgres") {
        $running = docker ps --filter "name=ragify-postgres" --format "{{.Names}}"
        if ($running -eq "ragify-postgres") {
            Write-Host "✅ PostgreSQL container already running" -ForegroundColor $successColor
        } else {
            Write-Host "🔄 Starting PostgreSQL container..." -ForegroundColor $infoColor
            docker start ragify-postgres | Out-Null
            Start-Sleep -Seconds 3
            Write-Host "✅ PostgreSQL started" -ForegroundColor $successColor
        }
    } else {
        Write-Host "🔄 Creating PostgreSQL container..." -ForegroundColor $infoColor
        docker run -d `
            --name ragify-postgres `
            -e POSTGRES_USER=ragify `
            -e POSTGRES_PASSWORD=ragify123 `
            -e POSTGRES_DB=ragify_db `
            -p 5432:5432 `
            postgres:15 | Out-Null
        Start-Sleep -Seconds 5
        Write-Host "✅ PostgreSQL started (user: ragify, db: ragify_db)" -ForegroundColor $successColor
    }
} else {
    Write-Host "`n[2/4] Skipping PostgreSQL (optional)" -ForegroundColor $warningColor
    Write-Host "    Use '-WithPostgres' flag to start PostgreSQL" -ForegroundColor $warningColor
}

# ============================================================================
# 3. Start Ollama
# ============================================================================
Write-Host "`n[3/4] Starting Ollama server..." -ForegroundColor $infoColor

try {
    $ollama = Get-Process -Name "ollama" -ErrorAction SilentlyContinue
    if ($ollama) {
        Write-Host "✅ Ollama is already running (PID: $($ollama.Id))" -ForegroundColor $successColor
    } else {
        Write-Host "🔄 Launching Ollama..." -ForegroundColor $infoColor
        Start-Process -FilePath "ollama" -ArgumentList "serve" -NoNewWindow -PassThru
        Start-Sleep -Seconds 5
        Write-Host "✅ Ollama started on http://127.0.0.1:11434" -ForegroundColor $successColor
    }
} catch {
    Write-Host "❌ Failed to start Ollama: $_" -ForegroundColor $errorColor
    Write-Host "   Install Ollama from https://ollama.ai" -ForegroundColor $warningColor
    exit 1
}

# ============================================================================
# 4. Start FastAPI Server
# ============================================================================
Write-Host "`n[4/4] Starting FastAPI server..." -ForegroundColor $infoColor

# Stop any existing uvicorn process
Stop-Process -Name "uvicorn" -Force -ErrorAction SilentlyContinue
Start-Sleep -Seconds 1

Write-Host "🔄 Launching uvicorn server in $Mode mode..." -ForegroundColor $infoColor

$env:RAGIFY_MODE = $Mode

# Start in a new window so it doesn't block
$serverProc = Start-Process -FilePath "python" `
    -ArgumentList "-m uvicorn main:app --reload --host 0.0.0.0 --port 8000" `
    -WorkingDirectory $scriptDir `
    -NoNewWindow `
    -PassThru

Start-Sleep -Seconds 3

Write-Host "✅ FastAPI server started (PID: $($serverProc.Id))" -ForegroundColor $successColor
Write-Host "   Running mode: $Mode" -ForegroundColor $successColor
Write-Host "   Access at: http://localhost:8000" -ForegroundColor $successColor

# ============================================================================
# Summary
# ============================================================================
Write-Host "`n╔════════════════════════════════════════════════════════════════╗"
Write-Host "║                   ✅ STARTUP COMPLETE                          ║"
Write-Host "╚════════════════════════════════════════════════════════════════╝"
Write-Host ""
Write-Host "Services Status:" -ForegroundColor $infoColor
Write-Host "  🔹 Ollama (Embeddings/LLM):  http://127.0.0.1:11434" -ForegroundColor $successColor
Write-Host "  🔹 FastAPI (Web Server):     http://localhost:8000" -ForegroundColor $successColor
if ($WithPostgres) {
    Write-Host "  🔹 PostgreSQL (Database):     localhost:5432" -ForegroundColor $successColor
}
Write-Host ""
Write-Host "Quick Commands:" -ForegroundColor $infoColor
Write-Host "  📤 Upload documents:  Open http://localhost:8000 in your browser" -ForegroundColor $infoColor
Write-Host "  🔍 Query documents:   Ask questions in the web UI" -ForegroundColor $infoColor
Write-Host "  📊 Check logs:        See console output above" -ForegroundColor $infoColor
Write-Host ""
Write-Host "To stop all services, run: shutdown.ps1" -ForegroundColor $warningColor
Write-Host ""
