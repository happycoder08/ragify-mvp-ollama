# RAGify Shutdown Script
# Stops all running services

Write-Host "╔════════════════════════════════════════════════════════════════╗"
Write-Host "║           RAGify MVP - Shutdown Script                         ║"
Write-Host "║           Stopping all services...                            ║"
Write-Host "╚════════════════════════════════════════════════════════════════╝"
Write-Host ""

$stoppedServices = @()
$failedServices = @()

# Colors
$successColor = "Green"
$errorColor = "Red"
$infoColor = "Cyan"

# ============================================================================
# 1. Stop FastAPI Server
# ============================================================================
Write-Host "[1/3] Stopping FastAPI server..." -ForegroundColor $infoColor

$uvicorn = Get-Process -Name "python" -ErrorAction SilentlyContinue | Where-Object {
    $_.CommandLine -like "*uvicorn*"
}

if ($uvicorn) {
    try {
        Stop-Process -InputObject $uvicorn -Force -ErrorAction Stop
        Write-Host "✅ FastAPI server stopped" -ForegroundColor $successColor
        $stoppedServices += "FastAPI"
    } catch {
        Write-Host "❌ Failed to stop FastAPI: $_" -ForegroundColor $errorColor
        $failedServices += "FastAPI"
    }
} else {
    Write-Host "⚠️  FastAPI server not running" -ForegroundColor $infoColor
}

# ============================================================================
# 2. Stop Ollama
# ============================================================================
Write-Host "`n[2/3] Stopping Ollama..." -ForegroundColor $infoColor

$ollama = Get-Process -Name "ollama" -ErrorAction SilentlyContinue

if ($ollama) {
    try {
        Stop-Process -InputObject $ollama -Force -ErrorAction Stop
        Write-Host "✅ Ollama stopped" -ForegroundColor $successColor
        $stoppedServices += "Ollama"
    } catch {
        Write-Host "❌ Failed to stop Ollama: $_" -ForegroundColor $errorColor
        $failedServices += "Ollama"
    }
} else {
    Write-Host "⚠️  Ollama not running" -ForegroundColor $infoColor
}

# ============================================================================
# 3. Stop PostgreSQL (Docker)
# ============================================================================
Write-Host "`n[3/3] Stopping PostgreSQL container..." -ForegroundColor $infoColor

try {
    $pgContainer = docker ps --filter "name=ragify-postgres" --format "{{.Names}}"
    
    if ($pgContainer -eq "ragify-postgres") {
        docker stop ragify-postgres | Out-Null
        Write-Host "✅ PostgreSQL stopped" -ForegroundColor $successColor
        $stoppedServices += "PostgreSQL"
    } else {
        Write-Host "⚠️  PostgreSQL container not found" -ForegroundColor $infoColor
    }
} catch {
    Write-Host "⚠️  Could not stop PostgreSQL (Docker may not be running)" -ForegroundColor $infoColor
}

# ============================================================================
# Summary
# ============================================================================
Write-Host "`n╔════════════════════════════════════════════════════════════════╗"
Write-Host "║                   ✅ SHUTDOWN COMPLETE                         ║"
Write-Host "╚════════════════════════════════════════════════════════════════╝"
Write-Host ""

if ($stoppedServices.Count -gt 0) {
    Write-Host "Stopped Services:" -ForegroundColor $successColor
    $stoppedServices | ForEach-Object {
        Write-Host "  ✅ $_" -ForegroundColor $successColor
    }
}

if ($failedServices.Count -gt 0) {
    Write-Host "`nFailed to Stop:" -ForegroundColor $errorColor
    $failedServices | ForEach-Object {
        Write-Host "  ❌ $_" -ForegroundColor $errorColor
    }
}

Write-Host ""
Write-Host "To start services again, run: startup.ps1" -ForegroundColor $infoColor
Write-Host ""
