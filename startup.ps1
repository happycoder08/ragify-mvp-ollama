param([switch]$WithPostgres, [string]$Mode = "demo")

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

Write-Host ""
Write-Host "========== RAGify MVP - Starting Services ==========" -ForegroundColor Cyan

if ($WithPostgres) {
    Write-Host "[1/3] PostgreSQL..." -ForegroundColor Cyan
    $pg = docker ps -a --filter "name=ragify-postgres" --format "{{.Names}}" 2>$null
    if ($pg -eq "ragify-postgres") {
        $running = docker ps --filter "name=ragify-postgres" --format "{{.Names}}" 2>$null
        if ($running -eq "ragify-postgres") {
            Write-Host "OK - Already running" -ForegroundColor Green
        } else {
            Write-Host "Starting..." -ForegroundColor Yellow
            docker start ragify-postgres 2>$null | Out-Null
            Start-Sleep -Seconds 3
            Write-Host "OK - Started (localhost:5432)" -ForegroundColor Green
        }
    } else {
        Write-Host "Creating container..." -ForegroundColor Yellow
        docker run -d --name ragify-postgres -e POSTGRES_USER=ragify -e POSTGRES_PASSWORD=ragify123 -e POSTGRES_DB=ragify_db -p 5432:5432 postgres:15 2>$null | Out-Null
        Start-Sleep -Seconds 5
        Write-Host "OK - Created (localhost:5432)" -ForegroundColor Green
    }
}

Write-Host "[2/3] Ollama..." -ForegroundColor Cyan
$ollama = Get-Process -Name "ollama" -ErrorAction SilentlyContinue
if ($ollama) {
    Write-Host "OK - Already running (PID: $($ollama.Id))" -ForegroundColor Green
} else {
    Write-Host "Starting..." -ForegroundColor Yellow
    Start-Process -FilePath "ollama" -ArgumentList "serve" -NoNewWindow -PassThru | Out-Null
    Start-Sleep -Seconds 5
    Write-Host "OK - Started (http://127.0.0.1:11434)" -ForegroundColor Green
}

Write-Host "[3/3] FastAPI (mode: $Mode)..." -ForegroundColor Cyan
Write-Host "Starting..." -ForegroundColor Yellow
$env:RAGIFY_MODE = $Mode
$serverProc = Start-Process -FilePath "python" -ArgumentList "-m uvicorn main:app --reload --host 0.0.0.0 --port 8000" -WorkingDirectory $scriptDir -NoNewWindow -PassThru
Start-Sleep -Seconds 3
Write-Host "OK - Started (http://localhost:8000)" -ForegroundColor Green

Write-Host ""
Write-Host "========== ALL SERVICES READY ==========" -ForegroundColor Green
Write-Host "Ollama:   http://127.0.0.1:11434" -ForegroundColor Green
Write-Host "FastAPI:  http://localhost:8000" -ForegroundColor Green
if ($WithPostgres) { Write-Host "PostgreSQL: localhost:5432" -ForegroundColor Green }
Write-Host ""
Write-Host "Next: Open http://localhost:8000 in your browser" -ForegroundColor Yellow
Write-Host "Stop: Run shutdown.ps1" -ForegroundColor Yellow
Write-Host ""
