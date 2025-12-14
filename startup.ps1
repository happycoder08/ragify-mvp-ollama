param([string]$Mode = "demo")

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

Write-Host ""
Write-Host "========== RAGify MVP - Starting Services ==========" -ForegroundColor Cyan

Write-Host "[1/4] Cleaning up old data..." -ForegroundColor Cyan
# Remove uploaded files
Remove-Item -Path ".\uploads\*" -Force -ErrorAction SilentlyContinue
# Remove vectorstore directory completely (so ChromaDB creates fresh)
Remove-Item -Path ".\vectorstore" -Recurse -Force -ErrorAction SilentlyContinue
Write-Host "OK - Uploads and vectorstore cleared" -ForegroundColor Green

Write-Host "[2/4] PostgreSQL..." -ForegroundColor Cyan
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

# Clear PostgreSQL data if container is running
$pgRunning = docker ps --filter "name=ragify-postgres" --format "{{.Names}}" 2>$null
if ($pgRunning -eq "ragify-postgres") {
    Write-Host "Clearing PostgreSQL data..." -ForegroundColor Yellow
    docker exec ragify-postgres psql -U ragify -d ragify_db -c "TRUNCATE TABLE documents, conversations, messages RESTART IDENTITY CASCADE;" 2>$null | Out-Null
    Write-Host "OK - Database tables cleared" -ForegroundColor Green
}

Write-Host "[3/4] Ollama..." -ForegroundColor Cyan
$ollama = Get-Process -Name "ollama" -ErrorAction SilentlyContinue
if ($ollama) {
    Write-Host "Process found (PID: $($ollama[0].Id)) - Verifying..." -ForegroundColor Yellow
    try {
        $response = Invoke-WebRequest -Uri "http://localhost:11434/api/tags" -Method GET -TimeoutSec 3 -UseBasicParsing -ErrorAction Stop
        Write-Host "OK - Already running and responsive" -ForegroundColor Green
    } catch {
        Write-Host "WARNING - Process exists but not responsive, restarting..." -ForegroundColor Yellow
        Stop-Process -Name "ollama" -Force -ErrorAction SilentlyContinue
        Start-Sleep -Seconds 2
        Start-Process -FilePath "ollama" -ArgumentList "serve" -WindowStyle Hidden
        Start-Sleep -Seconds 5
        Write-Host "OK - Restarted (http://127.0.0.1:11434)" -ForegroundColor Green
    }
} else {
    Write-Host "Starting..." -ForegroundColor Yellow
    Start-Process -FilePath "ollama" -ArgumentList "serve" -WindowStyle Hidden
    Start-Sleep -Seconds 5
    
    # Verify Ollama is responding
    $retries = 0
    $maxRetries = 3
    $started = $false
    while ($retries -lt $maxRetries -and -not $started) {
        try {
            $response = Invoke-WebRequest -Uri "http://localhost:11434/api/tags" -Method GET -TimeoutSec 3 -UseBasicParsing -ErrorAction Stop
            $started = $true
            Write-Host "OK - Started and verified (http://127.0.0.1:11434)" -ForegroundColor Green
        } catch {
            $retries++
            if ($retries -lt $maxRetries) {
                Write-Host "Waiting for Ollama to respond (attempt $retries/$maxRetries)..." -ForegroundColor Yellow
                Start-Sleep -Seconds 3
            }
        }
    }
    
    if (-not $started) {
        Write-Host "ERROR - Ollama started but not responding. Check if models are installed." -ForegroundColor Red
        Write-Host "       Run: ollama pull llama3.2:1b" -ForegroundColor Yellow
        Write-Host "       Run: ollama pull nomic-embed-text" -ForegroundColor Yellow
    }
}

Write-Host "[4/4] FastAPI (mode: $Mode)..." -ForegroundColor Cyan
Write-Host "Starting..." -ForegroundColor Yellow
$env:RAGIFY_MODE = $Mode
$serverProc = Start-Process -FilePath "python" -ArgumentList "-m uvicorn main:app --reload --host 0.0.0.0 --port 8000" -WorkingDirectory $scriptDir -NoNewWindow -PassThru
Start-Sleep -Seconds 3
Write-Host "OK - Started (http://localhost:8000)" -ForegroundColor Green

Write-Host ""
Write-Host "========== ALL SERVICES READY ==========" -ForegroundColor Green
Write-Host "PostgreSQL: localhost:5432" -ForegroundColor Green
Write-Host "Ollama:   http://127.0.0.1:11434" -ForegroundColor Green
Write-Host "FastAPI:  http://localhost:8000" -ForegroundColor Green
Write-Host ""
Write-Host "Next: Open http://localhost:8000 in your browser" -ForegroundColor Yellow
Write-Host "Stop: Run shutdown.ps1" -ForegroundColor Yellow
Write-Host ""
