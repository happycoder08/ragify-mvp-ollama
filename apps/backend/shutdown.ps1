Write-Host ""
Write-Host "========== RAGify MVP - Stopping Services ==========" -ForegroundColor Red

Write-Host "[1/4] Cleaning data..." -ForegroundColor Cyan
# Clear PostgreSQL data before stopping
$pgRunning = docker ps --filter "name=ragify-postgres" --format "{{.Names}}" 2>$null
if ($pgRunning -eq "ragify-postgres") {
    docker exec ragify-postgres psql -U ragify -d ragify_db -c "TRUNCATE TABLE documents, conversations, messages RESTART IDENTITY CASCADE;" 2>$null | Out-Null
    Write-Host "OK - Database cleared" -ForegroundColor Green
} else {
    Write-Host "PostgreSQL not running" -ForegroundColor Yellow
}
# Remove uploaded files and vectorstore
Remove-Item -Path ".\uploads\*" -Force -ErrorAction SilentlyContinue
Remove-Item -Path ".\vectorstore" -Recurse -Force -ErrorAction SilentlyContinue
Write-Host "OK - Files and vectorstore cleared" -ForegroundColor Green

Write-Host "[2/4] FastAPI..." -ForegroundColor Cyan
$uvicorn = Get-Process -Name "python" -ErrorAction SilentlyContinue | Where-Object { $_.CommandLine -like "*uvicorn*" }
if ($uvicorn) {
    Stop-Process -InputObject $uvicorn -Force
    Write-Host "OK - Stopped" -ForegroundColor Green
} else {
    Write-Host "Not running" -ForegroundColor Yellow
}

Write-Host "[3/4] Ollama..." -ForegroundColor Cyan
$ollama = Get-Process -Name "ollama" -ErrorAction SilentlyContinue
if ($ollama) {
    Stop-Process -Name "ollama" -Force -ErrorAction SilentlyContinue
    Start-Sleep -Seconds 2
    # Verify all Ollama processes stopped
    $remaining = Get-Process -Name "ollama" -ErrorAction SilentlyContinue
    if ($remaining) {
        Write-Host "WARNING - Some Ollama processes still running, force killing..." -ForegroundColor Yellow
        Stop-Process -Name "ollama" -Force -ErrorAction SilentlyContinue
    }
    Write-Host "OK - Stopped" -ForegroundColor Green
} else {
    Write-Host "Not running" -ForegroundColor Yellow
}

Write-Host "[4/4] PostgreSQL..." -ForegroundColor Cyan
$pg = docker ps --filter "name=ragify-postgres" --format "{{.Names}}" 2>$null
if ($pg -eq "ragify-postgres") {
    docker stop ragify-postgres 2>$null | Out-Null
    Write-Host "OK - Stopped" -ForegroundColor Green
} else {
    Write-Host "Not running" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "========== ALL SERVICES STOPPED ==========" -ForegroundColor Green
Write-Host ""
