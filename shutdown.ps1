Write-Host ""
Write-Host "========== RAGify MVP - Stopping Services ==========" -ForegroundColor Red

Write-Host "[1/3] FastAPI..." -ForegroundColor Cyan
$uvicorn = Get-Process -Name "python" -ErrorAction SilentlyContinue | Where-Object { $_.CommandLine -like "*uvicorn*" }
if ($uvicorn) {
    Stop-Process -InputObject $uvicorn -Force
    Write-Host "OK - Stopped" -ForegroundColor Green
} else {
    Write-Host "Not running" -ForegroundColor Yellow
}

Write-Host "[2/3] Ollama..." -ForegroundColor Cyan
$ollama = Get-Process -Name "ollama" -ErrorAction SilentlyContinue
if ($ollama) {
    Stop-Process -InputObject $ollama -Force
    Write-Host "OK - Stopped" -ForegroundColor Green
} else {
    Write-Host "Not running" -ForegroundColor Yellow
}

Write-Host "[3/3] PostgreSQL..." -ForegroundColor Cyan
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
