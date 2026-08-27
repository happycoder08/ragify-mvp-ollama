Write-Host ""
Write-Host "========== RAGify - Clean Storage ==========" -ForegroundColor Cyan

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$vectorstoreDir = Join-Path $scriptDir "vectorstore"
$uploadsDir = Join-Path $scriptDir "uploads"

Write-Host "Clearing vector database..." -ForegroundColor Yellow
if (Test-Path $vectorstoreDir) {
    Remove-Item -Path $vectorstoreDir -Recurse -Force
    Write-Host "OK - Removed $vectorstoreDir" -ForegroundColor Green
} else {
    Write-Host "Not found" -ForegroundColor Yellow
}

Write-Host "Clearing document uploads..." -ForegroundColor Yellow
if (Test-Path $uploadsDir) {
    Remove-Item -Path $uploadsDir -Recurse -Force
    Write-Host "OK - Removed $uploadsDir" -ForegroundColor Green
} else {
    Write-Host "Not found" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "========== STORAGE CLEARED ==========" -ForegroundColor Green
Write-Host "Vector database and uploads are now clean" -ForegroundColor Green
Write-Host ""
Write-Host "Tip: Restart services with startup.ps1 to reinitialize" -ForegroundColor Yellow
Write-Host ""
