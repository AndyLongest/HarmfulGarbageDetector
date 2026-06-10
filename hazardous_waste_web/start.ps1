$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $PSScriptRoot
$AppUrl = "http://127.0.0.1:8000"
$HealthUrl = "$AppUrl/api/health"
$LogFile = Join-Path $PSScriptRoot "server.log"
$ErrorLogFile = Join-Path $PSScriptRoot "server-error.log"
$Weights = Join-Path $Root "exp12\weights\best.pt"

function Test-Server {
    try {
        Invoke-WebRequest -UseBasicParsing $HealthUrl -TimeoutSec 2 | Out-Null
        return $true
    } catch {
        return $false
    }
}

function Open-App {
    if (-not $env:NO_BROWSER) {
        Start-Process $AppUrl
    }
}

Write-Host ""
Write-Host "==========================================" -ForegroundColor DarkGreen
Write-Host "  Hazardous Waste Detection - Quick Start" -ForegroundColor Green
Write-Host "==========================================" -ForegroundColor DarkGreen
Write-Host ""

if (Test-Server) {
    Write-Host "[OK] Service is already running." -ForegroundColor Green
    Open-App
    Start-Sleep -Seconds 2
    exit 0
}

if (-not (Get-Command conda -ErrorAction SilentlyContinue)) {
    Write-Host "[ERROR] Conda was not found. Install Anaconda/Miniconda or add Conda to PATH." -ForegroundColor Red
    Read-Host "Press Enter to close"
    exit 1
}

if (-not (Test-Path $Weights)) {
    Write-Host "[ERROR] Model weights were not found: $Weights" -ForegroundColor Red
    Read-Host "Press Enter to close"
    exit 1
}

Write-Host "[1/3] Checking the YOLO Conda environment..."
& conda run -n YOLO python -c "import torch, cv2, PIL" 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] The YOLO environment is missing or incomplete." -ForegroundColor Red
    Read-Host "Press Enter to close"
    exit 1
}

Write-Host "[2/3] Starting the detection service..."
Remove-Item $LogFile, $ErrorLogFile -Force -ErrorAction SilentlyContinue
$process = Start-Process -FilePath "conda" `
    -ArgumentList @("run", "-n", "YOLO", "--no-capture-output", "python", "hazardous_waste_web/app.py") `
    -WorkingDirectory $Root `
    -WindowStyle Hidden `
    -RedirectStandardOutput $LogFile `
    -RedirectStandardError $ErrorLogFile `
    -PassThru

Write-Host "[3/3] Waiting for the model to load..."
for ($attempt = 1; $attempt -le 40; $attempt++) {
    if (Test-Server) {
        Write-Host ""
        Write-Host "[OK] Application started: $AppUrl" -ForegroundColor Green
        Open-App
        Start-Sleep -Seconds 2
        exit 0
    }
    if ($process.HasExited) {
        break
    }
    Start-Sleep -Seconds 1
}

Write-Host ""
Write-Host "[ERROR] The service failed to start. See logs:" -ForegroundColor Red
Write-Host $LogFile
Write-Host $ErrorLogFile
Read-Host "Press Enter to close"
exit 1
