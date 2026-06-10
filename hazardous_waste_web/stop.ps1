$connections = Get-NetTCPConnection -LocalPort 8000 -State Listen -ErrorAction SilentlyContinue

if (-not $connections) {
    Write-Host "Detection service is not running."
    Start-Sleep -Seconds 2
    exit 0
}

$connections | Select-Object -ExpandProperty OwningProcess -Unique | ForEach-Object {
    Stop-Process -Id $_ -Force -ErrorAction SilentlyContinue
}

Write-Host "Detection service stopped."
Start-Sleep -Seconds 2
