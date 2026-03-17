# Nexus launcher for Windows
$ErrorActionPreference = "Stop"

# Activate venv if it exists
if (Test-Path ".venv\Scripts\Activate.ps1") {
    . .venv\Scripts\Activate.ps1
}

# Load .env if it exists
if (Test-Path ".env") {
    Get-Content ".env" | ForEach-Object {
        if ($_ -match "^\s*([^#][^=]+)=(.*)$") {
            [Environment]::SetEnvironmentVariable($matches[1].Trim(), $matches[2].Trim(), "Process")
        }
    }
}

Write-Host "Starting Nexus..." -ForegroundColor Cyan
uvicorn nexus.main:app --reload --host 0.0.0.0 --port 8000
