# Update DSAI on Windows, then check the result.
#
# Run from the project folder in PowerShell:
#     .\scripts\update.ps1
#
# Stop the running app first (Ctrl+C in its terminal). Streamlit keeps imported
# modules in memory, so a server left running serves the old code no matter how
# successful the pull was — that is the most common reason an update "does not
# work".

$ErrorActionPreference = "Stop"
$branch = "claude/ai-data-analytics-platform-pc429g"

Set-Location (Split-Path $PSScriptRoot -Parent)
Write-Host "Working in $(Get-Location)" -ForegroundColor Cyan

if (-not (Test-Path "pyproject.toml")) {
    Write-Host "This is not the project folder — pyproject.toml is not here." -ForegroundColor Red
    exit 1
}

# A running server on 8501 will keep serving the old code.
$listening = Get-NetTCPConnection -LocalPort 8501 -State Listen -ErrorAction SilentlyContinue
if ($listening) {
    Write-Host "Something is already serving port 8501. Stop it (Ctrl+C in its terminal) before" -ForegroundColor Yellow
    Write-Host "starting again, or the app you open will still be the old version." -ForegroundColor Yellow
}

if (Test-Path ".venv\Scripts\Activate.ps1") {
    Write-Host "Activating .venv" -ForegroundColor Cyan
    & ".\.venv\Scripts\Activate.ps1"
} else {
    Write-Host "No .venv found — installing into whichever Python is on PATH." -ForegroundColor Yellow
}

$dirty = git status --porcelain
if ($dirty) {
    Write-Host "You have local changes. Stashing them so the pull can proceed." -ForegroundColor Yellow
    git stash push -m "dsai update $(Get-Date -Format s)"
    $stashed = $true
}

Write-Host "Pulling $branch" -ForegroundColor Cyan
git pull origin $branch

if ($stashed) {
    Write-Host "Restoring your local changes" -ForegroundColor Cyan
    git stash pop
}

Write-Host "Installing dependencies" -ForegroundColor Cyan
pip install -e ".[full]" --quiet

Write-Host ""
dsai doctor
Write-Host ""
Write-Host "Start the app with:  dsai app" -ForegroundColor Green
