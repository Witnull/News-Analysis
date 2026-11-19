# migrate.ps1 — Run after adding new fields to Article model

param(
    [string]$Message = "auto_migration",
    [string]$DbUrl = "sqlite:///articles.db"
)

$env:DATABASE_URL = $DbUrl

Write-Host "📦 Installing alembic..." -ForegroundColor Cyan
pip install alembic | Out-Null

# Init only if not already initialized
if (-Not (Test-Path "alembic.ini")) {
    Write-Host "🔧 Initializing alembic..." -ForegroundColor Cyan
    alembic init migrations
}

Write-Host "🔍 Generating migration: '$Message'..." -ForegroundColor Yellow
alembic revision --autogenerate -m "$Message"

Write-Host "⬆️  Applying migration..." -ForegroundColor Yellow
alembic upgrade head

Write-Host "✅ Migration complete." -ForegroundColor Green