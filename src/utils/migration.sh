#!/usr/bin/env bash
# migrate.sh — Run after adding new fields to Article model

set -euo pipefail

MESSAGE="${1:-auto_migration}"
DATABASE_URL="${2:-sqlite:///articles.db}"
export DATABASE_URL

# Init only if not already initialized
if [ ! -f "alembic.ini" ]; then
    echo "🔧 Initializing alembic..."
    alembic init migrations
fi

echo "🔍 Generating migration: '$MESSAGE'..."
alembic revision --autogenerate -m "$MESSAGE"

echo "⬆️  Applying migration..."
alembic upgrade head

echo "✅ Migration complete."