#!/bin/bash
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT/backend"

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
fi

# Activate virtual environment
source .venv/bin/activate

# Use ffmpeg-full (includes libass for subtitle burning) if available
FFMPEG_FULL="$(brew --prefix ffmpeg-full 2>/dev/null)/bin"
[ -d "$FFMPEG_FULL" ] && export PATH="$FFMPEG_FULL:$PATH"

echo "Checking dependencies..."
pip install -r requirements.txt --quiet

# Load API keys from the untracked .env file (see .env.example)
if [ -f "$ROOT/.env" ]; then
    set -a
    . "$ROOT/.env"
    set +a
else
    echo "Warning: $ROOT/.env not found — copy .env.example to .env and fill in your keys."
fi

echo ""
echo "Starting Maqta3 server..."
echo "Open http://localhost:8000 in your browser"
echo "Press Ctrl+C to stop"
echo ""

# Wait for server to be ready, then open browser
(sleep 2 && open http://localhost:8000) &
python3 -m uvicorn main:app --host 0.0.0.0 --port 8000 --no-access-log
