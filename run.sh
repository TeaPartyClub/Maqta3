#!/bin/bash
cd "$(dirname "$0")/backend"

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

export HF_TOKEN=***REMOVED-HF-TOKEN***

echo ""
echo "Starting Maqta3 server..."
echo "Open http://localhost:8000 in your browser"
echo "Press Ctrl+C to stop"
echo ""

# Wait for server to be ready, then open browser
(sleep 2 && open http://localhost:8000) &
python3 -m uvicorn main:app --host 0.0.0.0 --port 8000
