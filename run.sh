#!/bin/bash
cd "$(dirname "$0")/backend"

echo "Checking dependencies..."
pip3 install -r requirements.txt --quiet

export HF_TOKEN=***REMOVED-HF-TOKEN***

echo ""
echo "Starting Maqta3 server..."
echo "Open http://localhost:8000 in your browser"
echo "Press Ctrl+C to stop"
echo ""

# Wait for server to be ready, then open browser
(sleep 2 && open http://localhost:8000) &
python3 -m uvicorn main:app --host 0.0.0.0 --port 8000
