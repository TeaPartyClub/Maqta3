@echo off
set PATH=C:\ffmpeg\bin;%PATH%
cd /d "%~dp0backend"

echo Checking dependencies...
pip install -r requirements.txt --quiet

set HF_TOKEN=***REMOVED-HF-TOKEN***
set ANTHROPIC_API_KEY=***REMOVED-ANTHROPIC-KEY***

echo.
echo Starting Maqta3 server...
echo Open http://localhost:8000 in your browser
echo Press Ctrl+C to stop
echo.

start "" http://localhost:8000
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --no-access-log

pause
