@echo off
set PATH=C:\ffmpeg\bin;%PATH%
set "ROOT=%~dp0"
cd /d "%~dp0backend"

echo Checking dependencies...
pip install -r requirements.txt --quiet

REM Load API keys from the untracked .env file (see .env.example)
if exist "%ROOT%.env" (
    for /f "usebackq eol=# tokens=1,* delims==" %%a in ("%ROOT%.env") do (
        if not "%%~a"=="" set "%%~a=%%~b"
    )
) else (
    echo Warning: %ROOT%.env not found - copy .env.example to .env and fill in your keys.
)

echo.
echo Starting Maqta3 server...
echo Open http://localhost:8000 in your browser
echo Press Ctrl+C to stop
echo.

start "" http://localhost:8000
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --no-access-log

pause
