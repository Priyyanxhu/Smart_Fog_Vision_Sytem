@echo off
echo Starting Smart Fog Vision on http://localhost:8001
python -m http.server 8001 --directory "%~dp0"
pause
