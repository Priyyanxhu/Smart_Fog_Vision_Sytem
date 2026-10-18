@echo off
title Smart Fog Vision - http://localhost:8001
echo Starting Smart Fog Vision on http://localhost:8001
echo.
echo Keep this window open (minimize it). Closing it stops the server
echo and the page will go blank in the browser.
echo.
python -m http.server 8001 --directory "%~dp0"
echo.
echo Server stopped. Run this file again, then press Ctrl+Shift+R in the browser.
pause
