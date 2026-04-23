@echo off
setlocal

powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0stop_insightvault.ps1" %*
exit /b %ERRORLEVEL%
