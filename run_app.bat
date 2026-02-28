@echo off
setlocal

cd /d "%~dp0"

set "VENV_PY=%~dp0chatbot_venv\Scripts\python.exe"
if not exist "%VENV_PY%" (
  set "VENV_PY=python"
)

start "FastAPI Backend" "%VENV_PY%" -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload

%VENV_PY% Frontend\frontend.py
