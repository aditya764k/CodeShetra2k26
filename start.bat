@echo off
cd /d "D:\Hackathon\DemoCodeShetra2"

:: Activate the virtual environment
call venv\Scripts\activate.bat

:: Start FastAPI backend in a new window
start "FastAPI Backend" cmd /k "uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload"

:: Start Streamlit frontend in another new window
start "Streamlit UI" cmd /k "streamlit run app\app.py --server.port=8501"

