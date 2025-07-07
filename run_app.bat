@echo off
echo Creating .streamlit directory...
if not exist "%USERPROFILE%\.streamlit" mkdir "%USERPROFILE%\.streamlit"

echo Starting Streamlit application...
streamlit run streamlit_app.py --server.port 8501 --server.address 0.0.0.0
pause 