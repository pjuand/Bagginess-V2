@echo off
echo ============================================
echo  BagginessControl - Control de Calidad Papel
echo ============================================
echo.

:: Instalar dependencias si es necesario
pip install -r requirements.txt --quiet

echo Iniciando aplicacion...
echo.
streamlit run app.py --server.port 8501

pause
