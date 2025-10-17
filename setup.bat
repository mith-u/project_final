@echo off
REM ============================================================================
REM  Project Setup Script for Network Anomaly Detection
REM ============================================================================

echo Creating project directory structure...

REM Create main directories
if not exist "data" mkdir data
if not exist "models" mkdir models
if not exist "results" mkdir results

REM Create subdirectories for results
if not exist "results\isolation_forest" mkdir results\isolation_forest
if not exist "results\lstm" mkdir results\lstm

echo.
echo [IMPORTANT] Directory structure created successfully!
echo.
echo Please place your NSL-KDD dataset files into the 'data' folder:
echo   - Place KDDTrain+.txt in network_anomaly_detection\data\
echo   - Place KDDTest+.txt  in network_anomaly_detection\data\
echo.
pause

echo.
echo Installing required Python libraries from requirements.txt...
pip install -r requirements.txt

echo.
echo ============================================================================
echo  Setup Complete!
echo ============================================================================
echo You can now run the main analysis script by executing: python 2_main.py
echo.
pause