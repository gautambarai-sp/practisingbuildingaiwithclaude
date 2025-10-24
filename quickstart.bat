@echo off
REM AI Inventory Optimizer - Quick Start Script for Windows

echo.
echo ===============================================
echo AI Inventory Optimization System - Quick Start
echo ===============================================
echo.

REM Check Python installation
echo Checking prerequisites...
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.9 or higher from python.org
    pause
    exit /b 1
)

for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
echo [OK] Found Python %PYTHON_VERSION%

REM Check Git installation
git --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Git is not installed
    echo Please install Git from git-scm.com
    pause
    exit /b 1
)
echo [OK] Git is installed
echo.

REM Create project directory
set PROJECT_DIR=ai-inventory-optimizer
if exist "%PROJECT_DIR%" (
    echo WARNING: Directory %PROJECT_DIR% already exists
    set /p OVERWRITE="Do you want to overwrite it? (Y/N): "
    if /i not "%OVERWRITE%"=="Y" (
        echo Aborted.
        pause
        exit /b 1
    )
    rmdir /s /q "%PROJECT_DIR%"
)

echo Creating project structure...
mkdir "%PROJECT_DIR%"
cd "%PROJECT_DIR%"
mkdir .streamlit
mkdir data
mkdir tests
echo [OK] Created project structure
echo.

REM Create virtual environment
echo Creating virtual environment...
python -m venv venv
if errorlevel 1 (
    echo ERROR: Failed to create virtual environment
    pause
    exit /b 1
)
echo [OK] Virtual environment created
echo.

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat
echo [OK] Virtual environment activated
echo.

REM Create requirements.txt
echo Creating requirements.txt...
(
echo streamlit==1.31.0
echo pandas==2.1.4
echo numpy==1.26.3
echo plotly==5.18.0
echo prophet==1.1.5
echo scikit-learn==1.4.0
echo openpyxl==3.1.2
) > requirements.txt
echo [OK] Created requirements.txt
echo.

REM Install dependencies
echo Installing dependencies ^(this may take a few minutes^)...
python -m pip install --upgrade pip --quiet
pip install -r requirements.txt --quiet
if errorlevel 1 (
    echo ERROR: Failed to install dependencies
    pause
    exit /b 1
)
echo [OK] Dependencies installed
echo.

REM Create .gitignore
echo Creating configuration files...
(
echo # Python
echo __pycache__/
echo *.py[cod]
echo venv/
echo env/
echo.
echo # Streamlit
echo .streamlit/secrets.toml
echo.
echo # Data
echo data/*.csv
echo !data/sample_*.csv
echo.
echo # Environment
echo .env
echo.
echo # OS
echo .DS_Store
echo Thumbs.db
) > .gitignore

REM Create Streamlit config
(
echo [theme]
echo primaryColor = "#3b82f6"
echo backgroundColor = "#ffffff"
echo secondaryBackgroundColor = "#f0f2f6"
echo textColor = "#262730"
echo font = "sans serif"
echo.
echo [server]
echo headless = true
echo port = 8501
) > .streamlit\config.toml
echo [OK] Configuration files created
echo.

REM Create sample data
echo Creating sample data...
(
echo date,sku,product_name,category,quantity_sold,revenue,price
echo 2024-01-01,TRD-BLK-M,Black Dress Medium,Dresses,12,959.88,79.99
echo 2024-01-01,JNS-BLU-32,Blue Jeans 32,Jeans,8,479.92,59.99
echo 2024-01-01,TSH-WHT-L,White T-Shirt Large,T-Shirts,15,374.85,24.99
) > data\sample_sales.csv

(
echo sku,product_name,current_stock,lead_time_days,category,price
echo TRD-BLK-M,Black Dress Medium,145,7,Dresses,79.99
echo JNS-BLU-32,Blue Jeans 32,89,5,Jeans,59.99
echo TSH-WHT-L,White T-Shirt Large,234,3,T-Shirts,24.99
) > data\sample_inventory.csv
echo [OK] Sample data created
echo.

REM Initialize git
echo Initializing Git repository...
git init --quiet
git add .
git commit -m "Initial commit: AI Inventory Optimization System" --quiet
echo [OK] Git repository initialized
echo.

REM Final instructions
echo.
echo ===============================================
echo            Setup Complete!
echo ===============================================
echo.
echo Next Steps:
echo.
echo 1. Add the main app.py file to this directory
echo    ^(Copy the content from the artifact provided^)
echo.
echo 2. Test locally:
echo    cd %PROJECT_DIR%
echo    venv\Scripts\activate
echo    streamlit run app.py
echo.
echo 3. Deploy to GitHub:
echo    - Create repository on GitHub
echo    - git remote add origin https://github.com/YOUR-USERNAME/ai-inventory-optimizer.git
echo    - git push -u origin main
echo.
echo 4. Deploy to Streamlit Cloud:
echo    - Visit https://share.streamlit.io
echo    - Connect your GitHub repository
echo    - Deploy!
echo.
echo Full documentation: See DEPLOYMENT_GUIDE.md
echo.
echo Happy coding!
echo.
pause
