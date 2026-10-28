@echo off
rem Naming the command window so the running project is easy to identify
title Privacy-First Job Recommendation System

rem Switching to the folder containing this launcher regardless of the opening location
cd /d "%~dp0"

rem Checking for the Windows Python launcher before attempting environment creation
where py >nul 2>nul
if errorlevel 1 (
    echo Python was not found. Install Python 3.11 and select Add Python to PATH.
    pause
    exit /b 1
)

rem Creating an isolated project environment only when it does not already exist
if not exist ".venv\Scripts\python.exe" (
    echo Creating the local Python environment...
    py -3.11 -m venv .venv
    if errorlevel 1 (
        py -3 -m venv .venv
    )
    if not exist ".venv\Scripts\python.exe" (
        echo Creating the Python environment failed. Install Python 3.11 and try again.
        pause
        exit /b 1
    )
)

rem Checking Streamlit as a quick indicator that project dependencies are already installed
".venv\Scripts\python.exe" -c "import streamlit" >nul 2>nul
if errorlevel 1 (
    echo Installing the required packages...
    ".venv\Scripts\python.exe" -m pip install --upgrade pip
    ".venv\Scripts\python.exe" -m pip install -r requirements.txt
    if errorlevel 1 (
        echo Installing packages failed. Check the internet connection and run this file again.
        pause
        exit /b 1
    )
)

rem Launching the local Streamlit server by using the isolated environment's Python executable
echo Opening the Privacy-First Job Recommendation System in the browser...
".venv\Scripts\python.exe" -m streamlit run app.py

rem Keeping the command window open after shutdown so startup errors remain readable
pause
