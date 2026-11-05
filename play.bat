@echo off
chcp 65001 >nul
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo 请先安装：python -m venv .venv
    echo 然后运行：.venv\Scripts\python.exe -m pip install -e .
    pause
    exit /b 1
)
if not "%~1"=="" goto args
echo.
echo  3x3 2048 - 选择版本
echo.
echo  1. MC - 蒙特卡洛树搜索 MCTS
echo  2. TD - 时序差分学习
echo  3. 完整规划 - 推荐
echo.
set "selected=3"
set /p "selected=输入1、2或3，直接回车使用推荐版本："
set "agent=best"
if "%selected%"=="1" set "agent=mc"
if "%selected%"=="2" set "agent=td"
if "%selected%"=="3" goto run
if "%selected%"=="1" goto run
if "%selected%"=="2" goto run
echo 输入无效，请重新运行并输入1、2或3。
pause
exit /b 1
:run
".venv\Scripts\python.exe" -m game2048.autoplay --agent %agent%
if errorlevel 1 pause
exit /b
:args
".venv\Scripts\python.exe" -m game2048.autoplay %*
if errorlevel 1 pause
