@echo off
chcp 65001 >nul
title Bibi 离线语音输入
cd /d "%~dp0"

REM 默认使用 SenseVoice 引擎（中英混说更准）；如需回退旧引擎，注释掉下面一行
set ASR_ENGINE=sensevoice

REM 优先使用项目自带 venv（本机已配置好的环境）
set VENV_PY=%~dp0..\.workbuddy_env\venv\Scripts\python.exe
if not exist "%VENV_PY%" set VENV_PY=C:\Users\tianw\.workbuddy\binaries\python\envs\vocotype\Scripts\python.exe
if not exist "%VENV_PY%" set VENV_PY=%~dp0.venv\Scripts\python.exe

echo ================================================
echo  Bibi 离线语音输入 (CPU 本地推理, 无需联网)
echo  引擎: %ASR_ENGINE%
echo  按 F2 开始/停止录音, 识别文字自动输入光标处
echo  退出请按 Ctrl+C 或直接关闭本窗口
echo ================================================
echo.

"%VENV_PY%" main.py
pause
