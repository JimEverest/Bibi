@echo off
chcp 65001 >nul
title Bibi 依赖安装（内网两步版）
cd /d "%~dp0"

echo ================================================
echo  Bibi 依赖安装 - 两步版（绕开 funasr_onnx 元数据限制）
echo  说明：funasr_onnx 声明的 numpy<=1.26.4 在 Python 3.13
echo  没有预编译包，实测 numpy 2.x 可正常运行，故绕过其检查。
echo ================================================
echo.

if not exist ".venv\Scripts\python.exe" (
    echo [错误] 未找到 .venv 虚拟环境，请先创建：
    echo     python -m venv .venv
    pause
    exit /b 1
)

echo [1/2] 先单独安装 funasr_onnx（跳过依赖元数据检查）...
".venv\Scripts\python.exe" -m pip install --no-deps funasr_onnx==0.4.1
if errorlevel 1 (
    echo [失败] funasr_onnx 安装失败，请检查网络后重试
    pause
    exit /b 1
)

echo.
echo [2/2] 安装其余依赖（已锁版本，全部有预编译 wheel）...
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 (
    echo [失败] 依赖安装失败，请把完整报错发给技术支持
    pause
    exit /b 1
)

echo.
echo ================================================
echo  安装完成！验证中...
echo ================================================
".venv\Scripts\python.exe" -c "import numpy, onnxruntime, funasr_onnx, librosa, sounddevice, keyboard; print('numpy', numpy.__version__); print('onnxruntime', onnxruntime.__version__); print('funasr_onnx OK'); print('ALL_IMPORTS_OK')"

echo.
echo 如上方显示 ALL_IMPORTS_OK 即安装成功，可运行 main.py
pause
