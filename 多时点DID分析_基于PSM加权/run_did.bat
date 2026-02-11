@echo off
chcp 65001 >nul
echo ================================================
echo 多时点DID分析（基于PSM加权）
echo ================================================
echo.

REM 激活conda环境（如果需要）
REM call conda activate your_env_name

echo 正在安装依赖包...
pip install -r requirements.txt

echo.
echo 开始运行DID分析...
python did_analysis.py

echo.
echo ================================================
echo 分析完成！
echo ================================================
pause
