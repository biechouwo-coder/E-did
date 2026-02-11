@echo off
chcp 65001 >nul
echo ================================================
echo 基期PSM分析 (2009年) - 有放回匹配
echo ================================================
echo.

REM 激活conda环境（如果需要）
REM call conda activate your_env_name

echo 正在安装依赖包...
pip install -r requirements.txt

echo.
echo 开始运行PSM分析...
python psm_analysis.py

echo.
echo ================================================
echo 分析完成！
echo ================================================
pause
