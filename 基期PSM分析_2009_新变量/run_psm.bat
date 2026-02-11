@echo off
chcp 65001 >nul
echo ========================================
echo 基期PSM分析 (2009年)
echo ========================================
echo.
echo 正在安装依赖包...
pip install -r requirements.txt -q
echo.
echo 正在运行PSM分析...
python psm_analysis.py
echo.
echo 分析完成! 请查看生成的输出文件。
echo.
pause
