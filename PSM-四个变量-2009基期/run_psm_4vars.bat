@echo off
chcp 65001 >nul
echo ========================================
echo PSM分析 - 四个匹配变量（2009基期）
echo ========================================
echo.
echo 匹配变量：
echo   - ln_real_gdp
echo   - In_人口密度
echo   - In_金融发展水平
echo   - 第二产业占GDP比重
echo.
echo 匹配设定：
echo   - 基期：2009年
echo   - 卡尺：倾向得分对数几率标准差的0.25倍
echo   - 匹配方法：1:1有放回匹配
echo.
echo 正在运行PSM分析...
echo ========================================
echo.

python psm_analysis_4vars.py

echo.
echo ========================================
echo 分析完成！
echo ========================================
pause
