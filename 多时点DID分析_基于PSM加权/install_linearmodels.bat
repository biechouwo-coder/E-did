@echo off
chcp 65001 >nul
echo ================================================
echo 安装 linearmodels 包
echo ================================================
echo.
echo 方案1: 从PyPI安装（需要网络连接）
echo --------------------------------
pip install linearmodels
echo.
echo 方案2: 从GitHub安装（如果PyPI无法访问）
echo --------------------------------
echo 请手动下载并安装：
echo.
echo 1. 访问 https://github.com/bashtage/linearmodels
echo 2. 下载最新版本的 zip 文件
echo 3. 解压后运行：python setup.py install
echo.
echo 或者使用 conda（如果安装了Anaconda）:
echo conda install -c conda-forge linearmodels
echo.
echo ================================================
pause
