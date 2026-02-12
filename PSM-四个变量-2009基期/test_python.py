"""
Python环境测试脚本
用于测试Python和所需库是否正确安装
"""

import sys

print("="*60)
print("Python环境测试")
print("="*60)
print(f"\nPython版本: {sys.version}")
print(f"Python路径: {sys.executable}")

# 测试所需的库
libraries = {
    'pandas': 'pandas',
    'numpy': 'numpy',
    'sklearn': 'scikit-learn',
    'matplotlib': 'matplotlib',
    'seaborn': 'seaborn',
    'scipy': 'scipy',
    'openpyxl': 'openpyxl'
}

print("\n" + "="*60)
print("检查所需的库:")
print("="*60)

missing = []
for module_name, package_name in libraries.items():
    try:
        __import__(module_name)
        print(f"✓ {package_name:20s} - 已安装")
    except ImportError:
        print(f"✗ {package_name:20s} - 未安装")
        missing.append(package_name)

if missing:
    print("\n" + "="*60)
    print("需要安装的库:")
    print("="*60)
    print(f"缺失的库: {', '.join(missing)}")
    print("\n请运行以下命令安装:")
    print(f"pip install {' '.join(missing)}")
else:
    print("\n" + "="*60)
    print("✓ 所有库都已安装，可以运行PSM分析！")
    print("="*60)

print("\n测试完成！")
