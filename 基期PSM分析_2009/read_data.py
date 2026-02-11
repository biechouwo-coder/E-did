import pandas as pd
import numpy as np

# 尝试读取Excel文件
try:
    df = pd.read_excel('../总数据集2007-2023_仅含emission城市_更新DID.xlsx')
    print("成功读取Excel文件")
    print(f"数据形状: {df.shape}")
    print(f"\n列名 ({len(df.columns)} 列):")
    for i, col in enumerate(df.columns):
        print(f"{i+1}. {col}")

    print(f"\n前10行数据:")
    print(df.head(10))

    # 查找可能的列名
    print(f"\n查找关键变量...")
    for col in df.columns:
        if 'gdp' in col.lower() or 'GDP' in col or '人口' in col or '金融' in col or '产业' in col:
            print(f"  - {col}")

except Exception as e:
    print(f"读取失败: {e}")
    print("\n尝试其他方式...")
