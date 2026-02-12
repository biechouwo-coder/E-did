"""
检查原始数据结构
"""
import pandas as pd
import numpy as np

# 读取数据
df = pd.read_excel('../总数据集2007-2023_仅含emission城市_更新DID.xlsx')

print("="*80)
print("原始数据结构检查")
print("="*80)

print(f"\n数据形状: {df.shape}")
print(f"列名: {df.columns.tolist()}")

print(f"\n年份范围: {df['year'].min()} - {df['year'].max()}")
print(f"各年份数据量:")
print(df['year'].value_counts().sort_index())

# 检查2007-2019年的数据
df_2007_2019 = df[df['year'].between(2007, 2019)]
print(f"\n2007-2019年数据量: {len(df_2007_2019)}")
print(f"城市数量: {df_2007_2019['city_name'].nunique()}")

# 检查2009年基期数据
df_2009 = df[df['year'] == 2009]
print(f"\n2009年（基期）数据量: {len(df_2009)}")
print(f"2009年城市数量: {df_2009['city_name'].nunique()}")

# 检查处理组和控制组
if 'Treat' in df.columns:
    print(f"\n2009年处理组城市数: {(df_2009['Treat'] == 1).sum()}")
    print(f"2009年控制组城市数: {(df_2009['Treat'] == 0).sum()}")
else:
    print("\n警告: 数据中没有'Treat'列")

# 检查四个变量是否存在
variables = ['ln_real_gdp', 'ln_人口密度', 'ln_金融发展水平', '第二产业占GDP比重']
print(f"\n检查PSM变量:")
for var in variables:
    if var in df.columns:
        missing_2009 = df_2009[var].isna().sum()
        missing_all = df_2007_2019[var].isna().sum()
        print(f"  ✓ {var}: 存在")
        print(f"    - 2009年缺失: {missing_2009}/{len(df_2009)}")
        print(f"    - 2007-2019年缺失: {missing_all}/{len(df_2007_2019)}")
    else:
        print(f"  ✗ {var}: 不存在")

print("\n"+"="*80)
