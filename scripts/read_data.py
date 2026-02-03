# -*- coding: utf-8 -*-
import pandas as pd

# 读取CSV文件
print("正在读取CSV文件...")
csv_df = pd.read_csv('Low_Carbon_Pilot_Cities_Complete.csv', encoding='gbk')
print(f"CSV列名: {csv_df.columns.tolist()}")
print(f"CSV形状: {csv_df.shape}")
print(f"\n前10行数据:")
print(csv_df.head(10))
print(f"\n唯一城市数: {csv_df['City'].nunique()}")

# 读取Excel文件
print("\n" + "="*50)
print("正在读取Excel文件...")
excel_df = pd.read_excel('总数据集_2007-2023_完整版.xlsx')
print(f"Excel列名: {excel_df.columns.tolist()[:20]}")  # 显示前20列
print(f"Excel总列数: {len(excel_df.columns)}")
print(f"Excel形状: {excel_df.shape}")
print(f"\n前5行数据:")
print(excel_df.head())
