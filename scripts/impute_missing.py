# -*- coding: utf-8 -*-
"""
改进的缺失值插值策略
"""
import pandas as pd
import numpy as np
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

print("="*70)
print("改进的缺失值插值处理")
print("="*70)

# 读取数据
df = pd.read_excel('总数据集_2007-2023_完整版_DID_with_treat.xlsx')

control_vars = ['gdp_per_capita', 'pop_density', 'industrial_upgrading', 'ln_fdi']

print("\n插值策略:")
print("  1. 时间序列线性插值（针对单个城市的时间序列）")
print("  2. 同组别同年份均值填充（针对整个序列缺失的城市）")
print("  3. 同组别均值填充（针对仍然缺失的值）")

# 创建数据的副本
df_imputed = df.copy()

# 步骤1: 时间序列线性插值
print("\n步骤1: 时间序列线性插值")
print("-"*70)

cities = df['city_name'].unique()

for var in control_vars:
    df_city_imputed = pd.DataFrame()

    for city in cities:
        city_data = df[df['city_name'] == city].copy()
        city_data = city_data.sort_values('year')

        # 线性插值
        city_data[var] = city_data[var].interpolate(method='linear', limit_direction='both')

        # 首尾填充
        city_data[var] = city_data[var].fillna(method='ffill').fillna(method='bfill')

        df_city_imputed = pd.concat([df_city_imputed, city_data])

    df_imputed[var] = df_city_imputed.sort_index()[var]

    missing_after_step1 = df_imputed[var].isnull().sum()
    print(f"{var}: 时间序列插值后仍有 {missing_after_step1} 个缺失")

# 步骤2: 同组别同年份均值填充
print("\n步骤2: 同组别同年份均值填充")
print("-"*70)

for var in control_vars:
    missing_before = df_imputed[var].isnull().sum()

    if missing_before > 0:
        # 对每个缺失值，用同treat组、同年份的均值填充
        for year in df_imputed['year'].unique():
            for treat_val in [0, 1]:
                mask = (df_imputed['year'] == year) & (df_imputed['treat'] == treat_val)
                group_mean = df_imputed[mask][var].mean()

                # 只填充缺失值
                missing_mask = mask & df_imputed[var].isnull()
                if not pd.isna(group_mean):
                    df_imputed.loc[missing_mask, var] = group_mean

    missing_after = df_imputed[var].isnull().sum()
    print(f"{var}: 同组同年填充后从 {missing_before} → {missing_after} 个缺失")

# 步骤3: 同组别全局均值填充
print("\n步骤3: 同组别全局均值填充")
print("-"*70)

for var in control_vars:
    missing_before = df_imputed[var].isnull().sum()

    if missing_before > 0:
        # 对每个缺失值，用同treat组的全局均值填充
        for treat_val in [0, 1]:
            group_mean = df_imputed[df_imputed['treat'] == treat_val][var].mean()

            missing_mask = (df_imputed['treat'] == treat_val) & df_imputed[var].isnull()
            if not pd.isna(group_mean):
                df_imputed.loc[missing_mask, var] = group_mean

    missing_after = df_imputed[var].isnull().sum()
    print(f"{var}: 同组全局填充后从 {missing_before} → {missing_after} 个缺失")

# 步骤4: 检查插值效果
print("\n步骤4: 插值效果验证")
print("-"*70)

df_2009_original = df[df['year'] == 2009]
df_2009_imputed = df_imputed[df_imputed['year'] == 2009]

print("\n2009年（基期）样本量变化:")
print(f"  原始总样本: {len(df_2009_original)}")
print(f"  插值后总样本: {len(df_2009_imputed)}")
print(f"\n  原始处理组: {(df_2009_original['treat']==1).sum()}")
print(f"  插值后处理组: {(df_2009_imputed['treat']==1).sum()}")
print(f"\n  原始对照组: {(df_2009_original['treat']==0).sum()}")
print(f"  插值后对照组: {(df_2009_imputed['treat']==0).sum()}")

print("\n2009年各变量缺失值情况:")
for var in control_vars:
    missing_orig = df_2009_original[var].isnull().sum()
    missing_imputed = df_2009_imputed[var].isnull().sum()
    print(f"  {var}: {missing_orig} → {missing_imputed}")

# 检查插值后的统计描述
print("\n\n步骤5: 插值后统计描述（2009年）")
print("-"*70)

for var in control_vars:
    print(f"\n{var}:")

    # 处理组
    treat_data = df_2009_imputed[df_2009_imputed['treat']==1][var]
    print(f"  处理组(treat=1): 均值={treat_data.mean():.2f}, 标准差={treat_data.std():.2f}, 样本={len(treat_data)}")

    # 对照组
    control_data = df_2009_imputed[df_2009_imputed['treat']==0][var]
    print(f"  对照组(treat=0): 均值={control_data.mean():.2f}, 标准差={control_data.std():.2f}, 样本={len(control_data)}")

# 保存插值后的数据
print("\n\n步骤6: 保存插值后的数据")
print("-"*70)

output_file = '总数据集_2007-2023_完整版_DID_with_treat_imputed.xlsx'
df_imputed.to_excel(output_file, index=False)
print(f"✓ 插值后数据已保存至: {output_file}")

print("\n" + "="*70)
print("插值处理完成！")
print("="*70)

# 计算总体改善情况
total_missing_before = sum(df[v].isnull().sum() for v in control_vars)
total_missing_after = sum(df_imputed[v].isnull().sum() for v in control_vars)

print(f"\n总体改善:")
print(f"  • 插值前总缺失: {total_missing_before} 个观测")
print(f"  • 插值后总缺失: {total_missing_after} 个观测")
print(f"  • 插值成功率: {(1 - total_missing_after/total_missing_before)*100:.1f}%")
print(f"  • 2009年可用样本: 203 → {len(df_2009_imputed)} ({len(df_2009_imputed)/203*100:.1f}%)")
