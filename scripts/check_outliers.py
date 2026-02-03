# -*- coding: utf-8 -*-
"""
检查 industrial_advanced 和 fdi_openness 变量的异常值
"""
import pandas as pd
import numpy as np
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

print("="*70)
print("变量异常值检查：industrial_advanced 和 fdi_openness")
print("="*70)

# 读取数据
df = pd.read_excel('总数据集_2007-2023_完整版_DID_with_treat.xlsx')

vars_to_check = ['industrial_advanced', 'fdi_openness']

print("\n步骤1: 基本统计信息")
print("-"*70)

for var in vars_to_check:
    print(f"\n【{var}】")
    print(f"缺失值: {df[var].isnull().sum()} / {len(df)} ({df[var].isnull().sum()/len(df)*100:.1f}%)")

    if df[var].notnull().sum() > 0:
        print(f"\n描述性统计:")
        print(df[var].describe())

print("\n\n步骤2: 异常值检测（多种方法）")
print("-"*70)

for var in vars_to_check:
    print(f"\n{'='*70}")
    print(f"【{var}】")
    print('='*70)

    data = df[var].dropna()

    if len(data) == 0:
        print("  无有效数据")
        continue

    # 方法1: IQR方法（箱线图规则）
    print("\n方法1: IQR方法（箱线图规则）")
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers_iqr = data[(data < lower_bound) | (data > upper_bound)]
    print(f"  Q1 (25%): {Q1:.4f}")
    print(f"  Q3 (75%): {Q3:.4f}")
    print(f"  IQR: {IQR:.4f}")
    print(f"  下界: {lower_bound:.4f}")
    print(f"  上界: {upper_bound:.4f}")
    print(f"  异常值数量: {len(outliers_iqr)} / {len(data)} ({len(outliers_iqr)/len(data)*100:.1f}%)")

    if len(outliers_iqr) > 0:
        print(f"  异常值范围: {outliers_iqr.min():.4f} ~ {outliers_iqr.max():.4f}")
        print(f"  异常值样本:")
        outlier_df = df[df[var].isin(outliers_iqr)][['year', 'city_name', var]].head(10)
        print(outlier_df.to_string(index=False))

    # 方法2: Z-score方法（3σ原则）
    print("\n方法2: Z-score方法（3σ原则）")
    mean = data.mean()
    std = data.std()

    z_scores = np.abs((data - mean) / std)
    outliers_zscore = data[z_scores > 3]

    print(f"  均值: {mean:.4f}")
    print(f"  标准差: {std:.4f}")
    print(f"  阈值: ±3倍标准差")
    print(f"  异常值数量: {len(outliers_zscore)} / {len(data)} ({len(outliers_zscore)/len(data)*100:.1f}%)")

    if len(outliers_zscore) > 0:
        print(f"  异常值范围: {outliers_zscore.min():.4f} ~ {outliers_zscore.max():.4f}")

    # 方法3: 极值检查
    print("\n方法3: 极值检查")
    print(f"  最小值: {data.min():.4f}")
    print(f"  最大值: {data.max():.4f}")
    print(f"  极差: {data.max() - data.min():.4f}")

    # 检查负值
    negative_values = data[data < 0]
    if len(negative_values) > 0:
        print(f"  负值数量: {len(negative_values)} ({len(negative_values)/len(data)*100:.1f}%)")
        print(f"  贍值范围: {negative_values.min():.4f} ~ {negative_values.max():.4f}")
    else:
        print(f"  无负值")

    # 检查零值
    zero_values = data[data == 0]
    if len(zero_values) > 0:
        print(f"  零值数量: {len(zero_values)} ({len(zero_values)/len(data)*100:.1f}%)")
    else:
        print(f"  无零值")

    # 方法4: 分布可视化信息
    print("\n方法4: 分布特征")
    print(f"  偏度 (Skewness): {data.skew():.4f}")
    print(f"  峰度 (Kurtosis): {data.kurtosis():.4f}")

    if abs(data.skew()) > 2:
        print(f"  ⚠️  高偏度警告: |偏度| > 2，分布严重偏斜")
    elif abs(data.skew()) > 1:
        print(f"  ⚠️  中等偏度: 1 < |偏度| < 2")

    if data.kurtosis() > 3:
        print(f"  ⚠️  高峰度警告: 峰度 > 3，存在厚尾分布")

print("\n\n步骤3: 按年份分组统计")
print("-"*70)

for var in vars_to_check:
    print(f"\n【{var}】按年份统计:")
    yearly_stats = df.groupby('year')[var].agg(['mean', 'std', 'min', 'max', 'count'])
    print(yearly_stats.to_string())

print("\n\n步骤4: 按城市分组检查（极端值的城市）")
print("-"*70)

for var in vars_to_check:
    print(f"\n【{var}】")

    # 计算每个城市的均值和标准差
    city_stats = df.groupby('city_name')[var].agg(['mean', 'std', 'count']).dropna()
    city_stats = city_stats[city_stats['count'] >= 5]  # 至少5年数据

    # 找出均值最高的10个城市
    top_cities = city_stats['mean'].nlargest(10)
    print(f"\n  均值最高的10个城市:")
    for city, mean_val in top_cities.items():
        std_val = city_stats.loc[city, 'std']
        print(f"    {city}: {mean_val:.4f} (std={std_val:.4f})")

    # 找出均值最低的10个城市
    bottom_cities = city_stats['mean'].nsmallest(10)
    print(f"\n  均值最低的10个城市:")
    for city, mean_val in bottom_cities.items():
        std_val = city_stats.loc[city, 'std']
        print(f"    {city}: {mean_val:.4f} (std={std_val:.4f})")

print("\n\n步骤5: 相关性检查")
print("-"*70)

# 检查这两个变量之间的相关性
data_corr = df[['industrial_advanced', 'fdi_openness']].dropna()
if len(data_corr) > 0:
    corr = data_corr['industrial_advanced'].corr(data_corr['fdi_openness'])
    print(f"industrial_advanced 与 fdi_openness 的相关系数: {corr:.4f}")

    if abs(corr) > 0.7:
        print("  ⚠️  高度相关，可能存在多重共线性")
    elif abs(corr) > 0.5:
        print("  ⚠️  中度相关")

print("\n" + "="*70)
print("异常值检查完成！")
print("="*70)
