# -*- coding: utf-8 -*-
"""
检查缺失值模式并实现线性插值
"""
import pandas as pd
import numpy as np
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

print("="*70)
print("缺失值分析与插值处理")
print("="*70)

# 读取数据
df = pd.read_excel('总数据集_2007-2023_完整版_DID_with_treat.xlsx')

control_vars = ['gdp_per_capita', 'pop_density', 'industrial_upgrading', 'ln_fdi']

print("\n步骤1: 检查缺失值模式")
print("-"*70)

# 按城市检查缺失值
print("\n各变量的缺失情况:")
for var in control_vars:
    total_missing = df[var].isnull().sum()
    pct_missing = total_missing / len(df) * 100
    print(f"{var}: {total_missing} / {len(df)} ({pct_missing:.1f}%)")

# 检查2009年的缺失情况
df_2009 = df[df['year'] == 2009]
print("\n2009年（基期）缺失情况:")
for var in control_vars:
    missing_by_treat = df_2009.groupby('treat')[var].apply(lambda x: x.isnull().sum())
    total_by_treat = df_2009.groupby('treat')[var].count()
    print(f"\n{var}:")
    for treat_val in [0, 1]:
        treat_name = "对照组" if treat_val == 0 else "处理组"
        if treat_val in missing_by_treat.index:
            missing = missing_by_treat[treat_val]
            total = total_by_treat.get(treat_val, 0)
            print(f"  {treat_name}(treat={treat_val}): {missing} / {total} ({missing/total*100:.1f}%)")

# 检查缺失值的连续性（是否可以插值）
print("\n\n步骤2: 检查缺失值的连续性")
print("-"*70)

# 按城市统计连续缺失年数
def get_max_consecutive_missing(series):
    """获取最长连续缺失的年数"""
    max_missing = 0
    current_missing = 0
    for val in series:
        if pd.isnull(val):
            current_missing += 1
            max_missing = max(max_missing, current_missing)
        else:
            current_missing = 0
    return max_missing

# 对每个控制变量，统计各城市的最长连续缺失年数
for var in control_vars:
    print(f"\n{var}:")
    city_missing = df.groupby('city_name')[var].apply(
        lambda x: get_max_consecutive_missing(x)
    )
    print(f"  平均最长连续缺失年数: {city_missing.mean():.1f}年")
    print(f"  最大最长连续缺失年数: {city_missing.max():.0f}年")
    print(f"  连续缺失≤2年的城市数: {(city_missing <= 2).sum()} / {len(city_missing)}")

print("\n\n步骤3: 执行线性插值")
print("-"*70)

# 创建数据的副本
df_imputed = df.copy()

# 对每个城市分别进行插值
cities = df['city_name'].unique()

print("\n插值方法:")
print("  • 对每个城市的时间序列进行线性插值")
print("  • 前向填充首年缺失值（如果有）")
print("  • 后向填充末年缺失值（如果有）")

for var in control_vars:
    # 对每个城市进行插值
    df_city_imputed = pd.DataFrame()

    for city in cities:
        city_data = df[df['city_name'] == city].copy()
        city_data = city_data.sort_values('year')

        # 记录插值前的缺失数
        missing_before = city_data[var].isnull().sum()

        # 线性插值
        city_data[var] = city_data[var].interpolate(method='linear', limit_direction='both')

        # 如果首年或末年仍然缺失，用前向或后向填充
        city_data[var] = city_data[var].fillna(method='ffill').fillna(method='bfill')

        # 记录插值后的缺失数
        missing_after = city_data[var].isnull().sum()

        df_city_imputed = pd.concat([df_city_imputed, city_data])

    # 替换原数据
    df_imputed[var] = df_city_imputed.sort_index()[var]

    total_imputed = df[var].isnull().sum() - df_imputed[var].isnull().sum()
    print(f"\n{var}:")
    print(f"  插值前缺失: {df[var].isnull().sum()}")
    print(f"  插值后缺失: {df_imputed[var].isnull().sum()}")
    print(f"  成功插值: {total_imputed} 个观测")

# 检查2009年的插值效果
print("\n\n步骤4: 插值效果验证（2009年基期）")
print("-"*70)

df_2009_imputed = df_imputed[df_imputed['year'] == 2009]

print("\n插值后2009年样本量:")
treat_count = (df_2009_imputed['treat'] == 1).sum()
control_count = (df_2009_imputed['treat'] == 0).sum()
print(f"  处理组(treat=1): {treat_count} 个城市")
print(f"  对照组(treat=0): {control_count} 个城市")
print(f"  总计: {len(df_2009_imputed)} 个城市")

print("\n各变量插值后缺失值:")
for var in control_vars:
    missing = df_2009_imputed[var].isnull().sum()
    print(f"  {var}: {missing} 个缺失")

# 统计描述对比
print("\n\n步骤5: 插值前后统计对比（2009年）")
print("-"*70)

for var in control_vars:
    print(f"\n{var}:")

    # 处理组
    treat_before = df_2009[df_2009['treat']==1][var].describe()
    treat_after = df_2009_imputed[df_2009_imputed['treat']==1][var].describe()

    print(f"  处理组(treat=1):")
    print(f"    插值前: 均值={treat_before['mean']:.2f}, 标准差={treat_before['std']:.2f}, 样本={int(treat_before['count'])}")
    print(f"    插值后: 均值={treat_after['mean']:.2f}, 标准差={treat_after['std']:.2f}, 样本={int(treat_after['count'])}")

    # 对照组
    control_before = df_2009[df_2009['treat']==0][var].describe()
    control_after = df_2009_imputed[df_2009_imputed['treat']==0][var].describe()

    print(f"  对照组(treat=0):")
    print(f"    插值前: 均值={control_before['mean']:.2f}, 标准差={control_before['std']:.2f}, 样本={int(control_before['count'])}")
    print(f"    插值后: 均值={control_after['mean']:.2f}, 标准差={control_after['std']:.2f}, 样本={int(control_after['count'])}")

# 保存插值后的数据
print("\n\n步骤6: 保存插值后的数据")
print("-"*70)

output_file = '总数据集_2007-2023_完整版_DID_with_treat_imputed.xlsx'
df_imputed.to_excel(output_file, index=False)
print(f"✓ 插值后数据已保存至: {output_file}")

print("\n" + "="*70)
print("插值处理完成！")
print("="*70)
print(f"\n主要改进:")
print(f"  • 2009年基期样本量从 203 → {len(df_2009_imputed)}")
print(f"  • 处理组样本从 98 → {treat_count}")
print(f"  • 对照组样本从 105 → {control_count}")
print(f"  • 样本保留率提升: {len(df_2009_imputed)/203*100:.1f}%")
