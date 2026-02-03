# -*- coding: utf-8 -*-
"""
对 industrial_advanced 和 fdi_openness 进行1%缩尾处理
"""
import pandas as pd
import numpy as np
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

print("="*70)
print("变量1%缩尾处理：industrial_advanced 和 fdi_openness")
print("="*70)

# 读取数据
df = pd.read_excel('总数据集_2007-2023_完整版_DID_with_treat.xlsx')

vars_to_winsorize = ['industrial_advanced', 'fdi_openness']

print("\n步骤1: 缩尾前统计")
print("-"*70)

for var in vars_to_winsorize:
    data = df[var].dropna()
    print(f"\n【{var}】")
    print(f"  有效观测: {len(data)}")
    print(f"  均值: {data.mean():.4f}")
    print(f"  标准差: {data.std():.4f}")
    print(f"  最小值: {data.min():.4f}")
    print(f"  1%分位数: {data.quantile(0.01):.4f}")
    print(f"  中位数: {data.median():.4f}")
    print(f"  99%分位数: {data.quantile(0.99):.4f}")
    print(f"  最大值: {data.max():.4f}")
    print(f"  偏度: {data.skew():.4f}")
    print(f"  峰度: {data.kurtosis():.4f}")

print("\n\n步骤2: 执行1%缩尾处理")
print("-"*70)

# 创建新列保存缩尾后的值
for var in vars_to_winsorize:
    # 创建缩尾后的列名
    winsorized_col = f'{var}_winsorized'
    df[winsorized_col] = df[var]

    # 计算上下界
    p01 = df[var].quantile(0.01)
    p99 = df[var].quantile(0.99)

    # 计算将被缩尾的观测数
    lower_outliers = (df[var] < p01).sum()
    upper_outliers = (df[var] > p99).sum()
    total_outliers = lower_outliers + upper_outliers

    print(f"\n【{var}】")
    print(f"  下界 (1%分位数): {p01:.4f}")
    print(f"  上界 (99%分位数): {p99:.4f}")
    print(f"  低于下界的观测: {lower_outliers}")
    print(f"  高于上界的观测: {upper_outliers}")
    print(f"  总异常值: {total_outliers} ({total_outliers/df[var].notnull().sum()*100:.1f}%)")

    # 执行缩尾
    df.loc[df[var] < p01, winsorized_col] = p01
    df.loc[df[var] > p99, winsorized_col] = p99

    print(f"  ✓ 已创建缩尾变量: {winsorized_col}")

print("\n\n步骤3: 缩尾后统计")
print("-"*70)

for var in vars_to_winsorize:
    winsorized_col = f'{var}_winsorized'
    data_orig = df[var].dropna()
    data_wins = df[winsorized_col].dropna()

    print(f"\n【{var}】")

    print(f"  缩尾前:")
    print(f"    均值: {data_orig.mean():.4f} → 标准差: {data_orig.std():.4f}")
    print(f"    最小值: {data_orig.min():.4f} → 最大值: {data_orig.max():.4f}")
    print(f"    偏度: {data_orig.skew():.4f} → 峰度: {data_orig.kurtosis():.4f}")

    print(f"  缩尾后:")
    print(f"    均值: {data_wins.mean():.4f} → 标准差: {data_wins.std():.4f}")
    print(f"    最小值: {data_wins.min():.4f} → 最大值: {data_wins.max():.4f}")
    print(f"    偏度: {data_wins.skew():.4f} → 峰度: {data_wins.kurtosis():.4f}")

    # 计算改善情况
    skew_improve = (abs(data_orig.skew()) - abs(data_wins.skew())) / abs(data_orig.skew()) * 100
    kurt_improve = (data_orig.kurtosis() - data_wins.kurtosis()) / data_orig.kurtosis() * 100

    print(f"  改善:")
    print(f"    |偏度|降低: {skew_improve:.1f}%")
    print(f"    峰度降低: {kurt_improve:.1f}%")

print("\n\n步骤4: 缩尾前后对比（被缩尾的样本）")
print("-"*70)

for var in vars_to_winsorize:
    winsorized_col = f'{var}_winsorized'

    # 找出被缩尾的观测
    p01 = df[var].quantile(0.01)
    p99 = df[var].quantile(0.99)

    lower_changed = df[(df[var] < p01) & df[var].notnull()]
    upper_changed = df[(df[var] > p99) & df[var].notnull()]

    print(f"\n【{var}】")

    if len(lower_changed) > 0:
        print(f"  下尾被缩尾的样本（前10个）:")
        sample = lower_changed[['year', 'city_name', var, winsorized_col]].head(10)
        for _, row in sample.iterrows():
            print(f"    {int(row['year'])}年 {row['city_name']}: {row[var]:.4f} → {row[winsorized_col]:.4f}")

    if len(upper_changed) > 0:
        print(f"  上尾被缩尾的样本（前10个）:")
        sample = upper_changed[['year', 'city_name', var, winsorized_col]].head(10)
        for _, row in sample.iterrows():
            print(f"    {int(row['year'])}年 {row['city_name']}: {row[var]:.4f} → {row[winsorized_col]:.4f}")

print("\n\n步骤5: 保存缩尾后的数据")
print("-"*70)

# 调整列顺序，将缩尾变量放在原变量后面
cols = df.columns.tolist()
for var in vars_to_winsorize:
    if var in cols:
        idx = cols.index(var)
        winsorized_col = f'{var}_winsorized'
        if winsorized_col in cols:
            cols.remove(winsorized_col)
            cols.insert(idx + 1, winsorized_col)

df = df[cols]

# 保存数据
output_file = '总数据集_2007-2023_完整版_DID_with_treat_winsorized.xlsx'
df.to_excel(output_file, index=False)

print(f"✓ 缩尾后数据已保存至: {output_file}")
print(f"  总列数: {len(df.columns)}")
print(f"  新增变量: {vars_to_winsorize[0]}_winsorized, {vars_to_winsorize[1]}_winsorized")

print("\n\n步骤6: 生成缩尾报告")
print("-"*70)

summary = f"""
{'='*70}
变量1%缩尾处理报告
{'='*70}

一、缩尾说明
-----------
缩尾方法：双侧1%缩尾（Winsorize）
• 将低于1%分位数的值替换为1%分位数
• 将高于99%分位数的值替换为99%分位数

处理变量：
• industrial_advanced（产业高级化）
• fdi_openness（外商直接投资开放度）

二、缩尾效果汇总
-----------

"""

for var in vars_to_winsorize:
    winsorized_col = f'{var}_winsorized'
    data_orig = df[var].dropna()
    data_wins = df[winsorized_col].dropna()

    p01 = df[var].quantile(0.01)
    p99 = df[var].quantile(0.99)
    lower_outliers = (df[var] < p01).sum()
    upper_outliers = (df[var] > p99).sum()

    summary += f"""
【{var}】
缩尾边界:
  • 下界 (1%): {p01:.4f}
  • 上界 (99%): {p99:.4f}

异常值数量:
  • 下尾异常值: {lower_outliers}
  • 上尾异常值: {upper_outliers}
  • 总异常值: {lower_outliers + upper_outliers}

缩尾前后对比:
                        缩尾前        缩尾后        变化
  • 均值:              {data_orig.mean():.4f}      {data_wins.mean():.4f}      {(data_wins.mean()-data_orig.mean())/data_orig.mean()*100:+.1f}%
  • 标准差:            {data_orig.std():.4f}      {data_wins.std():.4f}      {(data_wins.std()-data_orig.std())/data_orig.std()*100:+.1f}%
  • 偏度:              {data_orig.skew():.4f}      {data_wins.skew():.4f}
  • 峰度:              {data_orig.kurtosis():.4f}      {data_wins.kurtosis():.4f}

分布改善:
  • |偏度|降低: {(abs(data_orig.skew()) - abs(data_wins.skew())) / abs(data_orig.skew()) * 100:.1f}%
  • 峰度降低: {(data_orig.kurtosis() - data_wins.kurtosis()) / data_orig.kurtosis() * 100:.1f}%

"""

summary += f"""
三、使用建议
-----------
缩尾后的变量可以：
1. 作为控制变量使用，减少极端值的影响
2. 改善回归结果的稳健性
3. 在主回归中使用缩尾变量，在稳健性检验中使用原始变量对比

变量命名:
• 原始变量: {vars_to_winsorize[0]}, {vars_to_winsorize[1]}
• 缩尾变量: {vars_to_winsorize[0]}_winsorized, {vars_to_winsorize[1]}_winsorized

建议：
• 主回归使用缩尾变量
• 稳健性检验对比原始变量和缩尾变量的回归结果
• 如果结果差异较大，说明极端值对结果有较大影响

{'='*70}
"""

print(summary)

# 保存报告
report_file = 'scripts/winsorize_report.txt'
with open(report_file, 'w', encoding='utf-8') as f:
    f.write(summary)

print(f"\n✓ 缩尾报告已保存至: {report_file}")

print("\n" + "="*70)
print("缩尾处理完成！")
print("="*70)
