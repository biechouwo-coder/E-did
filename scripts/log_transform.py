# -*- coding: utf-8 -*-
"""
对 industrial_advanced 和 fdi_openness 取对数
"""
import pandas as pd
import numpy as np
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

print("="*70)
print("变量对数转换：industrial_advanced 和 fdi_openness")
print("="*70)

# 读取数据
df = pd.read_excel('总数据集_2007-2023_完整版_DID_with_treat.xlsx')

vars_to_log = ['industrial_advanced', 'fdi_openness']

print("\n步骤1: 检查数据范围")
print("-"*70)

for var in vars_to_log:
    data = df[var].dropna()
    print(f"\n【{var}】")
    print(f"  有效观测: {len(data)}")
    print(f"  最小值: {data.min():.6f}")
    print(f"  最大值: {data.max():.6f}")

    # 检查是否有≤0的值
    non_positive = (data <= 0).sum()
    if non_positive > 0:
        print(f"  ⚠️  ≤0的值: {non_positive} ({non_positive/len(data)*100:.1f}%)")
        print(f"  最小非正值: {data[data <= 0].min():.6f}")
    else:
        print(f"  ✓ 所有值都>0，可以直接取对数")

    print(f"  偏度: {data.skew():.4f}")
    print(f"  峰度: {data.kurtosis():.4f}")

print("\n\n步骤2: 执行对数转换")
print("-"*70)

for var in vars_to_log:
    # 创建对数变量名
    log_col = f'ln_{var}'

    print(f"\n【{var}】")

    # 检查是否有≤0的值
    data = df[var].dropna()
    non_positive = (data <= 0).sum()

    if non_positive > 0:
        print(f"  发现{non_positive}个≤0的值")
        print(f"  采用 ln(x + min_positive) 的方式处理")

        # 找到最小正数
        min_positive = data[data > 0].min()
        print(f"  最小正数: {min_positive:.6f}")
        print(f"  使用转换: ln(x + {min_positive:.6f})")

        # 取对数：ln(x + min_positive)
        df[log_col] = np.log(df[var] + min_positive)
    else:
        print(f"  所有值都>0，直接取对数")
        # 直接取对数
        df[log_col] = np.log(df[var])

    print(f"  ✓ 已创建对数变量: {log_col}")

print("\n\n步骤3: 对数转换后统计")
print("-"*70)

for var in vars_to_log:
    log_col = f'ln_{var}'
    data_orig = df[var].dropna()
    data_log = df[log_col].dropna()

    print(f"\n【{var}】")

    print(f"  原始变量:")
    print(f"    均值: {data_orig.mean():.4f}")
    print(f"    标准差: {data_orig.std():.4f}")
    print(f"    偏度: {data_orig.skew():.4f}")
    print(f"    峰度: {data_orig.kurtosis():.4f}")

    print(f"  对数变量:")
    print(f"    均值: {data_log.mean():.4f}")
    print(f"    标准差: {data_log.std():.4f}")
    print(f"    最小值: {data_log.min():.4f}")
    print(f"    最大值: {data_log.max():.4f}")
    print(f"    偏度: {data_log.skew():.4f}")
    print(f"    峰度: {data_log.kurtosis():.4f}")

    # 计算改善
    skew_improve = (abs(data_orig.skew()) - abs(data_log.skew())) / abs(data_orig.skew()) * 100
    kurt_improve = (data_orig.kurtosis() - data_log.kurtosis()) / data_orig.kurtosis() * 100

    print(f"  改善:")
    print(f"    |偏度|降低: {skew_improve:.1f}%")
    print(f"    峰度降低: {kurt_improve:.1f}%")

print("\n\n步骤4: 对数转换前后对比（按年份）")
print("-"*70)

for var in vars_to_log:
    log_col = f'ln_{var}'

    print(f"\n【{var}】")

    # 按年份统计原始变量
    yearly_orig = df.groupby('year')[var].agg(['mean', 'std'])
    print(f"\n  原始变量按年份均值:")
    print(f"    2007年: {yearly_orig.loc[2007, 'mean']:.4f}")
    print(f"    2010年: {yearly_orig.loc[2010, 'mean']:.4f}")
    print(f"    2020年: {yearly_orig.loc[2020, 'mean']:.4f}")
    print(f"    2023年: {yearly_orig.loc[2023, 'mean']:.4f}")

    # 按年份统计对数变量
    yearly_log = df.groupby('year')[log_col].agg(['mean', 'std'])
    print(f"\n  对数变量按年份均值:")
    print(f"    2007年: {yearly_log.loc[2007, 'mean']:.4f}")
    print(f"    2010年: {yearly_log.loc[2010, 'mean']:.4f}")
    print(f"    2020年: {yearly_log.loc[2020, 'mean']:.4f}")
    print(f"    2023年: {yearly_log.loc[2023, 'mean']:.4f}")

print("\n\n步骤5: 相关性检查")
print("-"*70)

# 检查对数转换后的相关性
log_vars = [f'ln_{v}' for v in vars_to_log]
data_corr = df[log_vars].dropna()

if len(data_corr) > 1:
    corr = data_corr[log_vars[0]].corr(data_corr[log_vars[1]])
    print(f"ln_industrial_advanced 与 ln_fdi_openness 的相关系数: {corr:.4f}")

    if abs(corr) > 0.7:
        print("  ⚠️  高度相关，可能存在多重共线性")
    elif abs(corr) > 0.5:
        print("  ⚠️  中度相关")
    else:
        print("  ✓ 相关性较低")

# 检查原始变量和对数变量的相关性
for var in vars_to_log:
    log_col = f'ln_{var}'
    data_corr = df[[var, log_col]].dropna()
    corr = data_corr[var].corr(data_corr[log_col])
    print(f"\n{var} 与 {log_col} 的相关系数: {corr:.4f}")
    print(f"  说明对数转换保留了原始变量的秩次信息")

print("\n\n步骤6: 极端值影响检查")
print("-"*70)

for var in vars_to_log:
    log_col = f'ln_{var}'

    print(f"\n【{var}】")

    # 找出原始变量中top 10的城市
    data = df[['year', 'city_name', var, log_col]].dropna(subset=[var])

    print(f"  原始变量最高的10个城市（对数转换后）:")
    top10 = data.nlargest(10, var)
    for _, row in top10.iterrows():
        orig_val = row[var]
        log_val = row[log_col]
        print(f"    {int(row['year'])}年 {row['city_name']}: {orig_val:.4f} → {log_val:.4f}")

print("\n\n步骤7: 保存对数转换后的数据")
print("-"*70)

# 调整列顺序，将对数变量放在原变量后面
cols = df.columns.tolist()
for var in vars_to_log:
    if var in cols:
        idx = cols.index(var)
        log_col = f'ln_{var}'
        if log_col in cols:
            cols.remove(log_col)
            cols.insert(idx + 1, log_col)

df = df[cols]

# 保存数据
output_file = '总数据集_2007-2023_完整版_DID_with_treat_log.xlsx'
df.to_excel(output_file, index=False)

print(f"✓ 对数转换后数据已保存至: {output_file}")
print(f"  总列数: {len(df.columns)}")
print(f"  新增变量: ln_industrial_advanced, ln_fdi_openness")

print("\n\n步骤8: 生成转换报告")
print("-"*70)

summary = f"""
{'='*70}
变量对数转换报告
{'='*70}

一、转换说明
-----------
转换方法：自然对数转换 ln(x)
• 如果变量存在≤0的值，使用 ln(x + min_positive) 方式处理
• 对数转换可以缓解右偏分布，使数据更接近正态分布

处理变量：
• industrial_advanced（产业高级化）
• fdi_openness（外商直接投资开放度）

二、转换效果汇总
-----------

"""

for var in vars_to_log:
    log_col = f'ln_{var}'
    data_orig = df[var].dropna()
    data_log = df[log_col].dropna()

    # 检查是否需要加常数
    non_positive = (data_orig <= 0).sum()

    summary += f"""
【{var}】
转换方法:
"""

    if non_positive > 0:
        min_positive = data_orig[data_orig > 0].min()
        summary += f"  • 使用转换: ln(x + {min_positive:.6f})\n"
        summary += f"  • 原因: 存在{non_positive}个≤0的值\n"
    else:
        summary += f"  • 使用转换: ln(x)\n"
        summary += f"  • 原因: 所有值都>0\n"

    summary += f"""
转换前后对比:
                        原始变量      对数变量      改善
  • 均值:              {data_orig.mean():.4f}      {data_log.mean():.4f}
  • 标准差:            {data_orig.std():.4f}      {data_log.std():.4f}
  • 偏度:              {data_orig.skew():.4f}      {data_log.skew():.4f}      ({(abs(data_orig.skew()) - abs(data_log.skew())) / abs(data_orig.skew()) * 100:.1f}%)
  • 峰度:              {data_orig.kurtosis():.4f}      {data_log.kurtosis():.4f}      ({(data_orig.kurtosis() - data_log.kurtosis()) / data_orig.kurtosis() * 100:.1f}%)

分布改善:
  • |偏度|降低: {(abs(data_orig.skew()) - abs(data_log.skew())) / abs(data_orig.skew()) * 100:.1f}%
  • 峰度降低: {(data_orig.kurtosis() - data_log.kurtosis()) / data_orig.kurtosis() * 100:.1f}%

"""

summary += f"""
三、使用建议
-----------
对数转换后的变量特点：
1. 缓解右偏分布，使数据更接近正态分布
2. 减少极端值的影响
3. 适用于回归分析，特别是当变量存在乘法关系时

变量命名:
• 原始变量: industrial_advanced, fdi_openness
• 对数变量: ln_industrial_advanced, ln_fdi_openness

建议：
• 主回归可使用对数变量
• 解释系数时注意：对数变量的系数表示弹性（百分比变化）
• 稳健性检验对比原始变量和对数变量的回归结果

优点：
• 分布更接近正态假设
• 减少异方差性
• 缓解多重共线性

注意事项：
• 对数变量的系数解释为"百分比变化"
• 零值或负值需要特殊处理（如加常数）
• 在论文中说明对数转换的理由和方法

{'='*70}
"""

print(summary)

# 保存报告
report_file = 'scripts/log_transform_report.txt'
with open(report_file, 'w', encoding='utf-8') as f:
    f.write(summary)

print(f"\n✓ 转换报告已保存至: {report_file}")

print("\n" + "="*70)
print("对数转换完成！")
print("="*70)
