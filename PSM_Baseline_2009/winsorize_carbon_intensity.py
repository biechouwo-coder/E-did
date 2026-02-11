import pandas as pd
import numpy as np
from scipy.stats.mstats import winsorize

print("=" * 80)
print("对ln_carbon_intensity进行上下1%缩尾处理")
print("=" * 80)

# 读取数据
print("\n步骤1: 读取数据...")
df = pd.read_excel('matched_data_full_period.xlsx')
print(f"数据集形状: {df.shape}")

# 检查ln_carbon_intensity的情况
print("\n步骤2: 检查ln_carbon_intensity数据情况...")
print(f"ln_carbon_intensity非空记录数: {df['ln_carbon_intensity'].notna().sum()}")
print(f"ln_carbon_intensity空值记录数: {df['ln_carbon_intensity'].isna().sum()}")

# 缩尾前统计
ln_ci_before = df['ln_carbon_intensity'].dropna()
print(f"\n缩尾前ln_carbon_intensity统计:")
print(f"  样本量: {len(ln_ci_before)}")
print(f"  均值: {ln_ci_before.mean():.4f}")
print(f"  中位数: {ln_ci_before.median():.4f}")
print(f"  标准差: {ln_ci_before.std():.4f}")
print(f"  最小值: {ln_ci_before.min():.4f}")
print(f"  1%分位数: {ln_ci_before.quantile(0.01):.4f}")
print(f"  99%分位数: {ln_ci_before.quantile(0.99):.4f}")
print(f"  最大值: {ln_ci_before.max():.4f}")

# 找出将被缩尾的极端值
lower_1pct = ln_ci_before.quantile(0.01)
upper_1pct = ln_ci_before.quantile(0.99)
extreme_low = (ln_ci_before < lower_1pct).sum()
extreme_high = (ln_ci_before > upper_1pct).sum()

print(f"\n极端值统计:")
print(f"  低于1%分位数的记录: {extreme_low}")
print(f"  高于99%分位数的记录: {extreme_high}")
print(f"  总极端值记录: {extreme_low + extreme_high}")

# 进行上下1%缩尾
print("\n步骤3: 进行上下1%缩尾处理...")

# 对有效数据进行缩尾
valid_mask = df['ln_carbon_intensity'].notna()
ln_ci_values = df.loc[valid_mask, 'ln_carbon_intensity'].values

# 使用scipy的winsorize函数
# limits=(0.01, 0.01)表示下限1%和上限1%
winsorized_values = winsorize(ln_ci_values, limits=(0.01, 0.01))

# 将缩尾后的值替换原值
df.loc[valid_mask, 'ln_carbon_intensity'] = winsorized_values

print(f"[OK] 缩尾处理完成")
print(f"  方法: 上下1%缩尾（winsorize）")
print(f"  下限1%分位值: {lower_1pct:.4f}")
print(f"  上限99%分位值: {upper_1pct:.4f}")

# 缩尾后统计
ln_ci_after = df['ln_carbon_intensity'].dropna()
print(f"\n缩尾后ln_carbon_intensity统计:")
print(f"  样本量: {len(ln_ci_after)}")
print(f"  均值: {ln_ci_after.mean():.4f}")
print(f"  中位数: {ln_ci_after.median():.4f}")
print(f"  标准差: {ln_ci_after.std():.4f}")
print(f"  最小值: {ln_ci_after.min():.4f}")
print(f"  最大值: {ln_ci_after.max():.4f}")

# 变化对比
print(f"\n缩尾前后对比:")
print(f"  均值变化: {ln_ci_before.mean():.4f} -> {ln_ci_after.mean():.4f} (差异: {ln_ci_after.mean() - ln_ci_before.mean():.4f})")
print(f"  标准差变化: {ln_ci_before.std():.4f} -> {ln_ci_after.std():.4f} (差异: {ln_ci_after.std() - ln_ci_before.std():.4f})")
print(f"  最小值变化: {ln_ci_before.min():.4f} -> {ln_ci_after.min():.4f}")
print(f"  最大值变化: {ln_ci_before.max():.4f} -> {ln_ci_after.max():.4f}")

# 2007-2019年数据统计
df_2007_2019 = df[
    (df['year'] >= 2007) &
    (df['year'] <= 2019) &
    (df['ln_carbon_intensity'].notna())
]

print(f"\n2007-2019年ln_carbon_intensity统计（缩尾后）:")
print(f"  有效记录数: {len(df_2007_2019)}")

# 处理组 vs 对照组
treat_stats = df_2007_2019[df_2007_2019['Treat'] == 1]['ln_carbon_intensity'].describe()
control_stats = df_2007_2019[df_2007_2019['Treat'] == 0]['ln_carbon_intensity'].describe()

print(f"\n处理组统计:")
print(f"  均值: {treat_stats['mean']:.4f}")
print(f"  中位数: {treat_stats['50%']:.4f}")
print(f"  标准差: {treat_stats['std']:.4f}")
print(f"  最小值: {treat_stats['min']:.4f}")
print(f"  最大值: {treat_stats['max']:.4f}")

print(f"\n对照组统计:")
print(f"  均值: {control_stats['mean']:.4f}")
print(f"  中位数: {control_stats['50%']:.4f}")
print(f"  标准差: {control_stats['std']:.4f}")
print(f"  最小值: {control_stats['min']:.4f}")
print(f"  最大值: {control_stats['max']:.4f}")

# 试点前后对比
pre_period = df_2007_2019[df_2007_2019['year'] < 2009]
post_period = df_2007_2019[df_2007_2019['year'] >= 2009]

pre_treat = pre_period[pre_period['Treat'] == 1]['ln_carbon_intensity'].mean()
pre_control = pre_period[pre_period['Treat'] == 0]['ln_carbon_intensity'].mean()
post_treat = post_period[post_period['Treat'] == 1]['ln_carbon_intensity'].mean()
post_control = post_period[post_period['Treat'] == 0]['ln_carbon_intensity'].mean()

print(f"\n试点前后对比（缩尾后对数值）:")
print(f"  试点前处理组均值: {pre_treat:.4f}")
print(f"  试点前对照组均值: {pre_control:.4f}")
print(f"  试点前差值: {pre_treat - pre_control:.4f}")
print(f"  试点后处理组均值: {post_treat:.4f}")
print(f"  试点后对照组均值: {post_control:.4f}")
print(f"  试点后差值: {post_treat - post_control:.4f}")
print(f"  DID效应: {(post_treat - post_control) - (pre_treat - pre_control):.4f}")

# 保存数据
print("\n步骤4: 保存数据...")
df.to_excel('matched_data_full_period.xlsx', index=False)
print("[OK] 已保存更新后的数据: matched_data_full_period.xlsx")

# 创建仅包含2007-2019年且ln_carbon_intensity非空的数据子集
df_complete = df[
    (df['year'] >= 2007) &
    (df['year'] <= 2019) &
    (df['ln_carbon_intensity'].notna())
].copy()

df_complete.to_excel('matched_data_2007-2019_complete.xlsx', index=False)
print(f"[OK] 已保存2007-2019年完整数据: matched_data_2007-2019_complete.xlsx")
print(f"    记录数: {len(df_complete)}")

# 生成统计报告
report = []
report.append("=" * 80)
report.append("ln_carbon_intensity缩尾处理报告")
report.append("=" * 80)
report.append("")
report.append("一、缩尾方法")
report.append("-" * 80)
report.append("方法: 上下1%缩尾（winsorize）")
report.append("说明: 将低于1%分位数的值替换为1%分位值")
report.append("      将高于99%分位数的值替换为99%分位值")
report.append("")
report.append("二、缩尾阈值")
report.append("-" * 80)
report.append(f"下限1%分位值: {lower_1pct:.4f}")
report.append(f"上限99%分位值: {upper_1pct:.4f}")
report.append(f"处理极端值数量: {extreme_low + extreme_high}")
report.append("  (低于下限: %d, 高于上限: %d)" % (extreme_low, extreme_high))
report.append("")
report.append("三、缩尾前后对比")
report.append("-" * 80)
report.append("均值: %.4f -> %.4f (变化: %.4f)" % (ln_ci_before.mean(), ln_ci_after.mean(), ln_ci_after.mean() - ln_ci_before.mean()))
report.append("标准差: %.4f -> %.4f (变化: %.4f)" % (ln_ci_before.std(), ln_ci_after.std(), ln_ci_after.std() - ln_ci_before.std()))
report.append("最小值: %.4f -> %.4f" % (ln_ci_before.min(), ln_ci_after.min()))
report.append("最大值: %.4f -> %.4f" % (ln_ci_before.max(), ln_ci_after.max()))
report.append("")
report.append("四、2007-2019年统计（缩尾后）")
report.append("-" * 80)
report.append("处理组均值: %.4f" % treat_stats['mean'])
report.append("对照组均值: %.4f" % control_stats['mean'])
report.append("均值差异: %.4f" % (treat_stats['mean'] - control_stats['mean']))
report.append("")
report.append("五、DID效应（缩尾后对数值）")
report.append("-" * 80)
report.append("试点前差值: %.4f" % (pre_treat - pre_control))
report.append("试点后差值: %.4f" % (post_treat - post_control))
report.append("DID效应: %.4f" % ((post_treat - post_control) - (pre_treat - pre_control)))
report.append("")
report.append("=" * 80)

report_text = "\n".join(report)
print(report_text)

# 保存报告
with open('winsorize_report.txt', 'w', encoding='utf-8') as f:
    f.write(report_text)

print(f"[OK] 已保存统计报告: winsorize_report.txt")

print("\n" + "=" * 80)
print("缩尾处理完成！")
print("=" * 80)
