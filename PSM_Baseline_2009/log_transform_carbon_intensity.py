import pandas as pd
import numpy as np

print("=" * 80)
print("对碳排放强度变量进行对数变换")
print("=" * 80)

# 读取数据
print("\n步骤1: 读取数据...")
df = pd.read_excel('matched_data_full_period.xlsx')
print(f"数据集形状: {df.shape}")

# 检查carbon_intensity的情况
print("\n步骤2: 检查carbon_intensity数据情况...")
print(f"carbon_intensity非空记录数: {df['carbon_intensity'].notna().sum()}")
print(f"carbon_intensity空值记录数: {df['carbon_intensity'].isna().sum()}")

# 检查是否有非正值
non_positive = (df['carbon_intensity'] <= 0).sum()
print(f"非正值记录数（≤0）: {non_positive}")

# 对数变换
print("\n步骤3: 进行对数变换...")
print("公式: ln_carbon_intensity = ln(carbon_intensity)")

# 只对正值取对数
df['ln_carbon_intensity'] = np.where(
    df['carbon_intensity'] > 0,
    np.log(df['carbon_intensity']),
    np.nan
)

# 统计结果
valid_ci = df['carbon_intensity'].notna().sum()
valid_ln_ci = df['ln_carbon_intensity'].notna().sum()

print(f"\n对数变换结果:")
print(f"  carbon_intensity有效记录: {valid_ci}")
print(f"  ln_carbon_intensity有效记录: {valid_ln_ci}")
print(f"  因非正值无法取对数的记录: {valid_ci - valid_ln_ci}")

# 描述性统计
print(f"\ncarbon_intensity原始值统计:")
print(df['carbon_intensity'].describe())

print(f"\nln_carbon_int对数统计:")
print(df['ln_carbon_intensity'].describe())

# 2007-2019年数据统计
df_2007_2019 = df[
    (df['year'] >= 2007) &
    (df['year'] <= 2019) &
    (df['ln_carbon_intensity'].notna())
]

print(f"\n2007-2019年ln_carbon_intensity统计:")
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

print(f"\n试点前后对比（对数值）:")
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
report.append("碳排放强度对数变换报告")
report.append("=" * 80)
report.append("")
report.append("一、变换公式")
report.append("-" * 80)
report.append("ln_carbon_intensity = ln(carbon_intensity)")
report.append("")
report.append("二、数据统计")
report.append("-" * 80)
report.append(f"carbon_intensity有效记录: {valid_ci}")
report.append(f"ln_carbon_intensity有效记录: {valid_ln_ci}")
report.append(f"因非正值无法取对数的记录: {valid_ci - valid_ln_ci}")
report.append("")
report.append("三、2007-2019年统计（对数值）")
report.append("-" * 80)
report.append(f"处理组均值: {treat_stats['mean']:.4f}")
report.append(f"对照组均值: {control_stats['mean']:.4f}")
report.append(f"均值差异: {treat_stats['mean'] - control_stats['mean']:.4f}")
report.append("")
report.append("四、DID效应（对数值）")
report.append("-" * 80)
report.append(f"试点前差值: {pre_treat - pre_control:.4f}")
report.append(f"试点后差值: {post_treat - post_control:.4f}")
report.append(f"DID效应: {(post_treat - post_control) - (pre_treat - pre_control):.4f}")
report.append("")
report.append("=" * 80)

report_text = "\n".join(report)
print(report_text)

# 保存报告
with open('ln_carbon_intensity_report.txt', 'w', encoding='utf-8') as f:
    f.write(report_text)

print(f"[OK] 已保存统计报告: ln_carbon_intensity_report.txt")

print("\n" + "=" * 80)
print("对数变换完成！")
print("=" * 80)
