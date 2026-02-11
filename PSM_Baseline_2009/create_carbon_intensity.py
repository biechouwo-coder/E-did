import pandas as pd
import numpy as np

print("=" * 80)
print("构建碳排放强度变量")
print("=" * 80)

# 读取匹配后的数据
print("\n步骤1: 读取匹配后的全期数据...")
df = pd.read_excel('matched_data_full_period.xlsx')
print(f"数据集形状: {df.shape}")
print(f"年份范围: {df['year'].min()} - {df['year'].max()}")
print(f"城市数: {df['city_name'].nunique()}")

# 检查emission和real_gdp的情况
print("\n步骤2: 检查emission和real_gdp数据情况...")
print(f"emission非空记录数: {df['emission'].notna().sum()}")
print(f"emission空值记录数: {df['emission'].isna().sum()}")
print(f"real_gdp非空记录数: {df['real_gdp'].notna().sum()}")
print(f"real_gdp空值记录数: {df['real_gdp'].isna().sum()}")

# 计算碳排放强度 = emission / real_gdp
print("\n步骤3: 计算碳排放强度...")
print("公式: 碳排放强度 = emission / real_gdp")

# 碳排放强度单位：万吨/亿元
df['carbon_intensity'] = df['emission'] / df['real_gdp']

# 统计信息
print(f"\n碳排放强度统计:")
print(f"  有效计算记录数: {df['carbon_intensity'].notna().sum()}")
print(f"  缺失记录数: {df['carbon_intensity'].isna().sum()}")

# 按年份统计有效记录
print(f"\n各年份碳排放强度有效记录数:")
yearly_stats = df.groupby('year').agg(
    total_records=('carbon_intensity', 'size'),
    valid_records=('carbon_intensity', lambda x: x.notna().sum()),
    mean_ci=('carbon_intensity', 'mean'),
    median_ci=('carbon_intensity', 'median'),
    std_ci=('carbon_intensity', 'std')
).reset_index()

print(yearly_stats.to_string(index=False))

# 处理组 vs 对照组对比
print(f"\n处理组 vs 对照组碳排放强度对比（2007-2019）:")
treat_stats = df[(df['Treat'] == 1) & (df['year'] >= 2007) & (df['year'] <= 2019)]['carbon_intensity'].describe()
control_stats = df[(df['Treat'] == 0) & (df['year'] >= 2007) & (df['year'] <= 2019)]['carbon_intensity'].describe()

comparison = pd.DataFrame({
    '处理组': treat_stats,
    '对照组': control_stats
})
print(comparison)

# 总体统计
print(f"\n总体统计（2007-2019年，emission非空）:")
overall_stats = df[
    (df['year'] >= 2007) &
    (df['year'] <= 2019) &
    (df['emission'].notna())
]['carbon_intensity'].describe()

print(f"  平均值: {overall_stats['mean']:.4f} 万吨/亿元")
print(f"  中位数: {overall_stats['50%']:.4f} 万吨/亿元")
print(f"  标准差: {overall_stats['std']:.4f} 万吨/亿元")
print(f"  最小值: {overall_stats['min']:.4f} 万吨/亿元")
print(f"  最大值: {overall_stats['max']:.4f} 万吨/亿元")

# 检查是否有异常值（负值或过大）
print(f"\n异常值检查:")
negative_count = (df['carbon_intensity'] < 0).sum()
print(f"  负值数量: {negative_count}")

# 定义异常值：超过平均值+3倍标准差
if overall_stats['std'] > 0:
    upper_bound = overall_stats['mean'] + 3 * overall_stats['std']
    outlier_count = (df['carbon_intensity'] > upper_bound).sum()
    print(f"  异常高值（>{upper_bound:.4f}）数量: {outlier_count}")

# 保存数据
print("\n步骤4: 保存数据...")
df.to_excel('matched_data_full_period.xlsx', index=False)
print(f"[OK] 已保存更新后的数据: matched_data_full_period.xlsx")

# 创建仅包含2007-2019年且carbon_intensity非空的数据子集
df_complete = df[
    (df['year'] >= 2007) &
    (df['year'] <= 2019) &
    (df['carbon_intensity'].notna())
].copy()

df_complete.to_excel('matched_data_2007-2019_complete.xlsx', index=False)
print(f"[OK] 已保存2007-2019年完整数据: matched_data_2007-2019_complete.xlsx")
print(f"    记录数: {len(df_complete)}")

# 生成统计报告
print("\n步骤5: 生成统计报告...")

report = []
report.append("=" * 80)
report.append("碳排放强度统计报告")
report.append("=" * 80)
report.append("")
report.append("一、变量定义")
report.append("-" * 80)
report.append("碳排放强度 = emission / real_gdp")
report.append("单位: 万吨/亿元")
report.append("")
report.append("二、数据覆盖")
report.append("-" * 80)
report.append(f"总记录数: {len(df)}")
report.append(f"有效计算记录数: {df['carbon_intensity'].notna().sum()}")
report.append(f"缺失记录数: {df['carbon_intensity'].isna().sum()}")
report.append("")
report.append("三、总体统计（2007-2019年）")
report.append("-" * 80)
report.append(f"平均值: {overall_stats['mean']:.4f} 万吨/亿元")
report.append(f"中位数: {overall_stats['50%']:.4f} 万吨/亿元")
report.append(f"标准差: {overall_stats['std']:.4f} 万吨/亿元")
report.append(f"最小值: {overall_stats['min']:.4f} 万吨/亿元")
report.append(f"最大值: {overall_stats['max']:.4f} 万吨/亿元")
report.append("")
report.append("四、处理组 vs 对照组")
report.append("-" * 80)
report.append(f"处理组平均值: {treat_stats['mean']:.4f} 万吨/亿元")
report.append(f"对照组平均值: {control_stats['mean']:.4f} 万吨/亿元")
report.append(f"差异: {treat_stats['mean'] - control_stats['mean']:.4f} 万吨/亿元")
report.append("")
report.append("五、输出文件")
report.append("-" * 80)
report.append("1. matched_data_full_period.xlsx - 更新后的全期数据（含carbon_intensity列）")
report.append("2. matched_data_2007-2019_complete.xlsx - 2007-2019年完整数据")
report.append("")
report.append("=" * 80)

report_text = "\n".join(report)
print(report_text)

# 保存报告
with open('carbon_intensity_report.txt', 'w', encoding='utf-8') as f:
    f.write(report_text)

print(f"[OK] 已保存统计报告: carbon_intensity_report.txt")

print("\n" + "=" * 80)
print("碳排放强度变量构建完成！")
print("=" * 80)
