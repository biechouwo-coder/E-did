import pandas as pd

# 读取结果
balance_df = pd.read_excel('balance_check_results.xlsx')
pairs_df = pd.read_excel('matched_pairs_details.xlsx')
baseline_df = pd.read_excel('baseline_2009_with_scores.xlsx')
full_df = pd.read_excel('matched_data_full_period.xlsx')

print("=" * 80)
print("基期倾向得分匹配 (PSM) 分析总结报告")
print("=" * 80)

print("\n一、分析设置")
print("-" * 80)
print("基期年份: 2009年")
print("匹配变量: ln_real_gdp, ln_人口密度, ln_金融发展水平, 第二产业占GDP比重")
print("匹配方法: 1:1 最近邻匹配")
print("卡尺: 0.05")

print("\n二、基期样本统计")
print("-" * 80)
print(f"2009年总城市数: {len(baseline_df)}")
print(f"处理组 (Treat=1): {baseline_df['Treat'].sum()} 个城市")
print(f"对照组 (Treat=0): {len(baseline_df) - baseline_df['Treat'].sum()} 个城市")

print("\n三、匹配结果")
print("-" * 80)
print(f"处理组城市数: 109 个")
print(f"成功匹配: {len(pairs_df)} 对")
print(f"匹配率: {len(pairs_df)/109*100:.2f}%")
print(f"唯一对照城市数: {pairs_df['control_city'].nunique()} 个")
print(f"平均得分差异: {pairs_df['score_diff'].mean():.4f}")
print(f"最大得分差异: {pairs_df['score_diff'].max():.4f}")

print("\n四、平衡性检验")
print("-" * 80)
print(balance_df.to_string(index=False))
print("\n说明: 标准化差异 < 10% 认为达到平衡")

balanced_count = (balance_df['std_diff'] < 10).sum()
total_count = len(balance_df)
print(f"\n平衡变量数: {balanced_count}/{total_count}")

print("\n五、匹配后数据集")
print("-" * 80)
print(f"全期数据总量: {len(full_df)} 条观测")
print(f"城市数量: {full_df['city_name'].nunique()} 个")
print(f"年份范围: {full_df['year'].min()} - {full_df['year'].max()}")
print(f"处理组观测数: {(full_df['Treat'] == 1).sum()}")
print(f"对照组观测数: {(full_df['Treat'] == 0).sum()}")

print("\n六、倾向得分统计")
print("-" * 80)
treat_scores = baseline_df[baseline_df['Treat'] == 1]['propensity_score']
control_scores = baseline_df[baseline_df['Treat'] == 0]['propensity_score']

print(f"处理组倾向得分:")
print(f"  均值: {treat_scores.mean():.4f}")
print(f"  标准差: {treat_scores.std():.4f}")
print(f"  最小值: {treat_scores.min():.4f}")
print(f"  最大值: {treat_scores.max():.4f}")

print(f"\n对照组倾向得分:")
print(f"  均值: {control_scores.mean():.4f}")
print(f"  标准差: {control_scores.std():.4f}")
print(f"  最小值: {control_scores.min():.4f}")
print(f"  最大值: {control_scores.max():.4f}")

print("\n七、输出文件")
print("-" * 80)
print("1. matched_data_full_period.xlsx - 匹配后全期数据（用于后续DID分析）")
print("2. baseline_2009_with_scores.xlsx - 基期数据及倾向得分")
print("3. balance_check_results.xlsx - 平衡性检验详细结果")
print("4. matched_pairs_details.xlsx - 匹配对详细信息")
print("5. psm_diagnostics.png - 诊断图表组合（4个子图）")
print("6. matched_pairs_scatter.png - 匹配对倾向得分对比散点图")

print("\n八、建议")
print("-" * 80)
if balanced_count < total_count:
    unbalanced_vars = balance_df[balance_df['std_diff'] >= 10]['variable'].tolist()
    print(f"注意: 以下变量平衡性检验未通过（标准化差异 >= 10%）:")
    for var in unbalanced_vars:
        std_diff = balance_df[balance_df['variable'] == var]['std_diff'].values[0]
        print(f"  - {var}: {std_diff:.2f}%")
    print("\n建议:")
    print("  1. 考虑调整卡尺大小")
    print("  2. 尝试其他匹配方法（如k近邻、核匹配等）")
    print("  3. 检查是否存在共同支撑域问题")
else:
    print("所有匹配变量均达到平衡性要求，可以进行后续DID分析。")

print("\n" + "=" * 80)
print("报告完成")
print("=" * 80)
