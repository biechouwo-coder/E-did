# -*- coding: utf-8 -*-
"""
PSM倾向得分匹配分析脚本（使用对数控制变量）

核心逻辑：
- 使用treat变量（静态分组变量）进行PSM匹配
- 基期：2009年（政策实施前）
- 卡尺：0.05

控制变量（对数形式）：
1. ln_pgdp - ln人均GDP
2. ln_pop_density - ln人口密度
3. ln_industrial_advanced - ln产业高级化
4. ln_fdi_openness - ln_FDI开放度
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import warnings
import sys
import io
warnings.filterwarnings('ignore')

# 设置输出编码为UTF-8
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

print("="*70)
print("PSM倾向得分匹配分析（对数控制变量版本）")
print("="*70)

# 设置输出路径
output_dir = 'PSM_Analysis'

# 步骤1: 读取数据
print("\n步骤1: 读取数据...")
print("使用数据集：总数据集_2007-2023_完整版_DID_with_treat_log.xlsx")
df = pd.read_excel('总数据集_2007-2023_完整版_DID_with_treat_log.xlsx')
print(f"✓ 读取数据: {df.shape[0]} 行 × {df.shape[1]} 列")

# 步骤2: 准备基期数据（使用2009年作为基期）
print("\n步骤2: 准备基期数据...")
base_year = 2009
df_base = df[df['year'] == base_year].copy()
print(f"✓ 基期年份: {base_year} (政策实施前)")
print(f"✓ 基期观测数: {len(df_base)}")

# 检查处理组和对照组（基于treat变量）
treatment_count = (df_base['treat'] == 1).sum()
control_count = (df_base['treat'] == 0).sum()
print(f"  处理组 (treat=1, 试点城市): {treatment_count} 个城市")
print(f"  对照组 (treat=0, 非试点城市): {control_count} 个城市")

# 步骤3: 准备控制变量（对数形式）
print("\n步骤3: 准备控制变量（对数形式）...")
control_vars = ['ln_pgdp', 'ln_pop_density', 'ln_industrial_advanced', 'ln_fdi_openness']
print(f"控制变量: {control_vars}")
print(f"  1. ln_pgdp - ln人均GDP")
print(f"  2. ln_pop_density - ln人口密度")
print(f"  3. ln_industrial_advanced - ln产业高级化")
print(f"  4. ln_fdi_openness - ln_FDI开放度")

# 检查缺失值
print("\n检查缺失值:")
for var in control_vars:
    missing = df_base[var].isnull().sum()
    print(f"  {var}: {missing} 个缺失值")

# 删除缺失值
df_base_clean = df_base.dropna(subset=control_vars + ['treat', 'city_name'])
print(f"\n✓ 清理后基期数据: {len(df_base_clean)} 个观测")

# 标准化控制变量
scaler = StandardScaler()
X = df_base_clean[control_vars].values
X_scaled = scaler.fit_transform(X)
y = df_base_clean['treat'].values

print(f"\n处理组 (treat=1, n={y.sum()}):")
print(df_base_clean[df_base_clean['treat']==1][control_vars].describe())
print(f"\n对照组 (treat=0, n={len(y)-y.sum()}):")
print(df_base_clean[df_base_clean['treat']==0][control_vars].describe())

# 步骤4: 计算倾向得分
print("\n" + "="*70)
print("步骤4: 计算倾向得分...")
print("="*70)

# 使用Logistic回归计算倾向得分
lr_model = LogisticRegression()
lr_model.fit(X_scaled, y)
propensity_scores = lr_model.predict_proba(X_scaled)[:, 1]

df_base_clean['propensity_score'] = propensity_scores

print(f"✓ 倾向得分计算完成")
print(f"\n倾向得分统计:")
print(f"  全体: 均值={propensity_scores.mean():.4f}, 标准差={propensity_scores.std():.4f}")
print(f"  处理组 (treat=1): 均值={propensity_scores[y==1].mean():.4f}, 标准差={propensity_scores[y==1].std():.4f}")
print(f"  对照组 (treat=0): 均值={propensity_scores[y==0].mean():.4f}, 标准差={propensity_scores[y==0].std():.4f}")

# 保存倾向得分数据
df_base_clean.to_excel(f'{output_dir}/propensity_scores_log_vars.xlsx', index=False)
print(f"\n✓ 倾向得分已保存至: {output_dir}/propensity_scores_log_vars.xlsx")

# 步骤5: 执行匹配（卡尺=0.05）
print("\n" + "="*70)
print("步骤5: 执行倾向得分匹配 (卡尺=0.05)...")
print("="*70)

caliper = 0.05

treated = df_base_clean[df_base_clean['treat'] == 1].copy()
control = df_base_clean[df_base_clean['treat'] == 0].copy()

print(f"\n匹配前:")
print(f"  处理组 (treat=1): {len(treated)} 个")
print(f"  对照组 (treat=0): {len(control)} 个")

# 为每个处理组观测找到对照组中的匹配
matched_pairs = []
matched_control_indices = []

for idx, treated_row in treated.iterrows():
    treated_score = treated_row['propensity_score']

    # 计算与所有对照组的得分差
    control['score_diff'] = np.abs(control['propensity_score'] - treated_score)

    # 找到卡尺范围内得分差最小的对照组
    eligible = control[control['score_diff'] <= caliper]

    if len(eligible) > 0:
        # 找到最近的匹配
        best_match = eligible.loc[eligible['score_diff'].idxmin()]
        matched_pairs.append({
            'treated_city': treated_row['city_name'],
            'treated_score': treated_score,
            'control_city': best_match['city_name'],
            'control_score': best_match['propensity_score'],
            'score_diff': best_match['score_diff']
        })
        matched_control_indices.append(best_match.name)

# 创建匹配结果DataFrame
matches_df = pd.DataFrame(matched_pairs)

print(f"\n匹配结果:")
print(f"  成功匹配的处理组: {len(matches_df)} 个")
print(f"  匹配率: {len(matches_df)/len(treated)*100:.1f}%")
print(f"  平均得分差: {matches_df['score_diff'].mean():.4f}")
print(f"  最大得分差: {matches_df['score_diff'].max():.4f}")

# 保存匹配结果
matches_df.to_excel(f'{output_dir}/matched_pairs_log_vars.xlsx', index=False)
print(f"\n✓ 匹配结果已保存至: {output_dir}/matched_pairs_log_vars.xlsx")

# 获取匹配后的样本
matched_treated_cities = matches_df['treated_city'].tolist()
matched_control_cities = matches_df['control_city'].tolist()

# 创建匹配后的数据集（用于后续DID分析）
matched_cities = matched_treated_cities + matched_control_cities
df_matched = df[df['city_name'].isin(matched_cities)].copy()

# 创建匹配后的处理组标识
df_matched['matched_treatment'] = df_matched.apply(
    lambda x: 1 if x['city_name'] in matched_treated_cities else 0,
    axis=1
)

# 为匹配后的样本创建新的DID变量（只对匹配成功的处理组）
def get_matched_did(row):
    if row['city_name'] in matched_treated_cities:
        return row['DID']
    else:
        return 0

df_matched['DID_matched'] = df_matched.apply(get_matched_did, axis=1)

df_matched.to_excel(f'{output_dir}/matched_dataset_log_vars.xlsx', index=False)
print(f"✓ 匹配后数据集已保存至: {output_dir}/matched_dataset_log_vars.xlsx")

# 步骤6: 匹配质量评估
print("\n" + "="*70)
print("步骤6: 匹配质量评估")
print("="*70)

# 6.1 倾向得分分布图
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 匹配前
axes[0].hist(propensity_scores[y==0], bins=30, alpha=0.5, label='对照组 (treat=0)', color='blue')
axes[0].hist(propensity_scores[y==1], bins=30, alpha=0.5, label='处理组 (treat=1)', color='red')
axes[0].set_xlabel('倾向得分')
axes[0].set_ylabel('频数')
axes[0].set_title(f'匹配前倾向得分分布 (基期{base_year}年)')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# 匹配后
matched_treated_scores = treated[treated['city_name'].isin(matched_treated_cities)]['propensity_score']
matched_control_scores = control.loc[matched_control_indices]['propensity_score']

axes[1].hist(matched_control_scores, bins=30, alpha=0.5, label='对照组 (匹配后)', color='blue')
axes[1].hist(matched_treated_scores, bins=30, alpha=0.5, label='处理组 (匹配后)', color='red')
axes[1].set_xlabel('倾向得分')
axes[1].set_ylabel('频数')
axes[1].set_title(f'匹配后倾向得分分布 (卡尺={caliper})')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{output_dir}/propensity_score_distribution_log_vars.png', dpi=300, bbox_inches='tight')
print(f"✓ 倾向得分分布图已保存: {output_dir}/propensity_score_distribution_log_vars.png")

# 6.2 计算标准化差异
def calculate_stdized_diff(treated_vals, control_vals):
    n_treat = len(treated_vals)
    n_ctrl = len(control_vals)
    mean_treat = treated_vals.mean()
    mean_ctrl = control_vals.mean()
    var_treat = treated_vals.var()
    var_ctrl = control_vals.var()
    pooled_std = np.sqrt((var_treat * (n_treat-1) + var_ctrl * (n_ctrl-1)) / (n_treat + n_ctrl - 2))
    return (mean_treat - mean_ctrl) / pooled_std * 100

# 匹配前的标准化差异
print("\n匹配前标准化差异 (%):")
before_std_diff = {}
for var in control_vars:
    treated_vals = df_base_clean[df_base_clean['treat']==1][var]
    control_vals = df_base_clean[df_base_clean['treat']==0][var]
    std_diff = calculate_stdized_diff(treated_vals, control_vals)
    before_std_diff[var] = std_diff
    print(f"  {var}: {std_diff:.2f}%")

# 匹配后的标准化差异
print("\n匹配后标准化差异 (%):")
after_std_diff = {}
for var in control_vars:
    treated_vals = df_base_clean[df_base_clean['city_name'].isin(matched_treated_cities)][var]
    control_vals = df_base_clean[df_base_clean['city_name'].isin(matched_control_cities)][var]
    std_diff = calculate_stdized_diff(treated_vals, control_vals)
    after_std_diff[var] = std_diff
    print(f"  {var}: {std_diff:.2f}%")

# 可视化标准化差异
fig, ax = plt.subplots(figsize=(10, 6))
x_pos = np.arange(len(control_vars))
width = 0.35

before_vals = [before_std_diff[v] for v in control_vars]
after_vals = [after_std_diff[v] for v in control_vars]

bars1 = ax.bar(x_pos - width/2, before_vals, width, label='匹配前', color='orange', alpha=0.7)
bars2 = ax.bar(x_pos + width/2, after_vals, width, label='匹配后', color='green', alpha=0.7)

ax.axhline(y=10, color='red', linestyle='--', linewidth=1, label='阈值(10%)')
ax.axhline(y=-10, color='red', linestyle='--', linewidth=1)
ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

ax.set_xlabel('控制变量')
ax.set_ylabel('标准化差异 (%)')
ax.set_title('匹配前后控制变量标准化差异对比（对数变量）')
ax.set_xticks(x_pos)
ax.set_xticklabels(['ln_pgdp', 'ln_pop_density', 'ln_industrial_advanced', 'ln_fdi_openness'],
                  fontsize=9, rotation=15, ha='right')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(f'{output_dir}/standardized_differences_log_vars.png', dpi=300, bbox_inches='tight')
print(f"\n✓ 标准化差异对比图已保存: {output_dir}/standardized_differences_log_vars.png")

# 6.3 匹配前后对比表
comparison_df = pd.DataFrame({
    '变量': control_vars,
    '变量名称': ['ln人均GDP', 'ln人口密度', 'ln产业高级化', 'ln_FDI开放度'],
    '匹配前标准化差异(%)': [before_std_diff[v] for v in control_vars],
    '匹配后标准化差异(%)': [after_std_diff[v] for v in control_vars],
    '处理组均值(匹配前)': [df_base_clean[df_base_clean['treat']==1][v].mean() for v in control_vars],
    '对照组均值(匹配前)': [df_base_clean[df_base_clean['treat']==0][v].mean() for v in control_vars],
    '处理组均值(匹配后)': [df_base_clean[df_base_clean['city_name'].isin(matched_treated_cities)][v].mean() for v in control_vars],
    '对照组均值(匹配后)': [df_base_clean[df_base_clean['city_name'].isin(matched_control_cities)][v].mean() for v in control_vars],
})

comparison_df.to_excel(f'{output_dir}/balance_check_log_vars.xlsx', index=False)
print(f"✓ 平衡性检验表已保存: {output_dir}/balance_check_log_vars.xlsx")

# 步骤7: 生成匹配摘要报告
print("\n" + "="*70)
print("PSM匹配分析摘要")
print("="*70)

summary = f"""
{'='*70}
PSM倾向得分匹配分析报告（对数控制变量版本）
{'='*70}

重要说明：
-----------
本分析使用treat变量进行PSM匹配，匹配的是"身份"而非"状态"

• treat: 静态分组变量（Who）
  - treat=1: 在低碳试点名单中的城市（无论何时开始实施）
  - treat=0: 从未在试点名单中的城市
  - treat不随时间变化

• DID: 动态处理变量（When）
  - DID=1: 该年已实施政策（年份 >= 该城市生效年份）
  - DID=0: 该年未实施政策
  - DID随时间变化

一、分析设定
-----------
• 基期年份: {base_year}年（政策实施前）
• 匹配变量: treat（静态分组变量）
• 控制变量（对数形式）: {', '.join(control_vars)}
  1. ln_pgdp - ln人均GDP
  2. ln_pop_density - ln人口密度
  3. ln_industrial_advanced - ln产业高级化
  4. ln_fdi_openness - ln_FDI开放度
• 卡尺设定: {caliper}
• 匹配方法: 1:1最近邻匹配

二、匹配结果
-----------
• 处理组数量 (treat=1): {len(treated)} 个城市
• 对照组数量 (treat=0): {len(control)} 个城市
• 成功匹配: {len(matches_df)} 对
• 匹配率: {len(matches_df)/len(treated)*100:.1f}%
• 平均得分差: {matches_df['score_diff'].mean():.4f}
• 最大得分差: {matches_df['score_diff'].max():.4f}

三、平衡性检验
-----------
匹配前标准化差异 (%):
{chr(10).join([f"  • {v}: {before_std_diff[v]:.2f}%" for v in control_vars])}

匹配后标准化差异 (%):
{chr(10).join([f"  • {v}: {after_std_diff[v]:.2f}%" for v in control_vars])}

四、匹配质量评估
-----------
"""

# 评估匹配质量
good_balance = sum([abs(after_std_diff[v]) < 10 for v in control_vars])
summary += f"• 标准化差异 < 10% 的变量: {good_balance}/{len(control_vars)}\n"

if good_balance == len(control_vars):
    summary += "• 结论: 匹配质量优秀，所有变量均达到平衡标准\n"
elif good_balance >= len(control_vars) * 0.75:
    summary += "• 结论: 匹配质量良好，大部分变量达到平衡标准\n"
elif good_balance >= len(control_vars) * 0.5:
    summary += "• 结论: 匹配质量一般，半数变量达到平衡标准\n"
else:
    summary += "• 结论: 匹配质量较差，建议调整匹配参数或增加控制变量\n"

summary += f"""
五、输出文件
-----------
所有结果已保存至目录: {output_dir}/
1. propensity_scores_log_vars.xlsx - 倾向得分数据
2. matched_pairs_log_vars.xlsx - 匹配对详细信息
3. matched_dataset_log_vars.xlsx - 匹配后完整数据集
4. propensity_score_distribution_log_vars.png - 倾向得分分布图
5. standardized_differences_log_vars.png - 标准化差异对比图
6. balance_check_log_vars.xlsx - 平衡性检验详细表
7. psm_summary_log_vars.txt - 本摘要报告

六、后续DID分析建议
-----------
使用匹配后数据集 (matched_dataset_log_vars.xlsx) 进行DID分析：

变量说明：
• treat: 原始分组变量（是否在试点名单中）
• matched_treatment: 匹配后处理组标识（1=匹配成功的试点城市）
• DID: 原始DID变量（是否已实施政策）
• DID_matched: 匹配后DID变量（仅对匹配成功的处理组有效）

推荐模型：
Y = β₀ + β₁×DID_matched + β₂×matched_treatment + γ×控制变量 + δ×年份固定效应 + θ×城市固定效应 + ε

其中：
• DID_matched: 核心解释变量（政策效应）
• matched_treatment: 处理组固定效应
• 控制变量（原始形式）: gdp_per_capita, pop_density, industrial_upgrading, fdi_openness
• 样本期：2007-2023年
• 样本城市：{len(matched_treated_cities) + len(matched_control_cities)} 个（{len(matches_df)}对）

{'='*70}
"""

print(summary)

# 保存摘要报告
with open(f'{output_dir}/psm_summary_log_vars.txt', 'w', encoding='utf-8') as f:
    f.write(summary)

print(f"\n✓ 分析完成！所有结果已保存至 {output_dir}/ 目录")
