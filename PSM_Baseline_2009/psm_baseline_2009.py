import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 60)
print("基期倾向得分匹配分析 (PSM) - 2009年基期")
print("=" * 60)

# 1. 读取数据
print("\n步骤1: 读取数据...")
df = pd.read_excel('../总数据集2007-2023_仅含emission城市.xlsx')
print(f"数据集形状: {df.shape}")
print(f"年份范围: {df['year'].min()} - {df['year'].max()}")

# 2. 筛选基期数据
print("\n步骤2: 筛选2009年基期数据...")
baseline_data = df[df['year'] == 2009].copy()
print(f"2009年数据量: {len(baseline_data)} 个城市")

# 检查处理组和对照组
treat_count = baseline_data['Treat'].sum()
control_count = len(baseline_data) - treat_count
print(f"处理组 (Treat=1): {treat_count} 个城市")
print(f"对照组 (Treat=0): {control_count} 个城市")

# 3. 准备匹配变量
print("\n步骤3: 准备匹配变量...")
match_vars = ['ln_real_gdp', 'ln_人口密度', 'ln_金融发展水平', '第二产业占GDP比重']

# 检查数据完整性
print("\n检查匹配变量的缺失值:")
for var in match_vars:
    missing_count = baseline_data[var].isna().sum()
    missing_pct = missing_count / len(baseline_data) * 100
    print(f"  {var}: {missing_count} 个缺失值 ({missing_pct:.2f}%)")

# 删除有缺失值的观测
baseline_clean = baseline_data.dropna(subset=match_vars).copy()
print(f"\n删除缺失值后的样本量: {len(baseline_clean)} 个城市")
print(f"处理组: {baseline_clean['Treat'].sum()} 个城市")
print(f"对照组: {len(baseline_clean) - baseline_clean['Treat'].sum()} 个城市")

# 标准化匹配变量（用于计算倾向得分）
X = baseline_clean[match_vars].values
y = baseline_clean['Treat'].values

# 标准化
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. 计算倾向得分
print("\n步骤4: 计算倾向得分...")
# 使用逻辑回归计算倾向得分
lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_scaled, y)

# 获取倾向得分
propensity_scores = lr.predict_proba(X_scaled)[:, 1]
baseline_clean['propensity_score'] = propensity_scores

print(f"倾向得分统计:")
print(f"  处理组: 均值={propensity_scores[y==1].mean():.4f}, 标准差={propensity_scores[y==1].std():.4f}")
print(f"  对照组: 均值={propensity_scores[y==0].mean():.4f}, 标准差={propensity_scores[y==0].std():.4f}")

# 5. 执行匹配（卡尺0.05）
print("\n步骤5: 执行倾向得分匹配...")
print(f"匹配方法: 1:1 最近邻匹配，卡尺=0.05")

caliper = 0.05
treated_indices = np.where(y == 1)[0]
control_indices = np.where(y == 0)[0]

# 使用K近邻进行匹配
nn = NearestNeighbors(n_neighbors=1)
nn.fit(propensity_scores[control_indices].reshape(-1, 1))

matched_pairs = []
matched_control_indices = []

for treat_idx in treated_indices:
    treat_score = propensity_scores[treat_idx]
    # 找到最近的对照
    distances, indices = nn.kneighbors([[treat_score]])
    control_idx = control_indices[indices[0][0]]
    distance = distances[0][0]

    # 检查是否在卡尺范围内
    if distance <= caliper:
        matched_pairs.append((treat_idx, control_idx, distance))
        matched_control_indices.append(control_idx)

# 检查是否有重复匹配
unique_matched_control = len(set(matched_control_indices))
total_matched = len(matched_pairs)

print(f"\n匹配结果:")
print(f"  处理组城市数: {len(treated_indices)}")
print(f"  成功匹配: {total_matched} 对")
print(f"  匹配率: {total_matched/len(treated_indices)*100:.2f}%")
print(f"  对照组去重后: {unique_matched_control} 个唯一对照")

# 6. 创建匹配后的数据集
print("\n步骤6: 创建匹配后的数据集...")
matched_treated_indices = [pair[0] for pair in matched_pairs]
matched_control_indices_final = [pair[1] for pair in matched_pairs]

matched_baseline = baseline_clean.iloc[matched_treated_indices + matched_control_indices_final].copy()
matched_baseline['matched_pair_id'] = list(range(len(matched_pairs))) * 2

print(f"匹配后基期数据: {len(matched_baseline)} 个观测（{len(matched_pairs)}对）")

# 7. 将匹配扩展到所有年份
print("\n步骤7: 将匹配扩展到所有年份...")
matched_cities = matched_baseline['city_name'].unique()
full_matched_data = df[df['city_name'].isin(matched_cities)].copy()

print(f"全期匹配数据: {len(full_matched_data)} 个观测")
print(f"城市数: {len(matched_cities)}")
print(f"年份范围: {full_matched_data['year'].min()} - {full_matched_data['year'].max()}")

# 8. 平衡性检验
print("\n步骤8: 平衡性检验...")

def calculate_std_diff(treated, control, var_name):
    """计算标准化差异"""
    treated_mean = np.mean(treated)
    control_mean = np.mean(control)
    pooled_std = np.sqrt((np.var(treated) + np.var(control)) / 2)
    std_diff = abs((treated_mean - control_mean) / pooled_std) * 100
    return std_diff

balance_results = []
for var in match_vars:
    treated_values = baseline_clean.iloc[matched_treated_indices][var].values
    control_values = baseline_clean.iloc[matched_control_indices_final][var].values

    std_diff = calculate_std_diff(treated_values, control_values, var)
    balance_results.append({
        'variable': var,
        'treated_mean': treated_values.mean(),
        'control_mean': control_values.mean(),
        'std_diff': std_diff,
        'balanced': 'Yes' if std_diff < 10 else 'No'  # 标准化差异<10%认为平衡
    })

balance_df = pd.DataFrame(balance_results)

print("\n平衡性检验结果:")
print(balance_df.to_string(index=False))

# 9. 保存结果
print("\n步骤9: 保存结果...")

# 保存匹配后的数据
full_matched_data.to_excel('matched_data_full_period.xlsx', index=False)
print(f"[OK] 已保存匹配后全期数据: matched_data_full_period.xlsx")

baseline_clean.to_excel('baseline_2009_with_scores.xlsx', index=False)
print(f"[OK] 已保存基期倾向得分: baseline_2009_with_scores.xlsx")

balance_df.to_excel('balance_check_results.xlsx', index=False)
print(f"[OK] 已保存平衡性检验结果: balance_check_results.xlsx")

# 保存匹配对信息
pairs_info = []
for i, (treat_idx, control_idx, distance) in enumerate(matched_pairs):
    treat_city = baseline_clean.iloc[treat_idx]['city_name']
    control_city = baseline_clean.iloc[control_idx]['city_name']
    treat_score = propensity_scores[treat_idx]
    control_score = propensity_scores[control_idx]

    pairs_info.append({
        'pair_id': i + 1,
        'treated_city': treat_city,
        'control_city': control_city,
        'treated_score': treat_score,
        'control_score': control_score,
        'score_diff': distance
    })

pairs_df = pd.DataFrame(pairs_info)
pairs_df.to_excel('matched_pairs_details.xlsx', index=False)
print(f"[OK] 已保存匹配对详细信息: matched_pairs_details.xlsx")

# 10. 生成可视化图表
print("\n步骤10: 生成可视化图表...")

# 图1: 倾向得分分布
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 倾向得分分布（匹配前）
ax = axes[0, 0]
ax.hist(propensity_scores[y==0], bins=20, alpha=0.5, label='对照组', color='blue')
ax.hist(propensity_scores[y==1], bins=20, alpha=0.5, label='处理组', color='red')
ax.set_xlabel('倾向得分')
ax.set_ylabel('频数')
ax.set_title('倾向得分分布（匹配前）')
ax.legend()

# 倾向得分分布（匹配后）
ax = axes[0, 1]
matched_treat_scores = propensity_scores[matched_treated_indices]
matched_control_scores = propensity_scores[matched_control_indices_final]
ax.hist(matched_control_scores, bins=20, alpha=0.5, label='对照组', color='blue')
ax.hist(matched_treat_scores, bins=20, alpha=0.5, label='处理组', color='red')
ax.set_xlabel('倾向得分')
ax.set_ylabel('频数')
ax.set_title('倾向得分分布（匹配后）')
ax.legend()

# 标准化差异（匹配前 vs 匹配后）
ax = axes[1, 0]
variables = balance_df['variable'].tolist()
x_pos = np.arange(len(variables))

# 计算匹配前的标准化差异
before_std_diffs = []
after_std_diffs = []

for var in match_vars:
    treated_before = baseline_clean[baseline_clean['Treat'] == 1][var].values
    control_before = baseline_clean[baseline_clean['Treat'] == 0][var].values
    std_diff_before = calculate_std_diff(treated_before, control_before, var)
    before_std_diffs.append(std_diff_before)

    treated_after = baseline_clean.iloc[matched_treated_indices][var].values
    control_after = baseline_clean.iloc[matched_control_indices_final][var].values
    std_diff_after = calculate_std_diff(treated_after, control_after, var)
    after_std_diffs.append(std_diff_after)

width = 0.35
ax.bar(x_pos - width/2, before_std_diffs, width, label='匹配前', color='orange', alpha=0.7)
ax.bar(x_pos + width/2, after_std_diffs, width, label='匹配后', color='green', alpha=0.7)
ax.set_xlabel('变量')
ax.set_ylabel('标准化差异 (%)')
ax.set_title('标准化差异对比')
ax.set_xticks(x_pos)
ax.set_xticklabels(variables, rotation=45, ha='right')
ax.axhline(y=10, color='red', linestyle='--', linewidth=1, label='10%阈值')
ax.legend()

# 匹配对得分差异分布
ax = axes[1, 1]
score_diffs = [pair[2] for pair in matched_pairs]
ax.hist(score_diffs, bins=20, color='purple', alpha=0.7, edgecolor='black')
ax.axvline(x=caliper, color='red', linestyle='--', linewidth=2, label=f'卡尺={caliper}')
ax.set_xlabel('倾向得分差异')
ax.set_ylabel('匹配对数量')
ax.set_title('匹配对倾向得分差异分布')
ax.legend()

plt.tight_layout()
plt.savefig('psm_diagnostics.png', dpi=300, bbox_inches='tight')
print("[OK] 已保存诊断图: psm_diagnostics.png")

# 11. 生成匹配对散点图
fig, ax = plt.subplots(figsize=(10, 8))

for i, (treat_idx, control_idx, distance) in enumerate(matched_pairs):
    treat_city = baseline_clean.iloc[treat_idx]['city_name']
    control_city = baseline_clean.iloc[control_idx]['city_name']
    treat_score = propensity_scores[treat_idx]
    control_score = propensity_scores[control_idx]

    ax.scatter([treat_score, control_score], [i, i],
               c=['red', 'blue'], s=50, alpha=0.6)
    ax.plot([treat_score, control_score], [i, i], 'k-', alpha=0.3)

ax.set_xlabel('倾向得分')
ax.set_ylabel('匹配对编号')
ax.set_title('匹配对倾向得分对比')
ax.legend(['处理组', '对照组'])
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('matched_pairs_scatter.png', dpi=300, bbox_inches='tight')
print("[OK] 已保存匹配对散点图: matched_pairs_scatter.png")

print("\n" + "=" * 60)
print("PSM分析完成！")
print("=" * 60)

print("\n输出文件总结:")
print("1. matched_data_full_period.xlsx - 匹配后全期数据（2007-2023）")
print("2. baseline_2009_with_scores.xlsx - 基期数据及倾向得分")
print("3. balance_check_results.xlsx - 平衡性检验结果")
print("4. matched_pairs_details.xlsx - 匹配对详细信息")
print("5. psm_diagnostics.png - 诊断图表组合")
print("6. matched_pairs_scatter.png - 匹配对散点图")
