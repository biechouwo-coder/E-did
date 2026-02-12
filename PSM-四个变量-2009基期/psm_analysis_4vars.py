# -*- coding: utf-8 -*-
"""
PSM分析脚本（四个匹配变量）
基期：2009年
匹配变量：ln_real_gdp、ln_人口密度、ln_金融发展水平、第二产业占GDP比重
匹配方法：1:1有放回匹配
卡尺：倾向得分对数几率的标准差的0.25倍
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import sys
import io

# 设置标准输出编码为UTF-8
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

print("="*80)
print("PSM倾向得分匹配分析（四个匹配变量）")
print("="*80)

# 1. 读取数据
print("\n[步骤1] 读取数据...")
df = pd.read_excel("../总数据集2007-2019_含碳排放强度.xlsx")
print(f"原始数据形状: {df.shape}")
print(f"年份范围: {df['year'].min()} - {df['year'].max()}")

# 2. 提取2009年基期数据
print("\n[步骤2] 提取2009年基期数据...")
df_base = df[df['year'] == 2009].copy()
print(f"2009年样本数: {len(df_base)}")

# 3. 检查变量是否存在
print("\n[步骤3] 检查变量...")
required_vars = ['ln_real_gdp', 'ln_人口密度', 'ln_金融发展水平', '第二产业占GDP比重']
missing_vars = [var for var in required_vars if var not in df_base.columns]

if missing_vars:
    print(f"错误：缺少以下变量: {missing_vars}")
    print("可用变量:", df_base.columns.tolist())
    sys.exit(1)

print("所有匹配变量均存在")

# 4. 数据预处理
print("\n[步骤4] 数据预处理...")
treatment_col = 'Treat'

if treatment_col not in df_base.columns:
    print(f"错误：未找到处理组变量 '{treatment_col}'")
    print("可用变量:", df_base.columns.tolist())
    sys.exit(1)

print(f"使用处理组变量: {treatment_col}")
print(f"处理组样本数: {df_base[treatment_col].sum()}")
print(f"对照组样本数: {(df_base[treatment_col]==0).sum()}")

# 删除含有缺失值的样本
df_clean = df_base[required_vars + [treatment_col]].dropna()
print(f"\n删除缺失值后样本数: {len(df_clean)}")
print(f"处理组样本数: {df_clean[treatment_col].sum()}")
print(f"对照组样本数: {(df_clean[treatment_col]==0).sum()}")

# 检查两组都有足够的样本
if df_clean[treatment_col].sum() < 2:
    print("警告：处理组样本数过少，无法进行匹配")
    sys.exit(1)

if (df_clean[treatment_col]==0).sum() < 2:
    print("警告：对照组样本数过少，无法进行匹配")
    sys.exit(1)

# 5. 描述性统计：匹配前
print("\n[步骤5] 匹配前描述性统计...")
vars_with_treat = required_vars + [treatment_col]
df_matched_before = df_clean[vars_with_treat].copy()

# 按处理组分组
treat_group = df_matched_before[df_matched_before[treatment_col] == 1]
control_group = df_matched_before[df_matched_before[treatment_col] == 0]

print("\n匹配前各变量均值比较:")
print("-" * 80)
comparison_results_before = []

for var in required_vars:
    treat_mean = treat_group[var].mean()
    control_mean = control_group[var].mean()
    diff = treat_mean - control_mean

    # t检验
    t_stat, p_value = stats.ttest_ind(
        treat_group[var].dropna(),
        control_group[var].dropna(),
        equal_var=False
    )

    # 标准化偏差
    pooled_std = np.sqrt((treat_group[var].std()**2 + control_group[var].std()**2) / 2)
    std_bias = (diff / pooled_std) * 100 if pooled_std > 0 else 0

    comparison_results_before.append({
        '变量': var,
        '处理组均值': treat_mean,
        '对照组均值': control_mean,
        '均值差异': diff,
        '标准化偏差(%)': std_bias,
        't统计量': t_stat,
        'p值': p_value
    })

    print(f"{var}:")
    print(f"  处理组均值: {treat_mean:.4f}")
    print(f"  对照组均值: {control_mean:.4f}")
    print(f"  均值差异: {diff:.4f}")
    print(f"  标准化偏差: {std_bias:.2f}%")
    print(f"  t统计量: {t_stat:.4f}, p值: {p_value:.4f}")
    print()

df_comparison_before = pd.DataFrame(comparison_results_before)
df_comparison_before.to_excel("匹配前比较_四个变量.xlsx", index=False)
print("匹配前比较结果已保存至: 匹配前比较_四个变量.xlsx")

# 6. 计算倾向得分
print("\n[步骤6] 计算倾向得分...")
X = df_clean[required_vars]
y = df_clean[treatment_col]

# 标准化协变量
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 使用Logistic回归计算倾向得分
logit = LogisticRegression(max_iter=1000, random_state=42)
logit.fit(X_scaled, y)

df_clean['propensity_score'] = logit.predict_proba(X_scaled)[:, 1]

print(f"倾向得分范围: {df_clean['propensity_score'].min():.4f} - {df_clean['propensity_score'].max():.4f}")
print(f"倾向得分均值: {df_clean['propensity_score'].mean():.4f}")
print(f"处理组倾向得分均值: {df_clean[df_clean[treatment_col]==1]['propensity_score'].mean():.4f}")
print(f"对照组倾向得分均值: {df_clean[df_clean[treatment_col]==0]['propensity_score'].mean():.4f}")

# 7. 计算卡尺（倾向得分对数几率的标准差的0.25倍）
print("\n[步骤7] 计算卡尺...")
# 计算对数几率
logit_score = np.log(df_clean['propensity_score'] / (1 - df_clean['propensity_score']))
# 计算标准差
caliper = 0.25 * logit_score.std()
caliper_original = 0.25 * df_clean['propensity_score'].std()
print(f"卡尺（对数几率标准差的0.25倍）: {caliper:.6f}")
print(f"卡尺（倾向得分原始尺度）: {caliper_original:.6f}")

# 8. 执行匹配（1:1有放回）
print("\n[步骤8] 执行1:1有放回匹配...")
treated = df_clean[df_clean[treatment_col] == 1]
control = df_clean[df_clean[treatment_col] == 0]

# 使用倾向得分进行匹配
treated_scores = treated['propensity_score'].values.reshape(-1, 1)
control_scores = control['propensity_score'].values.reshape(-1, 1)

# 创建NN模型（k=1表示1:1匹配）
nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree')
nbrs.fit(control_scores)

# 找到最近邻
distances, indices = nbrs.kneighbors(treated_scores)

# 应用卡尺限制
matched_control_indices = []
matched_treated_indices = []

for i, (dist, idx) in enumerate(zip(distances, indices)):
    if dist[0] <= caliper_original:
        matched_treated_indices.append(i)
        matched_control_indices.append(idx[0])

print(f"匹配成功数量: {len(matched_treated_indices)} / {len(treated)}")
print(f"匹配成功率: {len(matched_treated_indices)/len(treated)*100:.2f}%")

if len(matched_treated_indices) == 0:
    print("警告：没有匹配成功！请尝试增大卡尺或检查数据")
    sys.exit(1)

# 构建匹配后的数据集
matched_treated = treated.iloc[matched_treated_indices].copy()
matched_control = control.iloc[matched_control_indices].copy()

matched_treated['matched_id'] = range(len(matched_treated))
matched_control['matched_id'] = range(len(matched_control))

df_matched = pd.concat([matched_treated, matched_control], ignore_index=True)
print(f"匹配后总样本数: {len(df_matched)}")

# 9. 匹配后描述性统计
print("\n[步骤9] 匹配后描述性统计...")
treat_matched = df_matched[df_matched[treatment_col] == 1]
control_matched = df_matched[df_matched[treatment_col] == 0]

print("\n匹配后各变量均值比较:")
print("-" * 80)
comparison_results_after = []

for var in required_vars:
    treat_mean = treat_matched[var].mean()
    control_mean = control_matched[var].mean()
    diff = treat_mean - control_mean

    # t检验
    t_stat, p_value = stats.ttest_ind(
        treat_matched[var].dropna(),
        control_matched[var].dropna(),
        equal_var=False
    )

    # 标准化偏差
    pooled_std = np.sqrt((treat_matched[var].std()**2 + control_matched[var].std()**2) / 2)
    std_bias = (diff / pooled_std) * 100 if pooled_std > 0 else 0

    comparison_results_after.append({
        '变量': var,
        '处理组均值': treat_mean,
        '对照组均值': control_mean,
        '均值差异': diff,
        '标准化偏差(%)': std_bias,
        't统计量': t_stat,
        'p值': p_value
    })

    print(f"{var}:")
    print(f"  处理组均值: {treat_mean:.4f}")
    print(f"  对照组均值: {control_mean:.4f}")
    print(f"  均值差异: {diff:.4f}")
    print(f"  标准化偏差: {std_bias:.2f}%")
    print(f"  t统计量: {t_stat:.4f}, p值: {p_value:.4f}")
    print()

df_comparison_after = pd.DataFrame(comparison_results_after)
df_comparison_after.to_excel("匹配后比较_四个变量.xlsx", index=False)
print("匹配后比较结果已保存至: 匹配后比较_四个变量.xlsx")

# 10. 生成汇总结果
print("\n[步骤10] 生成汇总结果...")
summary = pd.DataFrame({
    '指标': [
        '基期年份',
        '匹配变量数量',
        '匹配变量',
        '匹配方法',
        '卡尺',
        '原始处理组样本数',
        '原始对照组样本数',
        '匹配成功数量',
        '匹配成功率',
        '匹配后处理组样本数',
        '匹配后对照组样本数'
    ],
    '值': [
        '2009',
        len(required_vars),
        ', '.join(required_vars),
        '1:1有放回匹配',
        f'{caliper_original:.6f}',
        len(treat_group),
        len(control_group),
        len(matched_treated),
        f'{len(matched_treated)/len(treated)*100:.2f}%',
        len(treat_matched),
        len(control_matched)
    ]
})

summary.to_excel("PSM匹配结果汇总_四个变量.xlsx", index=False)
print("汇总结果已保存至: PSM匹配结果汇总_四个变量.xlsx")

# 11. 可视化
print("\n[步骤11] 生成可视化图表...")

# 创建图表
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# 重新获取包含倾向得分的数据
treat_group_with_ps = df_clean[df_clean[treatment_col] == 1]
control_group_with_ps = df_clean[df_clean[treatment_col] == 0]

# (1) 倾向得分分布
ax1 = axes[0, 0]
ax1.hist(treat_group_with_ps['propensity_score'], bins=20, alpha=0.5, label='处理组', color='red')
ax1.hist(control_group_with_ps['propensity_score'], bins=20, alpha=0.5, label='对照组', color='blue')
ax1.set_xlabel('倾向得分', fontsize=12)
ax1.set_ylabel('频数', fontsize=12)
ax1.set_title('匹配前倾向得分分布', fontsize=14, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# (2) 匹配后倾向得分分布
ax2 = axes[0, 1]
ax2.hist(treat_matched['propensity_score'], bins=20, alpha=0.5, label='处理组', color='red')
ax2.hist(control_matched['propensity_score'], bins=20, alpha=0.5, label='对照组', color='blue')
ax2.set_xlabel('倾向得分', fontsize=12)
ax2.set_ylabel('频数', fontsize=12)
ax2.set_title('匹配后倾向得分分布', fontsize=14, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

# (3) 标准化偏差比较（匹配前后）
ax3 = axes[1, 0]
vars_names = required_vars
before_bias = [abs(r['标准化偏差(%)']) for r in comparison_results_before]
after_bias = [abs(r['标准化偏差(%)']) for r in comparison_results_after]

x = np.arange(len(vars_names))
width = 0.35

ax3.bar(x - width/2, before_bias, width, label='匹配前', color='orange', alpha=0.7)
ax3.bar(x + width/2, after_bias, width, label='匹配后', color='green', alpha=0.7)
ax3.set_xlabel('变量', fontsize=12)
ax3.set_ylabel('标准化偏差 (%)', fontsize=12)
ax3.set_title('匹配前后标准化偏差比较', fontsize=14, fontweight='bold')
ax3.set_xticks(x)
ax3.set_xticklabels(vars_names, rotation=45, ha='right', fontsize=9)
ax3.legend()
ax3.axhline(y=10, color='red', linestyle='--', alpha=0.5, label='10%阈值')
ax3.grid(True, alpha=0.3, axis='y')

# (4) 匹配质量评估
ax4 = axes[1, 1]
ax4.axis('off')

# 显示为表格
table_data = []
for i, var in enumerate(vars_names):
    table_data.append([
        var,
        f'{before_bias[i]:.2f}%',
        f'{after_bias[i]:.2f}%',
        f'{before_bias[i] - after_bias[i]:.2f}%'
    ])

table = ax4.table(cellText=table_data,
                  colLabels=['变量', '匹配前', '匹配后', '偏差减少'],
                  cellLoc='center',
                  loc='center')
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1.2, 1.5)
ax4.set_title('匹配质量评估表', fontsize=14, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig('PSM匹配结果可视化_四个变量.png', dpi=300, bbox_inches='tight')
print("可视化图表已保存至: PSM匹配结果可视化_四个变量.png")

# 12. 导出匹配后的数据
print("\n[步骤12] 导出匹配后的数据...")
df_matched_full = df_matched.drop(columns=['matched_id', 'propensity_score'])
df_matched_full.to_excel("匹配后数据集_四个变量.xlsx", index=False)
print("匹配后数据已保存至: 匹配后数据集_四个变量.xlsx")

# 13. 生成详细报告
print("\n[步骤13] 生成详细分析报告...")

avg_std_bias_before = np.mean([abs(r['标准化偏差(%)']) for r in comparison_results_before])
avg_std_bias_after = np.mean([abs(r['标准化偏差(%)']) for r in comparison_results_after])

report = f"""
{'='*80}
PSM倾向得分匹配分析报告（四个匹配变量）
{'='*80}

一、分析设定
------------
基期年份：2009年
匹配变量：{', '.join(required_vars)}
匹配方法：1:1有放回匹配
卡尺：倾向得分标准差的0.25倍（{caliper_original:.6f}）

二、样本情况
------------
原始处理组样本数：{len(treat_group)}
原始对照组样本数：{len(control_group)}
匹配成功数量：{len(matched_treated)}
匹配成功率：{len(matched_treated)/len(treated)*100:.2f}%

三、匹配效果评估
--------------
"""

for var in required_vars:
    before = next(r for r in comparison_results_before if r['变量'] == var)
    after = next(r for r in comparison_results_after if r['变量'] == var)

    report += f"""
{var}:
  匹配前标准化偏差：{before['标准化偏差(%)']:.2f}%
  匹配后标准化偏差：{after['标准化偏差(%)']:.2f}%
  偏差减少：{abs(before['标准化偏差(%)']) - abs(after['标准化偏差(%)']):.2f}%
  匹配前p值：{before['p值']:.4f}
  匹配后p值：{after['p值']:.4f}
"""

report += f"""
四、总体评估
----------
匹配前平均标准化偏差：{avg_std_bias_before:.2f}%
匹配后平均标准化偏差：{avg_std_bias_after:.2f}%
平均偏差减少：{avg_std_bias_before - avg_std_bias_after:.2f}%

"""

if avg_std_bias_after < 10:
    report += "✓ 匹配质量优秀：匹配后平均标准化偏差小于10%，说明处理组和对照组在匹配变量上具有良好的平衡性。\n"
elif avg_std_bias_after < 20:
    report += "△ 匹配质量良好：匹配后平均标准化偏差小于20%，处理组和对照组在匹配变量上基本平衡。\n"
else:
    report += "✗ 匹配质量需改进：匹配后平均标准化偏差较大，建议考虑调整匹配变量或匹配方法。\n"

report += f"""
{'='*80}
分析完成时间：{pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
{'='*80}
"""

with open("PSM分析报告_四个变量.txt", 'w', encoding='utf-8') as f:
    f.write(report)

print("详细分析报告已保存至: PSM分析报告_四个变量.txt")
print("\n" + report)

print("\n" + "="*80)
print("PSM分析完成！所有结果文件已生成。")
print("="*80)
