# -*- coding: utf-8 -*-
"""
PSM后DID回归分析
基于PSM匹配结果进行加权双重差分分析

分析步骤：
1. 读取PSM匹配后的数据
2. 计算每个城市在匹配中的权重
3. 将权重应用到完整数据集（2007-2019）
4. 进行固定效应回归（城市+年份）
5. 标准误聚类到城市层面
"""

import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import statsmodels.api as sm
from linearmodels.panel import PanelOLS
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import io

# 设置标准输出编码
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

print("="*80)
print("PSM后DID回归分析")
print("="*80)

# ============================================================================
# 步骤1: 读取数据
# ============================================================================
print("\n[步骤1] 读取数据...")

# 读取完整数据集
df_full = pd.read_excel("../总数据集2007-2019_含碳排放强度.xlsx")
print(f"完整数据集形状: {df_full.shape}")
print(f"年份范围: {df_full['year'].min()} - {df_full['year'].max()}")

# 读取PSM匹配后的数据
df_matched = pd.read_excel("../PSM-四个变量-2009基期/匹配后数据集_四个变量.xlsx")
print(f"PSM匹配后数据形状: {df_matched.shape}")
print(f"处理组样本数: {(df_matched['Treat']==1).sum()}")
print(f"对照组样本数: {(df_matched['Treat']==0).sum()}")

# ============================================================================
# 步骤2: 计算每个城市的匹配权重
# ============================================================================
print("\n[步骤2] 计算城市匹配权重...")

# 2.1 计算每个城市在匹配中出现的次数
city_counts = df_matched['city_name'].value_counts().reset_index()
city_counts.columns = ['city_name', 'match_count']

print(f"\n匹配后涉及的城市数量: {len(city_counts)}")
print(f"\n匹配次数统计:")
print(city_counts['match_count'].describe())

# 2.2 计算权重
# 处理组（Treat=1）：权重默认为1
# 对照组（Treat=0）：
#   - 未匹配上的城市：权重为0（剔除）
#   - 匹配上的城市：权重=它被选中的次数

weights_dict = {}

# 获取所有匹配中的城市
matched_cities = set(df_matched['city_name'])

# 获取处理组城市（Treat=1，来自2009年基期）
treated_cities_2009 = df_matched[df_matched['Treat'] == 1]['city_name'].unique()

# 获取对照组城市（Treat=0，来自2009年基期）
control_cities_2009 = df_matched[df_matched['Treat'] == 0]['city_name'].unique()

print(f"\n处理组城市数量（2009年）: {len(treated_cities_2009)}")
print(f"对照组城市数量（2009年）: {len(control_cities_2009)}")

# 为所有城市分配权重
for city in matched_cities:
    count = df_matched[df_matched['city_name'] == city].shape[0]

    if city in treated_cities_2009:
        # 处理组：权重为1
        weights_dict[city] = 1
    elif city in control_cities_2009:
        # 对照组：权重为匹配次数
        weights_dict[city] = count
    else:
        # 不在2009年基期的城市（不应该出现）
        weights_dict[city] = 0

# 创建权重数据框
df_weights = pd.DataFrame(list(weights_dict.items()), columns=['city_name', 'psm_weight'])
print(f"\n权重统计:")
print(df_weights['psm_weight'].describe())

print(f"\n权重分布:")
print(df_weights['psm_weight'].value_counts().sort_index())

# ============================================================================
# 步骤3: 将权重应用到完整数据集
# ============================================================================
print("\n[步骤3] 将权重应用到完整数据集...")

# 将权重合并到完整数据集
df_analysis = df_full.merge(df_weights, on='city_name', how='left')

# 未匹配上的城市权重设为0
df_analysis['psm_weight'] = df_analysis['psm_weight'].fillna(0)

# 筛选权重大于0的样本
df_analysis_weighted = df_analysis[df_analysis['psm_weight'] > 0].copy()

print(f"\n应用权重前的样本数: {len(df_analysis)}")
print(f"应用权重后的样本数: {len(df_analysis_weighted)}")
print(f"剔除的样本数: {len(df_analysis) - len(df_analysis_weighted)}")

print(f"\n加权后样本的时间分布:")
print(df_analysis_weighted['year'].value_counts().sort_index())

# 检查变量是否存在
required_vars = ['ln_emission_intensity_winzorized', 'ln_real_gdp', 'ln_人口密度',
                 'ln_金融发展水平', '第二产业占GDP比重', 'Treat', 'Post', 'DID']

missing_vars = [var for var in required_vars if var not in df_analysis_weighted.columns]
if missing_vars:
    print(f"\n警告：缺少以下变量: {missing_vars}")
    print("可用变量:", df_analysis_weighted.columns.tolist())
    sys.exit(1)

print(f"\n所有必需变量均存在")

# 删除缺失值
df_analysis_final = df_analysis_weighted[required_vars + ['city_name', 'year', 'psm_weight']].dropna()
print(f"\n删除缺失值后样本数: {len(df_analysis_final)}")

# ============================================================================
# 步骤4: 描述性统计
# ============================================================================
print("\n[步骤4] 描述性统计...")

# 按处理组和时期分组统计
desc_stats = df_analysis_final.groupby(['Treat', 'Post']).agg({
    'ln_emission_intensity_winzorized': ['count', 'mean', 'std'],
    'ln_real_gdp': 'mean',
    'ln_人口密度': 'mean',
    'ln_金融发展水平': 'mean',
    '第二产业占GDP比重': 'mean'
}).round(4)

print("\n按处理组和时期的描述性统计:")
print(desc_stats)

# 保存描述性统计
desc_stats.to_excel("描述性统计.xlsx")
print("\n描述性统计已保存至: 描述性统计.xlsx")

# ============================================================================
# 步骤5: 固定效应回归（城市+年份，标准误聚类到城市层面）
# ============================================================================
print("\n[步骤5: 固定效应回归...")
print("模型: ln_emission_intensity_winzorized = α + β·DID + γ·Controls + City_FE + Year_FE")
print("标准误: 聚类到城市层面")

# 准备数据：设置多级索引
df_panel = df_analysis_final.copy()
df_panel = df_panel.set_index(['city_name', 'year'])

# 定义模型公式
formula = 'ln_emission_intensity_winzorized ~ DID + ln_real_gdp + ln_人口密度 + ln_金融发展水平 + 第二产业占GDP比重'

# 使用linearmodels进行固定效应回归
try:
    # 城市固定效应 + 年份固定效应
    model = PanelOLS.from_formula(
        formula + ' + EntityEffects + TimeEffects',
        data=df_panel,
        weights=df_panel['psm_weight']
    )

    # 聚类标准误到城市层面
    results = model.fit(cov_type='clustered', cluster_entity=True)

    print("\n" + "="*80)
    print("回归结果")
    print("="*80)
    print(results)

    # 保存回归结果
    with open("回归结果.txt", 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("PSM后DID回归结果\n")
        f.write("="*80 + "\n\n")
        f.write("模型设定:\n")
        f.write(f"  因变量: ln_emission_intensity_winzorized\n")
        f.write(f"  核心解释变量: DID (Treat × Post)\n")
        f.write(f"  控制变量: ln_real_gdp, ln_人口密度, ln_金融发展水平, 第二产业占GDP比重\n")
        f.write(f"  固定效应: 城市固定效应 + 年份固定效应\n")
        f.write(f"  标准误: 聚类到城市层面\n")
        f.write(f"  样本权重: PSM匹配权重\n")
        f.write(f"\n{results}\n")

    print("\n回归结果已保存至: 回归结果.txt")

    # 导出回归结果表格
    summary_df = pd.DataFrame({
        '变量': results.params.index,
        '系数': results.params.values,
        '标准误': results.std_errors.values,
        't值': results.tstats.values,
        'p值': results.pvalues.values,
        '95%置信区间下限': results.conf_int()[0].values,
        '95%置信区间上限': results.conf_int()[1].values
    })

    # 添加星号表示显著性
    def add_stars(p):
        if p < 0.01:
            return '***'
        elif p < 0.05:
            return '**'
        elif p < 0.1:
            return '*'
        else:
            return ''

    summary_df['显著性'] = summary_df['p值'].apply(add_stars)

    summary_df.to_excel("回归结果表格.xlsx", index=False)
    print("回归结果表格已保存至: 回归结果表格.xlsx")

    print("\n核心结果:")
    print("-"*80)
    did_coef = results.params['DID']
    did_se = results.std_errors['DID']
    did_pval = results.pvalues['DID']

    print(f"DID系数: {did_coef:.6f}")
    print(f"标准误: {did_se:.6f}")
    print(f"t值: {did_coef/did_se:.4f}")
    print(f"p值: {did_pval:.4f}")

    if did_pval < 0.01:
        print("结论: 在1%水平上显著 ***")
    elif did_pval < 0.05:
        print("结论: 在5%水平上显著 **")
    elif did_pval < 0.1:
        print("结论: 在10%水平上显著 *")
    else:
        print("结论: 不显著")

except Exception as e:
    print(f"\n错误: 固定效应回归失败")
    print(f"错误信息: {str(e)}")
    print("\n尝试使用OLS回归（不带固定效应）...")

    # 备用方案：使用statsmodels的OLS
    X = df_analysis_final[['DID', 'ln_real_gdp', 'ln_人口密度', 'ln_金融发展水平', '第二产业占GDP比重']]
    X = sm.add_constant(X)
    y = df_analysis_final['ln_emission_intensity_winzorized']
    weights = df_analysis_final['psm_weight']

    model_ols = sm.WLS(y, X, weights=weights)
    results_ols = model_ols.fit(cov_type='cluster', cov_kwds={'groups': df_analysis_final['city_name']})

    print(results_ols.summary())

# ============================================================================
# 步骤6: 可视化
# ============================================================================
print("\n[步骤6] 生成可视化...")

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# (1) 权重分布
ax1 = axes[0, 0]
ax1.hist(df_weights['psm_weight'], bins=range(1, df_weights['psm_weight'].max()+2),
         alpha=0.7, color='steelblue', edgecolor='black')
ax1.set_xlabel('PSM匹配权重', fontsize=12)
ax1.set_ylabel('城市数量', fontsize=12)
ax1.set_title('城市PSM匹配权重分布', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3, axis='y')

# (2) 样本时间分布
ax2 = axes[0, 1]
year_counts = df_analysis_final['year'].value_counts().sort_index()
ax2.plot(year_counts.index, year_counts.values, marker='o', linewidth=2, markersize=8, color='coral')
ax2.set_xlabel('年份', fontsize=12)
ax2.set_ylabel('样本数', fontsize=12)
ax2.set_title('各年份样本数分布', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.axvline(x=2009, color='red', linestyle='--', alpha=0.5, label='PSM基期')
ax2.legend()

# (3) 处理组vs对照组碳排放强度趋势
ax3 = axes[1, 0]
for treat_val in [0, 1]:
    data = df_analysis_final[df_analysis_final['Treat'] == treat_val]
    trend = data.groupby('year')['ln_emission_intensity_winzorized'].mean()
    label = '处理组' if treat_val == 1 else '对照组'
    ax3.plot(trend.index, trend.values, marker='o', label=label, linewidth=2, markersize=6)

ax3.set_xlabel('年份', fontsize=12)
ax3.set_ylabel('ln(碳排放强度)', fontsize=12)
ax3.set_title('处理组与对照组碳排放强度趋势', fontsize=14, fontweight='bold')
ax3.legend()
ax3.grid(True, alpha=0.3)
ax3.axvline(x=2009, color='red', linestyle='--', alpha=0.5, label='政策基期')

# (4) 平行趋势检验（预处理期）
ax4 = axes[1, 1]
pre_data = df_analysis_final[df_analysis_final['year'] < 2009]
for treat_val in [0, 1]:
    data = pre_data[pre_data['Treat'] == treat_val]
    trend = data.groupby('year')['ln_emission_intensity_winzorized'].mean()
    label = '处理组' if treat_val == 1 else '对照组'
    ax4.plot(trend.index, trend.values, marker='o', label=label, linewidth=2, markersize=6)

ax4.set_xlabel('年份', fontsize=12)
ax4.set_ylabel('ln(碳排放强度)', fontsize=12)
ax4.set_title('平行趋势检验（2009年前）', fontsize=14, fontweight='bold')
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('PSM后DID分析可视化.png', dpi=300, bbox_inches='tight')
print("可视化图表已保存至: PSM后DID分析可视化.png")

# ============================================================================
# 步骤7: 生成分析报告
# ============================================================================
print("\n[步骤7] 生成分析报告...")

report = f"""
{'='*80}
PSM后DID回归分析报告
{'='*80}

一、数据准备
----------
1. PSM匹配结果（2009年基期）:
   - 处理组样本数: {len(treated_cities_2009)}
   - 对照组样本数: {len(control_cities_2009)}
   - 匹配后城市总数: {len(df_weights)}

2. 权重设置:
   - 处理组城市: 权重 = 1
   - 对照组城市: 权重 = 匹配次数（{df_weights['psm_weight'].min():.0f} - {df_weights['psm_weight'].max():.0f}）

3. 最终分析样本:
   - 时间范围: 2007-2019年
   - 样本量: {len(df_analysis_final)}
   - 城市数: {df_analysis_final['city_name'].nunique()}

二、模型设定
----------
因变量: ln_emission_intensity_winzorized

核心解释变量:
  - DID (Treat × Post): 双重差分项

控制变量:
  - ln_real_gdp: 实际GDP对数
  - ln_人口密度: 人口密度对数
  - ln_金融发展水平: 金融发展水平对数
  - 第二产业占GDP比重: 第二产业结构

固定效应:
  - 城市固定效应 (EntityEffects)
  - 年份固定效应 (TimeEffects)

标准误聚类:
  - 聚类到城市层面

样本权重:
  - PSM匹配权重

三、描述性统计
----------
详见文件: 描述性统计.xlsx

四、回归结果
----------
详见文件: 回归结果.txt 和 回归结果表格.xlsx

五、可视化
----------
详见文件: PSM后DID分析可视化.png

{'='*80}
分析完成时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
{'='*80}
"""

with open("PSM后DID分析报告.txt", 'w', encoding='utf-8') as f:
    f.write(report)

print("\n分析报告已保存至: PSM后DID分析报告.txt")
print(report)

print("\n" + "="*80)
print("PSM后DID分析完成！所有结果文件已生成。")
print("="*80)
