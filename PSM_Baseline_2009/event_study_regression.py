import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 80)
print("事件研究分析 - 回归方法（手动实现）")
print("=" * 80)

# 1. 读取数据
print("\n步骤1: 读取数据...")
df = pd.read_excel('matched_data_2007-2019_complete.xlsx')
print(f"数据集形状: {df.shape}")

# 2. 准备变量
print("\n步骤2: 准备回归变量...")
y_var = 'ln_carbon_intensity'
control_vars = ['ln_real_gdp', 'ln_人口密度', 'ln_金融发展水平', '第二产业占GDP比重']

# 创建城市和年份固定效应
df['city_fe'] = df['city_name'].astype('category')
df['year_fe'] = df['year'].astype('category')

# 创建相对年份变量（相对于2009年政策实施）
df['relative_year'] = df['year'] - 2009

print(f"因变量: {y_var}")
print(f"控制变量: {', '.join(control_vars)}")

# 3. 创建交互项（年份 × 处理组）
print("\n步骤3: 创建年份×处理组交互项...")
print("基准期: 2008年 (相对年份=-1)")

# 定义事件年份映射
event_years = {
    2007: -2,   # pre_2
    2009: 0,    # post_0
    2010: 1,    # post_1
    2011: 2,    # post_2
    2012: 3,    # post_3
    2013: 4,    # post_4
    2014: 5,    # post_5
    2015: 6,    # post_6
    2016: 7,    # post_7
    2017: 8,    # post_8
    2018: 9,    # post_9
    2019: 10    # post_10
}

# 创建交互项
for year_val, rel_year in event_years.items():
    col_name = f'post_{rel_year}' if rel_year >= 0 else f'pre_{abs(rel_year)}'
    df[col_name] = ((df['year'] == year_val) & (df['Treat'] == 1)).astype(int)

interact_terms = [f'pre_2'] + [f'post_{i}' for i in range(0, 11)]
print(f"交互项: {', '.join(interact_terms)}")

# 4. 删除缺失值
df_clean = df.dropna(subset=[y_var] + control_vars + interact_terms).copy()
print(f"\n删除缺失值后样本量: {len(df_clean)}")

# 5. 实现固定效应变换（Within Transformation）
print("\n步骤4: 应用固定效应变换...")
print("  城市固定效应 -> 去组内均值")

def within_transformation_group(data, group_col, vars_transform):
    """对指定变量进行组内去均值"""
    for var in vars_transform:
        group_means = data.groupby(group_col)[var].transform('mean')
        data[f'{var}_within'] = data[var] - group_means
    return data

# 先城市去均值
vars_to_transform = [y_var] + control_vars + interact_terms
df_clean = within_transformation_group(df_clean, 'city_name', vars_to_transform)

# 后年份去均值
for var in vars_to_transform:
    year_means = df_clean.groupby('year')[f'{var}_within'].transform('mean')
    df_clean[f'{var}_demeaned'] = df_clean[f'{var}_within'] - year_means

# 构建回归矩阵
y = df_clean[f'{y_var}_demeaned'].values
X_interact = df_clean[[f'{term}_demeaned' for term in interact_terms]].values
X_control = df_clean[[f'{var}_demeaned' for var in control_vars]].values
X = np.column_stack([X_interact, X_control])
X = np.column_stack([np.ones(X.shape[0]), X])  # 添加常数项

# 添加statsmodels的基本功能（手动实现）
def ols_with_cluster_se(y, X, cluster_groups):
    """OLS回归 + 聚类稳健标准误"""
    n, k = X.shape

    # OLS系数
    XtX = X.T @ X
    Xty = X.T @ y
    beta = np.linalg.inv(XtX) @ Xty

    # 残差
    residuals = y - X @ beta
    mse = np.sum(residuals**2) / (n - k)

    # 普通标准误
    XtX_inv = np.linalg.inv(XtX)
    var_ols = mse * XtX_inv
    se_ols = np.sqrt(np.diag(var_ols))

    # 聚类稳健标准误
    unique_clusters = np.unique(cluster_groups)
    G = len(unique_clusters)
    k_mat = XtX_inv

    # 计算聚类稳健协方差矩阵
    meat = np.zeros((k, k))
    for g in unique_clusters:
        mask = cluster_groups == g
        X_g = X[mask]
        e_g = residuals[mask].reshape(-1, 1)
        meat += X_g.T @ e_g @ e_g.T @ X_g

    # 调整因子
    adjustment = (n - 1) / (n - k) * G / (G - 1)
    var_cluster = adjustment * k_mat @ meat @ k_mat
    se_cluster = np.sqrt(np.diag(var_cluster))

    # t统计量和p值
    t_stats = beta / se_cluster
    p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), df=G - 1))

    # R方
    y_pred = X @ beta
    ss_total = np.sum((y - np.mean(y))**2)
    ss_residual = np.sum(residuals**2)
    r2 = 1 - ss_residual / ss_total

    return {
        'beta': beta,
        'std_errors': se_cluster,
        't_stats': t_stats,
        'p_values': p_values,
        'r2': r2,
        'nobs': n,
        'residuals': residuals
    }

# 6. 运行事件研究回归
print("\n步骤5: 运行事件研究回归...")
print("模型: Y_it = α + Στ_k (Year_k × Treat_i) + γ·X_it + μ_i + λ_t + ε_it")

# 创建城市聚类组（转换为数值）
city_labels = df_clean['city_name'].astype('category').cat.codes.values

result = ols_with_cluster_se(y, X, city_labels)

print("\n回归结果:")
print(f"  样本量: {result['nobs']}")
print(f"  R2: {result['r2']:.4f}")

# 7. 提取交互项结果
print("\n步骤6: 提取动态处理效应...")
event_results = []

# 创建年份列表（与interact_terms对应）
year_list = [2007] + [2009 + j for j in range(0, 11)]

for i, term in enumerate(interact_terms):
    coef = result['beta'][i + 1]  # +1 因为第0列是常数项
    se = result['std_errors'][i + 1]
    t_stat = result['t_stats'][i + 1]
    p_val = result['p_values'][i + 1]
    ci_lower = coef - 1.96 * se
    ci_upper = coef + 1.96 * se

    rel_year = event_years[year_list[i]]
    calendar_year = year_list[i]

    sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else ''

    event_results.append({
        'relative_year': rel_year,
        'calendar_year': calendar_year,
        'coefficient': coef,
        'std_error': se,
        't_statistic': t_stat,
        'p_value': p_val,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'significant': sig
    })

    sig_mark = ' ' if not sig else f' {sig}'
    print(f"  t={rel_year:2d} (年{calendar_year}): 系数={coef:7.4f}, SE={se:.4f}, t={t_stat:5.2f}, p={p_val:.4f}{sig_mark}")

event_df = pd.DataFrame(event_results)

# 8. 平行趋势检验
print("\n步骤7: 平行趋势检验...")
pre_trend = event_df[event_df['relative_year'] < 0]
pre_sig = pre_trend[pre_trend['p_value'] < 0.1]

if len(pre_sig) == 0:
    print("  [OK] 政策前期所有系数均不显著 (p >= 0.10)")
    print("  [OK] 平行趋势假设得到支持")
else:
    print(f"  [警告] {len(pre_sig)} 个政策前期系数显著")
    for _, row in pre_sig.iterrows():
        print(f"    相对年份 {row['relative_year']}: 系数={row['coefficient']:.4f}, p={row['p_value']:.4f}")
    print("  [警告] 平行趋势假设可能不成立")

# 9. 绘制事件研究图
print("\n步骤8: 绘制事件研究图...")

fig, ax = plt.subplots(figsize=(14, 7))

# 基准线
ax.axvline(x=-1, color='red', linestyle='--', linewidth=2, label='政策实施 (2009)')
ax.axvline(x=0, color='red', linestyle=':', linewidth=1, alpha=0.5)
ax.axhline(y=0, color='black', linestyle='-', linewidth=1.5, label='基准线 (系数=0)')

# 数据点
x_coords = event_df['relative_year'].values
y_coords = event_df['coefficient'].values
y_errors_lower = y_coords - event_df['ci_lower'].values
y_errors_upper = event_df['ci_upper'].values - y_coords

# 带误差线的散点图
colors = ['gray' if x < 0 else 'darkgreen' for x in x_coords]
ax.errorbar(x_coords, y_coords, yerr=[y_errors_lower, y_errors_upper],
            fmt='o', capsize=5, capthick=2, linewidth=2,
            color='steelblue', ecolor='steelblue', markersize=10,
            markerfacecolor='white', markeredgewidth=2.5, zorder=5)

# 添加显著性标记
for _, row in event_df.iterrows():
    if row['significant']:
        y_offset = 0.012 if row['coefficient'] >= 0 else -0.015
        ax.text(row['relative_year'], row['coefficient'] + y_offset,
               row['significant'], ha='center', va='bottom',
               fontsize=14, fontweight='bold', color='red')

ax.set_xlabel('相对于政策实施的年份', fontsize=13)
ax.set_ylabel('动态处理效应系数（对数值）', fontsize=13)
ax.set_title('事件研究：平行趋势检验与动态处理效应\n（含控制变量、固定效应、聚类稳健标准误）',
             fontsize=14, fontweight='bold')
ax.legend(fontsize=11, loc='upper left')
ax.grid(True, alpha=0.3, axis='y', linestyle='--')
ax.set_xticks(range(-2, 11))
ax.set_xticklabels([f'{i}' if i != 0 else '0' for i in range(-2, 11)], fontsize=10)

plt.tight_layout()
plt.savefig('event_study_regression.png', dpi=300, bbox_inches='tight')
print("[OK] 已保存事件研究图: event_study_regression.png")

# 10. 保存结果
event_df.to_excel('event_study_regression_results.xlsx', index=False)
print("[OK] 已保存事件研究数据: event_study_regression_results.xlsx")

# 11. 生成文本报告
report = []
report.append("=" * 80)
report.append("事件研究分析报告")
report.append("=" * 80)
report.append("")
report.append("一、模型设定")
report.append("-" * 80)
report.append("模型: Y_it = α + Στ_k (Year_k × Treat_i) + γ·X_it + μ_i + λ_t + ε_it")
report.append("")
report.append("  - Y_it: 因变量 (ln_carbon_intensity)")
report.append("  - Year_k × Treat_i: 年份与处理组交互项（动态处理效应）")
report.append("  - X_it: 控制变量 (ln_real_gdp, ln_人口密度, ln_金融发展水平, 第二产业占GDP比重)")
report.append("  - μ_i: 城市固定效应")
report.append("  - λ_t: 年份固定效应")
report.append("  - 标准误: 聚类到城市层面")
report.append("")
report.append(f"基准期: 2008年 (相对年份=-1)")
report.append(f"样本量: {result['nobs']}")
report.append(f"R2: {result['r2']:.4f}")
report.append("")
report.append("二、动态处理效应")
report.append("-" * 80)
for _, row in event_df.iterrows():
    period = "政策前" if row['relative_year'] < 0 else "政策后"
    sig_mark = row['significant'] if row['significant'] else ""
    report.append(f"t={row['relative_year']:2d} (年{row['calendar_year']}): "
                 f"系数={row['coefficient']:7.4f}, SE={row['std_error']:.4f}, "
                 f"t={row['t_statistic']:5.2f}, p={row['p_value']:.4f} {sig_mark}")
report.append("")
report.append("三、平行趋势检验")
report.append("-" * 80)
if len(pre_sig) == 0:
    report.append("结论: 政策前期所有系数均不显著，平行趋势假设得到支持。")
else:
    report.append(f"警告: {len(pre_sig)} 个政策前期系数显著，平行趋势假设可能不成立。")
report.append("")
report.append("=" * 80)

report_text = "\n".join(report)
print("\n" + report_text)

with open('event_study_report.txt', 'w', encoding='utf-8') as f:
    f.write(report_text)
print("\n[OK] 已保存事件研究报告: event_study_report.txt")

print("\n" + "=" * 80)
print("事件研究分析完成！")
print("=" * 80)

print("\n输出文件:")
print("1. event_study_regression.png - 事件研究图")
print("2. event_study_regression_results.xlsx - 事件研究数据")
print("3. event_study_report.txt - 分析报告")
