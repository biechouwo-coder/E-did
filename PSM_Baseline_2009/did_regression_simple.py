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
print("多时点DID回归分析 - 碳排放强度")
print("=" * 80)

# 1. 读取数据
print("\n步骤1: 读取PSM匹配后的数据...")
df = pd.read_excel('matched_data_2007-2019_complete.xlsx')
print(f"数据集形状: {df.shape}")
print(f"年份范围: {df['year'].min()} - {df['year'].max()}")
print(f"城市数: {df['city_name'].nunique()}")

# 2. 准备变量
print("\n步骤2: 准备回归变量...")

# 因变量
y_var = 'ln_carbon_intensity'

# 控制变量
control_vars = ['ln_real_gdp', 'ln_人口密度', 'ln_金融发展水平', '第二产业占GDP比重']

# DID变量
did_var = 'DID'

print(f"因变量: {y_var}")
print(f"DID变量: {did_var}")
print(f"控制变量: {', '.join(control_vars)}")

# 删除缺失值
df_clean = df.dropna(subset=[y_var, did_var] + control_vars).copy()
print(f"\n删除缺失值后样本量: {len(df_clean)}")

# 3. 双向固定效应变换（Within Transformation）
print("\n步骤3: 应用双向固定效应变换...")

# 先减去城市均值
for var in [y_var, did_var] + control_vars:
    city_means = df_clean.groupby('city_name')[var].transform('mean')
    df_clean[f'{var}_city demeaned'] = df_clean[var] - city_means

# 再减去年份均值（从城市去中心化的数据上）
for var in [y_var, did_var] + control_vars:
    year_means = df_clean.groupby('year')[f'{var}_city demeaned'].transform('mean')
    df_clean[f'{var}_demeaned'] = df_clean[f'{var}_city demeaned'] - year_means

print("[OK] 双向固定效应变换完成")

# 4. 构建回归矩阵
print("\n步骤4: 构建回归模型...")

# 准备回归变量（去中心化后的）
y = df_clean[f'{y_var}_demeaned'].values

# DID变量
X_did = df_clean[f'{did_var}_demeaned'].values.reshape(-1, 1)

# 控制变量
X_controls = df_clean[[f'{var}_demeaned' for var in control_vars]].values

# 模型1: 仅包含DID
X1 = np.column_stack([X_did, np.ones(len(X_did))])  # 添加常数项

# 模型2: 包含控制变量
X2 = np.column_stack([X_did, X_controls, np.ones(len(X_did))])

print("模型1: 仅包含DID项和固定效应")
print("模型2: 包含控制变量和固定效应")

# 5. OLS回归
def ols_regression(X, y):
    """OLS回归"""
    # X'X
    XtX = np.dot(X.T, X)
    # X'y
    Xty = np.dot(X.T, y)
    # (X'X)^(-1) X'y
    beta = np.dot(np.linalg.inv(XtX), Xty)
    # 预测值
    y_pred = np.dot(X, beta)
    # 残差
    residuals = y - y_pred
    # 残差平方和
    rss = np.sum(residuals ** 2)
    # 总平方和
    tss = np.sum((y - y.mean()) ** 2)
    # R²
    r2 = 1 - rss / tss
    # 自由度
    n, k = X.shape
    dof = n - k
    # 残差标准误
    sigma2 = rss / dof
    # 标准误
    var_covar = sigma2 * np.linalg.inv(XtX)
    std_errors = np.sqrt(np.diag(var_covar))
    # t统计量
    t_stats = beta / std_errors
    # p值（双尾）
    p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), dof))

    return {
        'beta': beta,
        'std_errors': std_errors,
        't_stats': t_stats,
        'p_values': p_values,
        'r2': r2,
        'residuals': residuals,
        'y_pred': y_pred,
        'nobs': n,
        'dof': dof
    }

print("\n步骤5: 运行OLS回归...")

# 模型1
result1 = ols_regression(X1, y)
print("\n模型1结果:")
print(f"  DID系数: {result1['beta'][0]:.4f}")
print(f"  标准误: {result1['std_errors'][0]:.4f}")
print(f"  t统计量: {result1['t_stats'][0]:.4f}")
print(f"  p值: {result1['p_values'][0]:.4f}")
print(f"  R2: {result1['r2']:.4f}")

# 模型2
result2 = ols_regression(X2, y)
print("\n模型2结果:")
print(f"  DID系数: {result2['beta'][0]:.4f}")
print(f"  标准误: {result2['std_errors'][0]:.4f}")
print(f"  t统计量: {result2['t_stats'][0]:.4f}")
print(f"  p值: {result2['p_values'][0]:.4f}")
print(f"  R2: {result2['r2']:.4f}")

# 控制变量系数
print("\n控制变量系数（模型2）:")
for i, var in enumerate(control_vars):
    idx = i + 1  # DID是第0个变量
    sig = '***' if result2['p_values'][idx] < 0.001 else '**' if result2['p_values'][idx] < 0.01 else '*' if result2['p_values'][idx] < 0.05 else ''
    print(f"  {var}: {result2['beta'][idx]:.4f} (SE={result2['std_errors'][idx]:.4f}, t={result2['t_stats'][idx]:.4f}, p={result2['p_values'][idx]:.4f}) {sig}")

# 6. 计算经济效应
print("\n步骤6: 计算经济效应...")
did_coef = result2['beta'][0]
did_se = result2['std_errors'][0]
did_pval = result2['p_values'][0]

# 对数DID系数转换为百分比效应
percent_effect = (np.exp(did_coef) - 1) * 100
lower_ci = did_coef - 1.96 * did_se
upper_ci = did_coef + 1.96 * did_se
percent_lower = (np.exp(lower_ci) - 1) * 100
percent_upper = (np.exp(upper_ci) - 1) * 100

print(f"\nDID效应解释（模型2 - 完整模型）:")
print(f"  对数DID系数: {did_coef:.4f}")
print(f"  标准误: {did_se:.4f}")
print(f"  t统计量: {result2['t_stats'][0]:.4f}")
print(f"  p值: {did_pval:.4f}")
print(f"  95%置信区间: [{lower_ci:.4f}, {upper_ci:.4f}]")
print(f"  百分比效应: {percent_effect:.2f}%")
print(f"  95% CI: [{percent_lower:.2f}%, {percent_upper:.2f}%]")
print(f"  解释: 低碳试点政策使处理组的碳排放强度相对变化了 {percent_effect:.2f}%")

# 7. 聚类稳健标准误（城市层面）
print("\n步骤7: 计算聚类稳健标准误...")

def cluster_se(df, residuals, X, cluster_col):
    """计算聚类稳健标准误"""
    n, k = X.shape
    clusters = df[cluster_col].unique()
    n_clusters = len(clusters)

    # 计算聚类稳健协方差矩阵
    meat = np.zeros((k, k))
    for cluster in clusters:
        cluster_mask = df[cluster_col].values == cluster
        X_cluster = X[cluster_mask]
        residuals_cluster = residuals[cluster_mask]
        u_cluster = residuals_cluster[:, np.newaxis] * X_cluster
        meat += np.dot(u_cluster.T, u_cluster)

    # 面包
    bread = np.linalg.inv(np.dot(X.T, X))

    # 聚类稳健协方差矩阵
    cov_cluster = (n / (n - 1)) * (n_clusters / (n_clusters - 1)) * np.dot(bread, np.dot(meat, bread))

    # 标准误
    cluster_se = np.sqrt(np.diag(cov_cluster))

    return cluster_se

# 计算聚类稳健标准误
cluster_se_model1 = cluster_se(df_clean, result1['residuals'], X1, 'city_name')
cluster_se_model2 = cluster_se(df_clean, result2['residuals'], X2, 'city_name')

print("\n模型1（聚类稳健标准误）:")
print(f"  DID系数: {result1['beta'][0]:.4f}")
print(f"  聚类SE: {cluster_se_model1[0]:.4f}")
print(f"  t统计量: {result1['beta'][0] / cluster_se_model1[0]:.4f}")

print("\n模型2（聚类稳健标准误）:")
print(f"  DID系数: {result2['beta'][0]:.4f}")
print(f"  聚类SE: {cluster_se_model2[0]:.4f}")
print(f"  t统计量: {result2['beta'][0] / cluster_se_model2[0]:.4f}")

# 8. 汇总结果
results_summary = {
    '模型': ['模型1 (无控制变量)', '模型2 (完整模型)'],
    '样本量': [result1['nobs'], result2['nobs']],
    'R平方': [result1['r2'], result2['r2']],
    'DID系数': [result1['beta'][0], result2['beta'][0]],
    'DID标准误(聚类)': [cluster_se_model1[0], cluster_se_model2[0]],
    'DID t值(聚类)': [result1['beta'][0] / cluster_se_model1[0], result2['beta'][0] / cluster_se_model2[0]],
    '显著性': ['***' if result1['p_values'][0] < 0.001 else '**' if result1['p_values'][0] < 0.01 else '*' if result1['p_values'][0] < 0.05 else '',
               '***' if result2['p_values'][0] < 0.001 else '**' if result2['p_values'][0] < 0.01 else '*' if result2['p_values'][0] < 0.05 else '']
}

results_df = pd.DataFrame(results_summary)
print("\n回归结果对比（聚类稳健标准误）:")
print(results_df.to_string(index=False))

# 9. 保存结果
print("\n步骤9: 保存结果...")

results_df.to_excel('did_regression_summary.xlsx', index=False)
print("[OK] 已保存回归结果汇总: did_regression_summary.xlsx")

with open('did_regression_details.txt', 'w', encoding='utf-8') as f:
    f.write("=" * 80 + "\n")
    f.write("多时点DID回归详细结果\n")
    f.write("=" * 80 + "\n\n")

    f.write("数据来源: PSM匹配后的数据集（83对，166个城市）\n")
    f.write(f"样本期间: 2007-2019年\n")
    f.write(f"总观测数: {result2['nobs']}\n")
    f.write(f"城市数: {df_clean['city_name'].nunique()}\n\n")

    f.write("模型设定:\n")
    f.write("-" * 80 + "\n")
    f.write("Y_it = α + β·DID_it + γ·X_it + μ_i + λ_t + ε_it\n")
    f.write("其中:\n")
    f.write("  - Y_it: ln_carbon_intensity (缩尾后的对数碳排放强度)\n")
    f.write("  - DID_it: Treat_i × Post_t (多时点DID交互项)\n")
    f.write("  - X_it: 控制变量 (ln_real_gdp, ln_人口密度, ln_金融发展水平, 第二产业占GDP比重)\n")
    f.write("  - μ_i: 城市固定效应\n")
    f.write("  - λ_t: 年份固定效应\n")
    f.write("  - 标准误聚类到城市层面\n\n")

    f.write("模型1: 仅包含DID项和固定效应\n")
    f.write("-" * 80 + "\n")
    f.write(f"  样本量: {result1['nobs']}\n")
    f.write(f"  R2: {result1['r2']:.4f}\n")
    f.write(f"  DID系数: {result1['beta'][0]:.4f}\n")
    f.write(f"  标准误（聚类）: {cluster_se_model1[0]:.4f}\n")
    f.write(f"  t统计量: {result1['beta'][0] / cluster_se_model1[0]:.4f}\n\n")

    f.write("模型2: 包含控制变量和固定效应（完整模型）\n")
    f.write("-" * 80 + "\n")
    f.write(f"  样本量: {result2['nobs']}\n")
    f.write(f"  R2: {result2['r2']:.4f}\n")
    f.write(f"  DID系数: {result2['beta'][0]:.4f}\n")
    f.write(f"  标准误（聚类）: {cluster_se_model2[0]:.4f}\n")
    f.write(f"  t统计量: {result2['beta'][0] / cluster_se_model2[0]:.4f}\n")
    f.write(f"  p值: {result2['p_values'][0]:.4f}\n\n")

    f.write("控制变量系数（模型2）:\n")
    f.write("-" * 80 + "\n")
    for i, var in enumerate(control_vars):
        idx = i + 1
        sig = '***' if result2['p_values'][idx] < 0.001 else '**' if result2['p_values'][idx] < 0.01 else '*' if result2['p_values'][idx] < 0.05 else ''
        f.write(f"  {var}:\n")
        f.write(f"    系数: {result2['beta'][idx]:.4f}\n")
        f.write(f"    SE: {result2['std_errors'][idx]:.4f}\n")
        f.write(f"    t值: {result2['t_stats'][idx]:.4f}\n")
        f.write(f"    p值: {result2['p_values'][idx]:.4f} {sig}\n\n")

    f.write("=" * 80 + "\n")
    f.write("经济效应解释\n")
    f.write("=" * 80 + "\n")
    f.write(f"DID系数: {did_coef:.4f}\n")
    f.write(f"标准误（聚类）: {cluster_se_model2[0]:.4f}\n")
    f.write(f"t统计量: {result2['beta'][0] / cluster_se_model2[0]:.4f}\n")
    f.write(f"p值: {did_pval:.4f}\n")
    f.write(f"95%置信区间: [{lower_ci:.4f}, {upper_ci:.4f}]\n")
    f.write(f"百分比效应: {percent_effect:.2f}%\n")
    f.write(f"95% CI: [{percent_lower:.2f}%, {percent_upper:.2f}%]\n")
    f.write(f"解释: 低碳试点政策使处理组的碳排放强度相对变化了 {percent_effect:.2f}%\n")

print("[OK] 已保存详细回归结果: did_regression_details.txt")

# 10. 可视化
print("\n步骤10: 生成可视化图表...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 图1: DID系数对比
ax = axes[0, 0]
models = ['模型1\n(无控制变量)', '模型2\n(完整模型)']
coefficients = [result1['beta'][0], result2['beta'][0]]
errors = [cluster_se_model1[0] * 1.96, cluster_se_model2[0] * 1.96]

colors = ['lightblue', 'lightcoral']
bars = ax.bar(range(len(models)), coefficients, yerr=errors,
              capsize=5, color=colors, alpha=0.7, edgecolor='black')
ax.axhline(y=0, color='red', linestyle='--', linewidth=1)
ax.set_xticks(range(len(models)))
ax.set_xticklabels(models, fontsize=10)
ax.set_ylabel('DID系数（对数值）')
ax.set_title('DID系数对比（聚类稳健标准误，95% CI）')
ax.grid(True, alpha=0.3, axis='y')

# 图2: R2对比
ax = axes[0, 1]
r_squared = [result1['r2'], result2['r2']]
bars = ax.bar(range(len(models)), r_squared, color=colors, alpha=0.7, edgecolor='black')
ax.set_xticks(range(len(models)))
ax.set_xticklabels(models, fontsize=10)
ax.set_ylabel('R2')
ax.set_title('模型拟合优度对比')
ax.grid(True, alpha=0.3, axis='y')

for i, bar in enumerate(bars):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, height + 0.005,
            f'{height:.4f}', ha='center', fontsize=10)

# 图3: 控制变量系数
ax = axes[1, 0]
control_coefs = result2['beta'][1:len(control_vars)+1]
control_errors = result2['std_errors'][1:len(control_vars)+1] * 1.96
control_pvals = result2['p_values'][1:len(control_vars)+1]

y_pos = np.arange(len(control_vars))
bars = ax.barh(y_pos, control_coefs, xerr=control_errors,
               capsize=5, color='lightgreen', alpha=0.7, edgecolor='black')
ax.axvline(x=0, color='red', linestyle='--', linewidth=1)
ax.set_yticks(y_pos)
ax.set_yticklabels(control_vars, fontsize=9)
ax.set_xlabel('系数值（对数值）')
ax.set_title('控制变量系数（模型2，95% CI）')
ax.grid(True, alpha=0.3, axis='x')

# 图4: 事件研究
ax = axes[1, 1]
event_results = []
base_year = 2009

for lead in range(-2, 12):
    year = base_year + lead
    if year < 2007 or year > 2019:
        continue
    year_data = df[df['year'] == year]
    if len(year_data) == 0:
        continue
    treat_mean = year_data[year_data['Treat'] == 1][y_var].mean()
    control_mean = year_data[year_data['Treat'] == 0][y_var].mean()
    diff = treat_mean - control_mean
    event_results.append({
        'relative_year': lead,
        'calendar_year': year,
        'treatment_effect': diff
    })

event_df = pd.DataFrame(event_results)
colors = ['gray' if x < -1 else 'green' for x in event_df['relative_year']]
ax.bar(event_df['relative_year'], event_df['treatment_effect'],
       color=colors, alpha=0.7, edgecolor='black')
ax.axvline(x=-1, color='red', linestyle='--', linewidth=2, label='政策实施')
ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
ax.set_xlabel('相对于政策实施的年份')
ax.set_ylabel('处理组-对照组差值（对数值）')
ax.set_title('事件研究：动态处理效应')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('did_regression_visualization.png', dpi=300, bbox_inches='tight')
print("[OK] 已保存可视化图表: did_regression_visualization.png")

# 保存事件研究数据
event_df.to_excel('event_study_results.xlsx', index=False)
print("[OK] 已保存事件研究数据: event_study_results.xlsx")

print("\n" + "=" * 80)
print("DID回归分析完成！")
print("=" * 80)

print("\n主要发现:")
print(f"1. DID系数（完整模型，聚类SE）: {result2['beta'][0]:.4f} ± {cluster_se_model2[0]:.4f}")
print(f"2. t统计量: {result2['beta'][0] / cluster_se_model2[0]:.4f}")
print(f"3. 经济效应: 低碳试点政策使碳排放强度相对变化 {percent_effect:.2f}%")
print(f"   95% CI: [{percent_lower:.2f}%, {percent_upper:.2f}%]")
print(f"4. R2: {result2['r2']:.4f}")
print(f"5. 样本量: {result2['nobs']} 个观测")
print(f"6. 城市数: {df_clean['city_name'].nunique()} 个")
print(f"7. 使用数据: PSM匹配后的83对城市（166个城市）")

print("\n输出文件:")
print("1. did_regression_summary.xlsx - 回归结果汇总")
print("2. did_regression_details.txt - 详细回归结果")
print("3. did_regression_visualization.png - 4合1可视化图表")
print("4. event_study_results.xlsx - 事件研究数据")
