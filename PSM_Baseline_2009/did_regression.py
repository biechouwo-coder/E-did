import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 80)
print("多时点DID回归分析 - 碳排放强度")
print("=" * 80)

# 1. 读取数据
print("\n步骤1: 读取数据...")
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

# DID变量（交互项Treat × Post）
did_var = 'DID'

print(f"因变量: {y_var}")
print(f"DID变量: {did_var}")
print(f"控制变量: {', '.join(control_vars)}")

# 检查缺失值
print("\n检查变量缺失值:")
for var in [y_var, did_var] + control_vars:
    missing = df[var].isna().sum()
    missing_pct = missing / len(df) * 100
    print(f"  {var}: {missing} ({missing_pct:.2f}%)")

# 删除缺失值
df_clean = df.dropna(subset=[y_var, did_var] + control_vars).copy()
print(f"\n删除缺失值后样本量: {len(df_clean)}")

# 3. 创建城市和年份虚拟变量
print("\n步骤3: 准备固定效应...")
# 创建城市和年份的虚拟变量（用于固定效应）
df_clean['city_fe'] = df_clean['city_name'].astype('category')
df_clean['year_fe'] = df_clean['year'].astype('category')

print(f"城市数量: {df_clean['city_fe'].nunique()}")
print(f"年份数量: {df_clean['year_fe'].nunique()}")

# 4. 构建回归公式
print("\n步骤4: 构建回归模型...")
print("模型设定: Y_it = α + β·DID_it + γ·X_it + μ_i + λ_t + ε_it")
print("其中:")
print("  - DID_it = Treat_i × Post_t (多时点DID交互项)")
print("  - X_it: 控制变量")
print("  - μ_i: 城市固定效应")
print("  - λ_t: 年份固定效应")
print("  - 标准误聚类到城市层面")

# 构建控制变量字符串
controls_str = ' + '.join(control_vars)

# 模型1: 仅包含DID和固定效应
formula1 = f"{y_var} ~ {did_var} + C(city_fe) + C(year_fe)"
print(f"\n模型1公式: {formula1}")

# 模型2: 包含控制变量
formula2 = f"{y_var} ~ {did_var} + {controls_str} + C(city_fe) + C(year_fe)"
print(f"模型2公式: {formula2}")

# 5. 运行回归（使用聚类稳健标准误）
print("\n步骤5: 运行回归...")

# 模型1
print("\n模型1: 仅包含DID项和固定效应（无控制变量）")
model1 = smf.ols(formula1, data=df_clean).fit(cov_type='cluster', cov_kwds={'groups': df_clean['city_name']})
print(model1.summary())

# 模型2
print("\n模型2: 包含控制变量和固定效应（完整模型）")
model2 = smf.ols(formula2, data=df_clean).fit(cov_type='cluster', cov_kwds={'groups': df_clean['city_name']})
print(model2.summary())

# 6. 提取关键结果
print("\n步骤6: 提取回归结果...")

# 计算组内R²
def within_r_squared(model, df, y_var):
    """计算组内R²"""
    y_pred = model.predict(df)
    y_actual = df[y_var].values

    # 计算组内均值
    df_with_mean = df.copy()
    df_with_mean['city_mean'] = df_with_mean.groupby('city_name')[y_var].transform('mean')

    # 组内平方和
    ss_within = ((y_actual - df_with_mean['city_mean']) ** 2).sum()
    ss_resid = ((y_actual - y_pred) ** 2).sum()

    r2_within = 1 - (ss_resid / ss_within)
    return r2_within

r2_within_1 = within_r_squared(model1, df_clean, y_var)
r2_within_2 = within_r_squared(model2, df_clean, y_var)

results_summary = {
    '模型': ['模型1 (无控制变量)', '模型2 (完整模型)'],
    '样本量': [int(model1.nobs), int(model2.nobs)],
    'R平方': [model1.rsquared, model2.rsquared],
    '组内R平方': [r2_within_1, r2_within_2],
    'DID系数': [model1.params[did_var], model2.params[did_var]],
    'DID标准误': [model1.bse[did_var], model2.bse[did_var]],
    'DID t值': [model1.tvalues[did_var], model2.tvalues[did_var]],
    'DID p值': [model1.pvalues[did_var], model2.pvalues[did_var]],
    '显著性': ['***' if model1.pvalues[did_var] < 0.001 else
               '**' if model1.pvalues[did_var] < 0.01 else
               '*' if model1.pvalues[did_var] < 0.05 else
               '不显著' if model1.pvalues[did_var] < 0.1 else '',
               '***' if model2.pvalues[did_var] < 0.001 else
               '**' if model2.pvalues[did_var] < 0.01 else
               '*' if model2.pvalues[did_var] < 0.05 else
               '不显著' if model2.pvalues[did_var] < 0.1 else '']
}

results_df = pd.DataFrame(results_summary)
print("\n回归结果对比:")
print(results_df.to_string(index=False))

# 7. 计算经济效应
print("\n步骤7: 计算经济效应...")
did_coef = model2.params[did_var]
did_se = model2.bse[did_var]
did_pval = model2.pvalues[did_var]

# 对数DID系数转换为百分比效应
percent_effect = (np.exp(did_coef) - 1) * 100
lower_ci = did_coef - 1.96 * did_se
upper_ci = did_coef + 1.96 * did_se
percent_lower = (np.exp(lower_ci) - 1) * 100
percent_upper = (np.exp(upper_ci) - 1) * 100

print(f"\nDID效应解释（模型2 - 完整模型）:")
print(f"  对数DID系数: {did_coef:.4f}")
print(f"  标准误: {did_se:.4f}")
print(f"  t统计量: {model2.tvalues[did_var]:.4f}")
print(f"  p值: {did_pval:.4f}")
print(f"  95%置信区间: [{lower_ci:.4f}, {upper_ci:.4f}]")
print(f"  百分比效应: {percent_effect:.2f}%")
print(f"  95% CI: [{percent_lower:.2f}%, {percent_upper:.2f}%]")
print(f"  解释: 低碳试点政策使处理组的碳排放强度相对变化了 {percent_effect:.2f}%")

# 8. VIF检验（多重共线性）
print("\n步骤8: 多重共线性检验（VIF）...")
# 准备数据（不含固定效应）
X_vif = df_clean[control_vars].copy()
X_vif = sm.add_constant(X_vif)

vif_data = []
for i in range(X_vif.shape[1]):
    vif_value = variance_inflation_factor(X_vif.values, i)
    var_name = X_vif.columns[i]
    if var_name == 'const':
        var_name = '常数项'
    vif_data.append({
        '变量': var_name,
        'VIF': vif_value,
        '多重共线性': '严重' if vif_value > 10 else '中等' if vif_value > 5 else '轻微'
    })

vif_df = pd.DataFrame(vif_data)
print(vif_df.to_string(index=False))

# 9. 保存结果
print("\n步骤9: 保存回归结果...")

# 保存回归结果表格
results_df.to_excel('did_regression_summary.xlsx', index=False)
print("[OK] 已保存回归结果汇总: did_regression_summary.xlsx")

# 保存详细回归结果
with open('did_regression_details.txt', 'w', encoding='utf-8') as f:
    f.write("=" * 80 + "\n")
    f.write("多时点DID回归详细结果\n")
    f.write("=" * 80 + "\n\n")

    f.write("数据来源: PSM匹配后的数据集（83对，166个城市）\n")
    f.write(f"样本期间: 2007-2019年\n")
    f.write(f"总观测数: {int(model2.nobs)}\n\n")

    f.write("模型1: 仅包含DID项和固定效应\n")
    f.write("-" * 80 + "\n")
    f.write(str(model1.summary()) + "\n\n")

    f.write("模型2: 包含控制变量和固定效应\n")
    f.write("-" * 80 + "\n")
    f.write(str(model2.summary()) + "\n\n")

    f.write("=" * 80 + "\n")
    f.write("经济效应解释\n")
    f.write("=" * 80 + "\n")
    f.write(f"DID系数: {did_coef:.4f}\n")
    f.write(f"标准误: {did_se:.4f}\n")
    f.write(f"t统计量: {model2.tvalues[did_var]:.4f}\n")
    f.write(f"p值: {did_pval:.4f}\n")
    f.write(f"95%置信区间: [{lower_ci:.4f}, {upper_ci:.4f}]\n")
    f.write(f"百分比效应: {percent_effect:.2f}%\n")
    f.write(f"95% CI: [{percent_lower:.2f}%, {percent_upper:.2f}%]\n")
    f.write(f"解释: 低碳试点政策使处理组的碳排放强度相对变化了 {percent_effect:.2f}%\n")

print("[OK] 已保存详细回归结果: did_regression_details.txt")

# 10. 生成可视化
print("\n步骤10: 生成可视化图表...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 图1: DID系数对比（带置信区间）
ax = axes[0, 0]
models = ['模型1\n(无控制变量)', '模型2\n(完整模型)']
coefficients = [model1.params[did_var], model2.params[did_var]]
errors = [model1.bse[did_var] * 1.96, model2.bse[did_var] * 1.96]

colors = ['lightblue', 'lightcoral']
bars = ax.bar(range(len(models)), coefficients, yerr=errors,
              capsize=5, color=colors, alpha=0.7, edgecolor='black')
ax.axhline(y=0, color='red', linestyle='--', linewidth=1)
ax.set_xticks(range(len(models)))
ax.set_xticklabels(models, fontsize=10)
ax.set_ylabel('DID系数（对数值）')
ax.set_title('DID系数对比（95%置信区间）')
ax.grid(True, alpha=0.3, axis='y')

# 添加显著性标记
sig_levels = [results_summary['显著性'][0], results_summary['显著性'][1]]
for i, (bar, sig) in enumerate(zip(bars, sig_levels)):
    height = bar.get_height()
    y_offset = errors[i] + 0.005 if height >= 0 else -errors[i] - 0.02
    if sig == '***':
        ax.text(bar.get_x() + bar.get_width()/2, height + y_offset + (0.01 if height >= 0 else -0.01),
                '***', ha='center', fontsize=12, fontweight='bold')
    elif sig == '**':
        ax.text(bar.get_x() + bar.get_width()/2, height + y_offset + (0.01 if height >= 0 else -0.01),
                '**', ha='center', fontsize=12, fontweight='bold')
    elif sig == '*':
        ax.text(bar.get_x() + bar.get_width()/2, height + y_offset + (0.01 if height >= 0 else -0.01),
                '*', ha='center', fontsize=12, fontweight='bold')

# 图2: 模型拟合优度对比
ax = axes[0, 1]
r_squared = [results_df['组内R平方'][0], results_df['组内R平方'][1]]
bars = ax.bar(range(len(models)), r_squared, color=colors, alpha=0.7, edgecolor='black')
ax.set_xticks(range(len(models)))
ax.set_xticklabels(models, fontsize=10)
ax.set_ylabel('组内R²')
ax.set_title('模型拟合优度对比（组内R²）')
ax.grid(True, alpha=0.3, axis='y')

# 添加数值标签
for i, bar in enumerate(bars):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, height + 0.005,
            f'{height:.4f}', ha='center', fontsize=10)

# 图3: 控制变量系数（模型2）
ax = axes[1, 0]
control_coefs = model2.params[control_vars]
control_errors = model2.bse[control_vars] * 1.96
control_pvals = model2.pvalues[control_vars]

y_pos = np.arange(len(control_vars))
bars = ax.barh(y_pos, control_coefs, xerr=control_errors,
               capsize=5, color='lightgreen', alpha=0.7, edgecolor='black')
ax.axvline(x=0, color='red', linestyle='--', linewidth=1)
ax.set_yticks(y_pos)
ax.set_yticklabels(control_vars, fontsize=9)
ax.set_xlabel('系数值（对数值）')
ax.set_title('控制变量系数（模型2，95%置信区间）')
ax.grid(True, alpha=0.3, axis='x')

# 添加显著性标记
for i, (bar, pval) in enumerate(zip(bars, control_pvals)):
    width = bar.get_width()
    sig = '***' if pval < 0.001 else '**' if pval < 0.01 else '*' if pval < 0.05 else ''
    if sig:
        x_offset = 0.002 if width >= 0 else -0.015
        ax.text(width + x_offset, i, sig, va='center', fontsize=10, fontweight='bold')

# 图4: 年份固定效应系数
ax = axes[1, 1]
# 提取年份固定效应
time_effects = model2.params[model2.params.index.str.contains('C(year_fe)')]
if len(time_effects) > 0:
    # 提取年份
    years = []
    for idx in time_effects.index:
        year_str = idx.split('T.')[1] if 'T.' in idx else idx.split('.')[-1]
        # 去掉右括号
        year_str = year_str.rstrip(')')
        years.append(int(year_str))

    coefficients = time_effects.values
    ax.plot(years, coefficients, 'o-', linewidth=2, markersize=6, color='purple')
    ax.axhline(y=0, color='red', linestyle='--', linewidth=1)
    ax.set_xlabel('年份')
    ax.set_ylabel('固定效应系数')
    ax.set_title('年份固定效应')
    ax.grid(True, alpha=0.3)
else:
    ax.text(0.5, 0.5, '年份固定效应\n请查看详细输出', ha='center', va='center',
            fontsize=12, transform=ax.transAxes)

plt.tight_layout()
plt.savefig('did_regression_visualization.png', dpi=300, bbox_inches='tight')
print("[OK] 已保存可视化图表: did_regression_visualization.png")

# 11. 事件研究（Event Study）
print("\n步骤11: 事件研究分析...")

# 计算相对于政策实施的年份
event_study_results = []

# 以2009年为基准年
base_year = 2009

for lead in range(-2, 12):  # 政策前2年到政策后11年
    year = base_year + lead
    if year < 2007 or year > 2019:
        continue

    # 筛选该年的数据
    year_data = df[df['year'] == year]

    if len(year_data) == 0:
        continue

    # 计算该年的处理组-对照组差值
    treat_mean = year_data[year_data['Treat'] == 1][y_var].mean()
    control_mean = year_data[year_data['Treat'] == 0][y_var].mean()
    diff = treat_mean - control_mean

    event_study_results.append({
        'relative_year': lead,
        'calendar_year': year,
        'treatment_effect': diff
    })

event_df = pd.DataFrame(event_study_results)

# 绘制事件研究图
if len(event_df) > 0:
    fig, ax = plt.subplots(figsize=(10, 6))

    # 绘制系数
    ax.axvline(x=-1, color='red', linestyle='--', linewidth=2, label='政策实施')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

    # 绘制各期效应
    colors = ['gray' if x < -1 else 'green' for x in event_df['relative_year']]
    ax.bar(event_df['relative_year'], event_df['treatment_effect'],
           color=colors, alpha=0.7, edgecolor='black')

    ax.set_xlabel('相对于政策实施的年份')
    ax.set_ylabel('处理组-对照组差值（对数值）')
    ax.set_title('事件研究：动态处理效应')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('event_study.png', dpi=300, bbox_inches='tight')
    print("[OK] 已保存事件研究图: event_study.png")

    # 保存事件研究数据
    event_df.to_excel('event_study_results.xlsx', index=False)
    print("[OK] 已保存事件研究数据: event_study_results.xlsx")

print("\n" + "=" * 80)
print("DID回归分析完成！")
print("=" * 80)

print("\n主要发现:")
print(f"1. DID系数（完整模型）: {did_coef:.4f} ({results_summary['显著性'][1]})")
print(f"2. 经济效应: 低碳试点政策使碳排放强度相对变化 {percent_effect:.2f}%")
print(f"   95% CI: [{percent_lower:.2f}%, {percent_upper:.2f}%]")
print(f"3. 组内R²: {results_df['组内R平方'][1]:.4f}")
print(f"4. 样本量: {int(model2.nobs)} 个观测")
print(f"5. 城市数: {df_clean['city_name'].nunique()} 个")
print(f"6. 使用数据: PSM匹配后的83对城市（166个城市）")

print("\n输出文件:")
print("1. did_regression_summary.xlsx - 回归结果汇总")
print("2. did_regression_details.txt - 详细回归结果")
print("3. did_regression_visualization.png - 4合1可视化图表")
print("4. event_study.png - 事件研究图")
print("5. event_study_results.xlsx - 事件研究数据")
