# -*- coding: utf-8 -*-
"""
DID双向固定效应回归分析（使用sklearn）

使用LSDV（最小二乘虚拟变量法）实现固定效应模型
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

print("="*70)
print("DID双向固定效应回归分析（sklearn版本）")
print("="*70)

# 步骤1: 读取数据
print("\n步骤1: 读取匹配后的数据...")
df = pd.read_excel('PSM_Analysis/matched_dataset_log_vars.xlsx')
print(f"✓ 读取数据: {df.shape[0]} 行 × {df.shape[1]} 列")
print(f"  年份范围: {df['year'].min()} - {df['year'].max()}")
print(f"  城市数量: {df['city_name'].nunique()}")

# 步骤2: 准备核心变量
print("\n步骤2: 准备核心变量...")

y_var = 'ln_carbon_intensity'
did_var = 'DID_matched'
control_vars = ['ln_pgdp', 'ln_pop_density', 'ln_industrial_advanced', 'ln_fdi_openness']

print(f"\n被解释变量 (Y): {y_var}")
print(f"核心解释变量 (DID): {did_var}")
print(f"控制变量: {', '.join(control_vars)}")

# 检查缺失值
core_vars = [y_var, did_var] + control_vars
for var in core_vars:
    missing = df[var].isnull().sum()
    print(f"  {var}: {missing} 个缺失值")

# 删除缺失值
df_reg = df.dropna(subset=core_vars).copy()
print(f"\n✓ 清理后数据: {len(df_reg)} 行")

# 步骤3: 描述性统计
print("\n" + "="*70)
print("步骤3: 描述性统计")
print("="*70)

print("\n核心变量描述性统计:")
print(df_reg[core_vars].describe())

print("\n按DID_matched分组统计:")
group_stats = df_reg.groupby(did_var)[y_var].agg(['mean', 'count'])
print(group_stats)

# 计算差异
treat_mean = df_reg[df_reg[did_var]==1][y_var].mean()
control_mean = df_reg[df_reg[df_reg==0][y_var].mean()
print(f"\n简单比较（未控制其他因素）:")
print(f"  处理组均值: {treat_mean:.4f}")
print(f"  对照组均值: {control_mean:.4f}")
print(f"  差异: {treat_mean - control_mean:.4f}")

# 步骤4: 创建固定效应虚拟变量
print("\n" + "="*70)
print("Step 4: Creating fixed effects model (LSDV method)")
print("="*70)

# 创建城市固定效应（删除一个作为基准）
city_dummies = pd.get_dummies(df_reg['city_name'], prefix='city', drop_first=True)
city_fe_count = city_dummies.shape[1]
print("  City fixed effects: {} dummy variables".format(city_fe_count))

# 创建年份固定效应（删除一个作为基准）
year_dummies = pd.get_dummies(df_reg['year'], prefix='year', drop_first=True)
year_fe_count = year_dummies.shape[1]
print("  Year fixed effects: {} dummy variables".format(year_fe_count))

# 组合所有变量
X_did = df_reg[[did_var]].values.reshape(-1, 1)
X_control = df_reg[control_vars].values
X_city = city_dummies.values
X_year = year_dummies.values

X = np.hstack([X_did, X_control, X_city, X_year])
y = df_reg[y_var].values

print(f"\n设计矩阵维度:")
print(f"  X: {X.shape} ({X.shape[0]}观测 × {X.shape[1]}变量)")
print(f"    - DID_matched: 1")
print(f"    - 控制变量: {len(control_vars)}")
print(f"    - 城市固定效应: {city_dummies.shape[1]}")
print(f"    - 年份固定效应: {year_dummies.shape[1]}")
print(f"  y: {y.shape}")

# 步骤5: 运行回归
print("\n正在运行OLS回归...")
model = LinearRegression(fit_intercept=True)
model.fit(X, y)

# 预测和R方
y_pred = model.predict(X)
ss_res = np.sum((y - y_pred)**2)
ss_tot = np.sum((y - np.mean(y))**2)
r_squared = 1 - ss_res/ss_tot
n = len(y)
k = X.shape[1]
adj_r_squared = 1 - (1 - r_squared) * (n - 1) / (n - k - 1)

print(f"✓ 回归完成!")

# 计算系数和标准误
# 使用普通最小二乘法计算标准误
X_with_intercept = np.column_stack([np.ones(n), X])
XX_inv = np.linalg.inv(X_with_intercept.T @ X_with_intercept)
residuals = y - y_pred
mse = ss_res / (n - k - 1)

# 方差-协方差矩阵
vcv = mse * XX_inv
se = np.sqrt(np.diag(vcv))

# 提取系数和标准误
coef_intercept = model.intercept_
coef = model.coef_
se_intercept = se[0]
se_coef = se[1:]

# t统计量和p值（双侧检验）
t_intercept = coef_intercept / se_intercept
t_coef = coef / se_coef

from scipy import stats
p_intercept = 2 * (1 - stats.t.cdf(np.abs(t_intercept), n - k - 1))
p_coef = 2 * (1 - stats.t.cdf(np.abs(t_coef), n - k - 1))

# 95%置信区间
t_crit = stats.t.ppf(0.975, n - k - 1)
ci_intercept = (coef_intercept - t_crit * se_intercept, coef_intercept + t_crit * se_intercept)
ci_coef = [(coef[i] - t_crit * se_coef[i], coef[i] + t_crit * se_coef[i]) for i in range(len(coef))]

# 步骤6: 结果解读
print("\n" + "="*70)
print("步骤5: 回归结果")
print("="*70)

# 提取DID系数
did_idx = 0
did_coef_val = coef[did_idx]
did_se_val = se_coef[did_idx]
did_t_val = t_coef[did_idx]
did_p_val = p_coef[did_idx]
did_ci_lower, did_ci_upper = ci_coef[did_idx]

print(f"\n核心解释变量 ({did_var}):")
print(f"  系数: {did_coef_val:.6f}")
print(f"  标准误: {did_se_val:.6f}")
print(f"  t值: {did_t_val:.4f}")
print(f"  p值: {did_p_val:.4f}")

if did_p_val < 0.01:
    sig = "*** (p<0.01)"
elif did_p_val < 0.05:
    sig = "**  (p<0.05)"
elif did_p_val < 0.1:
    sig = "*   (p<0.1)"
else:
    sig = "    (不显著)"

print(f"  显著性: {sig}")
print(f"  95%置信区间: [{did_ci_lower:.6f}, {did_ci_upper:.6f}]")

# 解释系数
print(f"\n系数解释:")
print(f"  在控制了城市固定效应、年份固定效应以及其他控制变量后，")
print(f"  低碳试点政策使碳排放强度变化了 {did_coef_val*100:.2f}%")
print(f"  {sig}")

if did_coef_val < 0:
    print(f"  ✓ 政策显著降低了碳排放强度")
else:
    print(f"  ✗ 政策未显著降低碳排放强度")

# 模型拟合优度
print(f"\n模型拟合优度:")
print(f"  R-squared: {r_squared:.4f}")
print(f"  Adj. R-squared: {adj_r_squared:.4f}")
print(f"  样本量: {n}")

# 步骤7: 控制变量结果
print("\n" + "="*70)
print("步骤6: 控制变量系数")
print("="*70)

# 创建变量名列表
var_names = [did_var] + control_vars + list(city_dummies.columns) + list(year_dummies.columns)

print(f"\n{'变量':<30s} {'系数':>12s} {'标准误':>10s} {'t值':>8s} {'显著性':>10s}")
print("-" * 80)

# DID变量
print(f"{did_var:<30s} {did_coef_val:>12.6f} {did_se_val:>10.4f} {did_t_val:>8.2f} {sig:>10s}")

# 控制变量（前4个）
for i, var in enumerate(control_vars):
    idx = 1 + i
    c = coef[idx]
    s = se_coef[idx]
    t = t_coef[idx]
    p = p_coef[idx]

    if p < 0.01:
        marker = '***'
    elif p < 0.05:
        marker = '** '
    elif p < 0.1:
        marker = '*  '
    else:
        marker = '   '

    print(f"{var:<30s} {c:>12.6f} {s:>10.4f} {t:>8.2f} {marker:>10s}")

# 截距项
print(f"{'常数项':<30s} {coef_intercept:>12.6f} {se_intercept:>10.4f} {t_intercept:>8.2f}")

# 步骤8: 保存结果
print("\n" + "="*70)
print("步骤7: 保存结果")
print("="*70)

# 创建结果DataFrame
results_data = []
results_data.append({
    '变量': 'DID_matched',
    '系数': did_coef_val,
    '标准误': did_se_val,
    't值': did_t_val,
    'p值': did_p_val,
    '显著性': sig.strip()
})

for i, var in enumerate(control_vars):
    idx = 1 + i
    p = p_coef[idx]
    if p < 0.01:
        marker = '***'
    elif p < 0.05:
        marker = '**'
    elif p < 0.1:
        marker = '*'
    else:
        marker = ''

    results_data.append({
        '变量': var,
        '系数': coef[idx],
        '标准误': se_coef[idx],
        't值': t_coef[idx],
        'p值': p,
        '显著性': marker
    })

results_df = pd.DataFrame(results_data)

# 保存Excel
output_file = 'DID_Regression_Results.xlsx'
with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
    results_df.to_excel(writer, sheet_name='回归结果', index=False)
    df_reg.describe().to_excel(writer, sheet_name='描述性统计', index=False)

    # 保存完整模型结果
    full_results = pd.DataFrame({
        '变量': ['常数项'] + var_names,
        '系数': [coef_intercept] + list(coef),
        '标准误': [se_intercept] + list(se_coef),
        't值': [t_intercept] + list(t_coef),
        'p值': [p_intercept] + list(p_coef)
    })
    full_results.to_excel(writer, sheet_name='完整系数', index=False)

print(f"✓ 回归结果已保存至: {output_file}")

# 保存文本报告
report_file = 'DID_Regression_Report.txt'
with open(report_file, 'w', encoding='utf-8') as f:
    f.write("="*70 + "\n")
    f.write("DID双向固定效应回归分析报告\n")
    f.write("="*70 + "\n\n")

    f.write("一、模型设定\n")
    f.write("-"*70 + "\n")
    f.write(f"数据集: matched_dataset_log_vars.xlsx\n")
    f.write(f"被解释变量: {y_var} (碳排放强度)\n")
    f.write(f"核心解释变量: {did_var} (匹配后的DID变量)\n")
    f.write(f"控制变量: {', '.join(control_vars)}\n")
    f.write(f"固定效应: 城市固定效应 + 年份固定效应\n")
    f.write(f"估计方法: 最小二乘虚拟变量法 (LSDV)\n\n")

    f.write("二、回归结果\n")
    f.write("-"*70 + "\n")
    f.write(f"样本量: {n}\n")
    f.write(f"R-squared: {r_squared:.4f}\n")
    f.write(f"Adj. R-squared: {adj_r_squared:.4f}\n\n")

    f.write("核心解释变量结果:\n")
    f.write(f"  系数: {did_coef_val:.6f}\n")
    f.write(f"  标准误: {did_se_val:.6f}\n")
    f.write(f"  t值: {did_t_val:.4f}\n")
    f.write(f"  p值: {did_p_val:.4f}\n")
    f.write(f"  95%CI: [{did_ci_lower:.6f}, {did_ci_upper:.6f}]\n\n")

    f.write("结论:\n")
    if did_p_val < 0.05:
        f.write(f"  低碳试点政策对碳排放强度有显著影响（{sig}）\n")
        f.write(f"  政策使碳排放强度{did_coef_val*100:+.2f}%\n")
    else:
        f.write(f"  低碳试点政策对碳排放强度无显著影响\n")

    f.write("\n三、系数解释\n")
    f.write("-"*70 + "\n")
    f.write(f"DID系数 = {did_coef_val:.6f}\n")
    f.write(f"解释: 在控制了其他因素后，低碳试点政策使碳排放强度")
    if did_coef_val < 0:
        f.write(f"降低了 {abs(did_coef_val)*100:.2f}%\n")
    else:
        f.write(f"增加了 {did_coef_val*100:.2f}%\n")

    f.write("\n四、建议\n")
    f.write("-"*70 + "\n")
    f.write("1. 这是基准回归结果，应作为论文主表\n")
    f.write("2. 已使用PSM匹配数据，样本更具代表性\n")
    f.write("3. 使用对数控制变量，符合统计假设\n")
    f.write("4. 后续可进行稳健性检验\n")

print(f"✓ 回归报告已保存至: {report_file}")

print("\n" + "="*70)
print("回归分析完成！")
print("="*70)

print(f"\n主要发现:")
print(f"  低碳试点政策使碳排放强度 {did_coef_val*100:+.2f}% {sig}")
if did_p_val < 0.05:
    print(f"  ✓ 政策效果显著！")
else:
    print(f"  ⚠️  政策效果不显著")

print(f"\n输出文件:")
print(f"  1. {output_file} - 回归结果Excel表")
print(f"  2. {report_file} - 回归报告文本")
