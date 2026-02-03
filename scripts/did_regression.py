# -*- coding: utf-8 -*-
"""
DID双向固定效应回归分析

第一阶段：准备工作
1. 数据集：matched_dataset_log_vars.xlsx
2. 被解释变量：ln_carbon_intensity（碳排放强度）
3. 核心解释变量：DID_matched
4. 控制变量：ln_pgdp, ln_pop_density, ln_industrial_advanced, ln_fdi_openness

第二阶段：核心回归
1. 双向固定效应模型（城市FE + 年份FE）
2. 聚类标准误：按城市聚类
"""

import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from statsmodels.stats.sandwich import cov_cluster
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

print("="*70)
print("DID双向固定效应回归分析")
print("="*70)

# 步骤1: 读取数据
print("\n步骤1: 读取匹配后的数据...")
df = pd.read_excel('PSM_Analysis/matched_dataset_log_vars.xlsx')
print(f"✓ 读取数据: {df.shape[0]} 行 × {df.shape[1]} 列")
print(f"  年份范围: {df['year'].min()} - {df['year'].max()}")
print(f"  城市数量: {df['city_name'].nunique()}")
print(f"  匹配对数: {(df['matched_treatment']==1).sum() / (df['year'].max() - df['year'].min() + 1)}")

# 步骤2: 确认核心变量
print("\n步骤2: 确认核心变量...")

# 被解释变量
y_var = 'ln_carbon_intensity'
print(f"\n被解释变量 (Y):")
print(f"  {y_var} - 碳排放强度（对数）")

# 核心解释变量
did_var = 'DID_matched'
print(f"\n核心解释变量 (DID):")
print(f"  {did_var} - 匹配后的DID变量")
print(f"  逻辑: 匹配上的试点城市实施政策后为1，其他情况为0")

# 控制变量
control_vars = ['ln_pgdp', 'ln_pop_density', 'ln_industrial_advanced', 'ln_fdi_openness']
print(f"\n控制变量 (Controls):")
for i, var in enumerate(control_vars, 1):
    print(f"  {i}. {var}")

# 检查缺失值
print(f"\n检查核心变量缺失值:")
core_vars = [y_var, did_var] + control_vars
for var in core_vars:
    missing = df[var].isnull().sum()
    print(f"  {var}: {missing} / {len(df)} ({missing/len(df)*100:.1f}%)")

# 删除缺失值
df_reg = df.dropna(subset=core_vars).copy()
print(f"\n✓ 清理后数据: {len(df_reg)} 行（删除了{len(df)-len(df_reg)}行）")

# 步骤3: 描述性统计
print("\n" + "="*70)
print("步骤3: 描述性统计")
print("="*70)

print("\n核心变量描述性统计:")
print(df_reg[core_vars].describe())

# 按DID分组统计
print("\n按DID_matched分组统计:")
print(df_reg.groupby(did_var)[y_var].agg(['mean', 'std', 'count']))

# 步骤4: 双向固定效应回归
print("\n" + "="*70)
print("步骤4: 双向固定效应回归（基准回归）")
print("="*70)

# 构建回归公式
# 使用C()表示分类变量（固定效应）
formula = f"{y_var} ~ {did_var} + {' + ', '.join(control_vars) + '} + C(city_name) + C(year)'

print(f"\n回归模型:")
print(f"  被解释变量: {y_var}")
print(f"  核心解释变量: {did_var}")
print(f"  控制变量: {', '.join(control_vars)}")
print(f"  固定效应: 城市固定效应 + 年份固定效应")

# 运行回归
print(f"\n正在运行回归...")
model = smf.ols(formula, data=df_reg)
results = model.fit(cov_type='cluster', cov_kwds={'groups': df_reg['city_name']})

print(f"✓ 回归完成!")

# 步骤5: 结果解读
print("\n" + "="*70)
print("步骤5: 回归结果")
print("="*70)

# 打印回归结果
print("\n回归结果摘要:")
print(results.summary())

# 提取关键信息
did_coef = results.params[did_var]
did_pval = results.pvalues[did_var]
did_std_err = results.bse[did_var]
did_ci_lower = results.conf_int()[0][0]
did_ci_upper = results.conf_int()[0][1]

print("\n" + "="*70)
print("核心结果解读")
print("="*70)

print(f"\n核心解释变量 ({did_var}):")
print(f"  系数 (β): {did_coef:.6f}")
print(f"  标准误: {did_std_err:.6f}")
print(f"  t值: {did_coef/did_std_err:.4f}")
print(f"  p值: {did_pval:.4f}")

if did_pval < 0.01:
    sig = "*** (p<0.01)"
elif did_pval < 0.05:
    sig = "**  (p<0.05)"
elif did_pval < 0.1:
    sig = "*   (p<0.1)"
else:
    sig = "    (不显著)"

print(f"  显著性: {sig}")
print(f"  95%置信区间: [{did_ci_lower:.6f}, {did_ci_upper:.6f}]")

# 解释系数
print(f"\n系数解释:")
print(f"  在控制了城市和年份固定效应以及其他控制变量后，")
print(f"  低碳试点政策使碳排放强度{did_coef*100:.2f}%")
print(f"  {sig}")

if did_coef < 0:
    print(f"  ✓ 政策显著降低了碳排放强度")
else:
    print(f"  ✗ 政策未显著降低碳排放强度（或效应为正）")

# 模型拟合优度
print(f"\n模型拟合优度:")
print(f"  R-squared: {results.rsquared:.4f}")
print(f"  Adj. R-squared: {results.rsquared_adj:.4f}")
print(f"  F-statistic: {results.fvalue:.2f}")
print(f"  AIC: {results.aic:.2f}")
print(f"  BIC: {results.bic:.2f}")

# 样本量
print(f"\n样本量:")
print(f"  总观测数: {int(results.nobs)}")
print(f"  城市数: {df_reg['city_name'].nunique()}")
print(f"  年份数: {df_reg['year'].nunique()}")

# 步骤6: 控制变量结果
print("\n" + "="*70)
print("步骤6: 控制变量系数")
print("="*70)

print(f"\n{'变量':<30s} {'系数':>12s} {'标准误':>10s} {'t值':>8s} {'显著性':>10s}")
print("-" * 80)

for var in control_vars:
    if var in results.params:
        coef = results.params[var]
        se = results.bse[var]
        t = coef / se
        pval = results.pvalues[var]

        if pval < 0.01:
            sig = '***'
        elif pval < 0.05:
            sig = '** '
        elif pval < 0.1:
            sig = '*  '
        else:
            sig = '   '

        print(f"{var:<30s} {coef:>12.6f} {se:>10.4f} {t:>8.2f} {sig:>10s}")

# 步骤7: 保存结果
print("\n" + "="*70)
print("步骤7: 保存结果")
print("="*70)

# 保存回归结果表（用于论文）
results_table = results.summary2().tables
results_df = results.summary2().tables[1]

# 创建更美观的结果表
summary_df = pd.DataFrame({
    '变量': ['DID_matched'] + control_vars + ['城市固定效应', '年份固定效应', '常数项'],
    '系数': [results.params.get(did_var, np.nan)] +
            [results.params.get(v, np.nan) for v in control_vars] +
            ['Yes', 'Yes', results.params.get('Intercept', np.nan)],
    '标准误': [results.bse.get(did_var, np.nan)] +
              [results.bse.get(v, np.nan) for v in control_vars] +
              ['', '', results.bse.get('Intercept', np.nan)],
    't值': [results.params.get(did_var, np.nan)/results.bse.get(did_var, np.nan) if did_var in results.params else np.nan] +
           [results.params.get(v, np.nan)/results.bse.get(v, np.nan) if v in results.params else np.nan for v in control_vars] +
           ['', '', ''],
})

# 添加显著性标记
def add_sig_marker(row):
    var = row['变量']
    if var == 'DID_matched':
        pval = did_pval
        if pval < 0.01:
            return f"{row['系数']:.6f}***"
        elif pval < 0.05:
            return f"{row['系数']:.6f}**"
        elif pval < 0.1:
            return f"{row['系数']:.6f}*"
        else:
            return f"{row['系数']:.6f}"
    elif var in control_vars:
        pval = results.pvalues.get(var, 1)
        coef = row['系数']
        if pd.notna(coef):
            if pval < 0.01:
                return f"{coef:.6f}***"
            elif pval < 0.05:
                return f"{coef:.6f}**"
            elif pval < 0.1:
                return f"{coef:.6f}*"
            else:
                return f"{coef:.6f}"
    else:
        return row['系数']

summary_df['系数_显著性'] = summary_df.apply(add_sig_marker, axis=1)

# 保存回归结果
output_file = 'DID_Regression_Results.xlsx'
with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
    summary_df.to_excel(writer, sheet_name='回归结果', index=False)
    results.summary2().tables[0].to_excel(writer, sheet_name='模型信息', index=False)
    df_reg.describe().to_excel(writer, sheet_name='描述性统计', index=False)

print(f"✓ 回归结果已保存至: {output_file}")

# 保存文本报告
report_file = 'DID_Regression_Report.txt'
with open(report_file, 'w', encoding='utf-8') as f:
    f.write("="*70 + "\n")
    f.write("DID双向固定效应回归分析报告\n")
    f.write("="*70 + "\n\n")

    f.write("一、模型设定\n")
    f.write("-"*70 + "\n")
    f.write(f"被解释变量: {y_var} (碳排放强度)\n")
    f.write(f"核心解释变量: {did_var} (匹配后的DID变量)\n")
    f.write(f"控制变量: {', '.join(control_vars)}\n")
    f.write(f"固定效应: 城市固定效应 + 年份固定效应\n")
    f.write(f"聚类标准误: 按城市聚类\n\n")

    f.write("二、回归结果\n")
    f.write("-"*70 + "\n")
    f.write(f"样本量: {int(results.nobs)}\n")
    f.write(f"R-squared: {results.rsquared:.4f}\n")
    f.write(f"Adj. R-squared: {results.rsquared_adj:.4f}\n\n")

    f.write("核心解释变量结果:\n")
    f.write(f"  系数: {did_coef:.6f}\n")
    f.write(f"  标准误: {did_std_err:.6f}\n")
    f.write(f"  t值: {did_coef/did_std_err:.4f}\n")
    f.write(f"  p值: {did_pval:.4f}\n")
    f.write(f"  95%CI: [{did_ci_lower:.6f}, {did_ci_upper:.6f}]\n\n")

    f.write("结论:\n")
    if did_pval < 0.05:
        f.write(f"  低碳试点政策对碳排放强度有显著影响（{sig}）\n")
        f.write(f"  政策使碳排放强度{did_coef*100:.2f}%\n")
    else:
        f.write(f"  低碳试点政策对碳排放强度无显著影响\n")

    f.write("\n三、建议\n")
    f.write("-"*70 + "\n")
    f.write("1. 这是基准回归结果，应作为论文主表\n")
    f.write("2. 后续可进行稳健性检验（如更换控制变量、不同匹配方法等）\n")
    f.write("3. 可以进一步分析政策效应的时间动态（动态DID）\n")
    f.write("4. 可以进行异质性分析（分地区、分批次等）\n")

print(f"✓ 回归报告已保存至: {report_file}")

print("\n" + "="*70)
print("回归分析完成！")
print("="*70)

print(f"\n主要发现:")
print(f"  低碳试点政策使碳排放强度 {did_coef*100:+.2f}% {sig}")
if did_pval < 0.05:
    print(f"  ✓ 政策效果显著！")
else:
    print(f"  ⚠️  政策效果不显著")

print(f"\n输出文件:")
print(f"  1. {output_file} - 回归结果Excel表")
print(f"  2. {report_file} - 回归报告文本")
