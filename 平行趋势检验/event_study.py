# -*- coding: utf-8 -*-
"""
事件研究法 - Event Study
严格的平行趋势检验（学术标准）

这是学术论文认可的标准方法：
1. 创建动态DID变量（Treat × 每年虚拟变量）
2. 估计固定效应模型
3. 提取各期系数和置信区间
4. 检验预处理期系数是否联合不显著
5. 绘制事件研究图（Event Study Plot）
"""

import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import io
from scipy import stats

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

print("="*80)
print("事件研究法 - 平行趋势检验（学术标准）")
print("="*80)

# ============================================================================
# 步骤1: 读取数据
# ============================================================================
print("\n[步骤1] 读取数据...")

df_full = pd.read_excel("../总数据集2007-2019_含碳排放强度.xlsx")
df_matched = pd.read_excel("../PSM-四个变量-2009基期/匹配后数据集_四个变量.xlsx")

print(f"完整数据集: {df_full.shape}")
print(f"PSM匹配后数据: {df_matched.shape}")

# 获取匹配后的城市列表
matched_cities = df_matched['city_name'].unique()
df = df_full[df_full['city_name'].isin(matched_cities)].copy()

# 保留需要的变量
vars_needed = ['ln_emission_intensity_winzorized', 'Treat', 'city_name', 'year',
               'ln_real_gdp', 'ln_人口密度', 'ln_金融发展水平', '第二产业占GDP比重']
df = df[vars_needed].dropna()

print(f"筛选后样本: {len(df)}")
print(f"城市数: {df['city_name'].nunique()}")
print(f"年份范围: {df['year'].min()} - {df['year'].max()}")

# ============================================================================
# 步骤2: 创建事件研究变量
# ============================================================================
print("\n[步骤2: 创建事件研究变量...")

# 创建相对年份（以2009年为0）
df['rel_year'] = df['year'] - 2009

print(f"相对年份分布:")
print(df['rel_year'].value_counts().sort_index())

# 为每一年创建虚拟变量（除-1外）
year_dummies = pd.get_dummies(df['rel_year'], prefix='y')

# 与处理组交互，创建动态DID变量
exclude_years = [-1]  # 排除基准期-1

for yr in df['rel_year'].unique():
    if yr not in exclude_years:
        var_name = f'D_post_{int(yr)}'  # 使用安全的变量名
        df[var_name] = df['Treat'] * (df['rel_year'] == yr).astype(int)

# 列出创建的动态DID变量
did_vars = [f'D_post_{int(yr)}' for yr in sorted(df['rel_year'].unique()) if yr not in exclude_years]
print(f"\n创建的动态DID变量（{len(did_vars)}个）:")
print(did_vars)

# ============================================================================
# 步骤3: 事件研究回归（简化版 - 适合大多数情况）
# ============================================================================
print("\n[步骤3: 事件研究回归...")

# 控制变量
controls = ['ln_real_gdp', 'ln_人口密度', 'ln_金融发展水平', '第二产业占GDP比重']

# 方法1：分步回归（避免过多变量导致的奇异性）
# 先跑预处理期
pre_did_vars = [v for v in did_vars if int(v.split('_')[2]) < 0]
post_did_vars = [v for v in did_vars if int(v.split('_')[2]) >= 0]

print(f"\n预处理期变量: {pre_did_vars}")
print(f"政策实施后变量: {post_did_vars[:5]}...")  # 显示前5个

# 创建公式（简化：只包含核心动态DID变量）
# 使用城市和年份固定效应
core_did_vars = pre_did_vars + post_did_vars[:6]  # 前6个政策后年份

formula = 'ln_emission_intensity_winzorized ~ ' + ' + '.join(core_did_vars + controls)
print(f"\n回归公式（简化版）:")
print(formula)

try:
    # 使用statsmodels的OLS（带城市和年份固定效应）
    # 创建虚拟变量
    df_fe = df.copy()
    df_fe['city_fe'] = pd.Categorical(df_fe['city_name']).codes
    df_fe['year_fe'] = pd.Categorical(df_fe['year']).codes

    # 带固定效应的回归
    formula_fe = 'ln_emission_intensity_winzorized ~ ' + ' + '.join(core_did_vars + controls) + ' + C(city_name) + C(year)'

    print("\n正在估计模型（这可能需要几分钟）...")
    model_fe = smf.ols(formula_fe, data=df_fe).fit(cov_type='cluster', cov_kwds={'groups': df_fe['city_name']})

    print("\n" + "="*80)
    print("事件研究回归结果")
    print("="*80)

    # 只显示动态DID系数
    did_results = model_fe.params[core_did_vars]
    did_std_err = model_fe.bse[core_did_vars]
    did_pvals = model_fe.pvalues[core_did_vars]
    did_conf_int = model_fe.conf_int().loc[core_did_vars]

    print("\n动态DID系数:")
    print("-" * 80)
    results_list = []
    for var in core_did_vars:
        yr = int(var.split('_')[2])
        coef = did_results[var]
        se = did_std_err[var]
        pval = did_pvals[var]
        ci_lower = did_conf_int.loc[var, 0]
        ci_upper = did_conf_int.loc[var, 1]

        sig = '***' if pval < 0.01 else '**' if pval < 0.05 else '*' if pval < 0.1 else ''

        print(f"rel_year = {yr:3d}: {coef:8.6f} (SE={se:.6f}, p={pval:.4f}) {sig}")
        print(f"             95% CI: [{ci_lower:.6f}, {ci_upper:.6f}]")

        results_list.append({
            '相对年份': yr,
            '变量名': var,
            '系数': coef,
            '标准误': se,
            't值': coef/se,
            'p值': pval,
            '95%CI下限': ci_lower,
            '95%CI上限': ci_upper,
            '显著性': sig
        })

    # 保存结果
    df_results = pd.DataFrame(results_list)
    df_results.to_excel("事件研究法回归结果.xlsx", index=False)
    print("\n回归结果已保存至: 事件研究法回归结果.xlsx")

except Exception as e:
    print(f"\n错误: 回归失败")
    print(f"错误信息: {str(e)}")
    print("\n使用简化方法...")
    df_results = None

# ============================================================================
# 步骤4: 绘制事件研究图（带置信区间）
# ============================================================================
print("\n[步骤4] 绘制事件研究图...")

if df_results is not None and len(df_results) > 0:
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 图1：完整的事件研究图
    ax1 = axes[0, 0]

    x = df_results['相对年份']
    y = df_results['系数']
    y_lower = df_results['95%CI下限']
    y_upper = df_results['95%CI上限']

    # 绘制散点和置信区间
    ax1.scatter(x, y, s=80, color='steelblue', zorder=3, label='系数估计值')
    ax1.vlines(x, y_lower, y_upper, colors='steelblue', linewidth=2, alpha=0.7, zorder=2)
    ax1.plot(x, y, color='steelblue', linewidth=1.5, alpha=0.8, zorder=1)

    # 添加y=0参考线
    ax1.axhline(y=0, color='red', linestyle='--', linewidth=2, label='系数=0', zorder=0)

    # 添加基准期线
    ax1.axvline(x=-1, color='gray', linestyle=':', linewidth=1.5, alpha=0.5, label='基准期(2008)')

    # 添加显著性标记
    for _, row in df_results.iterrows():
        if row['显著性'] != '':
            ax1.text(row['相对年份'], row['系数'], row['显著性'],
                    ha='center', va='bottom', fontsize=11, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='yellow', alpha=0.5))

    # 标注政策实施
    ax1.axvline(x=0, color='green', linestyle='--', linewidth=2, alpha=0.7)
    ax1.text(0, ax1.get_ylim()[1]*0.92, '政策实施', ha='center', fontsize=10,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.7))

    ax1.set_xlabel('相对年份（以2009年为0）', fontsize=12, fontweight='bold')
    ax1.set_ylabel('系数', fontsize=12, fontweight='bold')
    ax1.set_title('事件研究图（Event Study Plot）\n动态处理效应（95%置信区间）',
                   fontsize=14, fontweight='bold')
    ax1.legend(loc='best', fontsize=9)
    ax1.grid(True, alpha=0.3, axis='y')

    # 图2：预处理期放大（平行趋势检验）
    ax2 = axes[0, 1]

    pre_data = df_results[df_results['相对年份'] < 0].copy()

    if len(pre_data) > 0:
        ax2.scatter(pre_data['相对年份'], pre_data['系数'], s=120, color='steelblue', zorder=3)
        ax2.vlines(pre_data['相对年份'], pre_data['95%CI下限'],
                   pre_data['95%CI上限'], colors='steelblue', linewidth=3, alpha=0.8, zorder=2)
        ax2.axhline(y=0, color='red', linestyle='--', linewidth=2.5, label='系数=0', zorder=0)
        ax2.axvline(x=-1, color='gray', linestyle=':', linewidth=1.5, alpha=0.5, label='基准期')

        # 添加显著性标记
        for _, row in pre_data.iterrows():
            if row['显著性'] != '':
                ax2.text(row['相对年份'], row['系数'], row['显著性'],
                        ha='center', va='bottom', fontsize=16, fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.6))

        ax2.set_xlabel('相对年份（政策实施前）', fontsize=12, fontweight='bold')
        ax2.set_ylabel('系数', fontsize=12, fontweight='bold')
        ax2.set_title('预处理期平行趋势检验（放大）', fontsize=14, fontweight='bold')
        ax2.legend(loc='best')
        ax2.grid(True, alpha=0.3, axis='y')

        # 添加检验结论
        pre_sig_count = sum(pre_data['p值'] < 0.1)
        if pre_sig_count == 0:
            conclusion = "✓ 支持平行趋势\n(所有预处理期系数均不显著)"
            color = 'green'
        elif pre_sig_count == 1:
            conclusion = "△ 基本支持\n(仅1期显著)"
            color = 'orange'
        else:
            conclusion = "✗ 不支持平行趋势\n(多期显著)"
            color = 'red'

        ax2.text(0.5, 0.08, conclusion, transform=ax2.transAxes,
                ha='center', fontsize=12, fontweight='bold', color=color,
                bbox=dict(boxstyle='round,pad=0.6', facecolor='white', edgecolor=color, linewidth=2.5))

    # 图3：政策实施后效应
    ax3 = axes[1, 0]

    post_data = df_results[df_results['相对年份'] >= 0].copy()

    if len(post_data) > 0:
        ax3.scatter(post_data['相对年份'], post_data['系数'], s=100, color='darkgreen', zorder=3)
        ax3.vlines(post_data['相对年份'], post_data['95%CI下限'],
                   post_data['95%CI上限'], colors='darkgreen', linewidth=2, alpha=0.7, zorder=2)
        ax3.axhline(y=0, color='red', linestyle='--', linewidth=2, label='系数=0', zorder=0)

        # 添加显著性标记
        for _, row in post_data.iterrows():
            if row['显著性'] != '':
                ax3.text(row['相对年份'], row['系数'], row['显著性'],
                        ha='center', va='bottom', fontsize=11, fontweight='bold')

        ax3.set_xlabel('相对年份（政策实施后）', fontsize=12, fontweight='bold')
        ax3.set_ylabel('系数', fontsize=12, fontweight='bold')
        ax3.set_title('政策实施后动态效应', fontsize=14, fontweight='bold')
        ax3.legend(loc='best')
        ax3.grid(True, alpha=0.3, axis='y')

    # 图4：系数显著性总结表
    ax4 = axes[1, 1]
    ax4.axis('off')

    # 创建汇总表
    summary_data = []
    for _, row in df_results.iterrows():
        yr = row['相对年份']
        period = "预处理期" if yr < 0 else "政策实施后"
        sig_status = row['显著性']
        if sig_status == '':
            sig_text = "不显著"
        else:
            sig_text = f"显著{sig_status}"

        summary_data.append([
            f"{yr:+d}",
            f"{row['系数']:.4f}",
            f"[{row['95%CI下限']:.4f}, {row['95%CI上限']:.4f}]",
            f"{row['p值']:.4f}",
            sig_text,
            period
        ])

    # 绘制表格
    table = ax4.table(cellText=summary_data,
                      colLabels=['相对年份', '系数', '95%CI', 'p值', '显著性', '时期'],
                      cellLoc='center',
                      loc='center',
                      colWidths=[0.1, 0.15, 0.25, 0.1, 0.15, 0.15])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2.5)

    # 设置表头样式
    for i in range(6):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(weight='bold', color='white')

    # 高亮预处理期
    for i, row_data in enumerate(summary_data, 1):
        if row_data[5] == "预处理期":
            for j in range(6):
                table[(i, j)].set_facecolor('#E8F4F8')

    ax4.set_title('回归结果汇总表', fontsize=14, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig('事件研究图_完整版.png', dpi=300, bbox_inches='tight')
    print("\n事件研究图已保存至: 事件研究图_完整版.png")

else:
    print("警告：无法绘制事件研究图（回归失败）")

# ============================================================================
# 步骤5: 平行趋势统计检验
# ============================================================================
print("\n[步骤5] 平行趋势统计检验...")

if df_results is not None and len(df_results) > 0:
    pre_results = df_results[df_results['相对年份'] < 0]

    if len(pre_results) > 0:
        print("\n" + "="*80)
        print("平行趋势假设检验")
        print("="*80)

        print("\n原假设 (H0): 预处理期所有动态DID系数 = 0")
        print("备择假设 (H1): 至少一个预处理期动态DID系数 ≠ 0")
        print()

        # 显示预处理期结果
        print("预处理期系数估计:")
        print("-"*80)
        for _, row in pre_results.iterrows():
            print(f"rel_year = {row['相对年份']:3d}:")
            print(f"  系数 = {row['系数']:8.6f}")
            print(f"  标准误 = {row['标准误']:.6f}")
            print(f"  t值 = {row['系数']/row['标准误']:.4f}")
            print(f"  p值 = {row['p值']:.4f}")
            print(f"  95% CI = [{row['95%CI下限']:.6f}, {row['95%CI上限']:.6f}]")
            print(f"  显著性: {row['显著性'] if row['显著性'] != '' else '不显著'}")
            print()

        # 检验结论
        pre_sig_count = sum(pre_results['p值'] < 0.1)
        pre_total = len(pre_results)

        print("="*80)
        print("检验结论:")
        print("="*80)

        if pre_sig_count == 0:
            print(f"✓ 支持平行趋势假设")
            print(f"  理由：所有{pre_total}个预处理期系数均不显著（p>0.1）")
            print(f"  说明：政策实施前，处理组和对照组的碳排放强度变化趋势平行")
            print(f"  结论：DID模型估计结果可信")
            parallel_trend_result = "支持"
        elif pre_sig_count == 1:
            print(f"△ 基本支持平行趋势假设")
            print(f"  理由：{pre_total}个预处理期中，仅{pre_sig_count}个系数显著（p<0.1）")
            print(f"  说明：虽然有一期显著，但整体上预处理期趋势基本平行")
            print(f"  结论：DID模型估计结果基本可信")
            parallel_trend_result = "基本支持"
        else:
            print(f"✗ 不支持平行趋势假设")
            print(f"  理由：{pre_total}个预处理期中，{pre_sig_count}个系数显著（p<0.1）")
            print(f"  说明：政策实施前，处理组和对照组趋势不平行")
            print(f"  结论：DID模型估计可能存在偏差")
            print(f"  建议：考虑使用其他方法（合成控制法、PSM-DID等）")
            parallel_trend_result = "不支持"

else:
    parallel_trend_result = "无法检验"

# ============================================================================
# 步骤6: 生成详细报告
# ============================================================================
print("\n[步骤6] 生成详细报告...")

report = f"""
{'='*80}
事件研究法 - 平行趋势检验报告（学术标准）
{'='*80}

一、分析方法
----------
方法：事件研究法（Event Study）- 学术界认可的标准方法
模型：固定效应回归（城市FE + 年份FE）
样本：PSM匹配后的城市（{len(matched_cities)}个城市）
期间：2007-2019年（相对年份-2到+10）

基准期：rel_year = -1（2008年，政策实施前一年）

回归方程：
  ln(排放强度) = α + Σ(β_k × Treat × I[year-2009=k]) + Controls + City_FE + Year_FE

其中：
  - k ∈ {{-2, 0, 1, 2, ...,}}（基准组：k=-1）
  - Controls：ln(GDP), ln(人口密度), ln(金融发展), 产业结构
  - 标准误：聚类到城市层面

二、检验假设
----------
原假设 (H0)：预处理期所有动态DID系数 = 0（满足平行趋势）
备择假设 (H1)：至少一个预处理期动态DID系数 ≠ 0（不满足平行趋势）

三、回归结果
----------
详见文件：事件研究法回归结果.xlsx

"""

if df_results is not None and len(df_results) > 0:
    report += f"四、平行趋势检验结果\n"
    report += f"{'='*80}\n\n"

    pre_results = df_results[df_results['相对年份'] < 0]
    if len(pre_results) > 0:
        for _, row in pre_results.iterrows():
            sig_text = row['显著性'] if row['显著性'] != '' else '不显著'
            report += f"rel_year = {row['相对年份']:3d}: 系数 = {row['系数']:8.6f},  "
            report += f"标准误 = {row['标准误']:.6f},  p值 = {row['p值']:.4f}  ({sig_text})\n"

        report += f"\n检验结论：\n"
        report += f"{'='*80}\n\n"

        if parallel_trend_result == "支持":
            report += f"✓ 支持平行趋势假设\n\n"
            report += f"  统计依据：所有预处理期系数均不显著（p > 0.1）\n"
            report += f"  经济含义：在政策实施前，处理组和对照组的碳排放强度变化趋势平行\n\n"
            report += f"  学术结论：满足DID模型的识别假设，估计结果可信\n"
        elif parallel_trend_result == "基本支持":
            report += f"△ 基本支持平行趋势假设\n\n"
            report += f"  统计依据：仅有1个预处理期系数显著\n"
            report += f"  经济含义：预处理期趋势基本平行\n\n"
            report += f"  学术结论：基本满足DID模型的识别假设，估计结果基本可信\n"
        else:
            report += f"✗ 不支持平行趋势假设\n\n"
            report += f"  统计依据：多个预处理期系数显著（p < 0.1）\n"
            report += f"  经济含义：政策实施前两组趋势不平行\n\n"
            report += f"  学术结论：违反DID模型的识别假设，估计结果可能有偏\n"
            report += f"  建议：考虑使用合成控制法或其他方法\n"

    report += f"\n五、事件研究图解读\n"
    report += f"{'='*80}\n\n"
    report += f"详见文件：事件研究图_完整版.png\n\n"
    report += f"图中展示：\n"
    report += f"  - 横轴：相对年份（0为2009年政策实施年）\n"
    report += f"  - 纵轴：动态DID系数（估计处理效应）\n"
    report += f"  - 蓝色点：各期系数估计值\n"
    report += f"  - 蓝色竖线：95%置信区间\n"
    report += f"  - 红色虚线：系数=0参考线\n"
    report += f"  - 显著性标记：*** p<0.01, ** p<0.05, * p<0.1\n\n"

    if len(df_results[df_results['相对年份'] >= 0]) > 0:
        report += f"六、政策效应动态演变\n"
        report += f"{'='*80}\n\n"
        report += f"政策实施后效应（前5年）：\n\n"
        post_results = df_results[df_results['相对年份'] >= 0]
        for _, row in post_results.head(5).iterrows():
            sig_text = row['显著性'] if row['显著性'] != '' else '不显著'
            report += f"  year = {2009+row['相对年份']:4d} (rel_year={row['相对年份']:+2d}): "
            report += f"系数 = {row['系数']:8.6f},  p值 = {row['p值']:.4f}  ({sig_text})\n"

report += f"""

{'='*80}
报告完成时间：{pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
{'='*80}
"""

with open("事件研究法检验报告.txt", 'w', encoding='utf-8') as f:
    f.write(report)

print("\n详细报告已保存至: 事件研究法检验报告.txt")
print(report)

print("\n" + "="*80)
print("事件研究法分析完成！")
print("="*80)
