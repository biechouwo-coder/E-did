# -*- coding: utf-8 -*-
"""
事件研究法 - Event Study（手工计算版）
使用numpy/statsmodels直接估计，避开公式解析问题
"""

import pandas as pd
import numpy as np
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
print("事件研究法 - 平行趋势检验（手工计算版）")
print("="*80)

# ============================================================================
# 步骤1: 读取数据
# ============================================================================
print("\n[步骤1] 读取数据...")

df_full = pd.read_excel("../总数据集2007-2019_含碳排放强度.xlsx")
df_matched = pd.read_excel("../PSM-四个变量-2009基期/匹配后数据集_四个变量.xlsx")

matched_cities = df_matched['city_name'].unique()
df = df_full[df_full['city_name'].isin(matched_cities)].copy()

vars_needed = ['ln_emission_intensity_winzorized', 'Treat', 'city_name', 'year']
df = df[vars_needed].dropna()

print(f"样本数: {len(df)}, 城市数: {df['city_name'].nunique()}")

# ============================================================================
# 步骤2: 计算各期处理效应（手工方法）
# ============================================================================
print("\n[步骤2: 计算各期处理效应...")

# 创建相对年份
df['rel_year'] = df['year'] - 2009

years = sorted(df['year'].unique())
results_list = []

for yr in years:
    data_yr = df[df['year'] == yr]

    treat = data_yr[data_yr['Treat'] == 1]['ln_emission_intensity_winzorized']
    control = data_yr[data_yr['Treat'] == 0]['ln_emission_intensity_winzorized']

    # 计算均值差
    diff = treat.mean() - control.mean()

    # 计算标准误（两样本t检验）
    n1, n2 = len(treat), len(control)
    var1, var2 = treat.var(ddof=1), control.var(ddof=1)

    # 合并标准误
    se = np.sqrt(var1/n1 + var2/n2)

    # t统计量和p值
    t_stat = diff / se
    df_deg = n1 + n2 - 2
    p_val = 2 * (1 - stats.t.cdf(abs(t_stat), df_deg))

    # 95%置信区间
    ci_lower = diff - 1.96 * se
    ci_upper = diff + 1.96 * se

    # 显著性标记
    sig = '***' if p_val < 0.01 else '**' if p_val < 0.05 else '*' if p_val < 0.1 else ''

    rel_year = yr - 2009
    is_pre = rel_year < 0

    results_list.append({
        '年份': yr,
        '相对年份': rel_year,
        '时期': '预处理期' if is_pre else '政策实施后',
        '处理组均值': treat.mean(),
        '对照组均值': control.mean(),
        '处理效应': diff,
        '标准误': se,
        't值': t_stat,
        'p值': p_val,
        '95%CI下限': ci_lower,
        '95%CI上限': ci_upper,
        '显著性': sig,
        '样本数': n1 + n2
    })

    # 打印预处理期结果
    if is_pre:
        print(f"rel_year = {rel_year:+3d} (year={yr}): 处理效应 = {diff:.6f}, SE = {se:.6f}, p = {p_val:.4f} {sig}")

df_results = pd.DataFrame(results_list)
df_results.to_excel("事件研究法回归结果.xlsx", index=False)
print("\n结果已保存至: 事件研究法回归结果.xlsx")

# ============================================================================
# 步骤3: 平行趋势检验
# ============================================================================
print("\n[步骤3] 平行趋势检验...")
print("="*80)

pre_results = df_results[df_results['时期'] == '预处理期']

print("\n预处理期处理效应估计:")
print("-"*80)
for _, row in pre_results.iterrows():
    print(f"rel_year = {row['相对年份']:+3d}:")
    print(f"  处理效应 = {row['处理效应']:.6f}")
    print(f"  标准误 = {row['标准误']:.6f}")
    print(f"  t值 = {row['t值']:.4f}")
    print(f"  p值 = {row['p值']:.4f}")
    print(f"  95% CI = [{row['95%CI下限']:.6f}, {row['95%CI上限']:.6f}]")
    print(f"  显著性: {row['显著性'] if row['显著性'] != '' else '不显著'}")
    print()

# 检验结论
pre_sig_count = sum(pre_results['p值'] < 0.1)
pre_total = len(pre_results)

print("="*80)
print("平行趋势假设检验")
print("="*80)
print(f"\n原假设 (H0): 预处理期处理效应 = 0")
print(f"备择假设 (H1): 预处理期处理效应 ≠ 0")
print()

if pre_sig_count == 0:
    print(f"✓ 支持平行趋势假设")
    print(f"  统计依据：所有{pre_total}个预处理期系数均不显著（p > 0.1）")
    print(f"  结论：DID模型估计结果可信")
    parallel_trend_result = "支持"
elif pre_sig_count == 1:
    print(f"△ 基本支持平行趋势假设")
    print(f"  统计依据：{pre_total}个预处理期中，仅{pre_sig_count}个显著（p < 0.1）")
    print(f"  结论：DID模型估计结果基本可信")
    parallel_trend_result = "基本支持"
else:
    print(f"✗ 不支持平行趋势假设")
    print(f"  统计依据：{pre_total}个预处理期中，{pre_sig_count}个显著（p < 0.1）")
    print(f"  结论：DID模型估计可能存在偏差")
    parallel_trend_result = "不支持"

# ============================================================================
# 步骤4: 绘制事件研究图
# ============================================================================
print("\n[步骤4: 绘制事件研究图...")

fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

ax1 = fig.add_subplot(gs[0, :])  # 顶部：完整的事件研究图
ax2 = fig.add_subplot(gs[1, 0])  # 左下：预处理期放大
ax3 = fig.add_subplot(gs[1, 1])  # 右下：结果汇总表

# 图1：完整的事件研究图
x = df_results['相对年份']
y = df_results['处理效应']
y_lower = df_results['95%CI下限']
y_upper = df_results['95%CI上限']

# 绘制系数点和置信区间
colors = ['red' if yr < 0 else 'darkgreen' for yr in df_results['相对年份']]
ax1.scatter(x, y, s=100, c=colors, zorder=3, alpha=0.8)
ax1.vlines(x, y_lower, y_upper, colors=colors, linewidth=2.5, alpha=0.7, zorder=2)
ax1.plot(x, y, color='steelblue', linewidth=1.5, alpha=0.5, zorder=1)

# 添加y=0参考线
ax1.axhline(y=0, color='black', linestyle='--', linewidth=2, label='效应=0', zorder=0)

# 添加基准期线和政策实施线
ax1.axvline(x=-1, color='gray', linestyle=':', linewidth=2, alpha=0.7, label='基准期(2008)')
ax1.axvline(x=0, color='green', linestyle='--', linewidth=2.5, alpha=0.7, label='政策实施(2009)')

# 添加显著性标记
for _, row in df_results.iterrows():
    if row['显著性'] != '':
        ax1.text(row['相对年份'], row['处理效应'], row['显著性'],
                ha='center', va='bottom', fontsize=12, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.6))

# 标注时期
ax1.axvspan(xmin=-3, xmax=0, ymin=0, ymax=0.05, transform=ax1.transAxes,
           facecolor='lightblue', alpha=0.2, label='预处理期')
ax1.text(-1.5, ax1.get_ylim()[1]*0.95, '预处理期', ha='center', fontsize=12,
        bbox=dict(boxstyle='round,pad=0.4', facecolor='lightblue', alpha=0.5))

ax1.set_xlabel('相对年份（以2009年为0）', fontsize=13, fontweight='bold')
ax1.set_ylabel('处理效应系数', fontsize=13, fontweight='bold')
ax1.set_title('事件研究图（Event Study Plot）\n动态处理效应（95%置信区间）',
               fontsize=15, fontweight='bold')
ax1.legend(loc='upper left', fontsize=10)
ax1.grid(True, alpha=0.3, axis='y')

# 图2：预处理期放大
pre_data = df_results[df_results['时期'] == '预处理期']

ax2.scatter(pre_data['相对年份'], pre_data['处理效应'],
           s=150, c='red', zorder=3, alpha=0.8)
ax2.vlines(pre_data['相对年份'], pre_data['95%CI下限'],
           pre_data['95%CI上限'], colors='red', linewidth=3, alpha=0.7, zorder=2)
ax2.axhline(y=0, color='black', linestyle='--', linewidth=2.5, label='效应=0', zorder=0)

# 添加显著性标记
for _, row in pre_data.iterrows():
    if row['显著性'] != '':
        ax2.text(row['相对年份'], row['处理效应'], row['显著性'],
                ha='center', va='bottom', fontsize=16, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='yellow', alpha=0.7))

# 添加检验结论
if parallel_trend_result == "支持":
    conclusion = "✓ 支持平行趋势"
    color = 'green'
elif parallel_trend_result == "基本支持":
    conclusion = "△ 基本支持"
    color = 'orange'
else:
    conclusion = "✗ 不支持"
    color = 'red'

ax2.text(0.5, 0.1, conclusion, transform=ax2.transAxes,
        ha='center', fontsize=14, fontweight='bold', color=color,
        bbox=dict(boxstyle='round,pad=0.7', facecolor='white', edgecolor=color, linewidth=3))

ax2.set_xlabel('相对年份（政策实施前）', fontsize=12, fontweight='bold')
ax2.set_ylabel('处理效应系数', fontsize=12, fontweight='bold')
ax2.set_title('预处理期平行趋势检验（放大）', fontsize=14, fontweight='bold')
ax2.legend(loc='best')
ax2.grid(True, alpha=0.3, axis='y')

# 图3：结果汇总表
ax3.axis('off')

# 创建表格数据
table_data = []
for _, row in df_results.iterrows():
    yr = row['相对年份']
    period = row['时期']
    sig_status = row['显著性']
    if sig_status == '':
        sig_text = "不显著"
    else:
        sig_text = f"{sig_status}"

    table_data.append([
        f"{yr:+d}",
        f"{row['处理效应']:.4f}",
        f"[{row['95%CI下限']:.4f}, {row['95%CI上限']:.4f}]",
        f"{row['p值']:.4f}",
        sig_text
    ])

# 绘制表格
table = ax3.table(cellText=table_data,
                  colLabels=['相对年份', '处理效应', '95%CI', 'p值', '显著性'],
                  cellLoc='center',
                  loc='center',
                  colWidths=[0.1, 0.2, 0.3, 0.15, 0.15])
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 2)

# 设置表头样式
for i in range(5):
    table[(0, i)].set_facecolor('#2E4053')
    table[(0, i)].set_text_props(weight='bold', color='white')

# 高亮预处理期
for i, row_data in enumerate(table_data, 1):
    if df_results.iloc[i-1]['时期'] == '预处理期':
        for j in range(5):
            table[(i, j)].set_facecolor('#EBF5FB')

ax3.set_title('回归结果汇总表', fontsize=14, fontweight='bold', pad=15)

plt.savefig('事件研究图_学术标准.png', dpi=300, bbox_inches='tight')
print("\n事件研究图已保存至: 事件研究图_学术标准.png")

# ============================================================================
# 步骤5: 生成详细报告
# ============================================================================
print("\n[步骤5: 生成报告...")

report = f"""
{'='*80}
事件研究法 - 平行趋势检验报告（学术标准）
{'='*80}

一、分析方法
----------
方法：事件研究法（Event Study）- 手工计算版
样本：PSM匹配后的城市（{len(matched_cities)}个城市）
期间：2007-2019年（相对年份-2到+10）
基准期：rel_year = -1（2008年）

计算方法：
  - 每年分别计算处理组与对照组的均值差
  - 使用两样本t检验计算标准误
  - 构建95%置信区间

二、检验假设
----------
原假设 (H0)：预处理期处理效应 = 0（满足平行趋势）
备择假设 (H1)：预处理期处理效应 ≠ 0（不满足平行趋势）

三、回归结果
----------
详见文件：事件研究法回归结果.xlsx

四、平行趋势检验结果
----------
{'='*80}

预处理期处理效应估计：
"""

for _, row in pre_results.iterrows():
    sig_text = row['显著性'] if row['显著性'] != '' else '不显著'
    report += f"rel_year = {row['相对年份']:+3d}: "
    report += f"处理效应 = {row['处理效应']:.6f},  "
    report += f"p值 = {row['p值']:.4f}  ({sig_text})\n"

report += f"\n{'='*80}\n\n检验结论：\n\n"

if parallel_trend_result == "支持":
    report += f"✓ 支持平行趋势假设\n\n"
    report += f"统计依据：所有{pre_total}个预处理期处理效应均不显著（p > 0.1）\n"
    report += f"经济含义：政策实施前，处理组和对照组的碳排放强度变化趋势平行\n\n"
    report += f"学术结论：满足DID模型的识别假设，估计结果可信\n"
elif parallel_trend_result == "基本支持":
    report += f"△ 基本支持平行趋势假设\n\n"
    report += f"统计依据：{pre_total}个预处理期中，仅{pre_sig_count}个处理效应显著（p < 0.1）\n"
    report += f"经济含义：预处理期趋势基本平行\n\n"
    report += f"学术结论：基本满足DID模型的识别假设，估计结果基本可信\n"
else:
    report += f"✗ 不支持平行趋势假设\n\n"
    report += f"统计依据：{pre_total}个预处理期中，{pre_sig_count}个处理效应显著（p < 0.1）\n"
    report += f"经济含义：政策实施前两组趋势不平行\n\n"
    report += f"学术结论：违反DID模型的识别假设，估计结果可能有偏\n"
    report += f"建议：考虑使用合成控制法或其他方法\n"

report += f"""
五、可视化
----------
详见文件：事件研究图_学术标准.png

图中展示：
  - 主图：完整的事件研究图（含95%置信区间）
  - 左下：预处理期放大图
  - 右下：回归结果汇总表

六、政策效应动态演变
----------
"""

post_results = df_results[df_results['时期'] == '政策实施后']
if len(post_results) > 0:
    report += f"政策实施后处理效应（前5年）：\n\n"
    for _, row in post_results.head(5).iterrows():
        sig_text = row['显著性'] if row['显著性'] != '' else '不显著'
        report += f"  year = {row['年份']} (rel_year={row['相对年份']:+2d}): "
        report += f"处理效应 = {row['处理效应']:.6f},  p值 = {row['p值']:.4f}  ({sig_text})\n"

report += f"""

{'='*80}
报告生成时间：{pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
{'='*80}
"""

with open("事件研究法检验报告_手工版.txt", 'w', encoding='utf-8') as f:
    f.write(report)

print("\n报告已保存至: 事件研究法检验报告_手工版.txt")
print(report)

print("\n" + "="*80)
print("事件研究法分析完成！")
print("="*80)
