# -*- coding: utf-8 -*-
"""
事件研究法 - 手工版（Event Study - Manual Calculation）
平行趋势检验的学术标准方法

优势：
1. 避免patsy公式解析中负号的问题
2. 手工计算处理效应，更透明
3. 使用两样本t检验计算标准误
4. 生成带置信区间的事件研究图
"""

import pandas as pd
import numpy as np
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

print(f"完整数据集: {df_full.shape}")
print(f"PSM匹配后数据: {df_matched.shape}")

# 获取匹配后的城市列表
matched_cities = df_matched['city_name'].unique()
df = df_full[df_full['city_name'].isin(matched_cities)].copy()

# 保留需要的变量
vars_needed = ['ln_emission_intensity_winzorized', 'Treat', 'city_name', 'year']
df = df[vars_needed].dropna()

print(f"筛选后样本: {len(df)}")
print(f"城市数: {df['city_name'].nunique()}")
print(f"年份范围: {df['year'].min()} - {df['year'].max()}")

# ============================================================================
# 步骤2: 创建相对年份
# ============================================================================
print("\n[步骤2] 创建相对年份...")

df['rel_year'] = df['year'] - 2009
years = sorted(df['rel_year'].unique())
print(f"相对年份: {years}")

# ============================================================================
# 步骤3: 手工计算每年的处理效应
# ============================================================================
print("\n[步骤3] 计算每年的处理效应...")

results = []

for yr in years:
    # 筛选该年数据
    data_yr = df[df['rel_year'] == yr]

    # 分组
    treat = data_yr[data_yr['Treat'] == 1]['ln_emission_intensity_winzorized']
    control = data_yr[data_yr['Treat'] == 0]['ln_emission_intensity_winzorized']

    # 计算均值差（处理效应）
    diff = treat.mean() - control.mean()

    # 计算标准误（两样本t检验）
    n1 = len(treat)
    n2 = len(control)
    var1 = treat.var(ddof=1)  # 样本方差
    var2 = control.var(ddof=1)

    # 两样本t检验的标准误
    se = np.sqrt(var1/n1 + var2/n2)

    # t统计量
    t_stat = diff / se

    # 自由度（Welch's t-test）
    df_deg = (var1/n1 + var2/n2)**2 / ((var1/n1)**2/(n1-1) + (var2/n2)**2/(n2-1))

    # p值（双尾）
    p_val = 2 * (1 - stats.t.cdf(abs(t_stat), df_deg))

    # 95%置信区间
    ci_lower = diff - 1.96 * se
    ci_upper = diff + 1.96 * se

    # 显著性标记
    sig = '***' if p_val < 0.01 else '**' if p_val < 0.05 else '*' if p_val < 0.1 else ''

    results.append({
        '相对年份': yr,
        '实际年份': 2009 + yr,
        '处理组均值': treat.mean(),
        '对照组均值': control.mean(),
        '处理组样本': n1,
        '对照组样本': n2,
        '处理效应': diff,
        '标准误': se,
        't值': t_stat,
        'p值': p_val,
        '95%CI下限': ci_lower,
        '95%CI上限': ci_upper,
        '显著性': sig
    })

    print(f"rel_year = {yr:3d} ({2009+yr}): 处理效应 = {diff:8.6f}, SE = {se:.6f}, p = {p_val:.4f}  {sig}")

# 保存结果
df_results = pd.DataFrame(results)
df_results.to_excel("事件研究法回归结果_手工版.xlsx", index=False)
print("\n结果已保存至: 事件研究法回归结果_手工版.xlsx")

# ============================================================================
# 步骤4: 绘制事件研究图
# ============================================================================
print("\n[步骤4] 绘制事件研究图...")

fig = plt.figure(figsize=(18, 12))
gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.25)

# 图1：完整的事件研究图
ax1 = fig.add_subplot(gs[0, :])

# 区分预处理期和政策实施后
pre_data = df_results[df_results['相对年份'] < 0]
post_data = df_results[df_results['相对年份'] >= 0]

# 绘制预处理期
ax1.scatter(pre_data['相对年份'], pre_data['处理效应'],
           s=100, color='red', zorder=3, label='预处理期系数')
ax1.vlines(pre_data['相对年份'], pre_data['95%CI下限'], pre_data['95%CI上限'],
           colors='red', linewidth=2.5, alpha=0.7, zorder=2)
ax1.plot(pre_data['相对年份'], pre_data['处理效应'],
        color='red', linewidth=2, alpha=0.6, zorder=1)

# 绘制政策实施后
ax1.scatter(post_data['相对年份'], post_data['处理效应'],
           s=100, color='green', zorder=3, label='政策实施后系数')
ax1.vlines(post_data['相对年份'], post_data['95%CI下限'], post_data['95%CI上限'],
           colors='green', linewidth=2.5, alpha=0.7, zorder=2)
ax1.plot(post_data['相对年份'], post_data['处理效应'],
        color='green', linewidth=2, alpha=0.6, zorder=1)

# 添加y=0参考线
ax1.axhline(y=0, color='black', linestyle='--', linewidth=2, label='系数=0', zorder=0)

# 添加显著性标记
for _, row in df_results.iterrows():
    if row['显著性'] != '':
        color = 'red' if row['相对年份'] < 0 else 'green'
        ax1.text(row['相对年份'], row['处理效应'], row['显著性'],
                ha='center', va='bottom', fontsize=11, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7, edgecolor=color, linewidth=2))

# 标注政策实施
ax1.axvline(x=0, color='blue', linestyle='--', linewidth=2.5, alpha=0.7)
ax1.text(0, ax1.get_ylim()[1]*0.90, '政策实施\n(2009年)',
        ha='center', fontsize=11, fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='lightblue', alpha=0.8))

ax1.set_xlabel('相对年份（以2009年为0）', fontsize=13, fontweight='bold')
ax1.set_ylabel('处理效应系数', fontsize=13, fontweight='bold')
ax1.set_title('事件研究图（Event Study Plot）- 动态处理效应\n95%置信区间',
              fontsize=15, fontweight='bold')
ax1.legend(loc='best', fontsize=11)
ax1.grid(True, alpha=0.3, axis='y')

# 图2：预处理期放大（平行趋势检验）
ax2 = fig.add_subplot(gs[1, 0])

pre_data = df_results[df_results['相对年份'] < 0].copy()

if len(pre_data) > 0:
    ax2.scatter(pre_data['相对年份'], pre_data['处理效应'],
               s=150, color='red', zorder=3)
    ax2.vlines(pre_data['相对年份'], pre_data['95%CI下限'], pre_data['95%CI上限'],
               colors='red', linewidth=3.5, alpha=0.8, zorder=2)
    ax2.plot(pre_data['相对年份'], pre_data['处理效应'],
            color='red', linewidth=2.5, alpha=0.6, zorder=1)
    ax2.axhline(y=0, color='black', linestyle='--', linewidth=2.5, label='系数=0', zorder=0)

    # 添加显著性标记
    for _, row in pre_data.iterrows():
        if row['显著性'] != '':
            ax2.text(row['相对年份'], row['处理效应'], row['显著性'],
                    ha='center', va='bottom', fontsize=18, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.4', facecolor='yellow', alpha=0.7, edgecolor='red', linewidth=2.5))

    ax2.set_xlabel('相对年份（政策实施前）', fontsize=12, fontweight='bold')
    ax2.set_ylabel('处理效应系数', fontsize=12, fontweight='bold')
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

    ax2.text(0.5, 0.05, conclusion, transform=ax2.transAxes,
            ha='center', fontsize=12, fontweight='bold', color=color,
            bbox=dict(boxstyle='round,pad=0.7', facecolor='white', edgecolor=color, linewidth=3))

# 图3：政策实施后效应
ax3 = fig.add_subplot(gs[1, 1])

post_data = df_results[df_results['相对年份'] >= 0].copy()

if len(post_data) > 0:
    ax3.scatter(post_data['相对年份'], post_data['处理效应'],
               s=120, color='green', zorder=3)
    ax3.vlines(post_data['相对年份'], post_data['95%CI下限'], post_data['95%CI上限'],
               colors='green', linewidth=2.5, alpha=0.7, zorder=2)
    ax3.plot(post_data['相对年份'], post_data['处理效应'],
            color='green', linewidth=2, alpha=0.6, zorder=1)
    ax3.axhline(y=0, color='black', linestyle='--', linewidth=2, label='系数=0', zorder=0)

    # 添加显著性标记
    for _, row in post_data.iterrows():
        if row['显著性'] != '':
            ax3.text(row['相对年份'], row['处理效应'], row['显著性'],
                    ha='center', va='bottom', fontsize=11, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='yellow', alpha=0.6, edgecolor='green', linewidth=2))

    ax3.set_xlabel('相对年份（政策实施后）', fontsize=12, fontweight='bold')
    ax3.set_ylabel('处理效应系数', fontsize=12, fontweight='bold')
    ax3.set_title('政策实施后动态效应', fontsize=14, fontweight='bold')
    ax3.legend(loc='best')
    ax3.grid(True, alpha=0.3, axis='y')

# 图4：回归结果汇总表
ax4 = fig.add_subplot(gs[2, :])
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
        f"{2009+yr}",
        f"{row['处理效应']:.4f}",
        f"({row['标准误']:.4f})",
        f"[{row['95%CI下限']:.4f}, {row['95%CI上限']:.4f}]",
        f"{row['p值']:.4f}",
        sig_text,
        period
    ])

# 绘制表格
table = ax4.table(cellText=summary_data,
                  colLabels=['相对年份', '实际年份', '处理效应', '(标准误)', '95%置信区间', 'p值', '显著性', '时期'],
                  cellLoc='center',
                  loc='center',
                  colWidths=[0.08, 0.08, 0.12, 0.12, 0.20, 0.10, 0.15, 0.15])
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2.2)

# 设置表头样式
for i in range(8):
    table[(0, i)].set_facecolor('#4472C4')
    table[(0, i)].set_text_props(weight='bold', color='white')
    table[(0, i)].set_height(0.08)

# 高亮预处理期和政策实施后
for i, row_data in enumerate(summary_data, 1):
    if row_data[7] == "预处理期":
        for j in range(8):
            table[(i, j)].set_facecolor('#FFE6E6')  # 浅红色
    else:
        for j in range(8):
            table[(i, j)].set_facecolor('#E6FFE6')  # 浅绿色

ax4.set_title('回归结果汇总表（手工计算）', fontsize=14, fontweight='bold', pad=20)

plt.savefig('事件研究图_学术标准.png', dpi=300, bbox_inches='tight')
print("\n事件研究图已保存至: 事件研究图_学术标准.png")

# ============================================================================
# 步骤5: 平行趋势统计检验
# ============================================================================
print("\n[步骤5] 平行趋势统计检验...")

pre_results = df_results[df_results['相对年份'] < 0]

if len(pre_results) > 0:
    print("\n" + "="*80)
    print("平行趋势假设检验")
    print("="*80)

    print("\n原假设 (H0): 预处理期所有处理效应系数 = 0")
    print("备择假设 (H1): 至少一个预处理期处理效应系数 ≠ 0")
    print()

    print("预处理期系数估计:")
    print("-"*80)
    for _, row in pre_results.iterrows():
        sig_text = row['显著性'] if row['显著性'] != '' else '不显著'
        print(f"rel_year = {row['相对年份']:3d} (year={2009+int(row['相对年份']):4d}):")
        print(f"  处理效应 = {row['处理效应']:8.6f}")
        print(f"  标准误 = {row['标准误']:.6f}")
        print(f"  t值 = {row['t值']:.4f}")
        print(f"  p值 = {row['p值']:.4f}")
        print(f"  95% CI = [{row['95%CI下限']:.6f}, {row['95%CI上限']:.6f}]")
        print(f"  显著性: {sig_text}")
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
        parallel_trend_result = "不支持"

# ============================================================================
# 步骤6: 生成详细报告
# ============================================================================
print("\n[步骤6] 生成详细报告...")

report = f"""
{'='*80}
事件研究法 - 平行趋势检验报告（手工计算版 - 学术标准）
{'='*80}

一、分析方法
----------
方法：事件研究法（Event Study）- 手工计算处理效应
计算方式：
  - 每年分别计算处理组均值 - 对照组均值
  - 使用两样本t检验计算标准误
  - 构建95%置信区间
  - 统计显著性检验

样本：PSM匹配后的城市（{len(matched_cities)}个城市）
期间：2007-2019年（相对年份-2到+10）
观测值：{len(df)}个

二、检验假设
----------
原假设 (H0)：预处理期所有处理效应系数 = 0（满足平行趋势）
备择假设 (H1)：至少一个预处理期处理效应系数 ≠ 0（不满足平行趋势）

三、回归结果
----------
详见文件：事件研究法回归结果_手工版.xlsx

"""

if len(pre_results) > 0:
    report += f"四、平行趋势检验结果\n"
    report += f"{'='*80}\n\n"

    for _, row in pre_results.iterrows():
        sig_text = row['显著性'] if row['显著性'] != '' else '不显著'
        report += f"rel_year = {row['相对年份']:3d} (year={2009+int(row['相对年份']):4d}): "
        report += f"处理效应 = {row['处理效应']:8.6f},  "
        report += f"标准误 = {row['标准误']:.6f},  "
        report += f"p值 = {row['p值']:.4f}  ({sig_text})\n"

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
    report += f"详见文件：事件研究图_学术标准.png\n\n"
    report += f"图中展示：\n"
    report += f"  - 横轴：相对年份（0为2009年政策实施年）\n"
    report += f"  - 纵轴：处理效应系数（处理组均值 - 对照组均值）\n"
    report += f"  - 红色点/线：预处理期系数\n"
    report += f"  - 绿色点/线：政策实施后系数\n"
    report += f"  - 竖线：95%置信区间\n"
    report += f"  - 显著性标记：*** p<0.01, ** p<0.05, * p<0.1\n\n"

    if len(df_results[df_results['相对年份'] >= 0]) > 0:
        report += f"六、政策效应动态演变\n"
        report += f"{'='*80}\n\n"
        report += f"政策实施后效应：\n\n"
        post_results = df_results[df_results['相对年份'] >= 0]
        for _, row in post_results.iterrows():
            sig_text = row['显著性'] if row['显著性'] != '' else '不显著'
            report += f"  year = {2009+int(row['相对年份']):4d} (rel_year={row['相对年份']:+2d}): "
            report += f"处理效应 = {row['处理效应']:8.6f},  p值 = {row['p值']:.4f}  ({sig_text})\n"

report += f"""

{'='*80}
报告完成时间：{pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
{'='*80}
"""

with open("事件研究法检验报告_手工版.txt", 'w', encoding='utf-8') as f:
    f.write(report)

print("\n详细报告已保存至: 事件研究法检验报告_手工版.txt")
print(report)

print("\n" + "="*80)
print("事件研究法分析完成！（手工计算版）")
print("="*80)
print("\n生成文件:")
print("  1. 事件研究法回归结果_手工版.xlsx")
print("  2. 事件研究图_学术标准.png")
print("  3. 事件研究法检验报告_手工版.txt")
