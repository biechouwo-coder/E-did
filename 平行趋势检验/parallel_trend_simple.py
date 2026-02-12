# -*- coding: utf-8 -*-
"""
平行趋势检验 - 简化版
使用事件研究法进行平行趋势检验
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

print("="*80)
print("平行趋势检验 (Parallel Trend Test)")
print("="*80)

# 读取数据
print("\n[步骤1] 读取数据...")
df_full = pd.read_excel("../总数据集2007-2019_含碳排放强度.xlsx")
df_matched = pd.read_excel("../PSM-四个变量-2009基期/匹配后数据集_四个变量.xlsx")

matched_cities = df_matched['city_name'].unique()
df = df_full[df_full['city_name'].isin(matched_cities)].copy()

required_vars = ['ln_emission_intensity_winzorized', 'Treat', 'city_name', 'year']
df = df[required_vars].dropna()

print(f"筛选后样本数: {len(df)}")
print(f"城市数: {df['city_name'].nunique()}")

# 计算各年处理组和对照组均值
print("\n[步骤2] 计算各期均值...")
group_stats = df.groupby(['year', 'Treat'])['ln_emission_intensity_winzorized'].mean().unstack()

print("\n各年份处理组和对照组均值:")
print(group_stats)

# 创建图表
print("\n[步骤3] 绘制平行趋势图...")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 图1：完整趋势
ax1 = axes[0, 0]
for treat_val in [0, 1]:
    label = '处理组' if treat_val == 1 else '对照组'
    ax1.plot(group_stats.index, group_stats[treat_val], marker='o',
            label=label, linewidth=2.5, markersize=6)
ax1.axvline(x=2009, color='red', linestyle='--', linewidth=2, alpha=0.7, label='政策实施')
ax1.set_xlabel('年份', fontsize=12, fontweight='bold')
ax1.set_ylabel('ln(碳排放强度)', fontsize=12, fontweight='bold')
ax1.set_title('处理组与对照组碳排放强度趋势', fontsize=14, fontweight='bold')
ax1.legend(loc='best')
ax1.grid(True, alpha=0.3)

# 图2：预处理期放大
ax2 = axes[0, 1]
pre_data = group_stats[group_stats.index < 2009]
for treat_val in [0, 1]:
    label = '处理组' if treat_val == 1 else '对照组'
    ax2.plot(pre_data.index, pre_data[treat_val], marker='o',
            label=label, linewidth=3, markersize=8)
ax2.set_xlabel('年份', fontsize=12, fontweight='bold')
ax2.set_ylabel('ln(碳排放强度)', fontsize=12, fontweight='bold')
ax2.set_title('预处理期平行趋势检验（2007-2008）', fontsize=14, fontweight='bold')
ax2.legend(loc='best')
ax2.grid(True, alpha=0.3)

# 图3：差值趋势
ax3 = axes[1, 0]
diff = group_stats[1] - group_stats[0]
ax3.plot(diff.index, diff.values, marker='s', color='purple',
        linewidth=2.5, markersize=6, label='处理组-对照组')
ax3.axhline(y=0, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
ax3.axvline(x=2009, color='green', linestyle='--', linewidth=2, alpha=0.7, label='政策实施')
ax3.set_xlabel('年份', fontsize=12, fontweight='bold')
ax3.set_ylabel('差值', fontsize=12, fontweight='bold')
ax3.set_title('处理组与对照组之差', fontsize=14, fontweight='bold')
ax3.legend(loc='best')
ax3.grid(True, alpha=0.3)

# 图4：预处理期差值
ax4 = axes[1, 1]
pre_diff = diff[diff.index < 2009]
ax4.plot(pre_diff.index, pre_diff.values, marker='s', color='purple',
        linewidth=3, markersize=10)
ax4.axhline(y=0, color='red', linestyle='--', linewidth=2, alpha=0.7)
ax4.set_xlabel('年份', fontsize=12, fontweight='bold')
ax4.set_ylabel('差值', fontsize=12, fontweight='bold')
ax4.set_title('预处理期差值（检验平行趋势）', fontsize=14, fontweight='bold')
ax4.grid(True, alpha=0.3)

# 添加结论
pre_diff_vals = pre_diff.values
if len(pre_diff_vals) >= 2:
    # 简单检验：预处理期差值是否稳定
    diff_change = abs(pre_diff_vals[-1] - pre_diff_vals[0])
    mean_diff = np.mean(np.abs(pre_diff_vals))

    if diff_change < 0.1:
        conclusion = "✓ 预处理期差值稳定\n支持平行趋势假设"
        color = 'green'
    elif diff_change < 0.2:
        conclusion = "△ 预处理期差值基本稳定\n基本支持平行趋势"
        color = 'orange'
    else:
        conclusion = "✗ 预处理期差值变化较大\n可能违反平行趋势"
        color = 'red'

    ax4.text(0.5, 0.1, conclusion, transform=ax4.transAxes,
            ha='center', fontsize=12, fontweight='bold', color=color,
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor=color, linewidth=2))

plt.tight_layout()
plt.savefig('平行趋势检验.png', dpi=300, bbox_inches='tight')
print("\n图表已保存至: 平行趋势检验.png")

# 生成报告
print("\n[步骤4] 生成报告...")

report = f"""
{'='*80}
平行趋势检验报告（简化版）
{'='*80}

一、样本情况
----------
PSM匹配后城市数: {len(matched_cities)}
分析期间: 2007-2019年
样本量: {len(df)}

二、预处理期趋势
----------
{group_stats[group_stats.index < 2009].to_string()}

三、检验结论
----------
根据预处理期（2007-2008）处理组和对照组的趋势图：

"""

pre_diff_vals = diff[diff.index < 2009].values
if len(pre_diff_vals) >= 2:
    diff_change = abs(pre_diff_vals[-1] - pre_diff_vals[0])
    if diff_change < 0.1:
        report += "✓ 支持平行趋势假设\n"
        report += "  理由：预处理期两组差距变化较小（<0.1），趋势基本平行\n"
    elif diff_change < 0.2:
        report += "△ 基本支持平行趋势假设\n"
        report += f"  理由：预处理期两组差距变化为{diff_change:.4f}，基本平行\n"
    else:
        report += "✗ 可能违反平行趋势假设\n"
        report += f"  理由：预处理期两组差距变化较大（{diff_change:.4f}），趋势不平行\n"
else:
    report += "⚠ 预处理期数据不足，无法检验\n"

report += f"""
四、建议
----------
1. 此为简化版检验，建议结合事件研究法进行更严格的统计检验
2. 如预处理期趋势不平行，考虑：
   - 使用PSM-DID方法
   - 缩短研究窗口
   - 加入更多控制变量
   - 使用合成控制法

{'='*80}
生成时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
{'='*80}
"""

with open("平行趋势检验报告.txt", 'w', encoding='utf-8') as f:
    f.write(report)

print("\n报告已保存至: 平行趋势检验报告.txt")
print(report)

print("\n" + "="*80)
print("平行趋势检验完成！")
print("="*80)
