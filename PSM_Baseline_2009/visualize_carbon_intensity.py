import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

print("正在生成碳排放强度可视化图表...")

# 读取数据
df = pd.read_excel('matched_data_full_period.xlsx')

# 筛选2007-2019年的数据
df_complete = df[
    (df['year'] >= 2007) &
    (df['year'] <= 2019) &
    (df['carbon_intensity'].notna())
].copy()

# 创建图表
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. 年度趋势 - 处理组 vs 对照组
ax = axes[0, 0]
yearly_treat = df_complete[df_complete['Treat'] == 1].groupby('year')['carbon_intensity'].mean()
yearly_control = df_complete[df_complete['Treat'] == 0].groupby('year')['carbon_intensity'].mean()

ax.plot(yearly_treat.index, yearly_treat.values, 'ro-', label='处理组', linewidth=2, markersize=6)
ax.plot(yearly_control.index, yearly_control.values, 'bo-', label='对照组', linewidth=2, markersize=6)
ax.set_xlabel('年份')
ax.set_ylabel('碳排放强度 (万吨/亿元)')
ax.set_title('碳排放强度年度趋势（2007-2019）')
ax.legend()
ax.grid(True, alpha=0.3)

# 2. 箱线图 - 处理组 vs 对照组
ax = axes[0, 1]
treat_data = [df_complete[(df_complete['Treat'] == 1) & (df_complete['year'] == y)]['carbon_intensity'].values
              for y in range(2007, 2020)]
control_data = [df_complete[(df_complete['Treat'] == 0) & (df_complete['year'] == y)]['carbon_intensity'].values
                for y in range(2007, 2020)]

positions = np.arange(2007, 2020)
bp1 = ax.boxplot(treat_data, positions=positions - 0.2, widths=0.4, patch_artist=True,
                 boxprops=dict(facecolor='red', alpha=0.5),
                 medianprops=dict(color='darkred', linewidth=2),
                 showfliers=False)
bp2 = ax.boxplot(control_data, positions=positions + 0.2, widths=0.4, patch_artist=True,
                 boxprops=dict(facecolor='blue', alpha=0.5),
                 medianprops=dict(color='darkblue', linewidth=2),
                 showfliers=False)

ax.set_xlabel('年份')
ax.set_ylabel('碳排放强度 (万吨/亿元)')
ax.set_title('碳排放强度分布（处理组 vs 对照组）')
ax.set_xlim(2006.5, 2019.5)
ax.legend([bp1["boxes"][0], bp2["boxes"][0]], ['处理组', '对照组'])
ax.grid(True, alpha=0.3, axis='y')

# 3. 试点前后对比
ax = axes[1, 0]
pre_period = df_complete[df_complete['year'] < 2009]
post_period = df_complete[df_complete['year'] >= 2009]

pre_treat = pre_period[pre_period['Treat'] == 1]['carbon_intensity'].mean()
pre_control = pre_period[pre_period['Treat'] == 0]['carbon_intensity'].mean()
post_treat = post_period[post_period['Treat'] == 1]['carbon_intensity'].mean()
post_control = post_period[post_period['Treat'] == 0]['carbon_intensity'].mean()

categories = ['试点前\n(2007-2008)', '试点后\n(2009-2019)']
treat_means = [pre_treat, post_treat]
control_means = [pre_control, post_control]

x = np.arange(len(categories))
width = 0.35

ax.bar(x - width/2, treat_means, width, label='处理组', color='red', alpha=0.7)
ax.bar(x + width/2, control_means, width, label='对照组', color='blue', alpha=0.7)

ax.set_ylabel('平均碳排放强度 (万吨/亿元)')
ax.set_title('试点前后平均碳排放强度对比')
ax.set_xticks(x)
ax.set_xticklabels(categories)
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

# 添加数值标签
for i, (t, c) in enumerate(zip(treat_means, control_means)):
    ax.text(i - width/2, t + 0.001, f'{t:.4f}', ha='center', va='bottom', fontsize=9)
    ax.text(i + width/2, c + 0.001, f'{c:.4f}', ha='center', va='bottom', fontsize=9)

# 4. DID效应可视化
ax = axes[1, 1]
# 计算年度DID效应
did_effects = []
years = range(2007, 2020)
for year in years:
    year_data = df_complete[df_complete['year'] == year]
    treat_mean = year_data[year_data['Treat'] == 1]['carbon_intensity'].mean()
    control_mean = year_data[year_data['Treat'] == 0]['carbon_intensity'].mean()
    diff = treat_mean - control_mean
    did_effects.append(diff)

# 绘制差值趋势
ax.bar(years, did_effects, color=['gray' if y < 2009 else 'green' for y in years],
       alpha=0.7, edgecolor='black')
ax.axvline(x=2009, color='red', linestyle='--', linewidth=2, label='政策实施')
ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
ax.set_xlabel('年份')
ax.set_ylabel('处理组-对照组差值 (万吨/亿元)')
ax.set_title('年度DID效应（处理组-对照组）')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('carbon_intensity_visualization.png', dpi=300, bbox_inches='tight')
print("[OK] 已保存可视化图表: carbon_intensity_visualization.png")

# 打印关键统计信息
print("\n关键统计信息:")
print(f"试点前处理组平均: {pre_treat:.4f} 万吨/亿元")
print(f"试点前对照组平均: {pre_control:.4f} 万吨/亿元")
print(f"试点前差值: {pre_treat - pre_control:.4f} 万吨/亿元")
print(f"试点后处理组平均: {post_treat:.4f} 万吨/亿元")
print(f"试点后对照组平均: {post_control:.4f} 万吨/亿元")
print(f"试点后差值: {post_treat - post_control:.4f} 万吨/亿元")
print(f"DID效应: {(post_treat - post_control) - (pre_treat - pre_control):.4f} 万吨/亿元")

print("\n可视化完成！")
