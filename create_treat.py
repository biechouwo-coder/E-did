# -*- coding: utf-8 -*-
"""
创建treat变量（静态分组变量）

treat变量逻辑：
- 只要城市在低碳试点名单中（无论哪一批、哪一年开始），treat=1
- 从未入选的城市，treat=0
- treat不随时间变化
"""

import pandas as pd
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

print("="*70)
print("创建treat变量（静态分组变量）")
print("="*70)

# 步骤1: 读取数据
print("\n步骤1: 读取数据...")

# 读取低碳试点城市名单
csv_df = pd.read_csv('Low_Carbon_Pilot_Cities_Complete.csv', encoding='gbk')
print(f"✓ 低碳试点城市名单: {len(csv_df)} 个城市")

# 读取主数据集
main_df = pd.read_excel('总数据集_2007-2023_完整版_DID.xlsx')
print(f"✓ 主数据集: {len(main_df)} 行 × {len(main_df.columns)} 列")
print(f"  年份范围: {main_df['year'].min()} - {main_df['year'].max()}")
print(f"  城市数量: {main_df['city_name'].nunique()} 个")

# 步骤2: 创建城市集合
print("\n步骤2: 创建试点城市集合...")

# 处理城市名称匹配问题
csv_df['City_Clean'] = csv_df['City'].str.replace('市', '', regex=False)

# 创建试点城市集合（包含带"市"和不带"市"的版本）
pilot_cities_with_suffix = set(csv_df['City'])
pilot_cities_without_suffix = set(csv_df['City_Clean'])

# 合并两个集合
all_pilot_cities = pilot_cities_with_suffix | pilot_cities_without_suffix

print(f"✓ 试点城市集合包含 {len(all_pilot_cities)} 个城市名称")
print(f"  示例城市: {list(all_pilot_cities)[:10]}")

# 步骤3: 创建treat变量
print("\n步骤3: 创建treat变量...")

def is_pilot_city(city_name):
    """检查城市是否在试点名单中"""
    if city_name in pilot_cities_with_suffix:
        return 1
    # 尝试去除"市"后缀匹配
    city_clean = city_name.replace('市', '')
    if city_clean in pilot_cities_without_suffix:
        return 1
    return 0

# 应用函数创建treat变量
main_df['treat'] = main_df['city_name'].apply(is_pilot_city)

print(f"✓ treat变量创建完成")
print(f"  treat=1 (试点城市): {main_df['treat'].sum()} 个观测")
print(f"  treat=0 (非试点城市): {(main_df['treat']==0).sum()} 个观测")

# 检查treat=1的唯一城市数
pilot_cities_in_data = main_df[main_df['treat']==1]['city_name'].nunique()
non_pilot_cities_in_data = main_df[main_df['treat']==0]['city_name'].nunique()

print(f"\n  试点城市数: {pilot_cities_in_data}")
print(f"  非试点城市数: {non_pilot_cities_in_data}")

# 步骤4: 验证treat变量的时间不变性
print("\n步骤4: 验证treat变量的时间不变性...")

# 检查每个城市的treat值是否随时间变化
city_treat_check = main_df.groupby('city_name')['treat'].nunique()
cities_with_varying_treat = city_treat_check[city_treat_check > 1]

if len(cities_with_varying_treat) == 0:
    print("✓ 所有城市的treat值不随时间变化（验证通过）")
else:
    print(f"⚠ 警告: {len(cities_with_varying_treat)} 个城市的treat值随时间变化:")
    print(cities_with_varying_treat.index.tolist())

# 步骤5: 显示treat与DID的区别
print("\n步骤5: 显示treat与DID的区别...")
print("  treat: 静态分组变量（Who）- 是否在试点名单中")
print("  DID: 动态处理变量（When）- 是否已经实施政策")

# 选择一个试点城市示例
sample_city = main_df[main_df['treat']==1]['city_name'].iloc[0]
sample_data = main_df[main_df['city_name']==sample_city].sort_values('year')[['year', 'treat', 'DID']].head(10)

print(f"\n  示例城市: {sample_city}")
print(sample_data.to_string(index=False))

# 步骤6: 保存更新后的数据集
print("\n步骤6: 保存更新后的数据集...")

# 调整列顺序，将treat放在前面
cols = list(main_df.columns)
cols.insert(cols.index('city_name') + 1, cols.pop(cols.index('treat')))
main_df = main_df[cols]

output_file = '总数据集_2007-2023_完整版_DID_with_treat.xlsx'
main_df.to_excel(output_file, index=False)

print(f"✓ 数据集已保存至: {output_file}")
print(f"  总列数: {len(main_df.columns)}")
print(f"  列名: {main_df.columns.tolist()}")

# 生成统计摘要
print("\n" + "="*70)
print("treat变量创建完成！")
print("="*70)
print(f"\n统计摘要:")
print(f"  总观测数: {len(main_df)}")
print(f"  试点城市 (treat=1): {pilot_cities_in_data} 个城市, {main_df['treat'].sum()} 个观测")
print(f"  非试点城市 (treat=0): {non_pilot_cities_in_data} 个城市, {(main_df['treat']==0).sum()} 个观测")
print(f"\n变量说明:")
print(f"  treat: 分组变量（Who）- 1=试点城市, 0=非试点城市（不随时间变化）")
print(f"  DID: 处理变量（When）- 1=已实施政策, 0=未实施政策（随时间变化）")
