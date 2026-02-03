# -*- coding: utf-8 -*-
import pandas as pd
import sys
import io

# 设置输出编码
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

print("="*60)
print("低碳试点城市DID变量生成脚本")
print("="*60)

print("\n步骤1: 读取数据文件...")

# 读取CSV文件（低碳试点城市名单）
csv_df = pd.read_csv('Low_Carbon_Pilot_Cities_Complete.csv', encoding='gbk')
print(f"✓ 成功读取CSV文件: {csv_df.shape[0]} 个城市")
print(f"  批次分布: {csv_df['Source'].value_counts().to_dict()}")

# 读取主数据集
main_df = pd.read_excel('总数据集_2007-2023_完整版.xlsx')
print(f"✓ 成功读取主数据集: {main_df.shape[0]} 行 × {main_df.shape[1]} 列")
print(f"  年份范围: {main_df['year'].min()} - {main_df['year'].max()}")
print(f"  城市数量: {main_df['city_name'].nunique()} 个唯一城市")

print("\n步骤2: 创建城市-生效年份映射...")

# 创建城市-生效年份映射字典
city_year_map = dict(zip(csv_df['City'], csv_df['Start_Year']))

# 处理城市名称匹配问题
# 移除可能存在的"市"后缀，统一处理
csv_df['City_Clean'] = csv_df['City'].str.replace('市', '', regex=False)
city_year_map_clean = dict(zip(csv_df['City_Clean'], csv_df['Start_Year']))

print(f"  映射字典包含 {len(city_year_map)} 个城市")
print(f"  示例城市及生效年份:")
for i, (city, year) in enumerate(list(city_year_map.items())[:5]):
    print(f"    {city}: {year}年")

print("\n步骤3: 生成DID变量...")

# 定义函数来获取城市的生效年份
def get_start_year(city_name):
    # 先尝试精确匹配
    if city_name in city_year_map:
        return city_year_map[city_name]
    # 尝试去除"市"后缀匹配
    city_clean = city_name.replace('市', '')
    if city_clean in city_year_map_clean:
        return city_year_map_clean[city_clean]
    # 如果找不到，返回一个很大的年份（确保DID=0）
    return 9999

# 生成DID列
main_df['DID'] = main_df.apply(
    lambda row: 1 if row['year'] >= get_start_year(row['city_name']) else 0,
    axis=1
)

print(f"✓ 成功生成DID变量")
print(f"  DID=1的观测数: {main_df['DID'].sum()} ({main_df['DID'].sum()/len(main_df)*100:.1f}%)")
print(f"  DID=0的观测数: {(main_df['DID']==0).sum()} ({(main_df['DID']==0).sum()/len(main_df)*100:.1f}%)")

print("\n步骤4: 验证结果...")

# 验证几个城市的结果
print("\n  示例验证（前3个城市的前5年）:")
sample_cities = main_df['city_name'].unique()[:3]
for city in sample_cities:
    city_data = main_df[main_df['city_name'] == city].sort_values('year').head(5)
    start_year = get_start_year(city)
    print(f"\n  {city} (生效年份: {start_year if start_year != 9999 else '未在名单中'}):")
    for _, row in city_data.iterrows():
        print(f"    {int(row['year'])}年: DID={int(row['DID'])}")

print("\n步骤5: 保存结果...")

# 保存结果
output_file = '总数据集_2007-2023_完整版_DID.xlsx'
main_df.to_excel(output_file, index=False)
print(f"✓ 结果已保存至: {output_file}")

# 生成统计报告
print("\n" + "="*60)
print("DID变量生成完成！")
print("="*60)
print(f"\n统计摘要:")
print(f"  总观测数: {len(main_df)}")
print(f"  处理组(DID=1): {main_df['DID'].sum()} 个观测")
print(f"  对照组(DID=0): {(main_df['DID']==0).sum()} 个观测")
print(f"  涉及城市数: {main_df['city_name'].nunique()}")

# 检查哪些城市在名单中
cities_in_list = set(city_year_map.keys())
cities_matched = main_df[main_df['DID'] == 1]['city_name'].unique()
print(f"\n  CSV名单中的城市数: {len(cities_in_list)}")
print(f"  主数据集中成功匹配的城市数: {len(cities_matched)}")
