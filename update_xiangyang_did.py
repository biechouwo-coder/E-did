import pandas as pd
import numpy as np

print("=" * 80)
print("更新总数据集中襄阳市的DID变量")
print("=" * 80)

# 1. 读取数据
print("\n步骤1: 读取总数据集...")
file_path = '总数据集2007-2023_仅含emission城市.xlsx'
df = pd.read_excel(file_path)
print(f"数据集形状: {df.shape}")

# 2. 查找襄阳市
print("\n步骤2: 查找襄阳市...")
cities = df['city_name'].unique()
print(f"总城市数: {len(cities)}")

# 搜索包含"襄阳"的城市
xiangyang_mask = df['city_name'].str.contains('襄阳', na=False)
xiangyang_count = xiangyang_mask.sum()

if xiangyang_count == 0:
    print("未找到襄阳市，尝试其他匹配方式...")
    # 尝试湖北的城市
    hubei_mask = df['province'].str.contains('湖北', na=False)
    hubei_cities = df[hubei_mask]['city_name'].unique()
    print(f"湖北省城市: {list(hubei_cities)}")
    exit(1)

print(f"找到襄阳市观测数: {xiangyang_count}")

# 显示襄阳市当前数据
xiangyang_data = df[xiangyang_mask].copy()
print(f"襄阳市年份范围: {xiangyang_data['year'].min()} - {xiangyang_data['year'].max()}")
print(f"襄阳市当前Treat值: {xiangyang_data['Treat'].unique()}")
print(f"襄阳市当前Post值: {xiangyang_data['Post'].unique()}")
print(f"襄阳市当前DID值: {xiangyang_data['DID'].unique()}")

# 3. 更新襄阳市的DID变量
print("\n步骤3: 更新襄阳市的DID变量...")
print("规则: 襄阳市从2010年起受政策影响")
print("  - Treat = 1 (襄阳市是处理组城市)")
print("  - Post = 1 if year >= 2010 else 0")
print("  - DID = Treat × Post")

# 更新Treat（襄阳是处理组）
df.loc[xiangyang_mask, 'Treat'] = 1
print("  [OK] Treat = 1")

# 更新Post（2010年起为1）
post_mask = xiangyang_mask & (df['year'] >= 2010)
df.loc[post_mask, 'Post'] = 1
print("  [OK] Post = 1 (year >= 2010)")

# 确保之前的Post=0
pre_post_mask = xiangyang_mask & (df['year'] < 2010)
df.loc[pre_post_mask, 'Post'] = 0

# 更新DID
df.loc[xiangyang_mask, 'DID'] = df.loc[xiangyang_mask, 'Treat'] * df.loc[xiangyang_mask, 'Post']
print("  [OK] DID = Treat × Post")

# 4. 验证更新结果
print("\n步骤4: 验证更新结果...")
xiangyang_updated = df[xiangyang_mask].copy()

print("\n襄阳市2009-2012年的数据:")
print(xiangyang_updated[xiangyang_updated['year'].between(2009, 2012)][['city_name', 'year', 'Treat', 'Post', 'DID']].to_string(index=False))

# 验证
pre_2010_did = xiangyang_updated[xiangyang_updated['year'] < 2010]['DID'].unique()
post_2010_did = xiangyang_updated[xiangyang_updated['year'] >= 2010]['DID'].unique()

print(f"\n验证:")
print(f"  2010年前DID值: {pre_2010_did} (应该为0)")
print(f"  2010年起DID值: {post_2010_did} (应该为1)")

if len(pre_2010_did) == 1 and pre_2010_did[0] == 0:
    print("  [OK] 2010年前DID正确")
else:
    print("  [警告] 2010年前DID有误")

if len(post_2010_did) == 1 and post_2010_did[0] == 1:
    print("  [OK] 2010年起DID正确")
else:
    print("  [警告] 2010年起DID有误")

# 5. 保存更新后的数据
print("\n步骤5: 保存更新后的数据...")
output_file = '总数据集2007-2023_仅含emission城市_更新DID.xlsx'
df.to_excel(output_file, index=False)
print(f"[OK] 已保存: {output_file}")

# 6. 统计信息
print("\n步骤6: 统计信息...")
print(f"\n总数据集:")
print(f"  城市数: {df['city_name'].nunique()}")
print(f"  观测数: {len(df)}")
print(f"  年份范围: {df['year'].min()} - {df['year'].max()}")

print(f"\n处理组城市 (Treat=1):")
treat_cities = df[df['Treat'] == 1]['city_name'].unique()
print(f"  数量: {len(treat_cities)}")

print(f"\nDID=1的观测:")
did_obs = df[df['DID'] == 1]
print(f"  数量: {len(did_obs)}")

# 7. 检查是否有其他湖北省城市需要更新
print("\n步骤7: 检查湖北省其他城市...")
hubei_mask = df['province'].str.contains('湖北', na=False)
hubei_cities = df[hubei_mask]['city_name'].unique()
print(f"湖北省所有城市 ({len(hubei_cities)}个):")
for city in sorted(hubei_cities):
    city_data = df[df['city_name'] == city]
    treat_val = city_data['Treat'].iloc[0] if len(city_data) > 0 else 0
    post_vals = city_data['Post'].unique()
    did_vals = city_data['DID'].unique()
    print(f"  {city}: Treat={treat_val}, Post={post_vals}, DID={did_vals}")

print("\n" + "=" * 80)
print("襄阳市DID变量更新完成！")
print("=" * 80)
