import pandas as pd
import numpy as np

print("=" * 80)
print("更新总数据集中儋州市的DID变量")
print("=" * 80)

# 1. 读取数据
print("\n步骤1: 读取总数据集...")
file_path = '总数据集2007-2023_仅含emission城市.xlsx'
df = pd.read_excel(file_path)
print(f"数据集形状: {df.shape}")

# 2. 查找儋州市
print("\n步骤2: 查找儋州市...")
cities = df['city_name'].unique()
print(f"总城市数: {len(cities)}")

# 搜索包含"儋州"的城市
danzhou_mask = df['city_name'].str.contains('儋州', na=False)
danzhou_count = danzhou_mask.sum()

if danzhou_count == 0:
    print("未找到儋州市，检查海南省其他城市...")
    # 检查海南省的城市
    hainan_mask = df['province'].str.contains('海南', na=False)
    hainan_cities = df[hainan_mask]['city_name'].unique()
    print(f"海南省城市: {list(hainan_cities)}")
    exit(1)

print(f"找到儋州市观测数: {danzhou_count}")

# 显示儋州市当前数据
danzhou_data = df[danzhou_mask].copy()
print(f"儋州市年份范围: {danzhou_data['year'].min()} - {danzhou_data['year'].max()}")
print(f"儋州市当前Treat值: {danzhou_data['Treat'].unique()}")
print(f"儋州市当前Post值: {danzhou_data['Post'].unique()}")
print(f"儋州市当前DID值: {danzhou_data['DID'].unique()}")

# 3. 更新儋州市的DID变量
print("\n步骤3: 更新儋州市的DID变量...")
print("规则: 儋州市从2012年起受政策影响（海南省是全省试点）")
print("  - Treat = 1 (儋州市是处理组城市)")
print("  - Post = 1 if year >= 2012 else 0")
print("  - DID = Treat × Post")

# 更新Treat（儋州是处理组）
df.loc[danzhou_mask, 'Treat'] = 1
print("  [OK] Treat = 1")

# 更新Post（2012年起为1）
post_mask = danzhou_mask & (df['year'] >= 2012)
df.loc[post_mask, 'Post'] = 1
print("  [OK] Post = 1 (year >= 2012)")

# 确保之前的Post=0
pre_post_mask = danzhou_mask & (df['year'] < 2012)
df.loc[pre_post_mask, 'Post'] = 0

# 更新DID
df.loc[danzhou_mask, 'DID'] = df.loc[danzhou_mask, 'Treat'] * df.loc[danzhou_mask, 'Post']
print("  [OK] DID = Treat × Post")

# 4. 验证更新结果
print("\n步骤4: 验证更新结果...")
danzhou_updated = df[danzhou_mask].copy()

print("\n儋州市2011-2014年的数据:")
print(danzhou_updated[danzhou_updated['year'].between(2011, 2014)][['city_name', 'year', 'Treat', 'Post', 'DID']].to_string(index=False))

# 验证
pre_2012_did = danzhou_updated[danzhou_updated['year'] < 2012]['DID'].unique()
post_2012_did = danzhou_updated[danzhou_updated['year'] >= 2012]['DID'].unique()

print(f"\n验证:")
print(f"  2012年前DID值: {pre_2012_did} (应该为0)")
print(f"  2012年起DID值: {post_2012_did} (应该为1)")

if len(pre_2012_did) == 1 and pre_2012_did[0] == 0:
    print("  [OK] 2012年前DID正确")
else:
    print("  [警告] 2012年前DID有误")

if len(post_2012_did) == 1 and post_2012_did[0] == 1:
    print("  [OK] 2012年起DID正确")
else:
    print("  [警告] 2012年起DID有误")

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

# 7. 检查海南省所有城市
print("\n步骤7: 检查海南省所有城市...")
hainan_mask = df['province'].str.contains('海南', na=False)
hainan_cities = df[hainan_mask]['city_name'].unique()
print(f"海南省所有城市 ({len(hainan_cities)}个):")
for city in sorted(hainan_cities):
    city_data = df[df['city_name'] == city]
    treat_val = city_data['Treat'].iloc[0] if len(city_data) > 0 else 0
    post_vals = city_data['Post'].unique()
    did_vals = city_data['DID'].unique()
    print(f"  {city}: Treat={treat_val}, Post={post_vals}, DID={did_vals}")

print("\n" + "=" * 80)
print("儋州市DID变量更新完成！")
print("=" * 80)
print("\n说明: 海南省是全省试点（2012年），海口、三亚、儋州均为处理组城市")
