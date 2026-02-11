import pandas as pd
import numpy as np

print("=" * 80)
print("更新总数据集中拉萨市的DID变量")
print("=" * 80)

# 1. 读取数据
print("\n步骤1: 读取总数据集...")
file_path = '总数据集2007-2023_仅含emission城市.xlsx'
df = pd.read_excel(file_path)
print(f"数据集形状: {df.shape}")

# 2. 找到拉萨市
print("\n步骤2: 查找拉萨市...")
# 获取所有唯一城市
cities = df['city_name'].unique()
print(f"总城市数: {len(cities)}")

# 方法1：直接搜索包含"拉萨"的城市名
lhasa_mask = df['city_name'].str.contains('拉萨', na=False)
lhasa_count = lhasa_mask.sum()

# 方法2：如果方法1失败，尝试其他可能的名称
if lhasa_count == 0:
    # 尝试使用索引（根据前面的探索，拉萨在索引23）
    lhasa_city_name = cities[23]  # 从探索得知索引23是拉萨
    lhasa_mask = df['city_name'] == lhasa_city_name
    lhasa_count = lhasa_mask.sum()
    print(f"使用索引方法找到城市: {repr(lhasa_city_name)}")

print(f"拉萨市观测数: {lhasa_count}")

if lhasa_count == 0:
    print("错误: 未找到拉萨市！")
    exit(1)

# 显示拉萨市当前数据
lhasa_data = df[lhasa_mask].copy()
print(f"拉萨市年份范围: {lhasa_data['year'].min()} - {lhasa_data['year'].max()}")
print(f"拉萨市当前Treat值: {lhasa_data['Treat'].unique()}")
print(f"拉萨市当前Post值: {lhasa_data['Post'].unique()}")
print(f"拉萨市当前DID值: {lhasa_data['DID'].unique()}")

# 3. 更新拉萨市的DID变量
print("\n步骤3: 更新拉萨市的DID变量...")
print("规则: 拉萨市从2017年起受政策影响")
print("  - Treat = 1 (拉萨是处理组城市)")
print("  - Post = 1 if year >= 2017 else 0")
print("  - DID = Treat × Post")

# 更新Treat（拉萨是处理组）
df.loc[lhasa_mask, 'Treat'] = 1
print("  [OK] Treat = 1")

# 更新Post（2017年起为1）
post_mask = lhasa_mask & (df['year'] >= 2017)
df.loc[post_mask, 'Post'] = 1
print("  [OK] Post = 1 (year >= 2017)")

# 确保之前的Post=0
pre_post_mask = lhasa_mask & (df['year'] < 2017)
df.loc[pre_post_mask, 'Post'] = 0

# 更新DID
df.loc[lhasa_mask, 'DID'] = df.loc[lhasa_mask, 'Treat'] * df.loc[lhasa_mask, 'Post']
print("  [OK] DID = Treat × Post")

# 4. 验证更新结果
print("\n步骤4: 验证更新结果...")
lhasa_updated = df[lhasa_mask].copy()

print("\n拉萨市2016-2019年的数据:")
print(lhasa_updated[lhasa_updated['year'].between(2016, 2019)][['city_name', 'year', 'Treat', 'Post', 'DID']].to_string(index=False))

# 验证
pre_2017_did = lhasa_updated[lhasa_updated['year'] < 2017]['DID'].unique()
post_2017_did = lhasa_updated[lhasa_updated['year'] >= 2017]['DID'].unique()

print(f"\n验证:")
print(f"  2017年前DID值: {pre_2017_did} (应该为0)")
print(f"  2017年起DID值: {post_2017_did} (应该为1)")

if len(pre_2017_did) == 1 and pre_2017_did[0] == 0:
    print("  [OK] 2017年前DID正确")
else:
    print("  [警告] 2017年前DID有误")

if len(post_2017_did) == 1 and post_2017_did[0] == 1:
    print("  [OK] 2017年起DID正确")
else:
    print("  [警告] 2017年起DID有误")

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

print("\n" + "=" * 80)
print("拉萨市DID变量更新完成！")
print("=" * 80)
