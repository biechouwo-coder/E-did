# -*- coding: utf-8 -*-
"""
删除缩尾变量 industrial_advanced_winsorized 和 fdi_openness_winsorized
"""
import pandas as pd
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

print("="*70)
print("删除缩尾变量")
print("="*70)

# 读取数据
df = pd.read_excel('总数据集_2007-2023_完整版_DID_with_treat_log.xlsx')

winsorized_vars = ['industrial_advanced_winsorized', 'fdi_openness_winsorized']

print(f"\n原始数据集: {len(df)} 行 × {len(df.columns)} 列")
print(f"\n当前变量: {len(df.columns)} 个")

# 检查是否存在缩尾变量
existing_winsorized = [v for v in winsorized_vars if v in df.columns]

if existing_winsorized:
    print(f"\n发现缩尾变量: {existing_winsorized}")

    # 删除缩尾变量
    df = df.drop(columns=existing_winsorized)

    print(f"✓ 已删除缩尾变量")
    print(f"✓ 删除后变量数: {len(df.columns)}")

    # 保存数据
    output_file = '总数据集_2007-2023_完整版_DID_with_treat_log.xlsx'
    df.to_excel(output_file, index=False)

    print(f"\n✓ 数据已保存至: {output_file}")
    print(f"  总列数: {len(df.columns)}")

else:
    print(f"\n当前数据集中没有缩尾变量")
    print(f"  查找的变量: {winsorized_vars}")

print("\n" + "="*70)
print("删除完成！")
print("="*70)
