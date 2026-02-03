# -*- coding: utf-8 -*-
import pandas as pd
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

df = pd.read_excel('总数据集_2007-2023_完整版_DID.xlsx')

print('列名:')
print(df.columns.tolist())

print('\n检查treat变量:')
print('treat' in df.columns)

if 'treat' in df.columns:
    print('\ntreat变量值分布:')
    print(df['treat'].value_counts())
    print('\ntreat变量的唯一值:', df['treat'].unique())
else:
    print('\n数据集中没有treat变量')
    print('需要基于city_name判断城市是否在低碳试点名单中')
