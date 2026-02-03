# -*- coding: utf-8 -*-
import pandas as pd
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

df = pd.read_excel('总数据集_2007-2023_完整版_DID_with_treat.xlsx')
base = df[df['year']==2009]

print('2009年原始数据:')
print(f'  总观测: {len(base)}')
print(f'  处理组(treat=1): {(base["treat"]==1).sum()}')
print(f'  对照组(treat=0): {(base["treat"]==0).sum()}')

print('\n控制变量缺失值（按组别）:')
vars = ['gdp_per_capita', 'pop_density', 'industrial_upgrading', 'ln_fdi']

for v in vars:
    missing_treat = base[(base['treat']==1)][v].isnull().sum()
    missing_ctrl = base[(base['treat']==0)][v].isnull().sum()
    print(f'\n{v}:')
    print(f'  处理组缺失: {missing_treat} / {(base["treat"]==1).sum()}')
    print(f'  对照组缺失: {missing_ctrl} / {(base["treat"]==0).sum()}')

# 统计删除缺失值后的情况
print('\n' + '='*60)
print('删除缺失值后的情况:')
base_clean = base.dropna(subset=vars + ['treat', 'city_name'])
print(f'  清理后总观测: {len(base_clean)}')
print(f'  清理后处理组: {(base_clean["treat"]==1).sum()}')
print(f'  清理后对照组: {(base_clean["treat"]==0).sum()}')

print('\n删除的观测:')
print(f'  处理组被删除: {(base["treat"]==1).sum() - (base_clean["treat"]==1).sum()}')
print(f'  对照组被删除: {(base["treat"]==0).sum() - (base_clean["treat"]==0).sum()}')
