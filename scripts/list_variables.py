# -*- coding: utf-8 -*-
import pandas as pd
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

df = pd.read_excel('总数据集_2007-2023_完整版_DID_with_treat_log.xlsx')

print("="*70)
print("数据集变量列表")
print("="*70)
print(f"\n文件: 总数据集_2007-2023_完整版_DID_with_treat_log.xlsx")
print(f"总观测数: {len(df)}")
print(f"总列数: {len(df.columns)}")
print(f"年份范围: {df['year'].min()} - {df['year'].max()}")
print(f"城市数量: {df['city_name'].nunique()}")

print("\n" + "="*70)
print("变量列表（按类别分组）")
print("="*70)

# 按类别分组
identifiers = ['year', 'city_name', 'city_code', 'treat']
did_vars = ['DID']
pilot_vars = ['pilot_year']

original_vars = [
    'population',
    'pop_density',
    'gdp_real',
    'gdp_per_capita',
    'gdp_deflator',
    'carbon_intensity',
    'tertiary_share',
    'industrial_upgrading',
    'road_area',
    'financial_development',
    'nominal_gdp',
]

transformed_vars = [
    'ln_road_area',
    'ln_pop_density',
    'ln_pgdp',
    'ln_pop',
    'ln_carbon_intensity',
]

advanced_vars = [
    'industrial_advanced',
    'ln_industrial_advanced',
    'fdi',
    'fdi_openness',
    'ln_fdi_openness',
]

other_vars = [
    'tertiary_share_sq',
    'industrial_advanced_winsorized',
    'fdi_openness_winsorized',
]

# 打印各类变量
categories = [
    ("1. 标识变量（Identifiers）", identifiers),
    ("2. DID变量（政策变量）", did_vars),
    ("3. 试点年份（Pilot Year）", pilot_vars),
    ("4. 原始变量（Original Variables）", original_vars),
    ("5. 对数转换变量（Log Transformed）", transformed_vars),
    ("6. 产业高级化与FDI变量", advanced_vars),
    ("7. 其他变量", other_vars),
]

for category, vars_list in categories:
    existing_vars = [v for v in vars_list if v in df.columns]
    if existing_vars:
        print(f"\n{category}:")
        for var in existing_vars:
            missing = df[var].isnull().sum()
            missing_pct = missing / len(df) * 100
            print(f"  • {var:30s} (缺失: {missing:4d} / {len(df):4d} = {missing_pct:5.1f}%)")

print("\n" + "="*70)
print("变量说明")
print("="*70)

descriptions = {
    'year': '年份',
    'city_name': '城市名称',
    'city_code': '城市代码',
    'treat': '是否在试点名单中（1=是，0=否）',
    'DID': '双重差分变量（1=已实施政策，0=未实施）',
    'pilot_year': '该城市试点实施年份',
    'population': '人口',
    'pop_density': '人口密度',
    'gdp_real': '实际GDP',
    'gdp_per_capita': '人均GDP',
    'gdp_deflator': 'GDP平减指数',
    'carbon_intensity': '碳强度',
    'tertiary_share': '第三产业占比',
    'industrial_upgrading': '产业结构升级',
    'road_area': '道路面积',
    'financial_development': '金融发展水平',
    'nominal_gdp': '名义GDP',
    'ln_road_area': '道路面积（对数）',
    'ln_pop_density': '人口密度（对数）',
    'ln_pgdp': '人均GDP（对数）',
    'ln_pop': '人口（对数）',
    'ln_carbon_intensity': '碳强度（对数）',
    'industrial_advanced': '产业高级化（原始）',
    'ln_industrial_advanced': '产业高级化（对数）★推荐',
    'industrial_advanced_winsorized': '产业高级化（1%缩尾）',
    'fdi': '外商直接投资',
    'fdi_openness': 'FDI开放度（原始）',
    'ln_fdi_openness': 'FDI开放度（对数）★推荐',
    'fdi_openness_winsorized': 'FDI开放度（1%缩尾）',
    'tertiary_share_sq': '第三产业占比的平方',
}

print("\n核心变量（带★标记的为推荐使用）:\n")
for var, desc in descriptions.items():
    if var in df.columns:
        star = " ★" if var in ['ln_industrial_advanced', 'ln_fdi_openness'] else ""
        print(f"  {var:30s}: {desc}{star}")

print("\n" + "="*70)
print("数据集版本说明")
print("="*70)

datasets = [
    ("总数据集_2007-2023_完整版_DID_with_treat.xlsx", "原始版本（含DID和treat变量）"),
    ("总数据集_2007-2023_完整版_DID_with_treat_imputed.xlsx", "插值版本（缺失值已插补）"),
    ("总数据集_2007-2023_完整版_DID_with_treat_winsorized.xlsx", "缩尾版本（含1%缩尾变量）"),
    ("总数据集_2007-2023_完整版_DID_with_treat_log.xlsx", "对数版本（含对数转换变量）★推荐"),
]

print("\n可用数据集:\n")
for file, desc in datasets:
    print(f"  • {file}")
    print(f"    {desc}")

print("\n推荐使用: 总数据集_2007-2023_完整版_DID_with_treat_log.xlsx")
print("="*70)
