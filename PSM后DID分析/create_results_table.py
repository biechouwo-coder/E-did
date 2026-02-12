# -*- coding: utf-8 -*-
import pandas as pd

# 创建回归结果表格
results_data = {
    '变量': ['DID', 'ln_real_gdp', 'ln_人口密度', 'ln_金融发展水平', '第二产业占GDP比重'],
    '系数': [0.0851, -0.5473, -0.2550, 0.1279, 0.4768],
    '标准误': [0.0345, 0.2869, 0.1931, 0.0937, 0.2948],
    't值': [2.4698, -1.9074, -1.3208, 1.3658, 1.6176],
    'p值': [0.0136, 0.0567, 0.1868, 0.1722, 0.1059],
    '95%置信区间下限': [0.0175, -1.1101, -0.6337, -0.0558, -0.1013],
    '95%置信区间上限': [0.1527, 0.0155, 0.1237, 0.3117, 1.0550]
}

df_results = pd.DataFrame(results_data)

# 添加显著性标记
def add_stars(p):
    if p < 0.01:
        return '***'
    elif p < 0.05:
        return '**'
    elif p < 0.1:
        return '*'
    else:
        return ''

df_results['显著性'] = df_results['p值'].apply(add_stars)

# 保存到Excel
df_results.to_excel('回归结果表格_固定效应.xlsx', index=False)
print("回归结果表格已保存至: 回归结果表格_固定效应.xlsx")
print("\n核心结果:")
print(df_results.to_string(index=False))
