# -*- coding: utf-8 -*-
"""
基期PSM分析 (2009年基期) - 简化版
使用变量: ln_real_gdp, ln_人口密度, ln_金融发展水平, 第二产业占GDP比重
卡尺: 倾向得分对数几率标准差的0.25倍
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from scipy import stats
import warnings
import sys
import io

# 设置标准输出编码为UTF-8
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
warnings.filterwarnings('ignore')

def main():
    print("="*60)
    print("基期PSM分析 (2009年基期)")
    print("="*60)

    # ============================================================
    # 1. 加载数据
    # ============================================================
    print("\n1. 加载数据")
    print("-"*60)

    data_path = "../总数据集2007-2023_仅含emission城市_更新DID.xlsx"
    df = pd.read_excel(data_path, engine='openpyxl')
    print(f"✓ 数据加载成功: {df.shape[0]}行 × {df.shape[1]}列")

    # 显示列名（前40列）
    print("\n列名（前40列）:")
    for i, col in enumerate(df.columns[:40]):
        print(f"  {i+1:2d}. {col}")

    # ============================================================
    # 2. 筛选2009年基期数据
    # ============================================================
    print("\n2. 筛选2009年基期数据")
    print("-"*60)

    # 查找年份列
    year_col = None
    for col in df.columns:
        if '年' in col or 'year' in col.lower() or 'Year' in col:
            year_col = col
            break

    if year_col is None and len(df.columns) >= 3:
        year_col = df.columns[2]  # 假设第3列是年份

    print(f"年份列: {year_col}")

    # 筛选2009年
    df_baseline = df[df[year_col] == 2009].copy()
    print(f"✓ 基期样本数: {len(df_baseline)}")

    # ============================================================
    # 3. 检测变量
    # ============================================================
    print("\n3. 检测分析变量")
    print("-"*60)

    # 处理变量
    treatment_var = None
    for col in df_baseline.columns:
        if 'treat' in col.lower() or '处理' in col or 'group' in col.lower():
            treatment_var = col
            break

    if treatment_var is None:
        print("请输入处理变量名（如 treat, group 等）:")
        treatment_var = input().strip()

    print(f"✓ 处理变量: {treatment_var}")

    # 匹配变量
    covariates = []

    # ln_real_gdp (使用对数形式)
    for col in df_baseline.columns:
        if 'ln_real_gdp' == col or 'ln_gdp' in col.lower():
            covariates.append(col)
            print(f"✓ ln_real_gdp: {col}")
            break

    # ln_人口密度 (使用对数形式)
    for col in df_baseline.columns:
        if 'ln_人口密度' == col or 'ln_pop' in col.lower():
            covariates.append(col)
            print(f"✓ ln_人口密度: {col}")
            break

    # ln_金融发展水平 (使用对数形式)
    for col in df_baseline.columns:
        if 'ln_金融发展水平' == col or 'ln_finance' in col.lower():
            covariates.append(col)
            print(f"✓ ln_金融发展水平: {col}")
            break

    # 第二产业占GDP比重 (不需要对数)
    for col in df_baseline.columns:
        if '第二产业占GDP比重' == col or col == '第二产业占GDP比重':
            covariates.append(col)
            print(f"✓ 第二产业占GDP比重: {col}")
            break

    if len(covariates) < 4:
        print(f"\n警告: 仅找到 {len(covariates)}/4 个匹配变量")
        print("已找到:", covariates)

    # ============================================================
    # 4. 准备分析数据
    # ============================================================
    print("\n4. 准备分析数据")
    print("-"*60)

    analysis_vars = [treatment_var] + covariates
    df_analysis = df_baseline[analysis_vars].dropna().copy()
    print(f"✓ 有效样本数: {len(df_analysis)}")

    # 统计
    n_treat = int(df_analysis[treatment_var].sum())
    n_control = len(df_analysis) - n_treat
    print(f"\n样本分布:")
    print(f"  处理组: {n_treat}")
    print(f"  控制组: {n_control}")

    # ============================================================
    # 5. 估计倾向得分
    # ============================================================
    print("\n5. 估计倾向得分 (Logit模型)")
    print("-"*60)

    X = df_analysis[covariates].values
    y = df_analysis[treatment_var].values

    # 标准化协变量
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Logistic回归
    logit_model = LogisticRegression(solver='lbfgs', max_iter=1000)
    logit_model.fit(X_scaled, y)

    # 获取倾向得分
    propensity_scores = logit_model.predict_proba(X_scaled)[:, 1]

    print(f"✓ 倾向得分统计:")
    print(f"  均值: {propensity_scores.mean():.4f}")
    print(f"  标准差: {propensity_scores.std():.4f}")
    print(f"  最小值: {propensity_scores.min():.4f}")
    print(f"  最大值: {propensity_scores.max():.4f}")

    # ============================================================
    # 6. 计算卡尺并执行匹配
    # ============================================================
    print("\n6. 执行PSM匹配")
    print("-"*60)

    # 计算logit倾向得分
    eps = 1e-10
    logit_scores = np.log((propensity_scores + eps) / (1 - propensity_scores + eps))
    logit_std = logit_scores.std()

    # 卡尺 = 0.25 * logit倾向得分标准差
    caliper = 0.25 * logit_std

    print(f"卡尺设定:")
    print(f"  Logit倾向得分标准差: {logit_std:.4f}")
    print(f"  卡尺系数: 0.25")
    print(f"  卡尺值: {caliper:.4f}")

    # 执行1:1最近邻匹配
    treat_indices = np.where(y == 1)[0]
    control_indices = np.where(y == 0)[0]

    match_results = []

    for treat_idx in treat_indices:
        treat_logit = logit_scores[treat_idx]

        # 计算与所有控制组的logit距离
        control_logits = logit_scores[control_indices]
        distances = np.abs(control_logits - treat_logit)

        # 找到满足卡尺条件的最近邻
        valid_matches = np.where(distances <= caliper)[0]

        if len(valid_matches) > 0:
            best_idx = valid_matches[np.argmin(distances[valid_matches])]
            match_results.append({
                'treat_idx': treat_idx,
                'control_idx': control_indices[best_idx],
                'distance': distances[best_idx],
                'treat_score': propensity_scores[treat_idx],
                'control_score': propensity_scores[control_indices[best_idx]]
            })

    matched_pairs = pd.DataFrame(match_results)

    n_matched = len(matched_pairs)
    match_rate = n_matched / n_treat * 100 if n_treat > 0 else 0

    print(f"\n✓ 匹配结果:")
    print(f"  成功匹配对数: {n_matched}/{n_treat}")
    print(f"  匹配成功率: {match_rate:.2f}%")
    print(f"  平均匹配距离: {matched_pairs['distance'].mean():.4f}")

    # ============================================================
    # 7. 平衡性检验
    # ============================================================
    print("\n7. 平衡性检验")
    print("-"*60)

    balance_results = []

    for i, covar in enumerate(covariates):
        # 匹配前
        treat_before = df_analysis[df_analysis[treatment_var] == 1][covar]
        control_before = df_analysis[df_analysis[treatment_var] == 0][covar]

        mean_treat_b = treat_before.mean()
        mean_control_b = control_before.mean()

        # 标准化偏差 (%)
        pooled_std_b = np.sqrt((treat_before.var() + control_before.var()) / 2)
        bias_before = (mean_treat_b - mean_control_b) / pooled_std_b * 100 if pooled_std_b > 0 else 0

        # t检验
        t_b, p_b = stats.ttest_ind(treat_before, control_before)

        # 匹配后
        treat_idx_list = matched_pairs['treat_idx'].values
        control_idx_list = matched_pairs['control_idx'].values

        treat_after = df_analysis.iloc[treat_idx_list][covar]
        control_after = df_analysis.iloc[control_idx_list][covar]

        mean_treat_a = treat_after.mean()
        mean_control_a = control_after.mean()

        pooled_std_a = np.sqrt((treat_after.var() + control_after.var()) / 2)
        bias_after = (mean_treat_a - mean_control_a) / pooled_std_a * 100 if pooled_std_a > 0 else 0

        t_a, p_a = stats.ttest_ind(treat_after, control_after)

        # 偏差削减
        bias_reduction = ((bias_before - bias_after) / abs(bias_before) * 100) if abs(bias_before) > 0 else 0

        balance_results.append({
            '变量': covar,
            '处理组均值_匹配前': mean_treat_b,
            '控制组均值_匹配前': mean_control_b,
            '偏差%_匹配前': bias_before,
            'P值_匹配前': p_b,
            '处理组均值_匹配后': mean_treat_a,
            '控制组均值_匹配后': mean_control_a,
            '偏差%_匹配后': bias_after,
            'P值_匹配后': p_a,
            '偏差削减%': bias_reduction
        })

    balance_df = pd.DataFrame(balance_results)

    print("\n平衡性检验结果:")
    print(balance_df.to_string(index=False))

    # 保存
    balance_df.to_csv('基期PSM平衡性检验结果_2009.csv', index=False, encoding='utf-8-sig')
    print(f"\n✓ 结果已保存: 基期PSM平衡性检验结果_2009.csv")

    # ============================================================
    # 8. 导出匹配数据
    # ============================================================
    print("\n8. 导出匹配数据")
    print("-"*60)

    if n_matched > 0:
        matched_data = []

        for i, (idx, row) in enumerate(matched_pairs.iterrows()):
            # 处理组
            t_row = df_analysis.iloc[int(row['treat_idx'])].copy()
            t_row['匹配ID'] = i
            t_row['组别'] = '处理组'
            matched_data.append(t_row)

            # 控制组
            c_row = df_analysis.iloc[int(row['control_idx'])].copy()
            c_row['匹配ID'] = i
            c_row['组别'] = '控制组'
            matched_data.append(c_row)

        matched_df = pd.DataFrame(matched_data)
        matched_df.to_excel('PSM匹配数据_2009.xlsx', index=False, engine='openpyxl')
        print(f"✓ 匹配数据已导出: PSM匹配数据_2009.xlsx")

    # ============================================================
    # 9. 生成报告
    # ============================================================
    print("\n" + "="*60)
    print("PSM分析汇总报告")
    print("="*60)

    report = f"""
{'='*60}
基期PSM分析报告 (2009年基期)
{'='*60}

一、分析设置
{'─'*60}
  基期年份: 2009
  匹配变量:
"""
    for i, var in enumerate(covariates, 1):
        report += f"    {i}. {var}\n"

    report += f"""
  卡尺: 倾向得分对数几率标准差的0.25倍
  卡尺值: {caliper:.4f}
  匹配方法: 1:1 最近邻匹配

二、样本统计
{'─'*60}
  基期样本数: {len(df_baseline)}
  有效样本数: {len(df_analysis)}
  处理组: {n_treat}
  控制组: {n_control}

三、匹配结果
{'─'*60}
  成功匹配对数: {n_matched}
  匹配成功率: {match_rate:.2f}%
  平均匹配距离: {matched_pairs['distance'].mean():.4f}

四、平衡性检验
{'─'*60}
  见: 基期PSM平衡性检验结果_2009.csv

{'='*60}
生成的文件:
  1. 基期PSM平衡性检验结果_2009.csv
  2. PSM匹配数据_2009.xlsx
  3. PSM分析报告_2009.txt
{'='*60}
"""

    print(report)

    with open('PSM分析报告_2009.txt', 'w', encoding='utf-8') as f:
        f.write(report)
    print("✓ 报告已保存: PSM分析报告_2009.txt")

    print("\n" + "="*60)
    print("✓ PSM分析完成！")
    print("="*60)

if __name__ == "__main__":
    main()
