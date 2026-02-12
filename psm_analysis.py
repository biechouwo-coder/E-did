# -*- coding: utf-8 -*-
"""
PSM分析脚本 - 2009年基期
使用四个变量: ln_real_gdp, ln_人口密度, ln_金融发展水平, 第二产业占GDP比重
卡尺: 倾向得分对数几率标准差的0.25倍
匹配: 1:1有放回匹配
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def load_data(filepath):
    """加载数据"""
    df = pd.read_excel(filepath)
    print(f"数据加载成功，共 {len(df)} 行, {len(df.columns)} 列")

    # 修复编码问题 - 重新映射列名
    # 检查并重命名列（处理编码问题）
    column_mapping = {}
    for col in df.columns:
        if 'ln_' in str(col) and '人口' in str(col):
            column_mapping[col] = 'ln_pop_density'
        elif 'ln_' in str(col) and '金融' in str(col):
            column_mapping[col] = 'ln_financial_dev'
        elif '第二产业' in str(col):
            column_mapping[col] = 'secondary_industry_share'
        elif 'ln_real_gdp' not in str(col) and 'real_gdp' in str(col) and 'ln' in str(col):
            column_mapping[col] = 'ln_real_gdp'

    df = df.rename(columns=column_mapping)

    print(f"可用列: {df.columns.tolist()}\n")

    return df

def prepare_psm_data(df, base_year=2009):
    """准备PSM分析数据 - 提取基期数据"""
    df_base = df[df['year'] == base_year].copy()
    print(f"基期 {base_year} 年数据: {len(df_base)} 个城市")

    # 检查Treat变量
    if 'Treat' in df_base.columns:
        treat_count = df_base['Treat'].sum()
        control_count = len(df_base) - treat_count
        print(f"  - 处理组城市: {treat_count:.0f} 个")
        print(f"  - 控制组城市: {control_count:.0f} 个\n")
    else:
        print("警告: 数据中没有 'Treat' 列")

    return df_base

def calculate_propensity_scores(df_base, covariates):
    """计算倾向得分"""

    # 删除有缺失值的行
    df_clean = df_base.dropna(subset=covariates + ['Treat']).copy()

    print(f"PSM分析样本数: {len(df_clean)} 个城市 (删除缺失值后)")

    X = df_clean[covariates].values
    y = df_clean['Treat'].values

    # 标准化协变量
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 使用逻辑回归计算倾向得分（不使用class_weight以获得准确的倾向得分）
    logit = LogisticRegression(max_iter=1000, random_state=42)
    logit.fit(X_scaled, y)

    # 获取倾向得分
    propensity_scores = logit.predict_proba(X_scaled)[:, 1]

    df_clean['propensity_score'] = propensity_scores
    df_clean['logit_ps'] = np.log(propensity_scores / (1 - propensity_scores + 1e-10))

    print(f"\n倾向得分统计:")
    print(df_clean['propensity_score'].describe())

    return df_clean, logit, scaler

def perform_matching(df_psm, caliper_mult=0.25, ratio=1, replace=True):
    """
    执行匹配

    参数:
    - caliper_mult: 卡尺倍数 (默认0.25，即倾向得分对数几率标准差的0.25倍)
    - ratio: 匹配比例 (默认1，即1:1匹配)
    - replace: 是否有放回匹配 (默认True)
    """
    print(f"\n{'='*60}")
    print(f"匹配参数:")
    print(f"  - 卡尺: {caliper_mult} × 倾向得分对数几率标准差")
    print(f"  - 匹配比例: 1:{ratio}")
    print(f"  - 有放回匹配: {'是' if replace else '否'}")
    print(f"{'='*60}\n")

    # 计算卡尺
    caliper = caliper_mult * df_psm['logit_ps'].std()
    print(f"实际卡尺值: {caliper:.4f}\n")

    treated = df_psm[df_psm['Treat'] == 1].copy()
    control = df_psm[df_psm['Treat'] == 0].copy()

    matched_controls = []
    matches_info = []

    for idx, treated_row in treated.iterrows():
        treated_ps = treated_row['propensity_score']
        treated_logit = treated_row['logit_ps']

        # 计算距离（使用倾向得分对数几率）
        control['distance'] = np.abs(control['logit_ps'] - treated_logit)

        # 应用卡尺限制
        control_within_caliper = control[control['distance'] <= caliper].copy()

        if len(control_within_caliper) == 0:
            # 没有在卡尺内的控制组样本
            matches_info.append({
                'treated_city': treated_row.get('city_name', idx),
                'matched': False,
                'control_city': None,
                'distance': None
            })
            continue

        # 找到最近的控制组样本
        if replace:
            # 有放回: 从所有控制组中选择最近的
            best_match = control_within_caliper.nsmallest(1, 'distance')
        else:
            # 无放回: 从未匹配的控制组中选择
            unmatched_controls = control_within_caliper[~control_within_caliper.index.isin([m['control_idx'] for m in matches_info if m['control_idx'] is not None])]
            if len(unmatched_controls) == 0:
                matches_info.append({
                    'treated_city': treated_row.get('city_name', idx),
                    'matched': False,
                    'control_city': None,
                    'distance': None
                })
                continue
            best_match = unmatched_controls.nsmallest(1, 'distance')

        matched_controls.append(best_match.iloc[0])
        matches_info.append({
            'treated_city': treated_row.get('city_name', idx),
            'treated_idx': idx,
            'matched': True,
            'control_city': best_match.iloc[0].get('city_name', best_match.index[0]),
            'control_idx': best_match.index[0],
            'distance': best_match.iloc[0]['distance']
        })

        # 如果无放回，移除已匹配的控制组
        if not replace:
            control = control.drop(best_match.index[0])

    # 合并处理组和匹配的控制组
    if matched_controls:
        df_matched_controls = pd.DataFrame(matched_controls)
        df_matched = pd.concat([treated, df_matched_controls])
    else:
        df_matched = treated.copy()

    # 统计匹配结果
    n_treated = len(treated)
    n_matched = len([m for m in matches_info if m['matched']])
    n_unmatched = n_treated - n_matched
    match_rate = n_matched / n_treated * 100 if n_treated > 0 else 0

    print(f"匹配结果:")
    print(f"  - 处理组样本数: {n_treated}")
    print(f"  - 成功匹配样本数: {n_matched}")
    print(f"  - 未匹配样本数: {n_unmatched}")
    print(f"  - 匹配成功率: {match_rate:.1f}%\n")

    # 计算平均距离
    distances = [m['distance'] for m in matches_info if m['distance'] is not None]
    if distances:
        print(f"匹配距离统计:")
        print(f"  - 平均距离: {np.mean(distances):.4f}")
        print(f"  - 中位数距离: {np.median(distances):.4f}")
        print(f"  - 最大距离: {np.max(distances):.4f}")
        print(f"  - 最小距离: {np.min(distances):.4f}\n")

    return df_matched, matches_info

def check_balance(df_matched, covariates):
    """检验匹配后的平衡性"""
    print(f"{'='*60}")
    print("平衡性检验")
    print(f"{'='*60}\n")

    treated = df_matched[df_matched['Treat'] == 1]
    control = df_matched[df_matched['Treat'] == 0]

    balance_results = []

    for var in covariates:
        treated_mean = treated[var].mean()
        control_mean = control[var].mean()
        treated_std = treated[var].std()
        control_std = control[var].std()

        # 标准化偏差
        pooled_std = np.sqrt((treated_std**2 + control_std**2) / 2)
        bias = 100 * (treated_mean - control_mean) / pooled_std if pooled_std > 0 else 0

        balance_results.append({
            'variable': var,
            'treated_mean': treated_mean,
            'control_mean': control_mean,
            'bias': bias
        })

    df_balance = pd.DataFrame(balance_results)

    print(df_balance.to_string(index=False))
    print(f"\n说明: 标准化偏差绝对值小于20%表示平衡性较好\n")

    return df_balance

def visualize_matching_results(df_psm, df_matched, matches_info, covariates):
    """可视化匹配结果"""

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # 1. 倾向得分分布
    ax = axes[0, 0]
    treated = df_matched[df_matched['Treat'] == 1]
    control = df_matched[df_matched['Treat'] == 0]

    ax.hist(treated['propensity_score'], bins=20, alpha=0.6, label='处理组', color='red', edgecolor='black')
    ax.hist(control['propensity_score'], bins=20, alpha=0.6, label='控制组', color='blue', edgecolor='black')
    ax.set_xlabel('倾向得分', fontsize=12)
    ax.set_ylabel('频数', fontsize=12)
    ax.set_title('匹配后倾向得分分布', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. 协变量平衡性
    ax = axes[0, 1]
    x_pos = np.arange(len(covariates))
    treated_means = [treated[var].mean() for var in covariates]
    control_means = [control[var].mean() for var in covariates]

    width = 0.35
    ax.bar(x_pos - width/2, treated_means, width, label='处理组', color='red', alpha=0.7)
    ax.bar(x_pos + width/2, control_means, width, label='控制组', color='blue', alpha=0.7)

    ax.set_xlabel('协变量', fontsize=12)
    ax.set_ylabel('均值', fontsize=12)
    ax.set_title('协变量均值对比 (匹配后)', fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(covariates, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # 3. 倾向得分对数几率分布
    ax = axes[1, 0]
    ax.hist(treated['logit_ps'], bins=20, alpha=0.6, label='处理组', color='red', edgecolor='black')
    ax.hist(control['logit_ps'], bins=20, alpha=0.6, label='控制组', color='blue', edgecolor='black')
    ax.set_xlabel('倾向得分对数几率', fontsize=12)
    ax.set_ylabel('频数', fontsize=12)
    ax.set_title('匹配后倾向得分对数几率分布', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. 匹配距离分布
    ax = axes[1, 1]
    distances = [m['distance'] for m in matches_info if m['distance'] is not None]
    if distances:
        ax.hist(distances, bins=20, color='green', edgecolor='black', alpha=0.7)
        ax.axvline(np.mean(distances), color='red', linestyle='--', linewidth=2, label=f'平均值: {np.mean(distances):.4f}')
        ax.axvline(np.median(distances), color='blue', linestyle='--', linewidth=2, label=f'中位数: {np.median(distances):.4f}')
    ax.set_xlabel('匹配距离 (倾向得分对数几率)', fontsize=12)
    ax.set_ylabel('频数', fontsize=12)
    ax.set_title('匹配距离分布', fontsize=14, fontweight='bold')
    if distances:
        ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('PSM匹配结果_2009基期_四变量.png', dpi=300, bbox_inches='tight')
    print("\n图表已保存为: PSM匹配结果_2009基期_四变量.png")
    plt.show()

def main():
    """主函数"""
    print("="*70)
    print("PSM分析 - 2009年基期 (四变量)")
    print("="*70)
    print("匹配变量:")
    print("  1. ln_real_gdp - 实际GDP对数")
    print("  2. ln_pop_density - 人口密度对数")
    print("  3. ln_financial_dev - 金融发展水平对数")
    print("  4. secondary_industry_share - 第二产业占GDP比重")
    print("="*70)
    print()

    # 1. 加载数据
    filepath = '总数据集2007-2019_含碳排放强度.xlsx'
    df = load_data(filepath)

    # 2. 准备PSM数据 (基期2009)
    df_base = prepare_psm_data(df, base_year=2009)

    # 3. 定义协变量
    covariates = ['ln_real_gdp', 'ln_pop_density', 'ln_financial_dev', 'secondary_industry_share']

    # 检查协变量是否存在
    available_covariates = [col for col in covariates if col in df_base.columns]
    if len(available_covariates) < len(covariates):
        print(f"警告: 部分协变量不可用。可用: {available_covariates}")

    if len(available_covariates) == 0:
        print("错误: 没有可用的协变量，退出分析")
        return

    # 打印协变量描述性统计
    print("\n协变量描述性统计 (匹配前):")
    print(df_base[available_covariates].describe().to_string())
    print()

    # 4. 计算倾向得分
    df_psm, logit_model, scaler = calculate_propensity_scores(df_base, available_covariates)

    # 5. 执行匹配 (卡尺=0.25, 1:1, 有放回)
    df_matched, matches_info = perform_matching(
        df_psm,
        caliper_mult=0.25,
        ratio=1,
        replace=True
    )

    # 6. 平衡性检验
    df_balance = check_balance(df_matched, available_covariates)

    # 7. 可视化
    visualize_matching_results(df_psm, df_matched, matches_info, available_covariates)

    # 8. 保存结果
    print("\n" + "="*70)
    print("PSM分析完成!")
    print("="*70)

    # 保存匹配后的数据
    output_cols = ['city_name', 'Treat', 'propensity_score', 'logit_ps'] + available_covariates
    df_matched_export = df_matched[[col for col in output_cols if col in df_matched.columns]].copy()
    df_matched_export.to_excel('PSM匹配结果_2009基期_四变量.xlsx', index=False)
    print("\n匹配后数据已保存为: PSM匹配结果_2009基期_四变量.xlsx")

    # 保存匹配信息
    df_matches = pd.DataFrame(matches_info)
    df_matches.to_excel('PSM匹配详情_2009基期_四变量.xlsx', index=False)
    print("匹配详情已保存为: PSM匹配详情_2009基期_四变量.xlsx")

    # 保存平衡性检验结果
    df_balance.to_excel('PSM平衡性检验_2009基期_四变量.xlsx', index=False)
    print("平衡性检验结果已保存为: PSM平衡性检验_2009基期_四变量.xlsx")

    print("\n分析完成!")

if __name__ == "__main__":
    main()
