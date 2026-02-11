"""
基期PSM分析 (2009年基期)
使用变量: ln_real_gdp, ln_人口密度, ln_金融发展水平, 第二产业占GDP比重
卡尺: 倾向得分对数几率标准差的0.25倍
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

class PSMAnalyzer:
    """倾向得分匹配分析器"""

    def __init__(self, data_path):
        """初始化分析器"""
        self.data_path = data_path
        self.df = None
        self.df_baseline = None
        self.psm_model = None
        self.propensity_scores = None
        self.matched_pairs = None

    def load_data(self):
        """加载数据"""
        print("="*60)
        print("1. 加载数据")
        print("="*60)

        try:
            # 尝试读取Excel
            self.df = pd.read_excel(self.data_path, engine='openpyxl')
            print(f"✓ 成功读取数据: {self.df.shape}")
            print(f"  - 观测数: {self.df.shape[0]}")
            print(f"  - 变量数: {self.df.shape[1]}")
        except Exception as e:
            print(f"✗ 读取失败: {e}")
            raise

        # 显示前几列
        print("\n数据列名（前20列）:")
        for i, col in enumerate(self.df.columns[:20]):
            print(f"  {i+1}. {col}")

        return self.df

    def prepare_baseline_data(self, year_col='年份', baseline_year=2009):
        """准备基期数据"""
        print("\n" + "="*60)
        print("2. 准备基期数据 (2009年)")
        print("="*60)

        # 检测年份列名
        possible_year_cols = [col for col in self.df.columns
                             if '年' in col or 'year' in col.lower() or 'Year' in col]
        print(f"可能的年份列: {possible_year_cols}")

        if possible_year_cols:
            year_col = possible_year_cols[0]

        # 筛选基期数据
        self.df_baseline = self.df[self.df[year_col] == baseline_year].copy()
        print(f"✓ 基期样本数: {len(self.df_baseline)}")

        # 检测处理组变量
        possible_treat_cols = [col for col in self.df.columns
                              if 'treat' in col.lower() or '处理' in col or 'group' in col.lower()]
        print(f"可能的处理变量: {possible_treat_cols}")

        return self.df_baseline

    def run_psm(self, treatment_var, covariates, caliper=0.25):
        """
        运行PSM分析

        参数:
        - treatment_var: 处理组变量名
        - covariates: 匹配变量列表
        - caliper: 卡尺（默认为logit倾向得分标准差的0.25倍）
        """
        print("\n" + "="*60)
        print("3. 运行倾向得分匹配 (PSM)")
        print("="*60)

        df = self.df_baseline.copy()

        # 删除缺失值
        analysis_vars = [treatment_var] + covariates
        df_analysis = df[analysis_vars].dropna()
        print(f"✓ 有效样本数: {len(df_analysis)}")

        # 分离处理组和控制组
        treatment = df_analysis[treatment_var].values
        X = df_analysis[covariates].values

        n_treat = int(treatment.sum())
        n_control = len(treatment) - n_treat
        print(f"\n样本分布:")
        print(f"  - 处理组: {n_treat}")
        print(f"  - 控制组: {n_control}")

        # 1. 估计倾向得分 (Logit模型)
        print(f"\n步骤1: 估计倾向得分")
        X_const = sm.add_constant(X)
        logit_model = sm.Logit(treatment, X_const)
        self.psm_model = logit_model.fit(disp=0)

        # 获取倾向得分
        propensity_scores = self.psm_model.predict(X_const)
        self.propensity_scores = propensity_scores

        print(f"✓ 倾向得分统计:")
        print(f"  - 均值: {propensity_scores.mean():.4f}")
        print(f"  - 标准差: {propensity_scores.std():.4f}")
        print(f"  - 最小值: {propensity_scores.min():.4f}")
        print(f"  - 最大值: {propensity_scores.max():.4f}")

        # 计算logit倾向得分
        logit_scores = np.log(propensity_scores / (1 - propensity_scores) + 1e-10)
        logit_std = logit_scores.std()

        # 2. 计算卡尺
        caliper_value = caliper * logit_std
        print(f"\n步骤2: 设定卡尺")
        print(f"  - Logit倾向得分标准差: {logit_std:.4f}")
        print(f"  - 卡尺系数: {caliper}")
        print(f"  - 卡尺值: {caliper_value:.4f}")

        # 3. 执行匹配 (1:1 最近邻匹配，带卡尺)
        print(f"\n步骤3: 执行匹配 (1:1 最近邻匹配)")
        matched_control = []
        matched_treat = []

        treat_indices = np.where(treatment == 1)[0]
        control_indices = np.where(treatment == 0)[0]

        match_results = []

        for treat_idx in treat_indices:
            treat_score = propensity_scores[treat_idx]
            treat_logit = logit_scores[treat_idx]

            # 计算与所有控制组的距离
            control_scores = propensity_scores[control_indices]
            control_logits = logit_scores[control_indices]

            # 使用logit距离
            distances = np.abs(control_logits - treat_logit)

            # 找到满足卡尺条件的最近邻
            valid_matches = np.where(distances <= caliper_value)[0]

            if len(valid_matches) > 0:
                # 在有效匹配中选择最近的
                best_match_idx = valid_matches[np.argmin(distances[valid_matches])]
                control_idx = control_indices[best_match_idx]

                match_results.append({
                    'treat_idx': treat_idx,
                    'control_idx': control_idx,
                    'distance': distances[best_match_idx],
                    'treat_score': treat_score,
                    'control_score': control_scores[best_match_idx]
                })

        self.matched_pairs = pd.DataFrame(match_results)

        n_matched = len(self.matched_pairs)
        match_rate = n_matched / n_treat * 100 if n_treat > 0 else 0

        print(f"✓ 匹配结果:")
        print(f"  - 成功匹配对数: {n_matched}/{n_treat}")
        print(f"  - 匹配成功率: {match_rate:.2f}%")
        print(f"  - 平均匹配距离: {self.matched_pairs['distance'].mean():.4f}")

        return self.matched_pairs

    def check_balance(self, treatment_var, covariates):
        """检查匹配前后的平衡性"""
        print("\n" + "="*60)
        print("4. 平衡性检验")
        print("="*60)

        df = self.df_baseline.copy()
        analysis_vars = [treatment_var] + covariates
        df_analysis = df[analysis_vars].dropna()

        results = []

        for covar in covariates:
            # 匹配前
            treat_before = df_analysis[df_analysis[treatment_var] == 1][covar]
            control_before = df_analysis[df_analysis[treatment_var] == 0][covar]

            mean_treat_before = treat_before.mean()
            mean_control_before = control_before.mean()
            bias_before = (mean_treat_before - mean_control_before) / \
                          np.sqrt((treat_before.var() + control_before.var()) / 2) * 100

            # t检验
            t_stat_before, p_val_before = stats.ttest_ind(treat_before, control_before)

            # 匹配后
            if len(self.matched_pairs) > 0:
                treat_indices = self.matched_pairs['treat_idx'].values
                control_indices = self.matched_pairs['control_idx'].values

                treat_after = df_analysis.iloc[treat_indices][covar]
                control_after = df_analysis.iloc[control_indices][covar]

                mean_treat_after = treat_after.mean()
                mean_control_after = control_after.mean()
                bias_after = (mean_treat_after - mean_control_after) / \
                            np.sqrt((treat_after.var() + control_after.var()) / 2) * 100

                t_stat_after, p_val_after = stats.ttest_ind(treat_after, control_after)

                bias_reduction = (bias_before - bias_after) / abs(bias_before) * 100
            else:
                mean_treat_after = np.nan
                mean_control_after = np.nan
                bias_after = np.nan
                t_stat_after = np.nan
                p_val_after = np.nan
                bias_reduction = np.nan

            results.append({
                '变量': covar,
                '处理组均值(匹配前)': mean_treat_before,
                '控制组均值(匹配前)': mean_control_before,
                '偏差%(匹配前)': bias_before,
                'P值(匹配前)': p_val_before,
                '处理组均值(匹配后)': mean_treat_after,
                '控制组均值(匹配后)': mean_control_after,
                '偏差%(匹配后)': bias_after,
                'P值(匹配后)': p_val_after,
                '偏差削减%': bias_reduction
            })

        balance_df = pd.DataFrame(results)

        print("\n平衡性检验结果:")
        print(balance_df.to_string(index=False))

        # 保存结果
        balance_df.to_csv('基期PSM平衡性检验结果_2009.csv', index=False, encoding='utf-8-sig')
        print(f"\n✓ 结果已保存至: 基期PSM平衡性检验结果_2009.csv")

        return balance_df

    def plot_propensity_scores(self, treatment_var):
        """绘制倾向得分分布图"""
        print("\n绘制倾向得分分布图...")

        df = self.df_baseline.copy()

        # 获取处理组和控制组的倾向得分
        treatment = df[treatment_var].values

        treat_scores = self.propensity_scores[treatment == 1]
        control_scores = self.propensity_scores[treatment == 0]

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # 1. 直方图
        ax = axes[0, 0]
        ax.hist(treat_scores, bins=30, alpha=0.6, label='处理组', color='red', density=True)
        ax.hist(control_scores, bins=30, alpha=0.6, label='控制组', color='blue', density=True)
        ax.set_xlabel('倾向得分')
        ax.set_ylabel('密度')
        ax.set_title('倾向得分分布 (匹配前)')
        ax.legend()

        # 2. 箱线图
        ax = axes[0, 1]
        data_to_plot = [treat_scores, control_scores]
        bp = ax.boxplot(data_to_plot, labels=['处理组', '控制组'],
                       patch_artist=True, showmeans=True)
        bp['boxes'][0].set_facecolor('lightcoral')
        bp['boxes'][1].set_facecolor('lightblue')
        ax.set_ylabel('倾向得分')
        ax.set_title('倾向得分箱线图')

        # 3. Q-Q图
        ax = axes[1, 0]
        from scipy import stats as scipy_stats
        scipy_stats.probplot(self.propensity_scores, dist="uniform", plot=ax)
        ax.set_title('倾向得分 Q-Q图')

        # 4. 匹配前后对比
        ax = axes[1, 1]
        if len(self.matched_pairs) > 0:
            matched_treat = self.matched_pairs['treat_score']
            matched_control = self.matched_pairs['control_score']
            ax.scatter(matched_treat, matched_control, alpha=0.5, s=20)
            ax.plot([0, 1], [0, 1], 'r--', label='完美匹配线')
            ax.set_xlabel('处理组倾向得分')
            ax.set_ylabel('控制组倾向得分')
            ax.set_title('匹配对倾向得分对比')
            ax.legend()

        plt.tight_layout()
        plt.savefig('倾向得分分布图_2009.png', dpi=300, bbox_inches='tight')
        print("✓ 图表已保存: 倾向得分分布图_2009.png")
        plt.close()

    def export_matched_data(self, treatment_var, covariates):
        """导出匹配后的数据"""
        print("\n" + "="*60)
        print("5. 导出匹配数据")
        print("="*60)

        if len(self.matched_pairs) == 0:
            print("没有匹配数据可导出")
            return

        df = self.df_baseline.copy()
        analysis_vars = [treatment_var] + covariates
        df_analysis = df[analysis_vars].dropna().reset_index(drop=True)

        # 获取匹配样本
        treat_indices = self.matched_pairs['treat_idx'].values
        control_indices = self.matched_pairs['control_idx'].values

        matched_data = []

        for i, (t_idx, c_idx) in enumerate(zip(treat_indices, control_indices)):
            # 处理组
            treat_row = df_analysis.iloc[t_idx].copy()
            treat_row['匹配ID'] = i
            treat_row['组别'] = '处理组'
            matched_data.append(treat_row)

            # 控制组
            control_row = df_analysis.iloc[c_idx].copy()
            control_row['匹配ID'] = i
            control_row['组别'] = '控制组'
            matched_data.append(control_row)

        matched_df = pd.DataFrame(matched_data)

        # 保存
        output_file = 'PSM匹配数据_2009.xlsx'
        matched_df.to_excel(output_file, index=False, engine='openpyxl')
        print(f"✓ 匹配数据已导出: {output_file}")
        print(f"  - 总样本数: {len(matched_df)}")
        print(f"  - 匹配对数: {len(matched_df) // 2}")

        return matched_df

    def generate_summary_report(self, covariates):
        """生成汇总报告"""
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
  卡尺设定: 倾向得分对数几率标准差的0.25倍
  匹配方法: 1:1 最近邻匹配

二、模型估计
{'─'*60}
  Logit模型估计结果:
"""

        if self.psm_model is not None:
            report += f"{self.psm_model.summary().tables[1]}\n"

        report += f"""
三、匹配结果
{'─'*60
}"""

        if self.matched_pairs is not None:
            report += f"""
  成功匹配对数: {len(self.matched_pairs)}
  匹配成功率: {len(self.matched_pairs) / self.matched_pairs['treat_idx'].nunique() * 100:.2f}%
  平均匹配距离: {self.matched_pairs['distance'].mean():.4f}
  最小匹配距离: {self.matched_pairs['distance'].min():.4f}
  最大匹配距离: {self.matched_pairs['distance'].max():.4f}
"""

        report += f"""
{'='*60}
分析完成！生成的文件:
  1. 基期PSM平衡性检验结果_2009.csv
  2. 倾向得分分布图_2009.png
  3. PSM匹配数据_2009.xlsx
{'='*60}
"""

        print(report)

        # 保存报告
        with open('PSM分析报告_2009.txt', 'w', encoding='utf-8') as f:
            f.write(report)
        print("✓ 报告已保存: PSM分析报告_2009.txt")


def main():
    """主函数"""
    print("\n" + "="*60)
    print("基期PSM分析 (2009年基期)")
    print("="*60)

    # 数据文件路径
    data_path = "../总数据集2007-2023_仅含emission城市_更新DID.xlsx"

    # 创建分析器
    analyzer = PSMAnalyzer(data_path)

    # 1. 加载数据
    analyzer.load_data()

    # 2. 准备基期数据
    analyzer.prepare_baseline_data()

    # 定义变量
    # 注意：这里需要根据实际数据列名调整
    # 假设处理变量名为 'treat' 或 '处理组'
    # 匹配变量: ln_real_gdp, ln_人口密度, ln_金融发展水平, 第二产业占GDP比重

    # 检测实际列名
    df = analyzer.df_baseline
    print("\n" + "="*60)
    print("检测实际变量名...")
    print("="*60)

    # 查找处理变量
    possible_treat = [col for col in df.columns
                     if 'treat' in col.lower() or '处理' in col or 'group' in col.lower()]
    if possible_treat:
        treatment_var = possible_treat[0]
        print(f"✓ 处理变量: {treatment_var}")
    else:
        print("✗ 未找到处理变量，请手动指定")
        return

    # 查找匹配变量
    covariates = []
    covariate_mapping = {
        'ln_real_gdp': ['gdp', 'GDP'],
        '人口密度': ['人口密度', 'pop_density'],
        '金融发展': ['金融', 'finance'],
        '第二产业': ['第二产业', 'secondary']
    }

    for target_name, keywords in covariate_mapping.items():
        found = False
        for col in df.columns:
            if any(kw in col for kw in keywords):
                covariates.append(col)
                print(f"✓ 找到 {target_name}: {col}")
                found = True
                break
        if not found:
            print(f"✗ 未找到 {target_name}")

    if len(covariates) < 4:
        print("\n警告: 部分匹配变量未找到，请检查数据")
        print("已找到的变量:", covariates)

    # 3. 运行PSM
    try:
        analyzer.run_psm(
            treatment_var=treatment_var,
            covariates=covariates,
            caliper=0.25
        )

        # 4. 平衡性检验
        analyzer.check_balance(treatment_var, covariates)

        # 5. 绘图
        analyzer.plot_propensity_scores(treatment_var)

        # 6. 导出匹配数据
        analyzer.export_matched_data(treatment_var, covariates)

        # 7. 生成报告
        analyzer.generate_summary_report(covariates)

        print("\n" + "="*60)
        print("✓ PSM分析完成！")
        print("="*60)

    except Exception as e:
        print(f"\n✗ PSM分析出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
