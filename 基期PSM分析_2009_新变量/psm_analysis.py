"""
基期PSM分析 (2009年)
使用变量: ln_real_gdp, ln_人口密度, ln_金融发展水平, 第二产业占GDP比重
匹配方法: 1:1 无放回匹配
卡尺: 倾向得分对数几率标准差的0.25倍
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from scipy import stats

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


class PSMAnalyzer:
    """倾向得分匹配分析器"""

    def __init__(self, data_path, baseline_year=2009, caliper_mult=0.25):
        """
        初始化PSM分析器

        参数:
        - data_path: 数据文件路径
        - baseline_year: 基期年份，默认2009
        - caliper_mult: 卡尺倍数，默认0.25（标准差的倍数）
        """
        self.baseline_year = baseline_year
        self.caliper_mult = caliper_mult
        self.df = None
        self.baseline_data = None
        self.matched_data = None
        self.propensity_scores = None
        self.caliper = None

        # 读取数据
        self._load_data(data_path)

    def _load_data(self, data_path):
        """读取Excel数据"""
        print(f"正在读取数据: {data_path}")
        self.df = pd.read_excel(data_path)
        print(f"数据形状: {self.df.shape}")
        print(f"列名: {list(self.df.columns)}")
        print()

    def prepare_baseline_data(self, treatment_col='Treat',
                             covariates=None):
        """
        准备基期数据

        参数:
        - treatment_col: 处理组标识列名
        - covariates: 协变量列表，如果为None则使用默认四个变量
        """
        if covariates is None:
            covariates = ['ln_real_gdp', 'ln_人口密度', 'ln_金融发展水平', '第二产业占GDP比重']

        # 筛选基期数据
        self.baseline_data = self.df[self.df['year'] == self.baseline_year].copy()

        # 检查必要的列是否存在
        missing_cols = []
        if treatment_col not in self.baseline_data.columns:
            missing_cols.append(treatment_col)
        for col in covariates:
            if col not in self.baseline_data.columns:
                missing_cols.append(col)

        if missing_cols:
            print(f"警告: 以下列不存在于数据中: {missing_cols}")
            print("可用的列名:", list(self.baseline_data.columns))
            return False

        # 删除有缺失值的行
        n_before = len(self.baseline_data)
        self.baseline_data = self.baseline_data[
            [treatment_col] + covariates + ['city_name']].dropna()
        n_after = len(self.baseline_data)

        print(f"基期({self.baseline_year}年)数据准备完成:")
        print(f"  - 删除缺失值前: {n_before} 个观测")
        print(f"  - 删除缺失值后: {n_after} 个观测")
        print(f"  - 处理组: {self.baseline_data[treatment_col].sum()} 个城市")
        print(f"  - 控制组: {(self.baseline_data[treatment_col]==0).sum()} 个城市")
        print(f"  - 协变量: {covariates}")
        print()

        self.treatment_col = treatment_col
        self.covariates = covariates
        return True

    def calculate_propensity_scores(self):
        """计算倾向得分"""
        print("计算倾向得分...")

        X = self.baseline_data[self.covariates].values
        y = self.baseline_data[self.treatment_col].values

        # 标准化协变量
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # 使用Logistic回归计算倾向得分
        logit = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
        logit.fit(X_scaled, y)

        # 获取预测概率，并限制在合理范围内避免0和1
        raw_scores = logit.predict_proba(X_scaled)[:, 1]
        # 将概率限制在[0.01, 0.99]范围内以避免logit计算问题
        self.propensity_scores = np.clip(raw_scores, 0.01, 0.99)

        # 计算卡尺: 0.25 * logit(PS)的标准差
        logit_ps = np.log(self.propensity_scores / (1 - self.propensity_scores))
        logit_ps_std = np.std(logit_ps)
        self.caliper = self.caliper_mult * logit_ps_std

        print(f"倾向得分统计:")
        print(f"  - 均值: {self.propensity_scores.mean():.4f}")
        print(f"  - 标准差: {self.propensity_scores.std():.4f}")
        print(f"  - 最小值: {self.propensity_scores.min():.4f}")
        print(f"  - 最大值: {self.propensity_scores.max():.4f}")
        print(f"  - Logit(PS)标准差: {logit_ps_std:.4f}")
        print(f"  - 卡尺 (0.25*SD): {self.caliper:.4f}")
        print()

        # 保存倾向得分到数据中
        self.baseline_data = self.baseline_data.reset_index(drop=True)
        self.baseline_data['propensity_score'] = self.propensity_scores

        return self.propensity_scores

    def perform_matching(self, ratio=1, replacement=False):
        """
        执行倾向得分匹配

        参数:
        - ratio: 匹配比例，默认1:1
        - replacement: 是否有放回，默认False
        """
        print(f"执行匹配: {ratio}:1, {'有' if replacement else '无'}放回")
        print(f"卡尺: {self.caliper:.4f}")

        treated = self.baseline_data[self.baseline_data[self.treatment_col] == 1].copy()
        control = self.baseline_data[self.baseline_data[self.treatment_col] == 0].copy()

        # 重置索引
        treated = treated.reset_index(drop=True)
        control = control.reset_index(drop=True)

        matched_treated = []
        matched_control = []
        used_control_indices = []

        # 对每个处理组观测寻找匹配
        for i, treated_row in treated.iterrows():
            treated_ps = treated_row['propensity_score']

            # 计算与所有控制组的距离
            distances = np.abs(control['propensity_score'].values - treated_ps)

            # 如果是无放回，排除已使用的控制组
            if not replacement and used_control_indices:
                distances[used_control_indices] = np.inf

            # 找到最近的匹配
            best_idx = np.argmin(distances)
            min_distance = distances[best_idx]

            # 检查是否在卡尺范围内
            if min_distance <= self.caliper:
                matched_control.append(control.iloc[best_idx].copy())
                matched_treated.append(treated_row.copy())

                if not replacement:
                    used_control_indices.append(best_idx)
            else:
                # 没有找到符合条件的匹配，仍然记录但不标记匹配成功
                matched_treated.append(treated_row.copy())
                # 创建一个空的控制组记录
                empty_row = pd.Series({'propensity_score': np.nan})
                for col in self.covariates:
                    empty_row[col] = np.nan
                empty_row['city_name'] = '未匹配'
                matched_control.append(empty_row)

        matched_treated_df = pd.DataFrame(matched_treated)
        matched_control_df = pd.DataFrame(matched_control)

        # 统计匹配结果
        n_matched = (matched_control_df['city_name'] != '未匹配').sum()
        match_rate = n_matched / len(treated) * 100

        print(f"\n匹配结果:")
        print(f"  - 处理组总数: {len(treated)}")
        print(f"  - 成功匹配: {n_matched}")
        print(f"  - 匹配成功率: {match_rate:.2f}%")
        print()

        self.matched_data = {
            'treated': matched_treated_df,
            'control': matched_control_df,
            'match_rate': match_rate,
            'n_matched': n_matched
        }

        return self.matched_data

    def check_balance(self, save_path='平衡性检验结果.xlsx'):
        """
        检查匹配后的平衡性

        参数:
        - save_path: 结果保存路径
        """
        print("执行平衡性检验...")

        # 获取成功匹配的样本
        matched_mask = self.matched_data['control']['city_name'].values != '未匹配'
        treated_matched = self.matched_data['treated'][matched_mask].copy()
        control_matched = self.matched_data['control'][matched_mask].copy()

        # 计算匹配前的差异
        treated_all = self.baseline_data[self.baseline_data[self.treatment_col] == 1]
        control_all = self.baseline_data[self.baseline_data[self.treatment_col] == 0]

        results = []

        for covariate in self.covariates:
            # 匹配前
            mean_t_before = treated_all[covariate].mean()
            mean_c_before = control_all[covariate].mean()
            std_t_before = treated_all[covariate].std()
            std_c_before = control_all[covariate].std()

            # 标准化偏差匹配前
            pooled_std_before = np.sqrt((std_t_before**2 + std_c_before**2) / 2)
            if pooled_std_before > 0:
                bias_before = 100 * (mean_t_before - mean_c_before) / pooled_std_before
            else:
                bias_before = 0

            # t检验匹配前
            t_stat_before, p_value_before = stats.ttest_ind(
                treated_all[covariate].dropna(),
                control_all[covariate].dropna(),
                equal_var=False
            )

            # 匹配后
            mean_t_after = treated_matched[covariate].mean()
            mean_c_after = control_matched[covariate].mean()
            std_t_after = treated_matched[covariate].std()
            std_c_after = control_matched[covariate].std()

            # 标准化偏差匹配后
            pooled_std_after = np.sqrt((std_t_after**2 + std_c_after**2) / 2)
            if pooled_std_after > 0:
                bias_after = 100 * (mean_t_after - mean_c_after) / pooled_std_after
            else:
                bias_after = 0

            # t检验匹配后
            t_stat_after, p_value_after = stats.ttest_ind(
                treated_matched[covariate].dropna(),
                control_matched[covariate].dropna(),
                equal_var=False
            )

            # 偏差减少比例
            if abs(bias_before) > 0:
                bias_reduction = (abs(bias_before) - abs(bias_after)) / abs(bias_before) * 100
            else:
                bias_reduction = 0

            results.append({
                '变量': covariate,
                '匹配前_处理组均值': mean_t_before,
                '匹配前_控制组均值': mean_c_before,
                '匹配前_偏差(%)': bias_before,
                '匹配前_t值': t_stat_before,
                '匹配前_p值': p_value_before,
                '匹配后_处理组均值': mean_t_after,
                '匹配后_控制组均值': mean_c_after,
                '匹配后_偏差(%)': bias_after,
                '匹配后_t值': t_stat_after,
                '匹配后_p值': p_value_after,
                '偏差减少比例(%)': bias_reduction
            })

        balance_df = pd.DataFrame(results)

        # 打印结果
        print("\n平衡性检验结果:")
        print(balance_df.to_string(index=False))
        print()

        # 保存结果
        balance_df.to_excel(save_path, index=False, engine='openpyxl')
        print(f"平衡性检验结果已保存至: {save_path}")

        # 生成可视化
        self._plot_balance(balance_df)

        return balance_df

    def _plot_balance(self, balance_df):
        """绘制平衡性检验图"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # 1. 标准化偏差对比
        ax1 = axes[0, 0]
        x_pos = np.arange(len(self.covariates))
        width = 0.35

        ax1.bar(x_pos - width/2, balance_df['匹配前_偏差(%)'], width,
                label='匹配前', color='coral', alpha=0.7)
        ax1.bar(x_pos + width/2, balance_df['匹配后_偏差(%)'], width,
                label='匹配后', color='steelblue', alpha=0.7)
        ax1.axhline(y=10, color='red', linestyle='--', linewidth=1, label='10%阈值')
        ax1.axhline(y=-10, color='red', linestyle='--', linewidth=1)
        ax1.set_xlabel('变量')
        ax1.set_ylabel('标准化偏差 (%)')
        ax1.set_title('匹配前后标准化偏差对比')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(self.covariates, rotation=15, ha='right')
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)

        # 2. 偏差减少比例
        ax2 = axes[0, 1]
        colors = ['green' if x > 50 else 'orange' if x > 0 else 'red'
                  for x in balance_df['偏差减少比例(%)']]
        ax2.barh(self.covariates, balance_df['偏差减少比例(%)'], color=colors, alpha=0.7)
        ax2.set_xlabel('偏差减少比例 (%)')
        ax2.set_title('偏差减少比例')
        ax2.axvline(x=50, color='red', linestyle='--', linewidth=1, label='50%阈值')
        ax2.legend()
        ax2.grid(axis='x', alpha=0.3)

        # 3. p值对比
        ax3 = axes[1, 0]
        ax3.bar(x_pos - width/2, balance_df['匹配前_p值'], width,
                label='匹配前', color='coral', alpha=0.7)
        ax3.bar(x_pos + width/2, balance_df['匹配后_p值'], width,
                label='匹配后', color='steelblue', alpha=0.7)
        ax3.axhline(y=0.05, color='red', linestyle='--', linewidth=1, label='0.05显著性水平')
        ax3.set_xlabel('变量')
        ax3.set_ylabel('p值')
        ax3.set_title('匹配前后t检验p值对比')
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels(self.covariates, rotation=15, ha='right')
        ax3.legend()
        ax3.grid(axis='y', alpha=0.3)

        # 4. 倾向得分分布
        ax4 = axes[1, 1]
        treated_ps = self.baseline_data[
            self.baseline_data[self.treatment_col] == 1]['propensity_score']
        control_ps = self.baseline_data[
            self.baseline_data[self.treatment_col] == 0]['propensity_score']

        ax4.hist(treated_ps, bins=20, alpha=0.5, label='处理组', color='coral', density=True)
        ax4.hist(control_ps, bins=20, alpha=0.5, label='控制组', color='steelblue', density=True)
        ax4.set_xlabel('倾向得分')
        ax4.set_ylabel('密度')
        ax4.set_title('倾向得分分布')
        ax4.legend()
        ax4.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        plt.savefig('平衡性检验图.png', dpi=300, bbox_inches='tight')
        print("平衡性检验图已保存至: 平衡性检验图.png")
        plt.show()

    def export_matched_cities(self, save_path='匹配后城市列表.xlsx'):
        """导出匹配后的城市列表"""
        matched_mask = self.matched_data['control']['city_name'].values != '未匹配'
        treated_matched = self.matched_data['treated'][matched_mask][['city_name', 'propensity_score'] + self.covariates].copy()
        control_matched = self.matched_data['control'][matched_mask][['city_name', 'propensity_score'] + self.covariates].copy()

        treated_matched['组别'] = '处理组'
        control_matched['组别'] = '控制组'

        matched_cities = pd.concat([treated_matched, control_matched], ignore_index=True)
        matched_cities = matched_cities[['city_name', '组别', 'propensity_score'] + self.covariates]

        matched_cities.to_excel(save_path, index=False, engine='openpyxl')
        print(f"匹配后城市列表已保存至: {save_path}")

        return matched_cities

    def generate_report(self, output_path='PSM分析报告_2009.txt'):
        """生成PSM分析报告"""
        report = f"""
{'='*80}
基期倾向得分匹配(PSM)分析报告
{'='*80}

一、分析设置
{'-'*80}
基期年份: {self.baseline_year}
协变量: {', '.join(self.covariates)}
匹配比例: 1:1
是否放回: 无放回
卡尺设置: 倾向得分对数几率标准差的{self.caliper_mult}倍
实际卡尺值: {self.caliper:.4f}

二、匹配结果
{'-'*80}
处理组城市数量: {self.baseline_data[self.treatment_col].sum()}
控制组城市数量: {(self.baseline_data[self.treatment_col]==0).sum()}
成功匹配数量: {self.matched_data['n_matched']}
匹配成功率: {self.matched_data['match_rate']:.2f}%

三、倾向得分统计
{'-'*80}
倾向得分均值: {self.propensity_scores.mean():.4f}
倾向得分标准差: {self.propensity_scores.std():.4f}
倾向得分最小值: {self.propensity_scores.min():.4f}
倾向得分最大值: {self.propensity_scores.max():.4f}

四、匹配建议
{'-'*80}
"""
        # 读取平衡性检验结果
        try:
            balance_df = pd.read_excel('平衡性检验结果.xlsx')
            n_pass = (balance_df['匹配后_p值'] > 0.05).sum()
            n_total = len(balance_df)

            report += f"""
平衡性检验结果:
- 通过检验(p>0.05)的变量数: {n_pass}/{n_total}
- 平均偏差减少比例: {balance_df['偏差减少比例(%)'].mean():.2f}%

匹配质量评估: {'优良' if n_pass >= n_total*0.75 else '一般' if n_pass >= n_total*0.5 else '需改进'}
"""

            if n_pass < n_total * 0.75:
                report += """

改进建议:
1. 考虑增加协变量以提高匹配质量
2. 调整卡尺大小(当前为0.25倍标准差)
3. 尝试有放回匹配或不同的匹配比例
4. 检查是否有足够的共同支撑域
"""

        except:
            report += "无法生成平衡性评估(请先运行平衡性检验)\n"

        report += f"""
{'='*80}
报告生成时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
{'='*80}
"""

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report)

        print(f"\n分析报告已保存至: {output_path}\n")
        print(report)

        return report


def main():
    """主函数"""
    # 设置参数
    DATA_PATH = '../总数据集2007-2023_仅含emission城市_更新DID.xlsx'
    BASELINE_YEAR = 2009
    CALIPER_MULT = 0.25

    # 创建分析器
    print("="*80)
    print("基期PSM分析 (2009年)")
    print("="*80)
    print()

    analyzer = PSMAnalyzer(DATA_PATH, baseline_year=BASELINE_YEAR, caliper_mult=CALIPER_MULT)

    # 准备基期数据
    if not analyzer.prepare_baseline_data():
        print("数据准备失败，请检查列名")
        return

    # 计算倾向得分
    analyzer.calculate_propensity_scores()

    # 执行匹配
    analyzer.perform_matching(ratio=1, replacement=False)

    # 平衡性检验
    if analyzer.matched_data['n_matched'] > 0:
        analyzer.check_balance()

        # 导出匹配后的城市列表
        analyzer.export_matched_cities()
    else:
        print("警告: 没有成功匹配的样本，无法进行平衡性检验")

    # 生成分析报告
    analyzer.generate_report()

    print("="*80)
    print("PSM分析完成!")
    print("="*80)


if __name__ == '__main__':
    main()
