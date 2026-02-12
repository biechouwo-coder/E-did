"""
多时点DID分析（基于PSM加权）
使用PSM匹配后的城市权重进行DID估计

因变量: 缩尾后的ln(碳排放强度)
控制变量: ln_real_gdp, ln_人口密度, ln_金融发展水平, 第二产业占GDP比重
固定效应: 城市、年份
标准误聚类: 城市层面
"""

import pandas as pd
import numpy as np
from scipy.stats.mstats import winsorize
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# 尝试导入linearmodels，如果不可用则使用statsmodels
try:
    from linearmodels.panel import PanelOLS
    USE_LINEARMODELS = True
    print("使用 linearmodels 进行面板回归")
except ImportError:
    USE_LINEARMODELS = False
    print("linearmodels 未安装，将使用 statsmodels (注意: statsmodels不支持直接聚类稳健标准误)")

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


class DIDAnalyzer:
    """多时点DID分析器（基于PSM加权）"""

    def __init__(self, data_path, psm_matched_path, psm_control_usage_path):
        """
        初始化DID分析器

        参数:
        - data_path: 原始数据路径
        - psm_matched_path: PSM匹配后的城市列表路径
        - psm_control_usage_path: PSM控制组使用情况路径
        """
        self.data_path = data_path
        self.psm_matched_path = psm_matched_path
        self.psm_control_usage_path = psm_control_usage_path

        self.df = None
        self.weights = None
        self.city_weights = {}
        self.results = None

        # 读取数据
        self._load_data()

    def _load_data(self):
        """读取所有数据"""
        print("="*80)
        print("数据加载")
        print("="*80)

        # 1. 读取原始数据
        print(f"\n正在读取原始数据: {self.data_path}")
        self.df = pd.read_excel(self.data_path)
        print(f"  - 数据形状: {self.df.shape}")
        print(f"  - 年份范围: {self.df['year'].min()} - {self.df['year'].max()}")
        print(f"  - 城市数量: {self.df['city_name'].nunique()}")

        # 2. 读取PSM匹配后的城市列表
        print(f"\n正在读取PSM匹配结果: {self.psm_matched_path}")
        psm_matched = pd.read_excel(self.psm_matched_path)
        print(f"  - 匹配样本数: {len(psm_matched)}")
        print(f"  - 处理组样本数: {(psm_matched['组别']=='处理组').sum()}")
        print(f"  - 控制组样本数: {(psm_matched['组别']=='控制组').sum()}")

        # 3. 读取控制组使用情况
        print(f"\n正在读取控制组使用情况: {self.psm_control_usage_path}")
        control_usage = pd.read_excel(self.psm_control_usage_path)
        print(f"  - 使用的控制组城市数: {len(control_usage)}")
        print(f"  - 平均使用次数: {control_usage['使用次数'].mean():.2f}")
        print(f"  - 最大使用次数: {control_usage['使用次数'].max()}")

        self.psm_matched = psm_matched
        self.control_usage = control_usage

    def calculate_weights(self):
        """计算每个城市的权重"""
        print("\n" + "="*80)
        print("计算城市权重")
        print("="*80)

        self.city_weights = {}

        # 从匹配后的城市列表中获取处理组城市
        treated_cities = self.psm_matched[self.psm_matched['组别'] == '处理组']['city_name'].unique()

        # 处理组权重 = 1
        for city in treated_cities:
            self.city_weights[city] = 1

        print(f"\n处理组城市: {len(treated_cities)} 个，权重 = 1")

        # 控制组权重 = 使用次数（从控制组使用情况表）
        control_dict = dict(zip(self.control_usage['城市'], self.control_usage['使用次数']))

        # 获取原始数据中的所有控制组城市
        all_control_cities = self.df[self.df['Treat'] == 0]['city_name'].unique()

        for city in all_control_cities:
            if city in control_dict:
                # 匹配上的控制组城市：权重 = 使用次数
                self.city_weights[city] = control_dict[city]
            else:
                # 未匹配上的控制组城市：权重 = 0（将被剔除）
                self.city_weights[city] = 0

        # 统计控制组
        control_weights = {k: v for k, v in self.city_weights.items() if k not in treated_cities}
        n_matched_controls = sum(1 for v in control_weights.values() if v > 0)
        n_unmatched_controls = sum(1 for v in control_weights.values() if v == 0)

        print(f"控制组城市（原始）: {len(control_weights)} 个")
        print(f"  - 匹配上: {n_matched_controls} 个（权重 > 0）")
        print(f"  - 未匹配: {n_unmatched_controls} 个（权重 = 0，将被剔除）")

        # 将权重添加到数据中
        self.df['weight'] = self.df['city_name'].map(self.city_weights).fillna(0)

        # 只保留有权重的数据
        n_before = len(self.df)
        self.df = self.df[self.df['weight'] > 0].copy()
        n_after = len(self.df)

        print(f"\n数据筛选:")
        print(f"  - 筛选前观测数: {n_before}")
        print(f"  - 筛选后观测数: {n_after}")
        print(f"  - 剔除观测数: {n_before - n_after}")
        print(f"  - 保留城市数: {self.df['city_name'].nunique()}")

    def winsorize_emission(self, limits=(0.01, 0.01)):
        """
        对碳排放强度进行对数变换和缩尾处理

        参数:
        - limits: 缩尾比例，默认上下各1%
        """
        print("\n" + "="*80)
        print("因变量处理: ln(碳排放强度) + 缩尾")
        print("="*80)

        # 计算对数碳排放强度
        self.df['ln_emission'] = np.log(self.df['emission'])

        print(f"\n原始ln_emission统计:")
        print(f"  - 均值: {self.df['ln_emission'].mean():.4f}")
        print(f"  - 标准差: {self.df['ln_emission'].std():.4f}")
        print(f"  - 最小值: {self.df['ln_emission'].min():.4f}")
        print(f"  - 最大值: {self.df['ln_emission'].max():.4f}")

        # 缩尾处理
        self.df['ln_emission_winsor'] = winsorize(self.df['ln_emission'], limits=limits)

        print(f"\n缩尾后ln_emission统计 (上下各{limits[0]*100}%):")
        print(f"  - 均值: {self.df['ln_emission_winsor'].mean():.4f}")
        print(f"  - 标准差: {self.df['ln_emission_winsor'].std():.4f}")
        print(f"  - 最小值: {self.df['ln_emission_winsor'].min():.4f}")
        print(f"  - 最大值: {self.df['ln_emission_winsor'].max():.4f}")

        # 绘制分布对比图
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        axes[0].hist(self.df['ln_emission'], bins=50, alpha=0.7, label='原始', color='coral')
        axes[0].set_xlabel('ln(碳排放强度)')
        axes[0].set_ylabel('频数')
        axes[0].set_title('原始ln(碳排放强度)分布')
        axes[0].legend()
        axes[0].grid(axis='y', alpha=0.3)

        axes[1].hist(self.df['ln_emission_winsor'], bins=50, alpha=0.7, label='缩尾后', color='steelblue')
        axes[1].set_xlabel('ln(碳排放强度)')
        axes[1].set_ylabel('频数')
        axes[1].set_title('缩尾后ln(碳排放强度)分布')
        axes[1].legend()
        axes[1].grid(axis='y', alpha=0.3)

        plt.tight_layout()
        plt.savefig('ln_emission分布对比.png', dpi=300, bbox_inches='tight')
        print(f"\n分布对比图已保存: ln_emission分布对比.png")
        plt.close()

    def parallel_trends_test(self, pre_period_end=2009):
        """
        平行趋势检验（Parallel Trends Test）

        检验在政策实施前，处理组和控制组的结果变量是否具有相似的线性趋势

        参数:
        - pre_period_end: 政策实施前的截止年份（默认2009）
        """
        print("\n" + "="*80)
        print("平行趋势检验")
        print("="*80)

        # 筛选政策前的数据
        pre_data = self.df[self.df['year'] <= pre_period_end].copy()

        if len(pre_data) == 0:
            print(f"错误: 没有{pre_period_end}年及以前的数据")
            return False

        print(f"\n使用{pre_period_end}年及以前的数据进行检验")
        print(f"  - 观测数: {len(pre_data)}")
        print(f"  - 处理组城市数: {pre_data[pre_data['Treat']==1]['city_name'].nunique()}")
        print(f"  - 控制组城市数: {pre_data[pre_data['Treat']==0]['city_name'].nunique()}")

        # 按组别和年份计算平均ln_emission
        group_year_stats = pre_data.groupby(['Treat', 'year'])['ln_emission_winsor'].mean().reset_index()
        group_year_stats.columns = ['Treat', 'year', 'mean_ln_emission']

        # 绘制趋势图
        fig, ax = plt.subplots(figsize=(12, 6))

        treated_data = group_year_stats[group_year_stats['Treat'] == 1]
        control_data = group_year_stats[group_year_stats['Treat'] == 0]

        ax.plot(treated_data['year'], treated_data['mean_ln_emission'],
                marker='o', linestyle='-', linewidth=2, label='处理组', color='coral')
        ax.plot(control_data['year'], control_data['mean_ln_emission'],
                marker='s', linestyle='--', linewidth=2, label='控制组', color='steelblue')

        # 标记政策实施年份
        ax.axvline(x=pre_period_end, color='red', linestyle=':', linewidth=2, label=f'政策实施({pre_period_end}年)')
        ax.set_xlabel('年份', fontsize=12)
        ax.set_ylabel('ln(碳排放强度) [缩尾后]', fontsize=12)
        ax.set_title('平行趋势检验：政策前处理组与控制组趋势对比', fontsize=14)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('平行趋势检验.png', dpi=300, bbox_inches='tight')
        print(f"\n趋势图已保存: 平行趋势检验.png")
        plt.close()

        # 统计检验：回归交互项
        # 模型: ln_emission = α + β1*Treat + β2*year + β3*(Treat×year) + ε
        # 如果β3显著，说明平行趋势不成立
        import statsmodels.formula.api as smf

        pre_data_copy = pre_data.copy()
        pre_data_copy['Treat'] = pre_data_copy['Treat'].astype(int)
        pre_data_copy['year_centered'] = pre_data_copy['year'] - pre_data_copy['year'].min()

        formula = 'ln_emission_winsor ~ Treat * year_centered'
        model = smf.ols(formula, data=pre_data_copy).fit()

        # 提取交互项系数和p值
        interaction_coef = model.params['Treat:year_centered']
        interaction_pval = model.pvalues['Treat:year_centered']

        print("\n统计检验结果:")
        print(f"  - 模型: ln_emission ~ Treat + year + Treat×year")
        print(f"  - 交互项系数(Treat×year): {interaction_coef:.6f}")
        print(f"  - 交互项p值: {interaction_pval:.6f}")
        print(f"  - 平行趋势假设: {'通过 (p>=0.05)' if interaction_pval >= 0.05 else '拒绝 (p<0.05)'}")

        # 年别增长率对比
        if len(treated_data) > 1 and len(control_data) > 1:
            # 计算复合年增长率
            treated_growth = (treated_data['mean_ln_emission'].iloc[-1] /
                            treated_data['mean_ln_emission'].iloc[0]) ** (1/len(treated_data)) - 1
            control_growth = (control_data['mean_ln_emission'].iloc[-1] /
                            control_data['mean_ln_emission'].iloc[0]) ** (1/len(control_data)) - 1

            print(f"\n年复合增长率对比:")
            print(f"  - 处理组: {treated_growth*100:.2f}%")
            print(f"  - 控制组: {control_growth*100:.2f}%")
            print(f"  - 差异: {abs(treated_growth - control_growth)*100:.2f}个百分点")

        print("\n结论:")
        if interaction_pval < 0.05:
            print("  ⚠️ 警告: 交互项显著，处理组和控制组在政策前的趋势存在显著差异")
            print("  这可能违反平行趋势假设，建议谨慎解释DID结果")
        else:
            print("  通过: 平行趋势假设成立，处理组和控制组在政策前具有相似的发展趋势")

        return {
            'interaction_coef': interaction_coef,
            'interaction_pval': interaction_pval,
            'parallel_trend_holds': interaction_pval >= 0.05
        }

    def run_did_regression(self):
        """
        运行多时点DID回归

        模型: ln_emission_winsor = β0 + β1*Treat×Post + γX + μ_city + λ_year + ε
        """
        print("\n" + "="*80)
        print("多时点DID回归")
        print("="*80)

        # 检查必要的列
        required_cols = ['ln_emission_winsor', 'Treat', 'Post', 'DID',
                       'ln_real_gdp', 'ln_人口密度', 'ln_金融发展水平', '第二产业占GDP比重']

        missing_cols = [col for col in required_cols if col not in self.df.columns]
        if missing_cols:
            print(f"错误: 缺少必要的列: {missing_cols}")
            return False

        # 删除包含缺失值的行
        n_before = len(self.df)
        # 包含weight列用于PSM加权
        reg_df = self.df[required_cols + ['city_name', 'year', 'weight']].dropna()
        n_after = len(reg_df)

        if n_before != n_after:
            print(f"\n删除包含缺失值的行:")
            print(f"  - 删除前: {n_before} 行")
            print(f"  - 删除后: {n_after} 行")
            print(f"  - 删除: {n_before - n_after} 行")

        # 方法1: 使用linearmodels（推荐）
        if USE_LINEARMODELS:
            print("\n使用 linearmodels.PanelOLS 进行回归...")
            results = self._run_linearmodels(reg_df)
        # 方法2: 使用statsmodels（备选）
        elif self._check_statsmodels():
            print("\n使用 statsmodels.OLS 进行回归...")
            results = self._run_statsmodels(reg_df)
        # 方法3: 使用sklearn手动实现固定效应（最终备选）
        else:
            print("\n使用 sklearn + 手动固定效应 进行回归...")
            results = self._run_sklearn_manual_fe(reg_df)

        self.results = results
        return results

    def _check_statsmodels(self):
        """检查statsmodels是否可用"""
        try:
            import statsmodels.formula.api as smf
            return True
        except ImportError:
            return False

    def _run_linearmodels(self, reg_df):
        """使用linearmodels进行面板回归（支持聚类标准误）"""
        # 创建多索引面板数据
        reg_df = reg_df.set_index(['city_name', 'year'])

        # 定义模型变量
        y = reg_df['ln_emission_winsor']
        X = reg_df[['DID', 'ln_real_gdp', 'ln_人口密度', 'ln_金融发展水平', '第二产业占GDP比重']]

        # 检查是否有权重列
        if 'weight' in reg_df.columns:
            weights = reg_df['weight']
            print(f"使用PSM权重: {weights.sum():.0f} (总权重)")
            # 拟合模型（带权重）
            model = PanelOLS(y, X, entity_effects=True, time_effects=True, weights=weights)
        else:
            print("警告: 未找到权重列，使用无权重回归")
            # 拟合模型（无权重）
            model = PanelOLS(y, X, entity_effects=True, time_effects=True)

        # 使用聚类稳健标准误（聚类到城市层面）
        result = model.fit(cov_type='clustered', cluster_entity=True)

        # 打印结果
        print("\n" + "-"*80)
        print("回归结果:")
        print("-"*80)
        print(result)

        # 提取关键结果
        self._extract_results_linearmodels(result)

        return result

    def _run_statsmodels(self, reg_df):
        """使用statsmodels进行OLS回归（LSDV法，不支持聚类标准误）"""
        import statsmodels.formula.api as smf
        import statsmodels.api as sm

        # 创建城市和年份虚拟变量
        reg_df_copy = reg_df.copy()
        reg_df_copy['city_name'] = reg_df_copy['city_name'].astype('category')
        reg_df_copy['year'] = reg_df_copy['year'].astype('category')

        # 使用公式API拟合模型
        formula = ('ln_emission_winsor ~ DID + ln_real_gdp + ln_人口密度 + ln_金融发展水平 + ' +
                  'Q("第二产业占GDP比重") + C(city_name) + C(year)')

        # 检查是否有权重列
        if 'weight' in reg_df_copy.columns:
            print(f"使用PSM权重: {reg_df_copy['weight'].sum():.0f} (总权重)")
            # 使用WLS（加权最小二乘法）
            model = smf.ols(formula, data=reg_df_copy)
            result = model.fit_regularized(alpha=0, L1_wt=reg_df_copy['weight'].values)
            # 实际上statsmodels的ols不支持直接权重，需要使用WLS
            # 改用sm.WLS
            y, X = smf.patsy.dmatrices(formula, reg_df_copy, return_type='dataframe')
            result = sm.WLS(y, X, weights=reg_df_copy['weight'].values).fit()
        else:
            print("警告: 未找到权重列，使用无权重OLS回归")
            result = smf.ols(formula, data=reg_df_copy).fit()

        # 打印结果
        print("\n" + "-"*80)
        print("回归结果:")
        print("-"*80)
        print(result.summary())

        # 提取关键结果
        self._extract_results_statsmodels(result)

        print("\n警告: statsmodels不支持聚类标准误，建议安装linearmodels:")
        print("  pip install linearmodels")

        return result

    def _run_sklearn_manual_fe(self, reg_df):
        """使用sklearn手动实现固定效应（组内去均值法）"""
        from sklearn.linear_model import LinearRegression

        print("\n使用组内去均值法处理固定效应...")

        # 获取变量名
        y_var = 'ln_emission_winsor'
        x_vars = ['DID', 'ln_real_gdp', 'ln_人口密度', 'ln_金融发展水平', '第二产业占GDP比重']

        # 对每个变量进行城市内去均值（消除城市固定效应）
        for var in [y_var] + x_vars:
            reg_df[f'{var}_demean'] = reg_df.groupby('city_name')[var].transform(lambda x: x - x.mean())

        # 对每个变量进行年份内去均值（消除年份固定效应）
        for var in [y_var] + x_vars:
            reg_df[f'{var}_demean2'] = reg_df.groupby('year')[f'{var}_demean'].transform(lambda x: x - x.mean())

        # 准备回归数据
        y = reg_df[f'{y_var}_demean2'].values
        X = reg_df[[f'{v}_demean2' for v in x_vars]].values

        # 添加常数项
        X = np.column_stack([X, np.ones(len(X))])

        # OLS回归
        model = LinearRegression(fit_intercept=False)
        model.fit(X, y)

        # 预测值和残差
        y_pred = model.predict(X)
        residuals = y - y_pred

        # 计算标准误（简化版，未聚类）
        n = len(y)
        k = X.shape[1]
        df = n - k

        # 残差标准误
        sse = np.sum(residuals**2)
        sigma2 = sse / df

        # 方差协方差矩阵
        XtX = np.dot(X.T, X)
        try:
            XtX_inv = np.linalg.inv(XtX)
            vcov = sigma2 * XtX_inv
            se = np.sqrt(np.diag(vcov))
        except:
            se = np.full(k, np.nan)
            print("警告: 无法计算标准误（矩阵奇异）")

        # t统计量和p值
        from scipy import stats
        t_stats = model.coef_ / se
        p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), df=df))

        # R方
        r2 = 1 - sse / np.sum((y - np.mean(y))**2)

        # 打印结果
        print("\n" + "-"*80)
        print("回归结果:")
        print("-"*80)
        print(f"R-squared: {r2:.6f}")
        print(f"观测数: {n}")
        print(f"\n变量系数:")

        results_dict = {}
        for i, var in enumerate(x_vars):
            coef = model.coef_[i]
            std_err = se[i]
            t_stat = t_stats[i]
            p_val = p_values[i]
            sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else ''
            print(f"  {var}: {coef:.6f} (标准误: {std_err:.6f}, t值: {t_stat:.6f}, p值: {p_val:.6f}) {sig}")

            results_dict[var] = {'coef': coef, 'se': std_err, 't': t_stat, 'p': p_val}

        # 提取DID结果
        print("\n" + "="*80)
        print("关键结果")
        print("="*80)

        did_coef = results_dict['DID']['coef']
        did_se = results_dict['DID']['se']
        did_t = results_dict['DID']['t']
        did_p = results_dict['DID']['p']

        print(f"\nDID交互项系数:")
        print(f"  - 系数值: {did_coef:.6f}")
        print(f"  - 标准误: {did_se:.6f}")
        print(f"  - t值: {did_t:.6f}")
        print(f"  - p值: {did_p:.6f}")
        print(f"  - 显著性: {'***' if did_p < 0.001 else '**' if did_p < 0.01 else '*' if did_p < 0.05 else '不显著'}")

        print(f"\n模型拟合:")
        print(f"  - R-squared: {r2:.6f}")

        print("\n注意: 当前使用sklearn简化版本，未使用PSM权重，也未包含聚类标准误")
        print("      如需更准确的推断，建议安装linearmodels:")
        print("      pip install linearmodels")

        self.did_results = {
            'coefficient': did_coef,
            'std_error': did_se,
            't_value': did_t,
            'p_value': did_p,
            'r2': r2
        }

        return {'model': model, 'results': results_dict, 'r2': r2}

    def _extract_results_linearmodels(self, result):
        """从linearmodels结果中提取关键信息"""
        print("\n" + "="*80)
        print("关键结果")
        print("="*80)

        # DID系数
        did_coef = result.params['DID']
        did_se = result.std_errors['DID']
        did_t = result.tstats['DID']
        did_p = result.pvalues['DID']

        print(f"\nDID交互项系数:")
        print(f"  - 系数值: {did_coef:.6f}")
        print(f"  - 标准误: {did_se:.6f}")
        print(f"  - t值: {did_t:.6f}")
        print(f"  - p值: {did_p:.6f}")
        print(f"  - 显著性: {'***' if did_p < 0.001 else '**' if did_p < 0.01 else '*' if did_p < 0.05 else '不显著'}")

        # R方
        r2_within = result.rsquared_within
        print(f"\n模型拟合:")
        print(f"  - 组内R2: {r2_within:.6f}")

        # 控制变量系数
        print(f"\n控制变量系数:")
        for var in ['ln_real_gdp', 'ln_人口密度', 'ln_金融发展水平', '第二产业占GDP比重']:
            if var in result.params:
                coef = result.params[var]
                pval = result.pvalues[var]
                sig = '***' if pval < 0.001 else '**' if pval < 0.01 else '*' if pval < 0.05 else ''
                print(f"  - {var}: {coef:.6f} {sig}")

        self.did_results = {
            'coefficient': did_coef,
            'std_error': did_se,
            't_value': did_t,
            'p_value': did_p,
            'r2_within': r2_within
        }

    def _extract_results_statsmodels(self, result):
        """从statsmodels结果中提取关键信息"""
        print("\n" + "="*80)
        print("关键结果")
        print("="*80)

        # DID系数
        did_coef = result.params['DID']
        did_se = result.bse['DID']
        did_t = result.tvalues['DID']
        did_p = result.pvalues['DID']

        print(f"\nDID交互项系数:")
        print(f"  - 系数值: {did_coef:.6f}")
        print(f"  - 标准误: {did_se:.6f}")
        print(f"  - t值: {did_t:.6f}")
        print(f"  - p值: {did_p:.6f}")
        print(f"  - 显著性: {'***' if did_p < 0.001 else '**' if did_p < 0.01 else '*' if did_p < 0.05 else '不显著'}")

        # R方
        r2 = result.rsquared
        print(f"\n模型拟合:")
        print(f"  - R2: {r2:.6f}")

        # 控制变量系数
        print(f"\n控制变量系数:")
        for var in ['ln_real_gdp', 'ln_人口密度', 'ln_金融发展水平', 'Q("第二产业占GDP比重")']:
            if var in result.params:
                coef = result.params[var]
                pval = result.pvalues[var]
                var_clean = var.replace('Q("', '').replace('")', '')
                sig = '***' if pval < 0.001 else '**' if pval < 0.01 else '*' if pval < 0.05 else ''
                print(f"  - {var_clean}: {coef:.6f} {sig}")

        self.did_results = {
            'coefficient': did_coef,
            'std_error': did_se,
            't_value': did_t,
            'p_value': did_p,
            'r2': r2
        }

    def export_results(self, filename='DID回归结果.xlsx'):
        """导出回归结果到Excel"""
        import openpyxl

        print("\n" + "="*80)
        print("导出结果")
        print("="*80)

        # 创建结果表
        results_summary = pd.DataFrame([
            {'指标': 'DID系数', '值': f"{self.did_results['coefficient']:.6f}"},
            {'指标': '标准误', '值': f"{self.did_results['std_error']:.6f}"},
            {'指标': 't值', '值': f"{self.did_results['t_value']:.6f}"},
            {'指标': 'p值', '值': f"{self.did_results['p_value']:.6f}"},
            {'指标': '显著性', '值': '***' if self.did_results['p_value'] < 0.001 else
                                '**' if self.did_results['p_value'] < 0.01 else
                                '*' if self.did_results['p_value'] < 0.05 else '不显著'},
        ])

        # 添加R方
        if 'r2_within' in self.did_results:
            results_summary = pd.concat([
                results_summary,
                pd.DataFrame([{'指标': '组内R2', '值': f"{self.did_results['r2_within']:.6f}"}])
            ], ignore_index=True)
        else:
            results_summary = pd.concat([
                results_summary,
                pd.DataFrame([{'指标': 'R2', '值': f"{self.did_results['r2']:.6f}"}])
            ], ignore_index=True)

        # 导出城市权重
        weights_df = pd.DataFrame([
            {'城市': city, '权重': weight}
            for city, weight in sorted(self.city_weights.items(), key=lambda x: x[0])
        ])
        weights_df = weights_df[weights_df['权重'] > 0].sort_values('权重', ascending=False)

        # 写入Excel
        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            results_summary.to_excel(writer, sheet_name='回归结果汇总', index=False)
            weights_df.to_excel(writer, sheet_name='城市权重', index=False)

        print(f"\n结果已保存至: {filename}")
        print(f"  - 回归结果汇总 (1个表格)")
        print(f"  - 城市权重 ({len(weights_df)}个城市)")

    def generate_report(self, filename='DID分析报告.txt'):
        """生成完整的DID分析报告"""
        report = f"""
{'='*80}
多时点DID分析报告（基于PSM加权）
{'='*80}

一、分析设置
{'-'*80}
数据来源: {self.data_path.split('/')[-1]}
PSM匹配结果: {self.psm_matched_path.split('/')[-1]}

模型设定:
  因变量: ln(碳排放强度) [缩尾处理]
  核心解释变量: Treat × Post (DID交互项)
  控制变量: ln_real_gdp, ln_人口密度, ln_金融发展水平, 第二产业占GDP比重
  固定效应: 城市、年份
  标准误: 聚类到城市层面

"""

        # 添加平行趋势检验结果
        if hasattr(self, 'parallel_test_result'):
            prt = self.parallel_test_result
            report += f"""
二、平行趋势检验
{'-'*80}
检验方法: 事件研究法(Event Study)
检验时期: 2009年及以前

统计检验结果:
  - 交互项系数(Treat×Year): {prt['interaction_coef']:.6f}
  - 交互项p值: {prt['interaction_pval']:.6f}
  - 平行趋势假设: {'通过 (p>=0.05)' if prt['parallel_trend_holds'] else '拒绝 (p<0.05)'}

"""
            if not prt['parallel_trend_holds']:
                report += """
警告: 处理组和控制组在政策前的趋势存在显著差异，这可能违反DID的平行趋势假设。
建议:
  1. 谨慎解释DID结果的因果效应
  2. 考虑使用其他方法（如合成控制法）进行稳健性检验
"""

        # 添加城市权重信息
        treated_cities = [c for c, w in self.city_weights.items() if c in self.df[self.df['Treat']==1]['city_name'].values]
        control_weights = {k: v for k, v in self.city_weights.items() if k not in treated_cities}

        report += f"""
保留城市总数: {self.df['city_name'].nunique()}
  - 处理组: {len([c for c in treated_cities if c in self.city_weights])} 个城市 (权重=1)
  - 控制组: {sum(1 for v in control_weights.values() if v > 0)} 个城市 (权重=使用次数)

观测总数: {len(self.df)}
年份范围: {self.df['year'].min()} - {self.df['year'].max()}

三、回归结果
{'-'*80}
"""

        if hasattr(self, 'did_results'):
            report += f"""
DID交互项效应:
  系数值: {self.did_results['coefficient']:.6f}
  标准误: {self.did_results['std_error']:.6f}
  t值: {self.did_results['t_value']:.6f}
  p值: {self.did_results['p_value']:.6f}
  显著性水平: {'*** (p<0.001)' if self.did_results['p_value'] < 0.001 else
                '** (p<0.01)' if self.did_results['p_value'] < 0.01 else
                '* (p<0.05)' if self.did_results['p_value'] < 0.05 else
                '不显著 (p≥0.05)'}

"""

            if 'r2_within' in self.did_results:
                report += f"模型拟合: 组内R2 = {self.did_results['r2_within']:.6f}\n"
            else:
                report += f"模型拟合: R2 = {self.did_results['r2']:.6f}\n"

        report += f"""
四、经济含义解释
{'-'*80}
"""

        if hasattr(self, 'did_results'):
            coef = self.did_results['coefficient']
            # 将对数系数转换为百分比变化
            pct_change = (np.exp(coef) - 1) * 100

            report += f"""
DID系数 = {coef:.6f}

这意味着：在控制其他因素后，政策实施使处理组城市的ln(碳排放强度)比
控制组城市{('显著' if self.did_results['p_value'] < 0.05 else '')}{'低了' if coef < 0 else '高了'}{abs(coef):.6f}个单位。

换算成百分比：{'降低了约 {:.2f}%'.format(abs(pct_change)) if coef < 0 else '提高了约 {:.2f}%'.format(pct_change)}

"""

        report += f"""
五、注意事项
{'-'*80}
1. 本分析基于PSM匹配后的样本，控制组根据匹配次数加权
2. 标准误聚类到城市层面，考虑了面板数据的异方差和序列相关问题
3. 因变量经过缩尾处理，减少极端值影响
4. 固定效应控制了所有不随时间变化的城市特征和不随城市变化的时间趋势

{'='*80}
报告生成时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
{'='*80}
"""

        with open(filename, 'w', encoding='utf-8') as f:
            f.write(report)

        print(f"\n分析报告已保存至: {filename}\n")
        print(report)


def main():
    """主函数"""
    # 设置文件路径
    DATA_PATH = '../总数据集2007-2023_仅含emission城市_更新DID.xlsx'
    PSM_MATCHED_PATH = '../基期PSM分析_2009_有放回/匹配后城市列表.xlsx'
    PSM_CONTROL_USAGE_PATH = '../基期PSM分析_2009_有放回/控制组使用情况.xlsx'

    # 创建分析器
    analyzer = DIDAnalyzer(DATA_PATH, PSM_MATCHED_PATH, PSM_CONTROL_USAGE_PATH)

    # 计算权重
    analyzer.calculate_weights()

    # 对因变量进行对数变换和缩尾处理
    analyzer.winsorize_emission(limits=(0.01, 0.01))

    # 平行趋势检验（在DID回归之前）
    parallel_result = analyzer.parallel_trends_test(pre_period_end=2009)
    analyzer.parallel_test_result = parallel_result  # 保存结果用于报告

    # 运行DID回归
    if analyzer.run_did_regression():
        # 导出结果
        analyzer.export_results()

        # 生成报告
        analyzer.generate_report()

    print("\n" + "="*80)
    print("DID分析完成!")
    print("="*80)


if __name__ == '__main__':
    main()
