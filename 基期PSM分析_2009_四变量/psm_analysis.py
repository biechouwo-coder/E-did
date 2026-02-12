"""
基期PSM分析（倾向得分匹配）
====================================
研究范围: 2007-2019年
基期: 2009年
匹配变量: ln_real_gdp, ln_人口密度, ln_金融发展水平, 第二产业占GDP比重
卡尺: 倾向得分对数几率标准差的0.25倍
匹配方法: 1:1 有放回匹配
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


class PSMAnalyzer:
    """倾向得分匹配分析器（基期匹配）"""

    def __init__(self, data_path, base_year=2009, start_year=2007, end_year=2019,
                 caliper=0.25, ratio=1, replacement=True):
        """
        初始化PSM分析器

        参数:
        - data_path: 原始数据路径
        - base_year: 基期年份（用于匹配）
        - start_year: 研究起始年份
        - end_year: 研究结束年份
        - caliper: 卡尺（倾向得分对数几率标准差的倍数）
        - ratio: 匹配比例（控制组:处理组）
        - replacement: 是否有放回匹配
        """
        self.data_path = data_path
        self.base_year = base_year
        self.start_year = start_year
        self.end_year = end_year
        self.caliper = caliper
        self.ratio = ratio
        self.replacement = replacement

        # 匹配变量
        self.match_vars = ['ln_real_gdp', 'ln_人口密度', 'ln_金融发展水平', '第二产业占GDP比重']

        # 数据
        self.df = None
        self.base_data = None
        self.propensity_scores = None
        self.matches = None
        self.matched_data = None

        # 读取数据
        self._load_data()

    def _load_data(self):
        """读取并预处理数据"""
        print("="*80)
        print("数据加载与预处理")
        print("="*80)

        # 读取原始数据
        print(f"\n正在读取数据: {self.data_path}")
        df_full = pd.read_excel(self.data_path)
        print(f"  - 原始数据形状: {df_full.shape}")
        print(f"  - 原始年份范围: {df_full['year'].min()} - {df_full['year'].max()}")

        # 筛选研究年份范围
        self.df = df_full[df_full['year'].between(self.start_year, self.end_year)].copy()
        print(f"\n筛选研究范围 ({self.start_year}-{self.end_year}年):")
        print(f"  - 筛选后数据形状: {self.df.shape}")
        print(f"  - 城市数量: {self.df['city_name'].nunique()}")

        # 筛选基期数据
        self.base_data = self.df[self.df['year'] == self.base_year].copy()
        print(f"\n基期数据 ({self.base_year}年):")
        print(f"  - 观测数: {len(self.base_data)}")
        print(f"  - 处理组: {(self.base_data['Treat'] == 1).sum()} 个城市")
        print(f"  - 控制组: {(self.base_data['Treat'] == 0).sum()} 个城市")

        # 检查匹配变量
        print(f"\n匹配变量:")
        for var in self.match_vars:
            if var in self.base_data.columns:
                missing = self.base_data[var].isna().sum()
                print(f"  [OK] {var}: 缺失{missing}个")
            else:
                print(f"  [X] {var}: 不存在！")
                raise ValueError(f"变量 {var} 不存在")

        # 删除包含缺失值的基期数据
        n_before = len(self.base_data)
        self.base_data = self.base_data.dropna(subset=self.match_vars)
        n_after = len(self.base_data)
        if n_before != n_after:
            print(f"\n删除基期缺失值:")
            print(f"  - 删除前: {n_before}")
            print(f"  - 删除后: {n_after}")
            print(f"  - 删除: {n_before - n_after}")

    def calculate_propensity_scores(self):
        """计算倾向得分"""
        print("\n" + "="*80)
        print("计算倾向得分")
        print("="*80)

        # 准备数据
        X = self.base_data[self.match_vars].values
        y = self.base_data['Treat'].values

        print(f"\n使用Logistic回归计算倾向得分")
        print(f"  - 样本数: {len(X)}")
        print(f"  - 处理组: {y.sum()}")
        print(f"  - 控制组: {(y == 0).sum()}")
        print(f"  - 匹配变量数: {len(self.match_vars)}")

        # 标准化特征
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # 计算倾向得分（使用Logistic回归）
        logit = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
        logit.fit(X_scaled, y)

        # 获取倾向得分
        propensity_scores = logit.predict_proba(X_scaled)[:, 1]

        # 添加到基期数据
        self.base_data['propensity_score'] = propensity_scores

        # 统计
        print(f"\n倾向得分统计:")
        print(f"  - 处理组: 均值={propensity_scores[y==1].mean():.4f}, " +
              f"标准差={propensity_scores[y==1].std():.4f}")
        print(f"  - 控制组: 均值={propensity_scores[y==0].mean():.4f}, " +
              f"标准差={propensity_scores[y==0].std():.4f}")

        # 计算对数几率
        logit_ps = np.log(propensity_scores / (1 - propensity_scores) + 1e-10)
        self.base_data['logit_ps'] = logit_ps

        # 计算卡尺
        logit_std = np.std(logit_ps)
        self.caliper_value = self.caliper * logit_std

        print(f"\n卡尺设置:")
        print(f"  - 对数几率标准差: {logit_std:.4f}")
        print(f"  - 卡尺系数: {self.caliper}")
        print(f"  - 实际卡尺值: {self.caliper_value:.4f}")

        self.propensity_scores = propensity_scores

    def match(self):
        """执行倾向得分匹配（1:1有放回）"""
        print("\n" + "="*80)
        print("执行倾向得分匹配")
        print("="*80)

        # 分离处理组和控制组
        treated = self.base_data[self.base_data['Treat'] == 1].copy()
        control = self.base_data[self.base_data['Treat'] == 0].copy()

        print(f"\n匹配前:")
        print(f"  - 处理组: {len(treated)} 个城市")
        print(f"  - 控制组: {len(control)} 个城市")

        # 存储匹配结果
        matches_list = []
        control_usage = {}  # 记录每个控制组被使用的次数

        # 对每个处理组单位进行匹配
        for idx, treated_unit in treated.iterrows():
            treated_ps = treated_unit['propensity_score']
            treated_logit_ps = treated_unit['logit_ps']

            # 计算与所有控制组的距离（使用对数几率）
            control['distance'] = np.abs(control['logit_ps'] - treated_logit_ps)

            # 应用卡尺限制
            eligible_controls = control[control['distance'] <= self.caliper_value]

            if len(eligible_controls) > 0:
                # 找到最近的控制组
                best_match = eligible_controls.loc[eligible_controls['distance'].idxmin()]

                matches_list.append({
                    'treated_city': treated_unit['city_name'],
                    'treated_ps': treated_ps,
                    'control_city': best_match['city_name'],
                    'control_ps': best_match['propensity_score'],
                    'distance': best_match['distance']
                })

                # 更新控制组使用次数
                control_city = best_match['city_name']
                control_usage[control_city] = control_usage.get(control_city, 0) + 1
            else:
                # 没有找到符合条件的匹配
                matches_list.append({
                    'treated_city': treated_unit['city_name'],
                    'treated_ps': treated_ps,
                    'control_city': None,
                    'control_ps': None,
                    'distance': None
                })

        self.matches = pd.DataFrame(matches_list)

        # 统计匹配结果
        n_matched = self.matches['control_city'].notna().sum()
        n_unmatched = self.matches['control_city'].isna().sum()

        print(f"\n匹配结果:")
        print(f"  - 成功匹配: {n_matched} 个处理组城市")
        print(f"  - 未匹配: {n_unmatched} 个处理组城市")
        print(f"  - 匹配率: {n_matched/len(treated)*100:.1f}%")

        # 控制组使用情况
        print(f"\n控制组使用情况:")
        print(f"  - 被使用的控制组城市数: {len(control_usage)}")
        print(f"  - 平均使用次数: {np.mean(list(control_usage.values())):.2f}")
        print(f"  - 最大使用次数: {max(control_usage.values())}")

        # 创建控制组使用情况表
        control_usage_df = pd.DataFrame([
            {'城市': city, '使用次数': count}
            for city, count in sorted(control_usage.items(), key=lambda x: x[1], reverse=True)
        ])
        self.control_usage = control_usage_df

        # 创建匹配后的城市列表
        matched_cities = []
        for _, row in self.matches.iterrows():
            if pd.notna(row['control_city']):
                matched_cities.append({
                    'city_name': row['treated_city'],
                    '组别': '处理组',
                    '匹配城市': row['control_city'],
                    '倾向得分': row['treated_ps'],
                    '距离': row['distance']
                })
                matched_cities.append({
                    'city_name': row['control_city'],
                    '组别': '控制组',
                    '匹配城市': row['treated_city'],
                    '倾向得分': row['control_ps'],
                    '距离': row['distance']
                })

        self.matched_data = pd.DataFrame(matched_cities)

    def check_balance(self):
        """检验匹配后的平衡性"""
        print("\n" + "="*80)
        print("平衡性检验")
        print("="*80)

        if self.matches is None or self.matches['control_city'].isna().all():
            print("警告: 没有成功匹配的样本")
            return

        # 合并处理组和匹配的控制组
        matched_control_cities = self.matches[self.matches['control_city'].notna()]['control_city'].unique()
        treated_data = self.base_data[self.base_data['Treat'] == 1]
        control_data = self.base_data[self.base_data['city_name'].isin(matched_control_cities)]

        print(f"\n匹配后样本:")
        print(f"  - 处理组: {len(treated_data)} 个城市")
        print(f"  - 控制组: {len(control_data)} 个城市")

        # 计算标准化差异
        print(f"\n变量平衡性检验:")
        print(f"{'变量':<20} {'处理组均值':<12} {'控制组均值':<12} {'标准化差异(%)':<15} {'是否平衡'}")
        print("-"*80)

        balance_results = []
        for var in self.match_vars:
            treated_mean = treated_data[var].mean()
            control_mean = control_data[var].mean()
            treated_std = treated_data[var].std()
            control_std = control_data[var].std()

            # 标准化差异（Standardized Difference）
            pooled_std = np.sqrt((treated_std**2 + control_std**2) / 2)
            std_diff = abs(treated_mean - control_mean) / pooled_std * 100 if pooled_std > 0 else 0

            is_balanced = "[OK]" if std_diff < 20 else "[X]"

            print(f"{var:<20} {treated_mean:<12.4f} {control_mean:<12.4f} {std_diff:<15.2f} {is_balanced}")

            balance_results.append({
                '变量': var,
                '处理组均值': treated_mean,
                '控制组均值': control_mean,
                '标准化差异(%)': std_diff,
                '是否平衡': is_balanced
            })

        self.balance_results = pd.DataFrame(balance_results)

        self.balance_results = pd.DataFrame(balance_results)

    def visualize_propensity_scores(self):
        """可视化倾向得分分布"""
        print("\n" + "="*80)
        print("生成倾向得分分布图")
        print("="*80)

        if self.base_data is None or 'propensity_score' not in self.base_data.columns:
            print("错误: 尚未计算倾向得分")
            return

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # 1. 匹配前倾向得分分布（直方图）
        treated_ps = self.base_data[self.base_data['Treat'] == 1]['propensity_score']
        control_ps = self.base_data[self.base_data['Treat'] == 0]['propensity_score']

        axes[0, 0].hist(treated_ps, bins=30, alpha=0.6, label='处理组', color='coral', density=True)
        axes[0, 0].hist(control_ps, bins=30, alpha=0.6, label='控制组', color='steelblue', density=True)
        axes[0, 0].set_xlabel('倾向得分', fontsize=11)
        axes[0, 0].set_ylabel('密度', fontsize=11)
        axes[0, 0].set_title('匹配前倾向得分分布', fontsize=13, fontweight='bold')
        axes[0, 0].legend(fontsize=10)
        axes[0, 0].grid(axis='y', alpha=0.3)

        # 2. 匹配前后倾向得分对比（箱线图）
        if self.matches is not None and not self.matches['control_city'].isna().all():
            matched_treated_ps = self.matches[self.matches['control_city'].notna()]['treated_ps']
            matched_control_ps = self.matches[self.matches['control_city'].notna()]['control_ps']

            bp_data = [treated_ps.values, control_ps.values, matched_treated_ps.values, matched_control_ps.values]
            bp_labels = ['处理组\n(匹配前)', '控制组\n(匹配前)', '处理组\n(匹配后)', '控制组\n(匹配后)']
            bp_colors = ['coral', 'steelblue', 'coral', 'steelblue']

            bp = axes[0, 1].boxplot(bp_data, labels=bp_labels, patch_artist=True)
            for patch, color in zip(bp['boxes'], bp_colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.6)
            axes[0, 1].set_ylabel('倾向得分', fontsize=11)
            axes[0, 1].set_title('匹配前后倾向得分对比', fontsize=13, fontweight='bold')
            axes[0, 1].grid(axis='y', alpha=0.3)
            plt.setp(axes[0, 1].xaxis.get_majorticklabels(), fontsize=9)

        # 3. 对数几率分布
        treated_logit = self.base_data[self.base_data['Treat'] == 1]['logit_ps']
        control_logit = self.base_data[self.base_data['Treat'] == 0]['logit_ps']

        axes[1, 0].hist(treated_logit, bins=30, alpha=0.6, label='处理组', color='coral', density=True)
        axes[1, 0].hist(control_logit, bins=30, alpha=0.6, label='控制组', color='steelblue', density=True)
        axes[1, 0].set_xlabel('对数几率(Logit)', fontsize=11)
        axes[1, 0].set_ylabel('密度', fontsize=11)
        axes[1, 0].set_title('对数几率分布', fontsize=13, fontweight='bold')
        axes[1, 0].legend(fontsize=10)
        axes[1, 0].grid(axis='y', alpha=0.3)

        # 4. 匹配距离分布
        if self.matches is not None and not self.matches['control_city'].isna().all():
            distances = self.matches[self.matches['control_city'].notna()]['distance']
            axes[1, 1].hist(distances, bins=30, color='steelblue', alpha=0.7, edgecolor='black')
            axes[1, 1].axvline(distances.mean(), color='red', linestyle='--', linewidth=2,
                              label=f'平均距离={distances.mean():.4f}')
            axes[1, 1].axvline(self.caliper_value, color='orange', linestyle=':', linewidth=2,
                              label=f'卡尺={self.caliper_value:.4f}')
            axes[1, 1].set_xlabel('匹配距离（对数几率之差）', fontsize=11)
            axes[1, 1].set_ylabel('频数', fontsize=11)
            axes[1, 1].set_title('匹配距离分布', fontsize=13, fontweight='bold')
            axes[1, 1].legend(fontsize=10)
            axes[1, 1].grid(axis='y', alpha=0.3)

        plt.tight_layout()
        plt.savefig('倾向得分分布.png', dpi=300, bbox_inches='tight')
        print(f"\n图表已保存: 倾向得分分布.png")
        plt.close()

    def visualize_balance(self):
        """可视化平衡性检验结果"""
        print("\n" + "="*80)
        print("生成平衡性检验图")
        print("="*80)

        if not hasattr(self, 'balance_results'):
            print("警告: 尚未进行平衡性检验")
            return

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # 1. 标准化差异图
        y_pos = np.arange(len(self.balance_results))
        std_diffs = self.balance_results['标准化差异(%)'].values
        colors = ['green' if x < 20 else 'red' for x in std_diffs]

        axes[0, 0].barh(y_pos, std_diffs, color=colors, alpha=0.6, edgecolor='black')
        axes[0, 0].axvline(x=20, color='red', linestyle='--', linewidth=2, label='阈值(20%)')
        axes[0, 0].set_yticks(y_pos)
        axes[0, 0].set_yticklabels(self.balance_results['变量'])
        axes[0, 0].set_xlabel('标准化差异(%)', fontsize=11)
        axes[0, 0].set_title('变量标准化差异', fontsize=13, fontweight='bold')
        axes[0, 0].legend(fontsize=10)
        axes[0, 0].grid(axis='x', alpha=0.3)

        # 2-5. 各变量的箱线图对比
        if self.matches is not None and not self.matches['control_city'].isna().all():
            matched_control_cities = self.matches[self.matches['control_city'].notna()]['control_city'].unique()
            treated_data = self.base_data[self.base_data['Treat'] == 1]
            control_data = self.base_data[self.base_data['city_name'].isin(matched_control_cities)]

            positions = [(0, 1), (1, 0), (1, 1)]
            for i, var in enumerate(self.match_vars[:3]):
                row, col = positions[i]
                treated_values = treated_data[var].dropna().values
                control_values = control_data[var].dropna().values

                bp = axes[row, col].boxplot([treated_values, control_values],
                                           labels=['处理组', '控制组'],
                                           patch_artist=True)
                bp['boxes'][0].set_facecolor('coral')
                bp['boxes'][0].set_alpha(0.6)
                bp['boxes'][1].set_facecolor('steelblue')
                bp['boxes'][1].set_alpha(0.6)

                axes[row, col].set_ylabel(var, fontsize=11)
                axes[row, col].set_title(f'{var} 分布对比', fontsize=12, fontweight='bold')
                axes[row, col].grid(axis='y', alpha=0.3)

        plt.tight_layout()
        plt.savefig('平衡性检验.png', dpi=300, bbox_inches='tight')
        print(f"\n图表已保存: 平衡性检验.png")
        plt.close()

    def export_results(self):
        """导出匹配结果"""
        print("\n" + "="*80)
        print("导出结果")
        print("="*80)

        with pd.ExcelWriter('PSM匹配结果.xlsx', engine='openpyxl') as writer:
            # 1. 匹配后的城市列表
            if self.matched_data is not None:
                self.matched_data.to_excel(writer, sheet_name='匹配后城市列表', index=False)
                print(f"\n[OK] 匹配后城市列表: {len(self.matched_data)} 个城市")

            # 2. 匹配详情
            if self.matches is not None:
                self.matches.to_excel(writer, sheet_name='匹配详情', index=False)
                print(f"[OK] 匹配详情: {len(self.matches)} 个处理组匹配")

            # 3. 控制组使用情况
            if hasattr(self, 'control_usage'):
                self.control_usage.to_excel(writer, sheet_name='控制组使用情况', index=False)
                print(f"[OK] 控制组使用情况: {len(self.control_usage)} 个城市")

            # 4. 平衡性检验结果
            if hasattr(self, 'balance_results'):
                self.balance_results.to_excel(writer, sheet_name='平衡性检验', index=False)
                print(f"[OK] 平衡性检验: {len(self.balance_results)} 个变量")

            # 5. 基期数据（含倾向得分）
            if self.base_data is not None:
                export_cols = ['city_name', 'Treat', 'year'] + self.match_vars + ['propensity_score', 'logit_ps']
                self.base_data[export_cols].to_excel(writer, sheet_name='基期数据', index=False)
                print(f"[OK] 基期数据: {len(self.base_data)} 个观测")

        print(f"\n所有结果已保存至: PSM匹配结果.xlsx")

    def generate_report(self):
        """生成完整的PSM分析报告"""
        report = f"""
{'='*80}
基期PSM分析报告
{'='*80}

一、分析设置
{'-'*80}
数据来源: {self.data_path.split('/')[-1]}
研究范围: {self.start_year}-{self.end_year}年
基期年份: {self.base_year}年

匹配设置:
  - 匹配变量: {', '.join(self.match_vars)}
  - 匹配方法: 1:{self.ratio} {'有放回' if self.replacement else '无放回'}
  - 卡尺: 倾向得分对数几率标准差的 {self.caliper} 倍
  - 实际卡尺值: {self.caliper_value:.4f}

"""

        if self.base_data is not None:
            treated_count = (self.base_data['Treat'] == 1).sum()
            control_count = (self.base_data['Treat'] == 0).sum()
            report += f"""
基期数据 ({self.base_year}年):
  - 处理组: {treated_count} 个城市
  - 控制组: {control_count} 个城市
  - 总样本: {len(self.base_data)} 个城市

"""

        if self.matches is not None:
            n_matched = self.matches['control_city'].notna().sum()
            n_unmatched = self.matches['control_city'].isna().sum()
            match_rate = n_matched / len(self.matches) * 100 if len(self.matches) > 0 else 0

            avg_distance = self.matches[self.matches['control_city'].notna()]['distance'].mean()

            report += f"""
二、匹配结果
{'-'*80}
  - 成功匹配: {n_matched} 个处理组城市
  - 未匹配: {n_unmatched} 个处理组城市
  - 匹配率: {match_rate:.1f}%
  - 平均匹配距离: {avg_distance:.4f}

控制组使用情况:
  - 被使用的控制组城市数: {len(self.control_usage)} 个
  - 平均使用次数: {self.control_usage['使用次数'].mean():.2f} 次
  - 最大使用次数: {self.control_usage['使用次数'].max()} 次

"""

        if hasattr(self, 'balance_results'):
            report += f"""
三、平衡性检验
{'-'*80}
变量标准化差异:
"""
            for _, row in self.balance_results.iterrows():
                balanced = "[OK] 平衡" if row['是否平衡'] == "[OK]" else "[X] 不平衡"
                report += f"  - {row['变量']}: {row['标准化差异(%)']:.2f}% ({balanced})\n"

            n_balanced = (self.balance_results['是否平衡'] == "[OK]").sum()
            n_total = len(self.balance_results)
            report += f"\n平衡性总结: {n_balanced}/{n_total} 个变量达到平衡标准\n"

        report += f"""
四、输出文件
{'-'*80}
  - PSM匹配结果.xlsx: 包含所有匹配结果和检验
  - 倾向得分分布.png: 匹配前后倾向得分对比
  - 平衡性检验.png: 变量平衡性可视化

{'='*80}
报告生成时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
{'='*80}
"""

        # 保存报告
        with open('PSM分析报告.txt', 'w', encoding='utf-8') as f:
            f.write(report)

        print(f"\n分析报告已保存至: PSM分析报告.txt\n")
        print(report)


def main():
    """主函数"""
    print("\n" + "="*80)
    print("基期PSM分析（2009年基期，2007-2019年研究范围）")
    print("="*80)

    # 设置文件路径
    DATA_PATH = '../总数据集2007-2023_仅含emission城市_更新DID.xlsx'

    # 创建分析器
    analyzer = PSMAnalyzer(
        data_path=DATA_PATH,
        base_year=2009,
        start_year=2007,
        end_year=2019,
        caliper=0.25,
        ratio=1,
        replacement=True
    )

    # 计算倾向得分
    analyzer.calculate_propensity_scores()

    # 执行匹配
    analyzer.match()

    # 平衡性检验
    analyzer.check_balance()

    # 可视化
    analyzer.visualize_propensity_scores()
    analyzer.visualize_balance()

    # 导出结果
    analyzer.export_results()

    # 生成报告
    analyzer.generate_report()

    print("\n" + "="*80)
    print("PSM分析完成!")
    print("="*80)


if __name__ == '__main__':
    main()
