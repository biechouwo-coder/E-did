# DID-CEADs: 碳排放强度双重差分分析

本项目使用双重差分法（DID）和倾向得分匹配法（PSM）研究中国城市低碳试点的政策效果。

## 项目简介

本项目基于CEADs（中国碳排放数据库）数据，运用双重差分法（Difference-in-Differences, DID）和倾向得分匹配法（Propensity Score Matching, PSM）评估低碳城市试点政策对碳排放强度的影响。

## 数据说明

### 数据来源
- **数据来源**: CEADs（中国碳排放数据库）
- **时间跨度**: 2007-2019年
- **地理范围**: 中国地级市
- **主要变量**:
  - `emission`: 碳排放量
  - `emission_intensity`: 碳排放强度
  - `ln_emission_intensity_winzorized`: 经缩尾处理的碳排放强度对数
  - `Treat`: 处理组标识（1=低碳试点城市，0=非试点城市）
  - `Post`: 政策时间标识（1=政策实施后，0=政策实施前）

### 协变量
PSM分析使用以下四个匹配变量：

| 变量名 | 说明 | 预期影响 |
|--------|------|----------|
| `ln_real_gdp` | 实际GDP对数 | 经济发展水平 |
| `ln_pop_density` | 人口密度对数 | 城市密集程度 |
| `ln_financial_dev` | 金融发展水平对数 | 金融化程度 |
| `secondary_industry_share` | 第二产业占GDP比重 | 产业结构 |

## PSM分析方法

### 基期设定
- **基期年份**: 2009年
- **匹配方式**: 1:1 有放回匹配
- **卡尺设定**: 0.25 × 倾向得分对数几率标准差

### 匹配步骤
1. **数据准备**: 提取基期（2009年）数据，删除缺失值
2. **倾向得分估计**: 使用逻辑回归（Logistic Regression）估计倾向得分
3. **匹配执行**: 基于1:1有放回匹配，使用倾向得分对数几率计算距离
4. **平衡性检验**: 检查匹配后协变量平衡性（标准化偏差<20%为优）

### 技术细节
- **倾向得分模型**: 逻辑回归（Logistic Regression）
- **协变量标准化**: StandardScaler标准化
- **距离度量**: 倾向得分对数几率（logit PS）的绝对距离
- **匹配算法**: 最近邻匹配（Nearest Neighbor Matching）

## 使用说明

### 环境要求
```bash
pip install pandas numpy scikit-learn matplotlib openpyxl
```

### 运行PSM分析
Windows系统：
```bash
run_psm.bat
```

或直接运行Python脚本：
```bash
python psm_analysis.py
```

### 输出文件

分析完成后会生成以下文件：

1. **PSM匹配结果_2009基期_四变量.xlsx**
   - 匹配后的完整数据
   - 包含倾向得分、匹配标识等信息

2. **PSM匹配详情_2009基期_四变量.xlsx**
   - 每个处理组城市的匹配详情
   - 匹配距离、是否成功匹配等信息

3. **PSM平衡性检验_2009基期_四变量.xlsx**
   - 匹配前后协变量平衡性检验结果
   - 标准化偏差统计

4. **PSM匹配结果_2009基期_四变量.png**
   - 可视化图表（4个子图）
   - 倾向得分分布、协变量平衡性、匹配距离分布

## 分析结果示例

### 匹配成功率
- **处理组样本数**: 110个城市
- **成功匹配样本数**: 101个城市
- **匹配成功率**: 91.8%

### 平衡性检验结果
| 变量 | 处理组均值 | 控制组均值 | 标准化偏差 |
|------|-----------|-----------|-----------|
| ln_real_gdp | 6.78 | 6.57 | 22.16% |
| ln_pop_density | 5.93 | 5.77 | 17.64% |
| ln_financial_dev | 0.77 | 0.75 | 5.88% |
| secondary_industry_share | 0.49 | 0.48 | 10.53% |

注：标准化偏差绝对值小于20%表示平衡性较好。

## 项目结构

```
did-CEADs/
├── 总数据集2007-2019_含碳排放强度.xlsx  # 原始数据
├── psm_analysis.py                        # PSM分析主脚本
├── run_psm.bat                          # Windows运行脚本
├── PSM匹配结果_2009基期_四变量.xlsx     # 匹配结果数据
├── PSM匹配详情_2009基期_四变量.xlsx     # 匹配详情
├── PSM平衡性检验_2009基期_四变量.xlsx   # 平衡性检验
├── PSM匹配结果_2009基期_四变量.png      # 可视化图表
├── 参考文献/                             # 参考文献目录
├── 框架文件/                            # 框架说明文件
└── README.md                             # 项目说明文档（本文件）
```

## 参考文献

### 倾向得分匹配方法
- Rosenbaum, P. R., & Rubin, D. B. (1983). The central role of the propensity score in observational studies for causal effects. *Biometrika*, 70(1), 41-55.
- Austin, P. C. (2011). An introduction to propensity score methods for reducing the effects of confounding in observational studies. *Multivariate Behavioral Research*, 46(3), 399-424.

### 双重差分法
- Card, D., & Krueger, A. B. (1994). Minimum wages and employment: A case study of the fast-food industry. *American Economic Review*, 84(4), 772-793.
- Angrist, J. D., & Pischke, J. S. (2009). *Mostly harmless econometrics: An empiricist's companion*. Princeton university press.

### 低碳城市政策评估
- 相关低碳试点政策效果评估文献...

## 更新日志

### 2025-02-12
- 创建PSM分析脚本（四变量匹配）
- 基期设定为2009年
- 实现1:1有放回匹配
- 添加可视化输出功能

## 许可证

本项目仅供学术研究使用。

## 联系方式

如有问题或建议，请通过GitHub Issues联系。

---

**最后更新**: 2025年2月12日
