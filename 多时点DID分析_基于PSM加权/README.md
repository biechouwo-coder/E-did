# 多时点DID分析（基于PSM加权）

## 分析说明

本文件夹包含基于PSM匹配结果的多时点双重差分(DID)分析代码。

### 分析特点

- **样本筛选**：利用PSM匹配结果，根据城市在匹配中的出现次数加权
- **因变量处理**：ln(碳排放强度) + 缩尾处理（上下各1%）
- **固定效应**：城市固定效应 + 年份固定效应
- **标准误聚类**：聚类到城市层面

## 文件说明

- `did_analysis.py`: DID分析主脚本
- `requirements.txt`: Python依赖包列表
- `run_did.bat`: Windows运行脚本

## 使用方法

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

**推荐安装linearmodels**以获得更准确的标准误估计：

```bash
pip install linearmodels
```

### 2. 运行分析

**Windows用户**: 直接双击 `run_did.bat`

或在命令行中运行:

```bash
python did_analysis.py
```

### 3. 输出文件

运行后会生成以下文件:

- `DID回归结果.xlsx`: 包含回归结果汇总和城市权重
- `ln_emission分布对比.png`: 缩尾前后的分布对比图
- `DID分析报告.txt`: 完整的分析报告

## 分析流程

### 1. 权重计算

根据PSM匹配结果计算每个城市的权重：

| 组别 | 条件 | 权重 |
|------|------|------|
| 处理组 | Treat=1，且被匹配 | 1 |
| 控制组 | Treat=0，且被匹配 | 被使用的次数 |
| 控制组 | Treat=0，未被匹配 | 0（剔除） |

### 2. 样本筛选

只保留权重 > 0 的城市的所有观测

### 3. 因变量处理

- 计算ln(碳排放强度)
- 进行缩尾处理（winsorize），上下各1%

### 4. DID回归

**模型设定**:

```
ln_emission_winsor = β0 + β1×(Treat×Post) + γX + μ_city + λ_year + ε
```

**变量说明**:
- `Treat×Post`: DID交互项（核心解释变量）
- `X`: 控制变量（ln_real_gdp, ln_人口密度, ln_金融发展水平, 第二产业占GDP比重）
- `μ_city`: 城市固定效应
- `λ_year`: 年份固定效应

### 5. 结果解释

- DID系数表示政策效应
- 显著性水平：*** p<0.001, ** p<0.01, * p<0.05

## 模型细节

### PSM加权原理

传统PSM-DID分析中，控制组城市根据在匹配中被使用的次数进行加权：

- **处理组**：每个城市权重为1
- **控制组**：权重 = 在PSM匹配中被选中的次数

这种方法的优势：
1. 充分利用PSM匹配信息
2. 给更相似的控制组城市更高权重
3. 提高估计效率

### 缩尾处理

对ln(碳排放强度)进行1%水平的缩尾处理：
- 将低于1%分位数的值替换为1%分位数值
- 将高于99%分位数的值替换为99%分位数值

目的：减少极端值对回归结果的影响

### 固定效应

- **城市固定效应**：控制所有不随时间变化的城市特征（如地理、文化等）
- **年份固定效应**：控制所有城市共同的时间趋势（如宏观经济周期等）

### 标准误聚类

将标准误聚类到城市层面，以处理：
- 同一城市不同时期的误差相关性
- 异方差问题

## 注意事项

1. **数据依赖**: 需要先运行PSM分析，生成匹配后的城市列表和控制组使用情况

2. **列名要求**: 原始数据需包含以下列：
   - `city_name`: 城市名称
   - `year`: 年份
   - `emission`: 碳排放强度
   - `Treat`: 处理组标识
   - `Post`: 政策后标识
   - `DID`: Treat×Post交互项
   - 控制变量列名

3. **linearmodels vs statsmodels**:
   - `linearmodels`: 原生支持聚类标准误，推荐使用
   - `statsmodels`: 不支持聚类，但可以作为备选方案

## 参考文献

多时点DID模型：
- Callaway, B., & Sant'Anna, P. H. (2021). Difference-in-differences with multiple time periods. *Journal of Econometrics*.

PSM-DID方法：
- Heckman, J. J., Ichimura, H., & Todd, P. E. (1997). Matching as an econometric evaluation estimator. *Review of Economic Studies*.
