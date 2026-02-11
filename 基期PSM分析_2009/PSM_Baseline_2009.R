# ==============================================================================
# 基期PSM分析 (2009年基期)
# ==============================================================================
# 使用变量: ln_real_gdp, ln_人口密度, ln_金融发展水平, 第二产业占GDP比重
# 卡尺: 倾向得分对数几率标准差的0.25倍
# ==============================================================================

# 加载必要的包
library(MatchIt)
library(optmatch)
library(Match)
library(ggplot2)
library(cobalt)
library(openxlsx)
library(dplyr)

# 设置工作目录
setwd("基期PSM分析_2009")

# ==============================================================================
# 1. 数据加载
# ==============================================================================
cat("========================================\n")
cat("1. 加载数据\n")
cat("========================================\n")

# 读取Excel数据
df <- read.xlsx("../总数据集2007-2023_仅含emission城市_更新DID.xlsx")

cat(sprintf("✓ 数据加载成功: %d行 × %d列\n", nrow(df), ncol(df)))

# 显示列名
cat("\n数据列名（前30列）:\n")
for(i in 1:min(30, ncol(df))) {
  cat(sprintf("  %2d. %s\n", i, names(df)[i]))
}

# ==============================================================================
# 2. 准备基期数据
# ==============================================================================
cat("\n========================================\n")
cat("2. 准备基期数据 (2009年)\n")
cat("========================================\n")

# 查找年份列
year_col <- NULL
for(col in names(df)) {
  if(grepl("年|year|Year", col, ignore.case = TRUE)) {
    year_col <- col
    break
  }
}

if(is.null(year_col)) {
  # 假设第3列是年份
  year_col <- names(df)[3]
}

cat(sprintf("使用年份列: %s\n", year_col))

# 筛选2009年数据
df_baseline <- df[df[[year_col]] == 2009, ]
cat(sprintf("✓ 基期样本数: %d\n", nrow(df_baseline)))

# ==============================================================================
# 3. 变量检测
# ==============================================================================
cat("\n========================================\n")
cat("3. 检测变量名\n")
cat("========================================\n")

# 查找处理变量
treatment_var <- NULL
for(col in names(df_baseline)) {
  if(grepl("treat|处理|group|Group", col, ignore.case = TRUE)) {
    treatment_var <- col
    break
  }
}

if(is.null(treatment_var)) {
  cat("✗ 未找到处理变量，请手动指定\n")
  # 可以手动设置，例如: treatment_var <- "treat"
  stop("需要指定处理变量")
}

cat(sprintf("✓ 处理变量: %s\n", treatment_var))

# 查找匹配变量
covariates <- list()

# 1. ln_real_gdp (真实GDP对数)
found_gdp <- FALSE
for(col in names(df_baseline)) {
  if(grepl("gdp|GDP", col) && !grepl("per|增长率|growth", col, ignore.case = TRUE)) {
    covariates$ln_real_gdp <- col
    cat(sprintf("✓ 找到 ln_real_gdp: %s\n", col))
    found_gdp <- TRUE
    break
  }
}

# 2. ln_人口密度 (人口密度对数)
found_pop <- FALSE
for(col in names(df_baseline)) {
  if(grepl("人口密度|pop.*density|density", col, ignore.case = TRUE)) {
    covariates$ln_人口密度 <- col
    cat(sprintf("✓ 找到 ln_人口密度: %s\n", col))
    found_pop <- TRUE
    break
  }
}

# 3. ln_金融发展水平 (金融发展水平对数)
found_finance <- FALSE
for(col in names(df_baseline)) {
  if(grepl("金融|finance|loan|credit|deposit", col, ignore.case = TRUE)) {
    covariates$ln_金融发展水平 <- col
    cat(sprintf("✓ 找到 ln_金融发展水平: %s\n", col))
    found_finance <- TRUE
    break
  }
}

# 4. 第二产业占GDP比重
found_industry <- FALSE
for(col in names(df_baseline)) {
  if(grepl("第二产业|secondary|industry", col, ignore.case = TRUE)) {
    covariates$第二产业占GDP比重 <- col
    cat(sprintf("✓ 找到 第二产业占GDP比重: %s\n", col))
    found_industry <- TRUE
    break
  }
}

# 使用找到的变量名
covariate_names <- unlist(covariates)

if(length(covariate_names) < 4) {
  cat("\n警告: 部分匹配变量未找到\n")
  cat("已找到的变量:", paste(covariate_names, collapse = ", "), "\n")
}

# ==============================================================================
# 4. 准备分析数据
# ==============================================================================
cat("\n========================================\n")
cat("4. 准备分析数据\n")
cat("========================================\n")

# 选择分析变量
analysis_vars <- c(treatment_var, covariate_names)
df_analysis <- df_baseline[, analysis_vars, drop = FALSE]

# 删除缺失值
df_analysis <- na.omit(df_analysis)
cat(sprintf("✓ 有效样本数: %d\n", nrow(df_analysis)))

# 重命名变量以便分析
colnames(df_analysis)[1] <- "treatment"
colnames(df_analysis)[2:length(analysis_vars)] <- paste0("X", 1:(length(analysis_vars)-1))

# 统计处理组和控制组
n_treat <- sum(df_analysis$treatment == 1)
n_control <- sum(df_analysis$treatment == 0)

cat(sprintf("\n样本分布:\n"))
cat(sprintf("  - 处理组: %d\n", n_treat))
cat(sprintf("  - 控制组: %d\n", n_control))

# ==============================================================================
# 5. PSM匹配
# ==============================================================================
cat("\n========================================\n")
cat("5. 运行PSM匹配\n")
cat("========================================\n")

# 计算卡尺 (logit倾向得分标准差的0.25倍)
# 首先运行一个初步的logit模型获取标准差
formula_init <- as.formula(paste("treatment ~", paste(names(df_analysis)[-1], collapse = " + ")))
ps_init <- glm(formula_init, data = df_analysis, family = binomial())
ps_scores_init <- ps_init$fitted.values
logit_ps_init <- log(ps_scores_init / (1 - ps_scores_init))
caliper_value <- 0.25 * sd(logit_ps_init)

cat(sprintf("卡尺设定:\n"))
cat(sprintf("  - Logit倾向得分标准差: %.4f\n", sd(logit_ps_init)))
cat(sprintf("  - 卡尺系数: 0.25\n"))
cat(sprintf("  - 卡尺值: %.4f\n", caliper_value))

# 执行PSM匹配
# 使用MatchIt包的最近邻匹配，设置卡尺
formula <- as.formula(paste("treatment ~", paste(names(df_analysis)[-1], collapse = " + ")))

set.seed(12345)  # 设置随机种子以保证结果可重复

psm_match <- matchit(
  formula = formula,
  data = df_analysis,
  method = "nearest",
  distance = "logit",
  caliper = caliper_value,
  ratio = 1,
  replace = FALSE
)

cat(sprintf("\n✓ 匹配完成\n"))

# 获取匹配结果摘要
summary_results <- summary(psm_match, standardize = TRUE)

# 打印匹配结果
cat("\n匹配摘要:\n")
print(summary_results)

# 提取匹配信息
n_matched_treat <- summary_results$nn[1, 2]  # 匹配后的处理组
n_matched_control <- summary_results$nn[2, 2]  # 匹配后的控制组
match_rate <- (n_matched_treat / n_treat) * 100

cat(sprintf("\n匹配结果:\n"))
cat(sprintf("  - 处理组（匹配后）: %d\n", n_matched_treat))
cat(sprintf("  - 控制组（匹配后）: %d\n", n_matched_control))
cat(sprintf("  - 匹配成功率: %.2f%%\n", match_rate))

# ==============================================================================
# 6. 平衡性检验
# ==============================================================================
cat("\n========================================\n")
cat("6. 平衡性检验\n")
cat("========================================\n")

# 使用cobalt包进行平衡性检验
bal_tab <- bal.tab(psm_match, stats = c("mean.diffs", "var.ratios"), un = TRUE)

cat("\n平衡性检验结果:\n")
print(bal_tab)

# 提取匹配前后的均值差异
balance_df <- data.frame(
  变量 = names(df_analysis)[-1],
  处理组均值_匹配前 = summary_results$sum.matched[, "Mean.All"],
  控制组均值_匹配前 = summary_results$sum.matched[, "Mean.Control"],
  偏差_匹配前 = summary_results$sum.matched[, "Std.Mean.Diff"],
  偏差_匹配后 = NA
)

# 获取匹配后的统计
for(i in 1:nrow(balance_df)) {
  var_name <- balance_df$变量[i]
  if(var_name %in% names(summary_results$sum.matched)) {
    balance_df$偏差_匹配后[i] <- summary_results$sum.matched[i, "Std.Mean.Diff."]
  }
}

# 计算偏差削减率
balance_df$偏差削减 <- NA
for(i in 1:nrow(balance_df)) {
  before <- abs(balance_df$偏差_匹配前[i])
  after <- abs(balance_df$偏差_匹配后[i])
  if(!is.na(before) && !is.na(after) && before > 0) {
    balance_df$偏差削减[i] <- ((before - after) / before) * 100
  }
}

# 保存平衡性检验结果
write.csv(balance_df, "基期PSM平衡性检验结果_2009.csv", row.names = FALSE, fileEncoding = "UTF-8")
cat("\n✓ 平衡性检验结果已保存: 基期PSM平衡性检验结果_2009.csv\n")

# ==============================================================================
# 7. 绘制倾向得分分布图
# ==============================================================================
cat("\n========================================\n")
cat("7. 绘制倾向得分分布图\n")
cat("========================================\n")

# 获取匹配后的数据
matched_data <- match.data(psm_match)

# 绘制倾向得分分布
png("倾向得分分布图_2009.png", width = 1400, height = 1000, res = 300)

par(mfrow = c(2, 2), mar = c(4, 4, 3, 2))

# 1. 匹配前倾向得分直方图
hist(matched_data$distance[matched_data$treatment == 1],
     breaks = 30, col = rgb(1, 0, 0, 0.5),
     main = "倾向得分分布 (匹配前)", xlab = "倾向得分",
     ylab = "频数", density = TRUE)
hist(matched_data$distance[matched_data$treatment == 0],
     breaks = 30, col = rgb(0, 0, 1, 0.5), add = TRUE)
legend("topright", legend = c("处理组", "控制组"),
       fill = c(rgb(1, 0, 0, 0.5), rgb(0, 0, 1, 0.5)))

# 2. 匹配后倾向得分直方图
hist(matched_data$distance[matched_data$treatment == 1 & matched_data$weights > 0],
     breaks = 30, col = rgb(1, 0, 0, 0.5),
     main = "倾向得分分布 (匹配后)", xlab = "倾向得分",
     ylab = "频数", density = TRUE)
hist(matched_data$distance[matched_data$treatment == 0 & matched_data$weights > 0],
     breaks = 30, col = rgb(0, 0, 1, 0.5), add = TRUE)
legend("topright", legend = c("处理组", "控制组"),
       fill = c(rgb(1, 0, 0, 0.5), rgb(0, 0, 1, 0.5)))

# 3. QQ图
qqplot(matched_data$distance[matched_data$treatment == 1],
       matched_data$distance[matched_data$treatment == 0],
       main = "倾向得分QQ图 (匹配前)",
       xlab = "处理组分位数", ylab = "控制组分位数",
       col = "blue", pch = 16)
abline(0, 1, col = "red", lty = 2)

# 4. 箱线图
boxplot(distance ~ treatment, data = matched_data,
        names = c("控制组", "处理组"),
        main = "倾向得分箱线图 (匹配前)",
        col = c("lightblue", "lightcoral"))

dev.off()
cat("✓ 图表已保存: 倾向得分分布图_2009.png\n")

# 使用cobalt绘制love plot
png("Love图_2009.png", width = 1200, height = 800, res = 300)
love_plot(psm_match, threshold = 0.1)
dev.off()
cat("✓ Love图已保存: Love图_2009.png\n")

# ==============================================================================
# 8. 导出匹配数据
# ==============================================================================
cat("\n========================================\n")
cat("8. 导出匹配数据\n")
cat("========================================\n")

# 获取匹配后的数据
matched_df <- match.data(psm_analysis)

# 添加原始变量名
matched_df_export <- cbind(
  df_baseline[matched_df$.rownames, ],
  matched_df[, c("distance", "weights", "subclass")]
)

# 保存为Excel
write.xlsx(matched_df_export, "PSM匹配数据_2009.xlsx", rowNames = FALSE)

cat(sprintf("✓ 匹配数据已导出: PSM匹配数据_2009.xlsx\n"))
cat(sprintf("  - 总样本数: %d\n", nrow(matched_df_export)))

# ==============================================================================
# 9. 生成汇总报告
# ==============================================================================
cat("\n========================================\n")
cat("PSM分析汇总报告\n")
cat("========================================\n")

report <- sprintf("
============================================
基期PSM分析报告 (2009年基期)
============================================

一、分析设置
────────────────────────────────────
  基期年份: 2009
  匹配变量:
    1. %s
    2. %s
    3. %s
    4. %s
  卡尺设定: 倾向得分对数几率标准差的0.25倍
  卡尺值: %.4f
  匹配方法: 1:1 最近邻匹配

二、样本统计
────────────────────────────────────
  基期总样本数: %d
  有效样本数: %d
  处理组: %d
  控制组: %d

三、匹配结果
────────────────────────────────────
  匹配后处理组: %d
  匹配后控制组: %d
  匹配成功率: %.2f%%

四、平衡性检验
────────────────────────────────────
  请查看: 基期PSM平衡性检验结果_2009.csv
  Love图: Love图_2009.png

============================================
分析完成！生成的文件:
  1. 基期PSM平衡性检验结果_2009.csv
  2. 倾向得分分布图_2009.png
  3. Love图_2009.png
  4. PSM匹配数据_2009.xlsx
  5. PSM分析报告_2009.txt
============================================
",
  covariate_names[1],
  covariate_names[2],
  covariate_names[3],
  covariate_names[4],
  caliper_value,
  nrow(df_baseline),
  nrow(df_analysis),
  n_treat,
  n_control,
  n_matched_treat,
  n_matched_control,
  match_rate
)

cat(report)

# 保存报告
writeLines(report, "PSM分析报告_2009.txt")
cat("✓ 报告已保存: PSM分析报告_2009.txt\n")

cat("\n========================================\n")
cat("✓ PSM分析完成！\n")
cat("========================================\n")
