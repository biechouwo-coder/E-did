# -*- coding: utf-8 -*-
import pandas as pd

# 读取PSM匹配后的数据
df_matched = pd.read_excel("../PSM-四个变量-2009基期/匹配后数据集_四个变量.xlsx")

print("匹配后数据形状:", df_matched.shape)
print("\n列名:")
for i, col in enumerate(df_matched.columns):
    print(f"{i}: {repr(col)}")

print("\n前几行:")
print(df_matched.head())
