# -*- coding: utf-8 -*-
import pandas as pd
import sys

# 读取数据
df = pd.read_excel("../总数据集2007-2019_含碳排放强度.xlsx")

print("数据形状:", df.shape)
print("\n所有列名:")
for i, col in enumerate(df.columns):
    print(f"{i}: {repr(col)}")

print("\n前几行数据:")
print(df.head())

print("\n数据类型:")
print(df.dtypes)
