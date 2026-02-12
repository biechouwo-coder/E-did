import pandas as pd

# 读取数据
df = pd.read_excel("../总数据集2007-2019_含碳排放强度.xlsx")

print("数据形状:", df.shape)
print("\n列名:")
print(df.columns.tolist())
print("\n前几行数据:")
print(df.head(10))
print("\n数据类型:")
print(df.dtypes)
print("\n年份范围:")
print("年份范围:", df['年份'].min(), "-", df['年份'].max())
print("\n各年份样本数:")
print(df['年份'].value_counts().sort_index())
