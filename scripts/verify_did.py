# -*- coding: utf-8 -*-
import pandas as pd
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

df = pd.read_excel('总数据集_2007-2023_完整版_DID.xlsx')

cities = ['保定市', '杭州市', '北京市', '上海市', '深圳市']

print('不同批次城市的DID验证:\n')

for city in cities:
    city_data = df[df['city_name']==city].sort_values('year')
    if len(city_data) > 0:
        if (city_data['DID']==1).any():
            start_year = int(city_data[city_data['DID']==1]['year'].min())
        else:
            start_year = '无'

        print(f'{city}: 首次DID=1的年份 = {start_year}')

        early_years = city_data[city_data['year'].isin([2007,2008,2009,2010])]['DID'].tolist()
        late_years = city_data[city_data['year'].isin([2017,2018,2019,2020])]['DID'].tolist()

        print(f'  2007-2010年DID: {early_years}')
        print(f'  2017-2020年DID: {late_years}')
        print()
