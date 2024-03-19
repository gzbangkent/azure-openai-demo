import pywencai

res = pywencai.get(query='周涨跌幅 云计算行业', loop=True)
res.to_csv('stock.csv', index=False)