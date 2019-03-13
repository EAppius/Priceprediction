import datetime as dt
import matplotlib.pyplot as plt
from matplotlib import style
import pandas as pd
import pandas_datareader.data as web

style.use('ggplot')

start = dt.datetime(2017,1,1)
end = dt.datetime(2017,12,31)

df = web.DataReader('SIN', 'yahoo', start, end)
print(df.head(30))


print("test edit")
