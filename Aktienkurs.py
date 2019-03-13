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

df.to_csv('TSLA.csv')

df = pd.read_csv('tsla.csv', parse_dates=True, index_col=0)

df.plot()
plt.show()

df['Adj Close'].plot()
plt.show()

df[['High','Low']]