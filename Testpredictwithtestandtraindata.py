#https://github.com/johndehavilland/deeplearningseries/blob/master/stock_price_predictor.ipynb
#Mount Google Drive to the Colab Env
#On left Menu-->Files-->download desired gdrive files
from google.colab import drive
drive.mount('/content/drive/')
from keras import backend as K
import os
from importlib import reload
COLUMN = 'ABT'

#'/content/drive/My Drive/IC Tech/IC Test/sp500_joined_closes_correct.csv'
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from sklearn.model_selection import  train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import math
np.random.seed(7)

#load the dataset

dataset = pd.read_csv('/content/drive/My Drive/IC Tech/IC Test/sp500_joined_closes_correct.csv', delimiter = ";")

# Delete All NaN values from columns -> ['description','rate']
dataset = dataset[dataset['date'].notnull() & dataset[COLUMN].notnull()]
#dataset = dataset[dataset['date'].notnull() & dataset[COLUMN].notnull()& dataset['MMM'].notnull()]


dataset.head()
dataset.dtypes

#dataset['date'] = pd.to_datetime(dataset['date'])
#dataset[COLUMN] = pd.to_numeric(dataset[COLUMN], downcast='float')
dataset.set_index('date',inplace=True)
dataset.info()
dataset.sort_index(inplace=True)

#extract just close prices as that is what we want to predict
close = dataset[COLUMN]
close = close.values.reshape(len(close), 1)
plt.plot(close)
plt.show()

#normalize data
scaler = MinMaxScaler(feature_range=(0,1))

close = scaler.fit_transform(close)
close

#split data into train and test
train_size = int(len(close)* 0.7)
test_size = len(close) - train_size

train, test = close[0:train_size, :], close[train_size:len(close), :]

print('Split data into train and test: ', len(train), len(test))

#need to now convert the data into time series looking back over a period of days...e.g. use last 7 days to predict price

def create_ts(ds, series):
    X, Y =[], []
    for i in range(len(ds)-series - 1):
        item = ds[i:(i+series), 0]
        X.append(item)
        Y.append(ds[i+series, 0])
    return np.array(X), np.array(Y)

series = 5

trainX, trainY = create_ts(train, series)
testX, testY = create_ts(test, series)

#reshape into  LSTM format - samples, steps, features
trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
testX = np.reshape(testX, (testX.shape[0], testX.shape[1], 1))

model = Sequential()
model.add(LSTM(16, dropout=0.0, input_shape=(series, 1)))
model.add(Dense(16, activation='tanh'))#test sigmoid or tanh instead of relu
model.add(Dense(1))
model.add(Activation('relu'))#uses sigmoid because binary classifier
model.summary()

"""
#build the model
model = Sequential()
model.add(LSTM(32, input_shape=(series, 1)))
model.add(Dense(1))"""
model.compile(loss='mse', optimizer='adam')
#fit the model
model.fit(trainX, trainY, epochs=100, batch_size=32)#, validation_data = (testX, trainY))
