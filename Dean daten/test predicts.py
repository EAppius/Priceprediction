#Mount Google Drive to the Colab Env
#On left Menu-->Files-->download desired gdrive files
from google.colab import drive
drive.mount('/content/drive/')

DATA_COLUMN = 1
from sklearn.model_selection import train_test_split

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv("/content/drive/My Drive/IC Tech/IC Test/sp500_joined_closes_correct.csv", delimiter = ";")
#print(dataset)
#y = dataset.iloc[:,0]
#X = dataset.iloc[:,1]
#dataset = dataset[dataset[0].notnull() & dataset[DATA_COLUMN].notnull()]

#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
training_set, dataset_test = train_test_split(dataset.iloc[:,[0,DATA_COLUMN]], test_size = 0.25, shuffle=False)
print(training_set)

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(training_set)

# Creating a data structure with 60 timesteps and 1 output
X_train = []
y_train = []
for i in range(60, 231):
    X_train.append(training_set_scaled[i-60:i, 0])
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)

# Reshaping
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))



# Part 2 - Building the RNN

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

# Initialising the RNN
regressor = Sequential()

# Adding the first LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
regressor.add(Dropout(0.2))

# Adding a second LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

# Adding a third LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

# Adding a fourth LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))

# Adding the output layer
regressor.add(Dense(units = 1))

# Compiling the RNN
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

# Fitting the RNN to the Training set
regressor.fit(X_train, y_train, epochs = 50, batch_size = 32)


# Part 3 - Making the predictions and visualising the results

# Getting the real stock price of 2017

real_stock_price = dataset_test.iloc[,[,]].values
