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
