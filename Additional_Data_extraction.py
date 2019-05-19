import bs4 as bs
import datetime as dt
import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
import csv
import os
import pandas as pd
import pandas_datareader.data as web
import pickle
import quandl
from statistics import mean
import requests
from collections import Counter
from sklearn import svm, model_selection, neighbors
from sklearn.ensemble import VotingClassifier, RandomForestClassifier


##### Dateipfad angeben, wo mit GitKraken die GitHub Repo lokal gespeichert wurde ######
#für Dean's Computer
DATEIPFAD = "C:/Users/lea_m/Dropbox/Documents/MBI 3/IC Tech und Market Intelligence/Short"

##### TICKER FÜR GEWÜNSCHTE INFORMATIONEN ANGEBEN (Z.B. GOLD, DOLLAR, ÖL)######
lst = ['WGC/GOLD_DAILY_USD', 'EIA/PET_RWTC_D', 'FED/RXI_US_N_B_EU', 'BCIW/_INX']
#API Configuration für QUANDL
api_key = "4SKUTzkXLJmR-z9sPq5z"  # QUANDL API key

quandl.ApiConfig.api_key = api_key

#Funktion um die Daten der einzelnen Ticker aus Quandl zu lesen und jeweils als CSV speichern
def get_data_from_quandl(lst):
    tickers = lst
    print(tickers)

    start = "2017-01-01"  # dt.datetime(2010, 1, 1)
    end = "2019-01-01"  # dt.datetime.now()

    counter = 0

    for ticker in tickers:
        # just in case your connection breaks, we'd like to save our progress!
        counter += 1
        save_name = ticker.replace("/","")
        if not os.path.exists('{}/Data/Historical data/additional data/{}.csv'.format(DATEIPFAD, save_name)):

            df = quandl.get(ticker, date = { 'gte': start, 'lte': end })

            df.to_csv('{}/Data/Historical data/additional data/{}.csv'.format(DATEIPFAD, save_name))

        else:
            print('Already have {}'.format(save_name))



#Die einzelnen Funktionen Abrufen
get_data_from_quandl(lst)
