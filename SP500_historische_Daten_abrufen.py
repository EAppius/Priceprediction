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




#Funktion mit welcher aus Wikipedia die liste der S&P500 Ticker gelesen werden und in ein CSV gespeichert wird
def save_sp500_tickers():
    resp = requests.get('http://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    soup = bs.BeautifulSoup(resp.text, 'lxml')
    table = soup.find('table', {'class': 'wikitable sortable'})
    tickers = []
    counter = 0
    for row in table.findAll('tr')[1:]:
        ticker = row.findAll('td')[0].text
        ticker = ticker.replace("\n", "")
        tickers.append(ticker)
    with open("sp500tickers.pickle", "wb") as f:
        pickle.dump(tickers, f)
    with open(r'{}/Data/S&P 500 lists/SP500.csv'.format(DATEIPFAD), 'w', newline='') as csvFile:
        w = csv.writer(csvFile)
        for item in tickers:
            w.writerow([item])
    return tickers
    csvFile.close()

"""
#Alternativer Webscraper um die S&P 500 Liste von Wikipedia zu lesen, welche wir jedoch doch noch ersetzt haben
def save_sp500_tickers2():
    data = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')

    table = data[0]
    table.head(520)

    sliced_table = table[1:]
    header = table.iloc[0]

    corrected_table = sliced_table.rename(columns=header)

    tickers = corrected_table['Symbol'].tolist()

    with open("sp500tickers.pickle", "wb") as f:
        pickle.dump(tickers, f)
    print(tickers)
    with open(r'{}/Data/S&P 500 lists/SP500.csv'.format(DATEIPFAD), 'w', newline='') as csvFile:
        w = csv.writer(csvFile)
        for item in tickers:
            w.writerow([item])
    return tickers
    csvFile.close()
"""


#API Configuration für QUANDL
api_key = "4SKUTzkXLJmR-z9sPq5z"  # QUANDL API key

quandl.ApiConfig.api_key = api_key

#Funktion um die Daten der einzelnen Ticker aus Quandl zu lesen und jeweils als CSV speichern
def get_data_from_quandl(reload_sp500=True):
    if reload_sp500:
        tickers = save_sp500_tickers()
    else:
        with open("sp500tickers.pickle", "rb") as f:
            tickers = pickle.load(f)
    if not os.path.exists('stock_dfs'):
        os.makedirs('stock_dfs')
    tickers = save_sp500_tickers()

    start = "2017-01-01"  # dt.datetime(2010, 1, 1)
    end = "2019-01-01"  # dt.datetime.now()

    for ticker in tickers:
        # just in case your connection breaks, we'd like to save our progress!
        if not os.path.exists('{}/Data/Historical data/by_ticker/{}.csv'.format(DATEIPFAD, ticker)):

            df = quandl.get_table('WIKI/PRICES', qopts={'columns': ['ticker', 'date', 'close', 'adj_close']},
                                  ticker=[ticker], date={'gte': start, 'lte': end})

            df.to_csv('{}/Data/Historical data/by_ticker/{}.csv'.format(DATEIPFAD, ticker))

        else:
            print('Already have {}'.format(ticker))




#Funktion um die einzelnen CSV Dokumente in ein Dokument zusammenzubringen
def compile_data():
    with open("sp500tickers.pickle", "rb") as f:
        tickers = pickle.load(f)

    main_df = pd.DataFrame()

    for count, ticker in enumerate(tickers):
        df = pd.read_csv('{}/Data/Historical data/by_ticker/{}.csv'.format(DATEIPFAD, ticker))
        df.set_index('date', inplace=True)

        df.rename(columns={'adj_close': ticker}, inplace=True)
        df.drop(['close', 'ticker', 'None'], 1, inplace=True)

        if main_df.empty:
            main_df = df
        else:
            main_df = main_df.join(df, how='outer')

        if count % 10 == 0:
            print(count)
    print(main_df.head())
    main_df.to_csv('{}/Data/Historical data/sp500_joined_closes.csv'.format(DATEIPFAD))




#Funktion um die historischen Daten zu visualisieren um die Correlation zu sehen
def visualize_data():
    df = pd.read_csv('{}/Data/Historical data/sp500_joined_closes.csv'.format(DATEIPFAD))
    df_corr = df.corr()
    print(df_corr.head())
    df_corr.to_csv('{}/Data/Historical data/sp500corr.csv'.format(DATEIPFAD))
    data1 = df_corr.values
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)

    heatmap1 = ax1.pcolor(data1, cmap=plt.cm.RdYlGn)
    fig1.colorbar(heatmap1)

    ax1.set_xticks(np.arange(data1.shape[1]) + 0.5, minor=False)
    ax1.set_yticks(np.arange(data1.shape[0]) + 0.5, minor=False)
    ax1.invert_yaxis()
    ax1.xaxis.tick_top()
    column_labels = df_corr.columns
    row_labels = df_corr.index
    ax1.set_xticklabels(column_labels)
    ax1.set_yticklabels(row_labels)
    plt.xticks(rotation=90)
    heatmap1.set_clim(-1, 1)
    plt.tight_layout()
    plt.show()


#Die einzelnen Funktionen Abrufen
save_sp500_tickers()
get_data_from_quandl()
compile_data()
visualize_data()










############# Dies war eine alternativer Code für die Vorhersage ###########
"""
#Funktion um die Grafik anzuschreiben
def process_data_for_labels(ticker):
    hm_days = 7
    df = pd.read_csv('{}/Data/Historical data/sp500_joined_closes.csv'.format(DATEIPFAD), index_col=0)
    tickers = df.columns.values.tolist()
    df.fillna(0, inplace=True)
    for i in range(1, hm_days + 1):
        df['{}_{}d'.format(ticker, i)] = (df[ticker].shift(-i) - df[ticker]) / df[ticker]
    df.fillna(0, inplace=True)
    return tickers, df

#auswertung ob buy, sell or hold für diesen Ticker
def buy_sell_hold(*args):
    cols = [c for c in args]
    requirement = 0.02
    for col in cols:
        if col > requirement:
            return 1
        if col < -requirement:
            return -1
    return 0

#Vorbereitung der Daten für das Machine Learning
def extract_featuresets(ticker):
    tickers, df = process_data_for_labels(ticker)

    df['{}_target'.format(ticker)] = list(map( buy_sell_hold,
                                               df['{}_1d'.format(ticker)],
                                               df['{}_2d'.format(ticker)],
                                               df['{}_3d'.format(ticker)],
                                               df['{}_4d'.format(ticker)],
                                               df['{}_5d'.format(ticker)],
                                               df['{}_6d'.format(ticker)],
                                               df['{}_7d'.format(ticker)]))

    vals = df['{}_target'.format(ticker)].values.tolist()
    str_vals = [str(i) for i in vals]
    print('Data spread:', Counter(str_vals))

    df.fillna(0, inplace=True)
    df = df.replace([np.inf, -np.inf], np.nan)
    df.dropna(inplace=True)

    df_vals = df[[ticker for ticker in tickers]].pct_change()
    df_vals = df_vals.replace([np.inf, -np.inf], 0)
    df_vals.fillna(0, inplace=True)

    X = df_vals.values
    y = df['{}_target'.format(ticker)].values
    return X, y, df


#Ausführung des Machine Learning mit SVM.Linear, KNeighbors & RandomForest
def do_ml(ticker):
    X, y, df = extract_featuresets(ticker)

    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.25)

    clf = VotingClassifier([('lsvc', svm.LinearSVC()),
                            ('knn', neighbors.KNeighborsClassifier()),
                            ('rfor', RandomForestClassifier())])

    clf.fit(X_train, y_train)
    confidence = clf.score(X_test, y_test)
    print('accuracy:', confidence)
    predictions = clf.predict(X_test)
    print('predicted class counts:', Counter(predictions))
    print()
    print()
    return confidence



#Ausführen des Machine Learning codes zur Vorhersage
with open("sp500tickers.pickle","rb") as f:
    tickers = pickle.load(f)

accuracies = []
for count,ticker in enumerate(tickers):

    if count%10==0:
        print(count)

    accuracy = do_ml(ticker)
    accuracies.append(accuracy)
    print("{} accuracy: {}. Average accuracy:{}".format(ticker,accuracy,mean(accuracies)))

"""

