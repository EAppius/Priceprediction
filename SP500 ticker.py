import pandas as pd
data = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')


table = data[0]
table.head(520)

sliced_table = table[1:]
header = table.iloc[0]

corrected_table = sliced_table.rename(columns=header)
corrected_table

tickers = corrected_table['Symbol'].tolist()
print (tickers)

import csv
with open(r'C:\Users\flavio.hartmann\Desktop\EY Laptop Priavte Data\Studium\Master\UNISG\Unterlagen\2019-02 FS\IC Technology und Market Intelligence\Gruppenarbeit\Webcrawling\SP500.csv', 'w', newline='') as csvFile:
    w = csv.writer(csvFile)
    for item in tickers:
        w.writerow([item])

csvFile.close()
