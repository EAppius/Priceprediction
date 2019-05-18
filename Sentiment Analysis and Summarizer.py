import pandas as pd
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from gensim.summarization.summarizer import summarize



##### Dateipfad angeben, wo mit GitKraken die GitHub Repo lokal gespeichert wurde ######
#für Dean's Computer
DATEIPFAD = "C:/Users/lea_m/Dropbox/Documents/MBI 3/IC Tech und Market Intelligence/Short"


###### need to select the desired file here ######
# filename for Mattel news for negative 02.11.17
filename = '{}/Data/Manual News Data/MAT_news.csv'.format(DATEIPFAD)
save_name = "MAT"

# filename for Mattel news for positive 11.11.17
##filename = '{}/Data/Manual News Data/MAT2_news.csv'.format(DATEIPFAD)
##save_name = "MAT2"

# Filename for Loews news
##filename = '{}/Data/Manual News Data/LOEWS_news.csv'.format(DATEIPFAD)
##save_name = "LOEWS"

# read data from file name
data = pd.read_csv(filename, sep=";", encoding="Latin-1")
data.apply(np.random.permutation, axis=1)
data.head()

# prepare VadeSentimentanalyzer
analyser = SentimentIntensityAnalyzer()


# function for sentiment analysis that returns the sentiment score
def sentiment_analyzer_scores(sentence):
  score = analyser.polarity_scores(sentence)
  # print("{:-<5} {}".format(sentence, str(score)))
  return score


# create summary and score for each text
summaries = []
scores = []
for i in data.Text:
  # summary of the text
  try:
    summary = summarize(i)
    summaries.append(summary)
  except:
    print("an exception occured: e.g. too few sentences to summarize")
  # score of the full text
  score = sentiment_analyzer_scores(i)
  scores.append(score)

print(summaries)
print(scores)



  #save CSV with predictions for plotting in Tableau
with open('{}/Data/Manual News Data/Sentiment scores/{}.csv'.format(DATEIPFAD, save_name), mode='w') as file:
          file_writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
          file_writer.writerow(scores)
          text_lst = []
          for i in data.Title:
            text_lst.append(str(i))
          file_writer.writerow([text_lst])

