from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd

def get_sentiments(cleansed_df) :
    sid = SentimentIntensityAnalyzer()
    cleansed_df["sentiments"] = cleansed_df["review_clean"].apply(lambda x: sid.polarity_scores(x))
    return pd.concat([cleansed_df.drop(['sentiments'], axis=1), cleansed_df['sentiments'].apply(pd.Series)], axis=1)