import re
import tweepy
import pandas as pd
import numpy as np
import string 
from textblob import TextBlob




class Pred:
    def __init__(self):
        pass

    def clean(self,text):
        text  = "".join([char for char in text if char not in string.punctuation])
        text = re.sub(r"http\S+", "", text)
        text = re.sub(r"www.\S+", "", text)
        text = re.sub(r'RT[\s]+','',text)
        text = re.sub('[0-9]+', '', text)
        return text

    def read_tweets(self,df):
        df['cleaned_tweets'] = df['tweets'].apply(lambda x: self.clean(str(x)))
        return df

    def poles(self,tweets):
        polarity=lambda x:TextBlob(x).sentiment.polarity
        subjectivity = lambda x:TextBlob(x).sentiment.subjectivity

        tweet_polarity = np.zeros(len(tweets))
        tweet_subjectivity = np.zeros(len(tweets))

        for idx, tweet in enumerate(tweets):
            tweet_polarity[idx] = polarity(tweet)
            tweet_subjectivity[idx] = subjectivity(tweet)

        return (tweet_polarity,tweet_subjectivity)


    def graph_poles(self,tweet_polarity,tweet_subjectivity):
        fig,ax = plt.subplots(figsize=(6,6))
        ax = sns.scatterplot(tweet_polarity, #x-axis
                tweet_subjectivity, #y-axis
                s=100)


