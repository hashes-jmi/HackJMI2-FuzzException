from flask import Flask, config, render_template, request,redirect,url_for,send_file
from dotenv import dotenv_values
from torch.cuda import init
import tweepy
import pandas as pd
import numpy as np
from predict import Pred
from torch_process import torch_pred


app = Flask(__name__)



def auth(config):
    auth = tweepy.OAuthHandler(config['CONSUMER_KEY'],config['CONSUMER_SECRET'])
    auth.set_access_token(config['ACCESS_KEY'],config['ACCESS_SECRET'])
    auth.secret = True
    global api
    api = tweepy.API(auth,wait_on_rate_limit=True)
    return api


def grab_tweets(user):
    tweetsPerQry = 100
    fName = 'tweet.txt'
    sinceId = None
    max_id = -1
    output = []
    maxTweets = 1000
    tweetCount = 0
    print("Downloading max {0} tweets",format(maxTweets))
    with open(fName,'w') as f:
        while tweetCount < maxTweets:
            try:
                if(max_id<=0):    
                    if(not sinceId):
                        new_tweets = api.user_timeline(screen_name = user,lang='en',count=tweetsPerQry, tweet_mode ='extended')
                    else:
                        new_tweets = api.user_timeline(screen_name = user,lang='en',count=tweetsPerQry,since_id=sinceId, tweet_mode ='extended')
                else: 
                    if(not sinceId):
                        new_tweets = api.user_timeline(screen_name = user,lang='en',count=tweetsPerQry,max_id = str(max_id-1), tweet_mode ='extended')
                    else:
                        new_tweets = api.user_timeline(screen_name = user,lang='en',count=tweetsPerQry,max_id = str(max_id-1),since_id=sinceId, tweet_mode ='extended')

                if not new_tweets:
                    print("No more tweets found")
                    break
                for tweet in new_tweets:
                    output.append(tweet.full_text.replace('\n','').encode("utf-8"))
                    f.write(str(tweet.full_text.replace('\n','').encode("utf-8"))+"\n")

                    tweetCount += len(new_tweets)
                    print("Dowloaded {0} tweets".format(tweetCount))
                    max_id=new_tweets[-1].id
            except tweepy.TweepError as e:
                print('some error: '+str(e))
                break
    df = pd.DataFrame(output,columns=['tweets'])
    process = Pred()
    print("Next up => cleaning")
    cleaned = process.read_tweets(df)
    #cleaned.to_csv(r'tweets.csv',index=False)
    print("Init torch")
    init_pred = torch_pred(cleaned)
    print("trying to predict")
    output = init_pred.pred()
    print(output)





@app.route('/',methods=['POST','GET'])
def home():
    if request.method=='POST':
        global name
        name = request.form['userName'].strip()
        grab_tweets(name)
        return render_template('predict.html',userName=name)
    return render_template('home.html')
    
    

@app.route('/predict')
def predict():
    return render_template('predict.html')



#later
#@app.route('/visualize')
#def visualize():
#    sns.scatterplot(tweet_polarity, #x-axis
#                tweet_subjectivity, #y-axis
#                s=100)



if __name__ == '__main__':
    config = dotenv_values('app/.env')
    auth(config)
    app.run()
