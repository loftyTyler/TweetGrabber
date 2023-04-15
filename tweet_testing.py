from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
import snscrape.modules.twitter as sntwitter
import pandas as pd
from matplotlib import pyplot as plt

#Getting userinput for dates
userInput = input("Which economy sentiment do you want to see based on the following events.  Select the number."
                  "\n1.Banks Collapsing such as SVB\n2.Russia and China ditching USD for trade\n3.Release of Unemployment Rate in April\n")
if userInput == "1":
    query = "us economy (gdp OR jobs OR united OR states) (#us OR #unitedstates OR #economy) lang:en until:2023-03-017 since:2023-03-07"
    eventLabel = "Collaping Tech Banks \nNews on March 10th, data 5 days before and after"
elif userInput == "2":
    query = "us economy (gdp OR jobs OR united OR states) (#us OR #unitedstates OR #economy) lang:en until:2023-04-08 since:2023-03-029"
    eventLabel = "Russia and China ditching USD\nNews on April 3rd, data 5 days before and after"
else: 
    query = "us economy (gdp OR jobs OR united OR states) (#us OR #unitedstates OR #economy) lang:en until:2023-03-09 since:2023-03-02"
    eventLabel = "Unemployement Rate in April\nNews on April 7th, data 5 days before and after"

tweets = []
limit = 50
tweetsDf = []
processedTweet = []
scoresPolarity = []

#This code below will go through a search query of the keyword and grab each one
#if its more than the limit the loop will finish
#if its not at the limit then it will put the tweet within the tweets array
for tweet in sntwitter.TwitterSearchScraper(query).get_items():
    if len(tweets) == limit:
        break
    else:
        tweets.append(tweet.rawContent)
        tweetsDf.append([tweet.date, tweet.rawContent])

df = pd.DataFrame(tweetsDf, columns = ['Date', 'Tweet'])

# Preprocessing tweets and performing sentiment analysis
roberta = "cardiffnlp/twitter-roberta-base-sentiment"
model = AutoModelForSequenceClassification.from_pretrained(roberta)
tokenizer = AutoTokenizer.from_pretrained(roberta) #Load to convert tweet takes into appropriate numbers
labels = ['Negative', 'Neutral', 'Positive']


#Going through each tweet to preprocess it
for tweet in tweets:                                #Itterating through each element in tweets array
    tweet_words = []
    for word in tweet.split(' '):                   #Each word within each element of the tweets array will be split
        if word.startswith('@') and len(word) > 1:  
            word = '@user'
        elif word.startswith('http'):
            word = 'http'
        tweet_words.append(word)                    #Each word will within the tweet will be put into a new array called tweet_words
    tweet_proc = " ".join(tweet_words)
    processedTweet.append(tweet_proc)
    
    #Running preprocessed tweets trhough model
    encoded_tweet = tokenizer(tweet_proc, return_tensors='pt') #Convert to pytorch tensors
                                                               #Will map to a dictionary and convert tweet to numbers 
    output = model(**encoded_tweet)                            #Passing tweet through roberta model
    scores = output[0][0].detach().numpy()                     #Selects specifically the probility list
    scores = softmax(scores)                                   #Will convert numbers to between 0 and 1 

    #Subtracing the negative score from the positive to get its Polarity
    polarity = scores[2] - scores[0]
    scoresPolarity.append(polarity)


#Create new dataframe with only the graphable sections
df['Processed Tweet'] = processedTweet
df['Polarity'] = scoresPolarity 
graph = df[['Date', 'Polarity']].copy()
graph = df.sort_values(by="Date", ascending=True)
graph['MA Polarity'] = graph.Polarity.rolling(10, min_periods=3).mean()


#Good Graph Code
fig, axes = plt.subplots(figsize=(13,10))
axes.plot(graph['Date'], graph['MA Polarity'])
axes.set_title("\n".join(['Average Polarity']))
fig.suptitle('\n'.join(['Sentiment Analysis for ' + eventLabel]), y=0.98)
plt.show()

#Showing tweet data if desired
showData = input("If you wish to see tweets and thier coresponding polarity enter Y, or if not N\n")
if showData == 'Y':
    print(df[['Processed Tweet', 'Polarity']])

