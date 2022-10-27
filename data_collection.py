from textblob import TextBlob
import sys
import tweepy
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import nltk
import pycountry
import re
import string
import time
import datetime
import csv
import mysql.connector
from wordcloud import WordCloud, STOPWORDS
from PIL import Image
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from langdetect import detect
from nltk.stem import SnowballStemmer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import CountVectorizer

#DB configuration
mydb = mysql.connector.connect(
  host="localhost",
  user="root",
  password="",
  database="tweets"
)

mycursor = mydb.cursor()

while True:
	# Authentication
	consumerKey = ""
	consumerSecret = ""
	accessToken = ""
	accessTokenSecret = ""
	auth = tweepy.OAuthHandler(consumerKey, consumerSecret)
	auth.set_access_token(accessToken, accessTokenSecret)
	api = tweepy.API(auth)

	#Sentiment Analysis
	def percentage(part,whole):
	 return 100 * float(part)/float(whole)
	#keyword = input("Please enter keyword or hashtag to search: ")
	#noOfTweet = int(input ("Please enter how many tweets to analyze: "))
	keyword = 'oxford street'
	noOfTweet = 1300
	tweets = tweepy.Cursor(api.search_tweets, q=keyword + ' -filter:retweets', lang="en", tweet_mode="extended").items(noOfTweet)
	positive = 0
	negative = 0
	neutral = 0
	polarity = 0
	tweet_list = []
	neutral_list = []
	negative_list = []
	positive_list = []


	for tweet in tweets:
		#print(tweet)
		#print("\n")
		if (not tweet.retweeted) and ('RT @' not in tweet.full_text):
			#with open('all.txt', 'ab') as the_file:
				#hashtags  = ','.join(str(v) for v in tweet.entities.get('hashtags'))
				#line = tweet.full_text + " - " + hashtags +"\n\n"
				#the_file.write(line.encode("utf-8"))
			user_tweet = str(tweet.user.screen_name);
			coordinates = tweet.coordinates;
			if (coordinates):
				coordinates = str(tweet.coordinates['coordinates']);
				
			hashtags  = ','.join(str(v.get('text')) for v in tweet.entities.get('hashtags'))
			
			place = tweet.place;
			if (place):
				place = tweet.place.full_name;
			
			tweet_list.append(tweet.full_text)
			analysis = TextBlob(tweet.full_text)
			score = SentimentIntensityAnalyzer().polarity_scores(tweet.full_text)
			neg = score['neg']
			neu = score['neu']  
			pos = score['pos']
			comp = score['compound']
			polarity += analysis.sentiment.polarity
			type = ""
			if neg > pos:
				negative_list.append(tweet.full_text)
				negative += 1
				type = "NEGATIVE"
			elif pos > neg:
				positive_list.append(tweet.full_text)
				positive += 1
				type = "POSITIVE"		 
			elif pos == neg:
				neutral_list.append(tweet.full_text)
				neutral += 1
				type = "NEUTRAL"
				
			sql = "INSERT INTO tweets (created_at, id_str, full_text, hashtags, user_tweet, coordinates, place, lang, type) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)"
			val = (tweet.created_at, tweet.id_str, tweet.full_text, hashtags, user_tweet, coordinates, str(), tweet.lang, type)
			try:
				mycursor.execute(sql, val)
				mydb.commit()
			except:
				print("Duplicate tweet")
			
	positive = percentage(positive, noOfTweet)
	negative = percentage(negative, noOfTweet)
	neutral = percentage(neutral, noOfTweet)
	polarity = percentage(polarity, noOfTweet)
	positive = format(positive, '.1f')
	negative = format(negative, '.1f')
	neutral = format(neutral, '.1f')

	#for n in negative_list:
	#	with open('negative.txt', 'ab') as the_file:
	#		line = n +"\n\n"
	#		the_file.write(line.encode("utf-8"))
			
			
	#for n in positive_list:
	#	with open('positive.txt', 'ab') as the_file:
	#		line = n +"\n\n"
	#		the_file.write(line.encode("utf-8"))
			
			
	#for n in neutral_list:
	#	with open('neutral.txt', 'ab') as the_file:
	#		line = n +"\n\n"
	#		the_file.write(line.encode("utf-8"))


	#Number of Tweets (Total, Positive, Negative, Neutral)
	tweet_list = pd.DataFrame(tweet_list)
	neutral_list = pd.DataFrame(neutral_list)
	negative_list = pd.DataFrame(negative_list)
	positive_list = pd.DataFrame(positive_list)

	print("total number: ",len(tweet_list))
	print("positive number: ",len(positive_list))
	print("negative number: ", len(negative_list))
	print("neutral number: ",len(neutral_list))
	time.sleep(1000)