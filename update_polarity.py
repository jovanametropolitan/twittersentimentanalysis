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

mydb = mysql.connector.connect(
  host="localhost",
  user="root",
  password="",
  database="tweets"
)

mycursor = mydb.cursor()

mycursor.execute("SELECT * FROM tweets")

myresult = mycursor.fetchall()

for tweet in myresult:
	print(tweet[3])
	score = SentimentIntensityAnalyzer().polarity_scores(tweet[3])
	neg = score['neg']
	neu = score['neu']  
	pos = score['pos']
	comp = score['compound']
	sql = "UPDATE tweets SET score_negative = %s, score_positive = %s, score_neutral = %s, compound = %s where id_str = %s"
	val = (neg, pos, neu, comp, tweet[2])
	try:
		mycursor.execute(sql, val)
		mydb.commit()
	except:
		print("Invalid update")
