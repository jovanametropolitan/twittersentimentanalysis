import re # for regular expressions
import nltk # for text manipulation
import string
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import gensim

pd.set_option("display.max_colwidth", 200)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# reading the train and test csv file

train = pd.read_csv('tweets.csv')
test = pd.read_csv('tweets.csv')
# checking the tweets with non racist tweets with label 0

train[train['label'] == 0].head(10)
# checking the tweets with racist tweets with label 1

train[train['label'] == 1].head(10)

# checking the distribution of length of the tweets, in terms of words, in both train and test data.

length_train = train['tweet'].str.len()
length_test = test['tweet'].str.len()
plt.hist(length_train, bins=20, label="train_tweets")
plt.hist(length_test, bins=20, label="test_tweets")
plt.legend()
#plt.show()

print(test)
combi = train.append(test, ignore_index=True)
combi.shape


# function for removing patterns

def remove_pattern(input_txt, pattern):
    r = re.findall(pattern, input_txt)
    for i in r:
        input_txt = re.sub(i, '', input_txt)
    return input_txt



# removing @user handle

combi['tidy_tweet'] = np.vectorize(remove_pattern)(combi['tweet'], "@[\w]*")
combi.head()


# removing punctuation, numbers and special caharacters except small and large alphabets and #

combi['tidy_tweet'] = combi['tidy_tweet'].str.replace("[^a-zA-Z#]", " ")
combi.head(10)


# removing short words having length less than 3

combi['tidy_tweet'] = combi['tidy_tweet'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>3]))
combi.head()


# Text Normalization
# Tokens are individual terms or words, and tokenization is the process of splitting a string of text into tokens.

tokenized_tweet = combi['tidy_tweet'].apply(lambda x: x.split()) # tokenizing
tokenized_tweet.head()


# Now we can normalize the tokenized tweets.

from nltk.stem.porter import *
stemmer = PorterStemmer()
tokenized_tweet = tokenized_tweet.apply(lambda x: [stemmer.stem(i) for i in x]) # stemming



# Now let’s stitch these tokens back together. It can easily be done using nltk’s MosesDetokenizer function.

for i in range(len(tokenized_tweet)):
    tokenized_tweet[i] = ' '.join(tokenized_tweet[i])
combi['tidy_tweet'] = tokenized_tweet

combi['tidy_tweet'].head()

#wordcloud for all tweets
#all_words = ' '.join([text for text in combi['tidy_tweet']])
from wordcloud import WordCloud
#wordcloud = WordCloud(width=1100, height=700, random_state=1, max_font_size=120).generate(all_words)

#plt.figure(figsize=(10, 7))
#plt.imshow(wordcloud, interpolation="bilinear")
#plt.axis('off')
#plt.show()

# wordcloud for positive tweets
normal_words =' '.join([text for text in combi['tidy_tweet'][combi['label'] == 0]])
print(normal_words)
positive_dict = {}

x = normal_words.split()

for word in x:
	if word not in positive_dict:
		positive_dict[word] = 1
	else:
		positive_dict[word] += 1
		
#sorted_values = sorted(positive_dict.values()) # Sort the values
sorted_dict = {}

#for i in sorted_values:
#    for k in positive_dict.keys():
#        if positive_dict[k] == i:
#            sorted_dict[k] = positive_dict[k]
#            break

#print("\n")
#print("\n")
#print("POSITIVE")
#print(sorted_dict)

wordcloud = WordCloud(background_color='white', width=1100, height=700, random_state=1, max_font_size=120).generate(normal_words)
plt.figure(figsize=(10, 7))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.show()
#plt.show()

# wordcloud for negative tweets

negative_words = ' '.join([text for text in combi['tidy_tweet'][combi['label'] == 1]])
#print(negative_words)
wordcloud = WordCloud(background_color='white', width=1100, height=700, random_state=1, max_font_size=120).generate(negative_words)
plt.figure(figsize=(10, 7))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.show()

#positive_dict = {}

x = negative_words.split()

for word in x:
	if word not in positive_dict:
		positive_dict[word] = 1
	else:
		positive_dict[word] += 1
		
#sorted_values = sorted(positive_dict.values()) # Sort the values
#sorted_dict = {}

#for i in sorted_values:
    #for k in positive_dict.keys():
     #   if positive_dict[k] == i:
      #      sorted_dict[k] = positive_dict[k]
       #     break


#print("\n")
#print("\n")
#print("NEGATIVE")
#print(sorted_dict)

# wordcloud for neutral tweets

negative_words = ' '.join([text for text in combi['tidy_tweet'][combi['label'] == -1]])
#print(negative_words)
wordcloud = WordCloud(background_color='white', width=1100, height=700,random_state=1, max_font_size=120).generate(negative_words)
plt.figure(figsize=(10, 7))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.show()

#positive_dict = {}

x = negative_words.split()

for word in x:
	if word not in positive_dict:
		positive_dict[word] = 1
	else:
		positive_dict[word] += 1
		
sorted_values = sorted(positive_dict.values()) # Sort the values
sorted_dict = {}

for i in sorted_values:
    for k in positive_dict.keys():
        if positive_dict[k] == i:
            sorted_dict[k] = positive_dict[k]
            break


print("\n")
print("\n")
print("NEUTRAL")
print(sorted_dict)
# function to collect hashtags

def hashtag_extract(x):
    hashtags = []
    # Loop over the words in the tweet
    for i in x:
        ht = re.findall(r"#(\w+)", i)
        hashtags.append(ht)

    return hashtags

# extracting hashtags from non racist/sexist tweets

HT_regular = hashtag_extract(combi['tidy_tweet'][combi['label'] == 0])

# extracting hashtags from racist/sexist tweets

HT_negative = hashtag_extract(combi['tidy_tweet'][combi['label'] == 1])

# unnesting list

HT_regular = sum(HT_regular,[])
HT_negative = sum(HT_negative,[])

# plotting positive tweets hashtags

a = nltk.FreqDist(HT_regular)
d = pd.DataFrame({'Hashtag': list(a.keys()),
                  'Count': list(a.values())})
# selecting top 10 most frequent hashtags     
d = d.nlargest(columns="Count", n = 10) 
plt.figure(figsize=(16,5))
ax = sns.barplot(data=d, x= "Hashtag", y = "Count")
ax.set(ylabel = 'Count')
#plt.show()

# plotting negative tweets hashtags

b = nltk.FreqDist(HT_negative)
e = pd.DataFrame({'Hashtag': list(b.keys()), 'Count': list(b.values())})
# selecting top 10 most frequent hashtags
e = e.nlargest(columns="Count", n = 10)   
plt.figure(figsize=(16,5))
ax = sns.barplot(data=e, x= "Hashtag", y = "Count")
ax.set(ylabel = 'Count')
#plt.show()

# bag of words features

from sklearn.feature_extraction.text import CountVectorizer 
bow_vectorizer = CountVectorizer(max_df=0.90, min_df=2, max_features=1000, stop_words='english')
# bag-of-words feature matrix
bow = bow_vectorizer.fit_transform(combi['tidy_tweet'])

from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer(max_df=0.90, min_df=2, max_features=1000, stop_words='english')
# TF-IDF feature matrix
tfidf = tfidf_vectorizer.fit_transform(combi['tidy_tweet'])

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

train_bow = bow[:31962,:]
test_bow = bow[31962:,:]

# splitting data into training and validation set
xtrain_bow, xvalid_bow, ytrain, yvalid = train_test_split(train_bow, train['label'], random_state=42, test_size=0.3)

lreg = LogisticRegression()
lreg.fit(xtrain_bow, ytrain) # training the model

prediction = lreg.predict_proba(xvalid_bow) # predicting on the validation set
prediction_int = prediction[:,1] >= 0.3 # if prediction is greater than or equal to 0.3 than 1 else 0
prediction_int = prediction_int.astype(np.int)

f1_score(yvalid, prediction_int) # calculating f1 score

test_pred = lreg.predict_proba(test_bow)
test_pred_int = test_pred[:,1] >= 0.3
test_pred_int = test_pred_int.astype(np.int)
test['label'] = test_pred_int
submission = test[['id','label']]
submission.to_csv('sub_lreg_bow.csv', index=False) # writing data to a CSV file

train_tfidf = tfidf[:31962,:]
test_tfidf = tfidf[31962:,:]

xtrain_tfidf = train_tfidf[ytrain.index]
xvalid_tfidf = train_tfidf[yvalid.index]

lreg.fit(xtrain_tfidf, ytrain)

prediction = lreg.predict_proba(xvalid_tfidf)
prediction_int = prediction[:,1] >= 0.3
prediction_int = prediction_int.astype(np.int)

f1_score(yvalid, prediction_int)
