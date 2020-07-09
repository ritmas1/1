#downloading sample tweets:
import nltk
nltk.download('twitter_samples')
from nltk.corpus import twitter_samples
tweets = twitter_samples.docs()
docs= [t['text'] for t in tweets]

#initial number of tweets
len(docs)

#removing duplicate twitter lines:
docs_new=[]
for i in docs:
       if i not in docs_new:
            docs_new.append(i)
docs=docs_new

#number of tweets after removing duplicates:
len(docs)

# regex operations 
# regex operations performed on strings
# creating a string for string manipulations
import re
docs_new=[]
for tweet in docs:
    tweet = re.sub(r'\:{1}\(+', 'sad', tweet)
    tweet = re.sub(r'\:{1}\(+', 'sad', tweet)
    tweet = re.sub(r'\:{1}\-\(+', 'sad', tweet)
    tweet = re.sub(r'\:{1}\)+', 'happy ', tweet)
    tweet = re.sub(r'\:{1}\-\)+', 'happy ', tweet)
    tweet = re.sub(r'\:{1}\-?\*', 'kiss', tweet)
    tweet = re.sub(r'@[a-zA-Z0-9_]+', ' ', tweet) #removing user names
    tweet = re.sub(r'https?://t.co/[a-zA-Z0-9./]+', ' ', tweet) #removing links
    tweet = re.sub(r'[\U00010000-\U0010ffff]', ' ', tweet) #remocing emoji
    tweet = re.sub(r'#', '  ', tweet)  #removing only the hashtag symbol, since the word following the symbol might have a semantic meaning                   
    tweet = tweet.lower()
    tweet = re.sub(r'\d[th|rd|st)?]', ' ', tweet)
    tweet = re.sub(r'\d+', ' ', tweet)
    tweet = re.sub(r'\W', ' ', tweet)
    tweet = tweet.replace("  "," ") #removing extra spaces
    tweet = tweet.replace("   "," ") #removing extra spaces
    tweet = tweet.replace("    "," ") #removing extra spaces
    tweet = tweet.lstrip() #removing spaces at the begging and the end of the sentence
    docs_new.append(tweet)
docs=docs_new

# tokenization
from nltk.tokenize import word_tokenize

docs_tokenized=[]
for tweet in docs:
    tokens = word_tokenize(tweet)
    docs_tokenized.append(tokens)
print(docs_tokenized[0])

# removing  stop words
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
nltk.download('stopwords')
stopWords = set(stopwords.words('english'))
#stopWords.add()

doc_new=[]
    
for tweet in docs_tokenized:
    tweet_str_ = []
    for word in tweet:
        if word not in stopWords:
            tweet_str_.append(word)
    doc_new.append(tweet_str_)   
           
docs=doc_new
print(docs[0:10])

# stemming (applied to tokenized text)
from nltk.stem import SnowballStemmer
stemmer= SnowballStemmer("english")

stemmed_tweets=[]
for tweet in docs:
    tweets=[]
    for word in tweet:
        tweets.append(stemmer.stem(word))
    stemmed_tweets.append(tweets)

print(stemmed_tweets)

string_tweets=[]
for tweet in stemmed_tweets:
    tweet=" ".join(tweet)
    string_tweets.append(tweet)
    
#removing duplicate twitter lines after-preprocessing:

docs_new=[]
for i in string_tweets:
       if i not in docs_new:
            docs_new.append(i)
docs=docs_new

#number of tweets after removing duplicates:
len(docs)

#inverted_indexes
list_index=[]
for i in range(len(docs)):
    list_index.append((i,docs[i].split()))
list_index[0]


#Returning the documents that match “Farage” and “EU”
sent=['farag','eu']
search=[]
for i, j  in list_index:
    if all(word in j for word in sent):
            search.append(i)
                                                            
and_search = []    
for i in search:
    and_search.append(list_index[i])    

# removing duplicates
and_search_res= [[a, b] for i, [a, b] in enumerate(and_search) if not any(c == b for _, c in and_search[:i])]


sent=['camera','photo']
search=[]
for i, j  in list_index:
    if any(word in j for word in sent):
            search.append(i)
                                                            
or_search = []    
for i in search:
    or_search.append(list_index[i])    

# removing duplicates
or_search_res= [[a, b] for i, [a, b] in enumerate(or_search) if not any(c == b for _, c in or_search[:i])]


#Building the term-doc matrix and apply tf-idf to the counts (after-preprocessing)

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer   
vec = CountVectorizer()
X = vec.fit_transform(docs)
df = pd.DataFrame(X.toarray(), columns=vec.get_feature_names())
df_trans=df.T

matrix= df_trans.values
print(matrix)
matrix.shape

# MRR over the query “Nigel Farage leading new pro brexit party”
sent='Nigel Farage leading new pro brexit party'
sent=sent.lower()
sent=sent.split()
print(sent)

#before search it is necessary to preprocess the query:
 # removing  stop words
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
nltk.download('stopwords')
stopWords = set(stopwords.words('english'))

sent_new=[]
for word in sent:
    if word not in stopWords:
            sent_new.append(word)            
sent=sent_new


# stemming
from nltk.stem import SnowballStemmer
stemmer= SnowballStemmer("english")

sent_new=[]
for word in sent:
    sent_new.append(stemmer.stem(word))   

sent=sent_new
print(sent)

search=[]
for i, j in list_index:
    if all(word in j for word in sent):
            search.append(i)
                                                            
#and_search = []    
#for i in search:
#    and_search.append(list_index[i])    

#print(and_search)

#Result: apparently there is no tweet that contains all the element of the suggested query
    