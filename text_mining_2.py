import nltk
import pandas as pd
from itertools import chain
import pylab as pl
import numpy as np
import operator
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import scipy
from sklearn.feature_extraction.text import TfidfTransformer
from nltk.stem import WordNetLemmatizer
import re
import string
from sklearn.metrics.pairwise import cosine_similarity


#Plotting a histogram of the word vocabulary of “Alice in Wonderland" (raw text)

alice = 'carroll-alice.txt'
sents = nltk.corpus.gutenberg.sents(alice)
list_words=list(chain(*sents))

wordfreq = []
for w in list_words:
    wordfreq.append(list_words.count(w))

dic = dict(zip(list_words, wordfreq))

sorted_dic = sorted(dic.items(), key=operator.itemgetter(1))
sorted_dic.reverse()
sorted_dic=dict(sorted_dic)    

pl.rcParams['figure.figsize']=(20,20)
X = np.arange(len(sorted_dic))
pl.bar(X, sorted_dic.values(), align='center', width=1)
#pl.xticks(X)
ymax = max(sorted_dic.values())
pl.ylim(0, ymax)
pl.show()


# now lets see frequency distribution of words after pre-processing (normalization, lemmatization, stop word removal, etc..)
# 1. normalization 

list_words_string = " ".join(list_words)
list_words_string_1 = re.sub('[\(\?\]\[\.\!\/\;\:\@\>\)\<\"\*\\,\-\'\_\|]', ' ', list_words_string)
list_words_string_1 = list_words_string_1.lower()
list_words_string_1 = list_words_string_1.replace("can   t ", "can not ")
list_words_string_1 = list_words_string_1.replace("didn   t", "did not ")
list_words_string_1 = list_words_string_1.replace("you   d", "you would ")
list_words_string_1 = list_words_string_1.replace("shan   t", "shall not ")
list_words_string_1 = list_words_string_1.replace("doesn   t", "does not ")
list_words_string_1 = list_words_string_1.replace("don   t", "do not ")
list_words_string_1 = list_words_string_1.replace("wasn   t", "was not ")
list_words_string_1 = list_words_string_1.replace("said", " ")


#2. removing stop words

nltk.download('stopwords')
stopWords = set(stopwords.words('english'))

words = word_tokenize(list_words_string_1)
no_stopwords = []
 
for x in words:
    if x not in stopWords:
        no_stopwords.append(x)
        
normalized_list_words_string_1=" ".join(no_stopwords)


#3. #lemmatization (applied on normalized text)
nltk.download('wordnet')
lem_norm=[]


lemmatizer=WordNetLemmatizer()
input_str=word_tokenize(normalized_list_words_string_1)

for word in input_str:
    lem_norm.append(lemmatizer.lemmatize(word))

lem_norm = " ".join(lem_norm)


#creating a list of values
list_ = lem_norm.split(' ')
#counting frequencies
wordfreq_2 = []
for w in list_:
    wordfreq_2.append(list_.count(w))
#sorting
dic_2 = dict(zip(list_, wordfreq_2))

sorted_dic_2 = sorted(dic_2.items(), key=operator.itemgetter(1))
sorted_dic_2.reverse()
sorted_dic_2=dict(sorted_dic_2)


# plotting histogram
pl.rcParams['figure.figsize']=(20,20)
X = np.arange(len(sorted_dic_2))
pl.bar(X, sorted_dic_2.values(), align='center', width=1)
#pl.xticks(X)
ymax = max(sorted_dic_2.values())
pl.ylim(0, ymax)
pl.show()    

#RESULTS: After procedures of normalization, lemmatization and stop-words removal, the graph has flattened
#the most frequnent tokens in the initial book were symbols and stop-words and since they were 
#eliminated in the pre-processeing, now it is easier to identify the most common words that bring semantic meaning to the text



# Now building the term-document matrix for the collection in “books”.
# Finding the most similar document content to the “Alice in Wonderland” using tf-idf and cosine similarity.

macbeth='shakespeare-macbeth.txt'
ball='chesterton-ball.txt'
moby_dick='melville-moby_dick.txt'
sense='austen-sense.txt'
alice='carroll-alice.txt'

sents_macbeth = nltk.corpus.gutenberg.sents(macbeth)
list_macbeth=list(chain(*sents_macbeth))

sents_ball= nltk.corpus.gutenberg.sents(ball)
list_ball=list(chain(*sents_ball))

sents_moby_dick = nltk.corpus.gutenberg.sents(moby_dick)
list_moby_dick=list(chain(*sents_moby_dick))

sents_sense = nltk.corpus.gutenberg.sents(sense)
list_sense=list(chain(*sents_sense))

sents_alice = nltk.corpus.gutenberg.sents(alice)
list_alice=list(chain(*sents_alice))


# 1. normalization MACBETH

list_macbeth = " ".join(list_macbeth)
list_macbeth = re.sub('[\(\?\]\[\.\!\/\;\:\@\>\)\<\"\*\,\-\'\_\|]', ' ', list_macbeth)
list_macbeth = list_macbeth.lower()

#stop words
nltk.download('stopwords')
stopWords = set(stopwords.words('english'))

words = word_tokenize(list_macbeth)
list_macbeth = []
 
for x in words:
    if x not in stopWords:
        list_macbeth.append(x)
        
#list_macbeth=" ".join(no_stopwords)

# 1. normalization BALL

list_ball = " ".join(list_ball)
list_ball = re.sub('[\(\?\]\[\.\!\/\;\:\@\>\)\<\"\*\\,\-\'\_\|]', ' ', list_ball)
list_ball = list_ball.lower()

#stop words
nltk.download('stopwords')
stopWords = set(stopwords.words('english'))

words = word_tokenize(list_ball)
list_ball = []
 
for x in words:
    if x not in stopWords:
        list_ball.append(x)
        
#list_ball=" ".join(no_stopwords)

# 1. normalization MOBY DICK

list_moby_dick = " ".join(list_moby_dick)
list_moby_dick = re.sub('[\(\?\]\[\.\!\/\;\:\@\>\)\<\"\*\\,\-\'\_\|]', ' ', list_moby_dick )
list_moby_dick = list_moby_dick.lower()


#stop words
nltk.download('stopwords')
stopWords = set(stopwords.words('english'))

words = word_tokenize(list_moby_dick)
list_moby_dick = []
 
for x in words:
    if x not in stopWords:
        list_moby_dick.append(x)
        
#list_moby_dick=" ".join(no_stopwords)

# 1. normalization SENSE

list_sense = " ".join(list_sense)
list_sense = re.sub('[\(\?\]\[\.\!\/\;\:\@\>\)\<\"\*\\,\-\'\_\|]', ' ', list_sense)
list_sense = list_sense.lower()


#stop words
nltk.download('stopwords')
stopWords = set(stopwords.words('english'))

words = word_tokenize(list_sense)
list_sense = []
 
for x in words:
    if x not in stopWords:
        list_sense.append(x)
        
#list_sense = " ".join(no_stopwords)

# 1. normalization ALICE
list_alice = " ".join(list_alice)
list_alice = re.sub('[\(\?\]\[\.\!\/\;\:\@\>\)\<\"\*\\,\-\'\_\|]', ' ', list_alice)
list_alice = list_alice.lower()
list_alice = list_alice.replace("can   t ", "can not ")
list_alice = list_alice.replace("didn   t", "did not ")
list_alice = list_alice.replace("you   d", "you would ")
list_alice = list_alice.replace("shan   t", "shall not ")
list_alice = list_alice.replace("doesn   t", "does not ")
list_alice = list_alice.replace("don   t", "do not ")
list_alice = list_alice.replace("wasn   t", "was not ")
list_alice = list_alice.replace("said", " ")

#stop words

nltk.download('stopwords')
stopWords = set(stopwords.words('english'))

words = word_tokenize(list_alice)
list_alice = []
 
for x in words:
    if x not in stopWords:
        list_alice.append(x)
        
list_alice = " ".join(no_stopwords)

# word frequency
wordfreq_macbeth = []
for w in list_macbeth:
    wordfreq_macbeth.append(list_macbeth.count(w))

# word frequency
wordfreq_ball = []
for w in list_ball:
    wordfreq_ball.append(list_ball.count(w))
    
# word frequency
wordfreq_moby_dick = []
for w in list_moby_dick:
    wordfreq_moby_dick.append(list_moby_dick.count(w))
    
# word frequency
wordfreq_sense = []
for w in list_sense:
    wordfreq_sense.append(list_sense.count(w))
    
# word frequency
wordfreq_alice = []
for w in list_alice:
    wordfreq_alice.append(list_alice.count(w))    

# zipping and sorting
dic_mac = dict(zip(list_macbeth, wordfreq_macbeth))

sorted_mac = sorted(dic_mac.items(), key=operator.itemgetter(1))
sorted_mac.reverse()
sorted_mac=dict(sorted_mac)

dic_ball = dict(zip(list_ball, wordfreq_ball))

sorted_ball = sorted(dic_ball.items(), key=operator.itemgetter(1))
sorted_ball.reverse()
sorted_ball=dict(sorted_ball)

dic_moby_dick = dict(zip(list_moby_dick, wordfreq_moby_dick))

sorted_moby_dick = sorted(dic_moby_dick.items(), key=operator.itemgetter(1))
sorted_moby_dick.reverse()
sorted_moby_dick=dict(sorted_moby_dick)

dic_sense = dict(zip(list_sense, wordfreq_sense))

sorted_sense = sorted(dic_sense.items(), key=operator.itemgetter(1))
sorted_sense.reverse()
sorted_sense=dict(sorted_sense)

dic_alice = dict(zip(list_alice, wordfreq_sense))

sorted_alice = sorted(dic_alice.items(), key=operator.itemgetter(1))
sorted_alice.reverse()
sorted_alice=dict(sorted_alice)

# creating DataFrames
df_mac=pd.DataFrame.from_dict(sorted_mac, orient='index')
df_ball=pd.DataFrame.from_dict(sorted_ball, orient='index')
df_moby_dick=pd.DataFrame.from_dict(sorted_moby_dick, orient='index')
df_sense=pd.DataFrame.from_dict(sorted_sense, orient='index')
df_alice=pd.DataFrame.from_dict(sorted_alice, orient='index')

#dataframe_shape
print(df_mac.shape,df_ball.shape,df_moby_dick.shape, df_sense.shape,df_alice.shape)
df=df_moby_dick.join(df_ball, lsuffix='_caller', rsuffix='_other')
df=df.join(df_sense, lsuffix='_caller', rsuffix='_other')
df=df.join(df_mac, lsuffix='_caller', rsuffix='_other')
df=df.join(df_alice, lsuffix='_caller', rsuffix='_other')
df.columns=['Moby_dick','Ball','Sense','Macbeth','Alice']
df=df.fillna(0)

#TF-IDF matrix
transformer = TfidfTransformer(smooth_idf=False)
ls= list(zip(*[df[c].values.tolist() for c in df]))
tfidf = transformer.fit_transform(ls)                       
tfidf.toarray()    


tfidf=scipy.sparse.csr_matrix.toarray(tfidf)  


alice=np.transpose(tfidf[:,4])
moby=np.transpose(tfidf[:,0])
ball=np.transpose(tfidf[:,1])
sense=np.transpose(tfidf[:,2])
mac=np.transpose(tfidf[:,3])


## COSINE SIMILARITY


#alice vs moby_dick
alice_moby_cosine_similarity=cosine_similarity(alice,moby)
#alice vs ball
alice_ball_cosine_similarity=cosine_similarity(alice,ball)
#alice vs sense
alice_sense_cosine_similarity=cosine_similarity(alice,sense)
#alice vs Macbeth
alice_Macbeth_cosine_similarity=cosine_similarity(alice, mac)

