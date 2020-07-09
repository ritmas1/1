import unicodedata
from sklearn.datasets import fetch_20newsgroups
import nltk
import re
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import urllib

#opening file from github
url="https://raw.githubusercontent.com/ritmas1/1/master/india.txt"
urllib.request.urlretrieve(url,'india.txt')  
file=open('india.txt').read()


#2.2 Find every occurrence of a number and replace it with the word digit in this document

file = unicodedata.normalize("NFKD", file)
digits= re.sub('\\d+','digit', file)
digits1= re.sub('([0-9]+[t][h]|[0-9].[0-9]|[0-9])','digit', digits)
digits2= re.sub('([r][d])','digit', digits1)
#check
print(digits2)

#2.2 Find every occurrence of a number and replace it with the word digit in this document
file = unicodedata.normalize("NFKD", file)
digits= re.sub('\\d+','digit', file)
digits1= re.sub('([0-9]+[t][h]|[0-9].[0-9]|[0-9])','digit', digits)
digits2= re.sub('([r][d])','digit', digits1)
#check
print(digits2)

# 2.3 identify punctuation marks 
punctuation= re.findall('[\(\?\]\[\.\!\/\;\:\@\>\)\<\"\|\-\,]',file)
print(punctuation)

# TASK 3

nltk.download('punkt')
categories = ['sci.med']
twenty_train = fetch_20newsgroups(subset='train',categories=categories, shuffle=True, random_state=42)
sentence = twenty_train['data'][15]
print(sentence)

# 3.2 normalization

sentence1 = re.sub('[\(\?\]\[\.\!\/\;\:\@\>\)\<\"\|]', '', sentence)
sentence2 = re.sub('[\,\-]', ' ', sentence1)
sentence3 = sentence2.replace("shouldn't", "should not")
sentence3 = sentence3.replace("Lyme's", "Lyme")
sentence3 = sentence3.replace("Isn't", "Is not")
sentence3 = sentence3.replace("I'm", "I am")
sentence3 = sentence3.replace("I'd", "I had")
sentence3 = sentence3.replace("doesn't", "does not")
sentence3 = sentence3.lower()
sentence3 = re.sub('[\']', ' ', sentence3)


#stop words
nltk.download('stopwords')
stopWords = set(stopwords.words('english'))
words = word_tokenize(sentence3)
no_stopwords = []
 
for x in words:
    if x not in stopWords:
        no_stopwords.append(x)
        
normalized=" ".join(no_stopwords)


# 3.1 tokenization
from nltk.tokenize import word_tokenize
tokens = word_tokenize(sentence)
print(tokens)


tokenized=" ".join(tokens)

#lemmatization (applied on normalized text)
nltk.download('wordnet')

lemmatized_text=[]
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
lemmatizer=WordNetLemmatizer()
input_str=word_tokenize(normalized)
for word in input_str:
    lemmatized_text.append(lemmatizer.lemmatize(word))

lemmatized_text=" ".join(lemmatized_text)


#3.4 stemming 
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
stemmer= PorterStemmer()

stemmed_text=[]
input_str=word_tokenize(normalized)
for word in input_str:
    stemmed_text.append(stemmer.stem(word))

stemmed_text=" ".join(stemmed_text) 
