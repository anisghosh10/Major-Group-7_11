import pandas as pd
import os
import string
import numpy as np
import matplotlib.pyplot as plt
import nltk
from nltk.stem import SnowballStemmer

start = 1981
end = 2019

stopword = nltk.corpus.stopwords.words("english")
stop_updated = stopword  # ["said","United", "State", "today", "said", "Net", "COMPANY", "REPORT", "Government"]

ps = nltk.PorterStemmer()
ss = nltk.SnowballStemmer(language = 'english')
wn = nltk.WordNetLemmatizer()

import re
# Creating a user defined function
def clean_text(text):
    # Stripping white spaces before and after the text
    text = text.strip(" ")
    # Replacing multiple spaces with a single space
    text = re.sub("\s+"," ", text)
    text =re.sub("[^\w\s]"," ",text)
    #text = re.sub("\d+$","",text)
    # Replacing punctuations
    text = "".join([word.lower() for word in text if word not in string.punctuation])
    # Creating tokens
    tokens = re.split('\W+', text)
    # removing stopwords and stemming - snowball stemming
    text_final = [ss.stem(word) for word in tokens if word not in stop_updated and len(word)>2]
#     # creating a list of tokens
    text_final = " ".join(text_final)
    return text_final

# Creating a user defined function
def clean_text(text):
    # Stripping white spaces before and after the text
    text = text.strip(" ")
    # Replacing multiple spaces with a single space
    text = re.sub("\s+"," ", text)
    text =re.sub("[^\w\s]"," ",text)
    #text = re.sub("\d+$","",text)
    # Replacing punctuations
    text = "".join([word.lower() for word in text if word not in string.punctuation])
    # Creating tokens
    tokens = re.split('\W+', text)
    # removing stopwords and  lemmatization
    text_final = [wn.lemmatize(word) for word in tokens if word not in stop_updated and len(word)>2]
#     # creating a list of tokens
    text_final = " ".join(text_final)
    return text_final


for year in range(start, end+1):
    articles = pd.read_csv(f"articles/articles_{year}.csv", error_bad_lines=False)
    if articles.article.isna().sum()>0:
        articles.article = articles.article.dropna()
    articles['article_cleaned'] = articles['article'].apply(lambda x: clean_text(x))
    print('Done stemming')
    articles['article_cleaned_lemmatizer'] = articles['article'].apply(lambda x: clean_text(x))
    print('Done lemma')
    useful_columns = ['_id', 'url', 'word_count', 'section', 'date', 'type', 'headline',
       'abstract', 'article_cleaned', 'article_cleaned_lemmatizer']
    articles[useful_columns].to_csv(f'new_article_{year}.csv',index=False)
    print('Done',year)