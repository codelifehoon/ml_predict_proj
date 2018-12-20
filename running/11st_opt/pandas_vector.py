from sklearn.datasets import fetch_20newsgroups
from sklearn.naive_bayes import MultinomialNB
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, HashingVectorizer, CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, KFold
from scipy.stats import  sem
from sklearn import  metrics
import  pprint as pp
import pandas as pd
from urllib.parse   import quote
from urllib.request import urlopen
import json


url = 'https://open-korean-text-api.herokuapp.com/tokenize?text=' + quote('ABC/가나다/red/color/95')
token_str = urlopen(url).read()

token_json = json.loads(token_str)


rows = pd.DataFrame({'pumsa': token_json['tokens'],'word': token_json['token_strings'] })


r = []
for i,row in rows.iterrows():

    pumsa = row.pumsa
    word = row.word



    if pumsa.find('Punctuation')==-1 \
            and pumsa.find('Josa')==-1 \
            and pumsa.find('Eomi')==-1 \
            and not word in ['mm','MM'] :
        if pumsa in ['Number']:
            r = r + word.split(',')
        else:
            r.append(word)






