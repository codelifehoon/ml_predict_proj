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
from sklearn.externals import joblib
from konlpy.tag import Okt
import pickle
import os

pd.set_option('display.expand_frame_repr', False)
ver = '2';
filename = 'naive-bayes-multinomial_11st_opt'+ver+'.joblib.pkl'

# clf = joblib.load(filename)
with open(os.path.dirname(os.path.realpath(__file__)) + '/naive-bayes-multinomial_11st_opt2.pickle' , 'rb') as f:
    pipecls = pickle.load(f)


my_data = ['그레이','12','250','선택','민트','핑크','은색','silver','실버']

predict = pipecls.predict(my_data)
predict_proba = pipecls.predict_proba(my_data)
predict_data_max = []
print(predict)
print(predict_proba)

for data in predict_proba:
    predict_data_max.append(max(data))


predict_data = pd.DataFrame({'predict':predict
                                ,'maxval':predict_data_max
                                ,'validation':1})

predict_data.loc[predict_data['maxval']<0.95,'validation':'validation'] =  0
print(predict_data)


predict_data = predict_data.loc[predict_data['maxval']>0.98,:]
print(predict_data)
