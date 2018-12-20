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
import json


ver = '2';
pd.set_option('display.expand_frame_repr', False)
pipecls = joblib.load('naive-bayes-multinomial_11st_opt' + ver + '.joblib.pkl')
valdata = pd.read_excel('color_size_option_val.xlsx')

twitter = Okt()
colors = []
sizes = []

def word_analysis(opt_nm):
    # 확인 쿼리 형태소 분석
    malist = twitter.pos(opt_nm,norm=True,stem=False)
    r=[]
    for (word,pumsa) in malist:
        if not pumsa in ["Josa","Eomi","Punctuation"] and not word in ['mm','MM'] :
            if pumsa in ['Number']:
                r = r + word.split(',')
            else:
                r.append(word)
    return r


def predict_data(words):
    # 전달된 값에 대한 속성 및 값 전달

    predict = pipecls.predict(words)
    predict_proba = pipecls.predict_proba(words)
    predict_data_max = []

    for data in predict_proba:
        predict_data_max.append(max(data))


    predict_data = pd.DataFrame({ 'words':words
                                    ,'predict':predict
                                    ,'maxval':predict_data_max
                                    ,'validation':1})

    predict_data.loc[predict_data['maxval']<0.98,'validation':'validation'] =  0
    # print(predict_data)


    # predict_data = predict_data.loc[predict_data['maxval']>0.98,:]
    # print(predict_data)

    return predict_data


predicts =[]
for i , row in valdata.iterrows():

    # 형태소 분석으로 각각 문장을 구분하고
    # 각 문장을 preditct에 넣고 검증하고
    # 적합하면.. 값을 취한다 차후 dataframe에 넣을수 있도록 배열화 한다.

    words = word_analysis(row.OPT_NM)
    predicts.append(predict_data(words))



valdata['predict'] = predicts
print(valdata.head(3))

writer = pd.ExcelWriter('color_size_option_result.xlsx')
valdata.to_excel(writer,'Sheet1')
writer.save()
