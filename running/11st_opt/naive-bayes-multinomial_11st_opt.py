
from sklearn.feature_extraction.text import TfidfVectorizer, HashingVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, KFold
from sklearn.externals import joblib
from konlpy.tag import Okt
import numpy as np
import pandas as pd
from scipy.stats import  sem
from sklearn import  metrics
import  pprint as pp
import pickle


def word_analysis(opt_code, opt_nm):
    # 형태소 분리하고
    # row.OPT_CODE 에 따라서 색상 일때는 생상에 대한 문자만, 사이즈일때에는 사이즈일때의 문자만 취합하고
    # 그걸 다시 OPT_TRIM에 update

    malist = twitter.pos(opt_nm,norm=True,stem=True)
    r=[]
    for (word,pumsa) in malist:
        if not pumsa in ["Josa","Eomi","Punctuation"]:

            if opt_code == 0 and pumsa in ['Number'] :      ## size data
                r.append(word)
            if opt_code == 1 and pumsa not in ['Number'] :      ## color data
                r.append(word)
    return (" ".join(r)).strip()

ver = '2';
twitter = Okt()

pd.set_option('display.expand_frame_repr', False)
datas = pd.read_excel('color_size_option'+ver+'.xlsx')



opt_trim = []
for i, row in datas.iterrows():
    opt_trim.append(word_analysis(row.OPT_CODE, row.OPT_NM))


datas['OPT_TRIM'] = opt_trim


#################### 전처리 끝 ########################

def evaluate_cross_validation(clf, X, y, K):

    scores = cross_val_score(clf,X,y,cv=K)
    print('score:',scores)
    print('mean score:',np.mean(scores) , 'sem score:',sem(scores))

def train_and_evaluate(clf,X_train,X_test,y_train,y_test):

    clf.fit(X_train,y_train)
    print('Accurancy on training set:')
    print(clf.score(X_train,y_train))
    print('Accurancy on test set:')
    print(clf.score(X_test,y_test))

    y_pred = clf.predict(X_test)
    y_pred_proba = clf.predict_proba(X_test)

    print('predict_proba Repost:')
    print(y_pred_proba)

    print('Classification Repost:')
    pp.pprint(metrics.classification_report(y_test,y_pred))
    print('Classification Matrix:')
    pp.pprint(metrics.confusion_matrix(y_test,y_pred))



X = datas.OPT_TRIM
y = datas.OPT_CODE

# filename = news.filenames
# print(filename[0:10])
# print(news.keys())
# print(news.description)
# print(type(X),type(y))
# print(news.target_names)

split_rate = 0.75
split_size = int(len(X) * split_rate)
X_train,X_test = X[:split_size] , X[split_size:]
y_train,y_test = y[:split_size] , y[split_size:]
my_data = ['그레이','선택하시오']



clf_1 = Pipeline([
                    ('vect',CountVectorizer())
                    ,('clf',MultinomialNB(alpha=0.01))
                ])

clf_2 = Pipeline([
    ('vect',TfidfVectorizer())
    ,('clf',MultinomialNB())
])

clf_3 = Pipeline([
    ('vect',TfidfVectorizer())
    ,('clf',MultinomialNB(alpha=0.01))
])


clfs = [clf_1,clf_2,clf_3]




# for clf in clfs:
#     evaluate_cross_validation(clf, X_train, y_train, 5)

pipecls = clf_3

train_and_evaluate(pipecls, X_train, X_test, y_train, y_test)

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


predict_data = predict_data.loc[predict_data['maxval']>0.95,:]
print(predict_data)


filename = 'naive-bayes-multinomial_11st_opt'+ver+'.joblib.pkl'
_ = joblib.dump(pipecls, filename, compress=9)


with open('naive-bayes-multinomial_11st_opt'+ver+'.pickle', 'wb') as f:
    pickle.dump(pipecls, f, pickle.HIGHEST_PROTOCOL)







