import gzip

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
import sklearn.metrics
import  sys


twitter = Okt()
ver = '2';
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

splits_cnt = 10
accuracy_score = 0




def word_analysis(sentence):
    # 형태소 분리하고
    # row.OPT_CODE 에 따라서 색상 일때는 생상에 대한 문자만, 사이즈일때에는 사이즈일때의 문자만 취합하고
    # 그걸 다시 OPT_TRIM에 update

    try:
        malist = twitter.pos(sentence, norm=True, stem=True)
        r=[]
        for (word,pumsa) in malist:
            if not pumsa in ["Josa","Eomi","Punctuation"]:
                r.append(word)

        return (" ".join(r)).strip()

    except:
        print(" error:", sys.exc_info() ,'sentence:',sentence)


ctv = None
clf = None          # MultinomialNB(alpha=0.01)
y_classes = None


with open('cate_suggest_ctv.dump', 'rb') as f:
    ctv = pickle.load(f)

with open('cate_suggest_clf.dump', 'rb') as f:
    clf = pickle.load(f)

with open('cate_suggest_y_classes.dump', 'rb') as f:
    y_classes = pickle.load(f)


with open('cate_suggest_datas.dump', 'rb') as f:
    datas = pickle.load(f)



X = datas['goods_nm_ana'].values
y = datas['leaf_cate_no'].values

kf = KFold(n_splits=100)
break_cnt = 0
for train_index,test_index in kf.split(X,y):
    break_cnt +=1
    if break_cnt > 10:
        break


    test_X = X[test_index]
    test_y = y[test_index]
    test_X = ctv.transform(test_X).toarray()

    predict_Y = clf.predict(test_X)
    print(metrics.accuracy_score(test_y,predict_Y))

    # print('Classification Repost:')
    # pp.pprint(metrics.classification_report(test_y,predict_Y))
    # print('Classification Matrix:')
    # pp.pprint(metrics.confusion_matrix(test_y,predict_Y))







# predict_X = [word_analysis('스마트키 가죽 키홀더 지갑 열쇠고리')]
#
# predict_proba_y = clf.predict_proba(ctv.transform(predict_X).toarray())
# print(np.shape(predict_proba_y))
# predict_proba_y = np.reshape(predict_proba_y,[-1])
# predict_y_list = pd.DataFrame({'y_class':y_classes,'y_predict':predict_proba_y})
#
# predict_y_list = predict_y_list.sort_values(by=['y_predict'], ascending=False)
# predict_y_list = predict_y_list.iloc[:3]
# print(predict_y_list)

#GridSearchCV 최적의 학습 parameter 탐색


# CountVectorizer , MultinomialNB
# 0.7874833007912856
# 0.7895385880176755
# 0.7898468811016339
# 0.7880998869592025
# 0.7841948412290618
# 0.7911828177987874
# 0.7854280135648957
# 0.7840920768677423
# 0.7950878635289281
# 0.7867639502620491

# CountVectorizer , MultinomialNB

