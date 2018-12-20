import pickle

import numpy as np
import pandas as pd
from konlpy.tag import Okt
from sklearn.naive_bayes import MultinomialNB

twitter = Okt()
ver = '2';
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)


kfold_splits_cnt = 5
total_accuracy_score = 0
org_X_split = 10

ctv = None
datas = None



with open('cate_suggest_ctv.dump', 'rb') as f:
    ctv = pickle.load(f)

with open('cate_suggest_datas.dump', 'rb') as f:
    datas = pickle.load(f)


org_X_split = 100
org_X_size = datas.shape[0]
org_X_rowcount = int(org_X_size / org_X_split)

clf = MultinomialNB(alpha=0.01)
qunque_y = np.unique(datas['leaf_cate_no'])


for i in range(org_X_split):
    print('datas slice number:',i)
    split_X = None
    split_Y = None
    if i == org_X_split-1:
        split_X = datas['goods_nm_ana'].iloc[i * org_X_rowcount:org_X_size]
        split_X = ctv.transform(split_X).toarray()
        split_Y = datas['leaf_cate_no'].iloc[i * org_X_rowcount:org_X_size].values


    else:
        split_X = datas['goods_nm_ana'].iloc[i * org_X_rowcount:(i + 1) * org_X_rowcount]
        split_X = ctv.transform(split_X).toarray()
        split_Y = datas['leaf_cate_no'].iloc[i * org_X_rowcount:(i + 1) * org_X_rowcount].values



    if i==0:
        clf.partial_fit(split_X,split_Y,np.unique(datas['leaf_cate_no']))
    else:
        clf.partial_fit(split_X,split_Y)



    # kf = KFold(n_splits=kfold_splits_cnt)
    # for train_index , test_index in kf.split(split_X, split_Y):
    #     train_X = split_X[train_index]
    #     train_y = split_Y[train_index]
    #
    #     test_X = split_X[train_index]
    #     test_y = split_Y[train_index]
    #
    #     clf.partial_fit(train_X,train_y,)
    #
    #     predict_y = clf.predict(test_X)
    #     accuracy_score = metrics.accuracy_score(predict_y, test_y)
    #     total_accuracy_score += accuracy_score
    #     print('accuracy_score:', accuracy_score)


# print(goods_nm_fit.vocabulary_)
# print(goods_nm_vec)
y_classes =clf.classes_


with open('cate_suggest_clf.dump', 'wb') as f:
    pickle.dump(clf, f, pickle.HIGHEST_PROTOCOL)

with open('cate_suggest_y_classes.dump', 'wb') as f:
    pickle.dump(y_classes, f, pickle.HIGHEST_PROTOCOL)

print('clf.classes_:',np.shape(clf.classes_))
print('clf.feature_count_:',np.shape(clf.feature_count_))


#GridSearchCV 최적의 학습 parameter 탐색




