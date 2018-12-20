import json
import sys
from urllib.parse import quote
from urllib.request import urlopen

import numpy as np
import pandas as pd
from konlpy.tag import Okt


class cate_suggest_model:
    def __init__(self):
        print('__init__')


    def word_analysis_used_api(self,opt_nm):
        # 확인 쿼리 형태소 분석

        print('API 호출' + opt_nm)
        token_str = urlopen('https://open-korean-text-api.herokuapp.com/tokenize?text=' + quote(opt_nm)).read()
        token_json = json.loads(token_str)

        rows = pd.DataFrame({'pumsa': token_json['tokens'],'word': token_json['token_strings'] })

        r = []
        for i,row in rows.iterrows():

            pumsa = row.pumsa
            word = row.word

            if pumsa.find('Punctuation')==-1 and pumsa.find('Josa')==-1 and pumsa.find('Eomi')==-1:
                r.append(word)

        return (" ".join(r)).strip()

    def word_analysis(self,sentence):
        # 형태소 분리하고
        # row.OPT_CODE 에 따라서 색상 일때는 생상에 대한 문자만, 사이즈일때에는 사이즈일때의 문자만 취합하고
        # 그걸 다시 OPT_TRIM에 update

        twitter = Okt()
        try:
            malist = twitter.pos(sentence, norm=True, stem=True)
            r=[]
            for (word,pumsa) in malist:
                if not pumsa in ["Josa","Eomi","Punctuation"]:\
                    r.append(word)
            return (" ".join(r)).strip()

        except:
            print(" error:", sys.exc_info() ,'sentence:',sentence)


    def predict(self,goodsnm,rownum,ctv,clf,y_classes,y_classes_name):
        predict_X = [self.word_analysis_used_api(goodsnm)]

        predict_proba_y = clf.predict_proba(ctv.transform(predict_X).toarray())
        print(np.shape(predict_proba_y))
        predict_proba_y = np.reshape(predict_proba_y,[-1])
        predict_y_list = pd.DataFrame({'y_class':y_classes,'y_predict':predict_proba_y})

        predict_y_list = predict_y_list.sort_values(by=['y_predict'], ascending=False)
        predict_y_topn_list = predict_y_list.iloc[:rownum]


        mdata = pd.merge(predict_y_topn_list,y_classes_name)



        return mdata
