import datetime
import pickle
import sys
from multiprocessing import Pool, cpu_count

import pandas as pd
from konlpy.tag import Okt
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer

pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)


datas = pd.read_excel('goods_leaf_cate.xlsx')
splits_cnt = 10
accuracy_score = 0

# datas = datas.iloc[:1000]


def word_analysis(sentence):
    # 형태소 분리하고
    # row.OPT_CODE 에 따라서 색상 일때는 생상에 대한 문자만, 사이즈일때에는 사이즈일때의 문자만 취합하고
    # 그걸 다시 OPT_TRIM에 update

    try:
        twitter = Okt()
        malist = twitter.pos(sentence, norm=True, stem=True)
        r=[]
        for (word,pumsa) in malist:
            if not pumsa in ["Josa","Eomi","Punctuation"]:
                r.append(word)

        return (" ".join(r)).strip()

    except:
        print(" error:", sys.exc_info() ,'sentence:',sentence)
        return "";



goods_nms = []
goods_nms = datas['goods_nm'].values

with Pool(processes=cpu_count()) as pool:
    goods_nms = pool.map(word_analysis,goods_nms)

# for i, row in datas.iterrows():
#     goods_nms.append(word_analysis(row.goods_nm))
#     if  i%10000 == 0:
#         print(datetime.datetime.now(),': word_analysis:', i)
#

datas['goods_nm_ana'] = goods_nms      # 형태소 분석 완료
datas = datas.drop(columns=['goods_nm'])


ctv = TfidfVectorizer()
ctv.fit(datas['goods_nm_ana'])


with open('cate_suggest_ctv.dump', 'wb') as f:
    pickle.dump(ctv, f, pickle.HIGHEST_PROTOCOL)


with open('cate_suggest_datas.dump', 'wb') as f:
    pickle.dump(datas, f, pickle.HIGHEST_PROTOCOL)


print(datetime.datetime.now(),':_fin_')
