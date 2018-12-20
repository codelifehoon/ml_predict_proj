import pandas as pd
import json
from urllib.parse   import quote
from urllib.request import urlopen
from konlpy.tag import Okt


class word_analysis:
    def __init__(self,pipecls):
        print('__init__word_analysis_module')
        self.pipecls = pipecls
 

    def option_discover(self,prd_no,opt_nm):

        prd_nos = []
        opt_nms = []

        prd_nos.append(prd_no)
        opt_nms.append(opt_nm)


        valdata = pd.DataFrame({'PRD_NO' :prd_nos,'OPT_NM' :opt_nms })


        predicts =[]
        for i , row in valdata.iterrows():

            # 형태소 분석으로 각각 문장을 구분하고
            # 각 문장을 preditct에 넣고 검증하고
            # 적합하면.. 값을 취한다 차후 dataframe에 넣을수 있도록 배열화 한다.

            words = self.word_analysis_used_api(row.OPT_NM)
            predicts.append(self.predict_data(words))


        valdata['predict'] = predicts
        return valdata.to_json(orient='records')

    def word_analysis(self,opt_nm):
        # 확인 쿼리 형태소 분석
        twitter = Okt()
        malist = twitter.pos(opt_nm,norm=True,stem=False)
        r=[]
        for (word,pumsa) in malist:
            if not pumsa in ["Josa","Eomi","Punctuation"] and not word in ['mm','MM'] :
                if pumsa in ['Number']:
                    r = r + word.split(',')
                else:
                    r.append(word)
        return r

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

            if pumsa.find('Punctuation')==-1 and pumsa.find('Josa')==-1 and pumsa.find('Eomi')==-1 and not word in ['mm','MM'] :
                if pumsa.find('Number') >= 0:
                    r = r + word.split(',')
                else:
                    r.append(word)

        return r


    def predict_data(self,words):
        # 전달된 값에 대한 속성 및 값 전달

        predict = self.pipecls.predict(words)
        predict_proba = self.pipecls.predict_proba(words)
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


