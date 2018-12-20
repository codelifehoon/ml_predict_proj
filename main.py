# -*- coding: utf-8 -*-
# Copyright 2015 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


#ML include
import pickle
import os
from google.cloud import storage


# [START gae_flex_quickstart]
from flask import Flask
from flask_restful import reqparse
from flask_cors import CORS
import tensorflow as tf
import datetime
import logging
import pandas as pd
from model.cate_suggest_model import cate_suggest_model
from model import wordcomplete_model as wc_model, word_model as word_model
import loader.model_loader as modelloader

app = Flask(__name__)
# CORS(app, supports_credentials=True)
CORS(app)


# 옵션분류 학습자료
pipecls = None


# 문장 자동완서 학습자료
idx2word = None
word2idx = None
wordcomplete_saver = None
sess = None
graph = None

# 상품명 카테고리 제안
cate_suggest_ctv = None
cate_suggest_clf = None
cate_suggest_y_classes = None
cate_suggest_y_classes_name = None


logging.getLogger().setLevel(logging.INFO)

# test envs running vm option
# MODEL_BUCKET=codleife-ml-1101.appspot.com;SERVER=local

@app.before_first_request
def _load_model():
    global pipecls
    global idx2word,word2idx,wordcomplete_saver,sess,graph
    global  cate_suggest_ctv,cate_suggest_clf,cate_suggest_y_classes,cate_suggest_y_classes_name

    if   "MODEL_BUCKET" in os.environ:
        logging.info('_load_model real')
        pipecls = modelloader._load_wordanalysis_model_real()
        idx2word,word2idx,wordcomplete_saver,sess,graph = modelloader._load_charcomplete_model_real()
        cate_suggest_ctv,cate_suggest_clf,cate_suggest_y_classes,cate_suggest_y_classes_name = modelloader._load_cate_suggest_model_real()

    else:
        logging.info('_load_model local')
        pipecls = modelloader._load_wordanalysis_model_local()
        idx2word,word2idx,wordcomplete_saver,sess,graph = modelloader._load_charcomplete_model_local()
        cate_suggest_ctv,cate_suggest_clf,cate_suggest_y_classes,cate_suggest_y_classes_name = modelloader._load_cate_suggest_model_local()
        print(cate_suggest_y_classes_name)



@app.route('/')
def hello():
    """Return a friendly HTTP greeting."""

    return 'Hello ML.'

@app.route('/wordanalysis', methods=['GET', 'POST'])
def wordanalysis():

    model = word_model.word_analysis(pipecls)

    parser = reqparse.RequestParser()
    parser.add_argument('PRD_NO',type=str, location='json')
    parser.add_argument('OPT_NM',type=str, location='json')
    args = parser.parse_args()
    prd_no = args['PRD_NO']
    opt_nm = args['OPT_NM']

    return model.option_discover(prd_no,opt_nm)


@app.route('/wordcomplete', methods=['GET', 'POST'])
def wordcomplete():
    print('_wordcomplete_')

    parser = reqparse.RequestParser()
    parser.add_argument('sentence', type=str)
    sentence = parser.parse_args()['sentence']
    print('step1:',datetime.datetime.now())

    model = wc_model.sentence_model(wordcomplete_saver,idx2word,word2idx)

    return ''.join(model.predict_sentence2(sentence,1,sess,graph))

@app.route('/charcomplete', methods=['GET', 'POST'])
def charcomplete():
    print('charcomplete')

    parser = reqparse.RequestParser()
    parser.add_argument('sentence', type=str)
    sentence = parser.parse_args()['sentence']
    print('step1:',datetime.datetime.now())
    sequence_length = 17

    model = wc_model.sentence_model(wordcomplete_saver,idx2word,word2idx,sequence_length)
    predict_val= model.predict_sentence_char(sentence,sequence_length,0,sess,graph)

    return predict_val


@app.route('/catesuggest', methods=['GET', 'POST'])
def catesuggest():

    print('catesuggest',datetime.datetime.now())

    parser = reqparse.RequestParser()
    parser.add_argument('goodsnm', type=str)
    goodsnm = parser.parse_args()['goodsnm']
    predict_cnt = 3

    global  cate_suggest_ctv,cate_suggest_clf,cate_suggest_y_classes,cate_suggest_y_classes_name

    model = cate_suggest_model()
    predicts = model.predict(goodsnm,predict_cnt,cate_suggest_ctv,cate_suggest_clf,cate_suggest_y_classes,cate_suggest_y_classes_name)


    return predicts.to_json(orient='records')


@app.errorhandler(500)
def server_error(e):
    logging.exception('An error occurred during a request.')
    return """
    An internal error occurred: <pre>{}</pre>
    See logs for full stacktrace.
    """.format(e), 500


if __name__ == '__main__':
    # This is used when running locally. Gunicorn is used to run the
    # application on Google App Engine. See entrypoint in app.yaml.
    app.run(host='127.0.0.1', port=8080, debug=True)

# [END gae_flex_quickstart]

