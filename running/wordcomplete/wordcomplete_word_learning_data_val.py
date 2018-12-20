import datetime
import time
import pickle

import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.contrib.layers import fully_connected
import logging
import argparse
import os
from google.cloud import storage


if __name__ == '__main__':
    print('__name__:',__name__)

logging.getLogger().setLevel(logging.INFO)
tf.set_random_seed(777)
local_traning_path = '/Users/codelife/Developer/python_proj/ml_predict_proj/running/wordcomplete/char_train_data'






# global idx2word,word2idx,dataX,dataY
with open(local_traning_path + '/idx2word.dump', "rb") as fp:
    idx2word = pickle.load(fp)

with open(local_traning_path + '/word2idx.dump', "rb") as fp:
    word2idx = pickle.load(fp)

with open(local_traning_path + '/dataX.dump', "rb") as fp:
    dataX = pickle.load(fp)

with open(local_traning_path + '/dataY.dump' , "rb") as fp:
    dataY = pickle.load(fp)


print(dataX)
print(dataY)


'''
def make_sentence(idxs):

    sentence = []
    sentence2 = ''
    cnt = 0
    for s  in idxs:
        if cnt == 0:
            sentence  = s
            cnt=1
        else:
            sentence = sentence + [s[-1]]

    for c in sentence:
        sentence2 = sentence2 + idx2word[c]
    return sentence2

print('*'*100)
print(make_sentence(dataX))
print('*'*100)

print(make_sentence(dataY))

'''