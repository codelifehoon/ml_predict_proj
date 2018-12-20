from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.contrib.layers import fully_connected, batch_norm, dropout
import numpy as np
import  os
import pandas as pd
import pprint as pp
from konlpy.tag import  Okt
import pickle
import model.wordcomplete_model as wc
import datetime
import sys

print('__name__:',__name__)

sequence_length = 17
rootPath = '/Users/codelife/Developer/11st_escrow2/74.pas-pay-kafkaproject/src/main/java/com/skp/payment/pas/tgcorp/domain'
ext = '.java'
# with tf.variable_scope(self.scopeName):
# rootPath = '/Users/codelife/Developer/tensorFlow/DeepLearningZeroToAll/tests'
# ext = '.py'
print('start learning..' , datetime.datetime.now())

sentence_pret  = wc.sentence_pret(rootPath)

allFiles = sentence_pret.getListOfFiles(rootPath,ext)
# allFiles = allFiles[:50]

ana_docs,idx2word = sentence_pret.idx2word_by_char(allFiles)
word2idx = sentence_pret.word2idx(idx2word)

print('\n'.join(allFiles))
# print("*"*100)
# print("".join(ana_docs))

# sequence = [ana_docs[0]]
sequence = ana_docs

# sequence = ("if you want to build a ship, don't drum up people together to "
#             "collect wood and don't assign them tasks and work, but rather "
#             "teach them to long for the endless immensity of the sea.")


dataX , dataY = sentence_pret.make_word_xy_list(sequence,sequence_length,word2idx)



with open("./char_train_data/idx2word.dump", "wb") as fp:
    pickle.dump(idx2word, fp, pickle.HIGHEST_PROTOCOL)

with open("./char_train_data/word2idx.dump", "wb") as fp:
    pickle.dump(word2idx, fp, pickle.HIGHEST_PROTOCOL)


with open("./char_train_data/dataX.dump", "wb") as fp:
    pickle.dump(dataX, fp, pickle.HIGHEST_PROTOCOL)

with open("./char_train_data/dataY.dump", "wb") as fp:
    pickle.dump(dataY, fp, pickle.HIGHEST_PROTOCOL)

