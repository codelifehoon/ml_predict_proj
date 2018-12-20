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



rootPath = '/Users/codelife/Developer/tensorFlow/DeepLearningZeroToAll/tests'

twitter = Okt()
sequence_length = 10
learning_late = 0.1

sentence_pret = wc.sentence_pret(rootPath)

allFiles = sentence_pret.getListOfFiles(rootPath,'.java')[:1]
ana_docs,idx2word = sentence_pret.idx2word(allFiles)
word2idx = []

with open("./setence_train_data/wordcomplete_sentence_idx2word.dump", "rb") as fp:
    idx2word = pickle.load(fp)

with open("./setence_train_data/wordcomplete_sentence_word2idx.dump", "rb") as fp:
    word2idx = pickle.load(fp)



data_dim    = len(idx2word)
hidden_size = len(idx2word)
num_classes = len(idx2word)     #hidden_size

sequence = ana_docs[0]
# sequence = ("if you want to build a ship, don't drum up people together to "
#             "collect wood and don't assign them tasks and work, but rather "
#             "teach them to long for the endless immensity of the sea.")

dataX , dataY = sentence_pret.make_word_xy_list(sequence,sequence_length,word2idx)


tf.reset_default_graph()
saver = tf.train.import_meta_graph('./setence_train_data/wordcomplete_sentence_train_data.meta')

sess = tf.Session()
sess.run(tf.global_variables_initializer())
saver.restore(sess,tf.train.latest_checkpoint('./setence_train_data'))
graph = tf.get_default_graph()

# for op in graph.get_operations():
#     print(op.name)

X  = graph.get_tensor_by_name('X_holder:0')
Y  = graph.get_tensor_by_name('Y_holder:0')
batch_size  = graph.get_tensor_by_name('batch_size:0')
mean_loss  = graph.get_tensor_by_name('mean_loss:0')
train_mode = graph.get_tensor_by_name('train_mode:0')
# train  = graph.get_tensor_by_name('train_optimizer:0')
train  = graph.get_operation_by_name('train_optimizer')

for step in range(201):
    mean_loss_val, _ = sess.run([mean_loss,train],{X:dataX,Y:dataY,train_mode:True,batch_size:len(dataX)})
    if step % 100 == 0 :
        print(step,mean_loss_val)

saver = tf.train.Saver()
saver.save(sess,save_path='./setence_train_data/wordcomplete_sentence_train_data')
