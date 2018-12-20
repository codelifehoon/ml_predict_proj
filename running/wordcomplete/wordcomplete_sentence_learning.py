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

rootPath = '/Users/codelife/Developer/11st_escrow2/pas_ml_sample'
ext = '.java'
# with tf.variable_scope(self.scopeName):
# rootPath = '/Users/codelife/Developer/tensorFlow/DeepLearningZeroToAll/tests'
# ext = '.py'
print('start learning..' , datetime.datetime.now())

sentence_pret  = wc.sentence_pret(rootPath)

allFiles = sentence_pret.getListOfFiles(rootPath,ext)
allFiles = allFiles[:10]
ana_docs,idx2word = sentence_pret.idx2word(allFiles)
word2idx = sentence_pret.word2idx(idx2word)

print('\n'.join(allFiles))


sequence = ana_docs
sequence_length = 10
# sequence = ("if you want to build a ship, don't drum up people together to "
#             "collect wood and don't assign them tasks and work, but rather "
#             "teach them to long for the endless immensity of the sea.")

dataX , dataY = sentence_pret.make_word_xy_list(sequence,sequence_length,word2idx)

data_dim    = len(idx2word)
hidden_size = len(idx2word)
num_classes = len(idx2word)     #hidden_size
learning_late = 0.1
keep_prob = 0.7

# dic 삭제(memore save)
with open("./setence_train_data/wordcomplete_sentence_idx2word.dump", "wb") as fp:
    pickle.dump(idx2word, fp)

with open("./setence_train_data/wordcomplete_sentence_word2idx.dump", "wb") as fp:
    pickle.dump(word2idx, fp)

idx2word = []
word2idx = []



train_mode = tf.placeholder(tf.bool, name='train_mode')
X = tf.placeholder(tf.int32,shape=[None,sequence_length],name='X_holder')
Y = tf.placeholder(tf.int32,shape=[None,sequence_length],name='Y_holder')
batch_size= tf.placeholder(tf.int32,[],name='batch_size')
X_ONE_HOT = tf.one_hot(X,num_classes)

def lstm_cell():
    cell = rnn.BasicLSTMCell(hidden_size, state_is_tuple=True)
    return cell


cell = rnn.BasicLSTMCell(num_units=hidden_size,state_is_tuple=True)
multi_cells = rnn.MultiRNNCell([lstm_cell() for _ in range(3)], state_is_tuple=True)

rnnOutput,_state = tf.nn.dynamic_rnn(multi_cells,X_ONE_HOT,dtype=tf.float32)

X_for_fc = tf.reshape(rnnOutput,shape=[-1,hidden_size])
#
# hidden_layer1 = fully_connected(X_for_fc, num_classes, scope="h1")
# h1_drop = dropout(hidden_layer1, keep_prob, is_training=train_mode)
# hidden_layer2 = fully_connected(h1_drop, num_classes, scope="h2")
# h2_drop = dropout(hidden_layer2, keep_prob, is_training=train_mode)
# hidden_layer3 = fully_connected(h2_drop, num_classes, scope="h3")
# h3_drop = dropout(hidden_layer3, keep_prob, is_training=train_mode)
# hidden_layer4 = fully_connected(h3_drop, num_classes, scope="h4")
# h4_drop = dropout(hidden_layer4, keep_prob, is_training=train_mode)
# connected = fully_connected(h4_drop,num_classes,activation_fn=None)

connected = fully_connected(X_for_fc,num_classes,activation_fn=None)

logits  = tf.reshape(connected,shape=[batch_size,sequence_length,num_classes],name='logits')

weights = tf.ones(shape=[batch_size,sequence_length],dtype=tf.float32)  #   최종적으로 나오는 값에 대한 비중 설정

loss = tf.reduce_mean(tf.contrib.seq2seq.sequence_loss(logits=logits,targets=Y,weights=weights ))
train = tf.train.AdamOptimizer(learning_rate=learning_late).minimize(loss,name='train_optimizer')

mean_loss = tf.reduce_mean(loss,name='mean_loss')


sess = tf.Session()
sess.run(tf.global_variables_initializer())

print('prepare learning..' , datetime.datetime.now())

for step in range(1500):
    mean_loss_val, _ = sess.run([mean_loss,train],{X:dataX,Y:dataY,train_mode:True,batch_size:len(dataX)})
    if step % 100 == 0 :
        print('learning..',step ,mean_loss_val, datetime.datetime.now())

#
# saver = tf.train.Saver()
# saver.save(sess,save_path='./setence_train_data/wordcomplete_sentence_train_data')
#
#
#
