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
real_traning_path = 'gs://ml-codelife-20181031/char_train_data'


parser = argparse.ArgumentParser()

parser.add_argument(
    '--runtype',
    help='running location',
    choices=['local', 'real'],
    default='real')

parser.add_argument(
    '--isgraph',
    help='isgraph y/N',
    choices=['Y', 'N'],
    default='N')





parser.add_argument(
    '--verbosity',
    choices=['DEBUG', 'ERROR', 'FATAL', 'INFO', 'WARN'],
    default='DEBUG')

args, _ = parser.parse_known_args()

tf.logging.set_verbosity(args.verbosity)
log_level = str(tf.logging.__dict__[args.verbosity] / 10)
isgraph  = args.isgraph

# Set C++ Graph Execution level verbosity
os.environ['TF_CPP_MIN_LOG_LEVEL'] = log_level



idx2word = None
word2idx = None
dataX = None
dataY = None
data_dim    = 0
hidden_size = 0
num_classes = 0
learning_late = 0.001
sequence_length = 17
slice_per_size  = 50   # 한 슬라이스에 몇개의 row일지
slice_size = 0           # 전체 슬라이스가 몇개인지
keep_prob = 1
recursion_learning_cnt =  101    ## 반복 학습 횟수



if args.runtype == 'local':
    # global idx2word,word2idx,dataX,dataY
    with open(local_traning_path + '/idx2word.dump', "rb") as fp:
        idx2word = pickle.load(fp)

    with open(local_traning_path + '/word2idx.dump', "rb") as fp:
        word2idx = pickle.load(fp)

    with open(local_traning_path + '/dataX.dump', "rb") as fp:
        dataX = pickle.load(fp)

    with open(local_traning_path + '/dataY.dump' , "rb") as fp:
        dataY = pickle.load(fp)
else:
    # global idx2word,word2idx,dataX,dataY

    client = storage.Client()
    bucket = client.get_bucket('ml-codelife-20181031')

    dataX   = bucket.get_blob('data/dataX.dump')
    dataY   = bucket.get_blob('data/dataY.dump')
    word2idx = bucket.get_blob('data/word2idx.dump')
    idx2word = bucket.get_blob('data/idx2word.dump')

    print(bucket)
    print(dataX)


    dataX = pickle.loads(dataX.download_as_string())
    dataY = pickle.loads(dataY.download_as_string())
    word2idx = pickle.loads(word2idx.download_as_string())
    idx2word = pickle.loads(idx2word.download_as_string())


# 정상적으로 읽었는지 확인
print(dataX[:10])



dataX_len = len(dataX)
slice_size =  int(dataX_len / slice_per_size)
data_dim    = len(idx2word)
hidden_size = len(idx2word)
num_classes = len(idx2word)     #hidden_size


train_mode = tf.placeholder(tf.bool, name='train_mode')
X = tf.placeholder(tf.int32,shape=[None,sequence_length],name='X_holder')
Y = tf.placeholder(tf.int32,shape=[None,sequence_length],name='Y_holder')
BATCH_SIZE= tf.placeholder(tf.int32, shape=[], name='batch_size')
X_ONE_HOT = tf.one_hot(X,num_classes)


# print(X_ONE_HOT)


def lstm_cell():
    cell = rnn.BasicLSTMCell(hidden_size, state_is_tuple=True)
    return tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=1)


multi_cells = rnn.MultiRNNCell([lstm_cell() for _ in range(2)], state_is_tuple=True)

initial_state = multi_cells.zero_state(BATCH_SIZE, tf.float32)
rnnOutput,final_state = tf.nn.dynamic_rnn(multi_cells,X_ONE_HOT,initial_state=initial_state,dtype=tf.float32)


X_for_fc = tf.reshape(rnnOutput,shape=[-1,hidden_size])


connected = fully_connected(X_for_fc,num_classes,activation_fn=None)

logits  = tf.reshape(connected, shape=[BATCH_SIZE, sequence_length, num_classes], name='logits')

weights = tf.ones(shape=[BATCH_SIZE, sequence_length], dtype=tf.float32)  #   최종적으로 나오는 값에 대한 비중 설정

loss = tf.reduce_mean(tf.contrib.seq2seq.sequence_loss(logits=logits,targets=Y,weights=weights ))
train = tf.train.AdamOptimizer(learning_rate=learning_late).minimize(loss,name='train_optimizer')
mean_loss = tf.reduce_mean(loss,name='mean_loss')



# data batch 처리
dataset = tf.data.Dataset.from_tensor_slices((dataX, dataY))
dataset = dataset.batch(slice_per_size)

iterator = dataset.make_initializable_iterator()
f, l = iterator.get_next()



# sess = tf.Session()
sess =tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True)))
sess.run(tf.global_variables_initializer())

# tensor board graph_writer
graph_writer = None
summary_mean_loss  = tf.summary.histogram("mean_loss",mean_loss)
summary_merged = tf.summary.merge_all()

if args.runtype == 'local':
    graph_writer = tf.summary.FileWriter('/Users/codelife/Developer/python_proj/ml_predict_proj/running/wordcomplete/graph', sess.graph)
else:
    graph_writer = tf.summary.FileWriter(real_traning_path + '/graph', sess.graph)



logging.info('prepare learning..{}'.format(datetime.datetime.now()))




for step in range(recursion_learning_cnt):
    sess.run(iterator.initializer)
    state = sess.run(initial_state,feed_dict={BATCH_SIZE:slice_per_size})

    for i in range(slice_size):

        x, y = sess.run([f, l])
        feed_dict = {X:x, Y:y, BATCH_SIZE:slice_per_size}

        #mini batch로 전체의 자료를 분리해서 학습할때에는 state를 하나의 전체 학습 내에서는 전파 될 수 있도록 설정한다.
        for i, (c, h) in enumerate(initial_state):
            feed_dict[c] = state[i].c
            feed_dict[h] = state[i].h

        # state , summary,mean_loss_val, _  = sess.run([final_state,summary_merged,mean_loss,train], feed_dict=feed_dict)
        summary,mean_loss_val, state, _  = sess.run([summary_merged,mean_loss,final_state,train], feed_dict=feed_dict)



    if step % 1 == 0 :
        graph_writer.add_summary(summary,step)
        logging.info('learning.. {} {} {}'.format(step ,mean_loss_val, datetime.datetime.now()))


    if step % 50 == 0 :
        if args.runtype == 'local':
            saver = tf.train.Saver()
            saver.save(sess,save_path=local_traning_path + '/wordcomplete_word_train_data',global_step=step)
        else:
            saver = tf.train.Saver()
            saver.save(sess,save_path=real_traning_path + '/',global_step=step)



if args.runtype == 'local':
    saver = tf.train.Saver()
    saver.save(sess,save_path=local_traning_path + '/wordcomplete_word_train_data')
else:
    saver = tf.train.Saver()
    saver.save(sess,save_path=real_traning_path + '/')


