import numpy as np
import  os
from konlpy.tag import  Okt
import sys
import datetime
import re
import string

class consts:
    def __init__(self,type='sentence'):
        if type == 'sentence':
            self.checkpoint_path= '/Users/codelife/Developer/python_proj/ml_predict_proj/running/wordcomplete/setence_train_data'
            self.idx2word_path  = self.checkpoint_path + '/wordcomplete_sentence_idx2word.dump'
            self.word2idx_path  = self.checkpoint_path + '/wordcomplete_sentence_word2idx.dump'
            self.meta_graph_path= self.checkpoint_path + '/wordcomplete_sentence_train_data.meta'
        else:
            self.checkpoint_path= '/Users/codelife/Developer/python_proj/ml_predict_proj/data/char_train_data'
            self.idx2word_path  = self.checkpoint_path + '/idx2word.dump'
            self.word2idx_path  = self.checkpoint_path + '/word2idx.dump'
            self.xdata_path  = self.checkpoint_path + '/dataX.dump'
            self.ydata_path  = self.checkpoint_path + '/dataY.dump'
            self.meta_graph_path= self.checkpoint_path + '/wordcomplete_word_train_data-50.meta'

class sentence_model:
    def __init__(self,saver,idx2word,word2idx,sequence_length):
        self.sentence_after = sentence_after()

        self.idx2word = idx2word
        self.word2idx = word2idx
        self.saver = saver
        self.sequence_length = sequence_length
        self.X  = None
        self.batch_size  = None
        self.logits= None
        self.mean_loss  = None
        self.train_mode = None
        self.train  = None

        self.init_model()

        print('__init__sentence_model')


    def init_model(self):
        print('init_model')


    # def predict_sentence(self,sentence,next_step):
    #
    #
    #     sequence_next_step = 10 +next_step
    #
    #     predict_next = sentence_pret('').tokenizer(sentence)
    #
    #     # 예측 가능한 초기 셋이 10개 이상 되는지 확인
    #     dataX = self.sentence_after.make_word_x_list(predict_next, self.learning_step, self.word2idx)
    #     # print(dataX)
    #     dataX_len = np.reshape(dataX,-1)
    #     # print(dataX_len)
    #     if len(dataX_len) < 10 :
    #         return ['the sentence should be  10 words or more.']
    #
    #     next_sentence = []
    #
    #     # tf.reset_default_graph()
    #
    #
    #
    #     with tf.Session() as sess:
    #         sess.run(tf.global_variables_initializer())
    #
    #         self.saver.restore(sess, tf.train.latest_checkpoint(consts().checkpoint_path))
    #         graph = tf.get_default_graph()
    #
    #         # for op in graph.get_operations():
    #         #     print(op.name)
    #
    #         # self.X  = graph.get_tensor_by_name('X_holder:0')
    #         # self.batch_size  = graph.get_tensor_by_name('batch_size:0')
    #         # self.logits= graph.get_tensor_by_name('logits:0')
    #         # self.mean_loss  = graph.get_tensor_by_name('mean_loss:0')
    #         # self.train_mode = graph.get_tensor_by_name('train_mode:0')
    #         # self.train  = graph.get_operation_by_name('train_optimizer')
    #
    #         X  = graph.get_tensor_by_name('X_holder:0')
    #         batch_size  = graph.get_tensor_by_name('batch_size:0')
    #         logits= graph.get_tensor_by_name('logits:0')
    #         mean_loss  = graph.get_tensor_by_name('mean_loss:0')
    #         train_mode = graph.get_tensor_by_name('train_mode:0')
    #         train  = graph.get_operation_by_name('train_optimizer')
    #
    #
    #         for i in range(0, sequence_next_step):
    #
    #             predict_next  = predict_next + next_sentence
    #             dataX =  self.sentence_after.make_word_x_list(predict_next, self.learning_step, self.word2idx)
    #
    #             results = sess.run(logits, feed_dict={X: dataX, batch_size:len(dataX)})
    #             result = results[-1]
    #             index = np.argmax(result, axis=1)
    #             next_sentence = [self.idx2word[index[-1]]]
    #
    #     return ''.join(predict_next+next_sentence)


    def predict_sentence2(self,sentence,next_step,sess,graph):

        sequence_next_step = next_step

        predict_next = sentence_pret('').tokenizer(sentence)

        # 예측 가능한 초기 셋이 10개 이상 되는지 확인
        print('step2:',datetime.datetime.now())
        dataX = self.sentence_after.make_word_x_list(predict_next, self.sequence_length, self.word2idx)
        # print(dataX)
        dataX_len = np.reshape(dataX,-1)
        # print(dataX_len)
        if len(dataX_len) < 10 :
            return ['the sentence should be  10 words or more.']

        next_sentence = []
        print('step3:',datetime.datetime.now())
        X  = graph.get_tensor_by_name('X_holder:0')
        batch_size  = graph.get_tensor_by_name('batch_size:0')
        logits= graph.get_tensor_by_name('logits:0')
        print('step4:',datetime.datetime.now())

        for i in range(0, sequence_next_step):

            predict_next  = predict_next + next_sentence
            dataX =  self.sentence_after.make_word_x_list(predict_next, self.sequence_length, self.word2idx)

            results = sess.run(logits, feed_dict={X: dataX, batch_size:len(dataX)})
            result = results[-1]
            index = np.argmax(result, axis=1)
            next_sentence = [self.idx2word[index[-1]]]

        print('step5:',datetime.datetime.now())


        return ''.join(predict_next+next_sentence)

    def predict_sentence_char(self, sentence, sequence_length, next_step, sess, graph):


        if next_step == 0:
            sequence_next_step = 50
        else:
            sequence_next_step = next_step
        predict_next= sentence
        # 예측 가능한 초기 셋이 10개 이상 되는지 확인
        print('step2:',datetime.datetime.now())
        # dataX = self.sentence_after.make_word_x_list(sentence, self.learning_step, self.word2idx)

        if len(sentence) < sequence_length :
            return 'the sentence should be  {} words or more.'.format(sequence_length)

        next_sentence = []
        print('step3:',datetime.datetime.now())
        X  = graph.get_tensor_by_name('X_holder:0')
        batch_size  = graph.get_tensor_by_name('batch_size:0')
        logits= graph.get_tensor_by_name('logits:0')

        print('step4:',datetime.datetime.now())


        for i in range(0, sequence_next_step):

            dataX =  self.sentence_after.make_word_x_list(predict_next, self.sequence_length, self.word2idx)

            results = sess.run(logits, feed_dict={X: dataX, batch_size:len(dataX)})
            result = results[-1]

            index = np.argmax(result, axis=1)
            predict_char = self.idx2word[index[-1]]
            predict_next = predict_next + predict_char
            next_sentence.append(predict_char)

            # 문단단위로 완성을 요청하면
            if next_step == 0 and predict_char in [' ','\n','\t']:
                break

            # for i in index:
            #     print(self.idx2word[i],end='')


        return ''.join(predict_next)



class sentence_after:
    def __init__(self):
        print('__init__sentence_after')
    # 질문만 할떄만
    def make_word_x_list(self,sequence,learning_step,word2idx):
        cnt = len(sequence) - learning_step
        listx = []

        for i in range(0,cnt+1):
            x_str = sequence[i:i + learning_step]
            try:
                x =  [word2idx[c] for c in x_str]
                listx.append(x)
            except:
                print(" error:", sys.exc_info())

        return listx




class sentence_pret :
    def __init__(self,root_path):
        self.root_path = root_path
        self.konlpymodel = Okt()
        print('__init__sentence_pret')

    def getListOfFiles(self,dirName,ext):
        # create a list of file and sub directories
        # names in the given directory
        listOfFile = os.listdir(dirName)
        allFiles = list()
        # Iterate over all the entries
        for entry in listOfFile:
            # Create full path
            fullPath = os.path.join(dirName, entry)
            # If entry is a directory then get the list of files in this directory
            if os.path.isdir(fullPath):
                allFiles = allFiles + self.getListOfFiles(fullPath,ext)
            else:
                filename, file_extension = os.path.splitext(fullPath)
                if file_extension in [ext]:        # python file만 추가
                    allFiles.append(fullPath)

        return allFiles


    def tokenizer(self,document):
        words = []
        for word, tag in self.konlpymodel.pos(document):
            words.append(word)
        return words

    # 단어 기반 토큰
    def idx2word_by_char(self, allFiles):

        docs = []
        index_lists = []
        for file_name in allFiles:
            with open(file_name,'r', encoding='utf-8',errors='ignore') as r:
                document = ''.join(r.readlines())
                document = self.comment_korea_remove(document)
                document = self.comment_remover(document)    ## java comment remove

                index_lists = list(set(''.join(index_lists)+document))
                docs.append(document)
        return docs,index_lists

    # 문장 기반 토큰
    def idx2word(self, allFiles):

        docs = []
        index_lists = []
        for file_name in allFiles:
            with open(file_name,'r', encoding='utf-8',errors='ignore') as r:
                document = ''.join(r.readlines())

                words = self.tokenizer(document)

                index_lists = list(set(index_lists+words))
                docs.append(words)
        return docs,index_lists

    def word2idx(self, idx2word):
        return {w:i for i ,w in enumerate(idx2word)}


    #질문/응답 set
    def make_word_xy_list(self,sequences,learning_step,word2idx):

        listx = []
        listy = []

        for sequence in sequences:
            cnt = len(sequence) - learning_step

            for i in range(0,cnt):
                x_str = sequence[i:i + learning_step]
                y_str = sequence[i + 1: i + 1 + learning_step]
                # print(i,x_str , '->', y_str)

                x =  [word2idx[c] for c in x_str]
                y =  [word2idx[c] for c in y_str]

                listx.append(x)
                listy.append(y)

        return listx,listy


    def comment_remover(self,text):
        def replacer(match):
            s = match.group(0)
            if s.startswith('/'):
                return " " # note: a space and not an empty string
            else:
                return s
        pattern = re.compile(
            r'//.*?$|/\*.*?\*/|\'(?:\\.|[^\\\'])*\'|"(?:\\.|[^\\"])*"',
            re.DOTALL | re.MULTILINE
        )
        text = re.sub(pattern, replacer, text)

        return re.sub('  +',' ',text)



    def comment_korea_remove(self, text):

        ret = ''
        for c in text :
            if c in string.printable:
                ret = ret + c

        return ret

