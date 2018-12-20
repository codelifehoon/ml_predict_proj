import os
import pickle
import tensorflow as tf
from google.cloud import storage
from model import wordcomplete_model as wc_model


datapath = '/Users/codelife/Developer/python_proj/ml_predict_proj/data'

def _get_blob(filename):
    MODEL_BUCKET = os.environ['MODEL_BUCKET']
    client = storage.Client(project='codleife-ml-1101')
    bucket = client.get_bucket(MODEL_BUCKET)
    blob = bucket.get_blob(filename)


    return  blob.download_as_string()

def _load_wordanalysis_model_real():

    tmp = _get_blob('data/wordanalysis/naive-bayes-multinomial_11st_opt2.pickle')
    return  pickle.loads(_get_blob('data/wordanalysis/naive-bayes-multinomial_11st_opt2.pickle'))




def _load_charcomplete_model_real():

    idx2word = None
    word2idx = None
    wordcomplete_saver=None
    sess=None
    graph = None

    idx2word = pickle.loads(_get_blob('data/char_train_data/idx2word.dump'))  # pickle.loads vs pickle.load 다름 loads는  string형태에서 읽는거고 load는 binary에서..
    word2idx = pickle.loads(_get_blob('data/char_train_data/word2idx.dump'))

    # init wordcomplete sent ence learning data
    tf.reset_default_graph()
    wordcomplete_saver = tf.train.import_meta_graph('gs://' + os.environ['MODEL_BUCKET'] + '/data/char_train_data/wordcomplete_word_train_data-50.meta')
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    checkpointpath = 'gs://' +  os.environ['MODEL_BUCKET'] + '/data/char_train_data/wordcomplete_word_train_data-50'
    wordcomplete_saver.restore(sess, checkpointpath)
    graph = tf.get_default_graph()

    return idx2word,word2idx,wordcomplete_saver,sess,graph



def _load_cate_suggest_model_real():
    ctv = None
    clf = None
    y_classes = None
    y_classes_name = None

    ctv = pickle.loads(_get_blob('data/cate_analysis/cate_suggest_ctv.dump'))
    clf = pickle.loads(_get_blob('data/cate_analysis/cate_suggest_clf.dump'))
    y_classes = pickle.loads(_get_blob('data/cate_analysis/cate_suggest_y_classes.dump'))
    y_classes_name = pickle.loads(_get_blob('data/cate_analysis/cate_suggest_ylist.dump'))

    return ctv,  clf, y_classes,y_classes_name


def _load_wordanalysis_model_local():
    pipecls = None

    with open(datapath + '/wordanalysis/naive-bayes-multinomial_11st_opt2.pickle' , 'rb') as f:
        pipecls = pickle.load(f)
    return pipecls


def _load_charcomplete_model_local():
    idx2word = None
    word2idx = None
    wordcomplete_saver=None
    sess=None
    graph = None


    with open(datapath + '/char_train_data/idx2word.dump', "rb") as fp:
        idx2word = pickle.load(fp)

    with open(datapath + '/char_train_data/word2idx.dump' , "rb") as fp:
        word2idx = pickle.load(fp)

    # init wordcomplete sent ence learning data
    tf.reset_default_graph()
    # wordcomplete_saver = tf.train.Saver()
    wordcomplete_saver = tf.train.import_meta_graph(wc_model.consts('char').meta_graph_path)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    checkpoint_path = tf.train.latest_checkpoint(wc_model.consts('char').checkpoint_path)

    wordcomplete_saver.restore(sess, tf.train.latest_checkpoint(wc_model.consts('char').checkpoint_path))
    graph = tf.get_default_graph()

    return idx2word,word2idx,wordcomplete_saver,sess,graph



def _load_cate_suggest_model_local():
    ctv = None
    clf = None
    y_classes = None
    y_name = None

    with open(datapath + '/cate_analysis/cate_suggest_ctv.dump', 'rb') as f:
        ctv = pickle.load(f)

    with open(datapath + '/cate_analysis/cate_suggest_clf.dump', 'rb') as f:
        clf = pickle.load(f)

    with open(datapath + '/cate_analysis/cate_suggest_y_classes.dump', 'rb') as f:
        y_classes = pickle.load(f)

    with open(datapath + '/cate_analysis/cate_suggest_ylist.dump', 'rb') as f:
        y_name = pickle.load(f)


    return ctv,  clf, y_classes,y_name
