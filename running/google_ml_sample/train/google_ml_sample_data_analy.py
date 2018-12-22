import os
import argparse

import datetime
import tensorflow as tf
import numpy as np
import pandas as pd
import  keras
from keras import models, optimizers, losses, metrics
from keras import layers
from sklearn.preprocessing import MinMaxScaler
from google.cloud import storage
import logging
from tensorflow.python.lib.io import file_io

tf.set_random_seed(777)
logging.getLogger().setLevel(logging.DEBUG)
start_dt = datetime.datetime.now()





source_path  =os.path.dirname(os.path.realpath(__file__)) + '/data/google_ml_sample_data.csv'
df = pd.read_csv(source_path,dtype=np.float32)


print(df.head())
data_x = df.iloc[:, 0:10]
data_y = df.iloc[:, 10]

data_x = data_x.values
org_data_y = data_y.values
data_y = data_y.values

scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(data_x)
data_x = scaler.transform(data_x)
mean_y = 0 #np.mean(data_y)
data_y = data_y - mean_y

data_train_size = int(len(data_x) * 0.8)
data_val_size = int(len(data_x) * 0.1)
train_data_x = data_x[:data_train_size]
train_data_y = data_y[:data_train_size]
val_data_x = data_x[data_train_size:data_train_size+data_val_size]
val_data_y = data_y[data_train_size:data_train_size+data_val_size]
test_data_x = data_x[data_train_size+data_val_size:]
test_data_y = data_y[data_train_size+data_val_size:]
print(len(data_x),len(train_data_x),len(val_data_x),len(test_data_x))
print(np.mean(data_y),np.max(data_y))


data_x = np.arange(0,100).reshape((10,10))
print(data_x)
scaler.fit(data_x)
data_x = scaler.transform(data_x)
print('*'*100)
print(data_x)



