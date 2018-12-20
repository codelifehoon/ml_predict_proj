import os
import argparse
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


parser = argparse.ArgumentParser()

parser.add_argument(
    '--runtype',
    help='machine location',
    choices=['local', 'real'],
    default='real')
args, _ = parser.parse_known_args()

# Set C++ Graph Execution level verbosity(0:ALL)
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = 0


# np.seterr(divide='ignore', invalid='ignore')
df = None

if args.runtype == 'local':
    source_path  =os.path.dirname(os.path.realpath(__file__)) + '/data/google_ml_sample_data2.csv'
    df = pd.read_csv(source_path,dtype=np.float32)
else :
    with file_io.FileIO('gs://ml-codelife-20181219-data/train_data/google_ml_sample_data2.csv', 'r') as f:
        df = pd.read_csv(f,dtype=np.float32)

print(df.head())
data_x = df.iloc[:, 0:10]
data_y = df.iloc[:, 10]

data_x = data_x.values
org_data_y = data_y.values
data_y = data_y.values

scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(data_x)
data_x = scaler.transform(data_x)
max_y = 1 #np.max(data_y)
data_y =data_y / max_y

train_data_x = data_x[:700]
train_data_y = data_y[:700]
val_data_x = data_x[700:900]
val_data_y = data_y[700:900]
test_data_x = data_x[900:]
test_data_y = data_y[900:]

model = models.Sequential()
model.add(layers.Dense(10,activation='relu',input_shape=(data_x.shape[1],)))
model.add(layers.Dense(10,activation='relu'))
model.add(layers.Dense(10,activation='relu'))
model.add(layers.Dense(1))
model.compile(optimizer=optimizers.RMSprop(),loss=losses.mse,metrics=[metrics.mae])
model.fit(train_data_x,train_data_y,validation_data=(val_data_x,val_data_y),batch_size=10,epochs=300,verbose=0)

print(model.evaluate(test_data_x,test_data_y))
predcit_y = model.predict(test_data_x)
predcit_y = predcit_y * max_y
org_data_y = org_data_y[900:]
org_data_y = np.reshape(org_data_y,(len(org_data_y),1))

print(np.mean(predcit_y-org_data_y))
print(np.hstack([predcit_y,org_data_y]))