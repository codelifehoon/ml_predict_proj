import  tensorflow as tf
import numpy as np

tf.InteractiveSession()

params = tf.constant([10,20,30,40])
ids = tf.constant([1,1,2,3])
print(tf.nn.embedding_lookup(params,ids).eval())


params1 = tf.constant([1,2])
params2 = tf.constant([10,20])
ids = tf.constant([2,0,2,1,2,3])
result = tf.nn.embedding_lookup([params1, params2], ids).eval()
print(result)


#  배열 embedding
embeddings = tf.constant\
    ([
        [
            [1,1]
            ,[2,2]
            ,[3,3]
            ,[4,4]
        ]
        ,[
            [11,11]
            ,[12,12]
            ,[13,13]
            ,[14,14]
        ]
        ,[
            [21,21]
            ,[22,22]
            ,[23,23]
            ,[24,24]
        ]
    ])  # 3x4x2

print(np.shape(embeddings))

ids=tf.constant([0,2,1])
embed = tf.nn.embedding_lookup(embeddings, ids, partition_strategy='div')

with tf.Session() as session:
    result = session.run(embed)
    print (result)





#  배열 embedding2
embeddings = tf.constant \
        ([
        [1,1,2,2]
        ,[11,11,12,12]
        ,[21,21,22,22]
    ])  # 3x4

ids=tf.constant(
    [
        [0,2]
        ,[1,2]
    ])
print(np.shape(embeddings))
print(np.shape(ids))


embed = tf.nn.embedding_lookup(embeddings, ids, partition_strategy='div')



with tf.Session() as session:
    result = session.run(embed)
    print (result)

#
# [[[ 1  1  2  2]
#   [21 21 22 22]]
#
# [[11 11 12 12]
# [21 21 22 22]]]
#
#
#
# [[[ 1  1  2  2]
#   [21 21 22 22]]
#
# [[11 11 12 12]
# [21 21 22 22]]]

