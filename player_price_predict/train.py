import tensorflow as tf
import numpy as np
import data
import network
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

rate = 0.01
batch_size = 1000
train_step = 10000
filepath = "./data/train.csv"

data_load = data.load_data(filepath)
features,labels = data.preprocess(data_load)
labels = np.reshape(labels,(-1,1))
# features_train,features_validate,labels_train,labels_validate = train_test_split(features,labels,test_size=0.3)

x = tf.placeholder(tf.float32,[None,57],name="x")
y = tf.placeholder(tf.float32,[None,1])

pred = network.network(x)
loss = tf.reduce_mean(tf.abs(y-pred))
train_opration = tf.train.AdamOptimizer(rate).minimize(loss)
saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(train_step):
        features_train,labels_train = shuffle(features,labels)
        sess.run(train_opration,feed_dict={x:features_train,y:labels_train})
        if i%100==0:
            train_loss = sess.run(loss,feed_dict={x:features_train,y:labels_train})
            # validate_loss = sess.run(loss,feed_dict={x:features_validate,y:labels_validate})
            print("train step %s: training loss is %s, validate loss is %s" % (i,train_loss,0))

    # prediction = sess.run(pred,feed_dict={x:features_validate,y:labels_validate})
    # print(prediction)
        if i%1000==0 and i != 0:
            saver.save(sess, './checkpoints/model',global_step=i)
            print("model saved at step %s!"%i)
