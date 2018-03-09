import data
from sklearn.model_selection import train_test_split
import tensorflow as tf
import network
from sklearn.utils import shuffle
import numpy as np

rate = 0.02
filepath = "data/train.csv"
EPOCHS = 500
batch_size = 1500
print("reading data......")
features ,labels = data.read_data(filepath)
features = features.reshape(features.shape + (1,))
features = features-128.0/128.0
# x_train, x_validation, y_trian, y_validation = train_test_split(features, labels)

x = tf.placeholder(tf.float32, (None, 40, 40, 1),name="x")
y = tf.placeholder(tf.int32, (None))
one_hot_y = tf.one_hot(y, 2)

logits = network.network(x)
loss = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_y, logits=logits)
train_operation = tf.train.AdamOptimizer(rate).minimize(loss)

correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
predict = tf.argmax(logits,1,name="predict")

saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for i in range(EPOCHS):
        print("EPOCH {} ...".format(i + 1))
        shuffle(features, labels)
        num_train = len(features)
        num_validation = len(features)
        #train
        print("trianning......")
        for offset in range(0,num_train,batch_size):
            end = offset + batch_size
            batch_x = features[offset:end]
            batch_y = labels[offset:end]
            sess.run(train_operation,feed_dict={x:batch_x, y:batch_y})
        #validation
        total_accuracy = 0.0
        for offset in range(0, num_validation, batch_size):
            end = offset + batch_size
            batch_x = features[offset:end]
            batch_y = labels[offset:end]
            accuracy = sess.run(accuracy_operation,feed_dict={x:batch_x,y:batch_y})
            total_accuracy += (accuracy*len(batch_x))
        valid_accuracy = total_accuracy/num_validation

        print("Validation Accuracy = {:.3f}".format(valid_accuracy))
        print()

    saver.save(sess, './model')
    print("Model saved")
