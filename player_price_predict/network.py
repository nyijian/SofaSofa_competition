import tensorflow as tf
from tensorflow.contrib.layers import flatten


def weight_variable(shape, stddev=0.1):
    initial = tf.truncated_normal(shape, stddev=stddev)
    return tf.Variable(initial)


def bias_variable(shape, bais=0.1):
    initial = tf.constant(bais, shape=shape)
    return tf.Variable(initial)

def network(x):
    fc1_w = weight_variable([57,500])
    fc1_b = bias_variable([500])
    fc1 = tf.nn.relu(tf.matmul(x,fc1_w)+fc1_b)
    # fc1_drop = tf.nn.dropout(fc1,keep_prob=0.5)

    fc2_w = weight_variable([500, 1000])
    fc2_b = bias_variable([1000])
    fc2 = tf.nn.relu(tf.matmul(fc1, fc2_w) + fc2_b)
    fc2_drop = tf.nn.dropout(fc2, keep_prob=0.5)

    fc2_w_ = weight_variable([1000,500])
    fc2_b_ = bias_variable([500])
    fc2_ = tf.nn.relu(tf.matmul(fc2_drop,fc2_w_)+fc2_b_)
    # fc2_drop = tf.nn.dropout(fc2, keep_prob=0.5)

    fc3_w = weight_variable([500,250])
    fc3_b = bias_variable([250])
    fc3 = tf.nn.relu(tf.matmul(fc2_,fc3_w)+fc3_b)
    # fc3_drop = tf.nn.dropout(fc3, keep_prob=0.5)

    fc4_w = weight_variable([250,100])
    fc4_b = bias_variable([100])
    fc4 = tf.nn.relu(tf.matmul(fc3,fc4_w)+fc4_b)
    # fc4_drop = tf.nn.dropout(fc4, keep_prob=0.5)

    fc5_w = weight_variable([100,50])
    fc5_b = bias_variable([50])
    fc5 = tf.nn.relu(tf.matmul(fc4,fc5_w)+fc5_b)

    fc6_w = weight_variable([50,25])
    fc6_b = bias_variable([25])
    fc6 = tf.nn.relu(tf.matmul(fc5,fc6_w)+fc6_b)

    fc7_w = weight_variable([25,10])
    fc7_b = bias_variable([10])
    fc7 = tf.nn.relu(tf.matmul(fc6,fc7_w)+fc7_b)

    fc8_w = weight_variable([10,1])
    fc8_b = bias_variable([1])
    pred = tf.add(tf.matmul(fc7,fc8_w),fc8_b,name="pred")

    return pred