# -*- coding: utf-8 -*-
"""
Created on Tue Mar  6 14:57:53 2018

@author: yang
"""

import tensorflow as tf
from tensorflow.contrib.layers import flatten


def weight_variable(shape, stddev=0.1):
    initial = tf.truncated_normal(shape, stddev=stddev)
    return tf.Variable(initial)


def bias_variable(shape, bais=0.1):
    initial = tf.constant(bais, shape=shape)
    return tf.Variable(initial)


def conv2d(x, w):
    return tf.nn.conv2d(x, w, [1, 1, 1, 1], 'SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')


def max_pool_3x3(x):
    return tf.nn.max_pool(x, [1, 3, 3, 1], [1, 2, 2, 1], 'SAME')


def avg_pool_3x3(x):
    return tf.nn.avg_pool(x, [1, 3, 3, 1], [1, 2, 2, 1], 'SAME')

def network(features):
    conv1_w = weight_variable([5,5,1,6])
    conv1_b = bias_variable([6])
    conv1 = tf.nn.relu(conv2d(features,conv1_w) + conv1_b)
    max_pol1 = max_pool_2x2(conv1)
    
    conv2_w = weight_variable([5,5,6,16])
    conv2_b = bias_variable([16])
    conv2 = tf.nn.relu(conv2d(max_pol1,conv2_w) + conv2_b)
    max_pol2 = max_pool_2x2(conv2)
    
    flat = flatten(max_pol2)
    f_drop = tf.nn.dropout(flat,keep_prob=0.5)
    fc1_w = weight_variable([1600,120])
    fc1_b = bias_variable([120])
    fc1 = tf.nn.relu(tf.matmul(f_drop,fc1_w)+fc1_b)
    
    fc2_w = weight_variable([120,60])
    fc2_b = bias_variable([60])
    fc2 = tf.nn.relu(tf.matmul(fc1,fc2_w)+fc2_b)
    
    fc3_w = weight_variable([60,2])
    fc3_b = bias_variable([2])
    logits = tf.matmul(fc2,fc3_w)+fc3_b
    
    return logits


    
    
     
    
    