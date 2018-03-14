import tensorflow as tf
import numpy as np
import data
import network
import csv

filepath = "./data/test.csv"
data_load = data.load_data(filepath)
features,ids = data.preprocess_test(data_load)

with tf.Session() as sess:
    saver = tf.train.import_meta_graph("./checkpoints/model-9000.meta")
    ckpt = tf.train.get_checkpoint_state("./checkpoints/")
    saver.restore(sess, ckpt.model_checkpoint_path)

    graph = tf.get_default_graph()
    x = graph.get_tensor_by_name("x:0")
    pred = graph.get_tensor_by_name("pred:0")

    predictions = sess.run(pred,feed_dict={x:features})
    results = [(ids[i],predictions[i][0]) for i in range(len(predictions))]
    with open('results.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["id", "y"])
        writer.writerows(results)
