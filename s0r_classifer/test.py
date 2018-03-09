import tensorflow as tf
import data
import network
import csv
import matplotlib.pyplot as plt
batch_size=128
filepath = "data/test.csv"
features,ids = data.read_data_test(filepath)
features = features.reshape(features.shape + (1,))
features = features-128.0/128.0
ids_test = ids[:50]

# x = tf.placeholder(tf.float32,(None,40,40,1))
# y = tf.placeholder(tf.int32,(None))


with tf.Session() as sess:
    saver = tf.train.import_meta_graph('model.meta')
    saver.restore(sess,'./model')
    graph = tf.get_default_graph()
    x = graph.get_tensor_by_name("x:0")
    predict = graph.get_tensor_by_name("predict:0")
    num_features = len(features)
    total_pridictions = []
    for offset in range(0,num_features,batch_size):
        end = offset + batch_size
        batch_test = features[offset:end]
        predictions = sess.run(predict,feed_dict={x:batch_test})
        total_pridictions += list(predictions)
    print(len(total_pridictions))
    results = [(ids[i],total_pridictions[i]) for i in range(len(total_pridictions))]

    with open('results.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["id","y"])
        writer.writerows(results)

    print(features[0].shape)
    plt.figure(figsize=(25,100))
    for i in range(50):
        plt.subplot(25,2,i+1)
        title = "id:%s-predict:%s"%(ids_test[i],total_pridictions[i])
        plt.title(title)
        plt.imshow(features[i].reshape(40,40),cmap="gray")