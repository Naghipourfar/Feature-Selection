import numpy as np
import tensorflow as tf
import pandas as pd
import random

import matplotlib.pyplot as plt
import os

"""
    Created by Mohsen Naghipourfar on 2/18/18.
    Email : mn7697np@gmail.com
"""
SHRINK_THRESHOLD = 2000

# Hyper-Parameters
LEARNING_RATE = 0.05
DROP_OUT = 0.5
N_SAMPLES = 10787
N_FEATURES = 19670
N_DISEASES = 34
N_BATCHES = 250
N_EPOCHS = 100
N_BATCH_LEARN = 5


def weight_initializer(shape, stddev=0.01, name=None):
    initial = tf.random_normal(shape=shape, stddev=stddev)
    return tf.Variable(initial, name=name)


def bias_initializer(shape, init_value=0.1, name=None):
    initial = tf.constant(init_value, shape=shape)
    return tf.Variable(initial, name=name)


def shrink_data(filename):
    import csv
    new_filename = filename + "_new.csv"
    filename += ".csv"
    with open(filename, 'r') as csv_file:
        raw_data = csv.reader(csv_file)  # (10787, 19671)
        with open(new_filename, 'w') as new_file:
            wr = csv.writer(new_file, quoting=csv.QUOTE_NONE)
            row_number = 0
            for row in raw_data:
                row_number += 1
                wr.writerow(row)
                if row_number > SHRINK_THRESHOLD:
                    break
    return new_filename


def load_data(filename):
    import csv
    with open(filename, 'r') as csv_file:
        raw_data = csv.reader(csv_file)
        training_data = list(raw_data)
        training_data = pd.DataFrame(training_data)
    return training_data


def fully_connected(input_data, weight, bias, name=None):
    return tf.add(tf.matmul(input_data, weight), bias, name=name)


def train(x_data, y_data, k):
    # Split data into train/test = 80%/20%
    train_indices = np.random.choice(N_SAMPLES, round(N_SAMPLES * 0.8), replace=False)
    validation_indices = np.array(list(set(range(N_SAMPLES)) - set(train_indices)))

    x_train = x_data.iloc[train_indices]
    y_train = y_data.iloc[train_indices]

    N_TRAIN_DATA = x_train.shape[0]

    x_validation = x_data.iloc[validation_indices]
    y_validation = y_data.iloc[validation_indices]

    # Create Network and Variables
    x = tf.placeholder(tf.float32, shape=[None, N_FEATURES])
    y = tf.placeholder(tf.float32, shape=[None, N_DISEASES])
    neurons = {
        'in': N_FEATURES,
        'l1': 64,
        'l2': 128,
        'l3': 256,
        'l4': 128,
        'out': N_DISEASES
    }
    weights = {
        'l1': weight_initializer(shape=[neurons['in'], neurons['l1']], stddev=0.1, name='w1'),
        'l2': weight_initializer(shape=[neurons['l1'], neurons['l2']], stddev=0.1, name='w2'),
        'l3': weight_initializer(shape=[neurons['l2'], neurons['l3']], stddev=0.1, name='w3'),
        'l4': weight_initializer(shape=[neurons['l3'], neurons['l4']], stddev=0.1, name='w4'),
        'out': weight_initializer(shape=[neurons['l4'], neurons['out']], stddev=0.1, name='w_out')

    }

    biases = {
        'l1': bias_initializer(init_value=0.1, shape=[neurons['l1']], name='b1'),
        'l2': bias_initializer(init_value=0.1, shape=[neurons['l2']], name='b2'),
        'l3': bias_initializer(init_value=0.1, shape=[neurons['l3']], name='b3'),
        'l4': bias_initializer(init_value=0.1, shape=[neurons['l4']], name='b4'),
        'out': bias_initializer(init_value=0.1, shape=[neurons['out']], name='b_out')

    }
    # 1st Layer --> Fully Connected (16 Neurons)
    layer_1 = fully_connected(x, weights['l1'], biases['l1'], name='l1')

    # 2nd Layer --> Fully Connected (32 Neurons)
    layer_2 = fully_connected(layer_1, weights['l2'], biases['l2'], name='l2')

    # 3rd Layer --> Fully Connected (64 Neurons)
    layer_3 = fully_connected(layer_2, weights['l3'], biases['l3'], name='l3')

    # 4th Layer --> Fully Connected (128 Neurons)
    layer_4 = fully_connected(layer_3, weights['l4'], biases['l4'], name='l4')

    # Final Layer --> Fully Connected (N_DISEASES Neurons)
    final_output = fully_connected(layer_4, weights['out'], biases['out'], name='l_out')

    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=final_output, labels=y), name='loss')
    optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE, name='optimizer')
    train_step = optimizer.minimize(loss, name='train_step')

    init = tf.global_variables_initializer()

    training_acc = []
    validation_acc = []
    training_loss = []
    validation_loss = []
    test_acc = []

    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(N_EPOCHS):
            # Train Network
            for i in range(N_BATCH_LEARN):
                batch_indices = np.random.choice(N_TRAIN_DATA, size=N_BATCHES)
                x_train_batch = x_train.iloc[batch_indices]
                y_train_batch = y_train.iloc[batch_indices]

                feed_dict = {x: x_train_batch, y: y_train_batch}
                _, train_loss = sess.run([train_step, loss], feed_dict=feed_dict)
                training_loss.append(train_loss)
                prediction = tf.nn.softmax(final_output)
                correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(y_train_batch, 1))
                accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
                training_acc.append(accuracy.eval(feed_dict))

                # Test Validation set
                feed_dict = {x: x_validation, y: y_validation}
                validation_loss.append(sess.run(loss, feed_dict=feed_dict))
                prediction = tf.nn.softmax(final_output)
                correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(y_validation, 1))
                accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
                validation_acc.append(accuracy.eval(feed_dict))
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print("Epoch:", '%04d' % (epoch + 1), "\tValidation Accuracy={:.9f}".format(validation_acc[-1]),
                      "\tValidation Loss={:4.4f}".format(validation_loss[-1]))
            # Training Validation set
            # feed_dict = {x: x_validation_batch, y: y_validation_batch}
            # sess.run(train_step, feed_dict=feed_dict)
        model_path = "/tmp/model.ckpt"
        saver.save(sess, model_path)
    print("Training Finished!")

    # Test Model
    # feed_dict = {x: x_test, y: y_test}
    # with tf.Session() as sess:
    #     prediction = tf.nn.softmax(final_output)
    #     correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(y_test, 1))
    #     accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    #     sess.run(accuracy, feed_dict={x: x_test, y: y_test})

    # Plot accuracy over time
    import matplotlib.pyplot as plt
    plt.plot(training_acc, 'k-', label='Training Accuracy')
    plt.plot(validation_acc, 'b--', label='Validation Accuracy')
    plt.ylim(ymax=1.0, ymin=0.0)
    plt.title('Accuracy per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc="upper right")
    os.makedirs('./model_{0}_{1}'.format(k, validation_acc[-1]))
    plt.savefig('./model_{0}_{1}/accuracy.png'.format(k, validation_acc[-1]))
    plt.close()

    # Plot loss over time
    plt.plot(training_loss, 'k-', label='Training Loss')
    plt.plot(validation_loss, 'b--', label='Validation Loss')
    plt.title('cross_entropy Loss per Epoch (k = {0})'.format(k))
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc="upper right")
    plt.savefig('./model_{0}_{1}/loss.png'.format(k, validation_acc[-1]))


def modify_output(target):
    global N_DISEASES
    output_dict = {}
    a = target[0].value_counts()
    print(a)
    a = pd.DataFrame(a)
    i = 0
    for row in a.itertuples():
        output_dict[row[0]] = i
        i += 1
    N_DISEASES = i
    new_output = [[0 for _ in range(N_DISEASES)] for __ in range(N_SAMPLES)]
    for idx, y in target.iterrows():
        new_output[idx][output_dict[y[0]]] = 1
    return new_output


def random_train(k):
    global N_FEATURES, x_train, y_train
    print("k = {0}".format(k))
    random_feature_indices = np.random.choice(N_FEATURES, k)
    x_train = x_train.iloc[:, random_feature_indices]
    N_FEATURES = k
    train(x_train, y_train, k)


if __name__ == '__main__':
    x_filename = "../Data/fpkm_normalized.csv"
    y_filename = "../Data/disease.csv"
    # if not os.path.isfile(x_filename) or not os.path.isfile(y_filename):
    #     x_filename = shrink_data("../Data/fpkm_normalized")
    #     y_filename = shrink_data("../Data/disease")
    print("Loading data...")
    x_train, y_train = load_data(x_filename), load_data(y_filename)
    print("Data has been loaded successfully!")
    N_SAMPLES, N_FEATURES = x_train.shape
    y_train = modify_output(y_train)
    y_train = pd.DataFrame(y_train)
    print("Training neural network!")
    from multiprocessing import Pool
    N_PROCESSES = 1
    with Pool(N_PROCESSES) as p:
        p.map(random_train, [35+i for i in range(N_PROCESSES)])

    # for k in range(25, 35):
    #     print("k = {0}".format(k))
    #     random_feature_indices = np.random.choice(N_FEATURES, k)
    #     x_train = x_train.iloc[:, random_feature_indices]
    #     N_FEATURES = k
    #     train(x_train, y_train, k)
