import sys
import numpy as np
import pandas as pd
import itertools

import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn import metrics

import warnings
warnings.filterwarnings('ignore')

sys.setrecursionlimit(10000000)

"""
    Created by Mohsen Naghipourfar on 3/1/18.
    Email : mn7697np@gmail.com
"""

# Constants
LOCAL_LOCATION_X = "../Data/fpkm_normalized.csv"
LOCAL_LOCATION_Y = "../Data/optimal_encoded_scae_dropout.csv"
DAMAVAND_LOCATION_X = "~/f/Behrooz/dataset_local/fpkm_normalized.csv"
DAMAVAND_LOCATION_Y = "~/f/Behrooz/dataset_local/optimal_encoded_scae_dropout.csv"

# Hyper-Parameters
LEARNING_RATE = 1e-3
DROP_OUT = 0.5
N_SAMPLES = 10787
N_FEATURES = 19671
N_DISEASES = 34
N_BATCHES = 2000
N_EPOCHS = 100
N_BATCH_LEARN = 10


def weight_initializer(shape, stddev=0.01, name=None):
    initial = tf.random_normal(shape=shape, stddev=stddev)
    return tf.Variable(initial, name=name)


def bias_initializer(shape, init_value=0.1, name=None):
    initial = tf.constant(init_value, shape=shape)
    return tf.Variable(initial, name=name)


def load_data(filename):  # TODO search for faster way to load data
    return pd.read_csv(filename, header=None)


def fully_connected(input_data, weight, bias, name=None):
    return tf.nn.relu(tf.add(tf.matmul(input_data, weight), bias, name=name))


def drop_out(prev_output, keep_prob):
    return tf.nn.dropout(prev_output, keep_prob)


def train(x_data, y_data, y_col, num_epochs=10000, steps=50000, batch_size=2500):
    # Split data into train/test = 80%/20%
    X_train, X_test, Y_train, Y_test = train_test_split(x_data, y_data, test_size=0.20)
    print(X_train.shape)

    # training_size = X_train.shape[0]

    # Create Network and Variables
    with tf.Graph().as_default():
        feature_columns = [tf.feature_column.numeric_column('x', shape=X_train.shape[1:])]
        regressor = tf.contrib.learn.DNNRegressor(feature_columns=feature_columns,
                                                  activation_fn=tf.nn.relu, hidden_units=[1024, 512, 64])

        def input_fn(x, y=None):
            if y is not None:  # Training
                features = {k: tf.constant(x[k].values) for k in x.columns}
                responses = tf.constant(y.values, shape=y.shape)
                return features, responses
            else:  # Testing
                features = {k: tf.constant(x[k].values) for k in x_data.columns}
                return features

        # Training...
        train_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={'x': X_train.values}, y=Y_train.values, batch_size=batch_size, num_epochs=num_epochs, shuffle=True)
        print("Training...")
        regressor.fit(input_fn=train_input_fn, steps=steps)

        print("Testing...")
        test_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={'x': X_test.values}, y=Y_test.values, num_epochs=1, shuffle=False)
        predictions = regressor.predict_scores(input_fn=test_input_fn)

        predictions = [p for p in predictions]
        # print(predictions)
        # print(len(predictions))
        y_predicted = np.array(predictions)
        y_predicted = y_predicted.reshape(np.array(Y_test).shape)

        # Score with sklearn
        score_sklearn = metrics.mean_squared_error(Y_test, y_predicted)
        print('MSE (sklearn): {0:f}'.format(score_sklearn))

        # Score with tensorflow
        scores = regressor.evaluate(input_fn=test_input_fn)
        print('MSE (tensorflow): {0:f}'.format(scores['loss']))

        with open('../Results/encoded/result_{0}.csv'.format(y_col), 'a') as file:
            import csv
            writer = csv.writer(file)
            writer.writerow([scores['loss']])

        # regressor.fit(input_fn=lambda: input_fn(X_train, Y_train), steps=100)
        # evaluation = regressor.evaluate(input_fn=lambda: input_fn(X_test), steps=1)
        # loss_score = evaluation["loss"]
        # print("Final Loss on the testing set: {0:f}".format(loss_score))
        # y = regressor.predict(input_fn=lambda: input_fn(X_test))


def model_2(x_data, y_data, n_features=50, n_diseases=1, n_epochs=250, n_batches=1000):
    X_train, X_test, Y_train, Y_test = train_test_split(x_data, y_data, test_size=0.20)
    training_size = X_train.shape[0]

    with tf.Graph().as_default():
        x = tf.placeholder(tf.float32, shape=[None, n_features])
        y = tf.placeholder(tf.float32, shape=[None, ])
        keep_prob = tf.placeholder(tf.float32)
        neurons = {  # TODO train This new architecture
            'in': n_features,
            'l1': 1024,
            'l2': 256,
            'l3': 64,
            'out': n_diseases
        }
        weights = {
            'l1': weight_initializer(shape=[neurons['in'], neurons['l1']], stddev=0.1, name='w1'),
            'l2': weight_initializer(shape=[neurons['l1'], neurons['l2']], stddev=0.1, name='w2'),
            'l3': weight_initializer(shape=[neurons['l2'], neurons['l3']], stddev=0.1, name='w3'),
            'out': weight_initializer(shape=[neurons['l3'], neurons['out']], stddev=0.1, name='w_out')
        }

        biases = {
            'l1': bias_initializer(init_value=0.1, shape=[neurons['l1']], name='b1'),
            'l2': bias_initializer(init_value=0.1, shape=[neurons['l2']], name='b2'),
            'l3': bias_initializer(init_value=0.1, shape=[neurons['l3']], name='b3'),
            'out': bias_initializer(init_value=0.1, shape=[neurons['out']], name='b_out')
        }
        # 1st Layer --> Fully Connected (1024 Neurons)
        layer_1 = fully_connected(x, weights['l1'], biases['l1'], name='l1')
        layer_1 = drop_out(layer_1, keep_prob)

        # 2nd Layer --> Fully Connected (256 Neurons)
        layer_2 = fully_connected(layer_1, weights['l2'], biases['l2'], name='l2')
        layer_2 = drop_out(layer_2, keep_prob)

        # 3rd Layer --> Fully Connected (64 Neurons)
        layer_3 = fully_connected(layer_2, weights['l3'], biases['l3'], name='l3')
        layer_3 = drop_out(layer_3, keep_prob)

        # Final Layer --> Fully Connected (N_DISEASES Neurons)
        final_output = fully_connected(layer_3, weights['out'], biases['out'], name='l_out')

        loss = tf.reduce_mean(tf.square(final_output - y),name='loss')
        optimizer = tf.train.AdamOptimizer(learning_rate=0.001, name='optimizer')
        train_step = optimizer.minimize(loss, name='train_step')

        init = tf.global_variables_initializer()

        training_acc = []
        validation_acc = []
        training_loss = []
        validation_loss = []

        with tf.Session() as sess:
            sess.run(init)
            for epoch in range(n_epochs):
                # Train Network
                for i in range(10):
                    batch_indices = np.random.choice(training_size, size=n_batches)
                    x_train_batch = X_train.iloc[batch_indices]
                    y_train_batch = Y_train.iloc[batch_indices]

                    feed_dict = {x: x_train_batch, y: y_train_batch, keep_prob: DROP_OUT}
                    _, train_loss = sess.run([train_step, loss], feed_dict=feed_dict)
                    training_loss.append(train_loss)
                    prediction = tf.nn.softmax(final_output)
                    correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(y_train_batch, 1))
                    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
                    feed_dict = {x: x_train_batch, y: y_train_batch, keep_prob: 1.0}
                    training_acc.append(accuracy.eval(feed_dict))

                    # Test Validation set
                    feed_dict = {x: X_test, y: Y_test, keep_prob: 1.0}
                    validation_loss.append(sess.run(loss, feed_dict=feed_dict))
                    prediction = tf.nn.softmax(final_output)
                    correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y_test, 1))
                    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
                    validation_acc.append(accuracy.eval(feed_dict))
                if epoch % 5 == 0:
                    print("Epoch:", '%04d' % (epoch + 1) if epoch == 0 else epoch,
                          "\tValidation Accuracy =", '%01.9f' % (validation_acc[-1]),
                          "\tValidation Loss =", '%09.5f' % (validation_loss[-1]),
                          "\tTraining Accuracy =", '%01.9f' % (training_acc[-1]),
                          "\tTraining Loss =", '%09.5f' % (training_loss[-1]))
        print("Training Finished!")


def main():
    print("Loding Data...")
    # x_data = pd.DataFrame([[i for i in range(50)] for _ in range(100)])
    # y_data = pd.DataFrame([[i] for i in range(100)])
    x_data, y_data = load_data(DAMAVAND_LOCATION_X), load_data(DAMAVAND_LOCATION_Y)
    print(y_data.shape)
    # y_data = pd.DataFrame(np.reshape(y_data, newshape=[-1, 1]))
    print("Train Deep Neural Networks")
    for i in range(100):
        random_features = np.random.choice(19671, 50, replace=False)
        print(np.array(random_features))
        random_x = x_data[random_features]
        for y_col in range(y_data.shape[1]):
            new_y = y_data[y_col]
            new_y = pd.DataFrame(np.reshape(new_y, newshape=[-1, 1]))
            train(random_x, new_y, y_col, num_epochs=5000, steps=5000, batch_size=2500)


if __name__ == '__main__':
    main()
