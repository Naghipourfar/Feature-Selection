import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib

matplotlib.use('agg')
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split
import matplotlib

matplotlib.use('agg')
"""
    Created by Mohsen Naghipourfar on 2/18/18.
    Email : mn7697np@gmail.com
"""
# Constants
DAMAVAND_LOCATION_X = "~/f/Behrooz/dataset_local/fpkm_normalized.csv"
DAMAVAND_LOCATION_Y = "~/f/Behrooz/dataset_local/disease.csv"
DAMAVAND_LOCATION_SAVE = "./Results/Random_Features/"
LOCAL_LOCATION_X = "../Data/fpkm_normalized.csv"
LOCAL_LOCATION_Y = "../Data/disease.csv"
LOCAL_LOCATION_SAVE = "../Results/All_Features/"

# Hyper-Parameters
LEARNING_RATE = 1e-3
DROP_OUT = 0.5
N_SAMPLES = 10787
N_FEATURES = 19671
N_DISEASES = 34
N_BATCHES = 2000
N_EPOCHS = 100
N_BATCH_LEARN = 10


def normalize_data(data):
    return pd.DataFrame(normalize(data, norm='max', axis=0))


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


def make_directory(path):
    os.makedirs(path)


def save_model_results(k, path, validation_accuracy, validation_loss, training_accuracy, training_loss):
    plot_results(k, path, validation_accuracy, training_accuracy, result_type='Accuracy')
    plot_results(k, path, validation_loss, training_loss, result_type='Loss')
    write_result_data(path, validation_accuracy, validation_loss, training_accuracy, training_loss)


def write_result_data(path, validation_accuracy, validation_loss, training_accuracy, training_loss):
    with open(path + '/results.txt', 'w') as f:
        f.write("Validation Accuracy\tValidation Loss\tTraining Accuracy\tTraining Loss\n")
        for v_acc, v_loss, t_acc, t_loss in zip(validation_accuracy, validation_loss, training_accuracy, training_loss):
            f.write(str(v_acc) + "\t" + str(v_loss) + "\t" + str(t_acc) + "\t" + str(t_loss) + "\n")


def plot_results(k, path, validation_result, trainig_result, result_type='Accuracy'):
    plt.figure()
    plt.plot(trainig_result, 'k-', label='Training ' + result_type)
    plt.plot(validation_result, 'b--', label='Validation ' + result_type)
    if result_type == 'Accuracy':
        plt.ylim(ymax=1.0, ymin=0.0)
    plt.title('Accuracy per Epoch (k = {0})'.format(k))
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    if result_type == 'Accuracy':
        plt.legend(loc="lower right")
    else:
        plt.legend(loc="upper right")
    result_path = path + '/' + result_type + '.png'
    plt.savefig(result_path)
    plt.close()


def train(k, x_data, y_data,
          n_samples=N_SAMPLES,
          n_features=N_FEATURES,
          n_diseases=N_DISEASES,
          learning_rate=LEARNING_RATE,
          n_epochs=N_EPOCHS,
          n_batch_learn=N_BATCH_LEARN,
          n_batches=N_BATCHES):
    global LOCAL_LOCATION_SAVE, DAMAVAND_LOCATION_SAVE
    tf.reset_default_graph()
    # Split data into train/test = 80%/20%
    X_train, X_test, Y_train, Y_test = train_test_split(x_data, y_data, test_size=0.20)
    print(X_train.shape)
    # train_indices = np.random.choice(n_samples, round(n_samples * 0.85),
    #                                  replace=False)
    # validation_indices = np.array(list(set(range(n_samples)) - set(train_indices)))
    #
    # x_train = x_data.iloc[train_indices]
    # y_train = y_data.iloc[train_indices]

    training_size = X_train.shape[0]

    # x_validation = x_data.iloc[validation_indices]
    # y_validation = y_data.iloc[validation_indices]

    # Create Network and Variables
    with tf.Graph().as_default():
        x = tf.placeholder(tf.float32, shape=[None, n_features])
        y = tf.placeholder(tf.float32, shape=[None, n_diseases])
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

        loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(logits=final_output, labels=y),
            name='loss')
        optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE, name='optimizer')
        train_step = optimizer.minimize(loss, name='train_step')

        init = tf.global_variables_initializer()

        training_acc = []
        validation_acc = []
        training_loss = []
        validation_loss = []

        # saver = tf.train.Saver()
        # additional_path = 'model_{0}_{1}'.format(os.getpid(), k)
        # path = LOCAL_LOCATION_SAVE + additional_path
        # make_directory(path)
        # with open(path + '/parameters.txt', 'w') as f:
        #     f.write("n_samples\tn_features\tn_diseases\tLambda\tlearning_rate\tn_epochs\tn_batch_learn\tn_batches\n")
        #     f.write(
        #         str(n_samples) + "\t" + str(n_features) + "\t" +
        #         str(learning_rate) + "\t" + str(n_epochs) + "\t" +
        #         str(n_batch_learn) + "\t" + str(n_batches) + "\n")
        # log_filename = path + '/log.txt'
        # log_file = open(log_filename, 'w')
        with tf.Session() as sess:
            sess.run(init)
            for epoch in range(n_epochs):
                # Train Network
                for i in range(n_batch_learn):
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
                # message = "Epoch:" + '%04d' % (epoch + 1) \
                #           + "\tValidation Accuracy =" + '%01.9f' % (validation_acc[-1]) \
                #           + "\tValidation Loss =" + '%09.5f' % (validation_loss[-1]) \
                #           + "\tTraining Accuracy =" + '%01.9f' % (training_acc[-1]) \
                #           + "\tTraining Loss =" + '%09.5f' % (training_loss[-1]) \
                #           + "\n"
                # log_file.write(message)
                print("Epoch:", '%04d' % (epoch + 1),
                      "\tValidation Accuracy =", '%01.9f' % (validation_acc[-1]),
                      "\tValidation Loss =", '%09.5f' % (validation_loss[-1]),
                      "\tTraining Accuracy =", '%01.9f' % (training_acc[-1]),
                      "\tTraining Loss =", '%09.5f' % (training_loss[-1]))
            # model_path = path + '/session.ckpt'
            # saver.save(sess, model_path)
        print("Training Finished!")
        # log_file.close()
        # new_path = path + '_{0}'.format(validation_acc[-1])
        # os.rename(path, new_path)
        # path = new_path
        #
        # save_model_results(k, path, validation_acc, validation_loss, training_acc, training_loss)
        # print("Plots have been saved!")
        import csv
        with open('./result_{0}.csv'.format(k), 'a') as file:
            writer = csv.writer(file)
            writer.writerow([validation_acc[-1], validation_loss[-1]])


def modify_output(target):
    global N_DISEASES
    output_dict = {}
    a = target[0].value_counts()
    # print(a)
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


def random_train(k, x_train, y_train,
                 learning_rate=LEARNING_RATE,
                 n_batch_learn=N_BATCH_LEARN,
                 n_batches=N_BATCHES):
    print("k = {0}".format(k))
    global N_FEATURES
    new_X = []
    if k < 19671:
        random_feature_indices = np.random.choice(19671, k, replace=False)
        print(np.array(random_feature_indices))
        new_X = x_train[random_feature_indices]
        print(new_X.shape)
        N_FEATURES = new_X.shape[1]
        train(k, new_X, y_train,
              n_samples=N_SAMPLES,
              n_features=N_FEATURES,
              n_diseases=N_DISEASES,
              learning_rate=learning_rate,
              n_batch_learn=n_batch_learn,
              n_batches=n_batches)


def random_choice(n, features):
    random_arr = [1 for _ in range(n)] + [0 for _ in range(N_FEATURES - n)]
    np.random.shuffle(random_arr)
    a = features.copy()
    for i in range(N_FEATURES):
        if random_arr[i] == 0:
            a[i] = 0
    return a


if __name__ == '__main__':
    x_filename = LOCAL_LOCATION_X
    y_filename = LOCAL_LOCATION_Y
    print("Loading data...")
    x_train, y_train = load_data(x_filename), load_data(y_filename)
    # print("Normalizing Data")
    # x_train = normalize_data(x_train)
    print("Data has been loaded successfully!")
    # N_SAMPLES, N_FEATURES = x_train.shape
    y_train = modify_output(y_train)
    y_train = pd.DataFrame(y_train)
    print("Training neural network!")
    # from multiprocessing import Pool
    #
    # N_PROCESSES = 1
    # with Pool(N_PROCESSES) as p:  # Running multiple processes for training different networks (use cores of CPU)
    #     p.starmap(random_train,
    #               [[N_FEATURES, x_train, y_train, LAMBDA, LEARNING_RATE, N_BATCH_LEARN, N_BATCHES + i * 750] for i in
    #                range(N_PROCESSES)])
    for i in range(1000):
        random_train(100, x_train, y_train)
    # random_train(N_FEATURES, x_train, y_train, LEARNING_RATE, N_BATCH_LEARN, N_BATCHES)
    # random_train(N_FEATURES, x_train, y_train, 1.0, N_BATCH_LEARN, N_BATCHES)
    # random_train(N_FEATURES, x_train, y_train, 1.0, N_BATCH_LEARN, N_BATCHES)
    # random_train(N_FEATURES, x_train, y_train, LEARNING_RATE, N_BATCH_LEARN, N_BATCHES)
    # random_train(N_FEATURES, x_train, y_train, LEARNING_RATE, N_BATCH_LEARN, 1500)
    # for k in range(25, 35):
    #     random_train(k, x_train, y_train, LAMBDA, LEARNING_RATE, N_BATCH_LEARN, N_BATCHES)
