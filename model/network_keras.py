import numpy as np
import pandas as pd
import os, sys

import keras
import keras.backend as K
from keras.layers import Input, Dense, Dropout, GaussianNoise, BatchNormalization
from keras.models import Model
from keras.callbacks import ModelCheckpoint, CSVLogger

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

"""
    Created by Mohsen Naghipourfar on 3/26/18.
    Email : mn7697np@gmail.com
    Website: http://ce.sharif.edu/~naghipourfar
"""
# Constants
DAMAVAND_LOCATION_FPKM_NORMALIZED = "~/f/Behrooz/dataset_local/fpkm_normalized.csv"
DAMAVAND_LOCATION_CATEGORICAL_DISEASE = "~/f/Behrooz/dataset_local/disease.csv"
DAMAVAND_LOCATION_ENCODED = '../Data/encoded_scae_dropout.csv'
DAMAVAND_RESULTS_ENCODED = '../Results/CAE/encoded_results_{0}_{1}.csv'

LOCAL_LOCATION_FPKM_NORMALIZED = "../Data/fpkm_normalized.csv"
LOCAL_LOCATION_CATEGORICAL_DISEASE = "../Data/disease.csv"
LOCAL_LOCATION_ENCODED = "./Results/CAE/old/encoded_scae_dropout.csv"
LOCAL_RESULTS_ENCODED = './Results/CAE/encoded_results_{0}_{1}.csv'

# Hyper-Parameters
LEARNING_RATE = 1e-3
DROP_OUT = 0.5
N_SAMPLES = 10787
N_FEATURES = 19671
N_DISEASES = 34
N_BATCHES = 256
N_EPOCHS = 10
N_BATCH_LEARN = 10
N_RANDOM_FEATURES = 200
neurons = {
    'in': 12,
    'l1': 1024,
    'l2': 512,
    'l3': 256,
    'l4': 128,
    'out': N_DISEASES,
    'code': 12
}


def run(stddev, x_data, y_data, random_selection=True, seed=2018):
    # Train/Test Split
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.30)

    # Random Feature Selection
    if random_selection:
        np.random.seed(seed)
        random_feature_indices = np.random.choice(19671, N_RANDOM_FEATURES, replace=False)
        x_train = x_train[random_feature_indices]
        x_test = x_test[random_feature_indices]

    # Design Model
    input_layer = Input(shape=(neurons['in'],))

    noise_layer = GaussianNoise(stddev)(input_layer)

    l1 = Dense(neurons['l1'], activation='relu')(noise_layer)
    l1 = BatchNormalization()(l1)
    l1_dropout = Dropout(DROP_OUT)(l1)

    l2 = Dense(neurons['l2'], activation='relu')(l1_dropout)
    l2 = BatchNormalization()(l2)
    l2_dropout = Dropout(DROP_OUT)(l2)

    l3 = Dense(neurons['l3'], activation='relu')(l2_dropout)
    l3 = BatchNormalization()(l3)
    l3_dropout = Dropout(DROP_OUT)(l3)

    l4 = Dense(neurons['l4'], activation='relu')(l3_dropout)
    l4 = BatchNormalization()(l4)
    l4_dropout = Dropout(DROP_OUT)(l4)

    output_layer = Dense(neurons['out'], activation='softmax')(l4_dropout)

    # Compile Model
    network = Model(input_layer, output_layer)

    network.compile(optimizer='nadam', loss='categorical_crossentropy', metrics=['accuracy'])

    # network.summary()
    if not os._exists('./Results/Keras/{0}_{1}/'.format(os.getpid(), stddev)):
        os.makedirs('./Results/Keras/{0}_{1}/'.format(os.getpid(), stddev))
    save_path = './Results/Keras/{0}_{1}/'.format(os.getpid(), stddev) + 'model.{epoch:02d}-{val_acc:.4f}.hdf5'

    # Create a Callback for Model
    checkpointer = ModelCheckpoint(filepath=save_path,
                                   verbose=0,
                                   monitor='val_acc',
                                   save_best_only=True,
                                   mode='auto',
                                   period=1)

    # get_3rd_layer_output = K.function([network.layers[0].input, K.learning_phase()],
    #                                   [network.layers[3].output])
    # layer_output = get_3rd_layer_output([x_train, True])
    # print(layer_output[0].shape)
    # print(len(layer_output))
    # print("*" * 100)

    # Train Model

    # for epoch in range(N_EPOCHS):
    network.fit(x=x_train.as_matrix(),
                y=y_train.as_matrix(),
                epochs=N_EPOCHS,
                batch_size=N_BATCHES,
                shuffle=True,
                validation_data=(x_test.as_matrix(), y_test.as_matrix()),
                callbacks=[checkpointer],
                verbose=2)
    # layer_output.append(get_3rd_layer_output([x_train, True])[0])
    # print(layer_output)

    # print(layer_output[0].shape)
    # print(len(layer_output))
    # print("*" * 100)

    # Save Accuracy, Loss
    import csv
    with open('./result_noised_{0}.csv'.format(stddev), 'a') as file:
        writer = csv.writer(file)
        loss, accuracy = network.evaluate(x_test.as_matrix(), y_test.as_matrix(), verbose=0)
        writer.writerow([accuracy, loss])

    # class DummyCheckpoint(Callback):
    #     def on_train_begin(self, logs=None):
    #         self.accuracies = []
    #
    #     def on_epoch_end(self, epoch, logs=None):
    #         if (max(self.accuracies)) < logs.get('acc'):
    #             self.accuracies.append(logs.get('acc'))
    # return layer_output


def learn_code_layer(stddev=0.0, x_data=None, y_data=None, n_features=10, random_selection=False, seed=2018):
    noise_matrix = 0.5 * np.random.normal(loc=0.0, scale=stddev, size=y_data.shape)

    y_data_noised = y_data + noise_matrix
    # Train/Test Split
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data_noised, test_size=0.30, shuffle=True)

    # Random Feature Selection
    if random_selection:
        if random_selection:
            np.random.seed(seed)
            random_feature_indices = np.random.choice(19671, n_features, replace=False)
            x_train = x_train[random_feature_indices]
            x_test = x_test[random_feature_indices]

    def create_model():
        input_layer = Input(shape=(n_features,))

        l1 = Dense(neurons['l1'], activation='relu')(input_layer)
        l1 = BatchNormalization()(l1)
        l1_dropout = Dropout(DROP_OUT)(l1)

        l2 = Dense(neurons['l2'], activation='relu')(l1_dropout)
        l2 = BatchNormalization()(l2)
        l2_dropout = Dropout(DROP_OUT)(l2)

        l3 = Dense(neurons['l3'], activation='relu')(l2_dropout)
        l3 = BatchNormalization()(l3)
        l3_dropout = Dropout(DROP_OUT)(l3)

        l4 = Dense(neurons['l4'], activation='relu')(l3_dropout)
        l4 = BatchNormalization()(l4)
        l4_dropout = Dropout(DROP_OUT)(l4)

        output_layer = Dense(neurons['code'], activation='sigmoid')(l4_dropout)

        # Compile Model
        network = Model(input_layer, output_layer)

        network.compile(optimizer='nadam', loss='mse')

        return network

    model = create_model()

    model.fit(x=x_train.as_matrix(), y=y_train.as_matrix(),
              epochs=N_EPOCHS,
              batch_size=N_BATCHES,
              shuffle=True,
              validation_data=(x_test.as_matrix(), y_test.as_matrix()),
              verbose=2)
    import csv
    with open(LOCAL_RESULTS_ENCODED.format(stddev, n_features), 'a') as file:
        writer = csv.writer(file)
        score = model.evaluate(x_test.as_matrix(), y_test.as_matrix(), verbose=0)
        writer.writerow(score)


if __name__ == '__main__':
    # Load Data
    x_data = pd.read_csv(LOCAL_LOCATION_FPKM_NORMALIZED, header=None)
    y_data = pd.read_csv(LOCAL_LOCATION_ENCODED, header=None)

    noise_matrix = 0.5 * np.random.normal(loc=0.0, scale=1.0, size=y_data.shape)
    y_data += noise_matrix

    # label_encoder = LabelEncoder()
    # label_encoder.fit(y_data)
    # label_encoder = label_encoder.transform(y_data)
    # y_data = pd.DataFrame(keras.utils.to_categorical(label_encoder))

    print(x_data.shape, y_data.shape)
    for n_features in [2, 4, 8, 16, 32, 64, 128, 256, 512]:
        for i in range(100):
            learn_code_layer(0.01, x_data, y_data, n_features=n_features, random_selection=True, seed=2018 * i)

    print("Finished")
