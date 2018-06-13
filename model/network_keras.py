import numpy as np
import pandas as pd
import os

from keras import backend as K
from keras.layers import Input, Dense, Dropout, GaussianNoise, BatchNormalization, GaussianDropout
import keras.backend as backend
from keras.models import Model
from keras.callbacks import ModelCheckpoint, CSVLogger, History
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, normalize
from keras.utils import np_utils

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
LOCAL_RESULTS_ENCODED = './Results/CAE/encoded_results_{0}_{1}_notNoised.csv'

# Hyper-Parameters
LEARNING_RATE = 1e-3
DROP_OUT = 0.5
N_SAMPLES = 10787
N_FEATURES = 19671
N_DISEASES = 34
N_BATCHES = 256
N_EPOCHS = 200
N_BATCH_LEARN = 10
N_RANDOM_FEATURES = 200
neurons = {
    'in': 12,
    'l1': 1024,
    'l2': 512,
    'l3': 256,
    'l4': 128,
    'out': N_DISEASES,
    'code': 12,
    'features': N_FEATURES
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


def contractive_dropout_autoencoder(machine_name, local_data_folder, local_result_folder, model_specific,
                                    n_random_features, seed=2018):
    seed = seed
    np.random.seed(seed=seed)

    # dataset_folder = "/s/" + machine_name + local_data_folder
    # dataset_folder = '../Data/'
    dataset_folder = "/s/chopin/a/grad/asharifi/f/Behrooz/dataset_local/"

    df_m_rna_address = dataset_folder + "fpkm_normalized.csv"
    df_disease_address = dataset_folder + "disease.csv"

    df_m_rna = np.loadtxt(df_m_rna_address, delimiter=",")
    df_disease = np.ravel(pd.DataFrame.as_matrix(pd.read_csv(df_disease_address, delimiter=",", header=None)))

    # df_m_rna = normalize(X=df_m_rna, axis=0, norm="max")

    label_encoder_disease = LabelEncoder()
    label_encoder_disease.fit(df_disease)
    encoded_disease = label_encoder_disease.transform(df_disease)

    categorical_disease = np_utils.to_categorical(encoded_disease)
    m_rna = df_m_rna

    # noise_factor = 0.05
    # m_rna_noisy = m_rna + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=m_rna.shape)

    np.random.seed(seed)
    random_feature_indices = np.random.choice(19671, n_random_features, replace=False)

    indices = np.arange(m_rna.shape[0])
    indices = indices[0:10787]
    np.random.shuffle(indices)

    m_rna = m_rna[indices]
    # m_rna = m_rna[:, random_feature_indices]

    categorical_disease = categorical_disease[indices]

    m_rna_train = m_rna[0:9750, ]
    m_rna_train_input = m_rna[0:9750, random_feature_indices]
    m_rna_test = m_rna[9750:10787, ]
    m_rna_test_input = m_rna[9750:10787, random_feature_indices]

    categorical_disease_train = categorical_disease[0:9750, ]
    categorical_disease_test = categorical_disease[9750: 10787, ]

    print("data loading has just been finished")

    print(m_rna_train_input.shape, categorical_disease.shape)

    batch_size = 64
    nb_epochs = 200

    def create_model():
        inputs = Input(shape=(n_random_features,), name="inputs")
        inputs_noise = GaussianNoise(stddev=0.025)(inputs)
        inputs_noise = GaussianDropout(rate=0.025 ** 2 / (1 + 0.025 ** 2))(inputs_noise)
        inputs_0 = BatchNormalization(name="inputs_0")(inputs_noise)
        inputs_0 = Dropout(rate=0.0, name='dropout_1')(inputs_0)
        inputs_1 = Dense(1024, activation="softplus", name="inputs_1")(inputs_0)
        inputs_2 = BatchNormalization(name="inputs_2")(inputs_1)
        inputs_2 = Dropout(rate=0.0, name='dropout_2')(inputs_2)
        inputs_3 = Dense(256, activation="softplus", name="inputs_3")(inputs_2)
        inputs_4 = BatchNormalization(name="inputs_4")(inputs_3)
        inputs_4 = Dropout(rate=0.25, name='dropout_3')(inputs_4)

        encoded = Dense(units=12, activation='sigmoid', name='encoded')(inputs_4)
        encoded_noise = GaussianNoise(0.025)(encoded)

        # encoded_softmax = Dense(units=12, activation='softmax', name='encoded_softmax')(encoded)
        # encoded_attention = multiply([encoded, encoded_softmax])

        inputs_5 = Dense(512, activation="linear", name="inputs_5")(encoded_noise)
        inputs_5 = Dropout(rate=0.25, name='dropout_4')(inputs_5)

        decoded_tcga = Dense(units=m_rna.shape[1], activation='relu', name="m_rna")(inputs_5)
        cl_2 = Dense(units=categorical_disease.shape[1], activation="softmax", name="cl_disease")(encoded)

        scae = Model(inputs=inputs, outputs=[decoded_tcga, cl_2])

        lambda_value = 9.5581e-3

        def contractive_loss(y_pred, y_true):
            mse = backend.mean(backend.square(y_true - y_pred), axis=1)

            w = backend.variable(value=scae.get_layer('encoded').get_weights()[0])  # N inputs N_hidden
            w = backend.transpose(w)  # N_hidden inputs N
            h = scae.get_layer('encoded').output
            dh = h * (1 - h)  # N_batch inputs N_hidden

            # N_batch inputs N_hidden * N_hidden inputs 1 = N_batch inputs 1
            contractive = lambda_value * backend.sum(dh ** 2 * backend.sum(w ** 2, axis=1), axis=1)

            return mse + contractive

        # scae.compile(optimizer='nadam',
        #              loss=[contractive_loss, "mse", "cosine_proximity", "cosine_proximity"],
        #              loss_weights=[0.001, 0.001, 0.5, 0.5],
        #              metrics={"m_rna": ["mae", "mse"], "mi_rna": ["mae", "mse"], "cl_tissue": "acc",
        #                       "cl_disease": "acc"})

        scae.compile(optimizer='nadam',
                     loss=[contractive_loss, "mse"],
                     loss_weights=[0.001, 0.001],
                     metrics={"m_rna": ["mae", "mse"], "cl_disease": "acc"})

        return scae

    model = create_model()

    # result_folder = "/home/ermia/Desktop/Deep Learning-Bioinfo/Results/"
    # result_folder = '/s/' + machine_name + local_result_folder + model_specific
    result_folder = '../Results/CAE/'

    file_name = "best-scae-dropout.log"

    csv_logger = CSVLogger(result_folder + file_name)
    history = History()

    model.fit(x=m_rna_train_input, y=[m_rna_train, categorical_disease_train],
              batch_size=batch_size, epochs=nb_epochs,
              callbacks=[csv_logger, history],
              validation_data=(
                  m_rna_test_input, [m_rna_test, categorical_disease_test]), verbose=2)

    print(history.history.keys())
    print("fitting has just been finished")

    # save the model and encoded-layer output
    # model.save(filepath=result_folder + "scae-dropout.h5")
    #
    # layer_name = "encoded"
    # encoded_layer_model = Model(inputs=model.input, outputs=model.get_layer(layer_name).output)
    # encoded_output = encoded_layer_model.predict(df_m_rna)
    #
    # np.savetxt(X=encoded_output, fname=result_folder + "encoded_scae_dropout.csv", delimiter=",")
    #
    # noise_matrix = 0.5 * np.random.normal(loc=0.0, scale=1.0, size=encoded_output.shape)
    # np.savetxt(X=encoded_output + noise_matrix, fname=result_folder + "encoded_scae_dropout_noised_1.csv",
    #            delimiter=",")
    #
    # noise_matrix = 0.5 * np.random.normal(loc=0.0, scale=0.5, size=encoded_output.shape)
    # np.savetxt(X=encoded_output + noise_matrix, fname=result_folder + "encoded_scae_dropout_noised_0.5.csv",
    #            delimiter=",")
    #
    # noise_matrix = 0.5 * np.random.normal(loc=0.0, scale=0.01, size=encoded_output.shape)
    # np.savetxt(X=encoded_output + noise_matrix, fname=result_folder + "encoded_scae_dropout_noised_0.01.csv",
    #            delimiter=",")
    #
    # # save the result and prediction value
    #
    # data_pred = model.predict(m_rna, batch_size=batch_size, verbose=2)
    # np.savetxt(X=m_rna, fname=result_folder + "tcga_genes_scae_dropout.csv", delimiter=",", fmt='%1.3f')
    # np.savetxt(X=categorical_disease, fname=result_folder + "categorical_disease_scae_dropout.csv", delimiter=",",
    #            fmt='%1.3f')
    #
    # np.savetxt(X=data_pred[0], fname=result_folder + "tcga_genes_scae_dropout_pred.csv", delimiter=",", fmt='%1.3f')
    # np.savetxt(X=data_pred[1], fname=result_folder + "micro_rna_scae_dropout_pred.csv", delimiter=",", fmt='%1.3f')

    print("prediction process has just been finished")

    import csv
    with open(DAMAVAND_RESULTS_ENCODED.format(0.01, n_random_features), 'a') as file:
        writer = csv.writer(file)
        score = model.evaluate(m_rna_test_input, [m_rna_test, categorical_disease], verbose=0)[0]
        print('score is ', score)
        writer.writerow([float(score)])

    # for i in range(1, 51):
    #     print(i)
    #     dropout_factor = 1 - np.divide(i, 100)
    #     dropout_matrix = np.random.binomial(n=1, p=dropout_factor, size=m_rna.shape)
    #     m_rna_dropout = np.multiply(m_rna, dropout_matrix)
    #     m_rna_temp_test = m_rna_dropout[9750:10787, ]
    #     # score = model.evaluate(m_rna_temp_test,
    #     #                        [m_rna_temp_test, mi_rna_test, categorical_tissue_test, categorical_disease_test],
    #     #                        verbose=0, batch_size=batch_size)
    #
    #     score = model.evaluate(m_rna_temp_test,
    #                            [m_rna_temp_test, categorical_disease_test],
    #                            verbose=0, batch_size=batch_size)
    #     print(score)
    #
    #     # with open(result_folder + 'dropout-Dropout-CAE.txt', 'ab') as file:
    #     #     np.savetxt(file, score, delimiter=",")
    #
    # print("dropout has just been finished")
    #
    # for i in range(1, 51):
    #     print(i)
    #     noise_factor = np.divide(i, 100)
    #     noise_matrix = noise_factor * np.random.normal(loc=0.0, scale=1.0, size=m_rna.shape)
    #     m_rna_noisy = m_rna + noise_matrix
    #     m_rna_temp_test = m_rna_noisy[9750:10787, ]
    #     # score = model.evaluate(m_rna_temp_test,
    #     #                        [m_rna_temp_test, mi_rna_test, categorical_tissue_test, categorical_disease_test],
    #     #                        verbose=0, batch_size=batch_size)
    #
    #     score = model.evaluate(m_rna_temp_test,
    #                            [m_rna_temp_test, categorical_disease_test],
    #                            verbose=0, batch_size=batch_size)
    #     print(score)
    #
    #     # with open(result_folder + 'gaussian-Dropout-CAE.txt', 'ab') as file:
    #     #     np.savetxt(file, score, delimiter=",")

    print("gaussian has just been finished")


def auto_encoder(stddev=0.0, x_data=None, y_data=None, n_features=10, random_selection=False, seed=2018):
    # noise_matrix = 0.5 * np.random.normal(loc=0.0, scale=stddev, size=y_data.shape)

    # y_data_noised = y_data + noise_matrix
    # Train/Test Split
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.30, shuffle=True)

    # Random Feature Selection
    if random_selection:
        if random_selection:
            np.random.seed(seed)
            random_feature_indices = np.random.choice(19671, n_features, replace=False)
            x_train_random = x_train[random_feature_indices]
            x_test_random = x_test[random_feature_indices]

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

        encoded = Dense(neurons['code'], activation='sigmoid')(l4_dropout)

        inputs_5 = Dense(512, activation="linear")(encoded)
        inputs_5 = Dropout(rate=0.25)(inputs_5)

        decoded = Dense(units=N_FEATURES, activation='relu')(inputs_5)

        # Compile Model
        network = Model(input_layer, decoded)

        network.compile(optimizer='nadam', loss='mse')

        return network

    model = create_model()

    model.fit(x=x_train_random.as_matrix(), y=y_train.as_matrix(),
              epochs=N_EPOCHS,
              batch_size=N_BATCHES,
              shuffle=True,
              validation_data=(x_test_random.as_matrix(), y_test.as_matrix()),
              verbose=2)

    import csv
    with open(DAMAVAND_RESULTS_ENCODED.format(stddev, n_features), 'a') as file:
        writer = csv.writer(file)
        score = model.evaluate(x_test_random.as_matrix(), y_test.as_matrix(), verbose=0)
        print('score is ', score)
        writer.writerow([float(score)])


if __name__ == '__main__':
    # Load Data
    x_data = pd.read_csv(DAMAVAND_LOCATION_FPKM_NORMALIZED, header=None)
    # y_data = pd.read_csv(DAMAVAND_LOCATION_ENCODED, header=None)

    # noise_matrix = 0.0 * np.random.normal(loc=0.0, scale=1.0, size=y_data.shape)
    # y_data += noise_matrix

    # label_encoder = LabelEncoder()
    # label_encoder.fit(y_data)
    # label_encoder = label_encoder.transform(y_data)
    # y_data = pd.DataFrame(keras.utils.to_categorical(label_encoder))

    # print(x_data.shape, y_data.shape)
    for i in range(1000):
        for n_features in [2, 4, 8, 16, 32, 64, 128, 256, 512]:
            auto_encoder(0.01, x_data, x_data, n_features=n_features, random_selection=True, seed=2018 * n_features)
            # contractive_dropout_autoencoder(machine_name="damavand",
            #                                 local_data_folder=DAMAVAND_LOCATION_FPKM_NORMALIZED,
            #                                 local_result_folder="../Results/", model_specific="optimal_",
            #                                 n_random_features=n_features, seed=2018 * n_features * i)

    print("Finished")
