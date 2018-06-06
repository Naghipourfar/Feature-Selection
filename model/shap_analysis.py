import numpy as np
import pandas as pd

import tensorflow as tf
import matplotlib.pyplot as plt

import shap
import keras
import keras.backend as backend
from keras.layers import Input, Dense, Dropout, BatchNormalization
from keras.models import Model

"""
    Created by Mohsen Naghipourfar on 6/6/18.
    Email : mn7697np@gmail.com
    Website: http://ce.sharif.edu/~naghipourfar
"""


def load_model(path="./Results/CAE/old/scae-dropout.h5"):
    model = keras.models.load_model(path, custom_objects={'contractive_loss': contractive_loss})
    model.summary()
    return model


def contractive_loss(y_pred, y_true):
    mse = backend.mean(backend.square(y_true - y_pred), axis=1)

    # w = backend.variable(value=scae.get_layer('encoded').get_weights()[0])  # N inputs N_hidden
    # w = backend.transpose(w)  # N_hidden inputs N
    # h = scae.get_layer('encoded').output
    # dh = h * (1 - h)  # N_batch inputs N_hidden
    #
    # N_batch inputs N_hidden * N_hidden inputs 1 = N_batch inputs 1
    # contractive = 0.6 * backend.sum(dh ** 2 * backend.sum(w ** 2, axis=1), axis=1)

    # return mse + contractive
    return mse


# def analyze_with_shap(model, x_train, y_train):
# explainer = shap.KernelExplainer(lambda x : model.predict(x)[0], x_train.iloc[:10, :])
# shap_values = explainer.shap_values(x_train.iloc[299, :], nsamples=10)
# shap.force_plot(shap_values, X_display.iloc[299, :])

def auto_encoder(x_train, y_train):
    def create_model():
        inputs = Input(shape=x_train.shape[1])
        encoder_1 = Dense(1024, activation='relu', name="encoder_1")(inputs)
        encoder_1 = BatchNormalization()(encoder_1)
        encoder_1 = Dropout(0.0)(encoder_1)

        encoder_2 = Dense(256, activation='relu', name="encoder_2")(encoder_1)
        encoder_2 = BatchNormalization()(encoder_2)
        encoder_2 = Dropout(0.25)(encoder_2)

        code = Dense(12, activation='relu', name='code')(encoder_2)

        decoder_2 = Dense(256, activation='relu', name="decoder_2")(code)
        decoder_2 = BatchNormalization()(decoder_2)
        decoder_2 = Dropout(0.0)(decoder_2)

        decoder_3 = Dense(1024, activation='relu', name="decoder_3")(decoder_2)
        decoder_3 = BatchNormalization()(decoder_3)
        decoder_3 = Dropout(0.0)(decoder_3)

        decoder = Dense(y_train.shape[1], activation='relu', name="output")(decoder_3)

        model = Model(inputs=inputs, outputs=decoder)

        model.compile(optimizer='nadam', loss=keras.losses.mse)

        return model

    model = create_model()

    model.fit(x=x_train,
              y=x_train,
              epochs=250,
              batch_size=256,
              validation_split=0.25,
              verbose=2)

    model.save('./AE.h5')


if __name__ == '__main__':
    # model = load_model()
    x_train = pd.read_csv('../Data/fpkm_normalized.csv', header=None)
    # analyze_with_shap(model, x_train, y_train)
    auto_encoder(x_train, x_train)
