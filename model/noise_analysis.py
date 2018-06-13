import numpy as np
import pandas as pd

import tensorflow as tf
import matplotlib.pyplot as plt

"""
    Created by Mohsen Naghipourfar on 6/12/18.
    Email : mn7697np@gmail.com
    Website: http://ce.sharif.edu/~naghipourfar
"""

def auto_encoder(x_train, y_train):
    def create_model():
        inputs = Input(shape=(x_train.shape[1],))
        encoder_1 = Dense(1024, activation='relu', name="encoder_1")(inputs)
        encoder_1 = BatchNormalization()(encoder_1)
        encoder_1 = Dropout(0.0)(encoder_1)

        # encoder_2 = Dense(256, activation='relu', name="encoder_2")(encoder_1)
        # encoder_2 = BatchNormalization()(encoder_2)
        # encoder_2 = Dropout(0.25)(encoder_2)

        code = Dense(12, activation='relu', name='code')(encoder_1)

        # decoder_2 = Dense(256, activation='relu', name="decoder_2")(code)
        # decoder_2 = BatchNormalization()(decoder_2)
        # decoder_2 = Dropout(0.0)(decoder_2)

        decoder_3 = Dense(1024, activation='relu', name="decoder_3")(code)
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

    return model
