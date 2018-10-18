from keras import backend as K
from keras.engine.topology import Layer
import keras.initializers as initializers
import keras.regularizers as regularizers
import keras.constraints as constraints
import keras.activations as activations
from keras.layers import InputSpec, Input, Dense, KSparseDense, Lambda
from keras.models import Model
import numpy as np


# def KSparse(x, k=3):
#     # top_k_values = K.tf.nn.top_k(x, k=k)[0]
#     top_k_indices = K.tf.contrib.framework.argsort(x)[-k:]
#     output_shape = K.shape(x)
#     print(output_shape)
#     indices = np.zeros(shape=(1, output_shape.shape[0]))
#     print(indices.shape)
#     return x * indices

def func(x):
    # print(K.equal(K.learning_phase(), False))
    # f = K.function([K.learning_phase()], [x])
    return x

def nn(x_data, y_data):
    def create_model():
        input_layer = Input(shape=(x_data.shape[1],))
        lamda_layer = Lambda(func)(input_layer)
        output_layer = Dense(units=y_data.shape[1])(lamda_layer)

        model = Model(input_layer, output_layer)
        model.compile(optimizer="adam", loss="mse")
        return model

    model = create_model()
    model.fit(x_data,
            y_data,
            epochs=10,
            validation_split=0.25,
            verbose=0)
    # print(Code.get_layer("-1"))

x_data = np.array([[float(i) for i in range(10)] for j in range(20)])
y_data = np.array([[float(j) for j in range(5)] for i in range(20)])

# print(x_data.shape)
# print(y_data.shape)
nn(x_data, y_data)


