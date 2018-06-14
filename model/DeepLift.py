import numpy as np
import pandas as pd

import tensorflow as tf
import matplotlib.pyplot as plt
import keras

from keras.layers import Input, Dense, Dropout, BatchNormalization
from keras.models import Model
from keras.callbacks import History, CSVLogger

import deeplift
from deeplift.conversion import keras_conversion as kc
from deeplift.blobs import NonlinearMxtsMode
from deeplift.util import get_integrated_gradients_function

"""
    Created by Mohsen Naghipourfar on 6/14/18.
    Email : mn7697np@gmail.com or naghipourfar@ce.sharif.edu
    Website: http://ce.sharif.edu/~naghipourfar
    Github: https://github.com/naghipourfar
    Skype: mn7697np
"""


model = keras.models.load_model("./classifier.h5")
model.summary()
deeplift_model = kc.convert_functional_model(model, nonlinear_mxts_mode=NonlinearMxtsMode.Gradient)
gradient_function = deeplift_model.get_target_multipliers_func(find_scores_layer_idx=0, target_layer_idx=-2)

integrated_gradient_5 = get_integrated_gradients_function(gradient_function, 5)