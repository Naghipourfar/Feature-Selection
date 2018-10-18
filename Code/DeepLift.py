import numpy as np
import pandas as pd

import tensorflow as tf
import matplotlib.pyplot as plt
import keras

from keras.layers import Input, Dense, Dropout, BatchNormalization
from keras.models import Model, Sequential
from keras.callbacks import History, CSVLogger

import deeplift
from deeplift.conversion import kerasapi_conversion as kc
from deeplift.blobs import NonlinearMxtsMode
from deeplift.util import get_integrated_gradients_function

"""
    Created by Mohsen Naghipourfar on 6/14/18.
    Email : mn7697np@gmail.com or naghipourfar@ce.sharif.edu
    Website: http://ce.sharif.edu/~naghipourfar
    Github: https://github.com/naghipourfar
    Skype: mn7697np
"""

# Code = keras.models.load_model("./classifier.h5")

deeplift_model = kc.convert_model_from_saved_files("./classifier-noBatchNorm-noGaussian.h5",
                                                   nonlinear_mxts_mode=NonlinearMxtsMode.Gradient)
print(deeplift_model.get_name_to_layer().keys())

gradient_function = deeplift_model.get_target_multipliers_func(find_scores_layer_name="input_1_0",
                                                               pre_activation_target_layer_name="preact_dense_5_0")

integrated_gradient_5 = get_integrated_gradients_function(gradient_function, 5)

x = pd.read_csv("../Data/fpkm_normalized.csv", header=None)

for task_idx in range(1):
    print("\tComputing scores for task: " + str(task_idx))
    scores = np.array(integrated_gradient_5(
        task_idx=task_idx,
        input_data_list=[x],
        input_references_list=[np.zeros_like(x)],
        batch_size=1000,
        progress_update=None))
    print(scores.shape)
    print(scores)
    np.savetxt(fname="./deeplift_scores.csv", X=scores, delimiter=",")
print("Finished!")
