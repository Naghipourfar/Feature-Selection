import numpy as np
import pandas as pd

import tensorflow as tf
import matplotlib.pyplot as plt
import keras

from keras.layers import Input, Dense, Dropout, BatchNormalization
from keras.models import Model
from keras.callbacks import History, CSVLogger

"""
    Created by Mohsen Naghipourfar on 6/15/18.
    Email : mn7697np@gmail.com or naghipourfar@ce.sharif.edu
    Website: http://ce.sharif.edu/~naghipourfar
    Github: https://github.com/naghipourfar
    Skype: mn7697np
"""


def loadGradients(path="./integrated_gradients.csv"):
    return pd.read_csv(path, header=None)


def save_summaries_for_each_feature(feature_importance, path="./IntegratedGradient/Summaries/"):
    for i in range(feature_importance.shape[1]):
        description = feature_importance[i].describe()
        description.to_csv(path + "feature_{0}.txt".format(i))


def plot_distributions(feature_importance, path="./IntegratedGradient/DistPlots/"):
    import seaborn as sns
    for i in range(feature_importance.shape[1]):
        plt.figure()
        sns.distplot(feature_importance[i])
        plt.xlabel("Feature Importance")
        plt.ylabel("Density")
        plt.title("Feature_{0} Distribution of Importance".format(i))
        plt.savefig(path + "feature_{0}.png".format(i))
        plt.close()


if __name__ == '__main__':
    feature_importance = loadGradients()
    save_summaries_for_each_feature(feature_importance)
    plot_distributions(feature_importance)
