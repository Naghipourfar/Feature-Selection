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


def analyze_top_100_features_for_each_sample(feature_importance):
    top_importances = []
    for i in range(feature_importance.shape[0]):
        importances = feature_importance.iloc[i, :]
        importances = list(reversed(sorted(abs(importances))))
        top_100_importances = importances[:100]
        top_importances.append(top_100_importances)
    np.savetxt(fname="./top100.csv", X=np.array(top_importances), delimiter=',')


def plot_heatmaps(feature_importances, path="./IntegratedGradient/Heatmaps/"):
    plt.rcParams["figure.figsize"] = 5, 2

    for i in range(feature_importances.shape[0]):
        y = feature_importances.iloc[i, :]
        fig, ax = plt.subplots(nrows=1, sharex='all')
        # extent = [x[0] - (x[1] - x[0]) / 2., x[-1] + (x[1] - x[0]) / 2., 0, 1]
        heatmap = ax.imshow(y[np.newaxis, :], cmap="plasma", aspect="auto")
        ax.set_yticks([])
        # ax.set_xlim(extent[0], extent[1])
        plt.tight_layout()
        plt.colorbar(heatmap)
        plt.savefig(path + "sample_{0}.png".format(i))
        plt.close()


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
    print("Data has been loaded!")
    # save_summaries_for_each_feature(feature_importance)
    # plot_distributions(feature_importance)
    # analyze_top_100_features_for_each_sample(feature_importance)
    plot_heatmaps(feature_importance)
