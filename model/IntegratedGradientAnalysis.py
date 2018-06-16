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
    np.savetxt(fname="./top100_deeplift.csv",
               X=np.array(top_importances), delimiter=',')


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


def plot_distributions(feature_importance, path="../Results/IntegratedGradient/DistPlots/"):
    import seaborn as sns
    for i in range(feature_importance.shape[1]):
        plt.figure()
        sns.distplot(feature_importance[i])
        plt.xlabel("Feature Importance")
        plt.ylabel("Density")
        plt.title("Feature_{0} Distribution of Importance".format(i))
        plt.savefig(path + "feature_{0}.png".format(i))
        plt.close()


def plot_distribution(feature_importance, path="../Results/IntegratedGradient/"):
    file_name = "distribution.png"
    feature_importance = feature_importance.as_matrix()  # Convert to numpy ndarray
    new_shape = (feature_importance.shape[0] * feature_importance.shape[1], )
    feature_importance = np.reshape(feature_importance, newshape=new_shape)

    import seaborn as sns
    sns.distplot(feature_importance)
    plt.xlabel("Feature Importance")
    plt.ylabel("Density")
    plt.title("Distribution of all feature importances")
    plt.savefig(path + file_name)
    plt.close()


def box_plot(feature_importance, path="../Results/IntegratedGradient/"):
    pass


def calculate_statistical_criteria(feature_importance=None, criteria="absolute_error", path="../Results/IntegratedGradient/"):
    file_name = "intgrad_" + criteria + ".csv"
    if feature_importance:
        feature_importance = feature_importance.as_matrix()  # Convert to np.ndarray
        statistical_criteria = np.zeros(shape=(feature_importance.shape[1], 1))
        if criteria == "absolute_error":
            num_features = feature_importance.shape[1]
            statistical_criteria = np.array([[np.max(feature_importance[:, i]) - np.min(
                feature_importance[:, i])] for i in range(num_features)])
        elif criteria == "relative_error":
            statistical_criteria = np.array([[(np.max(feature_importance[:, i]) - np.min(
                feature_importance[:, i])) / (np.max(feature_importance[:, i]))]for i in range(feature_importance.shape[1])])
        np.savetxt(fname=path + file_name, X=statistical_criteria, delimiter=",")
        

def plot_statistical_criteria(criteria="absolute_error", data_path="../Results/IntegratedGradient/", save_path="../Results/IntegratedGradient/"):
    data_path = data_path + "intgrad_" + criteria + ".csv"
    save_path = save_path + "intgrad_" + criteria + ".png"
    statistical_criteria = pd.read_csv(data_path, header=None).as_matrix()

    import seaborn as sns
    sns.distplot(statistical_criteria)
    if criteria == "absolute_error":
        plt.xlabel("Absolute Error")
        plt.title("Distribution of Absolute Error")
    elif criteria == "relative_error":
        plt.xlabel("Relative Error")
        plt.title("Distribution of Relative Error")
    plt.ylabel("Density") 
    plt.savefig(save_path)
    plt.close()


if __name__ == '__main__':
    general_path = "../Results/IntegratedGradient/"

    data_path = "../Results/IntegratedGradient/integrated_gradients.csv"
    summary_path = general_path + "summary.csv"
    distplot_path = general_path + "distribution.png"

    # feature_importance = loadGradients(path=data_path)
    # print("Data has been loaded!")
    # save_summaries_for_each_feature(feature_importance)
    # print("Summaries has been written!")
    # plot_distributions(feature_importance)
    # print("Distplots are drawn!")
    # analyze_top_100_features_for_each_sample(feature_importance)
    # print("Top100.csv is made!")
    # plot_heatmaps(feature_importance)
    # print("Heatmaps are drawn!")
    # plot_distribution(feature_importance, path=distplot_path)
    # print("General Distribution has been drawn!")
    calculate_statistical_criteria(None, criteria="absolute_error")
    print("Statistical Criteria Calculation has been finished!")
    plot_statistical_criteria(criteria="absolute_error")
    print("Statistical Criteria Distribution plot has been drawn!")
    print("Finished!")
