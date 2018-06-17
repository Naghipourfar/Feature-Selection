import numpy as np
import pandas as pd

import sys
import os

import tensorflow as tf
import matplotlib.pyplot as plt
import keras

from keras.layers import Input, Dense, Dropout, BatchNormalization
from keras.models import Model
from keras.callbacks import History, CSVLogger

from IntegratedGradient import integrated_gradients


"""
    Created by Mohsen Naghipourfar on 6/15/18.
    Email : mn7697np@gmail.com or naghipourfar@ce.sharif.edu
    Website: http://ce.sharif.edu/~naghipourfar
    Github: https://github.com/naghipourfar
    Skype: mn7697np
"""


def loadGradients(path="../Results/IntegratedGradient/integrated_gradients.csv"):
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
    feature_importance = feature_importance.as_matrix()  # Convert to np.ndarray
    statistical_criteria = np.zeros(shape=(feature_importance.shape[1], 1))
    if criteria == "absolute_error":
        num_features = feature_importance.shape[1]
        statistical_criteria = np.array([[np.max(feature_importance[:, i]) - np.min(
            feature_importance[:, i])] for i in range(num_features)])
    elif criteria == "relative_error":
        statistical_criteria = np.array([[(np.max(feature_importance[:, i]) - np.min(
            feature_importance[:, i])) / (np.max(feature_importance[:, i]))]for i in range(feature_importance.shape[1])])
    np.savetxt(fname=path + file_name,
                X=statistical_criteria, delimiter=",")


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


def make_summary_data(feature_importance, path="../Results/IntegratedGradient/"):
    file_name = "summaries.csv"
    feature_importance = feature_importance
    num_features = feature_importance.shape[1]
    all_describtions = np.zeros(
        shape=(num_features, 4))  # mean - std - min - max
    for i in range(num_features):
        describtion = feature_importance[i].describe()
        describtion = describtion.iloc[[1, 2, 3, 7]].as_matrix()
        all_describtions[i] = describtion.T
    print(all_describtions.shape)
    np.savetxt(fname=path + file_name, X=all_describtions, delimiter=',')


def compute_integrated_gradient(machine="damavand", save_path="../Results/IntegratedGradient/", verbose=1):
    file_name = "integrated_gradient.csv"

    if machine == "damavand":
        mrna_address = "~/f/Behrooz/dataset_local/fpkm_normalized.csv"
    else:
        mrna_address = "../Data/fpkm_normalized.csv"

    m_rna = pd.read_csv(mrna_address, header=None)
    model = keras.models.load_model("../Results/classifier.h5")

    ig = integrated_gradients(model, verbose=verbose)
    num_samples = m_rna.shape[0]
    num_features = m_rna.shape[1]

    feature_importances = np.zeros(shape=(num_samples, num_features))
    for i in range(num_samples):
        feature_importances[i] = ig.explain(m_rna.as_matrix()[i, :])
        if verbose == 1:
            sys.stdout.flush()
            sys.stdout.write('\r')
            sys.stdout.write("Progress: " + str((i / 10787) * 100) + " %")
            sys.stdout.flush()
    if verbose == 1:
        sys.stdout.flush()
        sys.stdout.write('\r')
        sys.stdout.write("Progress: " + str((10787 / 10787) * 100) + " %")
        sys.stdout.flush()

    np.savetxt(fname="../Results/IntegratedGradient/integrated_gradients.csv",
               X=np.array(feature_importances), delimiter=',')

    return pd.DataFrame(feature_importances)


machine = "damavand"

if __name__ == '__main__':
    general_path = "../Results/IntegratedGradient/"

    data_path = general_path + "integrated_gradients.csv"
    summary_path = general_path + "summary.csv"
    distplot_path = general_path + "distribution.png"

    if os.path.exists(data_path):
        feature_importance = loadGradients(path=data_path)
        print("Data has been loaded!")
    else:
        feature_importance = compute_integrated_gradient(
            machine=machine, save_path=data_path, verbose=1)
        print("Data has been computed and saved!")

    plot_distribution(feature_importance, path=distplot_path)
    print("General Distribution has been drawn!")

    calculate_statistical_criteria(feature_importance, criteria="absolute_error")
    print("Statistical Criteria AE Calculation has been finished!")
    plot_statistical_criteria(criteria="absolute_error")
    print("Statistical Criteria AE Distribution plot has been drawn!")

    # calculate_statistical_criteria(None, criteria="relative_error")
    # print("Statistical Criteria RE Calculation has been finished!")
    # plot_statistical_criteria(criteria="relative_error")
    # print("Statistical Criteria RE Distribution plot has been drawn!")

    make_summary_data(feature_importance)
    print("Summary of all features has been made!")
    print("Finished!")
    
