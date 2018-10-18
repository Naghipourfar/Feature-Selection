import numpy as np
import pandas as pd

import tensorflow as tf
import matplotlib.pyplot as plt
from feature_selection import FeatureSelection
from network_keras import run, modify_output
import keras
"""
    Created by Mohsen Naghipourfar on 5/21/18.
    Email : mn7697np@gmail.com
    Website: http://ce.sharif.edu/~naghipourfar
"""


def main():
    LOCAL_LOCATION_X = "../Data/fpkm_normalized.csv"
    LOCAL_LOCATION_Y = "../Data/disease.csv"

    print("Loading Data...")
    features = pd.read_csv(LOCAL_LOCATION_X, header=None)
    labels = pd.read_csv(LOCAL_LOCATION_Y, header=None)
    print("Data Loaded!")

    feature_selection = FeatureSelection('mRMR', 5, features, labels)
    feature_indices = feature_selection.get_best_features()
    # feature_indices = [4929, 5345, 16381, 13656]
    print("Features has been selected!")

    selected_features = features[feature_indices]
    labels = pd.DataFrame(modify_output(labels))
    labels = pd.DataFrame(keras.utils.to_categorical(labels, num_classes=34))
    run(0, selected_features, labels, False)


if __name__ == '__main__':
    main()
