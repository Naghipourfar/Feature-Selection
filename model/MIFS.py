import numpy as np
import pandas as pd

import tensorflow as tf
import matplotlib.pyplot as plt
import mifs

"""
    Created by Mohsen Naghipourfar on 5/13/18.
    Email : mn7697np@gmail.com
    Website: http://ce.sharif.edu/~naghipourfar
"""
DAMAVAND_LOCATION_X = "~/f/Behrooz/dataset_local/fpkm_normalized.csv"
DAMAVAND_LOCATION_Y = "~/f/Behrooz/dataset_local/disease.csv"
LOCAL_LOCATION_X = "../Data/fpkm_normalized.csv"
LOCAL_LOCATION_Y = "../Data/disease.csv"

N_SAMPLES = 10787
N_FEATURES = 19671
N_DISEASES = 34

data = pd.read_csv(DAMAVAND_LOCATION_X, header=None)
labels = pd.read_csv(DAMAVAND_LOCATION_Y, header=None)

random_feature_indices = np.random.choice(19671, 1000, replace=False)
all_feature_indices = [i for i in range(19671)]
x_train = data

y = labels[0].iloc[:N_SAMPLES, ].astype('category').cat.codes
X = x_train.iloc[:N_SAMPLES, :]

# define MI_FS feature selection method
feature_selector = mifs.MutualInformationFeatureSelector(method='JMI', k=int(y.max()), n_features=200, verbose=2)

# find all relevant features
feature_selector.fit(X, y)

MI = feature_selector.mi_
with open('./MI.txt', 'a') as file:
    for i in range(len(feature_selector.mi_)):
        file.write("{0} : {1}\n".format(x_train.columns[feature_selector.ranking_[i]], MI[i]))
