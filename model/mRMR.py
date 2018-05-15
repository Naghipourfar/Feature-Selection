import numpy as np
import pandas as pd

import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import mutual_info_score

"""
    Created by Mohsen Naghipourfar on 5/13/18.
    Email : mn7697np@gmail.com
    Website: http://ce.sharif.edu/~naghipourfar
"""

DAMAVAND_LOCATION_X = "~/f/Behrooz/dataset_local/fpkm_normalized.csv"
DAMAVAND_LOCATION_Y = "~/f/Behrooz/dataset_local/disease.csv"

N_SAMPLES = 10787
N_FEATURES = 19671
N_DISEASES = 34

print("Loading Data...")
data = pd.read_csv(DAMAVAND_LOCATION_X, header=None)
labels = pd.read_csv(DAMAVAND_LOCATION_Y, header=None)
print("Data Loaded!")


def mRMR(number_of_selected_features):
    S = []
    F = [i for i in range(N_FEATURES)]
    x_train = data
    # y = labels[0].iloc[:, ].astype('category').cat.codes

    mi_features_classes = pd.read_csv('./MI_F_D.csv', header=None).as_matrix()
    max_value, idx = np.amax(mi_features_classes), np.argmax(mi_features_classes)
    S.append(idx[0])
    F.__delitem__(idx[0])
    print("Feature {0} has been added to S".format(idx[0]))
    sum_mi = 0
    for i in range(number_of_selected_features - 1):
        max_phi, max_idx = -10000, None
        for idx in F:
            feature = x_train[idx]
            dependency = mi_features_classes[idx]
            redundency = (1.0 / len(S)) * sum([mutual_info_score(feature, x_train[f]) for f in S])
            phi = dependency - redundency
            if phi > max_phi:
                max_phi = phi
                max_idx = idx
        S.append(max_idx)
        F.__delitem__(max_idx)
        print("Feature {0} has been added to S".format(max_idx))

    return S


if __name__ == '__main__':
    S = mRMR(200)
    np.savetxt('./mRMR.csv', np.array(S), delimiter=',')
    print("Finished!")
