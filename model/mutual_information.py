from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import mutual_info_score
import pandas as pd
import numpy as np

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


def mi_feature_class():
    # random_feature_indices = np.random.choice(19671, 10, replace=False)
    x_train = data
    y = labels[0].iloc[:, ].astype('category').cat.codes

    mi_features_classes = mutual_info_classif(x_train, y)
    np.savetxt("./MI_F_D.csv", mi_features_classes, delimiter=",")


def mi_pairwise():
    mutual_info_matrix = [[0 for _ in range(N_FEATURES)] for __ in range(N_FEATURES)]
    x_train = data
    for i in range(N_FEATURES):
        feature_1 = x_train[i]
        for j in range(i, N_FEATURES):
            feature_2 = x_train[j]
            mutual_information = mutual_info_score(feature_1, feature_2)
            mutual_info_matrix[i][j] = mutual_information
            mutual_info_matrix[j][i] = mutual_information
    np.savetxt('./pairwise_MI.csv', np.array(mutual_info_matrix), delimiter=',')


if __name__ == '__main__':
    mi_pairwise()
    print("Finished!")
