from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import mutual_info_score
import pandas as pd
import numpy as np
"""
    Created by Mohsen Naghipourfar on 5/13/18.
    Email : mn7697np@gmail.com
    Website: http://ce.sharif.edu/~naghipourfar
"""


def MI_between_features_and_classes(F, D):  # F is feature matrix with shape (n_samples, n_features)
    return mutual_info_classif(F, D)

DAMAVAND_LOCATION_X = "~/f/Behrooz/dataset_local/fpkm_normalized.csv"
DAMAVAND_LOCATION_Y = "~/f/Behrooz/dataset_local/disease.csv"

N_SAMPLES = 10787
N_FEATURES = 19671
N_DISEASES = 34


data = pd.read_csv(DAMAVAND_LOCATION_X, header=None)
labels = pd.read_csv(DAMAVAND_LOCATION_Y, header=None)

# random_feature_indices = np.random.choice(19671, 10, replace=False)
x_train = data
y = labels[0].iloc[:,].astype('category').cat.codes

mi_feature_class = mutual_info_classif(x_train, y)
np.savetxt("./MI_F_D.csv", mi_feature_class, delimiter=",")