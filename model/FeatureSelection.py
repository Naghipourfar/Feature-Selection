import numpy as np
import pandas as pd

import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutu

"""
    Created by Mohsen Naghipourfar on 5/20/18.
    Email : mn7697np@gmail.com
    Website: http://ce.sharif.edu/~naghipourfar
"""


class FeatureSelection(object):
    def __init__(self, method, k, features, target):
        self.method = method
        self.k = k
        self.features = features
        self.target = target
        self.number_of_features = features.shape[1]

    def _select_method(self):
        if self.method == 'mRMR':
            self._mRMR()

    def _mRMR(self):
        # S = []
        # F = [i for i in range(self.number_of_features)]
        # x_train = self.features
        # # y = labels[0].iloc[:, ].astype('category').cat.codes
        #
        # mi_features_classes = pd.read_csv('./MI_F_D.csv', header=None).as_matrix()
        # max_value, idx = np.amax(mi_features_classes), np.argmax(mi_features_classes)
        # S.append(idx[0])
        # F.__delitem__(idx[0])
        # print("Feature {0} has been added to S".format(idx[0]))
        # sum_mi = 0
        # for i in range(self.k - 1):
        #     max_phi, max_idx = -10000, None
        #     for idx in F:
        #         feature = x_train[idx]
        #         dependency = mi_features_classes[idx]
        #         redundency = (1.0 / len(S)) * sum([mutual_info_score(feature, x_train[f]) for f in S])
        #         phi = dependency - redundency
        #         if phi > max_phi:
        #             max_phi = phi
        #             max_idx = idx
        #     S.append(max_idx)
        #     F.__delitem__(max_idx)
        #     print("Feature {0} has been added to S".format(max_idx))

        return S

    def _MIFS(self):
        pass

    def _NMIFS_FS2(self):
        pass

    def _JMI(self):
        pass

    def _IWFS(self):
        pass

    def _DCSF(self):
        pass

