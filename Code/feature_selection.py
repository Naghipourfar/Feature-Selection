import numpy as np
import pandas as pd

import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import mutual_info_score
from sklearn.feature_selection import mutual_info_classif
import os

"""
    Created by Mohsen Naghipourfar on 5/20/18.
    Email : mn7697np@gmail.com
    Website: http://ce.sharif.edu/~naghipourfar
"""


class FeatureSelection(object):
    """
        FeatureSelection stands for mutual information based features selection methods.
        This class contains routines for selecting features using both
        continuous and discrete y variables. Three selection algorithms are
        implemented: mRMR, JMI, MIFS, NMIFS_FS2, IWFS, DCSF


        Parameters
        ----------

        method : string, default = 'JMI'
            Which mutual information based feature selection method to use:
            - 'mRMR' : Max-Relevance Min-Redundancy [1]
            - 'JMI' : Joint Mutual Information [2]
            - 'MIFS' : Mutual Information based Feature Selection [3]
            - 'NMIFS_FS2' : Mutual Information based Feature Selection (Modified) TODO Correction [4]
            - 'IWFS' : TODO Complete [5]
            - 'DCSF' : TODO Complete [6]

        k : int, default = 5
            Sets the number of features to be selected.

        features : pd.DataFrame, default = None
            features dataframe with shape (n_samples, n_features)

        target : pd.DataFrame, default = None
            target Dataframe with shape (n_samples, )


        Example
        --------

        import pandas as pd
        from FeatureSelection import FeatureSelection

        # load X and y
        X = pd.read_csv('my_X_table.csv', index_col=0).values
        y = pd.read_csv('my_y_vector.csv', index_col=0).values

        # define mRMR feature selection method
        feat_selector = FeatureSelection(method='mRMR', k=5, features=X, target=y)

        # find all relevant features
        features_indices = feat_selector.get_best_features()

        References
        ----------

        [1] H. Peng, Fulmi Long, C. Ding, "Feature selection based on mutual
            information criteria of max-dependency, max-relevance,
            and min-redundancy"
            Pattern Analysis & Machine Intelligence 2005

        [2] H. Yang and J. Moody, "Data Visualization and Feature Selection: New
            Algorithms for Nongaussian Data"
            NIPS 1999

        [3] Bennasar M., Hicks Y., Setchi R., "Feature selection using Joint Mutual
            Information Maximisation"
            Expert Systems with Applications, Vol. 42, Issue 22, Dec 2015

        """

    def __init__(self, method='mRMR', k=5, features=None, target=None):
        self.method = method
        self.k = k
        self.features = features
        self.target = target
        self.number_of_features = features.shape[1]

    def _select_features(self):
        if self.method == 'mRMR':
            return self._mRMR()
        elif self.method == 'JMI':
            return self._JMI()
        elif self.method == 'MIFS':
            return self._MIFS()
        elif self.method == 'NMIFS_FS2':
            return self._NMIFS_FS2()
        elif self.method == 'IWFS':
            return self._IWFS()
        elif self.method == 'DCSF':
            return self._DCSF()
        else:
            return Exception("Invalid Feature Selection Method!")

    def get_best_features(self):
        return self._select_features()

    def _mRMR(self):
        S = []
        F = [i for i in range(self.number_of_features)]
        x_train = self.features
        # y = labels[0].iloc[:, ].astype('category').cat.codes

        if os.path.isfile('../MI_Analysis/MI_FD.csv'):
            print("MI_FD Exists!")
            mi_features_classes = pd.read_csv('../MI_Analysis/MI_FD.csv', header=None).as_matrix()
        else:
            print("MI_FD doesn't Exists!")
            mi_features_classes = self._calculate_FD_MI()

        max_value, idx = np.amax(mi_features_classes), np.argmax(mi_features_classes)
        S.append(idx)
        F.__delitem__(idx)
        print("{1}: Feature {0} has been added to S".format(idx, 1))
        for i in range(self.k - 1):
            max_phi, max_idx = -10000, None
            for idx in F:
                feature = x_train[idx]
                dependency = mi_features_classes[idx]
                redundancy = (1.0 / len(S)) * sum([mutual_info_score(feature, x_train[f]) for f in S])
                phi = dependency - redundancy
                if phi > max_phi:
                    max_phi = phi
                    max_idx = idx
            S.append(max_idx)
            F.__delitem__(max_idx)
            print("{1}: Feature {0} has been added to S".format(max_idx, i + 2))

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

    def _calculate_pairwise_MI(self):
        mutual_info_matrix = [[0 for _ in range(self.number_of_features)] for __ in range(self.number_of_features)]
        x_train = self.features
        for i in range(self.number_of_features):
            feature_1 = x_train[i]
            for j in range(i, self.number_of_features):
                feature_2 = x_train[j]
                mutual_information = mutual_info_score(feature_1, feature_2)
                mutual_info_matrix[i][j] = mutual_information
                mutual_info_matrix[j][i] = mutual_information
        np.savetxt('../MI_Analysis/MI_pairwise.csv', np.array(mutual_info_matrix), delimiter=',')
        return mutual_info_matrix

    def _calculate_FD_MI(self):
        # random_feature_indices = np.random.choice(19671, 10, replace=False)
        x_train = self.features
        y = self.target[0].iloc[:, ].astype('category').cat.codes

        mi_features_classes = mutual_info_classif(x_train, y)
        np.savetxt("./MI_Analysis/MI_FD.csv", mi_features_classes, delimiter=",")
        return mi_features_classes
