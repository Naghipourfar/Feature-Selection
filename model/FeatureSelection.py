import numpy as np
import pandas as pd

import tensorflow as tf
import matplotlib.pyplot as plt

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

    def _select_method(self):
        if self.method == 'mRMR':
            self._mRMR()

    def _mRMR(self):
        pass

    def _MIFS(self):
        pass

    def _NMIFS_FS2(self):
        pass

