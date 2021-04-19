# -*- coding: utf-8 -*-
"""
@author: nicolas.posocco
"""

from sklearn.svm import SVC
import numpy as np


class SVCc(SVC):

    def __init__(self, **kwargs):
        kwargs["probability"] = True
        super().__init__(**kwargs)

    def predict(self, X):
        scores_matrix = self.predict_proba(X)
        return np.argmax(scores_matrix, axis=1)
