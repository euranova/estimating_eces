# -*- coding: utf-8 -*-
"""
@author: nicolas.posocco
"""

import numpy as np


class DecisionRule(object):

    def __init__(self):
        pass

    def __call__(self, batch):
        raise NotImplemented


class ThresholdBinaryDecisionRule(DecisionRule):

    def __init__(self, threshold):
        self.threshold = threshold
        super().__init__()

    def __call__(self, batch):
        """
        Computes predictions from scores matrix.
        Args:
            batch: np.ndarray of shape (n_samples, n_classes), score matrix.

        Returns:
        The prediction array, an np.ndarray of shape (n_samples, ).
        """

        assert type(batch) is np.ndarray
        assert batch.ndim == 2

        scores_class_1 = batch[:, 1]

        predictions = np.zeros(scores_class_1.shape[0])

        predictions[scores_class_1 < self.threshold] = 0
        predictions[scores_class_1 >= self.threshold] = 1

        return predictions
