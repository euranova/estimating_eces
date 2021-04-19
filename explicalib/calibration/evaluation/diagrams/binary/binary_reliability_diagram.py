# -*- coding: utf-8 -*-
"""
@author: nicolas.posocco
"""

from ..discrete_diagrams import reliability_diagram
import numpy as np


def binary_reliability_diagram(model=None, X=None, scores=None, Y=None, n_bins=None):
    if scores is None:

        # model
        assert hasattr(model, "predict_proba"), \
            "model must implement the predict_proba method."

        # X
        assert type(X) is np.ndarray, \
            "X should be an np.ndarray, not a {}.".format(type(X))
        assert X.ndim == 2, \
            "X should be a 1D array, not a {}D one.".format(X.ndim)

        scores = model.predict_proba(X)[:, 1]

    else:
        # model and X are ignored

        # scores
        assert type(scores) is np.ndarray, \
            "scores should be an np.ndarray, not a {}.".format(type(scores))
        assert scores.ndim == 1, \
            "scores should be a 2D array, not a {}D one.".format(scores.ndim)

    # Y
    assert type(Y) is np.ndarray, \
        "Y should be an np.ndarray, not a {}.".format(type(Y))
    assert Y.ndim == 1, \
        "Y should be a 1D array, not a {}D one.".format(Y.ndim)
    assert Y.shape[0] == scores.shape[0], \
        "Y and scores must have the same length, not {} and {}.".format(Y.shape[0], scores.shape[0])

    # n_bins
    assert type(n_bins) is int, \
        "n_bins should be an integer, not a {}.".format(type(n_bins))
    assert n_bins > 0, \
        "n_bins should be strictly positive, here is equal to {}.".format(n_bins)

    return reliability_diagram(label_of_interest=Y, scores=scores, n_bins=n_bins)