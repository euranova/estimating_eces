# -*- coding: utf-8 -*-
"""
@author: nicolas.posocco
"""

import numpy as np


def google_metric_formulation(model, X, Y):
    """
    Turns module formulation into one adapted to google uncertainty_metrics module.
    Args:
        model: a model following the standard sklearn api.
        X: 2D numpy.ndarray (n_samples, dimensionality)
        Y: 1D numpy.ndarray (n_samples, )

    Returns:
    Google's uncertainty_metrics module input arguments for metrics.
    """

    # model
    assert hasattr(model, "predict_proba")

    # X
    assert type(X) is np.ndarray
    assert X.ndim == 2

    # Y
    assert type(Y) is np.ndarray
    assert Y.ndim == 1

    # X & Y compatibility
    assert X.shape[0] == Y.shape[0]

    probabilities = model.predict_proba(X)

    labels = np.zeros(shape=Y.shape, dtype=np.int16)
    labels[Y == model.classes_[0]] = 0
    labels[Y == model.classes_[1]] = 1

    return probabilities, labels
