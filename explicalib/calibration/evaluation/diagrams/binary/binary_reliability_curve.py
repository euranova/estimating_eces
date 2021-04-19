# -*- coding: utf-8 -*-
"""
@author: nicolas.posocco
"""

from .....utils import bounded_1d_kde
import numpy as np


def binary_reliability_curve(model=None, X=None, Y=None,
                             kernel=None, bandwidth=None,
                             positive_scores=None,
                             positive_scores_for_positive_gt=None,
                             positive_class_probability=None):
    if positive_scores is None:
        # model
        assert model is not None, "model has to be provided when positive_scores is not."
        assert hasattr(model, "predict_proba")

        # X
        assert X is not None, "X has to be provided when positive_scores is not."
        assert type(X) is np.ndarray
        assert X.ndim == 2

        positive_scores = model.predict_proba(X)[:, 1]

    if positive_class_probability is None:
        # Y
        assert Y is not None, "Y has to be provided when positive_class_probability is not."
        assert type(Y) is np.ndarray
        assert Y.ndim == 1

        # model
        assert model is not None, "model has to be provided when positive_scores is not."
        assert hasattr(model, "classes_")

        unique, counts = np.unique(Y, return_counts=True)
        counts = dict(zip(unique, counts))
        positive_class_probability = counts[model.classes_[1]] / Y.shape[0]

    if positive_scores_for_positive_gt is None:
        # Y
        assert Y is not None, "Y has to be provided when positive_scores_for_positive_gt is not."
        assert type(Y) is np.ndarray
        assert Y.ndim == 1

        # model
        assert model is not None, "model has to be provided when positive_scores_for_positive_gt is not."
        assert hasattr(model, "classes_")

        positive_scores_for_positive_gt = positive_scores[Y == model.classes_[1]]

    # kernel
    assert kernel is not None

    # Defining the grid on which the density is estimated
    # (its support is [0, 1], yet we need to perform mirroring to constraint the kde to this domain)
    grid = np.linspace(-3, 4, 2 ** 20)

    # Estimating the density of P( S+ )
    s, ps_y = bounded_1d_kde(positive_scores,
                             low_bound=0, high_bound=1,
                             kernel=kernel, bandwidth=bandwidth,
                             output="discrete_signal",
                             grid=grid)

    # Rectification to avoid division by zero bellow
    epsilon = 1e-8
    ps_y[ps_y < epsilon] = epsilon

    # Estimating the density of P( S+ | Y=+ )
    _, psky1_y = bounded_1d_kde(positive_scores_for_positive_gt,
                                low_bound=0, high_bound=1,
                                kernel=kernel, bandwidth=bandwidth,
                                output="discrete_signal",
                                grid=grid)

    # Estimating the density of P( Y=+ | S+ )
    # using bayes inversion of the density
    py1ks_y = psky1_y * positive_class_probability / ps_y

    # Rectification : zero score density -> no calibration error prior
    py1ks_y[ps_y == epsilon] = s[ps_y == epsilon]

    reliability_curve = {"scores": s, "errors": py1ks_y}

    return reliability_curve

