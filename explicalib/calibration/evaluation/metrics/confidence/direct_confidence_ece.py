# -*- coding: utf-8 -*-
"""
@author: nicolas.posocco
"""

import numpy as np
from .....utils import confidences_from_scores
from .....utils import bounded_1d_kde, get_silvermans_bandwidth


def direct_confidence_ece(model=None, X=None, confidence_scores=None, predictions=None, Y=None,
                          bandwidth=0.05, kernel="triweight",
                          positive_class_probability=None):
    """

    Args:
        model: TODO.
        X: TODO.
        positive_scores: TODO.
        Y: TODO.
        bandwidth: TODO.
        positive_class_probability: TODO.

    Returns:
    TODO.
    """

    if confidence_scores is None:
        # model
        assert hasattr(model, "predict_proba")

        scores = model.predict_proba(X)
        predictions = model.predict(X)
        positive_scores = confidences_from_scores(model=model, predictions=predictions,
                                                  scores_matrix=scores)
    else:
        positive_scores = confidence_scores

    if positive_class_probability is None:
        positive_class_probability = np.sum(Y == predictions) / Y.shape[0]

    if bandwidth == "silverman":
        bandwidth = get_silvermans_bandwidth(X=positive_scores, kernel=kernel, bandwidth=bandwidth)

    # Defining the grid on which the density is estimated
    # (its support is [1/n_class, 1], yet we need to perform mirroring to constraint the kde to this domain)
    n_classes = len(model.classes_)
    grid = np.linspace(1 / n_classes - 3, 1 + 3, 2 ** 14)

    ps_x, ps_y = bounded_1d_kde(positive_scores,
                                low_bound=1 / n_classes, high_bound=1,
                                kernel=kernel,
                                bandwidth=bandwidth,
                                output="discrete_signal",
                                grid=grid)

    # Rectification to avoid division by zero bellow
    epsilon = 1e-8
    ps_y_no_null = np.copy(ps_y)
    ps_y_no_null[ps_y < epsilon] = epsilon

    positive_scores_for_positive_gt = positive_scores[Y == predictions]

    _, psky1_y = bounded_1d_kde(positive_scores_for_positive_gt,
                                low_bound=1 / n_classes, high_bound=1,
                                kernel=kernel,
                                bandwidth=bandwidth,
                                output="discrete_signal",
                                grid=grid)

    py1ks_y = psky1_y * positive_class_probability / ps_y_no_null

    # Rectification : zero score density -> no calibration error prior
    # Unnecessary here since its weight is zero bellow
    # py1ks_y[ps_y == epsilon] = ps_x[ps_y == epsilon]

    local_ce_y = py1ks_y - ps_x
    abs_local_ce_y = np.abs(local_ce_y)

    # Calculating the ECE
    metric_value = np.average(abs_local_ce_y, weights=ps_y)

    return metric_value
