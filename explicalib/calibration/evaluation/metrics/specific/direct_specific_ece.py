# -*- coding: utf-8 -*-
"""
@author: nicolas.posocco
"""

import numpy as np
from .....utils import bounded_1d_kde, get_silvermans_bandwidth


def direct_specific_ece(model=None, X=None, specific_scores=None, Y=None, kernel="triweight", bandwidth=None,
                        class_index=None, backend=None):
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

    if specific_scores is None:
        assert class_index is not None
        specific_scores = model.predict_proba(X)[:, class_index]

    unique, counts = np.unique(Y, return_counts=True)
    counts = dict(zip(unique, counts))
    if model.classes_[class_index] in counts:
        specific_class_probability = counts[model.classes_[class_index]] / Y.shape[0]
    else:
        # In case the specific class isn't represented
        # The following is the optimized body of the current function, with specific_class_probability = 0.

        grid = np.linspace(-2.5, 3.5, 2 ** 14)

        ps_x, ps_y = bounded_1d_kde(specific_scores,
                                    low_bound=0, high_bound=1,
                                    kernel=kernel,
                                    bandwidth=bandwidth,
                                    output="discrete_signal",
                                    grid=grid)
        local_ce_y = - ps_x
        abs_local_ce_y = np.abs(local_ce_y)

        # Calculating the ECE
        metric_value = np.average(abs_local_ce_y, weights=ps_y)

        return metric_value

    if bandwidth == "silverman":
        bandwidth = get_silvermans_bandwidth(X=specific_scores, kernel=kernel, bandwidth=bandwidth)

    grid = np.linspace(-2.5, 3.5, 2 ** 14)

    ps_x, ps_y = bounded_1d_kde(specific_scores,
                                low_bound=0, high_bound=1,
                                kernel=kernel,
                                bandwidth=bandwidth,
                                output="discrete_signal",
                                grid=grid)

    # Rectification to avoid division by zero bellow
    epsilon = 1e-8
    ps_y_no_null = np.copy(ps_y)
    ps_y_no_null[ps_y < epsilon] = epsilon

    specific_scores_for_specific_gt = specific_scores[Y == model.classes_[class_index]]

    _, pskyc_y = bounded_1d_kde(specific_scores_for_specific_gt,
                                low_bound=0, high_bound=1,
                                kernel=kernel,
                                bandwidth=bandwidth,
                                output="discrete_signal",
                                grid=grid)

    py1ks_y = pskyc_y * specific_class_probability / ps_y_no_null

    # Rectification : zero score density -> no calibration error prior
    # Unnecessary here since its weight is zero bellow
    py1ks_y[ps_y == epsilon] = ps_x[ps_y == epsilon]

    local_ce_y = py1ks_y - ps_x
    abs_local_ce_y = np.abs(local_ce_y)

    # Calculating the ECE
    metric_value = np.average(abs_local_ce_y, weights=ps_y)

    return metric_value
