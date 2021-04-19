# -*- coding: utf-8 -*-
"""
@author: nicolas.posocco
"""

import numpy as np
from tqdm import tqdm
from sklearn.utils import resample
from .binary_reliability_curve import binary_reliability_curve


def bootstrapped_binary_reliability_curve(model=None, X=None, scores=None, Y=None, bandwidth=0.025,
                                          confidence_interval=0.99, n_samplings=200, method="direct_convolution",
                                          kernel="cosine"):
    # model
    assert model is not None

    # Calculating scores for the positive class if not provided
    if scores is None:
        assert X is not None
        scores = model.predict_proba(X)[:, 1]

    else:

        # X is ignored

        assert type(scores) is np.ndarray
        assert scores.ndim == 1

    # n_samplings
    assert n_samplings is not None

    result = {}

    # Fixing the randomness used for the bootstraping
    reproductible = np.random.RandomState(8)

    # Calculating reliability curve on each bootstrap
    trajectories = []
    for _ in tqdm(range(n_samplings)):
        # Sampling a subdataset via bootstrap
        scores_b, y_test_b = resample(scores, Y, random_state=reproductible)

        # Calculating and saving curve
        diagram = binary_reliability_curve(model=model, positive_scores=scores_b, Y=y_test_b,
                                           bandwidth=bandwidth,  kernel=kernel)
        trajectories.append(diagram)

    calibration_error_plots = np.array([diagram["errors"] for diagram in trajectories])

    # Calculating median trajectory and uncertainty zone
    semi_uncertainty_mass = 0.5 * (1 - confidence_interval)
    median, quantile_low, quantile_high = np.quantile(calibration_error_plots, axis=0,
                                                      q=[0.5, semi_uncertainty_mass, 1 - semi_uncertainty_mass])

    result["median"] = median
    result["quantile_low"] = quantile_low
    result["quantile_high"] = quantile_high
    result["confidence_interval"] = confidence_interval
    result["scores"] = scores

    return result
