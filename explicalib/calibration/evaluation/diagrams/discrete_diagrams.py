# -*- coding: utf-8 -*-
"""
@author: nicolas.posocco
"""

from sklearn.calibration import calibration_curve
import numpy as np


def reliability_diagram(label_of_interest, scores, n_bins):
    # label_of_interest
    assert type(label_of_interest) is np.ndarray, \
        "label_of_interest should be an np.ndarray, not a {}.".format(type(label_of_interest))
    assert label_of_interest.ndim == 1, \
        "label_of_interest should be a 1D array, not a {}D one.".format(label_of_interest.ndim)
    assert label_of_interest.shape[0] > 0, \
        "label_of_interest can't be an empty array."

    # scores
    assert type(scores) is np.ndarray, \
        "scores should be an np.ndarray, not a {}.".format(type(scores))
    assert scores.ndim == 1, \
        "scores should be a 1D array, not a {}D one.".format(scores.ndim)
    assert scores.shape == label_of_interest.shape, \
        "scores and label_of_interest should share a common shape, here they are respectively {} and {}.".format(
            scores.shape, label_of_interest.shape)

    # Calculation via sklearn
    fraction_of_positives, mean_predicted_value = calibration_curve(label_of_interest, scores, n_bins=n_bins)

    return {"fraction_of_positives": fraction_of_positives,
            "mean_predicted_value": mean_predicted_value}
