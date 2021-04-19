# -*- coding: utf-8 -*-
"""
@author: nicolas.posocco
"""

from .....utils import get_silvermans_bandwidth
from ..specific.direct_specific_ece import direct_specific_ece


def direct_classwise_ece(model=None, scores=None, X=None, Y=None, kernel="triweight", bandwidth=None, backend=None):
    """
    Calculates the SCE of the model based on data (X,Y).
    Args:
        model: the model whose ECE we want.
        X: numpy.ndarray, variables of the calibration set.
        Y: numpy.ndarray, labels of the calibration set.
        n_bins: int, number of bins used to discretize score space [0,1].
        backend: string (default "google"), name of the backend used.

    Returns:
    The static calibration error (SCE) of the model.
    """

    if backend is None:
        backend = "accuracies_confidences"

    if backend == "accuracies_confidences":
        # Implementation in terms of accuracies and confidences

        if scores is None:
            scores = model.predict_proba(X)

        result = 0
        for i in range(scores.shape[1]):

            if bandwidth == "silverman":
                bandwidth = get_silvermans_bandwidth(X=scores[:, i], kernel=kernel, bandwidth=bandwidth)

            result += direct_specific_ece(model=model, specific_scores=scores[:, i], Y=Y, bandwidth=bandwidth,
                                          class_index=i, backend="accuracies_confidences")

        return result/scores.shape[1]

    else:
        raise NotImplementedError
