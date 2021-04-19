# -*- coding: utf-8 -*-
"""
@author: nicolas.posocco
"""

from ..discrete_metrics import classwise_binned_metric
import numpy as np
from ..specific.specific_ece_ac import specific_ece_ac


def classwise_ece_ac(model=None, X=None, scores=None, Y=None, n_bins=10, backend=None):
    """
    Calculates the SCE (adaptative binning, convex allocation) of the model based on data (X,Y).
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

        if n_bins == "sqrt":
            n_bins = int(np.sqrt(scores.shape[0]))

        result = 0
        for i in range(scores.shape[1]):
            result += specific_ece_ac(model=model, specific_scores=scores[:, i], Y=Y, n_bins=n_bins,
                                      class_index=i, backend="accuracies_confidences")

        return result/scores.shape[1]

    elif backend is "contributions":

        classwise_bins_weights = []
        for class_index in range(len(list(model.classes_))):
            # Calculating bins allocation
            bin_boundaries_policy = EqualAmountBinBoundariesPolicy()
            binning_policy = SpecificClassConvexAllocationBinningPolicy(bin_boundaries_policy=bin_boundaries_policy,
                                                                        n_bins=n_bins)
            bins_weights = binning_policy(model, X, class_index=class_index)

            classwise_bins_weights.append(bins_weights)

        return classwise_binned_metric(model, X, Y, classwise_bins_weights)

    elif backend == "prototype":
        return prototype_metrics.sce_c(model, X, Y, n_bins)

    elif backend == "google":
        raise NotImplementedError

    else:
        raise NotImplementedError
