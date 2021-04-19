# -*- coding: utf-8 -*-
"""
@author: nicolas.posocco
"""

from ...prototype.binning import ConfidenceConvexAllocationBinningPolicy
from ...prototype.bin_boundaries import EqualAmountBinBoundariesPolicy
from ..discrete_metrics import confidence_binned_metric
import numpy as np
from .....utils import confidences_from_scores


def confidence_ece_ac(model=None, X=None, confidence_scores=None, predictions=None, Y=None, n_bins=10, backend=None):
    """
    Calculates the ECE (adaptative binning, convex allocation) of the model based on data (X,Y).
    Args:
        model: the model whose ECE we want.
        X: numpy.ndarray, variables of the calibration set.
        Y: numpy.ndarray, labels of the calibration set.
        n_bins: int, number of bins used to discretize score space [0,1].
        backend: string (default "prototype"), name of the backend used.

    Returns:
    The expected calibration error (ECE) of the model.
    """

    if backend is None:
        backend = "accuracies_confidences"

    if backend == "accuracies_confidences":
        # Implementation in terms of accuracies and confidences

        if confidence_scores is None:
            # model
            assert hasattr(model, "predict_proba")

            scores = model.predict_proba(X)
            predictions = model.predict(X)
            confidence_scores = confidences_from_scores(model=model, predictions=predictions,
                                                        scores_matrix=scores)

        if n_bins == "sqrt":
            n_bins = int(np.sqrt(len(confidence_scores)))

        bin_boundaries_policy = EqualAmountBinBoundariesPolicy()
        binning_policy = ConfidenceConvexAllocationBinningPolicy(bin_boundaries_policy=bin_boundaries_policy,
                                                                 n_bins=n_bins)

        bins_weights = binning_policy(model=model, confidence_scores=confidence_scores)

        card_dataset = confidence_scores.shape[0]

        result = 0
        for bin_weights in bins_weights:
            card_bin = np.sum([sample_weight[1] for sample_weight in bin_weights])

            if card_bin > 0:
                bin_acc_unorm = 0
                bin_conf_unorm = 0
                for i, w in bin_weights:
                    bin_acc_unorm += w * int(Y[i] == predictions[i])
                    bin_conf_unorm += w * confidence_scores[i]
                bin_contribution = np.abs(bin_acc_unorm - bin_conf_unorm) / card_bin
                result += bin_contribution * card_bin / card_dataset

        return result

    elif backend is "contributions":

        # Calculating bins allocation
        bin_boundaries_policy = EqualAmountBinBoundariesPolicy()
        binning_policy = ConfidenceConvexAllocationBinningPolicy(bin_boundaries_policy=bin_boundaries_policy,
                                                                 n_bins=n_bins)
        bins_weights = binning_policy(model=model,  X=X)

        return confidence_binned_metric(model, X, Y, bins_weights)

    elif backend == "prototype":
        raise NotImplementedError

    elif backend == "google":
        raise NotImplementedError

    elif backend == "netcal":
        raise NotImplementedError

    else:
        raise NotImplementedError
