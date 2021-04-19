# -*- coding: utf-8 -*-
"""
@author: nicolas.posocco
"""

from ...prototype.bin_boundaries import EqualBinsBinBoundariesPolicy
from ...prototype.binning import ConfidenceConvexAllocationBinningPolicy
from ..discrete_metrics import confidence_binned_metric
import numpy as np
from .....utils import confidences_from_scores


def confidence_ece_c(model=None, X=None, confidence_scores=None, predictions=None, Y=None, n_bins=10, backend=None):
    """

    Args:
        model: the model whose ECE we want.
        X: numpy.ndarray, variables of the calibration set.
        Y: numpy.ndarray, labels of the calibration set.
        n_bins: int, number of bins used to discretise score space [0,1].
        backend: TODO.

    Returns:
    The convex expected calibration error (CECE) of the model.
    """

    if backend is None:
        backend = "accuracies_confidences"

    if backend == "accuracies_confidences":
        # Implementation in terms of accuracies and confidences

        if confidence_scores is None:
            # model
            assert hasattr(model, "predict_proba")

            scores = model.predict_proba(X)
            predictions = np.argmax(scores, axis=1)
            confidence_scores = confidences_from_scores(model=model, predictions=predictions,
                                                        scores_matrix=scores)

        if n_bins == "sqrt":
            n_bins = int(np.sqrt(len(confidence_scores)))

        bin_boundaries_policy = EqualBinsBinBoundariesPolicy()
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

    elif backend == "contributions":
        # Calculating bins allocation
        bin_boundaries_policy = EqualBinsBinBoundariesPolicy()
        binning_policy = ConfidenceConvexAllocationBinningPolicy(bin_boundaries_policy=bin_boundaries_policy,
                                                                 n_bins=n_bins)
        bins_weights = binning_policy(model=model,  X=X)

        return confidence_binned_metric(model, X, Y, bins_weights)

    else:
        raise NotImplementedError
