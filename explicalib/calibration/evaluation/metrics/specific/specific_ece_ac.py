# -*- coding: utf-8 -*-
"""
@author: nicolas.posocco
"""

from ...prototype.bin_boundaries import EqualAmountBinBoundariesPolicy
from ...prototype.binning import SpecificClassConvexAllocationBinningPolicy
import numpy as np


def specific_ece_ac(model=None, X=None, specific_scores=None, Y=None, class_index=None, n_bins=None, backend=None):
    """
    Calculates the binary ECE beta of the model based on data (X,Y).
    Args:
        model: the model whose ECE we want.
        X: numpy.ndarray, variables of the calibration set.
        Y: numpy.ndarray, labels of the calibration set.
        bandwidth: float, TODO.
        backend: string (default "prototype"), name of the backend used.

    Returns:
    The binary ece beta of the model.
    """

    if backend is None:
        backend = "accuracies_confidences"

    if backend == "accuracies_confidences":
        # Implementation in terms of accuracies and confidences

        if specific_scores is None:
            specific_scores = model.predict_proba(X)[:, class_index]

        if n_bins == "sqrt":
            n_bins = int(np.sqrt(len(specific_scores.shape[0])))

        bin_boundaries_policy = EqualAmountBinBoundariesPolicy()
        binning_policy = SpecificClassConvexAllocationBinningPolicy(
            bin_boundaries_policy=bin_boundaries_policy,
            n_bins=n_bins)

        bins_weights = binning_policy(specific_scores=specific_scores, class_index=class_index)

        card_dataset = specific_scores.shape[0]

        result = 0
        for bin_weights in bins_weights:
            card_bin = np.sum([sample_weight[1] for sample_weight in bin_weights])

            if card_bin > 0:
                bin_acc_unorm = 0
                bin_conf_unorm = 0
                for i, w in bin_weights:
                    bin_acc_unorm += w*int(Y[i] == model.classes_[class_index])
                    bin_conf_unorm += w*specific_scores[i]
                bin_contribution = np.abs(bin_acc_unorm - bin_conf_unorm) / card_dataset
                result += bin_contribution

        return result

    else:
        raise NotImplementedError
