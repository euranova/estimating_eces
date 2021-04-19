# -*- coding: utf-8 -*-
"""
@author: nicolas.posocco
"""

from ...prototype.bin_boundaries import EqualBinsBinBoundariesPolicy
from ...prototype.binning import ConfidenceBinningPolicy
from ..discrete_metrics import confidence_binned_metric
from ...utils import google_metric_formulation
import netcal.metrics as netcal_metrics
import uncertainty_metrics.numpy as um
import numpy as np
from .....utils import confidences_from_scores


def confidence_ece(model=None, X=None, Y=None, confidence_scores=None, predictions=None, n_bins=10, backend=None):
    """
    Calculates the ECE of the model based on data (X,Y).
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

    if backend is "contributions":

        if confidence_scores is None:
            # model
            assert hasattr(model, "predict_proba")

            scores = model.predict_proba(X)
            predictions = np.argmax(scores, axis=1)
            confidence_scores = confidences_from_scores(model=model, predictions=predictions,
                                                        scores_matrix=scores)

        # Calculating bins allocation
        bin_boundaries_policy = EqualBinsBinBoundariesPolicy()
        binning_policy = ConfidenceBinningPolicy(bin_boundaries_policy=bin_boundaries_policy,
                                                 n_bins=n_bins)
        bins_weights = binning_policy(model=model, confidence_scores=confidence_scores)

        return confidence_binned_metric(model, X, Y, bins_weights)

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
        binning_policy = ConfidenceBinningPolicy(bin_boundaries_policy=bin_boundaries_policy,
                                                 n_bins=n_bins)

        bins_weights = binning_policy(model=model, confidence_scores=confidence_scores)

        card_dataset = confidence_scores.shape[0]

        result = 0
        for bin_weights in bins_weights:
            card_bin = len(bin_weights)

            if card_bin > 0:
                bin_acc_unorm = 0
                bin_conf_unorm = 0
                for i, _ in bin_weights:
                    bin_acc_unorm += int(Y[i] == predictions[i])
                    bin_conf_unorm += confidence_scores[i]

                bin_contribution = np.abs(bin_acc_unorm - bin_conf_unorm) / card_dataset
                result += bin_contribution

        return result

    elif backend == "prototype":
        metric_value = prototype_metrics.confidence_ece(model, X, Y, n_bins=n_bins)
        return metric_value

    elif backend == "google":
        probabilities, labels = google_metric_formulation(model=model, X=X, Y=Y)
        metric_value = um.ece(labels, probabilities, num_bins=n_bins)
        return metric_value

    elif backend == "netcal":
        scores = model.predict_proba(X)
        ece = netcal_metrics.ECE(n_bins)
        metric_value = ece.measure(X=scores, y=Y)
        return metric_value

    else:
        raise NotImplementedError
