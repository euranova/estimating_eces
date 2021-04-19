# -*- coding: utf-8 -*-
"""
@author: nicolas.posocco
"""

import numpy as np
from ....utils import confidences_from_scores


def specific_class_binned_metric(model=None, X=None, specific_scores=None, Y=None, bins_weights=None, class_index=None):
    """
    Calculates a binary metric, given a binning allocation.
    Args:
        model: an ML model.
        X: np.ndarray (shape (n_samples, dimensionality)), data matrix.
        Y: np.ndarray (shape (n_samples, )), labels vector.
        bins_weights: list of floats, result of a BinningPolicy __call__.
        class_index: int, index of the class of interest.

    Returns:
    The value of the metric.
    """

    def sample_contribution_(bin_weights, sample_i, sample_weight, model, specific_scores, Y):
        """
        Defines the contribution of a sample.
        Args:
            bin_weights: element of a result of a BinningPolicy __call__.
            sample_i: int, index of the sample of interest in the data matrix.
            sample_weight: float, weight of the sample of interest for current bin.
            model: an ML model.
            scores: np.ndarray (shape (n_samples, n_classes)), scores matrix.
            Y: np.ndarray (shape (n_samples, )), labels vector.

        Returns:
        The contribution of this sample to current bin.
        """

        bin_weight = np.sum([w for _, w in bin_weights])
        return sample_weight / bin_weight * (
                (Y[sample_i] == model.classes_[class_index]) - specific_scores[sample_i])

    def bin_contribution_(bin_weights, card_dataset, sample_contributions):
        """
        Defines the contribution of a bin.
        Args:
            bin_weights: element of a result of a BinningPolicy __call__ for the bin of interest.
            card_dataset: amount of samples used to calculate the metric.
            sample_contributions: list of the contributions of samples in this bin.

        Returns:
        The contribution of this bin to the metric.
        """
        bin_weight = np.sum([w for _, w in bin_weights])
        return bin_weight / card_dataset * np.abs(np.sum(sample_contributions))

    card_dataset = specific_scores.shape[0]

    # For each bin
    bin_contributions = []
    for bin_weights in bins_weights:

        # For each sample contributing to this bin
        sample_contributions = []
        for i, w in bin_weights:
            # Calculating sample contribution to this bin.
            sample_contribution = sample_contribution_(bin_weights, i, w, model, specific_scores, Y)
            sample_contributions.append(sample_contribution)

        # The bin contribution is calculated from sample contributions to this bin
        bin_contribution = bin_contribution_(bin_weights, card_dataset, sample_contributions)
        bin_contributions.append(bin_contribution)

    # Returning the metric as the sum of its bin contributions.
    return np.sum(bin_contributions)


def binary_binned_metric(model=None, X=None, positive_scores=None, Y=None, bins_weights=None):
    """
    Calculates a binary metric, given a binning allocation.
    Args:
        model: an ML model.
        X: np.ndarray (shape (n_samples, dimensionality)), data matrix.
        Y: np.ndarray (shape (n_samples, )), labels vector.
        bins_weights: result of a BinningPolicy __call__.

    Returns:
    The value of the metric.
    """

    return specific_class_binned_metric(model=model, X=X, specific_scores=positive_scores,
                                        Y=Y, bins_weights=bins_weights, class_index=1)


def confidence_binned_metric(model, X, Y, bins_weights):
    def sample_contribution_(bin_weights, sample_i, sample_weight, confidences, predictions, Y):
        bin_weight = np.sum([w for _, w in bin_weights])
        return sample_weight / bin_weight * ((predictions[sample_i] == Y[sample_i]) - confidences[sample_i])

    def bin_contribution_(bin_weights, card_dataset, sample_contributions):
        bin_weight = np.sum([w for _, w in bin_weights])
        return bin_weight / card_dataset * np.abs(np.sum(sample_contributions))

    scores = model.predict_proba(X)
    predictions = model.predict(X)
    confidences = confidences_from_scores(scores_matrix=scores,
                                          predictions=predictions,
                                          model=model)

    bin_contributions = []
    for bin_weights in bins_weights:
        card_dataset = X.shape[0]

        sample_contributions = []
        for i, w in bin_weights:
            sample_contribution = sample_contribution_(bin_weights, i, w, confidences, predictions, Y)
            sample_contributions.append(sample_contribution)
        bin_contribution = bin_contribution_(bin_weights, card_dataset, sample_contributions)
        bin_contributions.append(bin_contribution)

    return np.sum(bin_contributions)


def classwise_binned_metric(model, X, Y, classwise_bins_weights):
    class_contributions = []
    for class_index in range(len(list(model.classes_))):
        bins_weights = classwise_bins_weights[class_index]

        class_contribution = specific_class_binned_metric(model, X, Y, bins_weights, class_index=class_index)
        class_contributions.append(class_contribution)

    return np.mean(class_contributions)
