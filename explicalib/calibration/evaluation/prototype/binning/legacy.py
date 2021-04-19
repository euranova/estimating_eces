# -*- coding: utf-8 -*-
"""
@author: nicolas.posocco
"""

from .primitives import FixedBinAmountBinningPolicy
from ..bin_boundaries import BinBoundariesPolicy
from .....utils import confidences_from_scores
import numpy as np


class SpecificClassBinningPolicy(FixedBinAmountBinningPolicy):

    def __init__(self, n_bins, bin_boundaries_policy):
        """
        Initializes a specific class binning policy,
        which sends samples into the n_bins bins created by the bin_boundaries_policy,
        based on the score the model gives for the class provided at runtime.
        Args:
            n_bins: int, number of bins used to divide the interval.
            bin_boundaries_policy: BinBoundariesPolicy, defining how to split the interval into bins.
        """

        # n_bins
        assert type(n_bins) is int
        assert n_bins > 0

        # bin_boundaries_policy
        assert isinstance(bin_boundaries_policy, BinBoundariesPolicy)

        super().__init__(n_bins=n_bins,
                         bin_boundaries_policy=bin_boundaries_policy)

    def __call__(self, model=None, X=None, specific_scores=None, class_index=None):
        """
        Affects every sample id to a bin based on the score given by the model for specified label.
        Bins placement is uniform, their amount is decided in the __init__ method.
        Args:
            model: ML model.
            X: np.ndarray (n_samples, dimensionality), data matrix.
            class_index: int, the index of the class on which we want the binning.

        Returns:
        A list of list of 2-tuples (int, float) defining the binning affectation of samples, each tuple being a
        (sample_index, weight).
        """

        if specific_scores is None:
            # model
            assert hasattr(model, "predict_proba"), "Model must implement the predict_proba method."

            # X
            assert type(X) is np.ndarray, "X must be a numpy array."
            assert X.ndim == 2, "Number of dimensions of array X must be 2."

            # class_index
            assert class_index is not None, "When specific_scores is not provided, class_index has to be."
            assert type(class_index) is int
            assert 0 <= class_index < len(model.classes_)

            # Calculating model scores on X for targeted class
            specific_scores = model.predict_proba(X)[:, class_index]

        # Grouping scores into n_bins interval bins, each of size 1/M
        bins = [[] for _ in range(self.n_bins)]
        bin_boundaries = self.bin_boundaries_policy(n_bins=self.n_bins, segment=[0, 1], elements=specific_scores)
        for i in range(specific_scores.shape[0]):
            current_score = specific_scores[i]

            # Getting corresponding bin
            m = self.bin_boundaries_policy.affect(bin_boundaries, current_score)

            # Adding sample id to corresponding bin (no specific weight -> 1)
            bins[m].append((i, 1))

        return bins


class BinaryBinningPolicy(SpecificClassBinningPolicy):

    def __init__(self, n_bins, bin_boundaries_policy):
        """
        Initializes a specific class binning policy,
        which sends samples into the n_bins bins created by the bin_boundaries_policy,
        based on the score the model gives for the positive class.
        Args:
            n_bins: int, number of bins used to divide the interval.
            bin_boundaries_policy: BinBoundariesPolicy, defining how to split the interval into bins.
        """

        # n_bins
        assert type(n_bins) is int
        assert n_bins > 0

        # bin_boundaries_policy
        assert isinstance(bin_boundaries_policy, BinBoundariesPolicy)

        super().__init__(n_bins=n_bins,
                         bin_boundaries_policy=bin_boundaries_policy)

    def __call__(self, model=None, X=None, positive_scores=None):
        """
        Affects every sample id to a bin based on the score given by the model for the positive class.
        Bins placement is uniform, their amount is decided in the __init__ method.
        Args:
            model: ML model.
            X: np.ndarray (n_samples, dimensionality), data matrix.

        Returns:
        A list of list of 2-tuples (int, float) defining the binning affectation of samples, each tuple being a
        (sample_index, weight).
        """

        if positive_scores is None:
            # model
            assert model is not None, "When positive_scores are not provided, model has to be, here is None."
            assert hasattr(model, "predict_proba"), "Model must implement the predict_proba method."

            # X
            assert type(X) is np.ndarray, "X must be a numpy array."
            assert X.ndim == 2, "Number of dimensions of array X must be 2."

            positive_scores = model.predict_proba(X)[:, 1]

        return SpecificClassBinningPolicy.__call__(self, model=model, specific_scores=positive_scores)


class ConfidenceBinningPolicy(FixedBinAmountBinningPolicy):

    def __init__(self, n_bins, bin_boundaries_policy):
        """
        Initializes a predicted class binning policy,
        which sends samples into the n_bins bins created by the bin_boundaries_policy,
        based on the score the model gives for the class provided at runtime.
        Args:
            n_bins: int, number of bins used to divide the interval.
            bin_boundaries_policy: BinBoundariesPolicy, defining how to split the interval into bins.
        """

        super().__init__(n_bins=n_bins, bin_boundaries_policy=bin_boundaries_policy)

    def __call__(self, model=None, X=None, confidence_scores=None):
        """

        Args:
            model:
            X:

        Returns:

        """

        if confidence_scores is None:
            # model
            assert hasattr(model, "predict_proba"), "Model <model> must implement the predict_proba method."

            # X
            assert type(X) is np.ndarray, "<X> must be a numpy array."
            assert X.ndim == 2, "Number of dimensions of array <X> must be 2."

            # Calculating model predictions and scores on X
            scores = model.predict_proba(X)
            predictions = model.predict(X)
            confidence_scores = confidences_from_scores(scores, predictions, model)

        # Grouping predictions into <bins> interval bins, each of size 1/M
        bins = [[] for _ in range(self.n_bins)]
        low = 1 / len(model.classes_)
        bin_boundaries = self.bin_boundaries_policy(n_bins=self.n_bins, segment=[low, 1], elements=confidence_scores)
        for i in range(confidence_scores.shape[0]):
            # Getting corresponding bin
            m = self.bin_boundaries_policy.affect(bin_boundaries, confidence_scores[i])

            # Adding sample id to corresponding bin (no specific weight -> 1)
            bins[m].append((i, 1))

        return bins
