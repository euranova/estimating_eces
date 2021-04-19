# -*- coding: utf-8 -*-
"""
@author: nicolas.posocco
"""

from .primitives import FixedBinAmountBinningPolicy
from .legacy import SpecificClassBinningPolicy, BinaryBinningPolicy
from .....utils import confidences_from_scores
import numpy as np


class SpecificClassConvexAllocationBinningPolicy(SpecificClassBinningPolicy):

    def __init__(self, n_bins, bin_boundaries_policy):
        """
        TODO
        Args:
            n_bins: int, number of bins used to subdivide the segment [0,1].
            bin_boundaries_policy: TODO
        """

        super().__init__(n_bins=n_bins,
                         bin_boundaries_policy=bin_boundaries_policy)

    def __call__(self, model=None, X=None, specific_scores=None, class_index=None):
        """
        TODO
        Args:
            model: ML model.
            X: np.ndarray (shape (n_samples, dimensionality)), data matrix.
            class_index: int, index of the class of interest.

        Returns:
        TODO
        """

        assert class_index is not None

        if specific_scores is None:
            # model
            assert hasattr(model, "predict_proba"), "Model must implement the predict_proba method."

            # X
            assert type(X) is np.ndarray, "X must be a numpy array."
            assert X.ndim == 2, "Number of dimensions of array X must be 2."

            # class_index
            assert type(class_index) is int
            assert 0 <= class_index < X.shape[1]

            # Calculating model predictions and scores on X
            specific_scores = model.predict_proba(X)[:, class_index]

        # Grouping predictions into <bins> interval bins, each of size 1/M
        bins = [[] for _ in range(self.n_bins)]
        bin_boundaries = self.bin_boundaries_policy(n_bins=self.n_bins, segment=[0, 1], elements=specific_scores)
        centroids = [(bin_boundaries[i] + bin_boundaries[i - 1]) / 2 for i in range(1, self.n_bins + 1)]
        for i in range(specific_scores.shape[0]):

            # Getting corresponding predicted score
            target_score = specific_scores[i]

            # Getting corresponding bin
            if target_score < centroids[0]:
                bins[0].append((i, 1))
                continue
            elif target_score > centroids[-1]:
                bins[-1].append((i, 1))
                continue

            # Usual case : item falls between two centroïds
            for b in range(self.n_bins):
                if centroids[b] < target_score:
                    continue
                else:
                    di = centroids[b] - target_score
                    dim1 = target_score - centroids[b - 1]
                    bins[b].append((i, di / (di + dim1)))
                    bins[b - 1].append((i, dim1 / (di + dim1)))
                    break

        return bins


class BinaryConvexAllocationBinningPolicy(SpecificClassConvexAllocationBinningPolicy, BinaryBinningPolicy):

    def __init__(self, n_bins, bin_boundaries_policy):
        """
        TODO
        Args:
            n_bins: int, number of bins used to subdivide the segment [0,1].
            bin_boundaries_policy: TODO
        """

        super().__init__(n_bins=n_bins,
                         bin_boundaries_policy=bin_boundaries_policy)

    def __call__(self, model=None, X=None, positive_scores=None):  # PEP is not happy, and it's normal.
        """
        TODO
        Args:
            model: ML model.
            X: np.ndarray (shape (n_samples, dimensionality)), data matrix.

        Returns:
        TODO
        """

        return SpecificClassConvexAllocationBinningPolicy.__call__(self, model=model, specific_scores=positive_scores,
                                                                   class_index=1)


class ConfidenceConvexAllocationBinningPolicy(FixedBinAmountBinningPolicy):

    def __init__(self, n_bins, bin_boundaries_policy):
        """
        TODO
        Args:
            n_bins: int, number of bins used to subdivide the segment [0,1].
            bin_boundaries_policy: BinBoundariesPolicy object, defining how the segment [0,1] is divided.
        """

        super().__init__(n_bins=n_bins,
                         bin_boundaries_policy=bin_boundaries_policy)

    def __call__(self, model=None, X=None, confidence_scores=None):
        """
        TODO
        Args:
            model: ML model.
            X: np.ndarray (shape (n_samples, dimensionality)), data matrix.

        Returns:
        TODO
        """

        if confidence_scores is None:

            # model
            assert hasattr(model, "predict_proba"), "Model must implement the predict_proba method."

            # X
            assert type(X) is np.ndarray, "X must be a numpy array."
            assert X.ndim == 2, "Number of dimensions of array X must be 2."

            # Calculating model predictions and scores on X
            scores = model.predict_proba(X)
            predictions = model.predict(X)
            confidences = confidences_from_scores(scores, predictions, model)

        # Grouping predictions into <bins> interval bins, each of size 1/M
        bins = [[] for _ in range(self.n_bins)]
        low = 1 / len(model.classes_)
        bin_boundaries = self.bin_boundaries_policy(n_bins=self.n_bins, segment=[low, 1], elements=confidence_scores)
        centroids = [(bin_boundaries[i] + bin_boundaries[i - 1]) / 2 for i in range(1, self.n_bins + 1)]
        for i in range(confidence_scores.shape[0]):

            # Getting corresponding predicted score
            target_score = confidence_scores[i]

            # Getting corresponding bin
            if target_score < centroids[0]:
                bins[0].append((i, 1))
                continue
            elif target_score > centroids[-1]:
                bins[-1].append((i, 1))
                continue

            # Usual case : item falls between two centroïds
            for b in range(self.n_bins):
                if centroids[b] < target_score:
                    continue
                else:
                    di = centroids[b] - target_score
                    dim1 = target_score - centroids[b - 1]
                    bins[b].append((i, di / (di + dim1)))
                    bins[b - 1].append((i, dim1 / (di + dim1)))
                    break

        return bins