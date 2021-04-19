# -*- coding: utf-8 -*-
"""
@author: nicolas.posocco
"""

import numpy as np
from ..bin_boundaries import BinBoundariesPolicy
from abc import ABC


class BinningPolicy(ABC):

    def __init__(self):
        """
        Initializes the binning policy.
        """

        pass

    def __call__(self, model=None, X=None, scores=None):
        """
        Sends samples into bins depending on model scores.
        Args:
            model: ML model.
            X: np.ndarray (n_samples, dimensionality), data matrix.

        Returns:
        A list of list of 2-tuples (int, float) defining the binning affectation of samples, each tuple being a
        (sample_index, weight).
        """

        raise NotImplementedError


class FixedBinAmountBinningPolicy(BinningPolicy):

    def __init__(self, n_bins, bin_boundaries_policy):
        """
        Initializes a fixed bin amount binning policy,
        which sends samples into the n_bins bins created by the bin_boundaries_policy.
        Args:
            n_bins: int, number of bins used to divide the interval.
            bin_boundaries_policy: BinBoundariesPolicy, defining how to split the interval into bins.
        """

        # n_bins
        assert type(n_bins) is int or n_bins in ("sqrt",)
        assert n_bins > 0

        # bin_boundaries_policy
        assert isinstance(bin_boundaries_policy, BinBoundariesPolicy)

        self.n_bins = n_bins
        self.bin_boundaries_policy = bin_boundaries_policy

        super().__init__()

    def __call__(self, model=None, X=None, scores=None):
        """
        Sends samples into bins depending on model scores.
        Args:
            model: ML model.
            X: np.ndarray (n_samples, dimensionality), data matrix.

        Returns:
        A list of list of 2-tuples (int, float) defining the binning affectation of samples, each tuple being a
        (sample_index, weight).
        """

        # model
        assert hasattr(model, "predict_proba")

        # X
        assert type(X) is np.ndarray
        assert X.ndim == 2

        raise NotImplemented
