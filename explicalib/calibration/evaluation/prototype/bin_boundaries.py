# -*- coding: utf-8 -*-
"""
@author: nicolas.posocco
"""

from math import floor, ceil
import numpy as np
from abc import ABC


class BinBoundariesPolicy(ABC):

    def __init__(self):
        """
        Initializes a bin boundaries policy.
        """

        pass

    def __call__(self, n_bins, segment, elements):
        """
        Returns the bin boundaries given the policy and context (an n_bins+1 array).
        Args:
            n_bins: int, number of bins to create.
            segment: list or tuple (length 2) containing the limits of the segment to subdivide, in increasing order.
            elements: np.ndarray (shape (n_elements, )) containing the elements to send to the bins.

        Returns:
        An np.ndarray (shape (n_bins+1, ) containing the boundaries defining the n_bins bins.
        """

        # n_bins
        assert type(n_bins) is int
        assert n_bins > 0

        # segment
        assert type(segment) is list or type(segment) is tuple
        assert len(segment) == 2
        assert np.isscalar(segment[0]) and np.isscalar(segment[1])
        assert segment[0] < segment[1]

        # elements
        assert type(elements) is np.ndarray, f"elements should be an np.ndarray, instead of {type(elements)}"
        assert elements.dtype == np.number

        raise NotImplemented

    def affect(self, bin_boundaries, element):
        """
        Describes how element is sent to its bin.
        Args:
            bin_boundaries: output of the __call__ method above.
            element: numeric object in the range of the segment

        Returns:
        The index of the bin in which the element is being sent.
        """

        # bin_boundaries
        assert type(bin_boundaries) is np.ndarray

        # element
        assert isinstance(element, (int, float, np.number)), \
            "element = {} should be of a numeric type, not {}.".format(element, type(element))
        assert bin_boundaries[0] <= element <= bin_boundaries[-1]

        # For all bins, in increasing order
        for m in range(1, len(bin_boundaries)):

            # If the element is too small to get into the mth bin
            if element < bin_boundaries[m]:
                # Returning the index of the previous one
                return m - 1

        # Boundary case : element belongs to the last bin.
        return len(bin_boundaries) - 2


class EqualBinsBinBoundariesPolicy(BinBoundariesPolicy):

    def __init__(self):
        """
        Initializes an equal bins bin boundaries policy,
        which splits given segment in n_bins equal bins
        """

        super().__init__()

    def __call__(self, n_bins, segment, elements):
        """
        Returns the bin boundaries corresponding to equal bins division.
        Args:
            n_bins: int, number of bins to create.
            segment: list or tuple (length 2) containing the limits of the segment to subdivide, in increasing order.
            elements: np.ndarray (shape (n_elements, )) containing the elements to send to the bins.

        Returns:
        An np.ndarray (shape (n_bins+1, ) containing the boundaries defining the n_bins bins.
        """

        # n_bins
        assert type(n_bins) is int
        assert n_bins > 0

        # segment
        assert type(segment) is list or type(segment) is tuple
        assert len(segment) == 2
        assert np.isscalar(segment[0]) and np.isscalar(segment[1])
        assert segment[0] < segment[1]

        # elements
        assert type(elements) is np.ndarray, f"elements should be an np.ndarray, instead of {type(elements)}"
        assert elements.dtype == np.number

        return np.array([segment[0] + i / n_bins * (segment[1] - segment[0])
                         for i in range(n_bins)]
                        + [float(segment[1])])

    def affect(self, bin_boundaries, element):
        """
        Describes how element is sent to its bin.
        Args:
            bin_boundaries: output of the __call__ method above.
            element: numeric object in the range of the segment

        Returns:
        The index of the bin in which the element is being sent.
        """

        # bin_boundaries
        assert type(bin_boundaries) is np.ndarray

        # element
        assert isinstance(element, (int, float, np.number)), \
            "element = {} should be of a numeric type, not {}.".format(element, type(element))
        assert bin_boundaries[0] <= element <= bin_boundaries[-1]

        n_bins = len(bin_boundaries) - 1
        m = floor(element * n_bins) if floor(element * n_bins) < n_bins else n_bins - 1

        return m


class EqualAmountBinBoundariesPolicy(BinBoundariesPolicy):

    def __init__(self):
        """
        Initializes a bin boundaries policy.
        """

        super().__init__()

    def __call__(self, n_bins, segment, elements):
        """
        Returns the bin boundaries corresponding to an adaptative binning policy
        (same amount of samples in each bin).
        Args:
            n_bins: int, number of bins to create.
            segment: list or tuple (length 2) containing the limits of the segment to subdivide, in increasing order.
            elements: np.ndarray (shape (n_elements, )) containing the elements to send to the bins.

        Returns:
        An np.ndarray (shape (n_bins+1, ) containing the boundaries defining the n_bins bins.
        """

        # n_bins
        assert type(n_bins) is int
        assert n_bins > 0

        # segment
        assert type(segment) is list or type(segment) is tuple
        assert len(segment) == 2
        assert np.isscalar(segment[0]) and np.isscalar(segment[1])
        assert segment[0] < segment[1]

        # elements
        assert type(elements) is np.ndarray, f"elements should be an np.ndarray, instead of {type(elements)}"
        assert elements.dtype == np.number

        sorted_elements = np.sort(elements)

        bin_card = int(floor(elements.shape[0]/n_bins))

        bin_boundaries = [segment[0]]

        for i in range(1, n_bins):
            boundary_l = sorted_elements[i*bin_card - 1]
            boundary_r = sorted_elements[i * bin_card]
            boundary = (boundary_l+boundary_r)/2

            bin_boundaries.append(boundary)

        bin_boundaries.append(segment[1])

        return np.array(bin_boundaries)
