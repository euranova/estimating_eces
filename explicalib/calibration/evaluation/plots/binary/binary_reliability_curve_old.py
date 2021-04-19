# -*- coding: utf-8 -*-
"""
@author: nicolas.posocco
"""

from ...diagrams.binary.binary_reliability_diagram import binary_reliability_diagram
from ...diagrams.binary.binary_reliability_curve_old import binary_reliability_curve_old
import numpy as np
from ..general_plots import plot_reliability


def plot_binary_reliability_curve_old(model=None, X=None, scores=None, Y=None, n_bins=None, bandwidth=None,
                                      kernel="cosine", font=None, method=None):
    # model
    assert model is not None

    # Calculating scores for the positive class if not provided
    if scores is None:
        assert X is not None
        scores = model.predict_proba(X)[:, 1]

    else:

        # X is ignored

        assert type(scores) is np.ndarray
        assert scores.ndim == 1

    if n_bins is not None:
        discrete_reliability_content = binary_reliability_diagram(scores=scores, Y=Y, n_bins=n_bins)
    else:
        discrete_reliability_content = None

    reliability_curve = binary_reliability_curve_old(model=model,
                                                     scores=scores, Y=Y,
                                                     method=method,
                                                     bandwidth=bandwidth, kernel=kernel)
    print(reliability_curve)

    # Calculating plot components
    reliability_content = {"discrete": discrete_reliability_content,
                           "continuous": reliability_curve,
                           "bootstrapped_continuous": None, "scores": scores, "is_confidence": False}

    # Plotting them
    plot_reliability(reliability_content=reliability_content, font=font)
