# -*- coding: utf-8 -*-
"""
@author: nicolas.posocco
"""

import numpy as np
from ...diagrams.binary.binary_reliability_diagram import binary_reliability_diagram
from ..general_plots import plot_reliability


def plot_binary_reliability_diagram(model=None, X=None, scores=None, Y=None, n_bins=None, font=None,
                                    show_score_distribution=True, title="Reliability diagram",
                                    legend_loc="best", legend_framealpha=0.8):
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

    # Calculating plot components
    reliability_content = {"discrete": binary_reliability_diagram(scores=scores, Y=Y, n_bins=n_bins),
                           "continuous": None, "bootstrapped_continuous": None, "scores": scores,
                           "is_confidence": False}

    # Plotting them
    plot_reliability(reliability_content=reliability_content, font=font,
                     show_score_distribution=show_score_distribution, title=title, legend_loc=legend_loc,
                     legend_framealpha=legend_framealpha)
