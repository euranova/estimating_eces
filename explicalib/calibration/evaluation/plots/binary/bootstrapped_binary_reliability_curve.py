# -*- coding: utf-8 -*-
"""
@author: nicolas.posocco
"""

from ...diagrams.binary.binary_reliability_diagram import binary_reliability_diagram
from ...diagrams.binary.bootstrapped_binary_reliability_curve import bootstrapped_binary_reliability_curve
import numpy as np
from ..general_plots import plot_reliability


def plot_bootstrapped_binary_reliability_curve(model=None, X=None, scores=None, Y=None, n_bins=None,
                                               bandwidth=None, n_samplings=None,
                                               confidence_interval=0.99, kernel="gaussian", font=None,
                                               show_score_distribution=True, title="Bootstrapped reliability curve",
                                               curve_color="#1f77b4",
                                               legend_loc="best", legend_framealpha=0.8):
    # model
    assert hasattr(model, "classes_"), "model must have a classes_ attribute."

    if scores is None:

        # model
        assert hasattr(model, "predict_proba"), "model must implement the predict_proba method."

        # X
        assert type(X) is np.ndarray, "X should be an np.ndarray, not a {}.".format(type(X))
        assert X.ndim == 2, "X should be a 1D array, not a {}D one.".format(X.ndim)

        scores = model.predict_proba(X)[:, 1]

    else:
        # model and X are ignored

        # scores
        assert type(scores) is np.ndarray, "scores should be an np.ndarray, not a {}.".format(type(scores))
        assert scores.ndim == 1, "scores should be a 2D array, not a {}D one.".format(scores.ndim)

    # Y
    # TODO

    # bandwidth
    # TODO

    # n_samplings
    # TODO

    # method
    # if method == "convolution":
    #    possibilities = ("gaussian", "cosine")
    #    assert kernel in possibilities, \
    #        "With convolution as method, kernel should be in {}, yet is equal to {}.".format(possibilities, kernel)

    # elif method == "moving_average":
    #    possibilities = ("gaussian", None)
    #    assert kernel in possibilities, \
    #        "With moving_average as method, kernel should be in {}, yet is equal to {}.".format(possibilities, kernel)

    # else:
    #    raise NotImplementedError(
    #        "method should be in {}, here is {}.".format(("convolution", "moving_average"), method))

    # font
    # TODO

    if n_bins is not None:
        discrete_reliability_content = binary_reliability_diagram(scores=scores, Y=Y, n_bins=n_bins)
    else:
        discrete_reliability_content = None

    bootstrapped_continuous_content = bootstrapped_binary_reliability_curve(model=model,
                                                                            scores=scores, Y=Y,
                                                                            confidence_interval=confidence_interval,
                                                                            n_samplings=n_samplings,
                                                                            bandwidth=bandwidth,
                                                                            kernel=kernel)

    # Calculating plot components
    reliability_content = {"discrete": discrete_reliability_content,
                           "continuous": None,
                           "bootstrapped_continuous": bootstrapped_continuous_content,
                           "scores": scores, "is_confidence": False}

    # Plotting them
    plot_reliability(reliability_content=reliability_content, font=font,
                     show_score_distribution=show_score_distribution, title=title,
                     curve_color=curve_color,
                     legend_loc=legend_loc, legend_framealpha=legend_framealpha)
