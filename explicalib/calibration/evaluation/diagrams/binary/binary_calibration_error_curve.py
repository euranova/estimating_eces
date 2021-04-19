# -*- coding: utf-8 -*-
"""
@author: nicolas.posocco
"""


from .binary_reliability_curve import binary_reliability_curve


def binary_calibration_error_curve(model=None, X=None, Y=None,
                                   kernel=None, bandwidth=None,
                                   positive_scores=None,
                                   positive_scores_for_positive_gt=None,
                                   positive_class_probability=None):
    reliability_curve = binary_reliability_curve(model=model, X=X, Y=Y,
                                                 kernel=kernel, bandwidth=bandwidth,
                                                 positive_scores=positive_scores,
                                                 positive_scores_for_positive_gt=positive_scores_for_positive_gt,
                                                 positive_class_probability=positive_class_probability)

    result = {"scores": reliability_curve["scores"],
              }

    return result
