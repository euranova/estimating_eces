# -*- coding: utf-8 -*-
"""
@author: nicolas.posocco
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.patches as patches


def plot_reliability(reliability_content, font=None, show_score_distribution=True, title=None, curve_color="#1f77b4",
                     legend_loc="best", legend_framealpha=0.8):
    # Setting font if provided
    if font is not None:
        matplotlib.rc('font', **font)

    # Initializing figure
    if reliability_content["scores"] is not None and show_score_distribution:
        fig = plt.figure(figsize=(10, 14))
        gs = fig.add_gridspec(2, 1, height_ratios=[11, 3])
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[1, 0])
    else:
        fig = plt.figure(figsize=(10, 10))
        gs = fig.add_gridspec(1, 1)
        ax1 = fig.add_subplot(gs[0, 0])

    # Plotting discrete reliability diagram
    if reliability_content["discrete"] is not None:
        ax1.scatter(reliability_content["discrete"]["mean_predicted_value"],
                    reliability_content["discrete"]["fraction_of_positives"],
                    color="black", label="Discrete reliability plot", s=10)
        for i in range(len(reliability_content["discrete"]["mean_predicted_value"])):
            x_p = reliability_content["discrete"]["mean_predicted_value"][i]
            y_p_bissec = x_p
            y_p_high = reliability_content["discrete"]["fraction_of_positives"][i]
            ax1.plot([x_p, y_p_bissec], [x_p, y_p_high], c="grey", alpha=0.6)

    # for diagram in trajectories:
    #    plot_binary_reliability_k2(diagram, axis=ax1)

    if reliability_content["bootstrapped_continuous"] is not None:

        if reliability_content["is_confidence"]:
            x = np.linspace(1/reliability_content["n_classes"], 1,
                            reliability_content["bootstrapped_continuous"]["median"].shape[0])
        else:
            x = np.linspace(0, 1, reliability_content["bootstrapped_continuous"]["median"].shape[0])

        # Plotting median reliability curve

        ax1.plot(x, reliability_content["bootstrapped_continuous"]["median"], label="Median reliability curve",
                 color=curve_color)

        # Adding confidence intervals
        ax1.fill_between(x,
                         reliability_content["bootstrapped_continuous"]["quantile_low"],
                         reliability_content["bootstrapped_continuous"]["quantile_high"],
                         alpha=0.2,
                         label="Confidence interval ({}%)".format(
                             int(reliability_content["bootstrapped_continuous"]["confidence_interval"] * 100)),
                         color=curve_color)

    if reliability_content["continuous"] is not None:
        # Plotting reliability curve
        ax1.plot(reliability_content["continuous"]["scores"],
                 reliability_content["continuous"]["errors"],
                 label="Reliability curve", color=curve_color)

    # Naming axes
    ax1.set_ylabel("Fraction of positives")
    ax1.set_xlabel("Score")

    # Setting axes span
    ax1.set_ylim([0, 1])
    ax1.set_xlim([0, 1])

    # Setting plot title
    if title is not None:
        ax1.set_title('Reliability diagram - Reliability curve')

    # Defining legend
    ax1.legend(loc=legend_loc, framealpha=legend_framealpha)

    if reliability_content["is_confidence"]:
        shadow_until_x = 1 / reliability_content["n_classes"]
        rect = patches.Rectangle((0, 0), shadow_until_x, 1, linewidth=0, facecolor='grey', alpha=0.5)
        ax1.add_patch(rect)

        # Plotting y=x (perfect calibration)
        ax1.plot([shadow_until_x, 1], [shadow_until_x, 1], "k:", label="Perfect calibration")

    else:
        # Plotting y=x (perfect calibration)
        ax1.plot([0, 1], [0, 1], "k:", label="Perfect calibration")

    if reliability_content["scores"] is not None and show_score_distribution:
        plot_scores_distribution(scores=reliability_content["scores"], target_axis=ax2)

    plt.show()


def plot_scores_distribution(scores, target_axis):
    target_axis.hist(scores, bins=100)
    target_axis.set_xlim([0, 1])
