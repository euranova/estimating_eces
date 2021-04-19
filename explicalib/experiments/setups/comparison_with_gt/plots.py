# -*- coding: utf-8 -*-
"""
@author: nicolas.posocco
"""

import copy
import numpy as np
from .utils import group_by, load_all
import matplotlib
import matplotlib.pyplot as plt


def add_normalized_trajectories(reports):
    res = []
    for report in reports:
        report2 = copy.deepcopy(report)

        gt = report["gt"]

        report2["medians_norm"] = report["medians"] / gt
        report2["highs_norm"] = report["highs"] / gt

        res.append(report2)

    return res


def aggregate_by_metric_complete(reports):
    def aggregate(group):
        result = {}

        d = np.stack([report["medians"] for report in group])
        distance_medians_gt_norm = np.median(d, axis=0)
        result["medians"] = distance_medians_gt_norm

        d = np.stack([report["highs"] for report in group])
        distance_highs_gt_norm = np.median(d, axis=0)
        result["highs"] = distance_highs_gt_norm

        result["n_samples_calibration_evaluations"] = group[0]["n_samples_calibration_evaluations"]

        # All groupped reports share the same metric settings
        result["metric.__name__"] = group[0]["metric.__name__"]
        result["metric_kwargs"] = group[0]["metric_kwargs"]
        result["metric_complete"] = group[0]["metric_complete"]

        return result

    reports_groupped = group_by(reports, key="metric_complete")
    reports_aggregated = [aggregate(group) for group in reports_groupped]

    return reports_aggregated


def plot_all_upper(seed_train_test_split=None, seed_distribution=None, setup_name=None, n_features=None, ylim=None,
                   filter_in=None, title=None, cut_at=5000, font=None, reports=None):
    metric_name_to_colors = {
        "classwise_ece": "k",
        "direct_classwise_ece": "g",
        "classwise_ece_a": "r",
        "classwise_ece_ma": "orange",
        "classwise_ece_c": "b",
        "classwise_ece_ac": "m",

        "confidence_ece": "k",
        "direct_confidence_ece": "g",
        "confidence_ece_a": "r",
        "confidence_ece_ma": "orange",
        "confidence_ece_c": "b",
        "confidence_ece_ac": "m",

        "binary_ece": "k",
        "direct_binary_ece": "g",
        "binary_ece_a": "r",
        "binary_ece_ma": "orange",
        "binary_ece_c": "b",
        "binary_ece_ac": "m",
    }

    metric_complete_to_linestyle = {
        "sqrt": "--",

        # "direct": ":", # if silverman, gets replaced by
        "silverman": "-.",
    }

    metric_complete_to_alpha = {
        "{'n_bins': 10}": 1.0,
        "{'n_bins': 20}": 0.7,
        "{'n_bins': 30}": 0.4,
        "{'bandwidth': 0.01}": 0.4,
        "{'bandwidth': 0.033}": 0.7,
        "{'bandwidth': 0.05}": 0.8,
        "{'bandwidth': 0.1}": 1.0,
    }

    metric_name_to_clean_metric_name = {
        "binary_ece": "$ECE^1$",
        "binary_ece_c": "$ECE^1_c$",
        "binary_ece_a": "$ECE^1_a$",
        "binary_ece_ma": "$ECE^1_{ma}$",
        "binary_ece_ac": "$ECE^1_{ac}$",
        "direct_binary_ece": "$ECE^1_{kde}$",

        "confidence_ece": "$ECE^{conf}$",
        "confidence_ece_c": "$ECE^{conf}_c$",
        "confidence_ece_a": "$ECE^{conf}_a$",
        "confidence_ece_ma": "$ECE^{conf}_{ma}$",
        "confidence_ece_ac": "$ECE^{conf}_{ac}$",
        "direct_confidence_ece": "$ECE^{conf}_{kde}$",

        "classwise_ece": "$ECE^{cw}$",
        "classwise_ece_c": "$ECE^{cw}_c$",
        "classwise_ece_a": "$ECE^{cw}_a$",
        "classwise_ece_ma": "$ECE^{cw}_{ma}$",
        "classwise_ece_ac": "$ECE^{cw}_{ac}$",
        "direct_classwise_ece": "$ECE^{cw}_{kde}$",
    }

    assert reports is not None

    # Setting font
    if font is None:
        font = {'family': 'DejaVu Sans', 'size': 15}
    matplotlib.rc('font', **font)

    #if reports is None:
    #    reports_df = load_all(setup_name=setup_name, protocol=4)
    #    reports = reports_df.to_dict(orient='records')

    #if seed_train_test_split is not None:
    #    reports = filter(lambda report: report["seed_train_test_split"] == seed_train_test_split, reports)
    #if seed_distribution is not None:
    #    reports = filter(lambda report: report["seed_distribution"] == seed_distribution, reports)
    #if n_features is not None:
    #    reports = filter(lambda report: report["n_features"] == n_features, reports)

    #if filter_in is not None:
    #    reports = filter(lambda report: filter_in(report), reports)

    print("Aggregating {} trajectories.".format(len(reports)))

    # Adding normalized trajectories
    print(type(reports))
    reports = add_normalized_trajectories(reports)

    # Adding the distance between the median estimation and the ground truth (with and without normalization)
    # reports = add_distances_to_gt(reports)

    # Adding the width of the 95% confidence interval (with and without normalization)
    # reports = add_confidence95_width(reports)

    # Aggregating reports with the same metric_complete
    reports = aggregate_by_metric_complete(reports)

    # Sorting reports for consistent colors
    reports = sorted(reports, key=lambda report: report["metric_complete"])

    # Creating figure
    fig = plt.figure(figsize=(14, 18))
    gs = fig.add_gridspec(2, 1)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0])

    legends = {}
    for report in reports:

        metric_complete = report["metric_complete"]
        metric_complete = metric_complete.replace("classwise_", "").replace("binary_", "").replace("confidence_", "")

        metric_name = report["metric.__name__"]

        if metric_name not in legends:
            legends[metric_name] = []
        legends[metric_name].append((metric_complete, report))

    max_entries = 0
    for metric_name in legends:
        max_entries = max(max_entries, len(legends[metric_name]))

    # Populating figure
    for metric_name in legends:

        clean_metric_name = metric_name_to_clean_metric_name[metric_name]
        ax2.plot([0, 1], [-1, -1], marker='o', linestyle="None", label=clean_metric_name,
                 color=metric_name_to_colors[metric_name])

        for metric_complete, report in legends[metric_name]:

            # Setting linestyle
            linestyle = "solid"
            for part in metric_complete_to_linestyle:
                if part in report["metric_complete"]:
                    linestyle = metric_complete_to_linestyle[part]

            # Setting color
            color = "fuchsia"
            for metric_name in metric_name_to_colors:
                if metric_name == report["metric.__name__"]:
                    color = metric_name_to_colors[metric_name]

            # Setting alpha
            alpha = 1
            for part in metric_complete_to_alpha:
                if part in report["metric_complete"]:
                    alpha = metric_complete_to_alpha[part]

            metric_complete = report["metric_complete"]
            metric_complete = metric_complete.replace("classwise_", "").replace("binary_", "").replace("confidence_",
                                                                                                       "")
            metric_name = report["metric.__name__"]

            metric_kwargs = report["metric_kwargs"]

            # if "direct" in report["metric_complete"]:
            #    linestyle = ":"
            # elif "sqrt" in report["metric_complete"]:
            #    linestyle = "--"
            # elif "silverman" in report["metric_complete"]:
            #    linestyle = "-."
            # else:
            #    linestyle="solid"

            if metric_kwargs == "{'n_bins': 'sqrt'}":
                metric_kwargs = "sqrt"
            elif metric_kwargs == "{'bandwidth': 'silverman'}":
                metric_kwargs = "Silverman"
            elif "n_bins" in metric_kwargs:
                n_bins = metric_kwargs.split(" ")[1][:-1]
                metric_kwargs = str(n_bins) + " bins"
            elif "bandwidth" in metric_kwargs:
                bandwidth = metric_kwargs.split(" ")[1][:-1]
                metric_kwargs = "bandwidth : " + str(bandwidth)

            p = ax1.loglog(report["n_samples_calibration_evaluations"], report["medians"],
                             label=metric_kwargs, linestyle=linestyle, color=color, alpha=alpha)

            ax2.loglog(report["n_samples_calibration_evaluations"], report["highs"],
                         label=metric_kwargs, linestyle=linestyle, color=color, alpha=alpha)

            # If a blank space needs to be created
            if metric_complete == legends[metric_name][-1][0] and len(legends[metric_name]) < max_entries:
                n_empties = max_entries - len(legends[metric_name])
                for i in range(n_empties):
                    ax2.plot([0, 1], [-1, -1], marker='o', label=".", color="white")

    ax1.set_xlim(30, cut_at)
    ax2.set_xlim(30, cut_at)
    #ax1.set_ylim(0, ylim)
    #ax2.set_ylim(0, ylim)
    ax1.set_xticks([])
    ax2.set_xticks([e for e in [30, 100, 200, 300, 400, 500, 5000] if e <= cut_at])

    # ax1.set_ylabel("Median deviation to ground truth trajectories")
    # ax2.set_ylabel("97.5th percentile deviation to ground truth trajectories")
    # ax2.set_xlabel("Number of samples used for the computation")

    # plt.legend([p], loc="lower center", bbox_to_anchor=(0.5, -0.3), ncol=3) #ax1.legend(loc=(1.025,0))
    plt.legend(loc="lower center", bbox_to_anchor=(0.5, -0.325),
               ncol=len(list(legends.keys())))  # ax1.legend(loc=(1.025,0))

    # if title is not None:
    #    fig.title(title)

    plt.tight_layout()

    ax1.set_ylabel("Median deviation to ground truth trajectories")
    ax2.set_ylabel("97.5th percentile deviation to ground truth trajectories")
    ax2.set_xlabel("Number of samples used for the computation")

    plt.show()


def plot_all(reports_classwise, reports_confidence, font=None):
    metric_name_to_colors = {
        "classwise_ece": "k",
        "direct_classwise_ece": "g",
        "classwise_ece_a": "r",
        "classwise_ece_ma": "orange",
        "classwise_ece_c": "b",
        "classwise_ece_ac": "m",

        "confidence_ece": "k",
        "direct_confidence_ece": "g",
        "confidence_ece_a": "r",
        "confidence_ece_ma": "orange",
        "confidence_ece_c": "b",
        "confidence_ece_ac": "m",

        "binary_ece": "k",
        "direct_binary_ece": "g",
        "binary_ece_a": "r",
        "binary_ece_ma": "orange",
        "binary_ece_c": "b",
        "binary_ece_ac": "m",
    }

    metric_complete_to_linestyle = {
        "sqrt": "--",

        # "direct": ":", # if silverman, gets replaced by
        "silverman": "-.",
    }

    metric_complete_to_alpha = {
        "{'n_bins': 10}": 1.0,
        "{'n_bins': 20}": 0.7,
        "{'n_bins': 30}": 0.4,
        "{'bandwidth': 0.01}": 0.4,
        "{'bandwidth': 0.033}": 0.7,
        "{'bandwidth': 0.05}": 0.8,
        "{'bandwidth': 0.1}": 1.0,
    }

    metric_name_to_clean_metric_name = {
        "binary_ece": "$ECE^1$",
        "binary_ece_c": "$ECE^1_c$",
        "binary_ece_a": "$ECE^1_a$",
        "binary_ece_ma": "$ECE^1_{ma}$",
        "binary_ece_ac": "$ECE^1_{ac}$",
        "direct_binary_ece": "$ECE^1_{kde}$",

        "confidence_ece": "$ECE^{conf}$",
        "confidence_ece_c": "$ECE^{conf}_c$",
        "confidence_ece_a": "$ECE^{conf}_a$",
        "confidence_ece_ma": "$ECE^{conf}_{ma}$",
        "confidence_ece_ac": "$ECE^{conf}_{ac}$",
        "direct_confidence_ece": "$ECE^{conf}_{kde}$",

        "classwise_ece": "$ECE^{cw}$",
        "classwise_ece_c": "$ECE^{cw}_c$",
        "classwise_ece_a": "$ECE^{cw}_a$",
        "classwise_ece_ma": "$ECE^{cw}_{ma}$",
        "classwise_ece_ac": "$ECE^{cw}_{ac}$",
        "direct_classwise_ece": "$ECE^{cw}_{kde}$",
    }

    assert reports_classwise is not None
    assert reports_confidence is not None

    # Setting font
    if font is None:
        font = {'family': 'DejaVu Sans', 'size': 15}
    matplotlib.rc('font', **font)

    print("CW   : Aggregating {} trajectories.".format(len(reports_classwise)))
    print("CONF : Aggregating {} trajectories.".format(len(reports_confidence)))

    # Adding normalized trajectories
    reports_classwise = add_normalized_trajectories(reports_classwise)
    reports_confidence = add_normalized_trajectories(reports_confidence)

    # Aggregating reports with the same metric_complete
    reports_classwise = aggregate_by_metric_complete(reports_classwise)
    reports_confidence = aggregate_by_metric_complete(reports_confidence)

    # Sorting reports for consistent colors
    reports_classwise = sorted(reports_classwise, key=lambda report: report["metric_complete"])

    # Creating figure
    fig = plt.figure(figsize=(14, 18))
    gs = fig.add_gridspec(2, 1)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0])

    # CLasswise
    legends = {}
    for report in reports_classwise:

        metric_complete = report["metric_complete"]
        metric_complete = metric_complete.replace("classwise_", "").replace("binary_", "").replace("confidence_", "")

        metric_name = report["metric.__name__"]

        if metric_name not in legends:
            legends[metric_name] = []
        legends[metric_name].append((metric_complete, report))

    max_entries = 0
    for metric_name in legends:
        max_entries = max(max_entries, len(legends[metric_name]))

    # Populating figure
    for metric_name in legends:

        clean_metric_name = metric_name_to_clean_metric_name[metric_name]
        ax2.plot([0, 1], [-1, -1], marker='o', linestyle="None", label=clean_metric_name,
                 color=metric_name_to_colors[metric_name])

        for metric_complete, report in legends[metric_name]:

            # Setting linestyle
            linestyle = "solid"
            for part in metric_complete_to_linestyle:
                if part in report["metric_complete"]:
                    linestyle = metric_complete_to_linestyle[part]

            # Setting color
            color = "fuchsia"
            for metric_name in metric_name_to_colors:
                if metric_name == report["metric.__name__"]:
                    color = metric_name_to_colors[metric_name]

            # Setting alpha
            alpha = 1
            for part in metric_complete_to_alpha:
                if part in report["metric_complete"]:
                    alpha = metric_complete_to_alpha[part]

            metric_complete = report["metric_complete"]
            metric_complete = metric_complete.replace("classwise_", "").replace("binary_", "").replace("confidence_",
                                                                                                       "")
            metric_name = report["metric.__name__"]

            metric_kwargs = report["metric_kwargs"]

            # if "direct" in report["metric_complete"]:
            #    linestyle = ":"
            # elif "sqrt" in report["metric_complete"]:
            #    linestyle = "--"
            # elif "silverman" in report["metric_complete"]:
            #    linestyle = "-."
            # else:
            #    linestyle="solid"

            if metric_kwargs == "{'n_bins': 'sqrt'}":
                metric_kwargs = "sqrt"
            elif metric_kwargs == "{'bandwidth': 'silverman'}":
                metric_kwargs = "Silverman"
            elif "n_bins" in metric_kwargs:
                n_bins = metric_kwargs.split(" ")[1][:-1]
                metric_kwargs = str(n_bins) + " bins"
            elif "bandwidth" in metric_kwargs:
                bandwidth = metric_kwargs.split(" ")[1][:-1]
                metric_kwargs = "bandwidth : " + str(bandwidth)

            p = ax1.loglog(report["n_samples_calibration_evaluations"], report["highs"],
                           label=metric_kwargs, linestyle=linestyle, color=color, alpha=alpha)

            #ax2.loglog(report["n_samples_calibration_evaluations"], report["highs"],
            #           label=metric_kwargs, linestyle=linestyle, color=color, alpha=alpha)

            # If a blank space needs to be created
            if metric_complete == legends[metric_name][-1][0] and len(legends[metric_name]) < max_entries:
                n_empties = max_entries - len(legends[metric_name])
                for i in range(n_empties):
                    ax2.plot([0, 1], [-1, -1], marker='o', label=".", color="white")

        # Confidence
        legends = {}
        for report in reports_confidence:

            metric_complete = report["metric_complete"]
            metric_complete = metric_complete.replace("classwise_", "").replace("binary_", "").replace("confidence_",
                                                                                                       "")

            metric_name = report["metric.__name__"]

            if metric_name not in legends:
                legends[metric_name] = []
            legends[metric_name].append((metric_complete, report))

        max_entries = 0
        for metric_name in legends:
            max_entries = max(max_entries, len(legends[metric_name]))

        # Populating figure
        for metric_name in legends:

            clean_metric_name = metric_name_to_clean_metric_name[metric_name]
            ax2.plot([0, 1], [-1, -1], marker='o', linestyle="None", label=clean_metric_name,
                     color=metric_name_to_colors[metric_name])

            for metric_complete, report in legends[metric_name]:

                # Setting linestyle
                linestyle = "solid"
                for part in metric_complete_to_linestyle:
                    if part in report["metric_complete"]:
                        linestyle = metric_complete_to_linestyle[part]

                # Setting color
                color = "fuchsia"
                for metric_name in metric_name_to_colors:
                    if metric_name == report["metric.__name__"]:
                        color = metric_name_to_colors[metric_name]

                # Setting alpha
                alpha = 1
                for part in metric_complete_to_alpha:
                    if part in report["metric_complete"]:
                        alpha = metric_complete_to_alpha[part]

                metric_complete = report["metric_complete"]
                metric_complete = metric_complete.replace("classwise_", "").replace("binary_", "").replace(
                    "confidence_",
                    "")
                metric_name = report["metric.__name__"]

                metric_kwargs = report["metric_kwargs"]

                # if "direct" in report["metric_complete"]:
                #    linestyle = ":"
                # elif "sqrt" in report["metric_complete"]:
                #    linestyle = "--"
                # elif "silverman" in report["metric_complete"]:
                #    linestyle = "-."
                # else:
                #    linestyle="solid"

                if metric_kwargs == "{'n_bins': 'sqrt'}":
                    metric_kwargs = "sqrt"
                elif metric_kwargs == "{'bandwidth': 'silverman'}":
                    metric_kwargs = "Silverman"
                elif "n_bins" in metric_kwargs:
                    n_bins = metric_kwargs.split(" ")[1][:-1]
                    metric_kwargs = str(n_bins) + " bins"
                elif "bandwidth" in metric_kwargs:
                    bandwidth = metric_kwargs.split(" ")[1][:-1]
                    metric_kwargs = "bandwidth : " + str(bandwidth)

                p = ax2.loglog(report["n_samples_calibration_evaluations"], report["highs"],
                               label=metric_kwargs, linestyle=linestyle, color=color, alpha=alpha)

                # ax2.loglog(report["n_samples_calibration_evaluations"], report["highs"],
                #           label=metric_kwargs, linestyle=linestyle, color=color, alpha=alpha)

                # If a blank space needs to be created
                if metric_complete == legends[metric_name][-1][0] and len(legends[metric_name]) < max_entries:
                    n_empties = max_entries - len(legends[metric_name])
                    for i in range(n_empties):
                        ax2.plot([0, 1], [-1, -1], marker='o', label=".", color="white")

    ax1.set_xlim(30, 500)
    #ax2.set_xlim(30, 500)
    # ax1.set_ylim(0, ylim)
    # ax2.set_ylim(0, ylim)
    ax1.set_xticks([])
    #ax2.set_xticks([e for e in [30, 100, 200, 300, 400, 500, 5000] if e <= cut_at])

    # ax1.set_ylabel("Median deviation to ground truth trajectories")
    # ax2.set_ylabel("97.5th percentile deviation to ground truth trajectories")
    # ax2.set_xlabel("Number of samples used for the computation")

    # plt.legend([p], loc="lower center", bbox_to_anchor=(0.5, -0.3), ncol=3) #ax1.legend(loc=(1.025,0))
    plt.legend(loc="lower center", bbox_to_anchor=(0.5, -0.325),
               ncol=len(list(legends.keys())))  # ax1.legend(loc=(1.025,0))

    # if title is not None:
    #    fig.title(title)

    plt.tight_layout()

    ax1.set_ylabel("Median deviation to ground truth trajectories")
    #ax2.set_ylabel("97.5th percentile deviation to ground truth trajectories")
    #ax2.set_xlabel("Number of samples used for the computation")

    plt.show()
