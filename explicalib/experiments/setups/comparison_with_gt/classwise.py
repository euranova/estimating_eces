# -*- coding: utf-8 -*-
"""
@author: nicolas.posocco
"""

from .utils import save, is_already_calculated, savefig, get_calculated
from ....distribution.multiclass_multimodal_multivariate_gaussian_distribution import \
    MulticlassMultimodalMultivariateGaussianDistribution
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from sklearn.utils import resample
from ....probabilistic_classifiers import SVCc, GaussianNB, LogisticRegression, RandomForestClassifier
import matplotlib.pyplot as plt
import os
from ....calibration.evaluation.metrics.classwise.classwise_ece import classwise_ece
from ....calibration.evaluation.metrics.classwise.classwise_ece_a import classwise_ece_a
from ....calibration.evaluation.metrics.classwise.classwise_ece_c import classwise_ece_c
from ....calibration.evaluation.metrics.classwise.classwise_ece_ac import classwise_ece_ac
from ....calibration.evaluation.metrics.classwise.direct_classwise_ece import direct_classwise_ece
from ....utils import logspace

# SETUP CONSTANTS

# MODEL
MODEL_INITIALIZATION_SEED = 1
TRAIN_SIZE = 300

# DISTRIBUTION / SAMPLING
N_DISTRIBUTIONS = 15
N_CLASSES = [3, 5, 10]
N_MODES_PER_CLASS = 4
N_FEATURES = [2, 5, 10]
MEAN_RADIUS = 1
COVARIANCE_RADIUS = 0.3
N_TOTAL_SAMPLES = 2000000 + TRAIN_SIZE
N_TRAIN_TEST_SPLITS = 3

# GROUND TRUTH ESTIMATION
N_BINS_GT = 2000

# EXPERIMENTS
N_SAMPLINGS = 200
n_samples_calibration_evaluations = logspace(30, 5000, 20)
evaluated_metric_payloads = [

    (direct_classwise_ece, {"bandwidth": "silverman"}),
    (direct_classwise_ece, {"bandwidth": 0.1}),
    (direct_classwise_ece, {"bandwidth": 0.033}),

    (classwise_ece, {"n_bins": 10}),
    (classwise_ece, {"n_bins": 30}),
    (classwise_ece, {"n_bins": "sqrt"}),

    (classwise_ece_a, {"n_bins": 10}),
    (classwise_ece_a, {"n_bins": 30}),
    (classwise_ece_a, {"n_bins": "sqrt"}),

    (classwise_ece_c, {"n_bins": 10}),
    (classwise_ece_c, {"n_bins": 30}),
    (classwise_ece_c, {"n_bins": "sqrt"}),

    (classwise_ece_ac, {"n_bins": 10}),
    (classwise_ece_ac, {"n_bins": 30}),
    (classwise_ece_ac, {"n_bins": "sqrt"}),

]


def plot_scores_distribution(scores, target_axis):
    target_axis.hist(scores, bins=100, density=True)
    target_axis.set_xlim([0, 1])


def figure_scores_distributions(scores_matrix):
    n_dims = scores_matrix.shape[1]

    # Creating figure
    fig = plt.figure(figsize=(11, 5 * n_dims))
    gs = fig.add_gridspec(n_dims, 1)

    for i in range(n_dims):
        ax = fig.add_subplot(gs[i, 0])
        plot_scores_distribution(scores=scores_matrix[:, i], target_axis=ax)
        ax.set_title("Scores for class {}".format(i))

    return fig


def compute(setup_name, n_features=None, seeds_distribution=None, n_classes=None, parallelism=True):
    if type(n_classes) is not list:
        n_classes = [n_classes]

    if seeds_distribution is None:
        seeds_distribution = [i for i in range(N_DISTRIBUTIONS)]
    elif type(seeds_distribution) is not list:
        seeds_distribution = [seeds_distribution]

    if n_features is None:
        n_features = N_FEATURES
    elif type(n_features) is not list:
        n_features = [n_features]

    confidence_interval = 0.95
    semi_uncertainty_mass = 0.5 * (1 - confidence_interval)

    already_calculated = get_calculated(setup_name=setup_name)

    for n_classes_i in n_classes:

        print("n_classes :", n_classes_i)

        for n_features_i in n_features:

            print("  n_features :", n_features_i)

            for seed_distribution in seeds_distribution:
                reproducible_distribution = np.random.RandomState(seed_distribution)

                print("    seed_distribution :", seed_distribution)

                # Sampling distribution
                distribution = MulticlassMultimodalMultivariateGaussianDistribution. \
                    randomly_set(n_classes=n_classes_i,
                                 n_features=n_features_i,
                                 means_radius=MEAN_RADIUS,
                                 n_modes_per_class=N_MODES_PER_CLASS,
                                 covariances_radius=COVARIANCE_RADIUS,
                                 mode_weights="uniform",
                                 reproducible=reproducible_distribution)

                # Sampling from distribution
                X, Y, P = distribution.sample(n_samples=N_TOTAL_SAMPLES, reproducible=reproducible_distribution)

                for seed_train_test_split in range(N_TRAIN_TEST_SPLITS):
                    reproducible_train_test_split = np.random.RandomState(seed_train_test_split)

                    print("      seed_train_test_split :", seed_train_test_split)

                    X_train, X_holdout, Y_train, Y_holdout = train_test_split(
                        X, Y, train_size=TRAIN_SIZE, random_state=reproducible_train_test_split)

                    # For each model family considered in the setup
                    for model_family in [SVCc, GaussianNB, LogisticRegression, RandomForestClassifier]:

                        # Instantiating a model
                        if model_family != GaussianNB:
                            model = model_family(random_state=MODEL_INITIALIZATION_SEED)
                        else:
                            model = model_family()

                        # Fitting such model
                        model.fit(X_train, Y_train)

                        # Calculating output scores on the holdout data
                        scores_holdout = model.predict_proba(X_holdout)

                        # Saving scores distribution
                        scores_dist = figure_scores_distributions(scores_holdout)
                        fig_name = \
                            model_family.__name__ + \
                            "_" + str(seed_train_test_split) + \
                            "_" + str(seed_distribution) + \
                            "_" + str(n_features_i) + \
                            "_" + str(n_classes_i)
                        savefig(setup_name=setup_name,
                                fig_dir=os.path.join("shared", "scores_distributions", str(n_features_i)),
                                fig_name=fig_name, fig=scores_dist)

                        # Calculating a very precise estimate of the ECE
                        gt = classwise_ece(model=model,
                                           scores=scores_holdout,
                                           Y=Y_holdout,
                                           n_bins=N_BINS_GT)

                        # For each evaluated estimator and set of associated hyperparameters
                        for metric, kwargs in evaluated_metric_payloads:

                            # Initializing the current report with its job identifier
                            identifier = \
                                metric.__name__ + \
                                "_" + str(kwargs) + \
                                "_" + model_family.__name__ + \
                                "_" + str(seed_train_test_split) + \
                                "_" + str(seed_distribution) + \
                                "_" + str(n_features_i) + \
                                "_" + str(n_classes_i)

                            # Skip if already calculated
                            if is_already_calculated(setup_name=setup_name, identifier=identifier,
                                                     already_calculated=already_calculated,
                                                     parallelism=parallelism):
                                print("    Already computed :", identifier)
                                continue

                            # Initializing report
                            report = {"identifier": identifier}

                            # For every evaluation set size
                            metric_values = []
                            for n_samples_eval in tqdm(n_samples_calibration_evaluations, postfix=identifier):

                                metric_values_n_samples = []

                                for _ in range(N_SAMPLINGS):
                                    # Getting evaluation set
                                    scores_eval, Y_eval = resample(scores_holdout, Y_holdout,
                                                                   n_samples=n_samples_eval,
                                                                   random_state=reproducible_train_test_split)

                                    # Calculating evaluated metric value
                                    metric_value = metric(scores=scores_eval, Y=Y_eval, model=model, **kwargs)
                                    metric_values_n_samples.append(metric_value)

                                metric_values.append(metric_values_n_samples)

                            # Calculating 50% and 95% quantile trajectories
                            metric_values = np.array(metric_values)
                            minus = gt * np.ones(metric_values.shape)
                            distances_to_gt = np.abs(metric_values - minus)

                            # Calculating 50% and 95% quantile trajectories
                            medians, highs = np.quantile(distances_to_gt, axis=1,
                                                         q=[0.5, 0.95])

                            # Saving job parameters
                            report["gt"] = gt
                            report["n_features"] = n_features_i
                            report["n_classes"] = n_classes_i
                            report["n_samples_calibration_evaluations"] = n_samples_calibration_evaluations
                            report["seed_train_test_split"] = seed_train_test_split
                            report["seed_distribution"] = seed_distribution
                            report["medians"] = medians
                            report["highs"] = highs
                            # report["lows"] = lows
                            report["metric.__name__"] = metric.__name__
                            report["model_family.__name__"] = model_family.__name__
                            report["metric_kwargs"] = str(kwargs)
                            report["metric_complete"] = report["metric.__name__"] + report["metric_kwargs"]

                            # Saving job report
                            save(setup_name=setup_name,
                                 filename=identifier,
                                 payload=report,
                                 protocol=4)

                            if parallelism is False:
                                already_calculated.append(identifier + ".pickle")
                            else:
                                already_calculated = get_calculated(setup_name=setup_name)
