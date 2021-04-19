# -*- coding: utf-8 -*-
"""
@author: nicolas.posocco
"""

import numpy as np
from sklearn.mixture import GaussianMixture
from scipy.stats import norm
import matplotlib.cm
import matplotlib.pyplot as plt
from .multiclass_distribution import MulticlassDistribution
from sklearn.utils import shuffle


class MulticlassMultimodalMultivariateGaussianDistribution(MulticlassDistribution):

    def __init__(self, means, covariances, mode_weights, mode_classes):
        """
        Initializes the distribution, allowing later sampling and posterior probabilities calculations.
        """

        # mode_classes
        assert type(mode_classes) is list
        self.n_modes = len(mode_classes)
        self.mode_classes = mode_classes

        # means
        assert type(means) is np.ndarray
        assert means.ndim == 2
        assert means.shape[0] == self.n_modes
        self.means = means
        self.n_features = self.means.shape[1]

        # covariances
        assert type(covariances) is np.ndarray
        assert covariances.ndim == 3
        assert covariances.shape[0] == self.n_modes
        assert covariances.shape[1] == self.n_features
        assert covariances.shape[2] == self.n_features
        self.covariances = covariances

        # mode_weights
        assert type(mode_weights) is np.ndarray
        assert mode_weights.ndim == 1
        assert mode_weights.shape[0] == self.n_modes
        self.mode_probabilities = mode_weights

        self.n_classes = len(np.unique(self.mode_classes))

        # Must be set in order to sample from the sklearn implementation
        self.distribution = GaussianMixture(n_components=self.n_modes,
                                            covariance_type="full",
                                            means_init=self.means,
                                            weights_init=self.mode_probabilities)
        self.distribution.means_ = self.means
        self.distribution.weights_ = self.mode_probabilities
        self.distribution.covariances_ = self.covariances
        self.distribution.precisions_cholesky_ = np.array(
            [np.linalg.inv(self.covariances[i]) for i in range(self.n_modes)])

        super().__init__()

    @classmethod
    def randomly_set(cls, n_classes, n_features, n_modes_per_class,
                     means_radius, covariances_radius,
                     mode_weights, reproducible):

        # The class each mode belongs to
        # Here we consider the same number of modes per class,
        # and thus bellow each class appears exactly n_modes_per_class times.
        mode_classes = [class_ for class_ in range(n_classes)] * n_modes_per_class

        # Sampling means in a unit ball (L1 norm) scaled by means_radius
        n_modes = n_classes * n_modes_per_class
        means = reproducible.uniform(low=-means_radius, high=means_radius,
                                     size=(n_modes, n_features))

        # Sampling covariances
        # We use there that
        # - transp(A)*A is positive definite for all invertible matrices A
        # - A random matrix has a probability of ~1 of being a full rank one, and thus of being invertible.
        As = reproducible.uniform(low=-covariances_radius,
                                  high=covariances_radius,
                                  size=(n_modes, n_features, n_features))
        covariances = np.array(
            [np.matmul(np.transpose(As[i, :, :]), As[i, :, :]) \
             for i in range(As.shape[0])]
        )

        # Sampling class weights
        if mode_weights == "uniform":
            mode_weights = 1 / n_modes * np.ones(n_modes)
        else:
            raise NotImplementedError

        return cls(means=means, covariances=covariances,
                   mode_weights=mode_weights, mode_classes=mode_classes)

    def sample(self, n_samples, reproducible=None):
        """
        Samples n_samples times from the distribution, and their label.
        Returns also the array of posterior probabilities if return_posterior=True.
        """

        # n_samples
        assert type(n_samples) is int, "n_samples should be an integer."
        assert n_samples > 0, "n_samples should be positive."

        # return_posterior
        # assert type(return_posterior) is bool, "return_posterior should be a boolean."

        # reproducible
        assert type(reproducible) is np.random.RandomState, "reproducible should be a np.random.RandomState object."

        # Sklearn's GMM implementation reproducibility is managed at object level
        # Thus we have to set the generative model random state if we want reproducibility.
        self.distribution.random_state = reproducible
        X, modality = self.distribution.sample(n_samples)
        Y = modality
        for mode in range(self.n_modes):
            class_ = self.mode_classes[mode]
            Y[Y == mode] = class_

        Xs, Ys = shuffle(X, Y, random_state=reproducible)
        Ps_modes = self.distribution.predict_proba(Xs)
        Ps = np.zeros((Ps_modes.shape[0], self.n_classes))
        for mode in range(self.n_modes):
            class_ = self.mode_classes[mode]
            Ps[:, class_] += Ps_modes[:, mode]

        return Xs, Ys, Ps

    def plot(self, X, Y, posteriors=None, font=None, axis_set=None, axis_bayes_optimal=None):

        assert self.n_features in (1, 2)

        # X
        assert type(X) is np.ndarray
        assert X.ndim == 2

        # Setting font if provided
        if font is not None:
            matplotlib.rc('font', **font)
        cmap = matplotlib.cm.get_cmap('tab10')

        if self.n_features == 1:

            assert X.shape[1] == 1

            # Initializing figure
            fig = plt.figure(figsize=(12, 12))
            gs = fig.add_gridspec(1, 1)
            ax1 = fig.add_subplot(gs[0, 0])

            means = np.reshape(self.means, (-1,))
            stdevs = np.reshape(self.covariances, (-1,))

            # Defining the span of visual interest
            min_x = np.min(means - 4 * stdevs)
            max_x = np.max(means + 5 * stdevs)
            x = np.linspace(min_x, max_x, 1000)

            # Calculating class pdfs
            yi = [self.mode_probabilities[i] * norm(means[i], stdevs[i]).pdf(x) for i in range(len(stdevs))]

            # Plotting each distribution
            colors = [cmap(0 + i / self.n_classes) for i in range(self.n_classes)]
            for i in range(self.n_classes):
                plt.plot(x, yi[i], c=colors[i])

            # Plotting data points with color corresponding to the argmax of the posterior probabilities
            y = -0.05
            s = 2
            point_colors_1 = np.array(colors)[np.argmax(posteriors, axis=1)]
            plt.scatter(X, y * np.ones(X.shape[0]), c=point_colors_1, s=3)  # 10*Px)
            plt.text(x=np.max(X) + 0.1, y=y - 0.003, s="- Optimal decision")

            # Plotting data points with color corresponding to the correct class
            y = -0.07
            s = 2
            point_colors_2 = np.array(colors)[Y]
            plt.scatter(X, y * np.ones(X.shape[0]), c=point_colors_2, s=3)  # 10*Px)
            plt.text(x=np.max(X) + 0.1, y=y - 0.003, s="- Ground Truth")

            # Setting axes span
            ax1.set_ylim([-0.1, 1.1 * np.max(yi)])
            ax1.set_xlim([min_x, max_x])

        elif self.n_features == 2:

            assert X.shape[1] == 2

            if axis_set is not None:
                ax1 = axis_set
                ax2 = axis_bayes_optimal
            else:

                # Initializing figure
                fig = plt.figure(figsize=(12, 24))
                gs = fig.add_gridspec(2, 1)
                ax1 = fig.add_subplot(gs[0, 0])
                ax2 = fig.add_subplot(gs[1, 0])

            # Defining the span of visual interest
            min_x = -3  # np.min(means_x - 10 * stdevs_x)
            max_x = 3  # np.max(means_x + 10 * stdevs_x)
            min_y = -3  # np.min(means_y - 10 * stdevs_y)
            max_y = 3  # np.max(means_y + 10 * stdevs_y)

            # Plotting data points with color corresponding to the argmax of the posterior probabilities
            colors = [cmap(0 + i / self.n_classes) for i in range(self.n_classes)]
            linewidths = 1
            posterior_colors = np.array(colors)[np.argmax(posteriors, axis=1)]
            gt_colors = np.array(colors)[Y]
            ax1.scatter(X[:, 0], X[:, 1], c=gt_colors, s=5, alpha=0.2)
            ax2.scatter(X[:, 0], X[:, 1], c=posterior_colors, s=5, alpha=0.2)

            # Setting axes span
            ax1.set_ylim([min_y, max_y])
            ax1.set_xlim([min_x, max_x])
            ax2.set_ylim([min_y, max_y])
            ax2.set_xlim([min_x, max_x])

            ax1.title.set_text('Ground Truth')
            ax2.title.set_text('Bayes Optimal Classifier predictions')

        else:

            raise NotImplementedError

        # Setting plot title
        ax1.set_title("Visualization of the samples")

        if axis_set is None:
            plt.show()
