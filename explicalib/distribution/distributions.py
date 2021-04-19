# -*- coding: utf-8 -*-
"""
@author: nicolas.posocco
"""

from scipy.stats import binom
from src.explicalib.dataset.utils import normal_expectancy_to_binomial
import matplotlib.pyplot as plt
import numpy as np


class MulticlassBinomialMultivatiateDistribution(object):

    def __init__(self, N, p, pyi):
        """
        Initializes a multivariate binomial multiclass distribution.
        Args:
            N: np.ndarray (shape (n_classes, dimensionality)), TODO.
            p: np.ndarray (shape (n_classes, dimensionality)), TODO.
            pyi: TODO.
        """

        # N
        assert type(N) is np.ndarray and N.ndim == 2 and N.dtype != np.object

        # p
        assert type(p) is np.ndarray and p.ndim == 2 and p.dtype != np.object

        # pyi
        assert type(pyi) is np.ndarray and pyi.ndim == 1 and pyi.dtype != np.object

        # Shapes compatibility
        assert p.shape[0] == N.shape[0] == pyi.shape[0], "Could not infer number of classes.\np.shape[0], N.shape[0] " \
                                                         "and pyi.shape[0] should be equal to the number of classes, " \
                                                         "found {}, {} and {}.".format(p.shape[0], N.shape[0],
                                                                                       pyi.shape[0])

        self.dimensionality = N.shape[1]  # Problem dimensionality
        assert p.shape[1] == self.dimensionality
        self.n_class = p.shape[0]  # Number of classes

        self.N = N
        self.p = p
        self.pyi = pyi

    def calculate_posterior(self, X):
        """
        Calculates the posterior probabilities for batch X based on the known sampling method.
        Args:
            X: numpy.ndarray (shape n_samples*dimensionality), batch of samples.

        Returns:
        The true posterior probabilities for batch X.
        """

        # X
        assert type(X) is np.ndarray and X.ndim == 2 and X.dtype != np.object

        size = X.shape[0]  # Size of sampled dataset

        # Allocating posterior array
        P = np.zeros(shape=(size, self.n_class))

        # For each sample
        for i in range(size):
            # The prior probability is the product of the probability of the individual vector components
            pxjkyi = np.array([[np.array(binom.pmf(k=X[i, k], n=self.N[j, k], p=self.p[j, k]))
                                for j in range(self.n_class)
                                ] for k in range(self.dimensionality)
                               ])
            pxkyi = np.prod(pxjkyi, axis=0)

            # Sample probability is calculated with marginalization
            px = np.sum(pxkyi * self.pyi)

            # Getting posterior for sample for each class via Bayes rule
            pyikx = pxkyi * self.pyi / px

            # Writing class posteriors for sample
            P[i, :] = np.array(pyikx)

        return P

    def sample(self, size, reproductible, normalize_x=False):
        """
        Generates a dataset based on init parameters and sampling method.
        Args:
            size: int, size of the dataset being generated.
            reproductible: numpy.random.RandomState object, used to guarantee reproductability.

        Returns:
        A numpy.ndarray (shape n_samples*dimensionality) containing the dataset.
        """

        # size
        # TODO

        # reproductible
        # TODO

        # Choosing from which class the data point is sampled
        classes = [i for i in range(len(self.N))]
        Y = reproductible.choice(classes, size, True, self.pyi)

        # Sampling data point from corresponding binomial distribution
        d = len(self.N[0])  # Problem dimensionality
        X = np.array([[reproductible.binomial(self.N[Y[i]][j], self.p[Y[i]][j])
                       for j in range(d)
                       ] for i in range(Y.shape[0])])

        # Calculating true posterior probabilities
        P = self.calculate_posterior(X)

        if normalize_x:
            for dim in range(X.shape[1]):
                X = X.astype(np.float32)
                X[:, dim] = (X[:, dim] - np.mean(X[:, dim])) / np.std(X[:, dim])

        return X, Y, P

    def visualize(self, size=10000, normalize_x=False):
        """
        Plots a visualization of the distribution.
        Args:
            size: TODO.

        Returns:
        None.
        """
        reproductible = np.random.RandomState(1)
        X, Y, P = self.sample(size=size, reproductible=reproductible)

        if normalize_x:
            for dim in range(X.shape[1]):
                X = X.astype(np.float32)
                X[:, dim] = (X[:, dim] - np.mean(X[:, dim])) / np.std(X[:, dim])

        fig = plt.figure(constrained_layout=True, figsize=(6, 8))
        gs = fig.add_gridspec(1, 1)

        # Plotting calibration vs profit on the top slot
        ax1 = fig.add_subplot(gs[0, 0])

        ax1.scatter(X[:, 0], X[:, 1], s=2, alpha=0.2, c=Y, cmap='Set1')
        plt.show()


class Line2dMulticlassBinomialMultivatiateDistribution(MulticlassBinomialMultivatiateDistribution):

    def __init__(self, n_class=5):
        """
        TODO
        Args:
            n_class: TODO
        """

        # n_class
        # TODO

        def line_2d_params(n_class):
            N = [[100, 100] for _ in range(n_class)]
            p = [[0.5 + 0.1 * i, 0.5 + 0.1 * i] for i in range(n_class)]
            pyi = [1 / n_class for _ in range(n_class)]
            return np.array(N), np.array(p), np.array(pyi)

        N, p, pyi = line_2d_params(n_class)

        super().__init__(N, p, pyi)


class Clusters2DMulticlassBinomialMultivatiateDistribution(MulticlassBinomialMultivatiateDistribution):

    def __init__(self, n_class):
        """
        TODO
        Args:
            n_class: TODO
        """

        # n_class
        # TODO

        # radius
        # TODO

        centers_x = np.random.randint(60, 100, size=n_class)
        centers_y = np.random.randint(60, 100, size=n_class)
        radii = np.random.randint(10, 30, size=n_class)

        N, p = normal_expectancy_to_binomial(centers_x, centers_y, radii)
        pyi = [1 / n_class for _ in range(n_class)]

        super().__init__(np.array(N), np.array(p), np.array(pyi))


class Flower2DMulticlassBinomialMultivatiateDistribution(MulticlassBinomialMultivatiateDistribution):

    def __init__(self, n_class, radius):
        """
        TODO
        Args:
            n_class: TODO
            radius: TODO
        """

        # n_class
        # TODO

        # radius
        # TODO

        def flower_2d_params(n_class, radius):
            # Center and radius for each class
            centres_x = [60] + [60 + radius * np.sin(2 * i * np.pi / (n_class - 1)) for i in range(1, n_class)]
            centres_y = [60] + [60 + radius * np.cos(2 * i * np.pi / (n_class - 1)) for i in range(1, n_class)]
            radii = [1.3 * radius for _ in range(n_class)]
            N, p = normal_expectancy_to_binomial(centres_x, centres_y, radii)
            pyi = [1 / n_class for _ in range(n_class)]

            return np.array(N), np.array(p), np.array(pyi)

        N, p, pyi = flower_2d_params(n_class, radius)

        super().__init__(N, p, pyi)
