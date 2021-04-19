# -*- coding: utf-8 -*-
"""
@author: nicolas.posocco
"""

from abc import ABC
import numpy as np


class MulticlassDistribution(ABC):

    def __init__(self):
        """
        Initializes the distribution, allowing later sampling and posterior probabilities calculations.
        """

        pass

    def sample(self, n_samples, return_posterior=True, reproducible=None):
        """
        Samples n_samples times from the distribution, and their label.
        Returns also the array of posterior probabilities if return_posterior=True.
        """

        # n_samples
        assert type(n_samples) is int, "n_samples should be an integer."
        assert n_samples > 0, "n_samples should be positive."

        # return_posterior
        assert type(return_posterior) is bool, "return_posterior should be a boolean."

        # reproducible
        assert type(reproducible) is np.random.RandomState, "reproducible should be a np.random.RandomState object."

        raise NotImplementedError

    def posteriors(self, X):

        # X
        assert type(X) is np.ndarray, "X should be a numpy array."
        assert X.ndim == 2, "X should be of shape (n_samples, n_features), here is of shape {}".format(X.shape)

        raise NotImplementedError

    def get_bayes_classifier(self):
        """
        Instanciates the optimal Bayes classifier for this distribution.
        """

        return BayesClassifier(distribution=self)


class BayesClassifier(ABC):

    def __init__(self, distribution):

        # distribution
        assert isinstance(distribution,
                          MulticlassDistribution), "distribution should inherit from MulticlassDistribution."

        self.distribution = distribution

    def fit(self, X):
        pass

    def predict_proba(self, X):

        # X
        assert type(X) is np.ndarray, "X should be a numpy array, here is a {}.".format(type(X))
        assert X.ndim == 2, "X should be of shape (n_samples, n_features), here is of shape {}".format(X.shape)

        posteriors = self.distribution.posteriors(X)

        return posteriors

    def predict(self, X):

        # X
        assert type(X) is np.ndarray, "X should be a numpy array, here is a {}.".format(type(X))
        assert X.ndim == 2, "X should be of shape (n_samples, n_features), here is of shape {}".format(X.shape)

        posteriors = self.predict_proba(X)

        return np.argmax(posteriors, axis=0)