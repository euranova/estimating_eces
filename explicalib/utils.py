# -*- coding: utf-8 -*-
"""
@author: nicolas.posocco
"""

from scipy.integrate import quad
from multiprocessing import Pool
import numpy as np
import tqdm
import os
from KDEpy import FFTKDE
from scipy.interpolate import interp1d

from scipy.optimize import curve_fit


class Model1D(object):

    def __init__(self, target_family, initial):
        self.target_family = target_family
        self.initial = initial
        self.params = None

    def fit(self, X, Y):
        res = curve_fit(self.target_family,
                        X, Y,
                        self.initial)
        self.params = res

    def predict(self, value):
        if type(value) is np.ndarray:
            return np.array([self.predict(value[v_i]) for v_i in range(value.shape[0])])
        else:
            return self.target_family(value, *list(self.params[0]))


class ExponentialRegression(Model1D):

    def __init__(self):
        def fun(x, alpha, beta, c, d):
            return alpha * np.exp(-beta * x + d) + c

        super().__init__(fun, (1, 1, 0, 0))


def get_all_subclasses(cls):
    all_subclasses = []

    for subclass in cls.__subclasses__():
        all_subclasses.append(subclass)
        all_subclasses.extend(get_all_subclasses(subclass))

    return all_subclasses


def compute(f, args):
    """
    f : python function with one argument
    args : list of tuples of arguments (one tuple per job), or list of elements (one element per job)
    """

    with Pool() as p:
        res = list(tqdm.tqdm(p.imap(f, args), total=len(args)))

    return res


def binomial_from_gaussian(mu, var):
    q = var / mu
    p = 1 - q
    n = mu / p

    return int(n), p


def fun_from_points(X, Y):
    def f(x):

        # Finding the index of the two surrounding xs
        ix1, ix2 = None, None
        for i in range(1, len(X)):
            if X[i] > x:
                ix1, ix2 = i - 1, i
                break
        if ix1 is None:  # if x is the right boundary
            # The last item is returned
            return Y[-1]

        # Calculating value at point x : f(x) = f(x1) + ((x-x1)/(x2-x1))*(f(x2)-f(x1))
        return Y[ix1] + ((x - X[ix1]) / (X[ix2] - X[ix1])) * (Y[ix2] - Y[ix1])

    return f


def fun_from_model(model):
    def fun(x):
        if type(x) is list or type(x) is np.array:
            x = np.array([x])
        else:
            x = np.array([[x]])
        return model.predict_proba(x)[0, 1]

    return fun


def fun_absolute_difference(f, g):
    def h(x):
        return abs(g(x) - f(x))

    return h


def fun_abs(f):
    def h(x):
        return abs(f(x))

    return h


def area_between(X, Y0, Y1, acceptable_error=0.001):
    f = fun_from_points(X, Y0)
    g = fun_from_points(X, Y1)
    h = fun_absolute_difference(f, g)

    area, error = quad(h, 0, 1, epsabs=acceptable_error)

    return area, error


def create_reports_directory(name="reports"):
    assert name not in os.listdir(), "Folder with this name is already there."

    # Creating folder
    os.mkdir(name)

    # Creating json index
    with open(os.path.join(name, "benchmark_reports.json"), "w") as json_file:
        json_file.write('''{"reports": {}}''')


def confidences_from_scores(scores_matrix, predictions, model):
    # Turning predictions into a numerical usable format
    corresps = {}
    for class_i in range(len(model.classes_)):
        corresps[model.classes_[class_i]] = class_i
    predictions_num = np.vectorize(corresps.get)(predictions)

    # Gathering scores relative to predicted class
    scores_predicted_class = []
    for i in range(len(predictions)):
        scores_predicted_class.append(scores_matrix[i, predictions_num[i]])

    return np.array(scores_predicted_class)


def voronoi_1D(X, domain):
    # X
    assert type(X) is np.ndarray
    assert X.ndim == 1

    # domain
    # TODO array-like 2 elements

    lower, upper = domain

    if len(X) == 1:
        return [(lower, upper)]

    result_sorted = [None for i in range(len(X))]

    # Sorting elements in X while tracking their indexes
    Xind_sorted = np.argsort(X)
    X_sorted = np.sort(X)

    result_sorted[0] = (lower, (X_sorted[0] + X_sorted[1]) / 2)
    for i in range(1, len(X_sorted) - 1):
        result_sorted[i] = ((float(X_sorted[i - 1]) + float(X_sorted[i])) / 2,
                            (float(X_sorted[i]) + float(X_sorted[i + 1])) / 2)
    result_sorted[-1] = ((X_sorted[-1] + X_sorted[-2]) / 2, upper)

    result = [None for i in range(len(X))]
    for i in range(len(result)):
        result[Xind_sorted[i]] = result_sorted[i]

    return result


def get_silvermans_bandwidth(X, kernel, bandwidth):

    # X
    assert X is not None
    assert type(X) is np.ndarray
    assert X.ndim == 1

    # kernel
    # assert kernel in ("triweight", )

    # bandwidth
    assert bandwidth in ("silverman",)

    kde = FFTKDE(bw=bandwidth, kernel=kernel)
    kde.fit(X)(2 ** 10)

    return kde.bw


def bounded_1d_kde(X, low_bound, high_bound, kernel='triweight', bandwidth=None, output="function", grid=None):
    #epsilon = 1e-8

    # output
    assert output in ("discrete_signal", "function")

    data = X

    if bandwidth is None:
        bandwidth = "silverman"

    if bandwidth in ("silverman", "ISJ"):
        # Determining bandwidth
        bandwidth = get_silvermans_bandwidth(X=X, kernel=kernel, bandwidth=bandwidth)
        print("Automatic bandwidth selection gave : {}".format(bandwidth))

    # Mirror the data about the domain boundary
    data = np.concatenate((data, 2 * low_bound - data))
    data = np.concatenate((data, 2 * high_bound - data))

    # Calculating kde
    fftkde = FFTKDE(bw=bandwidth, kernel=kernel).fit(data)
    if grid is not None:
        x, y = grid, fftkde(grid_points=grid)
    else:
        x, y = fftkde(2 ** 20)

    # Setting the KDE to zero outside of the domain
    y[x <= low_bound] = 0
    y[x > high_bound] = 0

    # Normalizing to get integral of ~1
    y = y * 3

    #y[y < epsilon] = epsilon

    if output == "function":
        return interp1d(x, y, kind='linear')
    else:
        # Removing out of support samples
        y = y[low_bound <= x]
        x = x[low_bound <= x]
        y = y[x <= high_bound]
        x = x[x <= high_bound]

        return x, y


def hasnan(array):
    array_sum = np.sum(array)
    return np.isnan(array_sum)


def logspace(start, end, steps):

    base = 10
    start_log = np.log(start)/np.log(base)
    end_log = np.log(end)/np.log(base)

    res = np.around(np.logspace(start_log, end_log, steps)).astype(np.int32)
    return res
