# -*- coding: utf-8 -*-
"""
@author: nicolas.posocco
"""

import os
import pickle
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd


def save(setup_name, filename, payload, protocol=4):
    identifier = filename

    if not os.path.exists("reports"):
        os.mkdir("reports")

    if not os.path.exists(os.path.join("reports", setup_name)):
        os.mkdir(os.path.join("reports", setup_name))

    with open(os.path.join("reports", setup_name, identifier + ".pickle"), 'wb') as file:
        pickle.dump(payload, file, protocol=protocol)


def is_already_calculated(setup_name, identifier, already_calculated=None, parallelism=False):

    if already_calculated is None:
        if not os.path.exists("reports"):
            os.mkdir("reports")

        if not os.path.exists(os.path.join("reports", setup_name)):
            os.mkdir(os.path.join("reports", setup_name))

        return os.path.exists(os.path.join("reports", setup_name, identifier + ".pickle"))

    else:
        if identifier+".pickle" in already_calculated:
            return True

        elif parallelism == True:

            if not os.path.exists("reports"):
                os.mkdir("reports")

            if not os.path.exists(os.path.join("reports", setup_name)):
                os.mkdir(os.path.join("reports", setup_name))

            return os.path.exists(os.path.join("reports", setup_name, identifier + ".pickle"))

        return False


def get_calculated(setup_name):

    if not os.path.exists("reports"):
        os.mkdir("reports")

    if not os.path.exists(os.path.join("reports", setup_name)):
        os.mkdir(os.path.join("reports", setup_name))

    files = os.listdir(os.path.join("reports", setup_name))

    return [file for file in files if ".pickle" in file]


def savefig(setup_name, fig_dir, fig_name, fig):
    if not os.path.exists(os.path.join("reports", setup_name)):
        os.mkdir(os.path.join("reports", setup_name))

    directory = os.path.join("reports", setup_name, fig_dir)

    os.makedirs(directory, exist_ok=True)

    fig.savefig(fname=os.path.join(directory, fig_name))

    plt.close(fig)


def load_all(setup_name, protocol=4):

    assert os.path.isdir(os.path.join("reports", setup_name)), "The experiments have to be run before loading and visualizing the results is possible."

    rows_list = []
    for filename in os.listdir(os.path.join("reports", setup_name)):

        if os.path.isdir(os.path.join("reports", setup_name, filename)):
            continue

        with open(os.path.join("reports", setup_name, filename), "rb") as file:
            report = pickle.load(file)

            rows_list.append(report)

    df = pd.DataFrame(rows_list)
    return df


def group_by(reports, key):
    res = {}

    for report in reports:
        value = report[key]

        if value not in res:
            res[value] = []
        res[value].append(report)

    res2 = []
    for key in res:
        res2.append(res[key])

    return res2

