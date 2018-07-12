# -*- coding: utf-8 -*-

"""
Classes to convert raw data to phuzzy numbers
"""
# https://en.wikipedia.org/wiki/Probability_distribution_fitting

import numpy as np
import pandas as pd


class Data():
    """Data"""

    def __init__(self, points, label="x"):
        """

        :param points:
        :param label:
        """

        points = np.asarray(points, dtype=float)
        self.df = pd.DataFrame(points, columns=[label])

    def estimate_probability(self):
        """
        .. math::

            z_{i}=\Phi ^{{-1}}\left({\frac  {i-a}{n+1-2a}}\right)

        """
        df = self.df
        df.sort_values(by="x", ascending=True, inplace=True)
        n = len(df)
        if n <= 10:
            a = 3. / 8.
        else:
            a = .5
        df["p_blom"] = (np.linspace(1, n, n) - a) / (n + 1. - 2. * a)
        df["p_weibull"] = (np.linspace(1, n, n)) / (n + 1)
        df["p_rossow"] = (3 * np.linspace(1, n, n) - 1) / (3 * n + 1)
        return df

    def bootstrap(self, n=10000):
        """"""
        n_points = len(self.df)
        xbar = np.empty((n, 2))
        for i in range(n):
            idxs = np.random.randint(n_points, size=n_points)
            sample = self.df.iloc[idxs]
            xbar[i, 0] = np.mean(sample)
            xbar[i, 1] = np.std(sample)
        df = pd.DataFrame(xbar, columns=["x_mean", "x_std"])
        return df

    def shuffling(self, n=10000, train_fraction=.8):
        """shuffling data"""
        n_points = len(self.df)
        xbar = np.empty((n, 2))
        sample_size = int(train_fraction * n_points)
        for i in range(n):
            idxs = np.random.randint(sample_size, size=n_points)
            sample = self.df.iloc[idxs]
            xbar[i, 0] = np.mean(sample)
            xbar[i, 1] = np.std(sample)
        df = pd.DataFrame(xbar, columns=["x_mean", "x_std"])
        return df


def get_histogram_data(values, bins=None, normed=False):
    if bins is None:
        bins = 'auto'
    frequency, edges = np.histogram(values, bins=bins, normed=normed)
    left, right = edges[:-1], edges[1:]
    center = (right + left) / 2
    df = pd.DataFrame({"frequency": frequency, "left": left, "right": right, "center": center})
    df["cum_sum"] = df.frequency.cumsum()
    df["cdf"] = df.cum_sum / df.cum_sum.max()
    return df
