# -*- coding: utf-8 -*-

import logging

logger = logging.getLogger("phuzzy")

import sys

is_py2 = sys.version_info.major == 2

import numpy as np
import pandas as pd
import copy
from scipy.integrate import cumtrapz


class FuzzyNumber(object):
    """convex fuzzy number"""

    def __init__(self, **kwargs):
        """base fuzzy number

        :param kwargs:
        """
        self.name = kwargs.get("name", "x")
        self._df = pd.DataFrame(columns=["alpha", "low", "high"])
        self._number_of_alpha_levels = kwargs.get("number_of_alpha_levels", 11)
        self.df = kwargs.get("df")

    def _get_number_of_alpha_levels(self):
        """number of alpha levels

        :rtype: int
        :return: number of alpha levels
        """
        return self._number_of_alpha_levels

    def _set_number_of_alpha_levels(self, value):
        self._number_of_alpha_levels = int(value)

    number_of_alpha_levels = property(fget=_get_number_of_alpha_levels, fset=_set_number_of_alpha_levels,
                                      doc="number of alpha levels")

    def _get_df(self):
        """returns alpha levels

        :rtype: pandas.Dataframe
        :return: alpha level dataframe
        """
        return self._df

    def _set_df(self, value):
        self._df = value

    df = property(fget=_get_df, fset=_set_df, doc="number of alpha levels")

    def discretize(self, alpha0, alpha1, alpha_levels):
        """discretize shape function

        :param alpha0: range at alpha=0
        :param alpha1: range at alpha=1
        :param alpha_levels: number of alpha levels
        :return: None
        """
        raise NotImplementedError

    def convert_df(self, alpha_levels=None, zero=1e-10):
        df = self.df.copy()
        if alpha_levels is not None:
            self.number_of_alpha_levels = alpha_levels
        df.sort_values(['alpha'], ascending=[True], inplace=True)
        # print("!",df)
        xs_l = df.l.values
        xs_l[xs_l == 0] = zero
        alphas_l = df["alpha"].values
        xs_r = df.r.values[::-1]
        xs_r[xs_r == 0] = zero
        alphas_r = df["alpha"].values[::-1]

        alphas_new = np.linspace(0., 1., self.number_of_alpha_levels)
        xs_l_new = np.interp(alphas_new, alphas_l, xs_l)
        xs_r_new = np.interp(alphas_new, alphas_r[::-1], xs_r[::-1])
        #
        # print((xs_l, alphas_l, xs_l_new))
        # print((xs_r, alphas_r, xs_r_new))
        #
        # print((np.vstack((alphas_new, xs_l_new, xs_r_new[::-1]))))
        data = np.vstack((alphas_new, xs_l_new, xs_r_new)).T
        self.df = pd.DataFrame(columns=["alpha", "l", "r"], data=data, dtype=np.float)

    def _unify(self, other):
        """equalize number of alpha levels

        :param other:
        :return: (fuzzy_number_1, fuzzy_number_2)
        """
        old0 = copy.deepcopy(self)
        old1 = copy.deepcopy(other)
        levels = max(len(old0.df), len(old1.df))
        old0.convert_df(levels)
        old1.convert_df(levels)
        return old0, old1

    @staticmethod
    def _get_cls(s, o):
        """get class after application of an operation

        :param s:
        :param o:
        :return: cls
        """
        if isinstance(s, (Uniform, Trapezoid, Triangle)) and isinstance(o, (int, float)):
            return s.__class__
        elif isinstance(o, (int, float)):
            return FuzzyNumber
        elif isinstance(s, Uniform) and isinstance(o, Uniform):
            return Uniform
        elif isinstance(s, (Triangle, Uniform)) and isinstance(o, (Triangle, Uniform)):
            return Triangle
        elif isinstance(s, (Triangle, Trapezoid, Uniform)) and isinstance(o, (Triangle, Trapezoid, Uniform)):
            return Trapezoid
        else:
            return FuzzyNumber

    def __add__(self, other):
        """adds a fuzzy number

        :param other: phuzzy.FuzzyNumber
        :return: fuzzy number
        """

        if isinstance(other, (int, float)):
            cls = self.__class__
            df = self.df.copy()
            df.update(df[["l", "r"]] + other)
            new = cls(alpha0=df.iloc[0][["l", "r"]].values,
                      alpha1=df.iloc[-1][["l", "r"]].values,
                      number_of_alpha_levels=len(df))
            new.df = df
            new.name = "{}+{}".format(self.name, other)
        else:
            old0, old1 = self._unify(other)
            quotients = np.vstack([old0.df.l + old1.df.l,
                                   old0.df.l + old1.df.r,
                                   old0.df.r + old1.df.l,
                                   old0.df.r + old1.df.r])
            df = pd.DataFrame.from_dict({"alpha": old0.df.alpha,
                                         "l": np.nanmin(quotients, axis=0),
                                         "r": np.nanmax(quotients, axis=0)})
            cls = self._get_cls(self, other)
            new = cls(alpha0=df.iloc[0][["l", "r"]].values,
                      alpha1=df.iloc[-1][["l", "r"]].values,
                      number_of_alpha_levels=len(df))
            new.df = df
            new.name = "{}+{}".format(self.name, other.name)
        return new

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        """substract a fuzzy number

        :param other: phuzzy.FuzzyNumber
        :return: fuzzy number
        """

        if isinstance(other, (int, float)):
            cls = self.__class__
            df = self.df.copy()
            df.update(df[["l", "r"]] - other)
            new = cls(alpha0=df.iloc[0][["l", "r"]].values,
                      alpha1=df.iloc[-1][["l", "r"]].values,
                      number_of_alpha_levels=len(df))
            new.df = df
            new.name = "{}-{}".format(self.name, other)
        else:
            cls = self._get_cls(self, other)
            old0, old1 = self._unify(other)
            quotients = np.vstack([old0.df.l - old1.df.l,
                                   old0.df.l - old1.df.r,
                                   old0.df.r - old1.df.l,
                                   old0.df.r - old1.df.r])
            df = pd.DataFrame.from_dict({"alpha": old0.df.alpha,
                                         "l": np.nanmin(quotients, axis=0),
                                         "r": np.nanmax(quotients, axis=0)})
            new = cls(alpha0=df.iloc[0][["l", "r"]].values,
                      alpha1=df.iloc[-1][["l", "r"]].values,
                      number_of_alpha_levels=len(df))
            new.df = df
            new.name = "{}-{}".format(self.name, other.name)
        return new

    def __rsub__(self, other):
        return self.__sub__(other)

    def __mul__(self, other):
        """multiply with a fuzzy number

        :param other: phuzzy.FuzzyNumber
        :return: fuzzy number
        """

        # fixme: zeros, infs, nans
        cls = self._get_cls(self, other)
        if isinstance(other, (int, float)):
            df = self.df.copy()
            df.update(df[["l", "r"]] * other)
            new = cls(alpha0=df.iloc[0][["l", "r"]].values,
                      alpha1=df.iloc[-1][["l", "r"]].values,
                      number_of_alpha_levels=len(df))
            new.df = df
            new.name = "{}*{}".format(self.name, other)
        else:
            old0, old1 = self._unify(other)
            quotients = np.vstack([old0.df.l * old1.df.l,
                                   old0.df.l * old1.df.r,
                                   old0.df.r * old1.df.l,
                                   old0.df.r * old1.df.r])
            df = pd.DataFrame.from_dict({"alpha": old0.df.alpha,
                                         "l": np.nanmin(quotients, axis=0),
                                         "r": np.nanmax(quotients, axis=0)})
            cls = self._get_cls(self, other)
            new = cls(alpha0=df.iloc[0][["l", "r"]].values,
                      alpha1=df.iloc[-1][["l", "r"]].values,
                      number_of_alpha_levels=len(df))
            new.df = df
            new.name = "{}*{}".format(self.name, other.name)
        return new

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        """divide by a fuzzy number

        :param other: phuzzy.FuzzyNumber
        :return: fuzzy number
        """

        # fixme: zeros, infs, nans
        cls = self._get_cls(self, other)
        if isinstance(other, (int, float)):
            df = self.df.copy()
            df.update(df[["l", "r"]] / other)
            new = cls(alpha0=df.iloc[0][["l", "r"]].values,
                      alpha1=df.iloc[-1][["l", "r"]].values,
                      number_of_alpha_levels=len(df))
            new.df = df
            new.name = "{}/{}".format(self.name, other)
        else:
            old0, old1 = self._unify(other)
            quotients = np.vstack([old0.df.l / old1.df.l,
                                   old0.df.l / old1.df.r,
                                   old0.df.r / old1.df.l,
                                   old0.df.r / old1.df.r])
            df = pd.DataFrame.from_dict({"alpha": old0.df.alpha,
                                         "l": np.nanmin(quotients, axis=0),
                                         "r": np.nanmax(quotients, axis=0)})
            cls = self._get_cls(self, other)
            new = cls(alpha0=df.iloc[0][["l", "r"]].values,
                      alpha1=df.iloc[-1][["l", "r"]].values,
                      number_of_alpha_levels=len(df))
            new.df = df
            new.name = "{}/{}".format(self.name, other.name)
        return new

    __div__ = __truediv__

    # __floordiv__ = __truediv__

    def __rdiv__(self, other):
        return self.__div__(other)

    def __pow__(self, other):
        """apply power of a fuzzy number

        :param other: phuzzy.FuzzyNumber
        :return: fuzzy number
        """

        # fixme: zeros, infs, nans
        cls = FuzzyNumber  # self._get_cls(self, other)
        if isinstance(other, (int, float)):
            if isinstance(self, Uniform):
                cls = Uniform
            df = self.df.copy()
            df.update(df[["l", "r"]] ** other)
            new = cls(alpha0=df.iloc[0][["l", "r"]].values,
                      alpha1=df.iloc[-1][["l", "r"]].values,
                      number_of_alpha_levels=len(df))
            new.df = df
            new.name = "{}^{}".format(self.name, other)
        else:
            old0, old1 = self._unify(other)
            quotients = np.vstack([old0.df.l ** old1.df.l,
                                   old0.df.l ** old1.df.r,
                                   old0.df.r ** old1.df.l,
                                   old0.df.r ** old1.df.r])
            df = pd.DataFrame.from_dict({"alpha": old0.df.alpha,
                                         "l": np.nanmin(quotients, axis=0),
                                         "r": np.nanmax(quotients, axis=0)})
            new = cls(alpha0=df.iloc[0][["l", "r"]].values,
                      alpha1=df.iloc[-1][["l", "r"]].values,
                      number_of_alpha_levels=len(df))
            new.df = df
            new.name = "{}^{}".format(self.name, other.name)
        new.make_convex()
        return new

    def __rpow__(self, other):
        """apply exponent of a fuzzy number

        :param other: phuzzy.FuzzyNumber
        :return: fuzzy number
        """

        # fixme: zeros, infs, nans
        cls = FuzzyNumber  # self._get_cls(self, other)
        if isinstance(other, (int, float)):
            if isinstance(self, Uniform):
                cls = Uniform
            df = self.df.copy()
            df.update(other ** df[["l", "r"]])
            new = cls(alpha0=df.iloc[0][["l", "r"]].values,
                      alpha1=df.iloc[-1][["l", "r"]].values,
                      number_of_alpha_levels=len(df))
            new.df = df
            new.name = "{}^{}".format(self.name, other)
        else:
            old0, old1 = self._unify(other)
            quotients = np.vstack([old0.df.l ** old1.df.l,
                                   old0.df.l ** old1.df.r,
                                   old0.df.r ** old1.df.l,
                                   old0.df.r ** old1.df.r])
            df = pd.DataFrame.from_dict({"alpha": old0.df.alpha,
                                         "l": np.nanmin(quotients, axis=0),
                                         "r": np.nanmax(quotients, axis=0)})
            new = cls(alpha0=df.iloc[0][["l", "r"]].values,
                      alpha1=df.iloc[-1][["l", "r"]].values,
                      number_of_alpha_levels=len(df))
            new.df = df
            new.name = "{}^{}".format(self.name, other.name)
        new.make_convex()
        return new

    def __neg__(self):
        """apply unary neg operator to a fuzzy number

        :param other: phuzzy.FuzzyNumber
        :return: fuzzy number
        """
        cls = self.__class__
        quotients = np.vstack([-self.df.l, -self.df.r])
        df = pd.DataFrame.from_dict({"alpha": self.df.alpha,
                                     "l": np.nanmin(quotients, axis=0),
                                     "r": np.nanmax(quotients, axis=0)})
        new = cls(alpha0=df.iloc[0][["l", "r"]].values,
                  alpha1=df.iloc[-1][["l", "r"]].values,
                  number_of_alpha_levels=len(df))
        new.df = df
        new.name = "-{}".format(self.name)
        new.make_convex()
        return new

    def __abs__(self):
        """apply abs operator to a fuzzy number

        :param other: phuzzy.FuzzyNumber
        :return: fuzzy number
        """
        dfl0 = self.df.l.copy()
        dfl0[dfl0 <= 0] = 0
        dfr0 = self.df.r.copy()
        dfr0[dfr0 <= 0] = 0

        if ((self.df[["l", "r"]].values <= 0).all()) or ((self.df[["l", "r"]].values >= 0).all()):
            cls = self.__class__
            quotients = np.vstack([abs(self.df.l), abs(self.df.r)])
        else:
            cls = FuzzyNumber
            quotients = np.vstack([dfl0, dfr0, abs(self.df.l), abs(self.df.r)])

        df = pd.DataFrame.from_dict({"alpha": self.df.alpha,
                                     "l": np.nanmin(quotients, axis=0),
                                     "r": np.nanmax(quotients, axis=0)})
        new = cls(alpha0=df.iloc[0][["l", "r"]].values,
                  alpha1=df.iloc[-1][["l", "r"]].values,
                  number_of_alpha_levels=len(df))
        new.df = df
        new.name = "|{}|".format(self.name)
        new.make_convex()
        return new

    def __lt__(self, other):
        """operation <

        :return: True or False
        """

        if isinstance(other, (int, float)):
            return self.max() < other
        else:
            return self.max() < other.min()

    # def __le__(self, other):
    #     """operation <=
    #
    #     :rtype: bool
    #     :return: True or False
    #     """
    #
    #     if isinstance(other, (int, float)):
    #         return self.max() <= other
    #     else:
    #         return self.max() <= other.max()

    def __eq__(self, other):
        """operation ==

        :rtype: bool
        :return: True or False
        """

        if isinstance(other, (int, float)):
            return False  # (self.min() >= other) and (self.max() <= other)
        else:
            return np.allclose(self.df.values, other.df.values)

    def __ne__(self, other):
        """operation !=

        :rtype: bool
        :return: True or False
        """

        return not self.__eq__(other)

    # def __ge__(self, other):
    #     """operation >=
    #
    #     :rtype: bool
    #     :return: True or False
    #     """
    #
    #     if isinstance(other, (int, float)):
    #         return self.min() <= other
    #     else:
    #         return self.min() <= other.max()

    def __gt__(self, other):
        """operation >

        :rtype: bool
        :return: True or False
        """

        if isinstance(other, (int, float)):
            return self.min() > other
        else:
            return self.min() > other.max()

    def __contains__(self, other):
        """operator in

        :rtype: bool
        :return: True or False
        """

        if isinstance(other, (int, float)):
            return (self.min() <= other) and (self.max() >= other)
        else:
            return (self.min() <= other.min()) and (self.max() >= other.max())

    def min(self):
        """minimum

        :rtype: float
        :return: min value of df
        """
        return self.df[["l", "r"]].values.min()

    def max(self):
        """maximal

        :rtype: float
        :return: max value of df
        """
        return self.df[["l", "r"]].values.max()

    def mean(self):
        """mean value

        :rtype: float
        :return: mean value
        """
        return self.ppf(.5)

    def make_convex(self):
        """make fuzzy number convex

        :return: None
        """
        for i in self.df.index:
            self.df.loc[i, "l"] = self.df.loc[i:, "l"].min()
            self.df.loc[i, "r"] = self.df.loc[i:, "r"].max()
            # self.df.loc[i, "l"] = np.nanmin(self.df.loc[i:, "l"])
            # self.df.loc[i, "r"] = np.nanmax(self.df.loc[i:, "r"])
            # print(i, self.df.loc[i:, "l"].min(), self.df.loc[i:, "r"].max())

    @property
    def alpha0(self):
        """row for alpha=0"""
        if self.df is not None:
            return self.df.iloc[0]

    @property
    def alpha1(self):
        """row for alpha=1"""
        if self.df is not None:
            return self.df.iloc[-1]

    # @property
    # def area(self):
    #     """integral of the fuzzy number
    #
    #     :return: area
    #     """
    #     raise NotImplementedError
    #     # A = ((self.alpha0.r - self.alpha0.l) - (self.df.r.values - self.df.l.values)).sum()
    #     # return A

    def import_csv(self, fh):
        """load alpha levels from csv

        :param fh: csv file path or file handle
        :return: alpha level dataframe
        """
        if isinstance(fh, str):
            fh = open(fh, "r")

        # self.df = pd.DataFrame.from_csv(fh)
        self.df = pd.read_csv(fh)
        # print(self.df.head())
        return self.df

    def export_csv(self, filepath=None):
        """export alpha levels to csv

        :param filepath: csv file path
        :return:
        """
        logger.info("export df '%s'" % filepath)
        if self.df is not None:
            res = self.df.to_csv(filepath)
            return res

    @classmethod
    def from_data(cls, **kwargs):
        """instantiate fuzzy number from attributes

        :param kwargs:
        :rtype: phuzzy.FuzzyNumber or derived object
        :return: fuzzy number
        """

        data = kwargs.get("data")
        data = np.asarray(data)
        kwargs["alpha0"] = [data.min(), data.max()]
        mean = data.mean()
        kwargs["alpha1"] = [mean]
        p = Triangle(**kwargs)
        return p

    def __str__(self):
        return "{0.__class__.__name__}({0.name}:{0.get_01})".format(self)

    def __repr__(self):
        return "{0.__class__.__name__}({0.name}:{0.get_01})".format(self)

    def to_str(self):
        """serialize fuzzy number to string

        :return: fuzzy number string
        """
        raise NotImplementedError

    @classmethod
    def from_str(cls, s):
        """deserialize fuzzy number to string

        :return: fuzzy number string
        """
        raise NotImplementedError

    def pdf(self, x):
        """Probability density function

        :param x: x values
        :return:
        """

        y_ = np.hstack((self.df.alpha, self.df.alpha[::-1]))
        x_ = np.hstack((self.df.l, self.df.r[::-1]))
        I = np.trapz(y_, x_)
        y = np.interp(x, x_, y_ / I, left=0., right=0.)
        return y

    def cdf(self, x, **kwargs):
        """Cumulative distribution function

        :param x: x values
        :param n: number of integration points
        :return: y
        """

        n = kwargs.get("n", 1000)
        y_ = np.hstack((self.df.alpha, self.df.alpha[::-1]))
        x_ = np.hstack((self.df.l, self.df.r[::-1]))

        x__ = np.linspace(self.alpha0.l, self.alpha0.r, n)
        y__ = np.interp(x__, x_, y_)

        I = cumtrapz(y__, x__, initial=0)
        I /= I[-1]
        y = np.interp(x, x__, I, left=0., right=1.)
        return y

    def ppf(self, x, **kwargs):
        """Percent point function (inverse of cdf-percentiles).

        :param x: x values
        :param n: number of integration points
        :return: y
        """

        n = kwargs.get("n", 1000)
        y_ = np.hstack((self.df.alpha, self.df.alpha[::-1]))
        x_ = np.hstack((self.df.l, self.df.r[::-1]))

        x__ = np.linspace(self.alpha0.l, self.alpha0.r, n)
        y__ = np.interp(x__, x_, y_)

        I = cumtrapz(y__, x__, initial=0)
        I /= I[-1]
        y = np.interp(x, I, x__, left=0., right=1.)
        return y

    @property
    def get_01(self):
        """get alpha=0 and alpha=1 values

        :return: [[a0_l, a0_r], [a1_l, a1_r]]
        """
        if self.df is not None and len(self.df) > 1:
            return self.df.iloc[[0, -1]][["l", "r"]].values.tolist()
        else:
            return []


class Triangle(FuzzyNumber):
    """triange fuzzy number"""

    def __init__(self, **kwargs):
        FuzzyNumber.__init__(self, **kwargs)
        alpha0 = kwargs.get("alpha0")
        alpha1 = kwargs.get("alpha1")
        self.discretize(alpha0=alpha0, alpha1=alpha1, alpha_levels=self.number_of_alpha_levels)

    def discretize(self, alpha0, alpha1, alpha_levels):
        """discretize shape function

        :param alpha0: range at alpha=0
        :param alpha1: range at alpha=1
        :param alpha_levels: number of alpha levels
        :return: None
        """
        # assert isinstance(alpha0, collections.Sequence) and len(alpha0) == 2
        # assert isinstance(alpha1, collections.Sequence) and len(alpha1) > 0
        self._a = alpha0[0]
        self._b = alpha0[1]
        self._c = alpha1[0]
        self.df = pd.DataFrame(columns=["alpha", "l", "r"],
                               data=[[0., alpha0[0], alpha0[1]], [1., alpha1[0], alpha1[0]]], dtype=np.float)
        self.df.sort_values(['alpha'], ascending=[True], inplace=True)
        self.convert_df(alpha_levels=alpha_levels)

    @classmethod
    def from_data(cls, **kwargs):
        """instantiate fuzzy number from attributes

        :param kwargs:
        :rtype: phuzzy.FuzzyNumber or derived object
        :return: fuzzy number
        """

        n = kwargs.get("n", 100)
        data = kwargs.get("data")
        data = np.asarray(data, float)
        datamin = data.min()
        datamax = data.max()
        kwargs["alpha0"] = [datamin, datamax]
        means = []
        mean = 3 * data.mean() - datamin - datamax
        means.append(mean)
        for _ in range(n):
            train_data = np.random.choice(data, int(len(data) * 50))
            # mean = 3 * train_data.mean() - train_data.min() - train_data.max()
            mean = 3 * train_data.mean() - (train_data.min() + datamin) / 2 - (train_data.max() + datamax) / 2
            means.append(mean)

        mean = np.array(means).mean()
        kwargs["alpha1"] = [mean]
        p = cls(**kwargs)
        return p

    def pdf(self, x):
        """https://en.wikipedia.org/wiki/Triangular_distribution"""
        a = self._a
        b = self._b
        c = self._c
        x = np.asarray(x)
        condlist = [x <= self._a, x <= self._c, x == c, x < self._b, x >= self._b]
        choicelist = [0.,
                      2. * (x - a) / (b - a) / (c - a),
                      2. / (b - a),
                      2. * (b - x) / (b - a) / (b - c),
                      0.]
        return np.select(condlist, choicelist)

    def cdf(self, x, **kwargs):
        """Cumulative distribution function

        :param x: x values
        :param n: number of integration points
        :return: y
        """

        a = self._a
        b = self._b
        c = self._c
        x = np.asarray(x)
        condlist = [x <= self._a, x <= self._c, x < self._b, x >= self._b]
        choicelist = [0.,
                      (x - a) ** 2 / (c - a) / (b - a),
                      1. - (b - x) ** 2 / (b - a) / (b - c),
                      1.]
        return np.select(condlist, choicelist)

    def to_str(self):
        if len(self.df) > 0:
            return "tri[{:.3g}, {:.3g}, {:.3g}]".format(self.df.iloc[0].l, self.df.iloc[0].r, self.df.iloc[-1].l)
        else:
            return "tri[nan, nan, nan]"


class Trapezoid(FuzzyNumber):
    """triange fuzzy number"""

    def __init__(self, **kwargs):
        FuzzyNumber.__init__(self, **kwargs)
        alpha0 = kwargs.get("alpha0")
        alpha1 = kwargs.get("alpha1")
        self.discretize(alpha0=alpha0, alpha1=alpha1, alpha_levels=self.number_of_alpha_levels)

    def discretize(self, alpha0, alpha1, alpha_levels):
        """discretize shape function

        :param alpha0: range at alpha=0
        :param alpha1: range at alpha=1
        :param alpha_levels: number of alpha levels
        :return: None
        """

        # assert isinstance(alpha0, collections.Sequence) and len(alpha0) == 2
        # assert isinstance(alpha1, collections.Sequence) and len(alpha1) == 2
        self._a = alpha0[0]
        self._b = alpha1[0]
        self._c = alpha1[1]
        self._d = alpha0[1]
        # todo: check a <= c <= d <= b
        self.df = pd.DataFrame(columns=["alpha", "l", "r"],
                               data=[[0., alpha0[0], alpha0[1]], [1., alpha1[0], alpha1[1]]], dtype=np.float)
        self.df.sort_values(['alpha'], ascending=[True], inplace=True)
        self.convert_df(alpha_levels=alpha_levels)

    def pdf(self, x):
        """pdf

        :param x:
        :return:
        """
        # https://en.wikipedia.org/wiki/Trapezoidal_distribution
        a = self._a
        b = self._b
        c = self._c
        d = self._d
        x = np.asarray(x)
        condlist = [x <= self._a, x <= self._b, x <= self._c, x < self._d, x >= self._d]
        choicelist = [0.,
                      2. / (c + d - a - b) * (x - a) / (b - a),
                      2. / (c + d - a - b),
                      2. / (c + d - a - b) * (d - x) / (d - c),
                      0.]
        return np.select(condlist, choicelist)

    def cdf(self, x, **kwargs):
        """Cumulative distribution function

        :param x: x values
        :param n: number of integration points
        :return: y
        """

        a = self._a
        b = self._b
        c = self._c
        d = self._d
        x = np.asarray(x)
        condlist = [x <= self._a, x <= self._b, x <= self._c, x < self._d, x >= self._d]
        choicelist = [0.,
                      (x - a) ** 2 / (c + d - a - b) / (b - a),
                      (2 * x - a - b) / (d + c - a - b),
                      1 - (d - x) ** 2 / (d - c) / (d + c - a - b),
                      1.]
        return np.select(condlist, choicelist)

    def to_str(self):
        if len(self.df) > 0:
            return "trap[{:.3g}, {:.3g}, {:.3g}, {:.3g}]".format(self.df.iloc[0].l, self.df.iloc[0].r,
                                                                 self.df.iloc[-1].l, self.df.iloc[-1].r)
        else:
            return "trap[nan, nan, nan, nan]"


class Uniform(FuzzyNumber):
    """triange fuzzy number"""

    def __init__(self, **kwargs):
        FuzzyNumber.__init__(self, **kwargs)
        alpha0 = kwargs.get("alpha0")
        self.discretize(alpha0=alpha0, alpha1=None, alpha_levels=self.number_of_alpha_levels)

    def discretize(self, alpha0, alpha1, alpha_levels):
        """discretize shape function

        :param alpha0: range at alpha=0
        :param alpha1: range at alpha=1
        :param alpha_levels: number of alpha levels
        :return: None
        """

        # assert isinstance(alpha0, collections.Sequence) and len(alpha0) == 2
        self._a = alpha0[0]
        self._b = alpha0[1]
        self.df = pd.DataFrame(columns=["alpha", "l", "r"],
                               data=[[0., alpha0[0], alpha0[1]], [1., alpha0[0], alpha0[1]]], dtype=np.float)
        self.df.sort_values(['alpha'], ascending=[True], inplace=True)
        self.convert_df(alpha_levels=alpha_levels)

    def pdf(self, x):
        """https://en.wikipedia.org/wiki/Uniform_distribution_(continuous)"""
        a = self._a
        b = self._b
        x = np.asarray(x)
        condlist = [x <= self._a, x < self._b, x >= self._b]
        choicelist = [0.,
                      1. / (b - a),
                      0.]
        return np.select(condlist, choicelist)

    def cdf(self, x, **kwargs):
        """Cumulative distribution function

        :param x: x values
        :param n: number of integration points
        :return: y
        """

        a = self._a
        b = self._b
        x = np.asarray(x)
        condlist = [x <= self._a, x < self._b, x >= self._b]
        choicelist = [0.,
                      (x - a) / (b - a),
                      1.]
        return np.select(condlist, choicelist)

    def to_str(self):
        return "Uniform[{:.4g},{:.4g}]".format(self.alpha0.l, self.alpha0.r)

    # @classmethod
    # def from_str(cls, s):
    #     raise NotImplementedError
