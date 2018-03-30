# -*- coding: utf-8 -*-

__title__ = "phuzzy"
__author__ = "lepy"
__email__ = "lepy@mailbox.org"
__description__ = """Fuzzy stuff"""
__long_description__ = """
fuzzy number tools
"""
__url__ = 'https://github.com/lepy/phuzzy'
__copyright__ = "Copyright (C) 2017-"
__version__ = "0.1.0"
__status__ = "3 - Alpha"
__credits__ = [""]
__license__ = """MIT"""

import logging
import os
import collections
import numpy as np
import pandas as pd
from scipy.stats import truncnorm, norm, gennorm
from scipy.integrate import cumtrapz
import copy

logger = logging.getLogger("phuzzy")


class FuzzyNumber(object):
    """convex fuzzy number"""

    def __init__(self, **kwargs):
        self.name = kwargs.get("name", "x")
        self._df = pd.DataFrame(columns=["alpha", "low", "high"])
        self._number_of_alpha_levels = kwargs.get("number_of_alpha_levels", 10)
        self.df = kwargs.get("df")

    def _get_number_of_alpha_levels(self):
        return self._number_of_alpha_levels

    def _set_number_of_alpha_levels(self, value):
        self._number_of_alpha_levels = int(value)

    number_of_alpha_levels = property(fget=_get_number_of_alpha_levels, fset=_set_number_of_alpha_levels,
                                      doc="number of alpha levels")

    def _get_df(self):
        return self._df

    def _set_df(self, value):
        self._df = value

    df = property(fget=_get_df, fset=_set_df, doc="number of alpha levels")

    def convert_df(self, alpha_levels=None, zero=1e-10):
        df = self.df.copy()
        if alpha_levels is not None:
            self.number_of_alpha_levels = alpha_levels
        df.sort_values(['alpha'], ascending=[True], inplace=True)
        # print("!",df)
        xs_l = df.low.values
        xs_l[xs_l == 0] = zero
        alphas_l = df["alpha"].values
        xs_r = df.high.values[::-1]
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
        self.df = pd.DataFrame(columns=["alpha", "low", "high"], data=data, dtype=np.float)

    def _unify(self, other):
        old0 = copy.deepcopy(self)
        old1 = copy.deepcopy(other)
        levels = max(len(old0.df), len(old1.df))
        old0.convert_df(levels)
        old1.convert_df(levels)
        return old0, old1

    def __add__(self, other):
        new = FuzzyNumber()
        old0, old1 = self._unify(other)
        quotients = np.vstack([old0.df.low + old1.df.low,
                               old0.df.low + old1.df.high,
                               old0.df.high + old1.df.low,
                               old0.df.high + old1.df.high])
        new.df = pd.DataFrame.from_dict({"alpha": old0.df.alpha,
                                         "low": np.nanmin(quotients, axis=0),
                                         "high": np.nanmax(quotients, axis=0)})
        return new

    def __sub__(self, other):
        new = FuzzyNumber()
        old0, old1 = self._unify(other)
        quotients = np.vstack([old0.df.low - old1.df.low,
                               old0.df.low - old1.df.high,
                               old0.df.high - old1.df.low,
                               old0.df.high - old1.df.high])
        new.df = pd.DataFrame.from_dict({"alpha": old0.df.alpha,
                                         "low": np.nanmin(quotients, axis=0),
                                         "high": np.nanmax(quotients, axis=0)})
        return new

    def __mul__(self, other):
        # fixme: zeros, infs, nans
        new = FuzzyNumber()
        old0, old1 = self._unify(other)
        quotients = np.vstack([old0.df.low * old1.df.low,
                               old0.df.low * old1.df.high,
                               old0.df.high * old1.df.low,
                               old0.df.high * old1.df.high])
        new.df = pd.DataFrame.from_dict({"alpha": old0.df.alpha,
                                         "low": np.nanmin(quotients, axis=0),
                                         "high": np.nanmax(quotients, axis=0)})
        return new

    def __div__(self, other):
        # fixme: zeros, infs, nans
        new = FuzzyNumber()
        old0, old1 = self._unify(other)
        quotients = np.vstack([old0.df.low / old1.df.low,
                               old0.df.low / old1.df.high,
                               old0.df.high / old1.df.low,
                               old0.df.high / old1.df.high])
        new.df = pd.DataFrame.from_dict({"alpha": old0.df.alpha,
                                         "low": np.nanmin(quotients, axis=0),
                                         "high": np.nanmax(quotients, axis=0)})
        return new

    def __pow__(self, other):
        # fixme: zeros, infs, nans
        new = FuzzyNumber()
        if isinstance(other, (int, float)):
            other = Trapezoid(alpha0=[other, other], alpha1=[other, other], number_of_alpha_levels=len(self.df))
        old0, old1 = self._unify(other)
        quotients = np.vstack([old0.df.low ** old1.df.low,
                               old0.df.low ** old1.df.high,
                               old0.df.high ** old1.df.low,
                               old0.df.high ** old1.df.high])
        new.df = pd.DataFrame.from_dict({"alpha": old0.df.alpha,
                                         "low": np.nanmin(quotients, axis=0),
                                         "high": np.nanmax(quotients, axis=0)})
        new.make_convex()
        return new

    def make_convex(self):
        for i in self.df.index:
            self.df.loc[i, "low"] = self.df.loc[i:, "low"].min()
            self.df.loc[i, "high"] = self.df.loc[i:, "high"].max()
            # self.df.loc[i, "low"] = np.nanmin(self.df.loc[i:, "low"])
            # self.df.loc[i, "high"] = np.nanmax(self.df.loc[i:, "high"])
            # print(i, self.df.loc[i:, "low"].min(), self.df.loc[i:, "high"].max())

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

    @property
    def area(self):
        A = ((self.alpha0.high - self.alpha0.low) - (self.df.high.values - self.df.low.values)).sum()
        return A

    def import_csv(self, fh):
        if isinstance(fh, str):
            fh = open(fh, "r")

        self.df = pd.DataFrame.from_csv(fh)
        print((self.df.head()))

    def export_csv(self, filepath):
        logger.info("export df '%s'" % filepath)
        if self.df is not None:
            self.df.to_csv(filepath)

    @classmethod
    def from_data(cls, **kwargs):
        p = cls(**kwargs)

        return p

    def __str__(self):
        return "({0.__class__.__name__}({0.name}))".format(self)

    def __repr__(self):
        return "(%s)" % self.name

    def to_str(self):
        raise NotImplementedError

    @classmethod
    def from_str(cls, s):
        raise NotImplementedError

    def pdf(self, x):
        y_ = np.hstack((self.df.alpha, self.df.alpha[::-1]))
        x_ = np.hstack((self.df.low, self.df.high[::-1]))
        I = np.trapz(y_, x_)
        y = np.interp(x, x_, y_ / I, left=0., right=0.)
        return y

    def cdf(self, x, n=100):
        y_ = np.hstack((self.df.alpha, self.df.alpha[::-1]))
        x_ = np.hstack((self.df.low, self.df.high[::-1]))

        x__ = np.linspace(self.alpha0.low, self.alpha0.high, n)
        y__ = np.interp(x__, x_, y_)

        I = cumtrapz(y__, x__, initial=0)
        I /= I[-1]
        y = np.interp(x, x__, I, left=0., right=1.)
        return y

    def get_01(self):
        return self.df.iloc[[0, -1]][["low", "high"]].values.tolist()


class Uniform(FuzzyNumber):
    """triange fuzzy number"""

    def __init__(self, **kwargs):
        FuzzyNumber.__init__(self, **kwargs)
        alpha0 = kwargs.get("alpha0")
        self.discretize(alpha0=alpha0, alpha1=None, alpha_levels=self.number_of_alpha_levels)

    def discretize(self, alpha0, alpha1, alpha_levels):
        assert isinstance(alpha0, collections.Sequence) and len(alpha0) == 2
        self._a = alpha0[0]
        self._b = alpha0[1]
        self.df = pd.DataFrame(columns=["alpha", "low", "high"],
                               data=[[0., alpha0[0], alpha0[1]], [1., alpha0[0], alpha0[1]]], dtype=np.float)
        self.df.sort_values(['alpha'], ascending=[True], inplace=True)
        self.convert_df(alpha_levels=alpha_levels)

    def pdf(self, x):
        """https://en.wikipedia.org/wiki/Uniform_distribution_(continuous)"""
        a = self._a
        b = self._b
        condlist = [x <= self._a, x < self._b, x >= self._b]
        choicelist = [0.,
                      1. / (b - a),
                      0.]
        return np.select(condlist, choicelist)

    def cdf(self, x):
        a = self._a
        b = self._b
        condlist = [x <= self._a, x < self._b, x >= self._b]
        choicelist = [0.,
                      (x - a) / (b - a),
                      1.]
        return np.select(condlist, choicelist)

    def to_str(self):
        return "Uniform[{:.4g},{:.4g}]".format(self.alpha0.low, self.alpha0.high)

    @classmethod
    def from_str(cls, s):
        raise NotImplementedError


class Triangle(FuzzyNumber):
    """triange fuzzy number"""

    def __init__(self, **kwargs):
        FuzzyNumber.__init__(self, **kwargs)
        alpha0 = kwargs.get("alpha0")
        alpha1 = kwargs.get("alpha1")
        self.discretize(alpha0=alpha0, alpha1=alpha1, alpha_levels=self.number_of_alpha_levels)

    def discretize(self, alpha0, alpha1, alpha_levels):
        assert isinstance(alpha0, collections.Sequence) and len(alpha0) == 2
        assert isinstance(alpha1, collections.Sequence) and len(alpha1) > 0
        self._a = alpha0[0]
        self._b = alpha0[1]
        self._c = alpha1[0]
        self.df = pd.DataFrame(columns=["alpha", "low", "high"],
                               data=[[0., alpha0[0], alpha0[1]], [1., alpha1[0], alpha1[0]]], dtype=np.float)
        self.df.sort_values(['alpha'], ascending=[True], inplace=True)
        self.convert_df(alpha_levels=alpha_levels)

    @classmethod
    def from_data(cls, **kwargs):
        n = kwargs.get("n", 100)
        data = kwargs.get("data")
        data = np.asarray(data)
        kwargs["alpha0"] = [data.min(), data.max()]
        means = []
        mean = 3 * data.mean() - data.min() - data.max()
        means.append(mean)
        print("!", mean)
        for i in range(n):
            train_data = np.random.choice(data, int(len(data) * 50))

            mean = 3 * train_data.mean() - train_data.min() - train_data.max()
            means.append(mean)

        mean = np.array(means).mean()
        kwargs["alpha1"] = [mean]
        p = cls(**kwargs)
        print("!!", mean)

        return p

    def pdf(self, x):
        """https://en.wikipedia.org/wiki/Triangular_distribution"""
        a = self._a
        b = self._b
        c = self._c
        condlist = [x <= self._a, x <= self._c, x == c, x < self._b, x >= self._b]
        choicelist = [0.,
                      2. * (x - a) / (b - a) / (c - a),
                      2. / (b - a),
                      2. * (b - x) / (b - a) / (b - c),
                      0.]
        return np.select(condlist, choicelist)

    def cdf(self, x):
        a = self._a
        b = self._b
        c = self._c
        condlist = [x <= self._a, x <= self._c, x < self._b, x >= self._b]
        choicelist = [0.,
                      (x - a) ** 2 / (c - a) / (b - a),
                      1. - (b - x) ** 2 / (b - a) / (b - c),
                      1.]
        return np.select(condlist, choicelist)

    def to_str(self):
        if len(self.df) > 0:
            return "tri[{:.3g}, {:.3g}, {:.3g}]".format(self.df.iloc[0].low, self.df.iloc[0].high, self.df.iloc[-1].low)
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
        assert isinstance(alpha0, collections.Sequence) and len(alpha0) == 2
        assert isinstance(alpha1, collections.Sequence) and len(alpha1) == 2
        self._a = alpha0[0]
        self._b = alpha1[0]
        self._c = alpha1[1]
        self._d = alpha0[1]
        # todo: check a <= c <= d <= b
        self.df = pd.DataFrame(columns=["alpha", "low", "high"],
                               data=[[0., alpha0[0], alpha0[1]], [1., alpha1[0], alpha1[1]]], dtype=np.float)
        self.df.sort_values(['alpha'], ascending=[True], inplace=True)
        self.convert_df(alpha_levels=alpha_levels)

    def pdf(self, x):
        """https://en.wikipedia.org/wiki/Trapezoidal_distribution"""
        a = self._a
        b = self._b
        c = self._c
        d = self._d
        condlist = [x <= self._a, x <= self._b, x <= self._c, x < self._d, x >= self._d]
        choicelist = [0.,
                      2. / (c + d - a - b) * (x - a) / (b - a),
                      2. / (c + d - a - b),
                      2. / (c + d - a - b) * (d - x) / (d - c),
                      0.]
        return np.select(condlist, choicelist)

    def cdf(self, x):
        a = self._a
        b = self._b
        c = self._c
        d = self._d
        condlist = [x <= self._a, x <= self._b, x <= self._c, x < self._d, x >= self._d]
        choicelist = [0.,
                      (x - a) ** 2 / (c + d - a - b) / (b - a),
                      (2 * x - a - b) / (d + c - a - b),
                      1 - (d - x) ** 2 / (d - c) / (d + c - a - b),
                      1.]
        return np.select(condlist, choicelist)

    def to_str(self):
        if len(self.df) > 0:
            return "trap[{:.3g}, {:.3g}, {:.3g}, {:.3g}]".format(self.df.iloc[0].low, self.df.iloc[0].high,
                                                                 self.df.iloc[-1].low, self.df.iloc[-1].high)
        else:
            return "trap[nan, nan, nan, nan]"


class Gauss(FuzzyNumber):
    """Symmetric Gaussian function depends on two parameters $\sigma$ and $c$
    $f(x;\sigma, c) = e ^ \frac {−(x−c)**2}{2 \sigma}$"""


class BELL(FuzzyNumber):
    """Symmetric Gaussian function depends on two parameters $\sigma$ and $c$
    $f(x; a,b,c) = \frac {1}{1+\left| \frac{x-c}{a} \right|^2b}$"""


class TruncNorm(FuzzyNumber):
    """Normal distibuted membership function"""

    def __init__(self, **kwargs):  # , mean=0., std=1., clip=None, ppf=None):
        FuzzyNumber.__init__(self, **kwargs)
        alpha0 = kwargs.get("alpha0")
        alpha1 = kwargs.get("alpha1")
        self.clip = kwargs.get("alpha0") or [0, np.inf]
        self.ppf = kwargs.get("ppf") or [.001, .999]
        self._loc = kwargs.get("mean") or np.array(alpha0).mean()
        self._scale = kwargs.get("std") or (alpha0[1] - alpha0[0]) / 6.
        # print("!", (alpha0[1]-alpha0[0])/6)
        self._distr = None
        self.discretize(alpha0=self.clip, alpha1=alpha1, alpha_levels=self.number_of_alpha_levels)

    # def __str__(self):
    #     return "tnorm(%s [%.3g,%.3g])" % (self.did, self.loc, self.std)
    #
    # __repr__ = __str__

    def _get_loc(self):
        return self._loc

    def _set_loc(self, value):
        self._loc = value

    mean = loc = property(fget=_get_loc, fset=_set_loc)

    def _get_scale(self):
        return self._scale

    def _set_scale(self, value):
        self._scale = value

    std = scale = property(fget=_get_scale, fset=_set_scale)

    @property
    def distr(self):
        if self._distr is None:
            a, b = (self.clip[0] - self.loc) / self.std, (self.clip[1] - self.loc) / self.std
            self._distr = truncnorm(a=a, b=b, loc=self.mean, scale=self.std)
        #             print "set_distr", self._distr, self.mean, self.std
        return self._distr

    def discretize(self, alpha0, alpha1, alpha_levels):
        print("alpha0", alpha0)
        assert isinstance(alpha0, collections.Sequence) and len(alpha0) == 2
        # assert isinstance(alpha1, collections.Sequence) and len(alpha1) > 0
        nn = 501
        pp = np.linspace(0, 1, nn)
        ppf = self.distr.ppf(pp)
        x = np.linspace(alpha0[0], alpha0[1], nn)
        pdf = self.distr.pdf(x)
        # alphas = np.linspace(0,pdf/pdf.max(),alpha_levels)
        alphas = pdf / pdf.max()
        data = []
        for i in range(len(x) // 2):
            data.append([alphas[i], x[i], x[::-1][i]])
        data.append([alphas[i + 1], x[i + 1], x[::-1][i + 1]])
        # print(alphas)
        # print(self.distr.mean(), self.distr.std())
        # print("x", x)
        # print("ppf", ppf)
        # print("pdf", pdf)
        self.df = pd.DataFrame(columns=["alpha", "low", "high"], data=data, dtype=np.float)
        self.convert_df(alpha_levels)
        self.df.sort_values(['alpha'], ascending=[True], inplace=True)
        self.convert_df(alpha_levels=alpha_levels)


class TruncGenNorm(FuzzyNumber):
    """Truncated generalized normal distibuted membership function"""

    def __init__(self, **kwargs):  # , mean=0., std=1., beta=2, clip=None, ppf=None):
        FuzzyNumber.__init__(self, **kwargs)
        alpha0 = kwargs.get("alpha0")
        alpha1 = kwargs.get("alpha1")
        self.beta = kwargs.get("beta") or 2.
        self.clip = kwargs.get("alpha0")
        self.ppf = kwargs.get("ppf") or [.001, .999]
        self._loc = kwargs.get("mean") or np.array(alpha0).mean()
        self._scale = kwargs.get("std") or (alpha0[1] - alpha0[0]) / 6.
        # print("!", (alpha0[1]-alpha0[0])/6)
        self._distr = None
        self.discretize(alpha0=alpha0, alpha1=alpha1, alpha_levels=self.number_of_alpha_levels)

    # def __str__(self):
    #     return "tnorm(%s [%.3g,%.3g])" % (self.did, self.loc, self.std)
    #
    # __repr__ = __str__

    def _get_loc(self):
        return self._loc

    def _set_loc(self, value):
        self._loc = value

    mean = loc = property(fget=_get_loc, fset=_set_loc)

    def _get_scale(self):
        return self._scale

    def _set_scale(self, value):
        self._scale = value

    std = scale = property(fget=_get_scale, fset=_set_scale)

    @property
    def distr(self):
        def obj(s, args=[1., 4., 4., .999]):
            """args = [min, max, beta, ppf]"""
            loc = (args[1]+args[0])/2.
            beta = args[2]
            ppf = args[3]
            d = gennorm(loc=loc, scale=s, beta=beta)
            r = sum((d.ppf([1.-ppf, .5,  ppf]) - np.array([args[0], loc, args[1]]))**2)
            return r
        if self._distr is None:
            from scipy.optimize import minimize
            res = minimize(obj, [.1], method='Nelder-Mead', tol=1e-6, args=[self.clip[0],self.clip[1], self.beta, .999])
            # res = scipy.optimize.minimize_scalar(obj, bounds=[1e-10, 1], args=[1.,4., beta, .999], tol=1e-10)
            # print("res", res.x)
            self._distr = gennorm(loc=self.mean, scale=res.x, beta=self.beta)
        return self._distr

    def discretize(self, alpha0, alpha1, alpha_levels):
        assert isinstance(alpha0, collections.Sequence) and len(alpha0) == 2
        nn = 501
        pp = np.linspace(0., 1., nn)
        ppf = self.distr.ppf(pp)
        x = np.linspace(alpha0[0], alpha0[1], nn)
        pdf = self.distr.pdf(x)
        alphas = pdf / pdf.max()
        data = []
        for i in range(len(x) // 2):
            data.append([alphas[i], x[i], x[::-1][i]])
        data.append([alphas[i + 1], x[i + 1], x[::-1][i + 1]])
        self.df = pd.DataFrame(columns=["alpha", "low", "high"], data=data, dtype=np.float)
        self.convert_df(alpha_levels=alpha_levels)
        self.df.sort_values(['alpha'], ascending=[True], inplace=True)
