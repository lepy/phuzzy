__title__     = "phuzzy"
__author__    = "lepy"
__email__     = "lepy@mailbox.org"
__description__ = """Fuzzy stuff"""
__long_description__ = """
fuzzy number tools
"""
__url__       = 'https://github.com/lepy/phuzzy'
__copyright__ = "Copyright (C) 2017-"
__version__   = "0.1.0"
__status__    = "3 - Alpha"
__credits__   = [""]
__license__   = """MIT"""

import logging
import os
import collections
import numpy as np
import pandas as pd
from scipy.stats import truncnorm, norm

logger = logging.getLogger("phuzzy")

class Fuzzy_Number(object):
    """convex fuzzy number"""
    def __init__(self, **kwargs):
        self.name = kwargs.get("name", "Fuzzy")
        self._df = pd.DataFrame(columns=["alpha", "min", "max"])
        self._alpha_levels = kwargs.get("number_of_alpha_levels", 10)
        self.df = kwargs.get("df")

    def _get_alpha_levels(self):
        return self._alpha_levels
    def _set_alpha_levels(self, value):
        self._alpha_levels = int(value)
    alpha_levels = property(fget=_get_alpha_levels, fset=_set_alpha_levels, doc="number of alpha levels")

    def _get_df(self):
        return self._df
    def _set_df(self, value):
        self._df = value
    df = property(fget=_get_df, fset=_set_df, doc="number of alpha levels")

    def convert_df(self, alpha_levels=None):
        df = self.df.copy()
        if alpha_levels is not None:
            self.alpha_levels = alpha_levels
        df.sort_values(['alpha'], ascending=[True], inplace=True)
        # print("!",df)
        xs_l = df["min"].values
        alphas_l = df["alpha"].values
        xs_r = df["max"].values[::-1]
        alphas_r = df["alpha"].values[::-1]

        alphas_new = np.linspace(0., 1., self.alpha_levels)
        xs_l_new = np.interp(alphas_new, alphas_l, xs_l)
        xs_r_new = np.interp(alphas_new, alphas_r[::-1], xs_r[::-1])
        #
        # print((xs_l, alphas_l, xs_l_new))
        # print((xs_r, alphas_r, xs_r_new))
        #
        # print((np.vstack((alphas_new, xs_l_new, xs_r_new[::-1]))))
        data = np.vstack((alphas_new, xs_l_new, xs_r_new)).T
        self.df = pd.DataFrame(columns=["alpha", "min", "max"], data=data, dtype=np.float)

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
        A = ((self.alpha0["max"] - self.alpha0["min"]) - (self.df["max"].values - self.df["min"].values)).sum()
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

    def __str__(self):
        return "({0.__class__.__name__}({0.name}))".format(self)

    def __repr__(self):
        return "(%s)" % self.name

class Triangle(Fuzzy_Number):
    """triange fuzzy number"""

    def __init__(self, **kwargs):
        Fuzzy_Number.__init__(self, **kwargs)
        alpha0 = kwargs.get("alpha0")
        alpha1 = kwargs.get("alpha1")
        self.discretize(alpha0=alpha0, alpha1=alpha1, alpha_levels=self.alpha_levels)

    def discretize(self, alpha0, alpha1, alpha_levels):
        assert isinstance(alpha0, collections.Sequence) and len(alpha0) == 2
        assert isinstance(alpha1, collections.Sequence) and len(alpha1) > 0
        self.df = pd.DataFrame(columns=["alpha", "min", "max"], data=[[0., alpha0[0], alpha0[1]], [1., alpha1[0], alpha1[0]]], dtype=np.float)
        self.df.sort_values(['alpha'], ascending=[True], inplace=True)

class Trapezoid(Fuzzy_Number):
    """triange fuzzy number"""

    def __init__(self, **kwargs):
        Fuzzy_Number.__init__(self, **kwargs)
        alpha0 = kwargs.get("alpha0")
        alpha1 = kwargs.get("alpha1")
        self.discretize(alpha0=alpha0, alpha1=alpha1, alpha_levels=self.alpha_levels)

    def discretize(self, alpha0, alpha1, alpha_levels):
        assert isinstance(alpha0, collections.Sequence) and len(alpha0) == 2
        assert isinstance(alpha1, collections.Sequence) and len(alpha1) == 2
        self.df = pd.DataFrame(columns=["alpha", "min", "max"], data=[[0., alpha0[0], alpha0[1]], [1., alpha1[0], alpha1[1]]], dtype=np.float)
        self.df.sort_values(['alpha'], ascending=[True], inplace=True)

class TruncNorm(Fuzzy_Number):
    """abgeschnittene Normalverteilung"""
    def __init__(self, **kwargs):#, mean=0., std=1., clip=None, ppf=None):
        Fuzzy_Number.__init__(self, **kwargs)
        alpha0 = kwargs.get("alpha0")
        alpha1 = kwargs.get("alpha1")

        self.color = "k"

        self.clip = kwargs.get("alpha0") or [0, np.inf]
        self.ppf = kwargs.get("ppf") or [.001, .999]
        self.xs = []
        self.ys = []
        self._loc = kwargs.get("mean") or 0.
        self._scale = kwargs.get("std") or 1.
        self._distr = None
        self.discretize(alpha0=self.clip, alpha1=alpha1, alpha_levels=self.alpha_levels)
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
        assert isinstance(alpha0, collections.Sequence) and len(alpha0) == 2
        assert isinstance(alpha1, collections.Sequence) and len(alpha1) > 0
        self.df = pd.DataFrame(columns=["alpha", "min", "max"], data=[[0., alpha0[0], alpha0[1]], [1., alpha1[0], alpha1[0]]], dtype=np.float)
        self.df.sort_values(['alpha'], ascending=[True], inplace=True)