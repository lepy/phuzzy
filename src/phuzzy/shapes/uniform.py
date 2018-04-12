# -*- coding: utf-8 -*-

from phuzzy.shapes import FuzzyNumber
import numpy as np
import pandas as pd

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




class Gauss(FuzzyNumber):
    """Symmetric Gaussian function depends on two parameters $\sigma$ and $c$
    $f(x;\sigma, c) = e ^ \frac {−(x−c)**2}{2 \sigma}$"""


class BELL(FuzzyNumber):
    """Symmetric Gaussian function depends on two parameters $\sigma$ and $c$
    $f(x; a,b,c) = \frac {1}{1+\left| \frac{x-c}{a} \right|^2b}$"""


