# -*- coding: utf-8 -*-

from phuzzy.shapes import FuzzyNumber
import numpy as np
import pandas as pd

class Superellipse(FuzzyNumber):
    """superelliptic fuzzy number

    ..code_block: python
        :linenos:

        Superellipse(alpha0=alpha0, alpha1=None, m=2, n=None, number_of_alpha_levels=17)

    """

    def __init__(self, **kwargs):
        FuzzyNumber.__init__(self, **kwargs)
        alpha0 = kwargs.get("alpha0")
        self.m = kwargs.get("m") or 2.
        self.n = kwargs.get("n") or self.m
        self.discretize(alpha0=alpha0, alpha1=None, alpha_levels=self.number_of_alpha_levels)

    def shape(self, x):
        """shape function

        :param x: x values
        :type x: array or float
        :return: y values
        """

        a = self._a
        b = self._b
        condlist = [x <= self._a, x < self._b, x >= self._b]

        r = (b - a) / 2.
        m = (a + b) / 2.
        choicelist = [0.,
                      (1. - (abs(x-m)/r)**self.m) ** (1./self.n),
                      0.]
        return np.select(condlist, choicelist)

    def to_str(self):
        """serialize fuzzy number to string

        :rtype: str
        :return: fuzzy string
        """
        return u"Superellipse[{:.4g},{:.4g}]".format(self.alpha0.l, self.alpha0.r)

    @classmethod
    def from_str(cls, s):
        """instantiate a fuzzy number from string

        :param s:
        :rtype: phuzzy.FuzzyNumber
        :return: fuzzy number
        """
        raise NotImplementedError

    def discretize(self, alpha0, alpha1, alpha_levels):
        """discretize shape function

        :param alpha0: range at alpha=0
        :param alpha1: range at alpha=1
        :param alpha_levels: number of alpha levels
        :return: None
        """
        self._a = alpha0[0]
        self._b = alpha0[1]
        # assert isinstance(alpha0, collections.Sequence) and len(alpha0) == 2
        nn = 501
        x = np.linspace(alpha0[0], alpha0[1], nn)
        shape = self.shape(x)
        alphas = shape / shape.max()
        data = []
        for i in range(len(x) // 2):
            data.append([alphas[i], x[i], x[::-1][i]])
        data.append([alphas[i + 1], x[i + 1], x[::-1][i + 1]])
        self.df = pd.DataFrame(columns=["alpha", "l", "r"], data=data, dtype=np.float)
        self.convert_df(alpha_levels=alpha_levels)
        self.df.sort_values(['alpha'], ascending=[True], inplace=True)
