# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

import phuzzy.contrib.tgo


class FuzzyAnalysis(object):
    """Fuzzy Analysis"""

    def __init__(self, **kwargs):
        """DOE(kwargs)"""

        self.name = kwargs.get("name", "FuzzyAnalysis N.N.")
        self.function = kwargs.get("function")
        self._designvars = []
        self.model = None

        if "designvars" in kwargs:
            self.add_designvars(kwargs.get("designvars"))

    def __str__(self):
        return "(FuzzyAnalysis:'{o.name}', dv={d}".format(o=self,
                                                          d=self._designvars,
                                                          )

    __repr__ = __str__

    @property
    def designvars(self):
        """returns all design variables of doe

        :return: dict of designvars
        """
        return self._designvars

    def add_designvar(self, designvar):
        """add design variable to doe

        :param designvar: design variable
        :return: None
        """
        self._designvars.append(designvar)

    def add_designvars(self, designvars):
        """add design variables to doe

        :param designvars: list of design variables
        :return: None
        """
        self._designvars.extend(designvars)

    def eval(self, ntgo=1000):
        """evaluate Function

        :return: FuzzyNumber
        """

        bounds = [(x.min(), x.max()) for x in self.designvars]

        try:
            res = phuzzy.contrib.tgo.tgo(self.function, bounds, args=(), g_cons=None, g_args=(), n=ntgo)
            print(res)
            print(res.xl)
            alphas = pd.DataFrame(
                np.vstack([x.get_alpha_from_value(res.xl[:, i]) for i, x in enumerate(self.designvars)]).T)
            alphas["alpha"] = alphas.min(axis=1)
            alphas["fun"] = res.funl
            print(alphas)
        except IndexError:
            alphas = None
        # TODO: find optimal max values

        z = self.function(self.designvars)
        print("!z0", z.df)
        return z

    def lcefa(self):
        """Local cost effectivness fuzzy analysis

        :return:
        """

        res = []
        s_sum = 0
        for i in range(len(self.designvars)):
            fvars = []
            for j, x in enumerate(self.designvars):
                if i == j:
                    fvars.append(x)
                else:
                    y = phuzzy.Uniform(alpha0=[x.min(), x.max()])
                    fvars.append(y)

            z = self.function(fvars)
            z.name = self.designvars[i].name
            dflr = z.df[["l", "r"]]
            E = 1. - (dflr.max(axis=1) - dflr.min(axis=1)) / (dflr.iloc[0].max() - dflr.iloc[0].min())
            z.E = E
            z.s = E.sum()
            s_sum += z.s

            res.append(z)

        sensibilities = []
        for i, z in enumerate(res):
            z.sk = z.s / s_sum
            self.designvars[i].sk = z.sk
            print("%s.sk=%.2g" % (z.name, z.sk))
            sensibilities.append([z, z.name, z.sk])
        return pd.DataFrame(sensibilities, columns=["x", "name", "sk"])

