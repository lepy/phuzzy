# -*- coding: utf-8 -*-
import logging
import pandas as pd
import numpy as np
import collections
import scipy.spatial.ckdtree


class DOE(object):
    """Design of Experiment"""

    MESHGRID = "meshgrid"
    HALTON = "halton"
    LHS = "lhs"

    def __init__(self, **kwargs):
        """DOE(kwargs)"""

        self.name = kwargs.get("name", "DOE N.N.")
        self._samples = pd.DataFrame()
        self._designvars = collections.OrderedDict()
        if "designvars" in kwargs:
            self.add_designvars(kwargs.get("designvars"))

    def __str__(self):
        return "(DOE:'{o.name}', dv={l}, {n} samples)".format(o=self,
                                                              l=self._designvars.keys(),
                                                              n=len(self.samples))

    __repr__ = __str__

    def _get_samples(self):
        """returns all design points

        :return: sample dataframe
        """
        return self._samples

    def _set_samples(self, value):
        self._samples = value

    samples = property(fget=_get_samples, fset=_set_samples, doc="dataframe of samples")

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
        self._designvars[designvar.name] = designvar

    def add_designvars(self, designvars):
        """add design variables to doe

        :param designvars: list of design variables
        :return: None
        """
        for designvar in designvars:
            self._designvars[designvar.name] = designvar

    def sample_doe(self, **kwargs):
        """generates samples for doe

        :param method: 'meshgrid', 'lhs', 'halton'
        :return: samples
        """
        methods = {self.MESHGRID: self.sample_meshgrid,
                   self.LHS     : self.sample_lhs,
                   self.HALTON  : self.sample_halton}
        methodname = kwargs.get("method", self.MESHGRID)
        method = methods.get(methodname)
        self.samples = method(**kwargs)
        return self.samples

    def sample_meshgrid(self, **kwargs):
        """return all combinations (np.meshgrid)

        :param kwargs:
        :return: doe
        """

        if len(self.designvars) == 1:
            designvar = self.designvars.values()[0]
            doe = pd.DataFrame.from_dict({designvar.name: designvar.samples})
        else:
            X = [designvar._disretize_range(n=0) for designvar in self.designvars.values()]
            Y = np.meshgrid(*X)

            d = {}
            for i, designvar in enumerate(self.designvars.values()):
                d[designvar.name] = Y[i].ravel()

            doe = pd.DataFrame.from_dict(d)
        return doe

    def sample_halton(self, **kwargs):
        sample = kwargs.get("n", 10)
        pass

    def sample_lhs(self, **kwargs):
        """Latin Hypercube Sampling

        :param n: number of sample points
        :return: doe
        """
        try:
            import pyDOE
        except ImportError:
            logging.error("Please install pyDOE (pip install pyDOE) to use LHS!")
            raise
        dim = len(self.designvars)
        n_samples = kwargs.get("n", 10)
        doe = pd.DataFrame(columns=[x.name for x in self.designvars.values()])
        doe.loc[0] = np.zeros(len(self.designvars))
        doelhs = pd.DataFrame(pyDOE.lhs(dim, n_samples - 1), columns=[x.name for x in self.designvars.values()])
        doe = pd.concat([doe, doelhs], ignore_index=True)
        for i, designvar in enumerate(self.designvars.values()):
            doe.iloc[:, i] = doe.iloc[:, i] * (designvar.max() - designvar.min()) + designvar.min()
        return doe
