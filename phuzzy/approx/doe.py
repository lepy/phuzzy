# -*- coding: utf-8 -*-
import logging
import pandas as pd
import numpy as np
import collections
import scipy.spatial.ckdtree
import phuzzy
import phuzzy.contrib.pydoe as pydoe
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV

class Job(object):
    """Run object

    """

    def __init__(self, **kwargs):
        """create Job

        :param kwargs:
        """

        self.name = kwargs.get("name", "Job N.N.")
        self.results = None

    def preprocessing(self, **kwargs):
        """pre-processing

        :param kwargs:
        :return:
        """

    def submit(self):
        """submit job

        :param kwargs:
        :return:
        """

    def postprocessing(self, **kwargs):
        """pos-processing

        :param kwargs:
        :return:
        """

    def __str__(self):
        return "(Job:'{o.name}')".format(o=self)

    __repr__ = __str__



class Project(object):
    """Project wrapper"""

    def __init__(self, **kwargs):
        """create Project

        :param kwargs:
        """

        self.name = kwargs.get("name", "Project N.N.")
        self._designvars = collections.OrderedDict()
        self._runs = collections.OrderedDict()

    @property
    def runs(self):
        """returns all design variables of doe

        :return: dict of designvars
        """
        return self._runs

    def add_run(self, run):
        """add run

        :param designvar: design variable
        :return: None
        """
        self._runs[run.name] = run

    @property
    def designvars(self):
        """returns all design variables of project

        :return: dict of designvars
        """
        return self._designvars

    def add_designvar(self, designvar):
        """add design variable to project

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

    def __str__(self):
        return "(Project:'{o.name}', dv={d}, runs={r})".format(o=self,
                                                              d=self._designvars.keys(),
                                                              r=self._runs.keys())

    __repr__ = __str__

class DOE(object):
    """Design of Experiment"""

    MESHGRID = "meshgrid"
    HALTON = "halton"
    LHS = "lhs"
    BOXBEHNKEN = "bb"
    CCDESIGN = "cc"

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

        :param method: 'meshgrid', 'lhs', 'halton', 'bb', 'cc'
        :return: samples
        """
        methods = {self.MESHGRID  : self.sample_meshgrid,
                   self.LHS       : self.sample_lhs,
                   self.HALTON    : self.sample_halton,
                   self.BOXBEHNKEN: self.sample_bbdesign,
                   self.CCDESIGN  : self.sample_ccdesign,
                   }
        methodname = kwargs.get("method", self.MESHGRID)
        method = methods.get(methodname)
        self.samples = method(**kwargs)
        return self.samples

    def sample_meshgrid(self, **kwargs):
        """return all combinations (np.meshgrid)

        :param kwargs:
        :return: doe
        """
        n = kwargs.get("n", 0)

        if len(self.designvars) == 1:
            designvar = self.designvars.values()[0]
            doe = pd.DataFrame.from_dict({designvar.name: designvar.samples})
            doe['alpha'] = 0.
        else:
            X = [designvar._disretize_range(n=0) for designvar in self.designvars.values()]
            Y = np.meshgrid(*X)

            d = {}
            for i, designvar in enumerate(self.designvars.values()):
                d[designvar.name] = Y[i].ravel()

            doe = pd.DataFrame.from_dict(d)
            # doe['alpha'] = 0
            for i, designvar in enumerate(self.designvars.values()):
                alpha = designvar.get_alpha_from_value(doe.iloc[:, i])
                doe["alpha_%d" % i] = alpha
                # print("alpha", designvar, alpha)
            alpha_cols = [x for x in doe.columns if x.startswith("alpha_")]
            doe["alpha"] = doe[alpha_cols].min(axis=1)

        return doe

    def sample_halton(self, **kwargs):
        sample = kwargs.get("n", 10)
        pass

    def sample_bbdesign(self, **kwargs):
        """Box-Behnken Sampling

        :param n: number of sample points
        :return: doe
        """
        dim = len(self.designvars)
        if dim < 3:
            logging.error("Box-Behnken requires at least 3 dimensions!")
            raise Exception("Box-Behnken requires at least 3 dimensions!")

        doe = pd.DataFrame(columns=[x.name for x in self.designvars.values()])
        doe['alpha'] = 0
        doe.loc[0] = np.zeros(len(self.designvars))
        doelhs = pd.DataFrame(pydoe.bbdesign(dim), columns=[x.name for x in self.designvars.values()])
        doe = pd.concat([doe, doelhs], ignore_index=True)
        for i, designvar in enumerate(self.designvars.values()):
            doe.iloc[:, i] = doe.iloc[:, i] * (designvar.max() - designvar.min()) + designvar.min()
        return doe

    def sample_ccdesign(self, **kwargs):
        """Central composite design

        :param n: number of sample points
        :return: doe
        """
        dim = len(self.designvars)
        dv0 = list(self.designvars.values())[0]
        # doe = pd.DataFrame([[x.ppf(.5) for x in self.designvars.values()]], columns=[x.name for x in self.designvars.values()])
        doe = pd.DataFrame(columns=[x.name for x in self.designvars.values()])
        doe_cc_raw = pd.DataFrame(pydoe.ccdesign(dim, face='ccf'), columns=[x.name for x in self.designvars.values()])
        doe_cc_raw['alpha'] = 0
        samples = []
        # for alphalevel in [0, -1]: # [0, -1, len(dv0.df)//2]:
        for alphalevel in [0, len(dv0.df) // 2, -1]:  # [0, -1, len(dv0.df)//2]:
            # for alphalevel in [0, len(dv0.df)//3, 2*len(dv0.df)//3, -1]: # [0, -1, len(dv0.df)//2]:
            doe_cc = doe_cc_raw.copy()
            for i, designvar in enumerate(self.designvars.values()):
                rmin = designvar.df.iloc[alphalevel].l
                rmax = designvar.df.iloc[alphalevel].r
                doe_cc.iloc[:, i] = (doe_cc.iloc[:, i] + 1.) / 2. * (rmax - rmin) + rmin

            alpha = designvar.df.iloc[alphalevel].alpha
            doe_cc.iloc[:, dim] = alpha

            samples.append(doe_cc)
        doe = pd.concat([doe] + samples, ignore_index=True)
        doe.drop_duplicates(inplace=True)
        doe.reset_index(inplace=True)
        return doe

    def sample_lhs(self, **kwargs):
        """Latin Hypercube Sampling

        :param n: number of sample points
        :return: doe
        """
        dim = len(self.designvars)
        n_samples = kwargs.get("n", 10)
        doe = pd.DataFrame(columns=[x.name for x in self.designvars.values()])
        doe.loc[0] = np.zeros(len(self.designvars))
        doelhs = pd.DataFrame(pydoe.lhs(dim, n_samples - 1), columns=[x.name for x in self.designvars.values()])
        doe = pd.concat([doe, doelhs], ignore_index=True)
        for i, designvar in enumerate(self.designvars.values()):
            doe.iloc[:, i] = doe.iloc[:, i] * (designvar.max() - designvar.min()) + designvar.min()
        for i, designvar in enumerate(self.designvars.values()):
            alpha = designvar.get_alpha_from_value(doe.iloc[:, i])
            doe["alpha_%d" % i] = alpha
            # print("alpha", designvar, alpha)
        alpha_cols = [x for x in doe.columns if x.startswith("alpha_")]
        doe["alpha"] = doe[alpha_cols].min(axis=1)
        # doe['alpha'] = 0
        print("doe", doe)
        print()
        return doe

    def eval(self, function, argnames, name=None):
        """evaluate function

        :param function:
        :param kwargs:
        :return:
        """
        # self.sample_doe(n=n_samples, method=method)
        eval_args = []
        for name in argnames:
            eval_args.append(self.samples[name])

        f_approx = function(*eval_args)
        self.df_res = pd.DataFrame({"alpha": self.samples.alpha,
                               "res"  : f_approx})

        fuzzynumber = phuzzy.approx.FuzzyNumber.from_results(self.df_res)
        fuzzynumber.df_res = self.df_res.copy()
        fuzzynumber.samples = self.samples.copy()
        if name is not None:
            fuzzynumber.name = name

        return fuzzynumber

    def fit_model(self, function, argnames, model=None, rel_train_size=.75):
        if model is None:
            model = "svm"

        if model == "svm":
            X = self.samples[argnames].values
            args = [self.samples[name] for name in argnames]
            y = function(*args)
            train_size = int(len(self.samples) * rel_train_size)
            self.model = GridSearchCV(SVR(kernel='rbf', gamma=.1), cv=5,
                       param_grid={"C": [1e0, 1e1, 1e2, 1e3, ],
                                   "gamma": np.logspace(-2, 2, num=5)})
            self.model.fit(X[:train_size], y[:train_size])

if __name__ == '__main__':
    p = Project(name="f")
    print(p)
