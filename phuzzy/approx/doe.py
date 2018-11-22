# -*- coding: utf-8 -*-
import collections
import logging

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR

import phuzzy
import phuzzy.contrib.pydoe as pydoe


class Expression(object):
    """Approximate an expression of fuzzy numbers

    """

    def __init__(self, **kwargs):
        """DOE(kwargs)"""

        self.name = kwargs.get("name", "DOE N.N.")
        self.function = kwargs.get("function")
        self._designvars = collections.OrderedDict()
        self.doe_training = None
        self.doe_prediction = None
        self.model = None

        if "designvars" in kwargs:
            self.add_designvars(kwargs.get("designvars"))

    def __str__(self):
        return "(Expression:'{o.name}', dv={d}".format(o=self,
                                                d=self._designvars.keys(),
                                                )

    __repr__ = __str__

    def generate_training_doe(self, name="train", n=10, method="lhs"):
        """generate train

        :param name:
        :param n:
        :param method:
        :return: doe
        """
        self.doe_training = DOE(designvars=self.designvars.values(), name="doe_training")
        self.doe_training.sample_doe(n=n, method=method)

    def generate_prediction_doe(self, name="train", n=10, method="lhs"):
        """generate prediction doe

        :param name:
        :param n:
        :param method:
        :return: doe
        """
        self.doe_prediction = DOE(designvars=self.designvars.values(), name="doe_prediction")
        self.doe_prediction.sample_doe(n=n, method=method)

    def predict(self, name=None):
        """predict function results"""
        X = self.doe_prediction.samples[list(self.designvars.keys())]
        y = self.model.predict(X)
        self.results_prediction = pd.DataFrame({"res": y, "alpha": self.doe_prediction.samples.alpha})
        print(1, self.results_training.head())
        print(2, self.results_prediction.head())
        df_res = pd.concat([self.results_training, self.results_prediction[["res", "alpha"]]], sort=False)

        if name is None:
            name = "z"
        z = phuzzy.FuzzyNumber.from_results(df_res, name=name)
        z.convert_df(alpha_levels=11)
        return z

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

    def update_training_results(self, df):
        """update training results for each data set of DoE sampling


        :param df:
        :return: None
        """
        # TODO: implement real update ()
        self.results_training = df

    def eval(self):
        """evaluate (expensive) function

        :param function:
        :param kwargs:
        :return:
        """
        eval_args = []
        for dv in self.designvars.values():
            eval_args.append(self.doe_training.samples[dv.name])

        # calculate results for each row/data set in sampling
        # TODO: add scheduler for long running (persistant) jobs
        f_approx = self.function(*eval_args)
        df_res = pd.DataFrame({"alpha": self.doe_training.samples.alpha,
                               "res": f_approx})

        self.update_training_results(df_res)

    def fit_model(self, model=None):
        """

        :return:
        """

        X = self.doe_training.samples[list(self.designvars.keys())].values
        y = self.results_training.res.values

        if model is None:
            model = "svr"

        models = {"svr": self._get_svr,
                  "knn": self._get_knn}
        get_model = models.get(model, "svr")
        get_model(X, y)

    def _get_svr(self, X, y):
        svr = GridSearchCV(SVR(kernel='rbf', gamma=.1), cv=5,
                           param_grid={"C": [1e0, 1e1, 1e2, 1e3, ],
                                       "gamma": np.logspace(-2, 2, num=5)})

        train_size = int(len(X) * .75)
        logging.debug("train_size %s" % train_size)
        svr.fit(X[:train_size], y[:train_size])
        self.model = svr

    def _get_knn(self, X, y):

        n_neighbors = 5
        weights = "distance"

        knn = KNeighborsRegressor(n_neighbors, weights=weights).fit(X, y)
        self.model = knn

    def get_fuzzynumber_from_results(self, name=None):
        """

        :return: FuzzyNumber
        """

        fuzzynumber = phuzzy.approx.FuzzyNumber.from_results(self.results_training)
        fuzzynumber.df_res = self.results_training.copy()
        fuzzynumber.samples = self.doe_training.samples.copy()
        if name is not None:
            fuzzynumber.name = name
        return fuzzynumber


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

        :param method: 'meshgrid', 'lhs', 'bb', 'cc'
        :return: samples
        """
        methods = {self.MESHGRID: self.sample_meshgrid,
                   self.LHS: self.sample_lhs,
                   # self.HALTON    : self.sample_halton,
                   self.BOXBEHNKEN: self.sample_bbdesign,
                   self.CCDESIGN: self.sample_ccdesign,
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

    # def sample_halton(self, **kwargs):
    #     sample = kwargs.get("n", 10)
    #     pass

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
        # for alphalevel in [0, len(dv0.df)-1]:  # [0, -1, len(dv0.df)//2]:
        for alphalevel in [0, len(dv0.df) // 2, len(dv0.df) - 1]:  # [0, -1, len(dv0.df)//2]:
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
        print("sample doe", doe.shape)
        return doe

    def gen_lhs_samples(self, **kwargs):
        """Latin Hypercube Sampling

        :param n: number of sample points
        :return: doe
        """
        dim = len(self.designvars)
        n_samples = kwargs.get("n", 10)
        doe = pd.DataFrame(columns=[x.name for x in self.designvars.values()])
        doe.loc[0] = np.zeros(len(self.designvars))
        doelhs = pd.DataFrame(pydoe.lhs(dim, n_samples), columns=[x.name for x in self.designvars.values()])
        return doelhs
