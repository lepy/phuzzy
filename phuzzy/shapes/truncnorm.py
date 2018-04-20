# -*- coding: utf-8 -*-

"""Normal distibuted membership function

.. code-block:: python

    TruncNorm(alpha0=[1, 3], alpha1=None, number_of_alpha_levels=15)

.. figure:: TruncNorm.png
    :scale: 90 %
    :alt: TruncNorm fuzzy number

    TruncNorm fuzzy number

.. code-block:: python

    TruncGenNorm(alpha0=[1, 4], alpha1=None, number_of_alpha_levels=15, beta=5)

.. figure:: TruncGenNorm.png
    :scale: 90 %
    :alt: TruncGenNorm fuzzy number

    TruncGenNorm fuzzy number


"""

from phuzzy.shapes import FuzzyNumber
import numpy as np
import pandas as pd
from scipy.stats import truncnorm, gennorm

class TruncNorm(FuzzyNumber):
    """Normal distibuted membership function

    """

    def __init__(self, **kwargs):  # , mean=0., std=1., clip=None, ppf=None):
        """create a TruncNorm object

        :param kwargs:

        .. code-block:: python

            TruncNorm(alpha0=[1, 3], alpha1=None, number_of_alpha_levels=17)

        """
        FuzzyNumber.__init__(self, **kwargs)
        alpha0 = kwargs.get("alpha0")
        alpha1 = kwargs.get("alpha1")
        self.clip = kwargs.get("alpha0", [0, np.inf])
        self.ppf_lim = kwargs.get("ppf", [.001, .999])
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
        """mean value

        :rtype: float
        :return: mean value aka location
        """
        return self._loc

    def _set_loc(self, value):
        self._loc = value

    mean = loc = property(fget=_get_loc, fset=_set_loc)

    def _get_scale(self):
        """standard deviation

        :rtype: float
        :return: standard deviation
        """
        return self._scale

    def _set_scale(self, value):
        self._scale = value

    std = scale = property(fget=_get_scale, fset=_set_scale)

    @property
    def distr(self):
        """calculate truncated normal distribution

        :return: distribution object
        """

        if self._distr is None:
            a, b = (self.clip[0] - self.loc) / self.std, (self.clip[1] - self.loc) / self.std
            self._distr = truncnorm(a=a, b=b, loc=self.mean, scale=self.std)
        #             print "set_distr", self._distr, self.mean, self.std
        return self._distr

    def discretize(self, alpha0, alpha1, alpha_levels):
        # print("alpha0", alpha0)
        # assert isinstance(alpha0, collections.Sequence) and len(alpha0) == 2
        # assert isinstance(alpha1, collections.Sequence) and len(alpha1) > 0
        nn = 501
        # pp = np.linspace(0, 1, nn)
        # ppf = self.distr.ppf(pp)
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
        self.df = pd.DataFrame(columns=["alpha", "l", "r"], data=data, dtype=np.float)
        self.convert_df(alpha_levels)
        self.df.sort_values(['alpha'], ascending=[True], inplace=True)
        self.convert_df(alpha_levels=alpha_levels)


class TruncGenNorm(FuzzyNumber):
    """Truncated generalized normal distibuted membership function"""

    def __init__(self, **kwargs):  # , mean=0., std=1., beta=2, clip=None, ppf=None):
        """create a TruncNorm object

        :param kwargs:

        .. code-block:: python

            TruncGenNorm(alpha0=[1, 3], alpha1=None, number_of_alpha_levels=17, beta=3)

        """
        FuzzyNumber.__init__(self, **kwargs)
        alpha0 = kwargs.get("alpha0")
        alpha1 = kwargs.get("alpha1")
        self.beta = kwargs.get("beta") or 2.
        self.clip = kwargs.get("alpha0")
        self.ppf_lim = kwargs.get("ppf") or [.001, .999]
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
        def obj(s, args=None):
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
        # assert isinstance(alpha0, collections.Sequence) and len(alpha0) == 2
        nn = 501
        # pp = np.linspace(0., 1., nn)
        # ppf = self.distr.ppf(pp)
        x = np.linspace(alpha0[0], alpha0[1], nn)
        pdf = self.distr.pdf(x)
        alphas = pdf / pdf.max()
        data = []
        for i in range(len(x) // 2):
            data.append([alphas[i], x[i], x[::-1][i]])
        data.append([alphas[i + 1], x[i + 1], x[::-1][i + 1]])
        self.df = pd.DataFrame(columns=["alpha", "l", "r"], data=data, dtype=np.float)
        self.convert_df(alpha_levels=alpha_levels)
        self.df.sort_values(['alpha'], ascending=[True], inplace=True)

