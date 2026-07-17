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
from scipy.stats import truncnorm, gennorm, lognorm, norm


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
        self.df = pd.DataFrame(columns=["alpha", "l", "r"], data=data, dtype=float)
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
        self.df = pd.DataFrame(columns=["alpha", "l", "r"], data=data, dtype=float)
        self.convert_df(alpha_levels=alpha_levels)
        self.df.sort_values(['alpha'], ascending=[True], inplace=True)


class TruncLogNorm(FuzzyNumber):
    r"""Log-normal distributed membership function, truncated to alpha0

    The membership is the log-normal pdf rescaled to peak at one,

    .. math::

        \mu(x) = \frac{f(x)}{f(x_{mode})}, \quad x_{mode} = c \cdot e^{-s^2}

    for a log-normal with median :math:`c` (``center``) and shape :math:`s`
    (``sigma``), truncated to the support :math:`[a, b]` given by ``alpha0``.
    Being right skewed, its core (:math:`\alpha=1`) sits at the mode, left of
    the median.

    ``center`` and ``sigma`` default to the values that place the ``ppf``
    quantiles exactly on the support:

    .. math::

        s = \frac{\ln(b/a)}{\Phi^{-1}(p_{hi}) - \Phi^{-1}(p_{lo})}, \quad
        c = a \cdot e^{-s\,\Phi^{-1}(p_{lo})}

    which for the default symmetric limits makes ``center`` the geometric mean
    :math:`\sqrt{ab}`.

    .. code-block:: python

        TruncLogNorm(alpha0=[1, 3], number_of_alpha_levels=17)

    :param alpha0: support [a, b], requires 0 < a < b
    :param mean: median of the distribution, alias ``center``
    :param std: shape parameter, the standard deviation of ln(x), alias ``sigma``
    :param ppf: quantile pair the support corresponds to, default [.001, .999]
    """

    def __init__(self, **kwargs):
        FuzzyNumber.__init__(self, **kwargs)
        alpha0 = kwargs.get("alpha0")
        if alpha0 is None or len(alpha0) != 2:
            raise ValueError("TruncLogNorm needs alpha0=[a, b], got {!r}".format(alpha0))

        a, b = float(alpha0[0]), float(alpha0[1])
        if not 0. < a < b:
            # a log-normal lives on (0, inf)
            raise ValueError("TruncLogNorm needs 0 < a < b, got alpha0=[{:g}, {:g}]".format(a, b))
        self.clip = [a, b]

        self.ppf_lim = kwargs.get("ppf") or [.001, .999]
        z_lo, z_hi = norm.ppf(self.ppf_lim[0]), norm.ppf(self.ppf_lim[1])
        if not z_lo < z_hi:
            raise ValueError("TruncLogNorm needs ppf=[p_lo, p_hi] with p_lo < p_hi, "
                             "got {!r}".format(self.ppf_lim))

        # place the ppf quantiles on a and b: solving
        #   a = c*exp(s*z_lo), b = c*exp(s*z_hi)  for s and c
        # `is None` rather than `or`: a given 0 must be rejected, not defaulted
        sigma, center = kwargs.get("std"), kwargs.get("mean")
        self._sigma = float(sigma) if sigma is not None else np.log(b / a) / (z_hi - z_lo)
        self._check_sigma(self._sigma)
        self._center = (float(center) if center is not None
                        else float(np.exp(np.log(a) - self._sigma * z_lo)))
        self._check_center(self._center)

        self._distr = None
        self.discretize(alpha0=self.clip, alpha1=kwargs.get("alpha1"),
                        alpha_levels=self.number_of_alpha_levels)

    @staticmethod
    def _check_center(value):
        if not value > 0.:
            raise ValueError("TruncLogNorm needs center > 0, got {:g}".format(value))

    @staticmethod
    def _check_sigma(value):
        if not value > 0.:
            raise ValueError("TruncLogNorm needs sigma > 0, got {:g}".format(value))

    def _refresh(self):
        """drop the cached distribution and rebuild the alpha levels"""
        self._distr = None
        self.discretize(alpha0=self.clip, alpha1=None,
                        alpha_levels=self.number_of_alpha_levels)

    def _get_center(self):
        """median of the distribution

        :rtype: float
        :return: median
        """
        return float(self._center)

    def _set_center(self, value):
        value = float(value)
        self._check_center(value)
        self._center = value
        self._refresh()

    mean = center = property(fget=_get_center, fset=_set_center)

    def _get_sigma(self):
        """shape parameter, the standard deviation of ln(x)

        :rtype: float
        :return: shape parameter
        """
        return float(self._sigma)

    def _set_sigma(self, value):
        value = float(value)
        self._check_sigma(value)
        self._sigma = value
        self._refresh()

    std = sigma = property(fget=_get_sigma, fset=_set_sigma)

    @property
    def mode(self):
        """peak of the membership function, left of the median

        :rtype: float
        :return: mode
        """
        return self.center * np.exp(-self.sigma ** 2)

    @property
    def distr(self):
        """log-normal distribution behind the membership function

        :return: distribution object
        """
        if self._distr is None:
            self._distr = lognorm(s=self.sigma, scale=self.center)
        return self._distr

    def discretize(self, alpha0, alpha1, alpha_levels):
        """discretize the membership function

        mu(x) = f(x)/f(mode) inverts in closed form: with u = ln(x/center),
        ln(mu) = -u^2/(2s^2) - u - s^2/2, so (u + s^2)^2 = -2 s^2 ln(mu) and

            x = mode * exp(+- s*sqrt(-2 ln(alpha)))

        :param alpha0: support [a, b]
        :param alpha1: unused, the core is the mode
        :param alpha_levels: number of alpha levels
        :return: None
        """
        a, b = float(alpha0[0]), float(alpha0[1])
        alphas = np.linspace(0., 1., int(alpha_levels))

        with np.errstate(divide="ignore"):
            # alpha=0 gives an infinite half width, which the clip below turns
            # into the support bounds -- that is what truncation means here
            half_width = self.sigma * np.sqrt(-2. * np.log(alphas))

        mode = self.mode
        l = np.clip(mode * np.exp(-half_width), a, b)
        r = np.clip(mode * np.exp(half_width), a, b)

        self.df = pd.DataFrame({"alpha": alphas, "l": l, "r": r}, dtype=float)
