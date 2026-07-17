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

    def __init__(self, **kwargs):
        """create a TruncNorm object

        .. code-block:: python

            TruncNorm(alpha0=[1, 3], number_of_alpha_levels=17)

        :param alpha0: support [a, b]
        :param mean: centre of the distribution, defaults to the midpoint
        :param std: standard deviation, defaults to spanning the ppf limits
        :param ppf: quantile pair the support corresponds to, default [.001, .999]
        """
        FuzzyNumber.__init__(self, **kwargs)
        a, b = self._validate_alpha0(kwargs.get("alpha0"))
        self.clip = [a, b]
        self.ppf_lim = kwargs.get("ppf") or [.001, .999]
        # `is None` rather than `or`: a given 0 is a valid mean and an invalid
        # std, so both must be honoured or rejected, never silently defaulted
        loc, scale = kwargs.get("mean"), kwargs.get("std")
        self._loc = float(loc) if loc is not None else (a + b) / 2.
        self._scale = float(scale) if scale is not None else self._scale_from_support(a, b)
        self._check_scale(self._scale)
        self._distr = None
        self.discretize(alpha0=self.clip, alpha1=kwargs.get("alpha1"),
                        alpha_levels=self.number_of_alpha_levels)

    def _scale_from_support(self, a, b):
        """standard deviation that spans the support between the ppf limits

        norm is a location-scale family, so ppf(q) = loc + scale*Phi^-1(q).
        Requiring ppf(p_lo) = a and ppf(p_hi) = b gives scale directly. For the
        default limits Phi^-1(.999) - Phi^-1(.001) is 6.18, so this is the
        familiar "support spans +-3 sigma" rule, only exact.

        :param a: lower bound of the support
        :param b: upper bound of the support
        :rtype: float
        :return: standard deviation
        """
        z_lo, z_hi = norm.ppf(self.ppf_lim[0]), norm.ppf(self.ppf_lim[1])
        if not z_lo < z_hi:
            raise ValueError("TruncNorm needs ppf=[p_lo, p_hi] with p_lo < p_hi, "
                             "got {!r}".format(self.ppf_lim))
        return (b - a) / (z_hi - z_lo)

    @staticmethod
    def _check_scale(value):
        if not value > 0.:
            raise ValueError("TruncNorm needs std > 0, got {:g}".format(value))

    def _refresh(self):
        """drop the cached distribution and rebuild the alpha levels"""
        self._distr = None
        self.discretize(alpha0=self.clip, alpha1=None,
                        alpha_levels=self.number_of_alpha_levels)

    def _get_loc(self):
        """mean value

        :rtype: float
        :return: mean value aka location
        """
        return self._loc

    def _set_loc(self, value):
        self._loc = float(value)
        self._refresh()

    mean = loc = property(fget=_get_loc, fset=_set_loc)

    def _get_scale(self):
        """standard deviation

        :rtype: float
        :return: standard deviation
        """
        return self._scale

    def _set_scale(self, value):
        value = float(value)
        self._check_scale(value)
        self._scale = value
        self._refresh()

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
        """discretize the membership function

        mu(x) = f(x)/f(loc) of a normal is exp(-(x-loc)^2 / (2 sigma^2)), which
        inverts in closed form:

            x = loc +- sigma*sqrt(-2 ln(alpha))

        :param alpha0: support [a, b]
        :param alpha1: unused, the core is loc
        :param alpha_levels: number of alpha levels
        :return: None
        """
        a, b = float(alpha0[0]), float(alpha0[1])
        alphas = np.linspace(0., 1., int(alpha_levels))

        with np.errstate(divide="ignore"):
            # alpha=0 gives an infinite half width, which the clip below turns
            # into the support bounds -- that is what truncation means here
            half_width = self.std * np.sqrt(-2. * np.log(alphas))

        l = np.clip(self.loc - half_width, a, b)
        r = np.clip(self.loc + half_width, a, b)
        self.df = pd.DataFrame({"alpha": alphas, "l": l, "r": r}, dtype=float)


class TruncGenNorm(FuzzyNumber):
    """Truncated generalized normal distibuted membership function"""

    def __init__(self, **kwargs):  # , mean=0., std=1., beta=2, clip=None, ppf=None):
        """create a TruncNorm object

        :param kwargs:

        .. code-block:: python

            TruncGenNorm(alpha0=[1, 3], alpha1=None, number_of_alpha_levels=17, beta=3)

        """
        FuzzyNumber.__init__(self, **kwargs)
        a, b = self._validate_alpha0(kwargs.get("alpha0"))
        self.clip = [a, b]
        # `is None` rather than `or`: a given 0 is a valid mean and an invalid
        # beta or std, so all three must be honoured or rejected
        beta, loc, scale = kwargs.get("beta"), kwargs.get("mean"), kwargs.get("std")
        self._beta = float(beta) if beta is not None else 2.
        self._check_beta(self._beta)
        self.ppf_lim = kwargs.get("ppf") or [.001, .999]
        self._loc = float(loc) if loc is not None else (a + b) / 2.
        self._distr = None
        # scale defaults to the value that puts the ppf limits on the support
        self._scale = float(scale) if scale is not None else self._scale_from_support(a, b)
        self._check_scale(self._scale)
        self.discretize(alpha0=self.clip, alpha1=kwargs.get("alpha1"),
                        alpha_levels=self.number_of_alpha_levels)

    @staticmethod
    def _check_beta(value):
        if not value > 0.:
            raise ValueError("TruncGenNorm needs beta > 0, got {:g}".format(value))

    @staticmethod
    def _check_scale(value):
        if not value > 0.:
            raise ValueError("TruncGenNorm needs std > 0, got {:g}".format(value))

    def _scale_from_support(self, a, b):
        """scale that spans the support between the ppf limits

        gennorm is a location-scale family, so ppf(q) = loc + scale*z(q, beta)
        with z the standardized quantile. Requiring ppf(p_lo) = a and
        ppf(p_hi) = b then gives scale directly, no optimizer needed.

        :param a: lower bound of the support
        :param b: upper bound of the support
        :rtype: float
        :return: scale
        """
        z_lo = gennorm.ppf(self.ppf_lim[0], beta=self.beta)
        z_hi = gennorm.ppf(self.ppf_lim[1], beta=self.beta)
        if not z_lo < z_hi:
            raise ValueError("TruncGenNorm needs ppf=[p_lo, p_hi] with p_lo < p_hi, "
                             "got {!r}".format(self.ppf_lim))
        return (b - a) / (z_hi - z_lo)

    def _refresh(self):
        """drop the cached distribution and rebuild the alpha levels"""
        self._distr = None
        self.discretize(alpha0=self.clip, alpha1=None,
                        alpha_levels=self.number_of_alpha_levels)

    def _get_loc(self):
        """mean value

        :rtype: float
        :return: mean value aka location
        """
        return self._loc

    def _set_loc(self, value):
        self._loc = float(value)
        self._refresh()

    mean = loc = property(fget=_get_loc, fset=_set_loc)

    def _get_scale(self):
        """standard deviation

        :rtype: float
        :return: standard deviation
        """
        return self._scale

    def _set_scale(self, value):
        value = float(value)
        self._check_scale(value)
        self._scale = value
        self._refresh()

    std = scale = property(fget=_get_scale, fset=_set_scale)

    def _get_beta(self):
        """shape parameter

        :rtype: float
        :return: beta
        """
        return self._beta

    def _set_beta(self, value):
        value = float(value)
        self._check_beta(value)
        self._beta = value
        self._refresh()

    beta = property(fget=_get_beta, fset=_set_beta)

    @property
    def distr(self):
        """generalized normal distribution behind the membership function

        :return: distribution object
        """
        if self._distr is None:
            self._distr = gennorm(loc=self.loc, scale=self.scale, beta=self.beta)
        return self._distr

    def discretize(self, alpha0, alpha1, alpha_levels):
        """discretize the membership function

        mu(x) = f(x)/f(loc) of a generalized normal is
        exp(-(|x-loc|/scale)^beta), which inverts in closed form:

            x = loc +- scale*(-ln(alpha))^(1/beta)

        :param alpha0: support [a, b]
        :param alpha1: unused, the core is loc
        :param alpha_levels: number of alpha levels
        :return: None
        """
        a, b = float(alpha0[0]), float(alpha0[1])
        alphas = np.linspace(0., 1., int(alpha_levels))

        with np.errstate(divide="ignore"):
            # alpha=0 gives an infinite half width, which the clip below turns
            # into the support bounds -- that is what truncation means here
            half_width = self.scale * (-np.log(alphas)) ** (1. / self.beta)

        l = np.clip(self.loc - half_width, a, b)
        r = np.clip(self.loc + half_width, a, b)
        self.df = pd.DataFrame({"alpha": alphas, "l": l, "r": r}, dtype=float)


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
