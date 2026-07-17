# -*- coding: utf-8 -*-
"""Construction contract shared by the shapes: support, core and parameters.

A fuzzy number needs a <= b for its support and its core inside that support.
Anything else is rejected rather than turned into a df whose levels are not
nested. TruncLogNorm has its own stricter rules, see test_trunclognorm.py.
"""

import numpy as np
import pytest
from scipy.stats import gennorm

import phuzzy


#: every shape, with the extra kwargs it needs beyond alpha0
SHAPES = [
    ("Triangle", {"alpha1": [2]}),
    ("Trapezoid", {"alpha1": [2, 3]}),
    ("Uniform", {}),
    ("Superellipse", {}),
    ("TruncNorm", {}),
    ("TruncGenNorm", {}),
]


def _make(clsname, alpha0, **kwargs):
    return getattr(phuzzy, clsname)(alpha0=alpha0, number_of_alpha_levels=5, **kwargs)


# -- support --------------------------------------------------------------

@pytest.mark.parametrize("clsname, kwargs", SHAPES)
def test_reversed_support_is_rejected(clsname, kwargs):
    """alpha0=[5, 1] used to yield NaN or l > r, depending on the shape."""
    with pytest.raises(ValueError, match=r"alpha0=\[a, b\] with a <= b"):
        _make(clsname, [5, 1], **kwargs)


@pytest.mark.parametrize("clsname, kwargs", SHAPES)
def test_missing_support_is_rejected(clsname, kwargs):
    with pytest.raises(ValueError, match=r"needs alpha0"):
        getattr(phuzzy, clsname)(number_of_alpha_levels=5, **kwargs)


@pytest.mark.parametrize("clsname, kwargs", SHAPES)
def test_one_sided_support_is_rejected(clsname, kwargs):
    with pytest.raises(ValueError, match=r"needs alpha0"):
        _make(clsname, [1], **kwargs)


@pytest.mark.parametrize("clsname, kwargs", SHAPES)
def test_the_error_names_the_shape(clsname, kwargs):
    with pytest.raises(ValueError, match=clsname):
        _make(clsname, [5, 1], **kwargs)


def test_a_degenerate_support_is_allowed():
    """a == b is a crisp number, not an error."""
    assert phuzzy.Triangle(alpha0=[2, 2], alpha1=[2], number_of_alpha_levels=3).min() == 2.
    assert phuzzy.Uniform(alpha0=[2, 2], number_of_alpha_levels=3).max() == 2.


def test_constant_needs_a_value():
    with pytest.raises(ValueError, match=r"Constant needs alpha0"):
        phuzzy.Constant(number_of_alpha_levels=3)


# -- core -----------------------------------------------------------------

def test_triangle_core_outside_the_support_is_rejected():
    with pytest.raises(ValueError, match=r"core alpha1=\[9, 9\] inside the support"):
        phuzzy.Triangle(alpha0=[1, 4], alpha1=[9], number_of_alpha_levels=3)


def test_trapezoid_core_wider_than_the_support_is_rejected():
    with pytest.raises(ValueError, match=r"core alpha1=\[0, 9\] inside the support"):
        phuzzy.Trapezoid(alpha0=[1, 4], alpha1=[0, 9], number_of_alpha_levels=3)


def test_trapezoid_reversed_core_is_rejected():
    with pytest.raises(ValueError, match=r"alpha1=\[c, d\] with c <= d"):
        phuzzy.Trapezoid(alpha0=[1, 4], alpha1=[3, 2], number_of_alpha_levels=3)


def test_trapezoid_needs_a_core():
    with pytest.raises(ValueError, match=r"needs alpha1"):
        phuzzy.Trapezoid(alpha0=[1, 4], number_of_alpha_levels=3)


def test_triangle_core_defaults_to_the_midpoint():
    p = phuzzy.Triangle(alpha0=[1, 5], number_of_alpha_levels=3)
    assert p.df.iloc[-1].l == pytest.approx(3.)


def test_triangle_accepts_a_two_point_core():
    """alpha1=[x, x] is the documented alternative to [x]."""
    p = phuzzy.Triangle(alpha0=[1, 4], alpha1=[2, 2], number_of_alpha_levels=3)
    assert p.df.iloc[-1].l == pytest.approx(2.)


def test_a_core_on_the_support_bound_is_allowed():
    p = phuzzy.Triangle(alpha0=[1, 4], alpha1=[4], number_of_alpha_levels=3)
    assert p.df.iloc[-1].l == pytest.approx(4.)


# -- explicit zero --------------------------------------------------------

def test_truncnorm_honours_an_explicit_zero_mean():
    """0 is a valid mean; `or` used to swap it for the midpoint."""
    assert phuzzy.TruncNorm(alpha0=[-1, 5], mean=0, number_of_alpha_levels=5).mean == 0.


def test_truncgennorm_honours_an_explicit_zero_mean():
    assert phuzzy.TruncGenNorm(alpha0=[-1, 5], mean=0, number_of_alpha_levels=5).mean == 0.


@pytest.mark.parametrize("clsname, kwargs, match", [
    ("TruncNorm", {"std": 0}, "std > 0"),
    ("TruncNorm", {"std": -1}, "std > 0"),
    ("TruncGenNorm", {"std": 0}, "std > 0"),
    ("TruncGenNorm", {"beta": 0}, "beta > 0"),
    ("TruncGenNorm", {"beta": -2}, "beta > 0"),
    ("Superellipse", {"m": 0}, "m > 0"),
    ("Superellipse", {"n": 0}, "n > 0"),
])
def test_an_invalid_zero_parameter_is_rejected_not_defaulted(clsname, kwargs, match):
    with pytest.raises(ValueError, match=match):
        _make(clsname, [1, 4], **kwargs)


def test_superellipse_n_defaults_to_m():
    assert phuzzy.Superellipse(alpha0=[-1, 2], m=1.5, number_of_alpha_levels=5).n == 1.5


# -- caching and setters --------------------------------------------------

def test_truncnorm_setters_rebuild_the_distribution_and_the_levels():
    p = phuzzy.TruncNorm(alpha0=[1, 3], number_of_alpha_levels=5)
    p.loc = 1.2
    assert p.distr.kwds["loc"] == pytest.approx(1.2)
    assert p.df.iloc[-1].l == pytest.approx(1.2, abs=1e-2)
    p.scale = .1
    assert p.distr.kwds["scale"] == pytest.approx(.1)


def test_truncgennorm_setters_rebuild_the_distribution():
    p = phuzzy.TruncGenNorm(alpha0=[1, 4], number_of_alpha_levels=5, beta=3.)
    p.loc = 1.2
    assert p.distr.kwds["loc"] == pytest.approx(1.2)
    p.scale = .5
    assert p.distr.kwds["scale"] == pytest.approx(.5)
    p.beta = 5.
    assert p.distr.kwds["beta"] == pytest.approx(5.)


def test_setters_reject_invalid_values():
    p = phuzzy.TruncNorm(alpha0=[1, 3], number_of_alpha_levels=5)
    with pytest.raises(ValueError, match="std > 0"):
        p.scale = 0.
    q = phuzzy.TruncGenNorm(alpha0=[1, 4], number_of_alpha_levels=5)
    with pytest.raises(ValueError, match="std > 0"):
        q.scale = -1.
    with pytest.raises(ValueError, match="beta > 0"):
        q.beta = 0.


# -- TruncGenNorm scale ---------------------------------------------------

def test_truncgennorm_scale_puts_the_ppf_limits_on_the_support():
    """Closed form: gennorm is a location-scale family, so no fit is needed."""
    a, b, beta = 1., 4., 3.
    p = phuzzy.TruncGenNorm(alpha0=[a, b], beta=beta, number_of_alpha_levels=5)
    assert p.distr.ppf(.001) == pytest.approx(a)
    assert p.distr.ppf(.999) == pytest.approx(b)
    expected = (b - a) / (gennorm.ppf(.999, beta=beta) - gennorm.ppf(.001, beta=beta))
    assert p.scale == pytest.approx(expected)


def test_truncgennorm_scale_is_a_scalar():
    """It used to be the (1,)-shaped array the optimizer returned."""
    p = phuzzy.TruncGenNorm(alpha0=[1, 4], number_of_alpha_levels=5)
    assert np.isscalar(p.distr.kwds["scale"]) or np.ndim(p.distr.kwds["scale"]) == 0


def test_truncgennorm_fit_and_use_agree_on_loc():
    """The scale used to be fitted for loc=(a+b)/2 and then applied with
    self.mean, so an explicit mean silently got a scale meant for another."""
    p = phuzzy.TruncGenNorm(alpha0=[0, 10], mean=1., number_of_alpha_levels=5)
    assert p.distr.kwds["loc"] == pytest.approx(1.)
    assert p.distr.median() == pytest.approx(1.)


def test_truncgennorm_honours_custom_ppf_limits():
    a, b = 1., 4.
    p = phuzzy.TruncGenNorm(alpha0=[a, b], ppf=[.01, .99], number_of_alpha_levels=5)
    assert p.distr.ppf(.01) == pytest.approx(a)
    assert p.distr.ppf(.99) == pytest.approx(b)


def test_truncgennorm_rejects_reversed_ppf_limits():
    with pytest.raises(ValueError, match=r"p_lo < p_hi"):
        phuzzy.TruncGenNorm(alpha0=[1, 4], ppf=[.99, .01], number_of_alpha_levels=5)


# -- from_data ------------------------------------------------------------

def test_triangle_from_data_clips_an_infeasible_peak():
    """3*mean - a - b leaves the support for a spike with outliers; the peak
    used to land at 0.006 for a support of [1, 5], which is not a triangle."""
    y = phuzzy.Triangle.from_data(data=[1, 2, 5] + [2] * 1000, number_of_alpha_levels=3)
    support = y.df.iloc[0]
    peak = y.df.iloc[-1].l
    assert support.l <= peak <= support.r
    assert (y.df.l.diff().dropna() >= -1e-12).all()
    assert (y.df.r.diff().dropna() <= 1e-12).all()


def test_triangle_from_data_keeps_a_feasible_peak():
    """Symmetric data leaves the moment estimate inside the support untouched."""
    data = [1., 2., 3.]
    y = phuzzy.Triangle.from_data(data=data, number_of_alpha_levels=3)
    assert y.df.iloc[-1].l == pytest.approx(2., abs=.2)
