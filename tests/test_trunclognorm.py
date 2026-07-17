# -*- coding: utf-8 -*-
"""Tests for phuzzy.TruncLogNorm.

The assertions are anchored to the properties the shape must have -- support,
core at the mode, monotonicity, scale invariance -- and to the closed form
being exact, rather than to recorded numbers.
"""

import numpy as np
import pytest
from scipy.stats import norm

import phuzzy


def _p(alpha0=(1, 3), levels=11, **kwargs):
    return phuzzy.TruncLogNorm(alpha0=list(alpha0), number_of_alpha_levels=levels,
                               name="p", **kwargs)


# -- validation -----------------------------------------------------------

def test_alpha0_is_required():
    with pytest.raises(ValueError, match=r"needs alpha0"):
        phuzzy.TruncLogNorm(number_of_alpha_levels=5)


def test_alpha0_needs_two_bounds():
    with pytest.raises(ValueError, match=r"needs alpha0"):
        phuzzy.TruncLogNorm(alpha0=[1], number_of_alpha_levels=5)


@pytest.mark.parametrize("alpha0", [[0, 3], [-2, 3], [3, 1], [2, 2]])
def test_support_must_be_positive_and_ordered(alpha0):
    """A log-normal lives on (0, inf), so 0 < a < b."""
    with pytest.raises(ValueError, match=r"0 < a < b"):
        phuzzy.TruncLogNorm(alpha0=alpha0, number_of_alpha_levels=5)


def test_ppf_limits_must_be_ordered():
    with pytest.raises(ValueError, match=r"p_lo < p_hi"):
        _p(ppf=[.999, .001])


@pytest.mark.parametrize("kwargs, match", [
    ({"mean": 0}, "center > 0"),
    ({"mean": -1}, "center > 0"),
    ({"std": 0}, "sigma > 0"),
    ({"std": -1}, "sigma > 0"),
])
def test_explicit_zero_is_rejected_not_defaulted(kwargs, match):
    """A given 0 is invalid input, not a request for the default."""
    with pytest.raises(ValueError, match=match):
        _p(**kwargs)


# -- default parameters ---------------------------------------------------

def test_default_center_is_the_geometric_mean():
    """Symmetric ppf limits put the median at sqrt(a*b)."""
    p = _p(alpha0=(1, 3))
    assert p.center == pytest.approx(np.sqrt(1 * 3))
    assert p.mean == p.center


def test_default_quantiles_land_exactly_on_the_support():
    a, b = 1., 3.
    p = _p(alpha0=(a, b))
    assert p.distr.ppf(.001) == pytest.approx(a)
    assert p.distr.ppf(.999) == pytest.approx(b)


def test_default_sigma_follows_the_log_width_of_the_support():
    a, b = 1., 3.
    p = _p(alpha0=(a, b))
    expected = np.log(b / a) / (norm.ppf(.999) - norm.ppf(.001))
    assert p.sigma == pytest.approx(expected)
    assert p.std == p.sigma


def test_custom_ppf_limits_are_honoured():
    a, b = 1., 3.
    p = _p(alpha0=(a, b), ppf=[.01, .99])
    assert p.distr.ppf(.01) == pytest.approx(a)
    assert p.distr.ppf(.99) == pytest.approx(b)


def test_narrower_ppf_limits_widen_the_distribution():
    """Claiming the support only spans 1%..99% leaves tails outside it, which
    takes a larger sigma than spanning 0.1%..99.9% over the same support."""
    a, b = 1., 3.
    assert _p(alpha0=(a, b), ppf=[.01, .99]).sigma > _p(alpha0=(a, b)).sigma


def test_explicit_center_and_sigma_win_over_the_defaults():
    p = _p(mean=2., std=.5)
    assert p.center == 2.
    assert p.sigma == .5


# -- shape ----------------------------------------------------------------

def test_alpha0_level_is_the_support():
    p = _p(alpha0=(1, 3))
    assert p.df.iloc[0].alpha == 0.
    assert p.df.iloc[0].l == pytest.approx(1.)
    assert p.df.iloc[0].r == pytest.approx(3.)


def test_core_is_the_mode():
    p = _p()
    top = p.df.iloc[-1]
    assert top.alpha == pytest.approx(1.)
    assert top.l == pytest.approx(p.mode)
    assert top.r == pytest.approx(p.mode)


def test_mode_sits_left_of_the_median():
    """The defining asymmetry of a log-normal: mode < median."""
    p = _p()
    assert p.mode < p.center
    assert p.mode == pytest.approx(p.center * np.exp(-p.sigma ** 2))


def test_the_number_is_right_skewed():
    p = _p()
    assert (p.df.iloc[-1].l - p.df.iloc[0].l) < (p.df.iloc[0].r - p.df.iloc[-1].r)


def test_alpha_levels_are_nested_and_ordered():
    p = _p(levels=21)
    assert (p.df.l <= p.df.r + 1e-12).all()
    assert (p.df.l.diff().dropna() >= -1e-12).all()   # l rises with alpha
    assert (p.df.r.diff().dropna() <= 1e-12).all()    # r falls with alpha


def test_alpha_levels_are_evenly_spaced():
    p = _p(levels=7)
    assert p.df.alpha.values == pytest.approx(np.linspace(0., 1., 7))


def test_alpha_cut_reproduces_the_membership_exactly():
    """The closed form must match pdf(x)/pdf(mode) to machine precision."""
    p = _p(levels=11)
    d, mode = p.distr, p.mode
    for _, row in p.df.iloc[1:].iterrows():
        assert d.pdf(row.l) / d.pdf(mode) == pytest.approx(row.alpha, abs=1e-12)
        assert d.pdf(row.r) / d.pdf(mode) == pytest.approx(row.alpha, abs=1e-12)


def test_membership_is_scale_invariant():
    """Scaling the support scales the number and leaves sigma untouched."""
    p1 = _p(alpha0=(1, 3), levels=7)
    p2 = _p(alpha0=(1000, 3000), levels=7)
    assert p2.df.l.values / 1000 == pytest.approx(p1.df.l.values)
    assert p2.df.r.values / 1000 == pytest.approx(p1.df.r.values)
    assert p2.sigma == pytest.approx(p1.sigma)


@pytest.mark.parametrize("alpha0", [(1, 100), (1, 1000), (1e-6, 1e-3), (1e6, 3e6)])
def test_wide_and_extreme_supports_stay_exact(alpha0):
    p = _p(alpha0=alpha0, levels=5)
    assert p.df.iloc[0].l == pytest.approx(alpha0[0])
    assert p.df.iloc[0].r == pytest.approx(alpha0[1])
    assert alpha0[0] < p.mode < alpha0[1]


def test_a_narrow_sigma_keeps_the_shape_inside_the_support():
    """An explicit sigma may pull the tails inside; the support still holds."""
    p = _p(alpha0=(1, 3), std=.01, levels=5)
    assert p.df.iloc[0].l == pytest.approx(1.)
    assert p.df.iloc[0].r == pytest.approx(3.)
    assert (p.df.l <= p.df.r).all()


def test_a_mode_below_the_support_collapses_to_the_lower_bound():
    """center and sigma may put the mode outside alpha0; truncation still wins."""
    p = _p(alpha0=(1, 3), mean=2., std=1.5, levels=5)
    assert p.mode < 1.
    assert (p.df.l >= 1. - 1e-12).all()
    assert (p.df.r <= 3. + 1e-12).all()


# -- distribution and setters --------------------------------------------

def test_distr_is_the_lognormal_behind_the_shape():
    p = _p()
    assert p.distr.kwds["s"] == pytest.approx(p.sigma)
    assert p.distr.kwds["scale"] == pytest.approx(p.center)


def test_distr_is_cached():
    p = _p()
    assert p.distr is p.distr


def test_setting_center_rebuilds_the_distribution_and_the_levels():
    p = _p()
    p.center = 2.5
    assert p.distr.kwds["scale"] == pytest.approx(2.5)
    assert p.mode == pytest.approx(2.5 * np.exp(-p.sigma ** 2))
    assert p.df.iloc[-1].l == pytest.approx(p.mode)


def test_setting_sigma_rebuilds_the_distribution_and_the_levels():
    p = _p()
    p.sigma = .3
    assert p.distr.kwds["s"] == pytest.approx(.3)
    assert p.df.iloc[-1].l == pytest.approx(p.mode)


def test_setters_reject_invalid_values():
    p = _p()
    with pytest.raises(ValueError, match="center > 0"):
        p.center = 0.
    with pytest.raises(ValueError, match="sigma > 0"):
        p.sigma = -1.


def test_setters_keep_the_support():
    p = _p()
    p.center = 2.2
    assert p.df.iloc[0].l == pytest.approx(1.)
    assert p.df.iloc[0].r == pytest.approx(3.)
