# -*- coding: utf-8 -*-
"""Coverage for the FuzzyNumber base API: statistics, serialization, operators.

test_fuzzy_number.py covers construction and the alpha-level dataframe; this
module drives the accessors, distribution functions and defuzzification.
"""

import numpy as np
import pandas as pd
import pytest

import phuzzy
from phuzzy.shapes import FuzzyNumber


def _tri(name="x", levels=5):
    return phuzzy.Triangle(alpha0=[1, 4], alpha1=[2], number_of_alpha_levels=levels, name=name)


def _tn(name="t", levels=5):
    """A shape that does not override pdf/cdf, so the base versions run."""
    return phuzzy.TruncNorm(alpha0=[1, 3], number_of_alpha_levels=levels, name=name)


# -- base class -----------------------------------------------------------

def test_base_discretize_is_abstract():
    with pytest.raises(NotImplementedError):
        FuzzyNumber().discretize(alpha0=[0, 1], alpha1=[.5], alpha_levels=3)


def test_update_clips_alpha1_into_alpha0():
    p = _tri()
    p.update(alpha0=[0, 5], alpha1=[-1, 9])
    # alpha1 is clamped to the alpha0 range before discretizing
    assert p.alpha0.l == pytest.approx(0.)
    assert p.alpha0.r == pytest.approx(5.)
    assert p.alpha1.l >= 0.
    assert p.alpha1.r <= 5.


def test_update_defaults_to_the_current_levels():
    p = _tri()
    p.update()
    assert p.number_of_alpha_levels == 5


def test_copy_is_deep():
    p = _tri()
    q = p.copy()
    q.name = "other"
    q.df.loc[0, "l"] = 99.
    assert p.name == "x"
    assert p.df.loc[0, "l"] != 99.


def test_setting_number_of_alpha_levels_reconverts():
    p = _tri()
    p.number_of_alpha_levels = 9
    assert p.number_of_alpha_levels == 9
    assert len(p.df) == 9


# -- membership -----------------------------------------------------------

def test_alpha_maps_x_back_to_membership():
    p = _tri()
    assert p.alpha(2.0) == pytest.approx(1.0)
    assert p.alpha(1.0) == pytest.approx(0.0)
    # outside the support
    assert p.alpha(-5.0) == pytest.approx(0.0)
    assert p.alpha(9.0) == pytest.approx(0.0)


def test_alpha_is_vectorized():
    p = _tri()
    assert p.alpha(np.array([1., 2., 4.])) == pytest.approx([0., 1., 0.])


def test_xy_traces_the_membership_outline():
    p = _tri(levels=3)
    xs, ys = p.xy()
    assert len(xs) == len(ys) == 6
    assert xs[0] == pytest.approx(p.df.l.iloc[0])
    assert ys[0] == pytest.approx(0.)
    assert ys[len(ys) // 2 - 1] == pytest.approx(1.)


def test_interval_is_the_support():
    p = _tri()
    assert p.interval == pytest.approx([1., 4.])


def test_interval_str_formats_a_range():
    assert _tri().interval_str == "[1.00-4.00]"


def test_interval_str_collapses_a_point():
    """A zero-width triangle. Constant does not qualify: it spreads by 1e-10."""
    t = phuzzy.Triangle(alpha0=[2, 2], alpha1=[2], number_of_alpha_levels=3, name="t")
    assert t.interval_str == "2.00"


def test_interval_str_of_a_constant_keeps_its_epsilon_width():
    c = phuzzy.Constant(alpha0=[2], number_of_alpha_levels=3, name="c")
    assert c.interval_str == "[2.00-2.00]"


def test_get_01_returns_the_two_key_levels():
    p = _tri()
    assert np.allclose(p.get_01, [[1., 4.], [2., 2.]])


def test_get_01_is_empty_without_a_dataframe():
    assert FuzzyNumber().get_01 == []


# -- distribution functions ----------------------------------------------

def test_base_pdf_integrates_to_one():
    """TruncNorm does not override pdf, so this exercises FuzzyNumber.pdf."""
    p = _tn(levels=21)
    xs = np.linspace(0.5, 3.5, 20001)
    assert np.trapezoid(p.pdf(xs), xs) == pytest.approx(1.0, abs=1e-3)


def test_base_pdf_is_zero_outside_the_support():
    p = _tn()
    assert p.pdf(-10.) == pytest.approx(0.)
    assert p.pdf(10.) == pytest.approx(0.)


def test_base_cdf_runs_from_zero_to_one():
    p = _tn(levels=11)
    assert p.cdf(-10., n=1e4) == pytest.approx(0.)
    assert p.cdf(10., n=1e4) == pytest.approx(1.)
    assert p.cdf(2., n=1e4) == pytest.approx(0.5, abs=1e-2)


def test_base_cdf_is_monotone():
    p = _tn(levels=11)
    y = p.cdf(np.linspace(1., 3., 50), n=1e4)
    assert np.all(np.diff(y) >= -1e-12)


def test_ppf_inverts_the_cdf():
    p = _tn(levels=11)
    x = p.ppf(.5, n=1e4)
    assert p.cdf(x, n=1e4) == pytest.approx(.5, abs=1e-2)


def test_rvs_samples_inside_the_support():
    p = _tri()
    s = p.rvs(200)
    assert len(s) == 200
    assert s.min() >= p.min()
    assert s.max() <= p.max()


def test_rvs_is_reproducible_with_a_positive_seed():
    p = _tri()
    assert p.rvs(50, seed=42) == pytest.approx(p.rvs(50, seed=42))


def test_rvs_with_a_negative_seed_sweeps_the_quantiles():
    """A negative seed swaps sampling for an even grid over [0, 1]."""
    p = _tri()
    s = p.rvs(50, seed=-1)
    assert len(s) == 50
    assert np.all(np.diff(s) >= -1e-12)


# -- defuzzification ------------------------------------------------------

def test_defuzzification_alpha_one_takes_the_peak():
    p = _tri()
    assert p.defuzzification_alpha_one() == pytest.approx(2.0)
    assert p.defuzzification(method="alpha_one") == pytest.approx(2.0)


def test_defuzzification_mean_averages_the_levels():
    p = _tri()
    assert p.defuzzification_mean() == pytest.approx(p.df[["l", "r"]].values.mean())


def test_defuzzification_p50_is_the_median():
    p = _tri()
    assert p.defuzzification_p50() == pytest.approx(p.ppf(.5))
    # 'mean' dispatches to p50, not to defuzzification_mean
    assert p.defuzzification(method="mean") == pytest.approx(p.defuzzification_p50())


def test_both_centroids_agree_on_a_symmetric_number():
    """A symmetric triangle's centre of gravity is its peak."""
    p = phuzzy.Triangle(alpha0=[0, 4], alpha1=[2], number_of_alpha_levels=11, name="s")
    assert p.defuzzification_centroid() == pytest.approx(2.0, abs=1e-6)
    assert p.defuzzification_centroid2() == pytest.approx(2.0, abs=1e-3)
    assert p.defuzzification(method="centroid") == pytest.approx(2.0, abs=1e-6)


def test_centroids_agree_on_a_skewed_number():
    p = _tri(levels=51)
    assert p.defuzzification_centroid() == pytest.approx(p.defuzzification_centroid2(), abs=1e-2)


def test_defuzzification_rejects_an_unknown_method():
    assert _tri().defuzzification(method="nonsense") is None


# -- csv round trip -------------------------------------------------------

def test_import_csv_accepts_a_path(tmp_path):
    p = _tri()
    out = tmp_path / "p.csv"
    p.df.to_csv(out, index=False)

    q = phuzzy.Triangle(alpha0=[0, 1], alpha1=[.5], number_of_alpha_levels=3)
    df = q.import_csv(str(out))
    assert df.alpha.values == pytest.approx(p.df.alpha.values)
    assert q.df.l.values == pytest.approx(p.df.l.values)


def test_import_csv_accepts_a_handle(tmp_path):
    p = _tri()
    out = tmp_path / "p.csv"
    p.df.to_csv(out, index=False)

    q = phuzzy.Triangle(alpha0=[0, 1], alpha1=[.5], number_of_alpha_levels=3)
    with open(out) as fh:
        q.import_csv(fh)
    assert q.df.r.values == pytest.approx(p.df.r.values)


# -- to_str ---------------------------------------------------------------

def test_triangle_to_str():
    assert _tri().to_str() == "tri[1, 4, 2]"


def test_triangle_to_str_without_levels():
    p = _tri()
    p.df = pd.DataFrame(columns=["alpha", "l", "r"])
    assert p.to_str() == "tri[nan, nan, nan]"


def test_trapezoid_to_str():
    p = phuzzy.Trapezoid(alpha0=[1, 4], alpha1=[2, 3], number_of_alpha_levels=5, name="t")
    assert p.to_str() == "trap[1, 4, 2, 3]"


def test_trapezoid_to_str_without_levels():
    p = phuzzy.Trapezoid(alpha0=[1, 4], alpha1=[2, 3], number_of_alpha_levels=5, name="t")
    p.df = pd.DataFrame(columns=["alpha", "l", "r"])
    assert p.to_str() == "trap[nan, nan, nan, nan]"


def test_constant_get_01_str_shows_the_single_value():
    c = phuzzy.Constant(alpha0=[2], number_of_alpha_levels=3, name="c")
    assert c.get_01_str == "2"


def test_constant_get_01_str_without_levels():
    c = phuzzy.Constant(alpha0=[2], number_of_alpha_levels=3, name="c")
    c.df = pd.DataFrame(columns=["alpha", "l", "r"])
    assert c.get_01_str == "[]"


# -- comparison and reflected operators -----------------------------------

def test_eq_compares_the_alpha_levels():
    assert _tri() == _tri()
    assert not (_tri() == _tri(levels=7))


def test_eq_against_a_scalar_is_false():
    assert not (_tri() == 3)


def test_eq_against_another_shape_is_not_implemented():
    p = _tri()
    q = phuzzy.Trapezoid(alpha0=[1, 4], alpha1=[2, 3], number_of_alpha_levels=5)
    assert p.__eq__(q) is NotImplemented


def test_rdiv_divides_a_scalar_by_the_number():
    p = phuzzy.Triangle(alpha0=[1, 4], alpha1=[2], number_of_alpha_levels=5, name="p")
    z = p.__rdiv__(2.)
    assert z.min() == pytest.approx((2. / p.df[["l", "r"]].values).min())


def test_rpow_raises_a_scalar_to_a_fuzzy_exponent():
    p = phuzzy.Triangle(alpha0=[1, 3], alpha1=[2], number_of_alpha_levels=5, name="p")
    z = 2. ** p
    assert z.alpha1.l == pytest.approx(4.0)
    assert z.min() == pytest.approx(2.0)
    assert z.max() == pytest.approx(8.0)


def test_rpow_keeps_uniform_uniform():
    u = phuzzy.Uniform(alpha0=[1, 3], number_of_alpha_levels=5, name="u")
    assert isinstance(2. ** u, phuzzy.Uniform)


def test_rpow_with_two_fuzzy_numbers():
    """x.__rpow__(y) is y ** x. Only reachable by calling it directly: for
    two fuzzy numbers, y ** x dispatches to y.__pow__(x)."""
    x = phuzzy.Triangle(alpha0=[2, 3], alpha1=[2.5], number_of_alpha_levels=5, name="x")
    y = phuzzy.Triangle(alpha0=[1, 2], alpha1=[1.5], number_of_alpha_levels=5, name="y")
    z = x.__rpow__(y)
    assert z.name == "y^x"
    # corners of y ** x: min 1**3 = 1, max 2**3 = 8
    assert z.min() == pytest.approx(1.0)
    assert z.max() == pytest.approx(8.0)
    # and it agrees with the forward operator
    assert (y ** x).min() == pytest.approx(z.min())
    assert (y ** x).max() == pytest.approx(z.max())


def test_pow_with_a_fuzzy_exponent_away_from_zero():
    x = phuzzy.Triangle(alpha0=[2, 3], alpha1=[2.5], number_of_alpha_levels=5, name="x")
    y = phuzzy.Triangle(alpha0=[1, 2], alpha1=[1.5], number_of_alpha_levels=5, name="y")
    z = x ** y
    assert z.min() == pytest.approx(2.0)   # 2**1
    assert z.max() == pytest.approx(9.0)   # 3**2


def test_pow_with_an_exponent_spanning_zero_reaches_one():
    """x**0 == 1 is attainable, so 1 must lie inside the result."""
    x = phuzzy.Triangle(alpha0=[2, 3], alpha1=[2.5], number_of_alpha_levels=5, name="x")
    y = phuzzy.Triangle(alpha0=[-1, 1], alpha1=[0], number_of_alpha_levels=5, name="y")
    z = x ** y
    assert z.min() <= 1.0 <= z.max()
