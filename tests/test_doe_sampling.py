# -*- coding: utf-8 -*-
"""Sampling and design-variable handling of phuzzy.approx.doe.DOE."""

import numpy as np
import pytest

import phuzzy
from phuzzy.approx.doe import DOE


def _dv(name, alpha0=(0, 4), alpha1=(2,), levels=5):
    return phuzzy.Triangle(alpha0=list(alpha0), alpha1=list(alpha1),
                           number_of_alpha_levels=levels, name=name)


@pytest.fixture
def xy():
    return _dv("x"), _dv("y", alpha0=(1, 5), alpha1=(3,))


def test_doe_default_name_and_empty_samples():
    doe = DOE()
    assert doe.name == "DOE N.N."
    assert doe.designvars == {}
    assert doe.samples.empty


def test_doe_str_reports_name_and_sample_count(xy):
    x, y = xy
    doe = DOE(designvars=[x, y], name="xy")
    assert str(doe) == "(DOE:'xy', dv=odict_keys(['x', 'y']), 0 samples)"
    assert repr(doe) == str(doe)


def test_add_designvar_keys_by_name(xy):
    x, _ = xy
    doe = DOE(name="xy")
    assert doe.add_designvar(x) is None
    assert doe.designvars == {"x": x}


def test_add_designvars_keeps_insertion_order(xy):
    x, y = xy
    doe = DOE(name="xy")
    doe.add_designvars([x, y])
    assert list(doe.designvars.keys()) == ["x", "y"]


def test_samples_are_settable(xy):
    x, _ = xy
    doe = DOE(designvars=[x], name="x")
    doe.samples = "sentinel"
    assert doe.samples == "sentinel"


def test_sample_meshgrid_is_the_default_method(xy):
    x, y = xy
    doe = DOE(designvars=[x, y], name="xy")
    samples = doe.sample_doe()
    assert set(samples.columns) == {"x", "y", "alpha_0", "alpha_1", "alpha"}
    assert samples.equals(doe.samples)


def test_sample_meshgrid_covers_every_combination(xy):
    x, y = xy
    doe = DOE(designvars=[x, y], name="xy")
    samples = doe.sample_doe(method="meshgrid")
    # 5 alpha levels -> 9 discretization points per designvar
    assert len(samples) == 9 * 9
    assert samples.alpha.min() == pytest.approx(0.0)
    assert samples.alpha.max() == pytest.approx(1.0)


def test_sample_meshgrid_with_a_single_designvar(xy):
    x, _ = xy
    doe = DOE(designvars=[x], name="x")
    samples = doe.sample_doe(method="meshgrid")
    assert len(samples) == 9
    assert samples.x.min() == pytest.approx(0.0)
    assert samples.x.max() == pytest.approx(4.0)
    # alpha peaks where the triangle peaks
    assert samples.alpha.max() == pytest.approx(1.0)


def test_sample_meshgrid_alpha_is_the_row_minimum(xy):
    x, y = xy
    doe = DOE(designvars=[x, y], name="xy")
    samples = doe.sample_doe(method="meshgrid")
    assert samples.alpha.equals(samples[["alpha_0", "alpha_1"]].min(axis=1))


def test_sample_lhs_stays_inside_the_designvar_ranges(xy):
    x, y = xy
    doe = DOE(designvars=[x, y], name="xy")
    samples = doe.sample_doe(n=20, method="lhs")
    assert len(samples) == 20
    assert samples.x.min() >= x.min()
    assert samples.x.max() <= x.max()
    assert samples.y.min() >= y.min()
    assert samples.y.max() <= y.max()


def test_sample_lhs2_samples_every_alpha_level(xy):
    x, y = xy
    doe = DOE(designvars=[x, y], name="xy")
    samples = doe.sample_lhs2(n=7)
    assert list(samples.columns) == ["x", "y", "alpha"]
    # one lhs pattern of n points per alpha level
    assert len(samples) == 7 * len(x.df)
    assert sorted(samples.alpha.unique()) == pytest.approx(sorted(x.df.alpha.unique()))


def test_sample_lhs2_maps_each_level_into_its_own_interval(xy):
    x, y = xy
    doe = DOE(designvars=[x, y], name="xy")
    samples = doe.sample_lhs2(n=25)
    for _, row in x.df.iterrows():
        level = samples[np.isclose(samples.alpha, row.alpha)]
        assert len(level) == 25
        assert level.x.min() >= row.l
        assert level.x.max() <= row.r


def test_sample_lhs2_defaults_to_100_points_per_level(xy):
    x, y = xy
    doe = DOE(designvars=[x, y], name="xy")
    assert len(doe.sample_lhs2()) == 100 * len(x.df)


def test_sample_bbdesign_needs_three_dimensions(xy):
    x, y = xy
    doe = DOE(designvars=[x, y], name="xy")
    with pytest.raises(Exception, match="at least 3 dimensions"):
        doe.sample_doe(method="bb")


def test_sample_bbdesign_stays_inside_the_designvar_ranges(xy):
    x, y = xy
    z = _dv("z", alpha0=(-2, 2), alpha1=(0,))
    doe = DOE(designvars=[x, y, z], name="xyz")
    samples = doe.sample_doe(method="bb")
    assert set(["x", "y", "z"]).issubset(samples.columns)
    for dv in (x, y, z):
        assert samples[dv.name].min() >= dv.min()
        assert samples[dv.name].max() <= dv.max()


def test_sample_ccdesign_covers_three_alpha_levels(xy):
    x, y = xy
    doe = DOE(designvars=[x, y], name="xy")
    samples = doe.sample_doe(method="cc")
    assert len(samples.alpha.unique()) == 3
    assert samples.x.min() >= x.min()
    assert samples.x.max() <= x.max()


def test_gen_lhs_samples_returns_the_raw_unit_hypercube(xy):
    x, y = xy
    doe = DOE(designvars=[x, y], name="xy")
    samples = doe.gen_lhs_samples(n=12)
    assert list(samples.columns) == ["x", "y"]
    assert len(samples) == 12
    # not scaled onto the designvar ranges
    assert samples.values.min() >= 0.0
    assert samples.values.max() <= 1.0
