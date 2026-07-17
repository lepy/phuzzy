# -*- coding: utf-8 -*-
"""Tests for the standalone plot helpers in phuzzy.mpl.plots."""

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
from mpl_toolkits.mplot3d import Axes3D

import phuzzy
import phuzzy.mpl as phm
import phuzzy.mpl.plots as mplots


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


@pytest.fixture
def raw():
    """Plain fuzzy numbers, without the mpl mixin."""
    x = phuzzy.Triangle(alpha0=[0, 4], alpha1=[2], number_of_alpha_levels=5, name="x")
    y = phuzzy.Triangle(alpha0=[1, 5], alpha1=[3], number_of_alpha_levels=5, name="y")
    return x, y


@pytest.fixture
def data():
    return np.random.default_rng(42).normal(5.0, 1.0, 200)


def test_plot_xy_mixes_in_plotting_on_a_copy(raw):
    x, y = raw
    fig, axs = mplots.plot_xy(x, y)
    assert len(axs) == 2
    # the originals stay untouched, only the copies get the mixin
    assert not hasattr(x, "plot")
    assert not hasattr(y, "plot")
    assert axs[0].get_xlabel() == "x"
    assert axs[1].get_xlabel() == "y"


def test_plot_xy_accepts_already_plottable_numbers():
    x = phm.Triangle(alpha0=[0, 4], alpha1=[2], number_of_alpha_levels=5, name="x")
    y = phm.Triangle(alpha0=[1, 5], alpha1=[3], number_of_alpha_levels=5, name="y")
    fig, axs = mplots.plot_xy(x, y)
    assert len(axs) == 2


def test_plot_xyz_draws_three_panels(raw):
    x, y = raw
    z = phuzzy.Triangle(alpha0=[2, 6], alpha1=[4], number_of_alpha_levels=5, name="z")
    fig, axs = mplots.plot_xyz(x, y, z)
    assert len(axs) == 3
    assert [a.get_xlabel() for a in axs] == ["x", "y", "z"]


def test_plot_xyz_accepts_already_plottable_numbers():
    args = [phm.Triangle(alpha0=[0, 4], alpha1=[2], number_of_alpha_levels=5, name=n)
            for n in "xyz"]
    fig, axs = mplots.plot_xyz(*args)
    assert len(axs) == 3


def test_plot_3d_builds_its_own_figure(raw):
    x, y = raw
    fig, ax = mplots.plot_3d(x, y)
    assert fig is not None
    assert isinstance(ax, Axes3D)
    assert ax.get_xlabel() == "x"
    assert ax.get_ylabel() == "y"
    assert ax.get_zlim() == (0, 1)


def test_plot_3d_uses_a_given_axes(raw):
    x, y = raw
    fig0 = plt.figure()
    ax0 = fig0.add_subplot(111, projection="3d")
    fig, ax = mplots.plot_3d(x, y, ax=ax0)
    assert ax is ax0
    assert fig is None


def test_plot_3d_can_show(raw, monkeypatch):
    x, y = raw
    shown = []
    monkeypatch.setattr(plt, "show", lambda *a, **k: shown.append(True))
    mplots.plot_3d(x, y, show=True)
    assert shown == [True]


def test_plot_xy_3d_returns_a_real_3d_axes(raw):
    """Regression: this used to store the (fig, ax) tuple from plot_3d."""
    x, y = raw
    fig, axs = mplots.plot_xy_3d(x, y)
    assert axs.shape == (2, 2)
    assert isinstance(axs[0, 1], Axes3D)


def test_plot_xy_3d_shares_the_xy_axes(raw):
    x, y = raw
    fig, axs = mplots.plot_xy_3d(x, y)
    axx, axxy = axs[0, 0], axs[1, 0]
    assert axx in axxy.get_shared_x_axes().get_siblings(axxy)
    assert axxy.get_xlabel() == "x"
    assert axxy.get_ylabel() == "y"


def test_plot_xy_3d_accepts_already_plottable_numbers():
    x = phm.Triangle(alpha0=[0, 4], alpha1=[2], number_of_alpha_levels=5, name="x")
    y = phm.Triangle(alpha0=[1, 5], alpha1=[3], number_of_alpha_levels=5, name="y")
    fig, axs = mplots.plot_xy_3d(x, y)
    assert isinstance(axs[0, 1], Axes3D)


def test_plot_hist_draws_a_line(data):
    fig, ax = mplots.plot_hist(data)
    assert len(ax.lines) == 1


def test_plot_hist_normed_scales_to_one():
    x = np.concatenate([np.zeros(10), np.ones(5)])
    _, ax = mplots.plot_hist(x, normed=True, bins=2)
    assert ax.lines[0].get_ydata().max() == pytest.approx(1.0)


def test_plot_hist_default_keeps_raw_counts():
    """normed defaults to 1, which is not True, so no scaling happens."""
    x = np.concatenate([np.zeros(10), np.ones(5)])
    _, ax = mplots.plot_hist(x, bins=2)
    assert ax.lines[0].get_ydata().max() == pytest.approx(10.0)


def test_plot_hist_filled_uses_a_patch(data):
    _, ax = mplots.plot_hist(data, filled=True, label="h", color="g", alpha=.4)
    assert len(ax.lines) == 0
    assert len(ax.collections) == 1


def test_plot_hist_uses_the_given_axes(data):
    _, ax0 = plt.subplots()
    _, ax = mplots.plot_hist(data, ax=ax0)
    assert ax is ax0


def test_plot_cdf_rises_monotonically(data):
    fig, ax = mplots.plot_cdf(data)
    y = ax.lines[0].get_ydata()
    assert np.all(np.diff(y) >= 0)
    assert y.min() > 0 and y.max() < 1


def test_plot_cdf_drops_non_positive_values():
    x = np.array([-2., -1., 0., 1., 2., 3.])
    _, ax = mplots.plot_cdf(x)
    assert len(ax.lines[0].get_xdata()) == 3


@pytest.mark.parametrize("method", ["weibull", "nn", "blom", "rossow", "cdf"])
def test_plot_cdf_supports_each_plotting_position(data, method):
    _, ax = mplots.plot_cdf(data, method=method)
    assert len(ax.lines) == 1


def test_plot_cdf_filled_uses_a_patch(data):
    _, ax = mplots.plot_cdf(data, filled=True, label="c", color="b", alpha=.3)
    assert len(ax.lines) == 0
    assert len(ax.collections) == 1


def test_plot_cdf_uses_the_given_axes(data):
    _, ax0 = plt.subplots()
    _, ax = mplots.plot_cdf(data, ax=ax0)
    assert ax is ax0
