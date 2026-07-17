# -*- coding: utf-8 -*-
"""Tests for phuzzy.data.plots and the remaining phuzzy.data estimators."""

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

import phuzzy.data
import phuzzy.data.plots


RAW = [84, 81, 72, 69, 61, 69, 74, 57, 45, 65, 76, 56, 87, 99, 44, 46, 63]


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


@pytest.fixture
def data():
    return phuzzy.data.Data(RAW)


def test_estimate_probability_uses_blom_constant_for_small_samples():
    """n <= 10 switches the plotting position constant a to 3/8."""
    df = phuzzy.data.Data(RAW[:5]).estimate_probability()
    n = 5
    a = 3. / 8.
    expected = (np.linspace(1, n, n) - a) / (n + 1. - 2. * a)
    assert df.p_blom.values == pytest.approx(expected)


def test_estimate_probability_uses_half_for_larger_samples(data):
    df = data.estimate_probability()
    n = len(RAW)
    a = .5
    expected = (np.linspace(1, n, n) - a) / (n + 1. - 2. * a)
    assert df.p_blom.values == pytest.approx(expected)
    assert set(["p_blom", "p_weibull", "p_rossow"]).issubset(df.columns)


def test_p_estimates_draws_one_line_per_estimator(data):
    df = data.estimate_probability()
    fig, ax = phuzzy.data.plots.p_estimates(df)
    assert len(ax.lines) == 3
    assert ax.get_xlabel() == "x"
    assert ax.get_ylabel() == "p"


def test_p_estimates_uses_the_given_axes(data):
    df = data.estimate_probability()
    _, ax0 = plt.subplots(1, 1)
    fig, ax = phuzzy.data.plots.p_estimates(df, ax=ax0)
    assert ax is ax0


def test_p_estimates_can_show(data, monkeypatch):
    shown = []
    monkeypatch.setattr(plt, "show", lambda *a, **k: shown.append(True))
    df = data.estimate_probability()
    phuzzy.data.plots.p_estimates(df, show=True)
    assert shown == [True]


def test_bootstrapping_marks_mean_and_std(data):
    df_boot = data.bootstrap(n=100)
    fig, axs = phuzzy.data.plots.bootstrapping(data, df_boot)
    ax_mean, ax_std = axs
    assert ax_mean.get_title() == "mean"
    assert ax_std.get_title() == "standard deviation"
    assert ax_mean.get_xlabel() == "x_mean"
    assert ax_std.get_xlabel() == "x_std"
    # one vline for the bootstrap mean, one for the sample mean
    assert len(ax_mean.lines) == 2
    assert len(ax_std.lines) == 2


def test_bootstrapping_can_show(data, monkeypatch):
    shown = []
    monkeypatch.setattr(plt, "show", lambda *a, **k: shown.append(True))
    df_boot = data.bootstrap(n=50)
    phuzzy.data.plots.bootstrapping(data, df_boot, show=True)
    assert shown == [True]


def test_shuffling_returns_mean_and_std_per_draw(data):
    df = data.shuffling(n=64)
    assert list(df.columns) == ["x_mean", "x_std"]
    assert len(df) == 64
    assert df.x_mean.notna().all()


def test_plot_hist_drops_nans():
    x = np.array([1., 2., np.nan, 3., 4.])
    fig, ax = phuzzy.data.plots.plot_hist(x)
    assert len(ax.lines) == 1
    assert np.isfinite(ax.lines[0].get_ydata()).all()


def test_plot_hist_normed_scales_to_one():
    x = np.concatenate([np.zeros(10), np.ones(5)])
    _, ax = phuzzy.data.plots.plot_hist(x, normed=True, bins=2)
    assert ax.lines[0].get_ydata().max() == pytest.approx(1.0)


def test_plot_hist_unnormed_keeps_raw_counts():
    x = np.concatenate([np.zeros(10), np.ones(5)])
    _, ax = phuzzy.data.plots.plot_hist(x, bins=2)
    assert ax.lines[0].get_ydata().max() == pytest.approx(10.0)


def test_plot_hist_filled_uses_a_patch_instead_of_a_line():
    x = np.array([1., 2., 3., 4.])
    _, ax = phuzzy.data.plots.plot_hist(x, filled=True, label="f", color="g", alpha=.5)
    assert len(ax.lines) == 0
    assert len(ax.collections) == 1


def test_plot_hist_uses_the_given_axes():
    _, ax0 = plt.subplots(1, 1)
    _, ax = phuzzy.data.plots.plot_hist(np.array([1., 2., 3.]), ax=ax0)
    assert ax is ax0
