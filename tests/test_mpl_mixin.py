# -*- coding: utf-8 -*-
"""Branch coverage for phuzzy.mpl.MPL_Mixin and the module's backend fallback."""

import importlib
import os

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pytest

import phuzzy
import phuzzy.mpl as phm


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


@pytest.fixture
def p():
    return phm.Triangle(alpha0=[1, 4], alpha1=[2, 3], number_of_alpha_levels=5, name="p")


def test_module_selects_agg_when_no_display(monkeypatch):
    """Importing without a DISPLAY forces the non-interactive backend."""
    used = []
    monkeypatch.setenv("DISPLAY", "")
    monkeypatch.setattr(matplotlib, "use", lambda backend, *a, **k: used.append(backend))
    try:
        importlib.reload(phm)
        assert used == ["Agg"]
    finally:
        monkeypatch.undo()
        importlib.reload(phm)


def test_plot_helper_mixes_in_and_plots(monkeypatch):
    shown = []
    monkeypatch.setattr(plt, "show", lambda *a, **k: shown.append(True))
    raw = phuzzy.Triangle(alpha0=[1, 4], alpha1=[2, 3], number_of_alpha_levels=5, name="raw")
    assert not hasattr(raw, "plot")

    phm.plot(raw)

    assert hasattr(raw, "plot")
    assert shown == [True]


def test_plot_helper_can_stay_quiet(monkeypatch):
    shown = []
    monkeypatch.setattr(plt, "show", lambda *a, **k: shown.append(True))
    raw = phuzzy.Triangle(alpha0=[1, 4], alpha1=[2, 3], number_of_alpha_levels=5, name="raw")
    phm.plot(raw, show=False)
    assert shown == []


def test_plot_reuses_the_given_axes(p):
    fig0, ax0 = plt.subplots()
    fig, ax = p.plot(ax=ax0)
    assert ax is ax0
    assert fig is fig0


def test_plot_sets_title_only_when_asked(p):
    _, ax = p.plot(title=True)
    assert ax.get_title() == "Triangle"

    _, ax = p.plot()
    assert ax.get_title() == ""


def test_plot_labels_axes_and_annotates_bounds(p):
    _, ax = p.plot(labels=True)
    assert ax.get_xlabel() == "p"
    assert ax.get_ylabel() == r"$\alpha$"
    # alpha0 and alpha1 bounds, four annotations
    assert len(ax.texts) == 4


def test_plot_without_labels_leaves_axes_bare(p):
    _, ax = p.plot(labels=False)
    assert ax.get_xlabel() == ""
    assert len(ax.texts) == 0


def test_plot_honours_explicit_xlim(p):
    _, ax = p.plot(xlim=(-10, 10))
    assert ax.get_xlim() == (-10, 10)


def test_plot_widens_xlim_for_a_degenerate_number():
    """A zero-width alpha0 would collapse the axis, so dx falls back to 1."""
    c = phm.Constant(alpha0=[2], number_of_alpha_levels=3, name="c")
    _, ax = c.plot()
    lo, hi = ax.get_xlim()
    assert hi - lo == pytest.approx(0.6)


def test_plot_writes_the_file(p, tmp_path):
    out = tmp_path / "p.png"
    p.plot(filepath=str(out))
    assert out.exists() and out.stat().st_size > 0


def test_plot_can_show(p, monkeypatch):
    shown = []
    monkeypatch.setattr(plt, "show", lambda *a, **k: shown.append(True))
    p.plot(show=True)
    assert shown == [True]


def test_p_property_returns_a_figure(p):
    assert p.p is not None


def test_vplot_returns_true_and_flips_the_axes(p):
    _, ax = plt.subplots()
    assert p.vplot(ax=ax) is True
    assert ax.get_xlabel() == r"$\alpha$"
    assert ax.get_ylabel() == "p"


def test_vplot_creates_its_own_axes(p):
    assert p.vplot() is True


def test_vplot_sets_title_only_when_asked(p):
    _, ax = plt.subplots()
    p.vplot(ax=ax, title=True)
    assert ax.get_title() == "Triangle"


def test_vplot_without_labels_leaves_axes_bare(p):
    _, ax = plt.subplots()
    p.vplot(ax=ax, labels=False)
    assert ax.get_xlabel() == ""
    assert len(ax.texts) == 0


def test_vplot_honours_explicit_xlim(p):
    _, ax = plt.subplots()
    p.vplot(ax=ax, xlim=(-5, 5))
    assert ax.get_xlim() == (-5, 5)


def test_vplot_writes_the_file_with_a_given_axes(p, tmp_path):
    """Regression: fig was unbound here, so the file was silently skipped."""
    out = tmp_path / "vplot.png"
    _, ax = plt.subplots()
    p.vplot(ax=ax, filepath=str(out))
    assert out.exists() and out.stat().st_size > 0


def test_vplot_can_show(p, monkeypatch):
    shown = []
    monkeypatch.setattr(plt, "show", lambda *a, **k: shown.append(True))
    _, ax = plt.subplots()
    p.vplot(ax=ax, show=True)
    assert shown == [True]


def test_vplot_accepts_explicit_ppf(p):
    _, ax = plt.subplots()
    assert p.vplot(ax=ax, ppf=[.25, .75]) is True


# Resolved by name at call time: the reload test above rebinds the module's
# classes, so a class captured at collection time would go stale.
@pytest.mark.parametrize("clsname, kwargs", [
    ("FuzzyNumber", {}),
    ("Constant", {"alpha0": [2]}),
    ("TrunclogNorm", {"alpha0": [1, 3]}),
])
def test_mpl_shape_subclasses_are_plottable(clsname, kwargs):
    obj = getattr(phm, clsname)(number_of_alpha_levels=5, name="n", **kwargs)
    assert hasattr(obj, "plot")
    assert hasattr(obj, "vplot")
