# -*- coding: utf-8 -*-
"""Branch coverage for phuzzy.analysis.alo.Alpha_Level_Optimization.

test_alo.py runs the headline six-variable optimization; this module drives
the individual branches, helpers and output formats with small problems.
"""

import numpy as np
import pandas as pd
import pytest

import phuzzy
from phuzzy.analysis.alo import Alpha_Level_Optimization as ALO
from phuzzy.contrib.shgo._shgo import SHGO


OBJ = '(x[0] - 1) ** 2 + (x[1] - 3) ** 2'
# same surface as OBJ, as a callable for the SHGO helpers
_sq = lambda x: (x[0] - 1) ** 2 + (x[1] - 3) ** 2


def _tri(lo, hi, peak, levels=3):
    return phuzzy.Triangle(alpha0=[lo, hi], alpha1=[peak], number_of_alpha_levels=levels)


def _alo(**over):
    """Two triangles; both collapse to a point at alpha=1."""
    kwargs = dict(var1=_tri(0, 4, 1), var2=_tri(1, 4, 3), obj_function=OBJ, name="t")
    kwargs.update(over)
    return ALO(**kwargs)


def _mixed_alo(**over):
    """A trapezoid keeps a non-degenerate interval at alpha=1."""
    kwargs = dict(var1=_tri(0, 4, 1),
                  var2=phuzzy.Trapezoid(alpha0=[0, 4], alpha1=[2, 3], number_of_alpha_levels=3),
                  obj_function=OBJ, name="t")
    kwargs.update(over)
    return ALO(**kwargs)


@pytest.fixture
def solved():
    a = _alo()
    a.calculation(n=10, iters=1)
    return a


# -- construction ---------------------------------------------------------

def test_objective_is_required():
    with pytest.raises(ValueError, match="PLEASE IMPORT OBJECTIVE"):
        ALO(var1=_tri(0, 4, 1), var2=_tri(1, 4, 3), name="t")


def test_default_name():
    a = ALO(var1=_tri(0, 4, 1), var2=_tri(1, 4, 3), obj_function=OBJ)
    assert repr(a) == "Fuzzy Objective Value"


def test_repr_is_the_name():
    assert repr(_alo()) == "t"


def test_dimensions_come_from_the_designvars():
    a = _alo()
    assert a.dim == 2
    assert a.number_of_alpha_lvls == 3


# -- calculation ----------------------------------------------------------

def test_all_constant_bounds_at_alpha_one(solved):
    """Both triangles pin to (1, 3) at alpha=1, the objective's exact minimum."""
    top = solved.df.iloc[-1]
    assert top.alpha == pytest.approx(1.0)
    assert top.l == pytest.approx(0.0)
    assert top.r == pytest.approx(0.0)


def test_membership_widens_towards_alpha_zero(solved):
    assert solved.df.l.iloc[0] <= solved.df.l.iloc[-1]
    assert solved.df.r.iloc[0] >= solved.df.r.iloc[-1]
    assert solved.total_nfev > 0


def test_mixed_constant_and_fuzzy_bounds_at_alpha_one():
    a = _mixed_alo()
    a.calculation(n=10, iters=1)
    top = a.df.iloc[-1]
    # var1 is pinned to 1, var2 still spans [2, 3] -> objective in [0, 1]
    assert top.l == pytest.approx(0.0, abs=1e-3)
    assert top.r == pytest.approx(1.0, abs=1e-3)


def test_simplicial_optimizer_agrees_with_sobol(solved):
    a = _alo()
    a.calculation(n=10, iters=1, optimizer="simplicial")
    assert a.df.l.values == pytest.approx(solved.df.l.values, abs=1e-2)


def test_start_at_skips_the_lower_alpha_levels():
    a = _alo(var1=_tri(0, 4, 1, levels=5), var2=_tri(1, 4, 3, levels=5))
    a.calculation(n=10, iters=1, start_at=2)
    assert a.number_of_alpha_lvls == 4
    assert a.orig_number_of_alpha_lvls == 5
    # alpha no longer runs up to 1.0
    assert a.df.alpha.max() < 1.0


def test_backup_writes_one_csv_per_alpha_level(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    a = _alo()
    a.calculation(n=10, iters=1, backup=True)
    files = sorted((tmp_path / "back_up").glob("*.csv"))
    assert len(files) == 3
    assert pd.read_csv(files[0], sep=";").columns.tolist() == ["alpha", "l", "r"]


def test_backup_reuses_an_existing_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "back_up").mkdir()
    a = _alo()
    a.calculation(n=10, iters=1, backup=True)
    assert len(list((tmp_path / "back_up").glob("*.csv"))) == 3


def test_backup_with_start_at(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    a = _alo(var1=_tri(0, 4, 1, levels=5), var2=_tri(1, 4, 3, levels=5))
    a.calculation(n=10, iters=1, backup=True, start_at=2)
    assert len(list((tmp_path / "back_up").glob("*.csv"))) == 4


# -- outputs --------------------------------------------------------------

def test_compact_output_can_round(solved):
    solved.compact_output(round=2)
    assert (solved.df.l.values == solved.df.l.round(2).values).all()


def test_extanded_output_carries_the_best_individuals(solved):
    solved.extanded_output()
    df = solved.df_extanded
    assert df.columns.tolist() == ["alpha", "l", "best_indi_l", "nfev_l", "nit_l",
                                   "r", "best_indi_r", "nfev_r", "nit_r"]
    assert len(df) == 3
    assert len(df.best_indi_l.iloc[0]) == 2


def test_extanded_output_can_round(solved):
    solved.extanded_output(round=1)
    assert solved.df_extanded.l.iloc[0] == pytest.approx(round(solved.df_extanded.l.iloc[0], 1))


def test_extanded_output_with_start_at():
    a = _alo(var1=_tri(0, 4, 1, levels=5), var2=_tri(1, 4, 3, levels=5))
    a.calculation(n=10, iters=1, start_at=2)
    a.extanded_output(round=2)
    assert a.df_extanded.alpha.max() < 1.0


def test_export_simple_csv_defaults_to_the_name(tmp_path, monkeypatch, solved):
    monkeypatch.chdir(tmp_path)
    solved.export_to_csv()
    assert (tmp_path / "t_results.csv").exists()


def test_export_simple_csv_to_a_filepath(tmp_path, solved):
    solved.export_to_csv(filepath=str(tmp_path))
    assert (tmp_path / "t_results.csv").exists()


def test_export_rejects_an_unknown_dataframe_kind(solved):
    with pytest.raises(ValueError, match="simple.*extended"):
        solved.export_to_csv(df="nonsense")


def test_export_extended_csv_defaults_to_the_name(tmp_path, monkeypatch, solved):
    monkeypatch.chdir(tmp_path)
    solved.extanded_output()
    solved.export_to_csv(df="extended")
    assert (tmp_path / "t_extended_results.csv").exists()


def test_export_extended_csv_to_a_filepath(tmp_path, solved):
    solved.extanded_output()
    solved.export_to_csv(df="extended", filepath=str(tmp_path))
    assert (tmp_path / "t_extended_results.csv").exists()


# -- defuzzification ------------------------------------------------------

def test_defuzzification_alpha_one(solved):
    solved.defuzzification(method="alpha_one")
    top = solved.df.iloc[-1]
    assert solved.determin_objective == pytest.approx((top.l + top.r) / 2)
    assert solved.determin_point[1] == pytest.approx(1.0)


def test_defuzzification_mean_sits_inside_the_support(solved):
    solved.defuzzification(method="mean")
    assert solved.df.l.min() <= solved.determin_objective <= solved.df.r.max()
    assert len(solved.determin_point) == 2


def test_defuzzification_centroid_sits_inside_the_support(solved):
    solved.defuzzification(method="centroid")
    assert solved.df.l.min() <= solved.determin_objective <= solved.df.r.max()


def test_defuzzification_interpolates_below_the_peak():
    """A negated objective peaks at alpha=1, so the mean lands left of it."""
    a = _alo(obj_function='-1*((x[0] - 1) ** 2 + (x[1] - 3) ** 2)')
    a.calculation(n=10, iters=1)
    a.defuzzification(method="mean")
    assert a.determin_objective < a.df.iloc[-1].l
    assert 0.0 < a.determin_point[1] < 1.0


def test_defuzzification_mean_when_the_peak_is_an_interval():
    a = _mixed_alo()
    a.calculation(n=10, iters=1)
    a.defuzzification(method="mean")
    assert np.isfinite(a.determin_objective)


def test_defuzzification_centroid_when_the_peak_is_an_interval():
    a = _mixed_alo()
    a.calculation(n=10, iters=1)
    a.defuzzification(method="centroid")
    assert np.isfinite(a.determin_objective)


# -- helpers --------------------------------------------------------------

def test_cut_global_bounds_needs_a_start_level():
    a = _alo()
    a.start_at = None
    with pytest.raises(ValueError, match="start_at=None"):
        a._cut_global_blounds()


def test_boundary_constraints_ignores_non_fuzzy_kwargs():
    da = ALO._boundary_constraints(var1=_tri(0, 4, 1), name="t", obj_function=OBJ)
    assert da["fuzzy_variables"].size == 1


def test_boundary_constraints_upsamples_to_the_finest_alpha_grid():
    """A coarser designvar is converted up to the largest level count."""
    coarse = _tri(0, 4, 1, levels=3)
    fine = _tri(1, 4, 3, levels=7)
    da = ALO._boundary_constraints(var1=coarse, var2=fine)
    assert da["number_of_alpha_levels"].size == 7
    assert coarse.number_of_alpha_levels == 7


def test_safe_z_values_enforces_monotone_minima():
    out = ALO._safe_z_values("min", [3.0, 5.0, 4.0])
    # each value is clamped to its predecessor, then flipped to ascending alpha
    assert out.tolist() == [3.0, 3.0, 3.0]


def test_safe_z_values_enforces_monotone_maxima():
    out = ALO._safe_z_values("max", [5.0, 3.0, 4.0])
    assert out.tolist() == [5.0, 5.0, 5.0]


def test_safe_z_values_keeps_already_monotone_input():
    assert ALO._safe_z_values("min", [5.0, 4.0, 3.0]).tolist() == [3.0, 4.0, 5.0]


def test_safe_z_values_rejects_other_directions():
    with pytest.raises(ValueError, match="min.*max"):
        ALO._safe_z_values("sideways", [1.0, 2.0])


def test_min_and_max_function_values_are_opposites():
    a = _alo()
    a.x_glob = []
    x = np.array([2.0, 2.0])
    assert a._min_function_value(x) == pytest.approx(2.0)
    assert a._max_function_value(x) == pytest.approx(-2.0)


def test_function_values_reinsert_popped_constants():
    """x_glob holds (index, value) pairs for designvars pinned at this level."""
    a = _alo()
    a.x_glob = np.array([[1.0, 3.0]])
    # only the free variable is passed; the constant is spliced back in at 1
    assert a._min_function_value(np.array([1.0])) == pytest.approx(0.0)


def test_pop_constants_removes_the_fixed_bounds():
    a = _alo()
    bounds = np.array([[0.0, 4.0], [3.0, 3.0]])
    out = a._pop_constants([False, True], bounds)
    assert out.tolist() == [[0.0, 4.0]]
    assert a.x_glob.tolist() == [[1.0, 3.0]]


def test_safe_best_array_merges_constants_and_optimized_values():
    a = _alo()
    res = type("R", (), {"x": np.array([2.5])})()
    best = a._safe_best_array([False, True], res)
    assert best[0] == pytest.approx(2.5)
    # the constant is read back from the alpha=1 bound
    assert best[1] == pytest.approx(3.0)


def test_find_result_passes_through_when_shgo_found_minima():
    shc = SHGO(_sq, bounds=np.array([[0., 4.], [1., 4.]]), n=10, iters=1,
               sampling_method="sobol", options={"ftol": 1e-4}, constraints=None)
    out = ALO._find_result(shc)
    assert out is shc
    assert len(shc.LMC.xl_maps) > 0
    assert shc.res.x == pytest.approx([1., 3.], abs=1e-3)


def test_find_result_falls_back_to_a_local_minimize():
    """A flat objective offers shgo no descent, so it builds no minimiser pool
    and _find_result has to run a local solve itself."""
    shc = SHGO(lambda x: 1.0, bounds=np.array([[0., 4.], [1., 4.]]), n=8, iters=1,
               sampling_method="sobol", options={"ftol": 1e-4}, constraints=None)

    out = ALO._find_result(shc)

    assert out is shc
    assert len(shc.LMC.xl_maps) == 1         # the fallback result, not shgo's
    assert shc.res.nlfev > 0                 # the extra local solve was counted
    assert shc.res.fun == pytest.approx(1.0)
    assert shc.f_lowest == pytest.approx(shc.res.fun)
    assert shc.x_lowest == pytest.approx(shc.res.x)


def test_from_str_and_to_str_are_placeholders():
    assert ALO.from_str("x") is None
    assert _alo().to_str() is None
