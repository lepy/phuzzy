# -*- coding: utf-8 -*-
"""Tests for phuzzy.Analysis, the design variable container in phuzzy/__init__.py.

Not to be confused with test_fuzzy_analysis.py, which covers
phuzzy.analysis.FuzzyAnalysis.
"""

import phuzzy


def _dv(name):
    return phuzzy.Triangle(alpha0=[0, 4], alpha1=[2], number_of_alpha_levels=3, name=name)


def test_analysis_default_name():
    a = phuzzy.Analysis()
    assert a.name == "FuzzyAnalysis N.N."
    assert a.designvars == []


def test_analysis_custom_name():
    assert phuzzy.Analysis(name="opti").name == "opti"


def test_analysis_takes_designvars_on_construction():
    x, y = _dv("x"), _dv("y")
    a = phuzzy.Analysis(name="xy", designvars=[x, y])
    assert a.designvars == [x, y]


def test_add_designvar_appends_one():
    a = phuzzy.Analysis(name="xy")
    x = _dv("x")
    assert a.add_designvar(x) is None
    assert a.designvars == [x]


def test_add_designvars_extends_with_many():
    a = phuzzy.Analysis(name="xy")
    x, y = _dv("x"), _dv("y")
    a.add_designvar(x)
    a.add_designvars([y])
    assert a.designvars == [x, y]


def test_designvars_are_not_shared_between_instances():
    a, b = phuzzy.Analysis(), phuzzy.Analysis()
    a.add_designvar(_dv("x"))
    assert b.designvars == []


def test_analysis_str_lists_name_and_designvars():
    a = phuzzy.Analysis(name="xy", designvars=[_dv("x")])
    assert str(a).startswith("(Analysis:'xy', dv=[")
    assert repr(a) == str(a)
