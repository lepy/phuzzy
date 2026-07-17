# -*- coding: utf-8 -*-
"""The last uncovered corners: accessors, property setters and error handlers."""

import numpy as np
import pytest

import phuzzy
import phuzzy.analysis
import phuzzy.approx
import phuzzy.approx.doe
import phuzzy.contrib.tgo
from phuzzy.approx.doe import DOE, Expression


def _f(x1, x2):
    return (x1 - 1) ** 2 + (x2 + .4) ** 2 + .1


def _xy():
    x = phuzzy.TruncNorm(alpha0=[1, 2], name="x", number_of_alpha_levels=3)
    y = phuzzy.Triangle(alpha0=[3, 6], alpha1=[4], name="y", number_of_alpha_levels=3)
    return x, y


# -- FuzzyAnalysis --------------------------------------------------------

def test_fuzzy_analysis_add_designvar_appends_one():
    x, y = _xy()
    pa = phuzzy.analysis.FuzzyAnalysis(function=_f)
    pa.add_designvar(x)
    pa.add_designvar(y)
    assert pa.designvars == [x, y]


def test_fuzzy_analysis_eval_survives_a_tgo_result_without_local_minima(monkeypatch):
    """A 1-D xl makes the res.xl[:, i] lookup raise, which eval swallows."""
    class _Res:
        xl = np.array([1.0, 2.0])
        funl = np.array([0.0])

    monkeypatch.setattr(phuzzy.contrib.tgo, "tgo", lambda *a, **k: _Res())

    x, y = _xy()
    pa = phuzzy.analysis.FuzzyAnalysis(designvars=[x, y], function=lambda v: v[0] + v[1])
    z = pa.eval(ntgo=10)
    assert z is not None


# -- Superellipse ---------------------------------------------------------

def test_superellipse_to_str():
    se = phuzzy.Superellipse(alpha0=[-1, 2.], alpha1=None, m=1.0, n=.5,
                             number_of_alpha_levels=5, name="se")
    assert se.to_str() == "Superellipse[-1,2]"


def test_superellipse_from_str_is_not_implemented():
    with pytest.raises(NotImplementedError):
        phuzzy.Superellipse.from_str("Superellipse[-1,2]")


# -- approx.doe.Expression ------------------------------------------------

def test_expression_str_lists_the_designvars():
    x, y = _xy()
    expr = Expression(designvars=[x, y], function=_f, name="f(x,y)")
    assert str(expr) == "(Expression:'f(x,y)', dv=odict_keys(['x', 'y'])"
    assert repr(expr) == str(expr)


def test_expression_add_designvar_keys_by_name():
    x, _ = _xy()
    expr = Expression(function=_f, name="f")
    expr.add_designvar(x)
    assert expr.designvars == {"x": x}


def test_expression_default_name():
    assert Expression(function=_f).name == "DOE N.N."


@pytest.fixture
def trained():
    x, y = _xy()
    expr = Expression(designvars=[x, y], function=_f, name="f(x,y)")
    expr.generate_training_doe(name="train", n=20, method="lhs")
    expr.eval()
    return expr


def test_fit_model_defaults_to_svr(trained):
    trained.fit_model()
    assert trained.model.__class__.__name__ == "GridSearchCV"


def test_fit_model_can_use_knn(trained):
    trained.fit_model(model="knn")
    assert trained.model.__class__.__name__ == "KNeighborsRegressor"
    # the knn is fitted on the full training set
    assert trained.model.n_samples_fit_ == 20


def test_predict_defaults_the_name_to_z(trained):
    trained.fit_model(model="knn")
    trained.generate_prediction_doe(name="prediction", n=50, method="lhs")
    z = trained.predict()
    assert z.name == "z"
    assert len(z.df) == 11


def test_predict_takes_a_name(trained):
    trained.fit_model(model="knn")
    trained.generate_prediction_doe(name="prediction", n=50, method="lhs")
    assert trained.predict(name="u").name == "u"


def test_get_fuzzynumber_from_results_keeps_the_default_name(trained):
    z = trained.get_fuzzynumber_from_results()
    assert z.samples is not None
    assert z.df_res is not None


def test_get_fuzzynumber_from_results_takes_a_name(trained):
    assert trained.get_fuzzynumber_from_results(name="q").name == "q"


# -- truncnorm property setters -------------------------------------------

def test_truncnorm_loc_and_scale_are_settable():
    p = phuzzy.TruncNorm(alpha0=[1, 3], number_of_alpha_levels=5, name="p")
    p.loc = 2.5
    assert p.loc == p.mean == 2.5
    p.scale = .75
    assert p.scale == p.std == .75


def test_truncgennorm_loc_and_scale_are_settable():
    p = phuzzy.TruncGenNorm(alpha0=[1, 4], alpha1=[2, 3], number_of_alpha_levels=5, beta=3.)
    p.loc = 2.5
    assert p.loc == p.mean == 2.5
    p.scale = .75
    assert p.scale == p.std == .75


def test_trunclognorm_center_and_sigma_are_settable():
    p = phuzzy.TruncLogNorm(alpha0=[1, 3], number_of_alpha_levels=5, name="p")
    p.center = 2
    assert p.center == p.mean == 2.0
    assert isinstance(p.center, float)
    p.sigma = 1
    assert p.sigma == p.std == 1.0
    assert isinstance(p.sigma, float)
