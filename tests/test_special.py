# -*- coding: utf-8 -*-
import os
import matplotlib

import phuzzy
import phuzzy.mpl as phm
from phuzzy.mpl import mix_mpl
import numpy as np


def test_zero():
    p = phuzzy.Trapezoid(alpha0=[-2, 2], alpha1=[-1, 1])
    assert p.has_zero() is True

    p = phuzzy.Trapezoid(alpha0=[1., 2], alpha1=[1, 1])
    assert p.has_zero() is False


def test_poor_mens_alpha_optimization_pow_scalar():
    p = phuzzy.Trapezoid(alpha0=[-2, 2], alpha1=[-1, 1])
    y = p ** 2
    print(y.df)
    # mix_mpl(y)
    # y.plot(show=False)

    assert np.isclose(y.min(), 0)
    assert np.isclose(y.max(), 4)
    assert np.isclose(y.df.iloc[-1][["l", "r"]].max(), 1)

    p = phuzzy.Trapezoid(alpha0=[-2, 2], alpha1=[-1, 1])
    y = p ** 0
    print(y.df)
    # mix_mpl(y)
    # y.plot(show=False)

    assert np.isclose(y.min(), 1)
    assert np.isclose(y.max(), 1)
    # assert np.isclose(y.df.iloc[-1][["l", "r"]].max(), 1)


def test_poor_mens_alpha_optimization_pow_fuzzy():
    p = phuzzy.Trapezoid(alpha0=[-2, 2], alpha1=[-1, 1])
    e = phuzzy.Trapezoid(alpha0=[2, 2], alpha1=[2, 2])
    y = p ** e
    print(y.df)
    # mix_mpl(y)
    # y.plot(show=True)

    assert np.isclose(y.min(), 0)
    assert np.isclose(y.max(), 4)
    assert np.isclose(y.df.iloc[-1][["l", "r"]].max(), 1)


def test_poor_mens_alpha_optimization_neg_pow_fuzzy():
    p = phuzzy.Trapezoid(alpha0=[-2, 2], alpha1=[-1, 1])
    e = phuzzy.Trapezoid(alpha0=[0, 0], alpha1=[0, 0])
    y = p ** e
    print(y.df)
    # mix_mpl(y)
    # y.plot(show=True)

    assert np.isclose(y.min(), 1)
    assert np.isclose(y.max(), 1)

    p = phuzzy.Trapezoid(alpha0=[0, 4], alpha1=[2, 3], number_of_alpha_levels=2, name="p")
    print(p)
    assert len(p.df) == 2


def test_poor_mens_alpha_optimization_pow_fuzzy2():
    t = phuzzy.TruncNorm(alpha0=[2, 3], alpha1=[], number_of_alpha_levels=5, name="t")
    p = phuzzy.Trapezoid(alpha0=[0, 4], alpha1=[2, 3], number_of_alpha_levels=5, name="p")
    # a = t ** p
    a = p ** t
    a.name = "t**p"
    print(a.df.values.tolist())
    print(a)
    print(a.df)
    assert np.allclose(a.df.values.tolist(),
                       [[0.0, 0.0, 64.0], [0.25, 0.125, 52.734375], [0.5, 1.0, 42.875], [0.75, 2.25, 34.328125],
                        [1.0, 4.0, 27.0]]
                       )
    # mix_mpl(a)
    # a.plot(show=True)

    print(a.df.values.tolist())
    a = t ** p
    print(a.df.values.tolist())
    # mix_mpl(a)
    # a.plot(show=True)
    assert np.allclose(a.df.values.tolist(),
                       [[0.0, 1.0, 81.0], [0.25, 1.0, 59.51536637372663], [0.5, 1.0, 52.84832630002708],
                        [0.75, 1.0, 47.58354751779905], [1.0, 1.0, 39.0625]]
                       )


def test_pow_zero():
    # p = phuzzy.Trapezoid(alpha0=[-1, 4], alpha1=[2, 3], number_of_alpha_levels=5, name="p")
    # y = p ** 0

    x = phuzzy.Trapezoid(alpha0=[-1, 4], alpha1=[.5, .8], number_of_alpha_levels=5, name="p")
    y = x ** 0
    mix_mpl(y)
    y.plot()
    print(y.df.values.tolist())
    assert np.isclose(y.min(), 1)
    assert np.isclose(y.max(), 1)
