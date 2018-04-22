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


def test_poor_mens_alpha_optimization_pow_fuzzy():
    p = phuzzy.Trapezoid(alpha0=[-2, 2], alpha1=[-1, 1])
    e = phuzzy.Trapezoid(alpha0=[2, 2], alpha1=[2, 2])
    z = p ** e
    print(z.df)
    mix_mpl(z)
    z.plot(show=True)
