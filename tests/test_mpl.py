# -*- coding: utf-8 -*-
import os
os.environ["DISPLAY"] = "localhost:0.0"
os.environ["MPLBACKEND"] = "AGG"

import matplotlib
matplotlib.use('Agg')

import phuzzy
import phuzzy.mpl as phm
from phuzzy.mpl import mix_mpl

def test_dyn_mix():
    p = phuzzy.Trapezoid(alpha0=[1, 4], alpha1=[2, 3], number_of_alpha_levels=5)
    print(p)
    print(p.df)
    p.convert_df(5)
    print(p.df)
    assert not hasattr(p, "plot")
    mix_mpl(p)
    assert hasattr(p, "plot")

def test_mpl_plot():
    p = phm.Uniform(alpha0=[1, 4], alpha1=[2, 3], number_of_alpha_levels=5)
    assert hasattr(p, "plot")

    p = phm.Triangle(alpha0=[1, 4], alpha1=[2, 3], number_of_alpha_levels=5)
    assert hasattr(p, "plot")

    p = phm.Trapezoid(alpha0=[1, 4], alpha1=[2, 3], number_of_alpha_levels=5)
    assert hasattr(p, "plot")

    p = phm.TruncNorm(alpha0=[1, 4], alpha1=[2, 3], number_of_alpha_levels=5)
    assert hasattr(p, "plot")

    p = phm.TruncGenNorm(alpha0=[1, 4], alpha1=[2, 3], number_of_alpha_levels=5)
    assert hasattr(p, "plot")

    p = phm.Superellipse(alpha0=[1, 4], alpha1=[2, 3], number_of_alpha_levels=5)
    assert hasattr(p, "plot")
    fig = p.plot()
    assert fig is not None

    p = phm.MPL_Mixin()
    assert hasattr(p, "plot")

    o = phuzzy.Triangle(alpha0=[1, 4], alpha1=[2, 3], number_of_alpha_levels=5)
    assert not hasattr(o, "plot")
    phm.extend_instance(o, phm.MPL_Mixin)
    assert hasattr(o, "plot")
