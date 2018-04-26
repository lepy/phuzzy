# -*- coding: utf-8 -*-

import phuzzy
import phuzzy.analysis
import numpy as np

def test_fuzzy_analysis():

    a = 1.23
    def f(x):
        x1 = x[0]
        x2 = x[1]
        return (x1+1.15)**2+(x2+.4)**2 + a

    x = phuzzy.TruncNorm(alpha0=[-6, 2], name="x", number_of_alpha_levels=2)
    y = phuzzy.Triangle(alpha0=[-6, 6], alpha1=[6], name="y", number_of_alpha_levels=2)

    pa = phuzzy.analysis.FuzzyAnalysis(designvars=[x, y], function=f)
    print(pa)

    z = pa.eval(ntgo=1000)
    print(z)
    assert np.isclose(z.min(), a)


def test_lcefa():

    a = 1.23
    def f(x):
        x1 = x[0]
        x2 = x[1]
        return (x1+1.15)**2+(x2+.4)**4 + a

    x = phuzzy.TruncNorm(alpha0=[-6, 2], name="x", number_of_alpha_levels=2)
    y = phuzzy.Triangle(alpha0=[-6, 6], alpha1=[6], name="y", number_of_alpha_levels=2)
    z = phuzzy.Triangle(alpha0=[-6, 6], alpha1=[6], name="z", number_of_alpha_levels=2)

    pa = phuzzy.analysis.FuzzyAnalysis(designvars=[x, y, z], function=f)
    print(pa)

    pa.lcefa()

def test_f1():
    def f(x):
        x1 = x[0]
        x2 = x[1]
        return x1 + 2

    x = phuzzy.TruncNorm(alpha0=[-6, 6], name="x", number_of_alpha_levels=11)
    y = phuzzy.Triangle(alpha0=[-6, 6], alpha1=[6], name="y", number_of_alpha_levels=11)

    pa = phuzzy.analysis.FuzzyAnalysis(designvars=[x, y], function=f)
    print(pa)

    vars = pa.lcefa()
    assert np.isclose(x.sk, 1)
    assert np.isclose(y.sk, 0)

def test_f2():
    def f(x):
        x1 = x[0]
        x2 = x[1]
        return x1**2 + x2**2

    x = phuzzy.TruncNorm(alpha0=[-6, 6], name="x", number_of_alpha_levels=11)
    y = phuzzy.TruncNorm(alpha0=[-6, 6], name="y", number_of_alpha_levels=11)
    # y = phuzzy.Triangle(alpha0=[-6, 6], alpha1=[6], name="y", number_of_alpha_levels=11)
    # y = phuzzy.Uniform(alpha0=[-6, 6], name="y", number_of_alpha_levels=11)

    pa = phuzzy.analysis.FuzzyAnalysis(designvars=[x, y], function=f)
    print(pa)

    sensibilities = pa.lcefa()
    print(sensibilities)
    assert np.isclose(x.sk, .5)
    assert np.isclose(y.sk, .5)

def atest_f3():
    def f(x):
        x1 = x[0]
        x2 = x[1]
        return np.sin(x1) + np.sin(4*x2)

    x = phuzzy.TruncNorm(alpha0=[0, 2*np.pi], name="x", number_of_alpha_levels=2)
    y = phuzzy.Triangle(alpha0=[0, 2*np.pi], alpha1=[6], name="y", number_of_alpha_levels=2)

    pa = phuzzy.analysis.FuzzyAnalysis(designvars=[x, y], function=f)
    print(pa)

    sensibilities = pa.lcefa()
    print(sensibilities)
    assert np.isclose(x.sk, .5)
    assert np.isclose(y.sk, .5)
