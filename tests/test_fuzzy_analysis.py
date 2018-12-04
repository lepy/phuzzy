# -*- coding: utf-8 -*-

import phuzzy
import phuzzy.analysis
import numpy as np
import phuzzy as ph

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

    df = pa.lcefa()
    print(df)

def test_lcefalr():

    a = 1.23
    def f(x):
        x1 = x[0]
        x2 = x[1]
        return (x1+1.15)**2+(x2+.4)**4 + a

    x = phuzzy.TruncNorm(alpha0=[-6, 2], name="x", number_of_alpha_levels=2)
    y = phuzzy.Triangle(alpha0=[-6, 6], alpha1=[-6], name="y", number_of_alpha_levels=2)
    z = phuzzy.Triangle(alpha0=[-6, 6], alpha1=[6], name="z", number_of_alpha_levels=2)

    pa = phuzzy.analysis.FuzzyAnalysis(designvars=[x, y, z], function=f)
    print(pa)

    df = pa.lcefalr()
    print(df)

def test_lcefalr2():
    a = 1.23
    def f(x):
        x2 = x[0]

        return x2

    # x = phuzzy.TruncNorm(alpha0=[-6, 2], name="x", number_of_alpha_levels=3)
    y = phuzzy.Triangle(alpha0=[-6, 6], alpha1=[6], name="y", number_of_alpha_levels=3)
    # z = phuzzy.Triangle(alpha0=[-6, 6], alpha1=[6], name="z", number_of_alpha_levels=3)

    pa = phuzzy.analysis.FuzzyAnalysis(designvars=[y], function=f)
    print(pa)

    df = pa.lcefalr()
    print(df)

def test_sst():

    number_of_alpha_levels = 31

    # load P
    P0 = 5000.  # N
    dP = 0.01 * P0  # N
    P = ph.Triangle(alpha0=[P0 - dP, P0 + dP], alpha1=[P0], name="P", number_of_alpha_levels=number_of_alpha_levels)

    # dimensions L, W, H
    W0 = 50  # mm
    H0 = 100  # mm
    L0 = 2000  # mm

    dW = 0.01 * W0  # mm
    dH = 0.01 * H0  # mm
    dL = 0.01 * L0  # mm

    L = ph.Triangle(alpha0=[L0 - dL, L0 + dL], alpha1=[L0], name="L", number_of_alpha_levels=number_of_alpha_levels)
    W = ph.Triangle(alpha0=[W0 - dW, W0 + dW], alpha1=[W0], name="W", number_of_alpha_levels=number_of_alpha_levels)
    H = ph.Triangle(alpha0=[H0 - dH, H0 + dH], alpha1=[H0], name="H", number_of_alpha_levels=number_of_alpha_levels)

    # material

    E0 = 30000.  # N/mm2
    dE = 0.1 * E0  # N/mm2
    E = ph.TruncNorm(alpha0=[E0 - dE, E0 + dE], alpha1=[E0], name="E", number_of_alpha_levels=number_of_alpha_levels)

    def calc_w(X):
        P = X[0]
        L = X[1]
        W = X[2]
        H = X[3]
        E = X[4]
        A = W * H
        A.name = "A"

        I = W * H** 3 / 12.
        I.name = "I"
        w = P * L ** 3 / (48 * E * I)
        w.name = r"P L^3 / (48 EI)"
        return w

    for x in [P, L, W, H, E]:
        print(x)

    pa = phuzzy.analysis.FuzzyAnalysis(designvars=[P, L, W, H, E], function=calc_w)
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
