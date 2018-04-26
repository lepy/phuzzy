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
        return (x1+1.15)**2+(x2+.4)**2 + a

    x = phuzzy.TruncNorm(alpha0=[-6, 2], name="x", number_of_alpha_levels=2)
    y = phuzzy.Triangle(alpha0=[-6, 6], alpha1=[6], name="y", number_of_alpha_levels=2)
    z = phuzzy.Triangle(alpha0=[-6, 6], alpha1=[6], name="y", number_of_alpha_levels=2)

    pa = phuzzy.analysis.FuzzyAnalysis(designvars=[x, y, z], function=f)
    print(pa)

    pa.lcefa()
