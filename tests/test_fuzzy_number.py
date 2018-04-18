# -*- coding: utf-8 -*-
import sys
is_py2 = sys.version_info.major == 2

import pytest
import phuzzy
import numpy as np
from io import StringIO

def test_fuzzy():
    n = phuzzy.FuzzyNumber()
    print(n)
    print(n.__class__.__name__)
    assert hasattr(n, "name")
    assert hasattr(n, "df")
    assert hasattr(n, "number_of_alpha_levels")

    t = phuzzy.FuzzyNumber.from_data(data=[1, 3, 5], number_of_alpha_levels=3)
    print(t.df)
    assert len(t.df) == 3
    print(t.__class__.__name__)
    s = t.to_str()
    print(s)
    pdf = t.pdf([2])
    print(pdf)
    cdf = t.cdf([2])
    print(cdf)

    n = phuzzy.FuzzyNumber()
    with pytest.raises(NotImplementedError) as exp:
        s = n.to_str()
        print(s)

    with pytest.raises(NotImplementedError) as exp:
        s = phuzzy.FuzzyNumber.from_str("!")
        print(s)

    t = phuzzy.Triangle(alpha0=[1, 3], alpha1=[2], number_of_alpha_levels=15)
    p = phuzzy.Trapezoid(alpha0=[1, 4], alpha1=[2, 3], number_of_alpha_levels=5)
    n = t + p

    pdf = n.pdf([2])
    print(pdf)
    cdf = n.cdf([2])
    print(cdf)


def test_triangle():
    t = phuzzy.Triangle(alpha0=[1, 3], alpha1=[2], number_of_alpha_levels=15)
    print(t)
    print(t.__class__.__name__)
    print(t.df)
    assert len(t.df) == 15
    print([t])
    print(t.get_01)
    print(t.df.columns)
    assert all([x == y for x, y in zip(sorted(t.df.columns), sorted(["alpha", "l", "r"]))])
    s = t.to_str()
    print(s)
    pdf = t.pdf([2])
    print(pdf)
    cdf = t.cdf([2])
    print(cdf)
    t.discretize(alpha0=[1, 3], alpha1=[2], alpha_levels=5)


def test_trapezoid():
    p = phuzzy.Trapezoid(alpha0=[1, 4], alpha1=[2, 3], number_of_alpha_levels=5)
    print(p)
    print("number_of_alpha_levels", p.number_of_alpha_levels)
    print(p.df)
    print(len(p.df))
    assert len(p.df) == 5
    print(p.get_01)
    p.discretize(alpha0=[1, 4], alpha1=[2, 3], alpha_levels=5)
    s = p.to_str()
    print(s)
    pdf = p.pdf([2])
    print(pdf)
    cdf = p.cdf([2])
    print(cdf)
    p.discretize(alpha0=[1, 3], alpha1=[2, 3], alpha_levels=5)


def test_uniform():
    p = phuzzy.Uniform(alpha0=[1, 4], number_of_alpha_levels=5)
    print(p.alpha0)
    print(p.df)
    print(p.to_str())
    print(p.get_01)

    s = p.to_str()
    print(s)
    assert s == "Uniform[1,4]"

    p.discretize(alpha0=[1, 4], alpha1=[], alpha_levels=5)

    pdf = p.pdf([2])
    print(pdf)
    cdf = p.cdf([2])
    print(cdf)

    # p2 = phuzzy.Uniform.from_str(s)
    # print(p2.alpha0)
    # assert p2.alpha0


def test_from_str():
    with pytest.raises(NotImplementedError) as exp:
        p2 = phuzzy.FuzzyNumber.from_str("WTF")
        print(p2.alpha0)


def test_unify():
    x = phuzzy.Uniform(alpha0=[1, 4], number_of_alpha_levels=5)
    y = phuzzy.Trapezoid(alpha0=[1, 4], alpha1=[2, 3], number_of_alpha_levels=7)
    a, b = x._unify(y)
    assert len(a.df) == 7
    assert len(b.df) == 7


def test_discretize():
    x = phuzzy.FuzzyNumber(alpha0=[1, 4], number_of_alpha_levels=5)
    with pytest.raises(NotImplementedError) as exp:
        x.discretize(alpha0=[1, 4], alpha1=[2, 3], alpha_levels=10)


def test_import_export():
    y = phuzzy.Trapezoid(alpha0=[1, 4], alpha1=[2, 3], number_of_alpha_levels=7)
    e = y.export_csv()
    print(e)

    if is_py2:
        e = e.decode()
    fh = StringIO(e)
    fh.seek(0)

    z = phuzzy.FuzzyNumber(alpha0=[1, 4], alpha1=[2, 3], number_of_alpha_levels=2)
    z.import_csv(fh)
    assert len(z.df) == 7


def test_from_data():
    x = phuzzy.FuzzyNumber.from_data(data=[1, 3, 5], number_of_alpha_levels=3)
    print(x.df)
    assert len(x.df) == 3
    print(x.__class__.__name__)

    y = phuzzy.FuzzyNumber.from_data(data=[1, 2, 5] + [2] * 1000, number_of_alpha_levels=3)
    print(y.df)
    assert len(y.df) == 3
    print(x.__class__.__name__)

    y = phuzzy.Triangle.from_data(data=[1, 2, 5] + [2] * 1000, number_of_alpha_levels=3)
    print(y.df)
    assert len(y.df) == 3
    print(x.__class__.__name__)


if __name__ == '__main__':
    test_fuzzy()
