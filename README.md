# phuzzy

[![GitPitch](https://gitpitch.com/assets/badge.svg)](https://gitpitch.com/lepy/phuzzy/master?grs=github&t=beige)
[![Codacy Badge](https://api.codacy.com/project/badge/Grade/4814372e95c543a69c69004c853b17be)](https://www.codacy.com/app/lepy/phuzzy?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=lepy/phuzzy&amp;utm_campaign=Badge_Grade)
[![Coverage Status](https://coveralls.io/repos/github/lepy/phuzzy/badge.svg?branch=master)](https://coveralls.io/github/lepy/phuzzy?branch=master)
[![Travis](https://img.shields.io/travis/lepy/phuzzy.svg)](https://travis-ci.org/lepy/phuzzy)
[![Dokumentation Status](https://readthedocs.org/projects/phuzzy/badge/?version=latest)](https://phuzzy.readthedocs.io/en/latest/?badge=latest)
[![Updates](https://pyup.io/repos/github/lepy/phuzzy/shield.svg)](https://pyup.io/repos/github/lepy/phuzzy/)
[![saythanks.io/to/lepy](https://img.shields.io/badge/Say%20Thanks-!-1EAEDB.svg)](https://saythanks.io/to/lepy)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.1219616.svg)](https://doi.org/10.5281/zenodo.1219616)

* python representation of fuzzy numbers|data
* specify uncertainty easily

# Usage

To use phuzzy in a project:

## phuzzy.Triangle

    p = phuzzy.Triangle(alpha0=[1,4], alpha1=[2], number_of_alpha_levels=5)

    p.df

       alpha     l    r
    0   0.00  1.00  4.0
    1   0.25  1.25  3.5
    2   0.50  1.50  3.0
    3   0.75  1.75  2.5
    4   1.00  2.00  2.0

![](doc/triangle.png)

## phuzzy.Trapezoid

    p = phuzzy.Trapezoid(alpha0=[1,4], alpha1=[2,3], number_of_alpha_levels=5)

    p.df

       alpha     l     r
    0   0.00  1.00  4.00
    1   0.25  1.25  3.75
    2   0.50  1.50  3.50
    3   0.75  1.75  3.25
    4   1.00  2.00  3.00

![](doc/trapezoid.png)

## phuzzy.TruncNorm

    p = phuzzy.TruncNorm(alpha0=[1,3], number_of_alpha_levels=15, name="x")

    p.df

           alpha         l         r
    0   0.000000  1.000000  3.000000
    1   0.071429  1.234184  2.765816
    2   0.142857  1.342402  2.657598
    3   0.214286  1.414912  2.585088
    4   0.285714  1.472370  2.527630
    5   0.357143  1.521661  2.478339
    6   0.428571  1.566075  2.433925
    7   0.500000  1.607529  2.392471
    8   0.571429  1.647354  2.352646
    9   0.642857  1.686656  2.313344
    10  0.714286  1.726558  2.273442
    11  0.785714  1.768503  2.231497
    12  0.857143  1.814923  2.185077
    13  0.928571  1.871675  2.128325
    14  1.000000  2.000000  2.000000

![](doc/truncnorm.png)

## phuzzy.TruncGenNorm

    import phuzzy.mpl as phm
    tgn = phm.TruncGenNorm(alpha0=[1, 4], alpha1=[2, 3], number_of_alpha_levels=15, beta=3.)
    tgn.plot(show=False, filepath="truncgennorm.png", title=True)

![](doc/truncgennorm.png)

## phuzzy.Superellipse

    import phuzzy.mpl as phm
    se = phm.Superellipse(alpha0=[-1, 2.], alpha1=None, m=1.0, n=.5, number_of_alpha_levels=17)
    se.plot(show=True, filepath="superellipse.png", title=True)

![](doc/superellipse.png)

## Addition

    x = phuzzy.Trapezoid(alpha0=[0, 4], alpha1=[2, 3], number_of_alpha_levels=5)
    y = phuzzy.TruncNorm(alpha0=[1, 3], number_of_alpha_levels=15, name="y")
    z = x + y
    z.name = "x+y"

![](docs/operations/x+y.png)

## Substaction

    x = phuzzy.Trapezoid(alpha0=[0, 4], alpha1=[2, 3], number_of_alpha_levels=5)
    y = phuzzy.TruncNorm(alpha0=[1, 3], number_of_alpha_levels=15, name="y")
    z = x - y
    z.name = "x-y"

![](docs/operations/x-y.png)

## Multiplication

    x = phuzzy.Trapezoid(alpha0=[0, 4], alpha1=[2, 3], number_of_alpha_levels=5)
    y = phuzzy.TruncNorm(alpha0=[1, 3], number_of_alpha_levels=15, name="y")
    z = x * y
    z.name = "x*y"

![](docs/operations/x_mul_y.png)

## Division

    x = phuzzy.Trapezoid(alpha0=[0, 4], alpha1=[2, 3], number_of_alpha_levels=5)
    y = phuzzy.TruncNorm(alpha0=[1, 3], number_of_alpha_levels=15, name="y")
    z = x / y
    z.name = "x/y"

![](docs/operations/x:y.png)

## Exponentiation

    x = phuzzy.Trapezoid(alpha0=[0, 4], alpha1=[2, 3], number_of_alpha_levels=5)
    y = phuzzy.TruncNorm(alpha0=[1, 3], number_of_alpha_levels=15, name="y")
    z = x ** y
    z.name = "x^y"

![](docs/operations/x_pow_y.png)

# cite

## plain

    Ingolf Lepenies (2018): phuzzy - a python package for fuzzy data.
    Zenodo. http://doi.org/10.5281/zenodo.1219616

## bibtex

    @article{phuzzy,
    title={phuzzy - a python package for fuzzy data},
    DOI={10.5281/zenodo.1219616},
    publisher={Zenodo},
    author={Ingolf Lepenies},
    year={2018}}


"I can live with doubt and uncertainty! 
I think it's much more exciting to live not knowing than to have answers 
which might be wrong."

Richard Feynman
