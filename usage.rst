.. image:: https://img.shields.io/pypi/v/phuzzy.svg
    :target: https://pypi.python.org/pypi/phuzzy

.. image:: https://readthedocs.org/projects/phuzzy/badge/?version=latest
    :target: https://phuzzy.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status

.. image:: https://travis-ci.org/lepy/phuzzy.svg?branch=master
    :target: https://travis-ci.org/lepy/phuzzy

.. image:: https://coveralls.io/repos/github/lepy/phuzzy/badge.svg
    :target: https://coveralls.io/github/lepy/phuzzy

.. image:: https://pyup.io/repos/github/lepy/phuzzy/shield.svg
    :target: https://pyup.io/repos/github/lepy/phuzzy/
    :alt: Updates

.. image:: https://api.codacy.com/project/badge/Grade/4814372e95c543a69c69004c853b17be
    :target: https://www.codacy.com/app/lepy/phuzzy?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=lepy/phuzzy&amp;utm_campaign=Badge_Grade

.. image:: https://img.shields.io/badge/Say%20Thanks-!-1EAEDB.svg
    :target: https://saythanks.io/to/lepy

.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.1219617.svg
   :target: https://doi.org/10.5281/zenodo.1219617


PHUZZY
======

* python representation of fuzzy numbers|data
* specify uncertainty easily

Usage
=====

To use phuzzy in a project:


.. code-block:: python

    # create a fuzzy number
    p = phuzzy.Triangle(alpha0=[1,4], alpha1=[2],
                        number_of_alpha_levels=5)
    # show alpha levels
    p.df

.. code-block::

       alpha     l    r
    0   0.00  1.00  4.0
    1   0.25  1.25  3.5
    2   0.50  1.50  3.0
    3   0.75  1.75  2.5
    4   1.00  2.00  2.0

Available shapes
----------------

Uniform
^^^^^^^

.. code-block:: python

    import phuzzy.mpl as phm
    uni = phm.Uniform(alpha0=[1, 4], number_of_alpha_levels=5, name="x")
    uni.plot(show=True, filepath="uniform.png", title=True)

Triangle
^^^^^^^^

.. code-block:: python

    import phuzzy.mpl as phm

    tri = phm.Triangle(alpha0=[1, 4], alpha1=[2], number_of_alpha_levels=5)
    tri.plot(show=False, filepath="triangle.png", title=True)

Trapezoid
^^^^^^^^^

.. code-block:: python

    import phuzzy.mpl as phm
    trap = phm.Trapezoid(alpha0=[1, 5], alpha1=[2, 3], number_of_alpha_levels=5)
    trap.plot(show=False, filepath="trapezoid.png", title=True)

TruncNorm
^^^^^^^^^

.. code-block:: python

    import phuzzy.mpl as phm
    tn = phm.TruncNorm(alpha0=[1, 3], number_of_alpha_levels=15, name="x")
    tn.plot(show=False, filepath="truncnorm.png", title=True)

TruncGenNorm
^^^^^^^^^^^^

.. code-block:: python

    import phuzzy.mpl as phm
    tgn = phm.TruncGenNorm(alpha0=[1, 4], alpha1=[2, 3], number_of_alpha_levels=15, beta=3.)
    tgn.plot(show=False, filepath="truncgennorm.png", title=True)

Superellipse
^^^^^^^^^^^^

.. code-block:: python

    import phuzzy.mpl as phm
    se = phm.Superellipse(alpha0=[-1, 2.], alpha1=None, m=1.0, n=.5, number_of_alpha_levels=17)
    se.plot(show=True, filepath="superellipse.png", title=True)

Basic operations
----------------

Addition
^^^^^^^^

.. code-block:: python

    x = phuzzy.Trapezoid(alpha0=[0, 4], alpha1=[2, 3], number_of_alpha_levels=5)
    y = phuzzy.TruncNorm(alpha0=[1, 3], number_of_alpha_levels=15, name="y")
    z = x + y
    z.name = "x+y"


Substraction
^^^^^^^^^^^^

.. code-block:: python

    x = phuzzy.Trapezoid(alpha0=[0, 4], alpha1=[2, 3], number_of_alpha_levels=5)
    y = phuzzy.TruncNorm(alpha0=[1, 3], number_of_alpha_levels=15, name="y")
    z = x - y
    z.name = "x-y"


Multiplication
^^^^^^^^^^^^^^

.. code-block:: python

    x = phuzzy.Trapezoid(alpha0=[0, 4], alpha1=[2, 3], number_of_alpha_levels=5)
    y = phuzzy.TruncNorm(alpha0=[1, 3], number_of_alpha_levels=15, name="y")
    z = x * y
    z.name = "x*y"

Division
^^^^^^^^

.. code-block:: python

    x = phuzzy.Trapezoid(alpha0=[0, 4], alpha1=[2, 3], number_of_alpha_levels=5)
    y = phuzzy.TruncNorm(alpha0=[1, 3], number_of_alpha_levels=15, name="y")
    z = x / y
    z.name = "x/y"


Exponentiation
^^^^^^^^^^^^^^

.. code-block:: python

    x = phuzzy.Trapezoid(alpha0=[0, 4], alpha1=[2, 3], number_of_alpha_levels=5)
    y = phuzzy.TruncNorm(alpha0=[1, 3], number_of_alpha_levels=15, name="y")
    z = x ** y
    z.name = "x^y"

