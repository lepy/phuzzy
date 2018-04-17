Usage
=====

To use phuzzy in a project:

.. code-block:: python

    import phuzzy
    tn = phuzzy.TruncNorm(alpha0=[2, 3], alpha1=[], number_of_alpha_levels=15, name="t")
    tri = phuzzy.Triangle(alpha0=[1, 4], alpha1=[2], number_of_alpha_levels=5)
    f = tn + tri
    print(f.df)


available shapes
----------------

Uniform
^^^^^^^

.. code-block:: python

    import phuzzy.mpl as phm
    uni = phm.Uniform(alpha0=[1, 4], number_of_alpha_levels=5, name="x")
    uni.plot(show=True, filepath="/tmp/uniform.png", title=True)

Triangle
^^^^^^^^

.. code-block:: python

    import phuzzy.mpl as phm

    tri = phm.Triangle(alpha0=[1, 4], alpha1=[2], number_of_alpha_levels=5)
    tri.plot(show=False, filepath="/tmp/triangle.png", title=True)

Trapezoid
^^^^^^^^^

.. code-block:: python

    import phuzzy.mpl as phm
    trap = phm.Trapezoid(alpha0=[1, 5], alpha1=[2, 3], number_of_alpha_levels=5)
    trap.plot(show=False, filepath="/tmp/trapezoid.png", title=True)

TruncNorm
^^^^^^^^^

.. code-block:: python

    import phuzzy.mpl as phm
    tn = phm.TruncNorm(alpha0=[1, 3], number_of_alpha_levels=15, name="x")
    tn.plot(show=False, filepath="/tmp/truncnorm.png", title=True)

TruncGenNorm
^^^^^^^^^^^^

.. code-block:: python

    import phuzzy.mpl as phm
    tgn = phm.TruncGenNorm(alpha0=[1, 4], alpha1=[2, 3], number_of_alpha_levels=15, beta=3.)
    tgn.plot(show=False, filepath="/tmp/truncgennorm.png", title=True)

Superellipse
^^^^^^^^^^^^

.. code-block:: python

    import phuzzy.mpl as phm
    se = phm.Superellipse(alpha0=[-1, 2.], alpha1=None, m=1.0, n=.5, number_of_alpha_levels=17)
    se.plot(show=True, filepath="/tmp/superellipse.png", title=True)

basic operations
----------------

Addition
^^^^^^^^

.. math::

    z = x + y

.. code-block:: python

    x = phuzzy.Trapezoid(alpha0=[0, 4], alpha1=[2, 3], number_of_alpha_levels=5)
    y = phuzzy.TruncNorm(alpha0=[1, 3], number_of_alpha_levels=15, name="y")
    z = x + y
    z.name = "x+y"


Substraction
^^^^^^^^^^^^

.. math::

    z = x - y

.. code-block:: python

    x = phuzzy.Trapezoid(alpha0=[0, 4], alpha1=[2, 3], number_of_alpha_levels=5)
    y = phuzzy.TruncNorm(alpha0=[1, 3], number_of_alpha_levels=15, name="y")
    z = x - y
    z.name = "x-y"


Multiplication
^^^^^^^^^^^^^^

.. math::

    z = x  y

.. code-block:: python

    x = phuzzy.Trapezoid(alpha0=[0, 4], alpha1=[2, 3], number_of_alpha_levels=5)
    y = phuzzy.TruncNorm(alpha0=[1, 3], number_of_alpha_levels=15, name="y")
    z = x * y
    z.name = "x*y"

Division
^^^^^^^^

.. math::

    z = \frac{x}{y}

.. code-block:: python

    x = phuzzy.Trapezoid(alpha0=[0, 4], alpha1=[2, 3], number_of_alpha_levels=5)
    y = phuzzy.TruncNorm(alpha0=[1, 3], number_of_alpha_levels=15, name="y")
    z = x / y
    z.name = "x/y"


Power
^^^^^

.. math::

    z = x^y

.. code-block:: python

    x = phuzzy.Trapezoid(alpha0=[0, 4], alpha1=[2, 3], number_of_alpha_levels=5)
    y = phuzzy.TruncNorm(alpha0=[1, 3], number_of_alpha_levels=15, name="y")
    z = x ** y
    z.name = "x^y"

