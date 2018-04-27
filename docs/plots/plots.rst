Plots
-----

.. code-block:: python

    import phuzzy
    from phuzzy.mpl import mix_mpl
    import phuzzy.mpl.plots
    x = phuzzy.TruncNorm(alpha0=[1, 2], name="x")
    y = phuzzy.Triangle(alpha0=[3, 6], alpha1=[4], name="y")
    mix_mpl(x)
    x.plot(filepath="FuzzyNumber_plot.png")

.. figure:: FuzzyNumber_plot.png
    :scale: 90 %
    :alt: FuzzyNumber_plot.png

    x.plot()

.. figure:: plot_xy.png
    :scale: 90 %
    :alt: plot_xy.png

    fig, ax = phuzzy.mpl.plots.plot_xy(x, y)

.. figure:: plot_xyz.png
    :scale: 90 %
    :alt: plot_xyz.png

    fig, ax = phuzzy.mpl.plots.plot_xyz(x, y, x+y)

.. figure:: plot_3d.png
    :scale: 90 %
    :alt: plot_3d.png

    fig, ax = phuzzy.mpl.plots.plot_3d(x, y)

.. figure:: plot_xy_3d.png
    :scale: 90 %
    :alt: plot_xy_3d.png

    fig, ax = phuzzy.mpl.plots.plot_xy_3d(x, y)
