TruncNorm
---------

.. code-block:: python
    :linenos:

    import phuzzy.mpl as phm
    tn = phm.TruncNorm(alpha0=[1, 3], number_of_alpha_levels=15, name="x")
    tn.plot(show=False, filepath="/tmp/truncnorm.png", title=True)

.. figure:: truncnorm.png
    :scale: 90 %
    :alt: TruncNorm fuzzy number

    TruncNorm fuzzy number

.. figure:: operations_TruncNorm.png
    :scale: 90 %
    :alt: TruncNorm fuzzy number operations

    TruncNorm fuzzy number operations
