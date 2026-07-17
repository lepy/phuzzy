=======
History
=======

0.4.0 (2018-04-13)
------------------

* First release on PyPI.

0.5.0 (2018-04-16)
------------------

* rename FuzzyNumber.df.columns = ["alpha", "l", "r"]
* add operators __add__
* add operators __sub__
* add operators __mul__
* add operators __div__

0.6.0 (2018-04-19)
------------------

* add operators __radd__, __rsub__, __rmul__, __rdiv__, __truediv__
* add operators __neg__, __abs__, __min__, __max__, __mean__
* add operators __lt__, __gt__, __eq__, __contain__

0.7.0 (2018-04-24)
------------------

* add plots
* add approximated function evaluation

0.10.0 (2026-07-17)
-------------------

Breaking changes
~~~~~~~~~~~~~~~~

* the shapes now reject invalid input instead of building a fuzzy number whose
  alpha levels are not nested: ``alpha0`` must satisfy ``a <= b``, and the core
  ``alpha1`` must lie inside it. ``TruncLogNorm`` requires ``0 < a < b``
* ``mean``, ``std``, ``beta``, ``m`` and ``n`` are read with ``is None`` rather
  than ``or``, so an explicit ``0`` is honoured where it is valid (``mean``) and
  rejected where it is not, instead of being replaced by the default
* ``TruncNorm``, ``TruncGenNorm`` and ``TruncLogNorm`` discretize in closed form
  and derive their spread from the ``ppf`` limits, which ``TruncNorm`` and
  ``TruncLogNorm`` previously accepted and ignored. Alpha levels are exact
  rather than sampled, and ``TruncNorm`` is 3% narrower by default
* removed ``phuzzy.approx.model``, a stale copy of ``phuzzy.approx.doe``; its
  ``sample_lhs2`` moved to ``phuzzy.approx.doe.DOE``
* ``phuzzy.data.get_histogram_data`` takes ``density`` instead of ``normed``,
  following numpy
* ``Triangle.from_data`` is deterministic: it no longer bootstraps
* requires ``pandas >= 1.5`` and ``scipy >= 1.6``, adds ``numexpr``, drops
  ``shgo``

Fixes
~~~~~

* ``Alpha_Level_Optimization`` works again -- ``__init__`` aborted on a leftover
  ``sys.exit()``, and the code behind it had rotted
* ``FuzzyNumber.pdf`` works on numpy 2, which removed ``np.trapz``
* the library works on pandas 3, whose copy-on-write made ``convert_df`` write
  to a read-only view
* ``TruncNorm`` and ``TruncGenNorm`` no longer invert their alpha levels when
  ``mean`` sits away from the middle of the support
* ``__eq__`` returns False instead of raising for shapes with different alpha
  level counts; ``__pow__`` covers ``x**0 == 1`` when the exponent spans zero;
  ``__rpow__`` computes ``other ** self``
* ``DOE.sample_meshgrid`` works with a single designvar; ``sample_bbdesign``
  scales its ``[-1, 1]`` design correctly
* ``mpl.plots.plot_xy_3d`` works on current matplotlib; ``vplot`` writes its
  ``filepath`` when handed an axes

Other
~~~~~

* 100% statement coverage, enforced via ``.coveragerc``
