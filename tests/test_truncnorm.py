import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

import numpy as np
import pytest
from scipy.stats import norm

from phuzzy.mpl import TruncNorm


def test_truncnorm():
    alpha0 = [0, 2]

    p = TruncNorm(alpha0=alpha0, alpha1=None, number_of_alpha_levels=7)
    print(p)
    print(p.df)

    assert len(p.df) == 7
    assert np.allclose(p.df.alpha.values, np.linspace(0., 1., 7))

    # the support, and the core at the centre
    assert np.allclose([p.df.iloc[0].l, p.df.iloc[0].r], alpha0)
    assert np.isclose(p.df.iloc[-1].l, p.mean)
    assert np.isclose(p.df.iloc[-1].r, p.mean)

    # A normal's membership is exp(-(x-loc)^2 / (2 sigma^2)), so the alpha cut
    # is loc -+ sigma*sqrt(-2 ln alpha). Checked against that definition rather
    # than against a recorded df, which would only pin whatever the code does.
    alphas = p.df.alpha.values[1:]
    half_width = p.std * np.sqrt(-2. * np.log(alphas))
    assert np.allclose(p.df.l.values[1:], p.mean - half_width)
    assert np.allclose(p.df.r.values[1:], p.mean + half_width)


def test_truncnorm_std_spans_the_support():
    """The default std places the ppf limits on the support, so the support is
    the 0.1%..99.9% range of the underlying normal."""
    a, b = 0., 2.
    p = TruncNorm(alpha0=[a, b], number_of_alpha_levels=7)
    underlying = norm(loc=p.mean, scale=p.std)
    assert underlying.ppf(.001) == pytest.approx(a)
    assert underlying.ppf(.999) == pytest.approx(b)


def test_truncnorm_off_centre_mean_stays_nested():
    """With loc away from the midpoint the shape is asymmetric and the far tail
    gets truncated. The levels must still nest -- they used to invert."""
    p = TruncNorm(alpha0=[0, 10], mean=1., number_of_alpha_levels=7)
    assert np.allclose([p.df.iloc[0].l, p.df.iloc[0].r], [0., 10.])
    assert np.isclose(p.df.iloc[-1].l, 1.)
    assert (p.df.l.diff().dropna() >= -1e-12).all()
    assert (p.df.r.diff().dropna() <= 1e-12).all()


def plot():
    alpha0 = [0, 2]

    p = TruncNorm(alpha0=alpha0, alpha1=None, number_of_alpha_levels=17, std=(alpha0[1] - alpha0[0]) / 6.)
    p.plot(show=True)
    plt.show()
