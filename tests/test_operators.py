import phuzzy
import numpy as np


def test_operation_class():
    tra = phuzzy.Trapezoid(alpha0=[1, 4], alpha1=[2, 3], number_of_alpha_levels=5)
    tri = phuzzy.Triangle(alpha0=[1, 4], alpha1=[2, 3], number_of_alpha_levels=5)
    uni = phuzzy.Uniform(alpha0=[1, 4], alpha1=[2, 3], number_of_alpha_levels=5)
    x = tra + tra
    assert isinstance(x, phuzzy.Trapezoid)

    x = uni + uni
    assert isinstance(x, phuzzy.Uniform)

    x = tri + uni
    assert isinstance(x, phuzzy.Triangle)

    x = uni + tri
    print(x.__class__)
    assert isinstance(x, phuzzy.Triangle)

    x = tra + uni
    assert isinstance(x, phuzzy.Trapezoid)

    x = uni + tra
    assert isinstance(x, phuzzy.Trapezoid)

    x = tra + tri
    assert isinstance(x, phuzzy.Trapezoid)

    x = tri + tra
    assert isinstance(x, phuzzy.Trapezoid)

    x = uni + tri + tra
    assert isinstance(x, phuzzy.Trapezoid)

    x = uni + 2.
    assert isinstance(x, phuzzy.Uniform)


def test_add():
    t = phuzzy.TruncNorm(alpha0=[1, 3], alpha1=[2], number_of_alpha_levels=3)
    print(t)
    assert len(t.df) == 3

    p = phuzzy.Trapezoid(alpha0=[1, 4], alpha1=[2, 3], number_of_alpha_levels=5)
    print(p)
    assert len(p.df) == 5

    a = t + p
    print(a)
    print(a.df)
    print(a.df.values.tolist())
    assert np.allclose(a.df.values.tolist(), [[0.0, 2.0, 7.0], [0.25, 2.5537645812426892, 6.446235418757311],
                                              [0.5, 3.1075291624853785, 5.892470837514622],
                                              [0.75, 3.5537645812426892, 5.446235418757311], [1.0, 4.0, 5.0]])
    # mix_mpl(t)
    # mix_mpl(p)
    # mix_mpl(a)
    # t.plot()
    # p.plot()
    # a.plot(show=True)

    b = t + 4.
    print(b)
    print(b.df)
    print(b.df.values.tolist())
    assert np.allclose(b.df.values.tolist(),
                       [[0.0, 5.0, 7.0], [0.5, 5.6075291624853785, 6.392470837514622], [1.0, 6.0, 6.0]]
                       )

    b = 4 - t
    print(b)
    print(b.df)
    print(b.df.values.tolist())
    assert np.allclose(b.df.values.tolist(),
                       [[0.0, -3.0, -1.0], [0.5, -2.3924708375146215, -1.607529162485378], [1.0, -2.0, -2.0]]
                       )


def test_sub():
    t = phuzzy.TruncNorm(alpha0=[1, 3], alpha1=[2], number_of_alpha_levels=3, name="t")
    print(t)
    assert len(t.df) == 3

    p = phuzzy.Trapezoid(alpha0=[1, 4], alpha1=[2, 3], number_of_alpha_levels=3, name="p")
    print(p)
    assert len(p.df) == 3

    a = t - p
    a.name = "t-p"

    print(a.df)
    print(a.df.values.tolist())
    assert np.allclose(a.df.values.tolist(),
                       [[0.0, -3.0, 2.0], [0.5, -1.8924708375146215, 0.892470837514622], [1.0, -1.0, 0.0]]
                       )

    b = t - 4.
    print(b)
    print(b.df)
    print(b.df.values.tolist())
    assert np.allclose(b.df.values.tolist(),
                       [[0.0, -3.0, -1.0], [0.5, -2.3924708375146215, -1.607529162485378], [1.0, -2.0, -2.0]]
                       )

    b = 4. - t
    print(b)
    print(b.df)
    print(b.df.values.tolist())
    assert np.allclose(b.df.values.tolist(),
                       [[0.0, -3.0, -1.0], [0.5, -2.3924708375146215, -1.607529162485378], [1.0, -2.0, -2.0]]
                       )


def test_mul():
    t = phuzzy.TruncNorm(alpha0=[1, 3], alpha1=[2], number_of_alpha_levels=3, name="t")
    print(t)
    print(t.df.values.tolist())
    assert len(t.df) == 3

    p = phuzzy.Trapezoid(alpha0=[1, 4], alpha1=[2, 3], number_of_alpha_levels=3, name="p")
    print(p)
    assert len(p.df) == 3

    a = t * p
    a.name = "t*p"

    print(a)
    print(a.df.values.tolist())
    assert np.allclose(a.df.values.tolist(),
                       [[0.0, 1.0, 12.0], [0.5, 2.4112937437280677, 8.373647931301177], [1.0, 4.0, 6.0]]
                       )

    b = t * 1.
    print(b)
    print(b.df)
    print(b.df.values.tolist())
    assert np.allclose(b.df.values.tolist(),
                       [[0.0, 1.0, 3.0], [0.5, 1.6075291624853785, 2.392470837514622], [1.0, 2.0, 2.0]]
                       )

    b = t * 2.
    print(b)
    print(b.df)
    print(b.df.values.tolist())
    assert np.allclose(b.df.values.tolist(),
                       [[0.0, 2.0, 6.0], [0.5, 3.215058324970757, 4.784941675029244], [1.0, 4.0, 4.0]]
                       )

    b = 2. * t
    print(b)
    print(b.df)
    print(b.df.values.tolist())
    assert np.allclose(b.df.values.tolist(),
                       [[0.0, 2.0, 6.0], [0.5, 3.215058324970757, 4.784941675029244], [1.0, 4.0, 4.0]]
                       )

    p = phuzzy.Trapezoid(alpha0=[-1, 4], alpha1=[2, 3], number_of_alpha_levels=3, name="p")
    z = p * p
    print(z.df)

def test_div():
    t = phuzzy.TruncNorm(alpha0=[2, 3], alpha1=[], number_of_alpha_levels=3, name="t")
    print(t.df.values.tolist())
    print(t)

    assert len(t.df) == 3

    p = phuzzy.Trapezoid(alpha0=[0, 4], alpha1=[2, 3], number_of_alpha_levels=3, name="p")
    print(p)
    assert len(p.df) == 3

    # a = t / p
    a = p / t
    a.name = "t/p"
    print(a.df.values.tolist())
    assert np.allclose(a.df.values.tolist(),
                       [[0.0, 3.3333333333333335e-11, 2.0], [0.5, 0.37088749487272066, 1.519252456825272],
                        [1.0, 0.8, 1.2]]
                       )

    print(a)
    print(a.df)
    print("_" * 80)
    b = t / 1.
    print(b)
    print(b.df)
    print(b.df.values.tolist())
    assert np.allclose(b.df.values.tolist(),
                       [[0.0, 2.0, 3.0], [0.5, 2.3037645812426892, 2.6962354187573108], [1.0, 2.5, 2.5]]
                       )

    print(a)
    print(a.df)
    print("_" * 80)
    b = t / 2.
    print(b)
    print(b.df)
    print(b.df.values.tolist())
    assert np.allclose(b.df.values.tolist(),
                       [[0.0, 1.0, 1.5], [0.5, 1.1518822906213446, 1.3481177093786554], [1.0, 1.25, 1.25]]
                       )


def test_power():
    t = phuzzy.TruncNorm(alpha0=[2, 3], alpha1=[], number_of_alpha_levels=2, name="t")
    print(t)
    assert len(t.df) == 2

    # p = phuzzy.Trapezoid(alpha0=[0, 4], alpha1=[2, 3], number_of_alpha_levels=2, name="p")
    # print(p)
    # assert len(p.df) == 2
    #
    # a = t ** p
    # a = p ** t
    # a.name = "t**p"
    # print(a.df.values.tolist())
    # assert np.allclose(a.df.values.tolist(),
    #                    [[0.0, 1e-30, 64.0], [1.0, 5.656854249492381, 15.588457268119896]]
    #                    )
    # print(a)
    # print(a.df)

    b = t ** 3.
    print(b)
    print(b.df)
    print(b.df.values.tolist())
    assert np.allclose(b.df.values.tolist(),
                       [[0.0, 8.0, 27.0], [1.0, 15.625, 15.625]]
                       )

    b = t ** 0.
    print(b)
    print(b.df)
    print(b.df.values.tolist())
    assert np.allclose(b.df.values.tolist(),
                       [[0.0, 1.0, 1.0], [1.0, 1.0, 1.0]]
                       )

    c = 3 ** t
    print(c)
    print(c.df)
    print(c.df.values.tolist())
    assert np.allclose(c.df.values.tolist(),
                       [[0.0, 9.0, 27.0], [1.0, 15.588457268119896, 15.588457268119896]]
                       )

    c = 0 ** t
    print(c)
    print(c.df)
    print(c.df.values.tolist())
    assert np.allclose(c.df.values.tolist(),
                       [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
                       )


def test_neg():
    p = phuzzy.Trapezoid(alpha0=[0, 4], alpha1=[2, 3], number_of_alpha_levels=3, name="p")
    print(p)
    print(p.df.values.tolist())
    assert len(p.df) == 3

    b = -p
    print(b.df.values.tolist())
    assert np.allclose(b.df.values.tolist(),
                       [[0.0, -4.0, -1e-10], [0.5, -3.5, -1.00000000005], [1.0, -3.0, -2.0]]
                       )


def test_abs():
    x = phuzzy.Trapezoid(alpha0=[-3, 5], alpha1=[1, 2], name="x", number_of_alpha_levels=3)
    y = abs(x)
    print(y.df.values.tolist())
    assert np.allclose(y.df.values.tolist(),
                       [[0.0, 0.0, 5.0], [0.5, 0.0, 3.5], [1.0, 1.0, 2.0]]
                       )

    x = phuzzy.Trapezoid(alpha0=[1, 5], alpha1=[2, 3], name="x", number_of_alpha_levels=3)
    y = abs(x)
    print(y.df.values.tolist())
    assert np.allclose(y.df.values.tolist(),
                       [[0.0, 1.0, 5.0], [0.5, 1.5, 4.0], [1.0, 2.0, 3.0]]
                       )

    x = phuzzy.Trapezoid(alpha0=[-5, -1], alpha1=[-3, -2], name="x", number_of_alpha_levels=3)
    y = abs(x)
    print(y.df.values.tolist())
    assert np.allclose(y.df.values.tolist(),
                       [[0.0, 1.0, 5.0], [0.5, 1.5, 4.0], [1.0, 2.0, 3.0]]
                       )
    y = x.abs()
    print(y.df.values.tolist())
    assert np.allclose(y.df.values.tolist(),
                       [[0.0, 1.0, 5.0], [0.5, 1.5, 4.0], [1.0, 2.0, 3.0]]
                       )



def test_lt():
    x = phuzzy.Trapezoid(alpha0=[1, 5], alpha1=[2, 3], name="x", number_of_alpha_levels=3)
    assert x < 6
    assert not x < 2

    y = phuzzy.Trapezoid(alpha0=[-5, -1], alpha1=[-3, -2], name="x", number_of_alpha_levels=3)
    print(y.max(), x.min())
    assert y < x


def test_gt():
    x = phuzzy.Trapezoid(alpha0=[1, 5], alpha1=[2, 3], name="x", number_of_alpha_levels=3)
    assert x > -1
    assert not x > 2

    y = phuzzy.Trapezoid(alpha0=[-5, -1], alpha1=[-3, -2], name="x", number_of_alpha_levels=3)
    print(y.max(), x.min())
    assert x > y


def test_eq():
    x = phuzzy.Trapezoid(alpha0=[1, 5], alpha1=[2, 3], name="x", number_of_alpha_levels=3)
    x2 = phuzzy.Trapezoid(alpha0=[1, 5], alpha1=[2, 3], name="x", number_of_alpha_levels=3)
    assert x == x2
    assert not x == 2

    y = phuzzy.Trapezoid(alpha0=[-5, -1], alpha1=[-3, -2], name="x", number_of_alpha_levels=3)
    print(y.max(), x.min())
    assert x != y


def test_contains():
    x = phuzzy.Trapezoid(alpha0=[1, 5], alpha1=[2, 3], name="x", number_of_alpha_levels=3)
    assert 2 in x
    y = phuzzy.Trapezoid(alpha0=[2, 4], alpha1=[2, 3], name="y", number_of_alpha_levels=3)
    z = phuzzy.Trapezoid(alpha0=[2, 7], alpha1=[2, 3], name="z", number_of_alpha_levels=3)
    assert y in x
    assert not x in y
    assert not z in x


def test_min():
    x = phuzzy.Trapezoid(alpha0=[-3, 5], alpha1=[1, 2], name="x", number_of_alpha_levels=3)
    print(x.min())
    assert np.isclose(x.min(), -3)


def test_max():
    x = phuzzy.Trapezoid(alpha0=[-3, 5], alpha1=[1, 2], name="x", number_of_alpha_levels=3)
    print(x.max())
    assert np.isclose(x.max(), 5)

    y = phuzzy.Trapezoid(alpha0=[-5, -1], alpha1=[-3, -2], name="x", number_of_alpha_levels=3)
    print(y.df)
    print(y.max())
    assert np.isclose(y.max(), -1)


def test_mean():
    x = phuzzy.Trapezoid(alpha0=[-2, 4], alpha1=[-1, 3], name="x", number_of_alpha_levels=3)
    print(x.mean())
    assert np.isclose(x.mean(), 1)


if __name__ == '__main__':
    test_add()
