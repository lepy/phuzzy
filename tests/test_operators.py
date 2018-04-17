import phuzzy
import numpy as np


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

    b = t + 3.
    print(b)
    print(b.df)
    print(b.df.values.tolist())
    assert np.allclose(b.df.values.tolist(),
                       [[3.0, 4.0, 6.0], [3.5, 4.6075291624853785, 5.392470837514622], [4.0, 5.0, 5.0]]
                       )


def test_sub():
    t = phuzzy.TruncNorm(alpha0=[1, 3], alpha1=[2], number_of_alpha_levels=2, name="t")
    print(t)
    assert len(t.df) == 2

    p = phuzzy.Trapezoid(alpha0=[1, 4], alpha1=[2, 3], number_of_alpha_levels=2, name="p")
    print(p)
    assert len(p.df) == 2

    a = t - p
    a.name = "t-p"

    print(a.df)
    print(a.df.values.tolist())
    assert np.allclose(a.df.values.tolist(), [[0.0, -3.0, 2.0], [1.0, -1.0, 0.0]])

    b = t - 3.
    print(b)
    print(b.df)
    print(b.df.values.tolist())
    assert np.allclose(b.df.values.tolist(),
                       [[-3.0, -2.0, 0.0], [-2.0, -1.0, -1.0]]
                       )


def test_mul():
    t = phuzzy.TruncNorm(alpha0=[1, 3], alpha1=[2], number_of_alpha_levels=3, name="t")
    print(t)
    assert len(t.df) == 3

    p = phuzzy.Trapezoid(alpha0=[1, 4], alpha1=[2, 3], number_of_alpha_levels=3, name="p")
    print(p)
    assert len(p.df) == 3

    a = t * p
    a.name = "t*p"

    print(a)
    print(a.df.values.tolist())
    assert np.allclose(a.df.values.tolist(),
                       [[0.0, 1.0, 12.0], [0.5, 2.4112937437280677, 8.373647931301177], [1.0, 4.0, 6.0]])

    b = t * 3.
    print(b)
    print(b.df)
    print(b.df.values.tolist())
    assert np.allclose(b.df.values.tolist(),
                       [[0.0, 3.0, 9.0], [1.5, 4.822587487456135, 7.1774125125438655], [3.0, 6.0, 6.0]]
                       )


def test_div():
    t = phuzzy.TruncNorm(alpha0=[2, 3], alpha1=[], number_of_alpha_levels=2, name="t")
    print(t)
    assert len(t.df) == 2

    p = phuzzy.Trapezoid(alpha0=[0, 4], alpha1=[2, 3], number_of_alpha_levels=2, name="p")
    print(p)
    assert len(p.df) == 2

    a = t / p
    # a = p / t
    a.name = "t/p"
    print(a.df.values.tolist())
    assert np.allclose(a.df.values.tolist(),
                       [[0.0, 3.3333333333333335e-11, 2.0], [1.0, 0.8, 1.2]]
                       )

    print(a)
    print(a.df)

    b = t / 3.
    print(b)
    print(b.df)
    print(b.df.values.tolist())
    assert np.allclose(b.df.values.tolist(),
                       [[0.0, 0.6666666666666666, 1.0], [0.3333333333333333, 0.8333333333333334, 0.8333333333333334]]
                       )

def test_power():
    t = phuzzy.TruncNorm(alpha0=[2, 3], alpha1=[], number_of_alpha_levels=2, name="t")
    print(t)
    assert len(t.df) == 2

    p = phuzzy.Trapezoid(alpha0=[0, 4], alpha1=[2, 3], number_of_alpha_levels=2, name="p")
    print(p)
    assert len(p.df) == 2

    a = t ** p
    a = p ** t
    a.name = "t**p"
    print(a.df.values.tolist())
    assert np.allclose(a.df.values.tolist(),
                       [[0.0, 1e-30, 64.0], [1.0, 5.656854249492381, 15.588457268119896]]
                       )

    print(a)
    print(a.df)

    b = t ** 3.
    print(b)
    print(b.df)
    print(b.df.values.tolist())
    assert np.allclose(b.df.values.tolist(),
                       [[0.0, 8.0, 27.0], [1.0, 15.625, 15.625]]
                       )

if __name__ == '__main__':
    atest_add()
