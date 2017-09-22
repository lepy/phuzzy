import phuzzy
from phuzzy.mpl import mix_mpl


def test_fuzzy():


    t = phuzzy.TruncNorm(alpha0=[1,3], alpha1=[2], number_of_alpha_levels=15)
    print(t)
    assert len(t.df)==15

    p = phuzzy.Trapezoid(alpha0=[1,4], alpha1=[2,3], number_of_alpha_levels=5)
    print(p)
    assert len(p.df)==5

    a = t + p

    print(a)
    print(a.df)
    mix_mpl(t)
    mix_mpl(p)
    mix_mpl(a)
    t.plot()
    p.plot()
    a.plot(show=True)

