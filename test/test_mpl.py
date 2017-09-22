import phuzzy

from phuzzy.mpl import mix_mpl

def test_dyn_mix():
    p = phuzzy.Trapezoid(alpha0=[1,4], alpha1=[2,3])
    print(p)
    print(p.df)
    p.convert_df(5)
    print(p.df)
    assert not hasattr(p, "plot")
    mix_mpl(p)
    assert hasattr(p, "plot")

    p.plot(show=False)
    print(p.__class__)



    p = phuzzy.Triangle(alpha0=[1,4], alpha1=[2,3])
    print(p)
    print(p.df)
    p.convert_df(5)
    print(p.df)

    mix_mpl(p)
    p.plot(show=False)

    p = phuzzy.TruncNorm(alpha0=[0,2], alpha1=[2,3])
    print(p)
    print(p.df)
    p.convert_df(5)
    print(p.df)
    print(p.distr.ppf([.05,.5, .95]))

    mix_mpl(p)
    p.plot(show=True)
