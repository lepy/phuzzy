import phuzzy

def test_fuzzy():

    n = phuzzy.Fuzzy_Number()
    print(n)
    print(n.__class__.__name__)

    t = phuzzy.Triangle(alpha0=[1,3], alpha1=[2], number_of_alpha_levels=15)
    print(t)
    print(t.__class__.__name__)
    print(t.df)
    assert len(t.df)==15

    p = phuzzy.Trapezoid(alpha0=[1,4], alpha1=[2,3], number_of_alpha_levels=5)
    print(p)
    print("number_of_alpha_levels", p.number_of_alpha_levels)
    print(p.df)
    print(len(p.df))
    assert len(p.df)==5
