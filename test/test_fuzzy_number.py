import phuzzy

def test_fuzzy():

    n = phuzzy.FuzzyNumber()
    print(n)
    print(n.__class__.__name__)

def test_triangle():
    t = phuzzy.Triangle(alpha0=[1,3], alpha1=[2], number_of_alpha_levels=15)
    print(t)
    print(t.__class__.__name__)
    print(t.df)
    assert len(t.df)==15
    print([t])
    print(t.get_01())

def test_trapezoid():
    p = phuzzy.Trapezoid(alpha0=[1,4], alpha1=[2,3], number_of_alpha_levels=5)
    print(p)
    print("number_of_alpha_levels", p.number_of_alpha_levels)
    print(p.df)
    print(len(p.df))
    assert len(p.df)==5
    print(p.get_01())

def test_uniform():
    p = phuzzy.Uniform(alpha0=[1,4], number_of_alpha_levels=5)
    print(p.alpha0)
    print(p.df)
    print(p.to_str())
    print(p.get_01())

if __name__ == '__main__':
    test_fuzzy()