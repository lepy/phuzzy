import phuzzy
from phuzzy.mpl import mix_mpl


def test_defuzzification():
    p = phuzzy.Trapezoid(alpha0=[-1, 6], alpha1=[1, 5], number_of_alpha_levels=5)**2
    print(p)
    x = p.defuzzification(method='alpha_one')
    print(x)

if __name__ == '__main__':
    test_defuzzification()
