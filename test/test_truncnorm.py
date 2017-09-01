import matplotlib.pyplot as plt

import numpy as np
import phuzzy
from phuzzy.mpl import MPL_Mixin, Trapezoid, Triangle, TruncNorm

def test_truncnorm():

    alpha0 = [0,2]
    # alpha1 = [2]

    p = TruncNorm(alpha0=alpha0, alpha1=None, number_of_alpha_levels=21)
    print(p)
    print(p.df)
    print(p.df)
    print(p.distr.ppf([.05,.5, .95]))
    pp = np.linspace(0,1,5)
    ppf = p.distr.ppf(pp)
    x = np.linspace(alpha0[0],alpha0[1],5)
    # x = np.linspace(-2, 2,5)
    pdf = p.distr.pdf(x)
    # print(p.distr.mean(), p.distr.std())
    # print("x", x)
    # print("ppf", ppf)
    # print("pdf", pdf)
    p.plot()

    # assert 1==2
    plt.show()
