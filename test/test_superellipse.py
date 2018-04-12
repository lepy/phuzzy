import matplotlib.pyplot as plt

import numpy as np
import phuzzy
from phuzzy.mpl import Superellipse

def test_superellipse():

    alpha0 = [-1,2]
    # alpha1 = [2]

    p = Superellipse(alpha0=alpha0, alpha1=None, m=3, n=1, number_of_alpha_levels=7)
    print(p)
    print(p.df)
    # print(p.distr.ppf([.05,.5, .95]))
    # pp = np.linspace(0,1,5)
    # ppf = p.distr.ppf(pp)
    # x = np.linspace(alpha0[0],alpha0[1],5)
    # # x = np.linspace(-2, 2,5)
    # pdf = p.distr.pdf(x)
    # print(p.distr.mean(), p.distr.std())
    # print("x", x)
    # print("ppf", ppf)
    # print("pdf", pdf)
    print(p.df.values.tolist())
    # ref = [[0.0, 0.0, 2.0], [0.16666666666666666, 0.36898774621220265, 1.6310122537877976], [0.3333333333333333, 0.505893899432985, 1.4941061005670149], [0.5, 0.6075291624853785, 1.3924708375146215], [0.6666666666666666, 0.6998279866511387, 1.3001720133488615], [0.8333333333333333, 0.7987198538648325, 1.2012801461351676], [1.0, 1.0, 1.0]]

    # assert np.allclose(p.df.values.tolist(), ref)


    p = Superellipse(alpha0=alpha0, alpha1=None, m=2, n=None, number_of_alpha_levels=17)
    p.plot(show=True)
    plt.show()
