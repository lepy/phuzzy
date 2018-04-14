import matplotlib.pyplot as plt

import numpy as np
from phuzzy.mpl import Superellipse

def test_superellipse():
    alpha0 = [-1, 2]

    p = Superellipse(alpha0=alpha0, alpha1=None, m=3, n=1, number_of_alpha_levels=7)
    print(p)
    print(p.df)
    print(p.df.values.tolist())
    ref = [[0.0, -1.0, 2.0], [0.16666666666666666, -0.9115491552521021, 1.9115491552521022],
           [0.3333333333333333, -0.8103641352538279, 1.810364135253828], [0.5, -0.6905434026749685, 1.6905434026749686],
           [0.6666666666666666, -0.5400341501766197, 1.54003415017662],
           [0.8333333333333333, -0.3254711824961655, 1.3254711824961656], [1.0, 0.5, 0.5]]

    assert np.allclose(p.df.values.tolist(), ref)

def plot():
    alpha0 = [-1, 2]

    p = Superellipse(alpha0=alpha0, alpha1=None, m=2, n=None, number_of_alpha_levels=17)
    p.plot(show=True)
    plt.show()
