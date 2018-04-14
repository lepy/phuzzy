import numpy as np

import phuzzy
from phuzzy.mpl import mix_mpl

def normal_estimator():
    import matplotlib.pyplot as plt
    H = 100.  # mm
    B = 300.  # mm
    fig, axs = plt.subplots(1, 2, dpi=90, facecolor='w', edgecolor='k', figsize=(B / 25.4, H / 25.4))

    alpha0 = [-3, 8]
    alpha1 = [-1]
    size= 5000
    data = np.random.triangular(alpha0[0], alpha1[0], alpha0[1], size)
    p0 = phuzzy.Triangle(name="org", alpha0=alpha0, alpha1=alpha1)
    p = phuzzy.Triangle.from_data(data=data, name="estimate")
    print(p.df)
    mix_mpl(p)
    mix_mpl(p0)
    p0.plot(ax=axs[1])
    p.plot(ax=axs[1], labels=False)

    print(data.mean())
    print(data.shape)
    p.plot(ax=axs[0])
    h = axs[0].hist(data, bins=50, normed=True, alpha=.5)
    fig.tight_layout()
    plt.show()


if __name__ == '__main__':
    normal_estimator()
