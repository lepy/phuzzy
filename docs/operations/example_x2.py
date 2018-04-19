# -*- coding: utf-8 -*-
import phuzzy
import phuzzy.mpl as phm
import matplotlib.pyplot as plt

def x2(x):
    return x**2

x = phm.Triangle(alpha0=[-1, 2], alpha1=[1], name="x")

y = x2(x)
print(y)
print(y.df)


# phm.mix_mpl(A)
# A.plot(show=True)


def plot_op_x(x, show=False, ftype=True):
    H = 100.  # mm
    B = 300.  # mm
    for i in [x,y]:
        phm.mix_mpl(i)

    fig, axs = plt.subplots(1, 2, dpi=90, facecolor='w', edgecolor='k', figsize=(B / 25.4, H / 25.4))
    if ftype is True:
        axs[0].set_title(x.__class__.__name__)
        axs[1].set_title(y.__class__.__name__)
    x.plot(ax=axs[0])
    y.plot(ax=axs[1])

    fig.tight_layout()
    # fig.savefig("A.png")
    if show==True:
        plt.show()

plot_op_x(x, show=True)

