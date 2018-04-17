# -*- coding: utf-8 -*-
import phuzzy
import phuzzy.mpl as phm
import matplotlib.pyplot as plt

w = phm.Triangle(alpha0=[2.9, 3.1], alpha1=[3], name="width")
h = phm.Trapezoid(alpha0=[4.5, 5.5], alpha1=[4.8, 5.2], name="height")

A = w * h
A.name = "area"

# phm.mix_mpl(A)
# A.plot(show=True)


def plot_mul(x,y,z):
    H = 100.  # mm
    B = 300.  # mm
    for i in [x,y,z]:
        phm.mix_mpl(i)

    fig, axs = plt.subplots(1, 3, dpi=90, facecolor='w', edgecolor='k', figsize=(B / 25.4, H / 25.4))
    x.plot(ax=axs[0])
    y.plot(ax=axs[1])
    z.plot(ax=axs[2])

    fig.tight_layout()
    # fig.savefig("A.png")
    plt.show()

plot_mul(w, h, A)
