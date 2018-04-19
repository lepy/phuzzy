# -*- coding: utf-8 -*-
import phuzzy
import phuzzy.mpl as phm
import matplotlib.pyplot as plt

def plot_op_x(x, show=False, ftype=True):
    H = 250.  # mm
    B = 300.  # mm

    fig, axs = plt.subplots(3, 3, dpi=90, facecolor='w', edgecolor='k', figsize=(B / 25.4, H / 25.4))

    ys = [x, -x, abs(x),
          x+2, 2+x, x-2,
          x*2, x/2, x**2]

    ys[4].name = "2+x"

    for i, y in enumerate(ys):
        phm.mix_mpl(y)
        ax = axs.ravel()[i]
        y.plot(ax=ax)
        if ftype is True:
            ax.set_title(y.__class__.__name__)

    fig.tight_layout()
    fig.savefig("operations_%s.png" % x.__class__.__name__)
    if show==True:
        plt.show()

x = phm.Triangle(alpha0=[-1, 2], alpha1=[1], name="x")
plot_op_x(x, show=False)

x = phm.Uniform(alpha0=[-2, 3], alpha1=[1], name="x")
plot_op_x(x, show=False)

x = phm.Trapezoid(alpha0=[-2, 3], alpha1=[1,2], name="x")
plot_op_x(x, show=False)

x = phm.TruncNorm(alpha0=[-2, 3], alpha1=[1,2], name="x")
plot_op_x(x, show=False)

x = phm.TruncGenNorm(alpha0=[-2, 3], alpha1=[1,2], name="x", beta=1, number_of_alpha_levels=21)
plot_op_x(x, show=False)

x = phm.Superellipse(alpha0=[-2, 6], alpha1=[1,2], name="x", m=2, n=.5, number_of_alpha_levels=21)
plot_op_x(x, show=True)
