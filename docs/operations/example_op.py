# -*- coding: utf-8 -*-
import phuzzy
import phuzzy.mpl as phm
import matplotlib.pyplot as plt

def plot_xy(x, y, show=False, ftype=True):
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

x = phm.Triangle(alpha0=[-1, 2], alpha1=[1], name="x")
plot_xy(x,-x, show=False)

x = phm.Trapezoid(alpha0=[-3, 5], alpha1=[0,1], name="x")
plot_xy(x,-x, show=False)

x = phm.Trapezoid(alpha0=[-3, 5], alpha1=[1,2], name="x")
plot_xy(x,abs(x), show=False)

x = phm.Trapezoid(alpha0=[1, 5], alpha1=[2,3], name="x")
plot_xy(x,abs(x), show=False)

x = phm.Trapezoid(alpha0=[-5, -1], alpha1=[-3,-2], name="x")
plot_xy(x,abs(x), show=True)

# plot_xy(x,x-2, show=False)
# plot_xy(x,x*2, show=False)
# plot_xy(x,x/2, show=False)
# plot_xy(x,x**2, show=True)
#
