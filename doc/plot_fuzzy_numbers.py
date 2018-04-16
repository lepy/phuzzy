# -*- coding: utf-8 -*-
import phuzzy

from phuzzy.mpl import mix_mpl
import matplotlib.pyplot as plt
import numpy as np

import phuzzy.mpl as phm


def plot():

    uni = phm.Uniform(alpha0=[1, 4], number_of_alpha_levels=5, name="x")
    uni.plot(show=True, filepath="~/tmp/uniform.png", title=True)
    print(uni)
    print(uni.df)


    tri = phm.Triangle(alpha0=[1, 4], alpha1=[2], number_of_alpha_levels=5)
    tri.plot(show=False, filepath="~/tmp/triangle.png", title=True)
    print(tri)
    print(tri.df)

    trap = phm.Trapezoid(alpha0=[1, 5], alpha1=[2, 3], number_of_alpha_levels=5)
    trap.plot(show=False, filepath="~/tmp/trapezoid.png", title=True)
    print(trap)
    print(trap.df)

    tn = phm.TruncNorm(alpha0=[1, 3], number_of_alpha_levels=15, name="x")
    tn.plot(show=False, filepath="~/tmp/truncnorm.png", title=True)
    print(tn)
    print(tn.df)
    print(tn.distr.ppf([.05, .5, .95]))

    tgn = phm.TruncGenNorm(alpha0=[1, 4], alpha1=[2, 3], number_of_alpha_levels=15, beta=3.)
    tgn.plot(show=False, filepath="~/tmp/truncgennorm.png", title=True)
    print(tgn)
    print(tgn.df)
    print(tgn.distr.ppf([.05, .5, .95]))

    se = phm.Superellipse(alpha0=[-1, 2.], alpha1=None, m=1.0, n=.5, number_of_alpha_levels=15)
    se.plot(show=True, filepath="~/tmp/superellipse.png", title=True)
    print(se)
    print(se.df)


def plot_add():
    print("plot_add")
    H = 100.  # mm
    B = 300.  # mm
    fig, axs = plt.subplots(1, 3, dpi=90, facecolor='w', edgecolor='k', figsize=(B / 25.4, H / 25.4))
    x = phuzzy.Trapezoid(alpha0=[0, 4], alpha1=[2, 3], number_of_alpha_levels=5)
    mix_mpl(x)
    x.plot(ax=axs[0])

    y = phuzzy.TruncNorm(alpha0=[1, 3], number_of_alpha_levels=15, name="y")
    mix_mpl(y)
    y.plot(ax=axs[1])

    z = x + y
    z.name = "x+y"
    mix_mpl(z)
    z.plot(ax=axs[2])

    fig.tight_layout()
    fig.savefig("x+y.png")
    plt.show()


def plot_sub():
    H = 100.  # mm
    B = 300.  # mm
    fig, axs = plt.subplots(1, 3, dpi=90, facecolor='w', edgecolor='k', figsize=(B / 25.4, H / 25.4))
    x = phuzzy.Trapezoid(alpha0=[0, 4], alpha1=[2, 3], number_of_alpha_levels=5)
    mix_mpl(x)
    x.plot(ax=axs[0])

    y = phuzzy.TruncNorm(alpha0=[1, 3], number_of_alpha_levels=15, name="y")
    mix_mpl(y)
    y.plot(ax=axs[1])

    z = x - y
    z.name = "x-y"
    mix_mpl(z)
    z.plot(ax=axs[2])

    fig.tight_layout()
    fig.savefig("x-y.png")
    plt.show()


def plot_mul():
    H = 100.  # mm
    B = 300.  # mm
    fig, axs = plt.subplots(1, 3, dpi=90, facecolor='w', edgecolor='k', figsize=(B / 25.4, H / 25.4))
    x = phuzzy.Trapezoid(alpha0=[0, 4], alpha1=[2, 3], number_of_alpha_levels=5)
    mix_mpl(x)
    x.plot(ax=axs[0])

    y = phuzzy.TruncNorm(alpha0=[1, 3], number_of_alpha_levels=15, name="y")
    mix_mpl(y)
    y.plot(ax=axs[1])

    z = x * y
    z.name = "x*y"
    mix_mpl(z)
    z.plot(ax=axs[2])

    fig.tight_layout()
    fig.savefig("x*y.png")
    plt.show()


def plot_div():
    H = 100.  # mm
    B = 300.  # mm
    fig, axs = plt.subplots(1, 3, dpi=90, facecolor='w', edgecolor='k', figsize=(B / 25.4, H / 25.4))
    x = phuzzy.Trapezoid(alpha0=[0, 4], alpha1=[2, 3], number_of_alpha_levels=5)
    mix_mpl(x)
    x.plot(ax=axs[0])

    y = phuzzy.TruncNorm(alpha0=[1, 3], number_of_alpha_levels=15, name="y")
    mix_mpl(y)
    y.plot(ax=axs[1])

    z = x / y
    z.name = "x/y"
    mix_mpl(z)
    z.plot(ax=axs[2])

    fig.tight_layout()
    fig.savefig("x:y.png")
    plt.show()


def plot_pow():
    H = 100.  # mm
    B = 300.  # mm
    fig, axs = plt.subplots(1, 3, dpi=90, facecolor='w', edgecolor='k', figsize=(B / 25.4, H / 25.4))
    x = phuzzy.Trapezoid(alpha0=[0, 4], alpha1=[2, 3], number_of_alpha_levels=5)
    mix_mpl(x)
    x.plot(ax=axs[0])

    y = phuzzy.TruncNorm(alpha0=[1, 3], number_of_alpha_levels=15, name="y")
    mix_mpl(y)
    y.plot(ax=axs[1])

    z = x ** y
    z.name = "x^y"
    mix_mpl(z)
    z.plot(ax=axs[2])

    fig.tight_layout()
    fig.savefig("x**y.png")
    plt.show()


def plot_pow2():
    H = 300.  # mm
    B = 300.  # mm
    fig, axs = plt.subplots(3, 3, dpi=90, facecolor='w', edgecolor='k', figsize=(B / 25.4, H / 25.4))

    for i in range(3):
        axs[0,i].get_shared_x_axes().join(axs[0,i],axs[1,i],axs[2,i])
    for i in range(1,3):
        axs[i,0].set_xlabel("x")
        axs[i,1].set_xlabel("y")
        axs[i,2].set_xlabel("x^y")

    i=2
    axs[i,0].get_shared_y_axes().join(axs[i,0],axs[i,1],axs[i,2])

    x = phuzzy.Trapezoid(alpha0=[0, 4], alpha1=[2, 3], number_of_alpha_levels=5)
    mix_mpl(x)
    x.plot(ax=axs[0, 0])

    y = phuzzy.TruncNorm(alpha0=[1, 3], number_of_alpha_levels=15, name="y")
    mix_mpl(y)
    y.plot(ax=axs[0, 1])

    z = x ** y
    z.name = "x^y"
    mix_mpl(z)
    z.plot(ax=axs[0, 2])

    b = np.linspace(-1, 5, 200)

    px = x.pdf(b)
    Px = x.cdf(b)
    axs[1, 0].fill_between(b, 0, px, alpha=.2)
    axs[1, 0].plot(b, px, label="pdf", lw=1)
    axs[2, 0].fill_between(b, 0, Px, alpha=.2)
    axs[2, 0].plot(b, Px, label="cdf", lw=1)

    py = y.pdf(b)
    Py = y.cdf(b)
    axs[1, 1].fill_between(b, 0, py, alpha=.2)
    axs[1, 1].plot(b, py, label="pdf", lw=1)
    axs[2, 1].fill_between(b, 0, Py, alpha=.2)
    axs[2, 1].plot(b, Py, label="cdf", lw=1)

    b = np.linspace(z.alpha0["low"], z.alpha0["high"], 200)
    pz = z.pdf(b)
    Pz = z.cdf(b)
    axs[1, 2].fill_between(b, 0, pz, alpha=.2)
    axs[1, 2].plot(b, pz, label="pdf", lw=1)
    axs[2, 2].fill_between(b, 0, Pz, alpha=.2)
    axs[2, 2].plot(b, Pz, label="cdf", lw=1)

    for i in range(3):
        axs[2,i].axhline(1, alpha=.4, c="k", lw=.5)
        # axs[1,i].set_ylabel("pdf")
        # axs[2,i].set_ylabel("cdf")
        axs[1,i].annotate('pdf', (0.01, 0.99), xycoords='axes fraction', size=8, ha='left', va='top', textcoords='axes fraction')
        axs[2,i].annotate('cdf', (0.01, 0.99), xycoords='axes fraction', size=8, ha='left', va='top', textcoords='axes fraction')

    # axs[1,1].sharex = axs[0,1]
    # axs[2,1].sharex = axs[0,1]
    # axs[1,1].share_x_axes(axs[0,1])

    for ax in axs.ravel():
        ax.set_ylim(0, None)

    fig.tight_layout()
    fig.savefig("x**y_pdf_cdf.png")
    plt.show()


def plot_gennorm_mix():
    n = 50
    p = phuzzy.TruncGenNorm(alpha0=[1, 4], alpha1=[2, 3], number_of_alpha_levels=n, beta=1)
    p2 = phuzzy.TruncGenNorm(alpha0=[1, 4], alpha1=[2, 3], number_of_alpha_levels=n, beta=2)
    p10 = phuzzy.TruncGenNorm(alpha0=[1, 4], alpha1=[2, 3], number_of_alpha_levels=n, beta=5)
    print(p)
    # print(p.df)
    # # p.convert_df(5)
    # print(p.df)

    mix_mpl(p)
    mix_mpl(p2)
    mix_mpl(p10)


    H = 100.  # mm
    B = 300.  # mm
    fig, axs = plt.subplots(1, 3, dpi=90, facecolor='w', edgecolor='k', figsize=(B / 25.4, H / 25.4))
    p.plot(ax=axs[0])
    p2.plot(ax=axs[1])
    p10.plot(ax=axs[2])
    ps = [p, p2, p10]
    for p, ax in zip(ps, axs):
        ax.set_title(r"$\beta=%.1f$" % p.beta)
    fig.tight_layout()
    fig.savefig("truncgennorm.png")
    plt.show()

def plot_superellipse():
    alpha0 = [-1, 2]

    H = 250.  # mm
    B = 300.  # mm
    fig, axs = plt.subplots(3, 3, dpi=90, facecolor='w', edgecolor='k', figsize=(B / 25.4, H / 25.4))
    ps = []
    for m in [.5, 1, 2]:
        for n in [.5, 1, 2]:
            p = phm.Superellipse(alpha0=alpha0, alpha1=None, m=m, n=n, number_of_alpha_levels=17)
            ps.append(p)

    for p, ax in zip(ps, axs.ravel()):
        p.plot(ax=ax)
        ax.set_title(r"$m=%.1f$, $n=%.1f$" % (p.m, p.n))
    fig.tight_layout()
    fig.savefig("/tmp/superellipse_var.png")
    plt.show()


    p.plot(show=True)
    plt.show()

if __name__ == '__main__':
    # plot_add()
    # plot_sub()
    # plot_mul()
    # plot_div()
    # plot_pow()
    # plot_pow2()
    # plot_gennorm_mix()
    plot_superellipse()
    # plot()
