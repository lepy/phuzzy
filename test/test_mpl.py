# -*- coding: utf-8 -*-
import phuzzy

from phuzzy.mpl import mix_mpl
import matplotlib.pyplot as plt
import numpy as np


def test_dyn_mix():
    p = phuzzy.Trapezoid(alpha0=[1, 4], alpha1=[2, 3], number_of_alpha_levels=5)
    print(p)
    print(p.df)
    p.convert_df(5)
    print(p.df)
    assert not hasattr(p, "plot")
    mix_mpl(p)
    assert hasattr(p, "plot")


def plot():

    p = phuzzy.Trapezoid(alpha0=[1, 4], alpha1=[2, 3], number_of_alpha_levels=5)
    print(p)
    print(p.df)
    p.convert_df(5)
    print(p.df)
    assert not hasattr(p, "plot")
    mix_mpl(p)
    assert hasattr(p, "plot")

    p.plot(show=False, filepath="trapezoid.png", title=True)
    print(p.__class__)

    p = phuzzy.Triangle(alpha0=[1, 4], alpha1=[2, 3], number_of_alpha_levels=5)
    print(p)
    print(p.df)
    p.convert_df(5)
    print(p.df)

    mix_mpl(p)
    p.plot(show=False, filepath="triangle.png", title=True)

    # p = phuzzy.TruncNorm(alpha0=[0,2], alpha1=[2,3])
    p = phuzzy.TruncNorm(alpha0=[1, 3], number_of_alpha_levels=15, name="x")

    print(p)
    print(p.df)
    print(p.distr.ppf([.05, .5, .95]))

    mix_mpl(p)
    p.plot(show=False, filepath="truncnorm.png", title=True)

    p = phuzzy.Uniform(alpha0=[1, 4], number_of_alpha_levels=5, name="x")

    print(p)
    print(p.df)
    mix_mpl(p)
    p.plot(show=True, filepath="uniform.png", title=True)


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
    fig.savefig("x^y.png")
    plt.show()


def plot_pow2():
    H = 300.  # mm
    B = 300.  # mm
    fig, axs = plt.subplots(3, 3, dpi=90, facecolor='w', edgecolor='k', figsize=(B / 25.4, H / 25.4))
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

    b = np.linspace(0, 5, 200)

    px = x.pdf(b)
    Px = x.cdf(b)
    axs[1, 0].plot(b, px, label="pdf", lw=2)
    axs[2, 0].plot(b, Px, label="cdf", lw=2)

    py = y.pdf(b)
    Py = y.cdf(b)
    axs[1, 1].plot(b, py, label="pdf", lw=2)
    axs[2, 1].plot(b, Py, label="cdf", lw=2)

    b = np.linspace(z.alpha0["low"], z.alpha0["high"], 200)
    pz = z.pdf(b)
    Pz = z.cdf(b)
    axs[1, 2].plot(b, pz, label="pdf", lw=2)
    axs[2, 2].plot(b, Pz, label="cdf", lw=2)

    fig.tight_layout()
    fig.savefig("x^y.png")
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
    assert not hasattr(p, "plot")
    mix_mpl(p)
    mix_mpl(p2)
    mix_mpl(p10)

    assert hasattr(p, "plot")
    H = 100.  # mm
    B = 300.  # mm
    fig, axs = plt.subplots(1, 3, dpi=90, facecolor='w', edgecolor='k', figsize=(B / 25.4, H / 25.4))
    p.plot(ax=axs[0])
    p2.plot(ax=axs[1])
    p10.plot(ax=axs[2])
    fig.tight_layout()
    # fig.savefig("x^y.png")
    plt.show()


if __name__ == '__main__':
    # plot_add()
    # plot_sub()
    # plot_mul()
    # plot_div()
    # plot_pow()
    # plot_pow2()
    plot_gennorm_mix()
