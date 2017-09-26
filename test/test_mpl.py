import phuzzy

from phuzzy.mpl import mix_mpl
import matplotlib.pyplot as plt

def test_dyn_mix():
    p = phuzzy.Trapezoid(alpha0=[1,4], alpha1=[2,3], number_of_alpha_levels=5)
    print(p)
    print(p.df)
    p.convert_df(5)
    print(p.df)
    assert not hasattr(p, "plot")
    mix_mpl(p)
    assert hasattr(p, "plot")

    p.plot(show=False, filepath="trapezoid.png")
    print(p.__class__)



    p = phuzzy.Triangle(alpha0=[1,4], alpha1=[2,3], number_of_alpha_levels=5)
    print(p)
    print(p.df)
    p.convert_df(5)
    print(p.df)

    mix_mpl(p)
    p.plot(show=False, filepath="triangle.png")

    # p = phuzzy.TruncNorm(alpha0=[0,2], alpha1=[2,3])
    p = phuzzy.TruncNorm(alpha0=[1,3], number_of_alpha_levels=15, name="x")

    print(p)
    print(p.df)
    print(p.distr.ppf([.05,.5, .95]))

    mix_mpl(p)
    p.plot(show=True, filepath="truncnorm.png")

def plot_add():
    print("plot_add")
    H = 100.  # mm
    B = 300.  # mm
    fig, axs = plt.subplots(1, 3, dpi=90, facecolor='w', edgecolor='k', figsize=(B / 25.4, H / 25.4))
    x = phuzzy.Triangle(alpha0=[1,4], alpha1=[2,3], number_of_alpha_levels=5)
    mix_mpl(x)
    x.plot(ax=axs[0])

    y = phuzzy.TruncNorm(alpha0=[1,3], number_of_alpha_levels=15, name="y")
    mix_mpl(y)
    y.plot(ax=axs[1])

    z = x + y
    z.name = "x+y"
    mix_mpl(z)
    z.plot(ax=axs[2])

    fig.tight_layout()
    plt.show()

def plot_sub():
    H = 100.  # mm
    B = 300.  # mm
    fig, axs = plt.subplots(1, 3, dpi=90, facecolor='w', edgecolor='k', figsize=(B / 25.4, H / 25.4))
    x = phuzzy.Triangle(alpha0=[1,4], alpha1=[2,3], number_of_alpha_levels=5)
    mix_mpl(x)
    x.plot(ax=axs[0])

    y = phuzzy.TruncNorm(alpha0=[1,3], number_of_alpha_levels=15, name="y")
    mix_mpl(y)
    y.plot(ax=axs[1])

    z = x - y
    z.name = "x-y"
    mix_mpl(z)
    z.plot(ax=axs[2])

    fig.tight_layout()
    plt.show()

def plot_mul():
    H = 100.  # mm
    B = 300.  # mm
    fig, axs = plt.subplots(1, 3, dpi=90, facecolor='w', edgecolor='k', figsize=(B / 25.4, H / 25.4))
    x = phuzzy.Trapezoid(alpha0=[0,4], alpha1=[2,3], number_of_alpha_levels=5)
    mix_mpl(x)
    x.plot(ax=axs[0])

    y = phuzzy.TruncNorm(alpha0=[1,3], number_of_alpha_levels=15, name="y")
    mix_mpl(y)
    y.plot(ax=axs[1])

    z = x * y
    z.name = "x*y"
    mix_mpl(z)
    z.plot(ax=axs[2])

    fig.tight_layout()
    plt.show()

def plot_div():
    H = 100.  # mm
    B = 300.  # mm
    fig, axs = plt.subplots(1, 3, dpi=90, facecolor='w', edgecolor='k', figsize=(B / 25.4, H / 25.4))
    x = phuzzy.Trapezoid(alpha0=[0,4], alpha1=[2,3], number_of_alpha_levels=5)
    mix_mpl(x)
    x.plot(ax=axs[0])

    y = phuzzy.TruncNorm(alpha0=[1,3], number_of_alpha_levels=15, name="y")
    mix_mpl(y)
    y.plot(ax=axs[1])

    z = x / y
    z.name = "x/y"
    mix_mpl(z)
    z.plot(ax=axs[2])

    fig.tight_layout()
    plt.show()

def plot_pow():
    H = 100.  # mm
    B = 300.  # mm
    fig, axs = plt.subplots(1, 3, dpi=90, facecolor='w', edgecolor='k', figsize=(B / 25.4, H / 25.4))
    x = phuzzy.Trapezoid(alpha0=[0,4], alpha1=[2,3], number_of_alpha_levels=5)
    mix_mpl(x)
    x.plot(ax=axs[0])

    y = phuzzy.TruncNorm(alpha0=[1,3], number_of_alpha_levels=15, name="y")
    mix_mpl(y)
    y.plot(ax=axs[1])

    z = x ** y
    z.name = "x^y"
    mix_mpl(z)
    z.plot(ax=axs[2])

    fig.tight_layout()
    plt.show()

if __name__ == '__main__':
    # plot_add()
    # plot_sub()
    # plot_mul()
    # plot_div()
    plot_pow()

