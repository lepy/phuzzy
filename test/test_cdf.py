import phuzzy
import numpy as np


def test_trapz():

    x = np.linspace(-2, 5, 11)

    p = phuzzy.Trapezoid(alpha0=[1,4], alpha1=[2,3], number_of_alpha_levels=5)
    P = p.cdf(x)
    print(P)

def plot_traz_cdf():

    import matplotlib.pyplot as plt
    H = 100.  # mm
    B = 300.  # mm
    fig, axs = plt.subplots(1, 2, dpi=90, facecolor='w', edgecolor='k', figsize=(B / 25.4, H / 25.4))

    x = np.linspace(0, 6, 200)

    f = phuzzy.Trapezoid(alpha0=[1,5], alpha1=[2,3], number_of_alpha_levels=5)
    p = f.pdf(x)
    P = f.cdf(x)
    axs[0].plot(x, p, label="pdf", lw=2)
    axs[1].plot(x, P, label="cdf", lw=2)
    axs[0].legend(loc="best")
    axs[1].legend(loc="best")
    plt.show()

def plot_tria_cdf():

    import matplotlib.pyplot as plt
    H = 100.  # mm
    B = 300.  # mm
    fig, axs = plt.subplots(1, 2, dpi=90, facecolor='w', edgecolor='k', figsize=(B / 25.4, H / 25.4))

    x = np.linspace(0, 6, 200)

    f = phuzzy.Triangle(alpha0=[1,4], alpha1=[2], number_of_alpha_levels=5)
    p = f.pdf(x)
    P = f.cdf(x)
    axs[0].plot(x, p, label="pdf", lw=2)
    axs[1].plot(x, P, label="cdf", lw=2)
    axs[0].legend(loc="best")
    axs[1].legend(loc="best")
    plt.show()

def plot_uniform_cdf():

    import matplotlib.pyplot as plt
    H = 100.  # mm
    B = 300.  # mm
    fig, axs = plt.subplots(1, 2, dpi=90, facecolor='w', edgecolor='k', figsize=(B / 25.4, H / 25.4))

    x = np.linspace(0, 6, 200)

    f = phuzzy.Uniform(alpha0=[1,4], number_of_alpha_levels=5)
    p = f.pdf(x)
    P = f.cdf(x)
    axs[0].plot(x, p, label="pdf", lw=2)
    axs[1].plot(x, P, label="cdf", lw=2)
    axs[0].legend(loc="best")
    axs[1].legend(loc="best")
    plt.show()

if __name__ == '__main__':
    # plot_tria_cdf()
    plot_uniform_cdf()