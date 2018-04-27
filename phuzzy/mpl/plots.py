# -*- coding: utf-8 -*-

import numpy as np
from phuzzy.mpl import mix_mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import mpl_toolkits.mplot3d.art3d as art3d

def plot_xy(x, y, height=100, width=200):
    """plot two fuzzy numbers

    :param x: first fuzzy number
    :param y: second fuzzy number
    :param height: figure height
    :param width: figure width
    :return: fig, axs (1x2)
    """

    if not hasattr(x, "plot"):
        x = x.copy()
        mix_mpl(x)
    if not hasattr(y, "plot"):
        y = y.copy()
        mix_mpl(y)

    fig, axs = plt.subplots(1, 2, dpi=90, facecolor='w', edgecolor='k', figsize=(width / 25.4, height / 25.4))

    x.plot(ax=axs[0])
    y.plot(ax=axs[1])
    fig.tight_layout()

    return fig, axs

def plot_xyz(x, y, z, height=70, width=200):
    """plot two fuzzy numbers

    :param x: first fuzzy number
    :param y: second fuzzy number
    :param height: figure height
    :param width: figure width
    :return: fig, axs (1x2)
    """

    if not hasattr(x, "plot"):
        x = x.copy()
        mix_mpl(x)
    if not hasattr(y, "plot"):
        y = y.copy()
        mix_mpl(y)
    if not hasattr(z, "plot"):
        z = z.copy()
        mix_mpl(z)

    fig, axs = plt.subplots(1, 3, dpi=90, facecolor='w', edgecolor='k', figsize=(width / 25.4, height / 25.4))

    x.plot(ax=axs[0])
    y.plot(ax=axs[1])
    z.plot(ax=axs[2])
    fig.tight_layout()

    return fig, axs

def plot_xy_3d(x, y, height=200, width=200):
    """plot two fuzzy numbers

    :param x: first fuzzy number
    :param y: second fuzzy number
    :return: fig, axs (2x2)
    """

    if not hasattr(x, "plot"):
        x = x.copy()
        mix_mpl(x)
    if not hasattr(y, "plot"):
        y = y.copy()
        mix_mpl(y)

    fig, axs = plt.subplots(2, 2, dpi=90, facecolor='w', edgecolor='k', figsize=(width / 25.4, height / 25.4))

    axx = axs[0, 0]
    axy = axs[1, 1]
    axxy = axs[1, 0]
    ax3d_ = axs[0, 1]

    axxy.set_xlabel("x")
    axxy.set_ylabel("y")

    axx.get_shared_x_axes().join(axx, axxy)
    axy.get_shared_y_axes().join(axy, axxy)

    x.plot(ax=axx)
    y.vplot(ax=axy)

    for i, xi in x.df.iterrows():
        yi = y.df.loc[i]

        axxy.plot([xi.l, xi.r, xi.r, xi.l, xi.l],
                  [yi.l, yi.l, yi.r, yi.r, yi.l],
                  c="k", alpha=.5, lw=.5, dashes=[2, 2])


    fig.tight_layout()
    bbox = ax3d_.get_position()
    ax3d = fig.gca(projection='3d')
    ax3d.set_position(bbox)
    ax3d_.axis('off')
    ax3d = plot_3d(x, y, ax=ax3d)
    axs[0,1] = ax3d

    return fig, axs


def plot_3d(x, y, ax=None, show=False, height=200, width=200):
    """plot two fuzzy numbers

    :param x: first fuzzy number
    :param y: second fuzzy number
    :return: fig, axs (2x2)
    """
    if ax is None:
        fig = plt.figure(dpi=90, facecolor='w', edgecolor='k',
                         figsize=(width / 25.4, height / 25.4))
        ax = fig.add_subplot(111, projection='3d')
    else:
        fig = None

    for i, xi in x.df.iterrows():
        yi = y.df.loc[i]
        polygon = Polygon(np.vstack([[xi.l, xi.r, xi.r, xi.l, xi.l],
                                     [yi.l, yi.l, yi.r, yi.r, yi.l]]).T,
                          alpha=.2)
        ax.add_patch(polygon)
        art3d.pathpatch_2d_to_3d(polygon, z=yi.alpha)

        ax.plot([xi.l, xi.r, xi.r, xi.l, xi.l],
                [yi.l, yi.l, yi.r, yi.r, yi.l],
                [yi.alpha, yi.alpha, yi.alpha, yi.alpha, yi.alpha],
                c="k", alpha=.5, lw=1.5)

    ax.plot(x.df.l, y.df.l, y.df.alpha, c="k", alpha=.5, lw=1.5)
    ax.plot(x.df.l, y.df.r, y.df.alpha, c="k", alpha=.5, lw=1.5)
    ax.plot(x.df.r, y.df.l, y.df.alpha, c="k", alpha=.5, lw=1.5)
    ax.plot(x.df.r, y.df.r, y.df.alpha, c="k", alpha=.5, lw=1.5)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel(r"$\alpha$")

    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(y.min(), y.max())
    ax.set_zlim(0, 1)

    if show is True:
        plt.show()

    return fig, ax
