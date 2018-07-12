# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def p_estimates(df, show=False):

    fig, ax = plt.subplots(1, 1, figsize=(10,5))
    for col in [c for c in df.columns if c.startswith("p_")]:
        ax.plot(df.x, df[col])

    ax.set_xlabel("x")
    ax.set_ylabel("p")

    ax.legend(loc="best")
    fig.tight_layout()
    if show is True:
        plt.show()

    return fig, ax

def bootstrapping(data, df_boot, show=False):

    fig, axs = plt.subplots(1, 2, figsize=(10,5))

    ax_mean = axs[0]
    ax_std = axs[1]

    ax_mean.hist(df_boot.x_mean)
    ax_std.hist(df_boot.x_std)

    ax_mean.axvline(df_boot.x_mean.mean(), color="m", lw=2, alpha=.6, dashes=[10,10])
    ax_mean.axvline(data.df.x.mean(), color="r", lw=1, alpha=.6)

    ax_std.axvline(df_boot.x_std.mean(), color="m", lw=2, alpha=.6, dashes=[10,10])
    ax_std.axvline(data.df.x.std(), color="r", lw=1, alpha=.6)


    ax_std.set_xlabel("x_std")
    ax_mean.set_xlabel("x_mean")
    ax_std.set_ylabel("frequency")
    ax_mean.set_ylabel("frequency")
    ax_std.set_title("standard deviation")
    ax_mean.set_title("mean")

    fig.tight_layout()
    if show is True:
        plt.show()

    return fig, axs
