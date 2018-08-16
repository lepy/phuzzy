# -*- coding: utf-8 -*-
import os
import matplotlib.pyplot as plt

import phuzzy.data
import phuzzy.data.plots


def test_bootstrapping():
    raw_data = [84, 81, 72, 69, 61, 69, 74, 57, 65, 76, 56, 87, 99, 44, 46, 63]
    raw_data = [84, 81, 72, 46, 63]

    data = phuzzy.data.Data(raw_data)
    # print(data.df)
    df_boot = data.bootstrap(n=1000)
    print(df_boot.head())

    # phuzzy.data.plots.bootstrapping(data, df_boot, show=True)


def test_shuffling():
    raw_data = [84, 81, 72, 69, 61, 69, 74, 57, 65, 76, 56, 87, 99, 44, 46, 63]
    # raw_data = [84, 81, 72, 46, 63]

    data = phuzzy.data.Data(raw_data)
    # print(data.df)
    df_boot = data.shuffling(n=1000, train_fraction=.7)
    print(df_boot.head())

    # phuzzy.data.plots.bootstrapping(data, df_boot, show=True)


def test_estimate_probability():
    raw_data = [84, 81, 72, 69, 61, 69, 74, 57, 45, 65, 76, 56, 87, 99, 44, 46, 63]
    # raw_data = [84, 81, 72, 46, 63]

    data = phuzzy.data.Data(raw_data)
    df = data.estimate_probability()
    print(df)
    # phuzzy.data.plots.p_estimates(df, show=True)


def test_histogram():
    raw_data = [84, 81, 72, 69, 61, 69, 74, 57, 45, 65, 76, 56, 87, 99, 44, 46, 63]
    data = phuzzy.data.Data(raw_data)
    df_boot = data.bootstrap(n=1000)
    df = phuzzy.data.get_histogram_data(df_boot.x_mean)
    data2 = phuzzy.data.Data(df_boot.x_mean)
    df_esti = data2.estimate_probability()

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    ax = axs[0]
    axcdf = axs[1]

    ax.plot(df.center, df.frequency, "s-", alpha=.7)
    phuzzy.data.plots.plot_hist(df_boot.x_mean, ax=ax, alpha=.2, filled=True)

    axcdf.plot(df.center, df.cdf, "s-", label="cdf from hist")
    phuzzy.data.plots.p_estimates(df_esti, ax=axcdf)

    ax.legend(loc="best")
    ax.set_xlabel("x")
    ax.set_ylabel("frequency")
    axcdf.set_xlabel("x")
    axcdf.set_ylabel("p")
    # plt.show()
