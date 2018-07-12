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

    phuzzy.data.plots.bootstrapping(data, df_boot, show=True)


def test_shuffling():

    raw_data = [84, 81, 72, 69, 61, 69, 74, 57, 65, 76, 56, 87, 99, 44, 46, 63]
    # raw_data = [84, 81, 72, 46, 63]


    data = phuzzy.data.Data(raw_data)
    # print(data.df)
    df_boot = data.shuffling(n=1000, train_fraction=.7)
    print(df_boot.head())

    phuzzy.data.plots.bootstrapping(data, df_boot, show=True)

def test_estimate_probability():
    raw_data = [84, 81, 72, 69, 61, 69, 74, 57, 45, 65, 76, 56, 87, 99, 44, 46, 63]
    # raw_data = [84, 81, 72, 46, 63]

    data = phuzzy.data.Data(raw_data)
    df = data.estimate_probability()
    print(df)



    phuzzy.data.plots.p_estimates(df, show=True)
