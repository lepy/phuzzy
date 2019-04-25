# -*- coding: utf-8 -*-

from shgo._shgo import SHGO

import datetime
import os
import subprocess

import phuzzy
from phuzzy.mpl import MPL_Mixin
from phuzzy.shapes import FuzzyNumber

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.rcsetup as rcsetup
from phuzzy.optimization import alphaOpt
from phuzzy.optimization import sensitivity_analysis

from phuzzy.fuzzification import data_fitting
import math

import numpy as np
import pandas as pd
import xarray as xr

#from math import *





if __name__ == "__main__":

    link = 'C:\\Users\\boos\\Desktop\\histotest.csv'

    kwargs = {'name': 'test', 'input_link': link}

    fuzzy_variable = data_fitting.New_Data_Fitting()
    fuzzy_variable.create_fuzzy_variable(input_link=link)
    histogram_df = fuzzy_variable.histogram_df
    fuzzy_variable_i = fuzzy_variable.fuzzy_variable

    fig, ax = plt.subplots()
    binsSize = histogram_df['widthR'].values[0]-histogram_df['widthL'].values[0]
    ax.bar(x=histogram_df['center'].values.tolist(),height=histogram_df['absBinFrequencyNorm'].values.tolist(),
           width=binsSize.tolist(),align='center',color='royalblue')
    plt.show()


    r = 1
