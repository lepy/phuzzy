# -*- coding: utf-8 -*-

from shgo._shgo import SHGO

import datetime
import os
import subprocess

import phuzzy
from phuzzy.mpl import MPL_Mixin
from phuzzy.shapes import FuzzyNumber

import matplotlib.pyplot as plt
from phuzzy.optimization import alphaOpt

import numpy as np
import pandas as pd
import xarray as xr

#from math import *





if __name__ == "__main__":



    v1 = phuzzy.Triangle(alpha0=[0,4], alpha1=[1], number_of_alpha_levels=4)
    v2 = phuzzy.Superellipse(alpha0=[-1, 2.], alpha1=None, m=1.0, n=.5, number_of_alpha_levels=4)
    v3 = phuzzy.TruncGenNorm(alpha0=[1, 4], alpha1=[2, 3], number_of_alpha_levels=4, beta=3.)
    v4 = phuzzy.Trapezoid(alpha0=[0, 4], alpha1=[2, 3], number_of_alpha_levels=8)
    v5 = phuzzy.TruncNorm(alpha0=[1, 3], number_of_alpha_levels=4, name="y")
    v6 = phuzzy.Triangle(alpha0=[1,4], alpha1=[3], number_of_alpha_levels=4)

    obj_function = '-1*((x[0] - 1) ** 2 + (x[1] + .1) ** 2 + .1 - (x[2] + 2) ** 2 - (x[3] - 0.1) ** 2 - (x[4] * x[5]) ** 2)'

    kwargs = {'var1': v1, 'var2': v2, 'var3': v3,
              'var4': v4,'var5': v5, 'var6': v6,
              'obj_function': obj_function}


    z = alphaOpt.Alpha_Level_Optimization(**kwargs)
    z.calculation()

    z.plot()
    plt.show()
