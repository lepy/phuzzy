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
from phuzzy.optimization import sensitivity_analysis
import math

import numpy as np
import pandas as pd
import xarray as xr

#from math import *




if __name__ == "__main__":

    """
    v1 = phuzzy.Triangle(alpha0=[1,4], alpha1=[2], number_of_alpha_levels=3)
    v2 = phuzzy.Triangle(alpha0=[0, 4], alpha1=[1], number_of_alpha_levels=3)
    v3 = phuzzy.Triangle(alpha0=[1, 4], alpha1=[2], number_of_alpha_levels=3)
    v4 = phuzzy.Triangle(alpha0=[0,4], alpha1=[2.5], number_of_alpha_levels=6)
    v5 = phuzzy.Triangle(alpha0=[1, 4], alpha1=[3], number_of_alpha_levels=3)
    v6 = phuzzy.Triangle(alpha0=[1,5], alpha1=[3], number_of_alpha_levels=3)
    obj_function = '-1*((x[0] - 1) ** 2 + (x[1] + .1) ** 2 + .1 - (x[2] + 2) ** 2 - (x[3] - 0.1) ** 2 - (x[4] * x[5]) ** 2)'
    kwargs = {'var6': v6, 'var2': v2, 'var3': v3,
              'var4': v4,'var5': v5, 'var1': v1,
              'obj_function': obj_function}
    
    """

    """
    v1 = phuzzy.Uniform(alpha0=[-3.14159265359,3.14159265359], number_of_alpha_levels=5)
    v2 = phuzzy.Uniform(alpha0=[-3.14159265359,3.14159265359], number_of_alpha_levels=5)
    v3 = phuzzy.Uniform(alpha0=[-3.14159265359,3.14159265359], number_of_alpha_levels=5)

    obj_function = 'sin(x[0]) + 7*(asin(x[1])**2) + 0.1*(x[2]**4)*sin(x[0])'

    kwargs = {'var1': v1, 'var2': v2, 'var3': v3,
              'obj_function': obj_function}
    """

    """
    #z = alphaOpt.Alpha_Level_Optimization(**kwargs)
    #z.calculation(progressbar_disable=True)
    #z.plot()
    #plt.show()


    rr = sensitivity_analysis.Fuzzy_Sensitivity_Analysis(**kwargs)
    rr.lcefa(error=False, **kwargs)
    rr.barchart_plot()
    plt.show()

    """
