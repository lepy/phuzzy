# -*- coding: utf-8 -*-
from shgo._shgo import SHGO
from scipy.optimize import minimize

import phuzzy
from phuzzy.mpl import MPL_Mixin
from phuzzy.shapes import FuzzyNumber

from asteval import Interpreter
from pathlib import Path
#from tqdm import tqdm

import numpy as np
import pandas as pd
import xarray as xr
from phuzzy.optimization import alphaOpt, sensitivity_analysis

from SALib.sample import saltelli
from SALib.analyze import sobol

#from math import *





if __name__ == "__main__":


    v1 = phuzzy.Triangle(alpha0=[1,4], alpha1=[2], name='v1', number_of_alpha_levels=3)
    v2 = phuzzy.Triangle(alpha0=[0, 4], alpha1=[1], name='v2', number_of_alpha_levels=5)
    v3 = phuzzy.Triangle(alpha0=[1, 4], alpha1=[2], name='v3', number_of_alpha_levels=3)
    v4 = phuzzy.Triangle(alpha0=[0,4], alpha1=[2.5], name='v4', number_of_alpha_levels=3)
    v5 = phuzzy.Triangle(alpha0=[1, 4], alpha1=[3], name='v5', number_of_alpha_levels=3)
    v6 = phuzzy.Triangle(alpha0=[1,5], alpha1=[3], name='v6', number_of_alpha_levels=3)

    input = [v1,v2,v3,v4,v5,v6]
    obj_function = '-1*((x[0] - 1) ** 2 + (x[1] + .1) ** 2 + .1 - (x[2] + 2) ** 2 - (x[3] - 0.1) ** 2 - (x[4] * x[5]) ** 2)'

    kwargs = {'fuzzy_variables': input,'obj_function': obj_function}


    #z = alphaOpt.Alpha_Level_Optimization(**kwargs)
    #z.calculation()

    p = phuzzy.Triangle(alpha0=[1,5], alpha1=[3],
                         number_of_alpha_levels=11, name='x')

    rr = sensitivity_analysis.Fuzzy_Sensitivity_Analysis(**kwargs)
    rr.calculation(**kwargs)
    rr.barchart_plot(plot=True)

