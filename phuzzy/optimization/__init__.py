# -*- coding: utf-8 -*-

from shgo._shgo import SHGO

import datetime

import os
import subprocess

import phuzzy
from phuzzy.mpl import MPL_Mixin
from phuzzy.shapes import FuzzyNumber

import matplotlib.pyplot as plt

from scipy.optimize import minimize
import numpy as np
import pandas as pd
import xarray as xr


class ObjFunction(object):
    def __init__(self):
        self.x_glob = []
        self.local = 1

        self.ob_func_link = 'C:\\Users\\boos\\Desktop\\Topo_exe_deg\\Topo.exe'

    def min_function_value(self, x):
        if isinstance(self.x_glob, np.ndarray):
            for row in self.x_glob:
                x = np.insert(x, (row[0].astype(int)), row[1])
        return self.objective_function(x)

    def max_function_value(self, x):
        if isinstance(self.x_glob, np.ndarray):
            for row in self.x_glob:
                x = np.insert(x, (row[0].astype(int)), row[1])
        return -1 * (self.objective_function(x))

    def objective_function(self, x):
        if self.local == 1:
            return -1*((x[0] - 1) ** 2 + (x[1] + .1) ** 2 + .1 - (x[2] + 2) ** 2 - (x[3] - 0.1) ** 2 - (x[4] * x[5]) ** 2)
        else:
            # external Routine
            np.savetxt('C:\\Users\\boos\\Desktop\\Topo_exe_deg\\load.txt', x, fmt='%1.5e')
            subprocess.call(self.ob_func_link, shell=0)

            f1 = open("C:\\Users\\boos\\Desktop\\Topo_exe_deg\\compliance.txt", 'r')
            c = f1.readlines()
            c = np.loadtxt(c, delimiter=',', skiprows=0)
            f1.close()
            os.remove("C:\\Users\\boos\\Desktop\\Topo_exe_deg\\compliance.txt")

            # f2 = open("C:\\Users\\boos\\Desktop\\Topo_exe_deg\\x_phys.txt", 'r')
            # x_phys = f2.readlines()
            # x_phys = np.loadtxt(x_phys, delimiter=',', skiprows=0)
            # x_phys = x_phys[:,(0,5)]
            # f2.close()
            # os.remove("C:\\Users\\boos\\Desktop\\Topo_exe_deg\\x_phys.txt")
            # self.cc.append(c)
            # self.xphys.append(x_phys)
            return c


class Constraints(object):
    def __init__(self, bound=True, **kwargs):
        """
        Optimization Constraints
        """
        if bound == True:
            self.boundary_constraints(**kwargs)

    def boundary_constraints(self, **kwargs):

        filter_fuzzy_variables_dict = {}
        list_of_n_alpha_levels = np.zeros(1, dtype='int')

        # check if number_of_alpha_levels is the same
        for key, value in kwargs.items():
            if isinstance(value, phuzzy.FuzzyNumber):
                if list_of_n_alpha_levels[0] == 0:
                    np.put(list_of_n_alpha_levels, 0, value.number_of_alpha_levels)
                else:
                    list_of_n_alpha_levels = np.append(list_of_n_alpha_levels, value.number_of_alpha_levels)

        # extract fuzzy variables from kwargs and safe in dict
        for key, value in kwargs.items():
            # if number_of_alpha_levels are different
            if (len(set(list_of_n_alpha_levels)) == 1) == False:
                max = np.max(list_of_n_alpha_levels)
                if isinstance(value, phuzzy.FuzzyNumber):
                    if value.number_of_alpha_levels < max: value.convert_df(alpha_levels=max)
                    filter_fuzzy_variables_dict[key] = value._df
            elif isinstance(value, phuzzy.FuzzyNumber):
                # if number_of_alpha_levels are the same
                filter_fuzzy_variables_dict[key] = value._df

        # extract fuzzy values from dict and safe as DataArray
        fuzzy_variables = {k: xr.DataArray(v, dims=['number_of_alpha_levels', 'alpha_level_bounds'])
                           for k, v in filter_fuzzy_variables_dict.items()}

        self.global_bounds_DataArray = xr.Dataset(fuzzy_variables).to_array(dim='fuzzy_variables')

    def inequality_constraints(self):
        # cons = ({'type': 'ineq', 'fun': g1}, #>=
        #       {'type': 'ineq', 'fun': g2},
        #       {'type': 'eq', 'fun': h1})
        pass

    def equality_constraints(self):
        # cons = ({'type': 'ineq', 'fun': g1}, #>=
        #       {'type': 'ineq', 'fun': g2},
        #       {'type': 'eq', 'fun': h1})
        pass


class Alpha_Level_Optimization(Constraints, ObjFunction, FuzzyNumber, MPL_Mixin):

    def __init__(self, n = 15, iters=3, optimizer = 'sobol', **kwargs):
        ObjFunction.__init__(self)
        Constraints.__init__(self, **kwargs)
        #super().__init__(**kwargs)

        if kwargs.get('name') is None: self.name = 'Fuzzy Objective Value'
        else: self.name = kwargs.get('name')

        self.dim = self.global_bounds_DataArray['fuzzy_variables'].size
        self.number_of_alpha_levels = len(self.global_bounds_DataArray['number_of_alpha_levels'].values)

        self.n = n
        self.iters = iters
        self.optimizer = optimizer          # simplicial / sobol

        self.best_indi_list_min = []
        self.best_indi_list_max = []

    def __repr__(self):
        return "Z(x:[[{:.3g}, {:.3g}], [{:.3g}, {:.3g}]])".format(self._df.iloc[0].l, self._df.iloc[0].r,
                                                                  self._df.iloc[-1].l, self._df.iloc[-1].r)

    def calculation(self):
        global bounds
        zmin_value_list = []
        zmax_value_list = []
        boundlist = []

        for i in range(1, self.global_bounds_DataArray['number_of_alpha_levels'].size + 1):
            boundlist.append(np.delete(self.global_bounds_DataArray.values[:, -i, :], 0, 1))

        for lvl, bounds in enumerate(boundlist):
            comp_bounds = []
            for item_i, item_j in zip(bounds[:, 0], bounds[:, 1]):  comp_bounds.extend([item_i == item_j])

            if lvl == 0:  # Alpha-Level = 1
                if all(comp_bounds) == True: # if all bounds are constants
                    ## calculate objective value
                    zmin = self.objective_function(bounds[:, 0])

                    ## safe values in list
                    zmin_value_list.append(np.array(zmin))
                    zmax_value_list.append(np.array(zmin))

                    best_indi_min = np.array(bounds[:, 0])
                    best_indi_max = np.array(bounds[:, 0])

                elif all(comp_bounds) == False and any(comp_bounds) == True: # if one or more bounds are constants
                    ## prepare bounds / pop constants
                    bounds = self.pop_constants(comp_bounds, bounds)

                    ## optimization routine
                    if self.optimizer == 'simplicial':
                        shc_const_min = SHGO(self.min_function_value, bounds, n=self.n, iters=self.iters,
                                             options={'ftol': 1e-6})
                        shc_const_max = SHGO(self.max_function_value, bounds, n=self.n, iters=self.iters,
                                             options={'ftol': 1e-6})
                    else:
                        shc_const_min = SHGO(self.min_function_value, bounds, n=self.n, iters=self.iters,
                                             sampling_method='sobol', options={'ftol': 1e-4})
                        shc_const_max = SHGO(self.max_function_value, bounds, n=self.n, iters=self.iters,
                                             sampling_method='sobol', options={'ftol': 1e-4})

                    #shc_const_min.construct_complex()
                    #shc_const_max.construct_complex()
                    shc_const_min = self.find_result(shc_const_min)
                    shc_const_max = self.find_result(shc_const_max)

                    ## safe results in lists
                    best_indi_min = self.safe_best_array(comp_bounds, shc_const_min.res)
                    best_indi_max = self.safe_best_array(comp_bounds, shc_const_max.res)

                    zmin_value_list.append(shc_const_min.res.fun * (1))
                    zmax_value_list.append(shc_const_max.res.fun * (-1))
                    self.x_glob = []

            else:  # Alpha-Level < 1
                # if all bounds are Fuzzy-Intervalls
                if 'shc_fuzzy_min' in locals():
                    shc_fuzzy_min.bounds = bounds
                    shc_fuzzy_min.iterate()
                    shc_fuzzy_min.find_minima()

                    shc_fuzzy_max.bounds = bounds
                    shc_fuzzy_max.iterate()
                    shc_fuzzy_max.find_minima()
                else:
                    if self.optimizer == 'simplicial':
                        shc_fuzzy_min = SHGO(self.min_function_value, bounds, n=self.n, iters=self.iters,
                                             options={'ftol': 1e-6})
                        shc_fuzzy_max = SHGO(self.max_function_value, bounds, n=self.n, iters=self.iters,
                                             options={'ftol': 1e-6})
                    else:
                        shc_fuzzy_min = SHGO(self.min_function_value, bounds, n=self.n, iters=self.iters,
                                             sampling_method='sobol', options={'ftol': 1e-4})
                        shc_fuzzy_max = SHGO(self.max_function_value, bounds, n=self.n, iters=self.iters,
                                             sampling_method='sobol', options={'ftol': 1e-4})

                    #shc_fuzzy_min.construct_complex()
                    #shc_fuzzy_max.construct_complex()
                    shc_fuzzy_min = self.find_result(shc_fuzzy_min)
                    shc_fuzzy_max = self.find_result(shc_fuzzy_max)

                ## safe results in lists
                best_indi_min = shc_fuzzy_min.res.x
                best_indi_max = shc_fuzzy_max.res.x

                zmin_value_list.append(shc_fuzzy_min.res.fun * (1))
                zmax_value_list.append(shc_fuzzy_max.res.fun * (-1))

            self.best_indi_list_min.append(best_indi_min)
            self.best_indi_list_max.append(best_indi_max)

        self.zmin_values = self.safe_z_values(min_max='min', z_value_list=zmin_value_list)
        self.zmax_values = self.safe_z_values(min_max='max', z_value_list=zmax_value_list)

        self.simple_dataframe()


    def find_result(self, shc):
        shc.construct_complex()
        if len(shc.LMC.xl_maps) > 0:
            return shc
        else:
            lres = minimize(shc.func, shc.x_lowest,
                                           **shc.minimizer_kwargs)
            shc.res.nlfev += lres.nfev
            try:
                lres.fun = lres.fun[0]
            except (IndexError, TypeError):
                lres.fun

            shc.LMC[shc.x_lowest]
            shc.LMC.add_res(shc.x_lowest, lres)
            shc.sort_result()
            # Lowest values used to report in case of failures
            shc.f_lowest = shc.res.fun
            shc.x_lowest = shc.res.x
            return shc


    def separat_minimization(self):

        zmin_value_list = []
        boundlist = []
        self.best_indi_list_min = []

        for i in range(1, self.global_bounds_DataArray['number_of_alpha_levels'].size + 1):
            boundlist.append(np.delete(self.global_bounds_DataArray.values[:, -i, :], 0, 1))

        for lvl, bounds in enumerate(boundlist):

            comp_bounds = []
            for item_i, item_j in zip(bounds[:, 0], bounds[:, 1]):  comp_bounds.extend([item_i == item_j])

            if lvl == 0:
                if all(comp_bounds) == True:
                    # if all bounds are constants (equal)
                    # calculate objective value
                    zmin = self.objective_function(bounds[:, 0])

                    # safe values in list
                    zmin_value_list.append(np.array(zmin))
                    best_indi = np.array(bounds[:, 0])

                elif all(comp_bounds) == False and any(comp_bounds) == True:
                    # if one or more bounds are constants
                    # prepare bounds / pop constants
                    bounds = self.pop_constants(comp_bounds, bounds)

                    # optimization routine
                    if self.optimizer == 'simplicial':
                        shc_const_min = SHGO(self.min_function_value, bounds, n=self.n, iters=self.iters,
                                             options={'ftol': 1e-6})
                    else:
                        shc_const_min = SHGO(self.min_function_value, bounds, n=self.n, iters=self.iters,
                                             sampling_method='sobol', options={'ftol': 1e-4})
                    shc_const_min.construct_complex()

                    # safe values in list
                    best_indi = self.safe_best_array(comp_bounds, shc_const_min.res)
                    zmin_value_list.append(shc_const_min.res.fun * (1))
                    self.x_glob = []
            else:
                # if all bounds are intervalls
                if 'shc_fuzzy_min' in locals():
                    shc_fuzzy_min.bounds = bounds
                    shc_fuzzy_min.iterate()
                    shc_fuzzy_min.find_minima()

                else:
                    if self.optimizer == 'simplicial':
                        shc_fuzzy_min = SHGO(self.min_function_value, bounds, n=self.n, iters=self.iters,
                                             options={'ftol': 1e-6})
                    else:
                        shc_fuzzy_min = SHGO(self.min_function_value, bounds, n=self.n, iters=self.iters,
                                             sampling_method='sobol', options={'ftol': 1e-4})
                    shc_fuzzy_min.construct_complex()

                best_indi = shc_fuzzy_min.res.x
                zmin_value_list.append(shc_fuzzy_min.res.fun * (1))

            self.best_indi_list_min.append(best_indi)

        self.zmin_values = self.safe_z_values('min', zmin_value_list)


    def separat_maximization(self):

        zmax_value_list = []
        boundlist = []
        self.best_indi_list_max = []

        for i in range(1, self.global_bounds_DataArray['number_of_alpha_levels'].size + 1):
            boundlist.append(np.delete(self.global_bounds_DataArray.values[:, -i, :], 0, 1))

        for lvl, bounds in enumerate(boundlist):

            comp_bounds = []
            for item_i, item_j in zip(bounds[:, 0], bounds[:, 1]):  comp_bounds.extend([item_i == item_j])

            if lvl == 0:
                if all(comp_bounds) == True:
                    # if all bounds are constants (equal)
                    # calculate objective value
                    zmax = self.objective_function(bounds[:, 0])

                    # safe values in list
                    zmax_value_list.append(np.array(zmax))
                    best_indi = np.array(bounds[:, 0])

                elif all(comp_bounds) == False and any(comp_bounds) == True:
                    # if one or more bounds are constants
                    # prepare bounds / pop constants
                    bounds = self.pop_constants(comp_bounds, bounds)

                    # optimization routine
                    if self.optimizer == 'simplicial':
                        shc_const_max = SHGO(self.max_function_value, bounds, n=self.n, iters=self.iters,
                                             options={'ftol': 1e-6})
                    else:
                        shc_const_max = SHGO(self.max_function_value, bounds, n=self.n, iters=self.iters,
                                             sampling_method='sobol', options={'ftol': 1e-4})
                    shc_const_max.construct_complex()

                    # safe values in list
                    best_indi = self.safe_best_array(comp_bounds, shc_const_max.res)

                    zmax_value_list.append(shc_const_max.res.fun * (-1))
                    self.x_glob = []
            else:
                # if all bounds are intervalls
                if 'shc_fuzzy_max' in locals():
                    shc_fuzzy_max.bounds = bounds
                    shc_fuzzy_max.iterate()
                    shc_fuzzy_max.find_minima()
                else:
                    if self.optimizer == 'simplicial':
                        shc_fuzzy_max = SHGO(self.max_function_value, bounds, n=self.n, iters=self.iters,
                                             options={'ftol': 1e-6})
                    else:
                        shc_fuzzy_max = SHGO(self.max_function_value, bounds, n=self.n, iters=self.iters,
                                             sampling_method='sobol', options={'ftol': 1e-4})
                    shc_fuzzy_max.construct_complex()

                best_indi = shc_fuzzy_max.res.x
                zmax_value_list.append(shc_fuzzy_max.res.fun * (-1))

            self.best_indi_list_max.append(best_indi)

        self.zmax_values = self.safe_z_values('max', zmax_value_list)


    def simple_dataframe(self):
        self._df = pd.DataFrame(data={'alpha': np.linspace(0, 1.0, self.number_of_alpha_levels), 'l': self.zmin_values, 'r': self.zmax_values})
        self.df = self._df





    def expanded_dataframe(self):
        self.best_indi_list_min.reverse()
        self.best_indi_list_max.reverse()
        self._df = pd.DataFrame(data={'alpha': np.linspace(0, 1.0, self.number_of_alpha_levels),
                                      'l': self.zmin_values, 'best_indi_l': self.best_indi_list_min,
                                      'r': self.zmax_values, 'best_indi_r': self.best_indi_list_max})


    def export_to_csv(self, filepath= None):
        if filepath is None:
            datatype_str = '_results.csv'
            self._df.to_csv(self.name+datatype_str, sep=';', encoding='utf8', index=None, header=True)
        else:
            if filepath[-2:] != '\\': filepath = filepath + '\\'
            datatype_str = '_results.csv'
            self._df.to_csv(filepath+self.name+datatype_str,  sep=';', encoding='utf8', index=None, header=True)


    def safe_best_array(self, comp_bounds, z_res):
        best_indi = np.zeros((self.dim), dtype=float)
        j = 0
        for i, comp in enumerate(comp_bounds):
            if comp:
                best_indi[i] = self.global_bounds_DataArray[i, -1, 1].values
            else:
                best_indi[i] = z_res.x[j]
                j += 1
        return best_indi


    def pop_constants(self, comp_bounds, bounds):
        pop = np.array([i for i, x in enumerate(comp_bounds) if x])
        self.x_glob = np.concatenate((np.atleast_2d(pop).T, np.atleast_2d(bounds[pop, 0]).T), axis=1)
        bounds = np.delete(bounds, pop, 0)
        return bounds


    def defuzzification(self, method = 'centroid'):

        self.zmax_values = np.flip(self.zmax_values)
        if method == 'alpha_one':
            self.deter_objective = ((self.alpha1['l']+self.alpha1['r']) / 2)

        elif method == 'mean':
            if self.zmin_values[-1]==self.zmax_values[0]:
                self.deter_objective = np.mean(np.concatenate((self.zmin_values[:-1],self.zmax_values), axis=0))
            else:
                self.deter_objective = np.mean(np.concatenate((self.zmin_values,self.zmax_values), axis=0))
                #self.deter_objective = np.mean((self.df['l'].values+self.df['r'].values)/2)
        elif method == 'centroid':
            A = 0
            B = 0
            if self.zmin_values[-1]==self.zmax_values[0]:
                X = np.hstack((np.hstack((self.zmin_values[:-1],self.zmax_values)),self.zmin_values[0]))
                Y = np.hstack((np.hstack((np.linspace(0,1,self.number_of_alpha_levels)[:-1],np.linspace(1,0,self.number_of_alpha_levels))),np.array([0])))
            else:
                X = np.hstack((np.hstack((self.zmin_values,self.zmax_values)),self.zmin_values[0]))
                Y = np.hstack((np.hstack((np.linspace(0,1,self.number_of_alpha_levels),np.linspace(1,0,self.number_of_alpha_levels))),np.array([0])))

            for i in range(0,len(X)-1):
                a = (X[i]*Y[i+1]-X[i+1]*Y[i])
                b = (X[i]+X[i+1])*(X[i]*Y[i+1]-X[i+1]*Y[i])
                A = A + a
                B = B + b
            self.deter_objective = (1/(3*A))*B

        if self.deter_objective < self.zmin_values[-1]:
            index_i = np.where(self.zmin_values < self.deter_objective)[0][-1]
            index_ii = np.where(self.zmin_values > self.deter_objective)[0][0]
            self.y_interpol = np.interp(self.deter_objective, [self.zmin_values[index_i],self.zmin_values[index_ii]],
                                   [np.linspace(0,1,self.number_of_alpha_levels)[index_i],
                                    np.linspace(0,1,self.number_of_alpha_levels)[index_ii]])
        elif self.deter_objective > self.zmax_values[0]:
            index_i = np.where(self.deter_objective > self.zmax_values)[0][-1]
            index_ii = np.where(self.deter_objective < self.zmax_values)[0][0]
            self.y_interpol = np.interp(self.deter_objective, [self.zmax_values[index_i],self.zmax_values[index_ii]],
                                   [np.linspace(1,0,self.number_of_alpha_levels)[index_i],
                                    np.linspace(1,0,self.number_of_alpha_levels)[index_ii]])
        elif self.deter_objective == ((self.alpha1['l']+self.alpha1['r']) / 2):
            self.y_interpol = 1.0


    @staticmethod
    def safe_z_values(min_max, z_value_list):
        if min_max == 'min':
            for (i, current_item), next_item in zip(enumerate(z_value_list), z_value_list[1:]):
                if current_item < next_item:
                    z_value_list[i + 1] = current_item
        elif min_max == 'max':
            for (i, current_item), next_item in zip(enumerate(z_value_list), z_value_list[1:]):
                if current_item > next_item:
                    z_value_list[i + 1] = current_item
        else:
            print('Please define -min- or -max- in min_max')
        return np.flip(z_value_list, axis=0)


    @classmethod
    def from_str(cls, s):
        pass

    def to_str(self):
        pass

    def discretize(self, alpha0, alpha1, alpha_levels):
        pass


# FUZZY ALPHA LEVEL OPT ROUTINE
var_1 = phuzzy.Trapezoid(alpha0=[0, 4], alpha1=[2, 3], number_of_alpha_levels=5)
var_2 = phuzzy.Trapezoid(alpha0=[2, 4], alpha1=[3, 3], number_of_alpha_levels=5)
var_3 = phuzzy.Triangle(alpha0=[-3, 1], alpha1=[1, 1], number_of_alpha_levels=5)
var_4 = phuzzy.Triangle(alpha0=[0, 5], alpha1=[3, 3], number_of_alpha_levels=8)
var_5 = phuzzy.Triangle(alpha0=[-10, 1], alpha1=[-4, -4], number_of_alpha_levels=5)
var_6 = phuzzy.Trapezoid(alpha0=[-10, 10], alpha1=[-2, 2], number_of_alpha_levels=5)

kwargs = {"var_1": var_1, "var_2": var_2, "var_3": var_3, "var_4": var_4, "var_5": var_5,  "var_6": var_6}

a = datetime.datetime.now()
z = Alpha_Level_Optimization(**kwargs)
z.calculation()
b = datetime.datetime.now()
print(b-a)

z.defuzzification()
z.plot(defuzzy=[z.deter_objective, z.y_interpol])
plt.show()

r = 1





