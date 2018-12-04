# -*- coding: utf-8 -*-

# import phuzzy.contrib.shgo

import phuzzy
from phuzzy.mpl import MPL_Mixin
from phuzzy.shapes import FuzzyNumber

from asteval import Interpreter
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
from asteval import Interpreter
from scipy.optimize import minimize
from shgo._shgo import SHGO
from tqdm import tqdm

import phuzzy

import sys
class Alpha_Level_Optimization():

    def __init__(self, **kwargs):
        """
        :param kwargs:      Inputvariables / Fuzzyvariables / Name (name) / Objective Function (obj_function) / Objective Link (obj_link)
        """
        self.fuzzynumber = FuzzyNumber()

        # Define Bounds of Each Alpha Level
        self.global_bounds_DataArray = self._boundary_constraints(**kwargs)
        print(self.global_bounds_DataArray)
        print(self.global_bounds_DataArray.to_dataframe(name="a"))
        sys.exit()
        self.x_glob = []

        # Filter Objective Function / Link and Safe it
        if kwargs.get('obj_function') is not None:
            self.objective = kwargs.get('obj_function')
        else:
            raise ValueError('PLEASE IMPORT OBJECTIVE')

        # Define Setup Parameters
        self.name = kwargs.get('name', 'Fuzzy Objective Value')

        self.dim = self.global_bounds_DataArray['fuzzy_variables'].size
        self.number_of_alpha_lvls = self.global_bounds_DataArray['number_of_alpha_levels'].size
        self.best_indi_list_min = []
        self.best_indi_list_max = []
        self.nfev_list_min = []
        self.nfev_list_max = []
        self.nit_list_min = []
        self.nit_list_max = []

    def __repr__(self):
        return "{}".format(self.name)

    def calculation(self, n=60, iters=3, optimizer='sobol', backup=False, start_at=None):
        """
        Main Routine calculating the Minimum and Maximum of the Objective on each Alpha Level to generate
        the Fuzzy Objective Membershipfunction.
        :param n:           Number of Sampling Points / Individuals for the Optimization Algorithm
        :param iters:       Number of max. Iterations per Optimization Loop
        :param optimizer:   Selected Optimizer Strategy: "sobol" / "simplicial"
        :param backup:      Creates a Backup Folder saving the result of each Alpha Level Result
        :param start_at:    Start at certain Alpha Level (Counts starts from Alpha Level 1)
        """

        # Input Variables
        self.n = n
        self.iters = iters
        self.optimizer = optimizer  # simplicial / sobol
        self.backup = backup
        self.start_at = start_at

        zmin_value_list = []
        zmax_value_list = []
        boundlist = []

        if self.start_at is not None: self._cut_global_blounds()

        for i in range(1, self.global_bounds_DataArray['number_of_alpha_levels'].size + 1):
            boundlist.append(np.delete(self.global_bounds_DataArray.values[:, -i, :], 0, 1))

        with tqdm(total=len(boundlist)) as pbar:
            for lvl, bounds in enumerate(boundlist):
                comp_bounds = []

                for item_i, item_j in zip(bounds[:, 0], bounds[:, 1]):  comp_bounds.extend([item_i == item_j])

                if lvl == 0:  # Alpha-Level = 1
                    if self.start_at is None:
                        if all(comp_bounds) == True:  # if all bounds are constants
                            ## calculate objective value
                            zmin = self.objective_function(bounds[:, 0])

                            ## safe values in list
                            zmin_value_list.append(np.array(zmin))
                            zmax_value_list.append(np.array(zmin))

                            best_indi_min = np.array(bounds[:, 0])
                            best_indi_max = np.array(bounds[:, 0])
                            nfev_min = np.array(0)
                            nfev_max = np.array(0)
                            nit_min = np.array(0)
                            nit_max = np.array(0)

                        elif all(comp_bounds) == False and any(
                            comp_bounds) == True:  # if one or more bounds are constants
                            ## prepare bounds / pop constants
                            bounds = self._pop_constants(comp_bounds, bounds)

                            ## optimization routine
                            shc_const_min = self._call_minimizer_shgo(bounds)
                            shc_const_max = self._call_maximizer_shgo(bounds)

                            shc_const_min = self._find_result(shc_const_min)
                            shc_const_max = self._find_result(shc_const_max)

                            ## safe results in lists
                            best_indi_min = self._safe_best_array(comp_bounds, shc_const_min.res)
                            best_indi_max = self._safe_best_array(comp_bounds, shc_const_max.res)
                            nfev_min = shc_const_min.res.nfev
                            nfev_max = shc_const_max.res.nfev
                            nit_min = shc_const_min.res.nit
                            nit_max = shc_const_max.res.nit

                            zmin_value_list.append(shc_const_min.res.fun * (1))
                            zmax_value_list.append(shc_const_max.res.fun * (-1))
                            self.x_glob = []
                    else:
                        shc_const_min = self._call_minimizer_shgo(bounds)
                        shc_const_max = self._call_maximizer_shgo(bounds)

                        shc_const_min = self._find_result(shc_const_min)
                        shc_const_max = self._find_result(shc_const_max)

                        ## safe results in lists
                        best_indi_min = self._safe_best_array(comp_bounds, shc_const_min.res)
                        best_indi_max = self._safe_best_array(comp_bounds, shc_const_max.res)
                        nfev_min = shc_const_min.res.nfev
                        nfev_max = shc_const_max.res.nfev
                        nit_min = shc_const_min.res.nit
                        nit_max = shc_const_max.res.nit

                        zmin_value_list.append(shc_const_min.res.fun * (1))
                        zmax_value_list.append(shc_const_max.res.fun * (-1))

                else:  # Alpha-Level < 1
                    # if all bounds are Fuzzy-Intervalls
                    if 'shc_fuzzy_min' not in locals():
                        shc_const_min = self._call_minimizer_shgo(bounds)
                        shc_const_max = self._call_maximizer_shgo(bounds)

                        shc_fuzzy_min = self._find_result(shc_const_min)
                        shc_fuzzy_max = self._find_result(shc_const_max)
                    else:
                        shc_fuzzy_min.bounds = bounds
                        shc_fuzzy_min.iterate()
                        shc_fuzzy_min.find_minima()

                        shc_fuzzy_max.bounds = bounds
                        shc_fuzzy_max.iterate()
                        shc_fuzzy_max.find_minima()

                    ## safe results in lists
                    best_indi_min = shc_fuzzy_min.res.x
                    best_indi_max = shc_fuzzy_max.res.x
                    if lvl <= 1:
                        nfev_min = shc_const_min.res.nfev
                        nfev_max = shc_const_max.res.nfev
                    else:
                        nfev_min = np.absolute((shc_const_min.res.nfev - np.sum(self.nfev_list_min[0:lvl - 1])))
                        nfev_max = np.absolute((shc_const_max.res.nfev - np.sum(self.nfev_list_max[0:lvl - 1])))
                    nit_min = shc_const_min.res.nit
                    nit_max = shc_const_max.res.nit

                    zmin_value_list.append(shc_fuzzy_min.res.fun * (1))
                    zmax_value_list.append(shc_fuzzy_max.res.fun * (-1))

                self.best_indi_list_min.append(best_indi_min)
                self.best_indi_list_max.append(best_indi_max)
                self.nfev_list_min.append(nfev_min)
                self.nfev_list_max.append(nfev_max)
                self.nit_list_min.append(nit_min)
                self.nit_list_max.append(nit_max)

                if self.backup == True:
                    self.zmin_values = self._safe_z_values(min_max='min', z_value_list=zmin_value_list)
                    self.zmax_values = self._safe_z_values(min_max='max', z_value_list=zmax_value_list)
                    self._call_backup(iteration=lvl)

                pbar.update()

        self.zmin_values = self._safe_z_values(min_max='min', z_value_list=zmin_value_list)
        self.zmax_values = self._safe_z_values(min_max='max', z_value_list=zmax_value_list)

        self.total_nfev = sum(self.nfev_list_min) + sum(self.nfev_list_max)
        self.compact_output()

    def compact_output(self, round=None):
        """
        Returns Dataframe of Objective Memebership Function
        :param round:       Round Values of Dataframe
        :return:            Dataframe
        """
        if self.start_at is None:
            self._df = pd.DataFrame(data={'alpha': np.linspace(0, 1.0, self.number_of_alpha_lvls),
                                          'l': self.zmin_values, 'r': self.zmax_values})
        else:
            self._df = pd.DataFrame(data={'alpha': np.linspace(0,
                                                               np.linspace(0, 1.0, self.orig_number_of_alpha_lvls)
                                                               [self.orig_number_of_alpha_lvls - self.start_at],
                                                               self.number_of_alpha_lvls),
                                          'l': self.zmin_values,
                                          'r': self.zmax_values})

        self.df = self._df
        if round is not None:
            self.df = self.df.round(round)
            self._df = self._df.round(round)

    def extanded_output(self, round=None):
        """
        Create extanded Solution Dataframe of Objective
        :param round:   Round Values of Dataframe
        :return:        Dataframe
        """
        self.best_indi_list_min.reverse()
        self.best_indi_list_max.reverse()
        self.nfev_list_min.reverse()
        self.nfev_list_max.reverse()
        self.nit_list_min.reverse()
        self.nit_list_max.reverse()
        if round is not None:
            min_arr = np.round(self.best_indi_list_min, round)
            max_arr = np.round(self.best_indi_list_max, round)
        else:
            min_arr = np.array(self.best_indi_list_min)
            max_arr = np.array(self.best_indi_list_max)

        if self.start_at is None:
            self.df_extanded = pd.DataFrame(data={'alpha': np.linspace(0, 1.0, self.number_of_alpha_lvls),
                                                  'l': self.zmin_values, 'best_indi_l': min_arr.tolist(),
                                                  'nfev_l': self.nfev_list_min, 'nit_l': self.nit_list_min,
                                                  'r': self.zmax_values, 'best_indi_r': max_arr.tolist(),
                                                  'nfev_r': self.nfev_list_max, 'nit_r': self.nit_list_max})
        else:
            self.df_extanded = pd.DataFrame(data={'alpha':
                                                      np.linspace(0, np.linspace(0, 1.0, self.orig_number_of_alpha_lvls)
                                                      [self.orig_number_of_alpha_lvls - self.start_at],
                                                                  self.number_of_alpha_lvls),
                                                  'l': self.zmin_values, 'best_indi_l': min_arr.tolist(),
                                                  'nfev_l': self.nfev_list_min, 'nit_l': self.nit_list_min,
                                                  'r': self.zmax_values, 'best_indi_r': max_arr.tolist(),
                                                  'nfev_r': self.nfev_list_max, 'nit_r': self.nit_list_max})

        if round is not None: self.df_extanded = self.df_extanded.round(round)

    def export_to_csv(self, df='simple', filepath=None):
        """
        Export Dataframe as CSV
        :param df               Define Dataframe-Type which is to be exported simple / extended
        :param filepath:        Filepath where to save the DF
        :return:                CSV of Dataframe
        """
        if df == 'simple':
            if filepath is None:
                datatype_str = '_results.csv'
                self._df.to_csv(self.name + datatype_str, sep=';', encoding='utf8', index=None, header=True)
            else:
                if filepath[-2:] != '\\': filepath = filepath + '\\'
                datatype_str = '_results.csv'
                self._df.to_csv(filepath + self.name + datatype_str, sep=';', encoding='utf8', index=None, header=True)
        elif df == 'extended':
            if filepath is None:
                datatype_str = '_extended_results.csv'
                self.df_extanded.to_csv(self.name + datatype_str, sep=';', encoding='utf8', index=None, header=True)
            else:
                if filepath[-2:] != '\\': filepath = filepath + '\\'
                datatype_str = '_extended_results.csv'
                self.df_extanded.to_csv(filepath + self.name + datatype_str, sep=';', encoding='utf8', index=None,
                                        header=True)

    def defuzzification(self, method='mean'):
        """
        Defuzzyfication of Uncertain Objective
        :param method:          Select Method of Defuzzyfication - alpha_one / mean / centroid
        """
        self.zmax_values = np.flip(self.zmax_values)
        if method == 'alpha_one':
            self.determin_objective = ((self.alpha1['l'] + self.alpha1['r']) / 2)

        elif method == 'mean':
            if self.zmin_values[-1] == self.zmax_values[0]:
                self.determin_objective = np.mean(np.concatenate((self.zmin_values[:-1], self.zmax_values), axis=0))
            else:
                self.determin_objective = np.mean(np.concatenate((self.zmin_values, self.zmax_values), axis=0))
                # self.determin_objective = np.mean((self.df['l'].values+self.df['r'].values)/2)
        elif method == 'centroid':
            A = 0
            B = 0
            if self.zmin_values[-1] == self.zmax_values[0]:
                X = np.hstack((np.hstack((self.zmin_values[:-1], self.zmax_values)), self.zmin_values[0]))
                Y = np.hstack((np.hstack(
                    (np.linspace(0, 1, self.number_of_alpha_lvls)[:-1], np.linspace(1, 0, self.number_of_alpha_lvls))),
                               np.array([0])))
            else:
                X = np.hstack((np.hstack((self.zmin_values, self.zmax_values)), self.zmin_values[0]))
                Y = np.hstack((np.hstack(
                    (np.linspace(0, 1, self.number_of_alpha_lvls), np.linspace(1, 0, self.number_of_alpha_lvls))),
                               np.array([0])))

            for i in range(0, len(X) - 1):
                a = (X[i] * Y[i + 1] - X[i + 1] * Y[i])
                b = (X[i] + X[i + 1]) * (X[i] * Y[i + 1] - X[i + 1] * Y[i])
                A = A + a
                B = B + b
            self.determin_objective = (1 / (3 * A)) * B

        if self.determin_objective < self.zmin_values[-1]:
            index_i = np.where(self.zmin_values < self.determin_objective)[0][-1]
            index_ii = np.where(self.zmin_values > self.determin_objective)[0][0]
            y_interpol = np.interp(self.determin_objective, [self.zmin_values[index_i], self.zmin_values[index_ii]],
                                   [np.linspace(0, 1, self.number_of_alpha_lvls)[index_i],
                                    np.linspace(0, 1, self.number_of_alpha_lvls)[index_ii]])
        elif self.determin_objective > self.zmax_values[0]:
            index_i = np.where(self.determin_objective > self.zmax_values)[0][-1]
            index_ii = np.where(self.determin_objective < self.zmax_values)[0][0]
            y_interpol = np.interp(self.determin_objective, [self.zmax_values[index_i], self.zmax_values[index_ii]],
                                   [np.linspace(1, 0, self.number_of_alpha_lvls)[index_i],
                                    np.linspace(1, 0, self.number_of_alpha_lvls)[index_ii]])
        else:
            y_interpol = 1.0

        self.determin_point = np.array([self.determin_objective, y_interpol])

    def _call_minimizer_shgo(self, bounds):
        """
        Minimizes Objective Function with the SHGO Optimizer
        :param bounds: Boundary of the current Alpha Level
        :return: Minimum
        """
        if self.optimizer == 'sobol':
            return SHGO(self._min_function_value, bounds=bounds, n=self.n, iters=self.iters,
                        sampling_method='sobol', options={'ftol': 1e-4}, constraints=None)
        else:
            return SHGO(self._min_function_value, bounds=bounds, n=self.n, iters=self.iters,
                        options={'ftol': 1e-4}, constraints=None)

    def _call_maximizer_shgo(self, bounds):
        """
        Maximizes Objective Function with the SHGO Optimizer
        :param bounds: Boundary of the current Alpha Level
        :return: Maximum
        """
        if self.optimizer == 'sobol':
            return SHGO(self._max_function_value, bounds=bounds, n=self.n, iters=self.iters,
                        sampling_method=self.optimizer, options={'ftol': 1e-4}, constraints=None)
        else:
            return SHGO(self._max_function_value, bounds=bounds, n=self.n, iters=self.iters,
                        options={'ftol': 1e-4}, constraints=None)

    def _objective_function(self, x):
        """
        Abstract Objective Function Call
        :param x:   Input Variable
        :return:    Function Value of Objective
        """
        aeval = Interpreter()
        exprc = aeval.parse(self.objective)
        aeval.symtable['x'] = x
        return aeval.run(exprc)

    def _cut_global_blounds(self):
        """
        Cut global Bounds for the Option to -start at- a selected Alpha Level
        """
        if self.start_at is not None:
            self.orig_number_of_alpha_lvls = self.number_of_alpha_lvls
            self.global_bounds_DataArray = self.global_bounds_DataArray[:,
                                           0:self.orig_number_of_alpha_lvls - self.start_at + 1, :]
            self.dim = self.global_bounds_DataArray['fuzzy_variables'].size
            self.number_of_alpha_lvls = self.global_bounds_DataArray['number_of_alpha_levels'].size
        else:
            raise ValueError('Please select starting Alpha Level or keep variable "start_at=None"')

    def _min_function_value(self, x):
        """
        Call Minimization Opt Routine
        :param x:       Input Variable
        :return:        Objective Value
        """
        if isinstance(self.x_glob, np.ndarray):
            for row in self.x_glob:
                x = np.insert(x, (row[0].astype(int)), row[1])
        return self._objective_function(x)

    def _max_function_value(self, x):
        """
        Call Maximization Opt Routine
        :param x:       Input Variable
        :return:        Objective Value
        """
        if isinstance(self.x_glob, np.ndarray):
            for row in self.x_glob:
                x = np.insert(x, (row[0].astype(int)), row[1])
        return -1 * (self._objective_function(x))

    def _safe_best_array(self, comp_bounds, z_res):
        """
        Post Preparation to extrapolate sampling Point with min/max Objective Value from Optimization Result
        :param comp_bounds:     Boolean Array of all Fuzzy Input Variables which are constants for a certain Alpha Level
        :param z_res:           Result of Optimization Loop
        :return:
        best_indi               Best Individual
        """
        best_indi = np.zeros((self.dim), dtype=float)
        j = 0
        for i, comp in enumerate(comp_bounds):
            if comp:
                best_indi[i] = self.global_bounds_DataArray[i, -1, 1].values
            else:
                best_indi[i] = z_res.x[j]
                j += 1
        return best_indi

    def _pop_constants(self, comp_bounds, bounds):
        """
        Pop Constant Fuzzy Variables of Input Variable
        :param comp_bounds:         Boolean Array of all Fuzzy Input Variables which are constants for a certain Alpha Level
        :param bounds:              All Optimization Bound of cetrain Alpha Level
        :return:
            bounds
        """
        pop = np.array([i for i, x in enumerate(comp_bounds) if x])
        self.x_glob = np.concatenate((np.atleast_2d(pop).T, np.atleast_2d(bounds[pop, 0]).T), axis=1)
        bounds = np.delete(bounds, pop, 0)
        return bounds

    def _call_backup(self, iteration):
        """
        Create Backup Dataframe for each Alpha Level and export it as CSV
        :param iteration:       Current Iterationstep / Alpha Level
        """
        if Path('back_up').exists():
            pass
        else:
            Path('back_up').mkdir()

        cache_str = 'backup_' + str(iteration) + '_iteration_step.csv'
        filepath = Path.cwd() / 'back_up' / cache_str
        if self.start_at is None:
            df_cache = pd.DataFrame(data={'alpha': np.linspace(np.linspace(0, 1.0, self.number_of_alpha_lvls)
                                                               [self.number_of_alpha_lvls - 1 - iteration], 1.0,
                                                               iteration + 1),
                                          'l': self.zmin_values,
                                          'r': self.zmax_values})
        else:
            df_cache = pd.DataFrame(data={'alpha': np.linspace(np.linspace(0, 1.0, self.number_of_alpha_lvls)
                                                               [self.number_of_alpha_lvls - 1 - iteration],
                                                               np.linspace(0, 1.0, self.orig_number_of_alpha_lvls)
                                                               [self.orig_number_of_alpha_lvls - self.start_at],
                                                               iteration + 1),
                                          'l': self.zmin_values,
                                          'r': self.zmax_values})

        df_cache.to_csv(filepath, sep=';', encoding='utf8', index=None, header=True)

    @staticmethod
    def _boundary_constraints(**kwargs):
        """Calculating the Optimization Boundaries based on the Fuzzy Variables Membership Function

        :param kwargs:      Input Fuzzy Variables
        :return:            3D DataArray representing each Fuzzy Inputvariables Boundaries for each Alpha Level
                            prepared for the Optimization Routine
        """
        filter_fuzzy_variables_dict = {}
        list_of_n_alpha_levels = np.zeros(1, dtype=int)

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
                max_value = np.max(list_of_n_alpha_levels)
                if isinstance(value, phuzzy.FuzzyNumber):
                    if value.number_of_alpha_levels < max_value: value.convert_df(alpha_levels=max_value)
                    filter_fuzzy_variables_dict[key] = value._df
            elif isinstance(value, phuzzy.FuzzyNumber):
                # if number_of_alpha_levels are the same
                filter_fuzzy_variables_dict[key] = value._df

        # extract fuzzy values from dict and safe as DataArray
        fuzzy_variables = {k: xr.DataArray(v, dims=['number_of_alpha_levels', 'alpha_level_bounds'])
                           for k, v in filter_fuzzy_variables_dict.items()}

        return xr.Dataset(fuzzy_variables).to_array(dim='fuzzy_variables')

    @staticmethod
    def _safe_z_values(min_max, z_value_list):
        """
        Post Preparation for Results of Optimization Routines for creation of Fuzzy Dataframe
        :param min_max:             Define if Result is from a Minimization or Maximization
        :param z_value_list:        List of all Objective Value Results (for each Alpha Level)
        :return:
            z_value_list            Adapted List of all Objective Value Results
        """
        if min_max == 'min':
            for (i, current_item), next_item in zip(enumerate(z_value_list), z_value_list[1:]):
                if current_item < next_item:
                    z_value_list[i + 1] = current_item
        elif min_max == 'max':
            for (i, current_item), next_item in zip(enumerate(z_value_list), z_value_list[1:]):
                if current_item > next_item:
                    z_value_list[i + 1] = current_item
        else:
            raise ValueError('Please define -min- or -max- in min_max')
        return np.flip(z_value_list, axis=0)

    @staticmethod
    def _find_result(shc):
        """
        Post Calculation for Shgo-Optimization Algorithm
        """
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

    @classmethod
    def from_str(cls, s):
        pass

    def to_str(self):
        pass


if __name__ == "__main__":
    pass
