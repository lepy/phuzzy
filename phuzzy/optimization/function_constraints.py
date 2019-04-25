import numpy as np



class ObjFunction(object):
    def __init__(self):
        self.x_glob = []

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
        return -1*((x[0] - 1) ** 2 + (x[1] + .1) ** 2 + .1 - (x[2] + 2) ** 2 - (x[3] - 0.1) ** 2 - (x[4] * x[5]) ** 2)


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
