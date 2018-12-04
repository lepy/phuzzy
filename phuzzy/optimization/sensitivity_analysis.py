import numpy as np
import pandas as pd

import phuzzy
import phuzzy.mpl as phm
from phuzzy.mpl import mix_mpl
from phuzzy.optimization import alphaOpt

from tqdm import tqdm
import matplotlib.pyplot as plt




class Fuzzy_Sensitivity_Analysis(object):

    def __init__(self,**kwargs):
        """
        :param kwargs:      Inputvariables / Fuzzyvariables / Name (name) / Objective Function (obj_function) / Objective Link (obj_link)
        """

        # Define Bounds of Each Alpha Level
        self.input_dict = kwargs
        self.fuzzy_variables_dict = self._prepare_fuzzy_variables(**kwargs)
        self.x_glob = []

        # Filter Objective Function / Link and Safe it
        if kwargs.get('obj_function') is not None: self.objective = kwargs.get('obj_function')
        elif kwargs.get('obj_link') is not None: self.objective = kwargs.get('obj_link')
        else: raise ValueError('PLEASE IMPORT OBJECTIVE')

        # Define Setup Parameters
        if kwargs.get('name') is None: self.name = 'Fuzzy Objective Value'
        else: self.name = kwargs.get('name')


    def _prepare_fuzzy_variables(self, alpha_Level=None, **kwargs):
        """
        Calculating the Optimization Boundaries based on the Fuzzy Variables Membership Funciton
        :param kwargs:      Input Fuzzy Variables
        :return:            3D DataArray representing each Fuzzy Inputvariables Boundaries for each Alpha Level
                            prepared for the Optimization Routine
        """
        filter_fuzzy_variables_dict = {}
        list_of_n_alpha_levels = np.zeros(1, dtype='int')

        if alpha_Level is None:
            # check if number_of_alpha_levels is the same
            for key, value in kwargs.items():
                if isinstance(value, phuzzy.FuzzyNumber):
                    if list_of_n_alpha_levels[0] == 0:
                        np.put(list_of_n_alpha_levels, 0, value.number_of_alpha_levels)
                    else:
                        list_of_n_alpha_levels = np.append(list_of_n_alpha_levels, value.number_of_alpha_levels)

        if alpha_Level is None:
            # extract fuzzy variables from kwargs and safe in dict
            for key, value in kwargs.items():
                # if number_of_alpha_levels are different
                if (len(set(list_of_n_alpha_levels)) == 1) == False:
                    max_value = np.max(list_of_n_alpha_levels)
                    if isinstance(value, phuzzy.FuzzyNumber):
                        if value.number_of_alpha_levels < max_value: value.convert_df(alpha_levels=max_value)
                        filter_fuzzy_variables_dict[key] = value._df
                    self.number_of_alpha_lvl = max_value
                elif isinstance(value, phuzzy.FuzzyNumber):
                    # if number_of_alpha_levels are the same
                    filter_fuzzy_variables_dict[key] = value._df
                    self.number_of_alpha_lvl = value.number_of_alpha_levels
        else:
            for key, value in kwargs.items():
                if isinstance(value, phuzzy.FuzzyNumber):
                    value.convert_df(alpha_levels=alpha_Level)
                    filter_fuzzy_variables_dict[key] = value._df
                    self.number_of_alpha_lvl = alpha_Level

        return filter_fuzzy_variables_dict


    def barchart_plot(self):


        y_pos = np.arange(len(self.name_list))
        fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, sharex=False,figsize=(10,len(y_pos)))
        width = 0.6
        width2 = 0.3


        if hasattr(self,'sensis_total_var') == True:
            fig.suptitle('Average Local Cost Effectivness Fuzzy Analysis')

            ax1.set_title('Average Total Relative Effectivness')
            ax1.set_yticks(y_pos)
            ax1.set_yticklabels(self.name_list)
            ax1.invert_yaxis()  # labels read top-to-bottom
            ax1.set_ylabel('Variables')
            ax1.set_xlabel('Total Sensitivity')
            ax1.set_axisbelow(True)
            ax1.set_xlim(0,max(self.sensis_total)+0.1)
            ax1.grid(b=True, which='major')
            ax1.grid(b=True, which='minor')
            formatted_sensis = [round(elem, 3) for elem in self.sensis_total]
            for i, v in enumerate(formatted_sensis):
                ax1.text(v+0.001, i-0.055, " "+str(v), color='black', va='center',fontweight='bold')
            ax1.barh(y_pos, self.sensis_total, width, xerr=self.sensis_total_var, align='center', color='lightslategrey', ecolor='black')

            ax2.set_title('Average Absolute Left / Right Effectivness')
            ax2.set_yticks(y_pos)
            ax2.set_yticklabels(self.name_list)
            ax2.invert_yaxis()  # labels read top-to-bottom
            ax2.set_ylabel('Variables')
            ax2.set_xlabel('Relative Partial Sensitivity')
            ax2.set_axisbelow(True)
            ax2.set_xlim(0,max(max(self.sensis_l),max(self.sensis_r))+0.1)
            ax2.grid(b=True, which='major')
            ax2.grid(b=True, which='minor')
            formatted_sensis_l = [round(elem, 3) for elem in self.sensis_l]
            formatted_sensis_r = [round(elem, 3) for elem in self.sensis_r]
            for (i_l, v_l),(i_r, v_r) in zip(enumerate(formatted_sensis_l),enumerate(formatted_sensis_r)):
                ax2.text(v_l+0.001, i_l-0.5*width2-0.055, " "+str(v_l), color='black', va='center',fontweight='bold')
                ax2.text(v_r+0.001, i_r+0.5*width2+0.055, " "+str(v_r), color='black', va='center',fontweight='bold')
            ax2.barh(y_pos-0.5*width2, self.sensis_l, width2, xerr=self.sensis_l_var, color='SkyBlue', label='Left', ecolor='black')
            ax2.barh(y_pos+0.5*width2, self.sensis_r, width2, xerr=self.sensis_r_var, color='IndianRed', label='Right', ecolor='black')
            ax2.legend()

        else:

            fig.suptitle('Local Cost Effectivness Fuzzy Analysis')

            ax1.set_title('Total Relative Effectivness')
            ax1.set_yticks(y_pos)
            ax1.set_yticklabels(self.name_list)
            ax1.invert_yaxis()  # labels read top-to-bottom
            ax1.set_ylabel('Variables')
            ax1.set_xlabel('Total Sensitivity')
            ax1.set_axisbelow(True)
            ax1.set_xlim(left=0,right=max(self.sensis_total)+0.1)
            ax1.grid(b=True, which='major')
            ax1.grid(b=True, which='minor')
            formatted_sensis = [round(elem, 3) for elem in self.sensis_total]
            for i, v in enumerate(formatted_sensis):
                ax1.text(v+0.001, i-0.055, " "+str(v), color='black', va='center',fontweight='bold')
            ax1.barh(y_pos, self.sensis_total, width, align='center', color='lightslategrey')

            ax2.set_title('Absolute Left / Right Effectivness')
            ax2.set_yticks(y_pos)
            ax2.set_yticklabels(self.name_list)
            ax2.invert_yaxis()  # labels read top-to-bottom
            ax2.set_ylabel('Variables')
            ax2.set_xlabel('Relative Partial Sensitivity')
            ax2.set_axisbelow(True)
            ax2.set_xlim(0,max(max(self.sensis_l),max(self.sensis_r))+0.1)
            ax2.grid(b=True, which='major')
            ax2.grid(b=True, which='minor')
            formatted_sensis_l = [round(elem, 3) for elem in self.sensis_l]
            formatted_sensis_r = [round(elem, 3) for elem in self.sensis_r]
            for (i_l, v_l),(i_r, v_r) in zip(enumerate(formatted_sensis_l),enumerate(formatted_sensis_r)):
                ax2.text(v_l+0.001, i_l-0.5*width2-0.055, " "+str(v_l), color='black', va='center',fontweight='bold')
                ax2.text(v_r+0.001, i_r+0.5*width2+0.055, " "+str(v_r), color='black', va='center',fontweight='bold')
            ax2.barh(y_pos-0.5*width2, self.sensis_l, width2, color='SkyBlue', label='Left')
            ax2.barh(y_pos+0.5*width2, self.sensis_r, width2, color='IndianRed', label='Right')
            ax2.legend()

        plt.show()


    def lcefa(self, error=False, lsa=None, **kwargs):
        """
        Local cost effectivness fuzzy analysis
        :return:
        """

        if error == False:
            self.name_list, self.sensis_total, self.sensis_l, self.sensis_r = self.lcefa_calculation(**kwargs)
        else:
            sensi_total_dict = {}
            sensi_l_dict = {}
            sensi_r_dict = {}
            #with tqdm(total=7,desc='Average Sensitivity Analysis',leave=True, disable=True) as pbar_glob:
            for alpha_Level in range(3,9):
                name_list, sensis_total, sensis_l, sensis_r = self.lcefa_calculation(alpha_Level=alpha_Level, lsa=lsa, leave=False,**kwargs,)
                sensi_total_dict[str(alpha_Level)] = sensis_total
                sensi_l_dict[str(alpha_Level)] = sensis_l
                sensi_r_dict[str(alpha_Level)] = sensis_r
                #pbar_glob.update()


            self.name_list = name_list
            self.sensi_total_df = pd.DataFrame.from_dict(sensi_total_dict)
            self.sensi_l_df = pd.DataFrame.from_dict(sensi_l_dict)
            self.sensi_r_df = pd.DataFrame.from_dict(sensi_r_dict)

            self.sensis_total = self.sensi_total_df.mean(axis=1).values
            self.sensis_total_var = self.sensi_total_df.var(axis=1).values
            self.sensis_l = self.sensi_l_df.mean(axis=1).values
            self.sensis_l_var = self.sensi_l_df.var(axis=1).values
            self.sensis_r = self.sensi_r_df.mean(axis=1).values
            self.sensis_r_var = self.sensi_r_df.var(axis=1).values


    def lcefa_calculation(self, alpha_Level=None, lsa=None, leave=False, **kwargs):
        res_total = []
        res_l = []
        res_r = []
        name_list = []

        s_sum = 0
        l_sum = 0
        r_sum = 0

        if alpha_Level is not None:
            self.fuzzy_variables_dict = self._prepare_fuzzy_variables(alpha_Level=alpha_Level, **kwargs)

        index = np.arange(len(self.fuzzy_variables_dict))

        with tqdm(total=len(index),desc='Sensitivity Analysis',leave=leave) as pbar_loc:
            for (var_i, values_i) in self.fuzzy_variables_dict.items():
                fvars = {}
                name_list.append(var_i)
                for (_, j) , (var_j, values_j) in zip(enumerate(index),self.fuzzy_variables_dict.items()):
                    if var_i == var_j:
                        fvars[var_j]=self.input_dict[var_i]
                    else:
                        if lsa is None:
                            y = phuzzy.Uniform(alpha0=[values_j['l'].iloc[0], values_j['r'].iloc[0]],
                                               number_of_alpha_levels=self.number_of_alpha_lvl)
                        else:
                            y = phuzzy.Uniform(alpha0=[lsa[j],lsa[j]], number_of_alpha_levels=self.number_of_alpha_lvl)

                        fvars[var_j]=y

                kwargs = {**self.input_dict , **fvars}

                z = alphaOpt.Alpha_Level_Optimization(**kwargs)
                z.calculation(n=30,iters=2,progressbar_disable=True)

                E = 1. - (z.df['r']-z.df['l'])/(z.df['r'].iloc[0]-z.df['l'].iloc[0])
                full_length = abs(z.df['r']-z.df['l'])
                E_l = abs(((z.df['l'].iloc[-1]-z.df['l'])/full_length)/(len(z.df)-1))
                E_r = abs(((z.df['r']-z.df['r'].iloc[-1])/full_length)/(len(z.df)-1))

                z.plot()

                #del z
                sensi_i = sum(E)
                sensi_l = sum(E_l)
                sensi_r = sum(E_r)

                s_sum += sensi_i
                l_sum += sensi_l
                r_sum += sensi_r

                res_total.append(sensi_i)
                res_l.append(sensi_l)
                res_r.append(sensi_r)

                pbar_loc.update()

        sensis_total = []
        sensis_l = []
        sensis_r = []

        for i in range(len(res_total)):
            sensis_total_j = res_total[i] / s_sum
            sensis_l_j = res_l[i]
            sensis_r_j = res_r[i]

            sensis_total.append(sensis_total_j)
            sensis_l.append(sensis_l_j)
            sensis_r.append(sensis_r_j)


        """
        for i, z in enumerate(res_total):
            sensi_j = z / s_sum
            sensis_total.append(sensi_j)
        """

        return name_list, sensis_total, sensis_l, sensis_r



if __name__ == "__main__":
    pass
