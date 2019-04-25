import numpy as np
import pandas as pd

import phuzzy
from phuzzy.optimization.alphaOpt import Alpha_Level_Optimization

from tqdm import tqdm
import matplotlib.pyplot as plt




class Fuzzy_Sensitivity_Analysis(object):

    def __init__(self, **kwargs):
        """
        :param input_fuzzy_var_list:    List of all Fuzzy Variables
        :param var_names_list:          List of Names of all Fuzzy Variables
        :param kwargs:      Inputvariables / Fuzzyvariables / Name (name) / Objective Function (obj_function) / Objective Link (obj_link)
        """

        # READ FUZZY INPUT VARIABLES
        if kwargs.get('fuzzy_variables') is not None: input_fuzzy_var_list = kwargs.get('fuzzy_variables')
        else: raise ValueError('PLEASE IMPORT FUZZY VARIABLES as -- fuzzy_variables --')

        # Check for identical number_of_alpha_levels and adapt
        self.fuzzy_var_list = Alpha_Level_Optimization._alpha_level_check(input_fuzzy_var_list)

        # Define Bounds of Each Alpha Level
        self.global_bounds_DataArray = Alpha_Level_Optimization._boundary_constraints(self.fuzzy_var_list)

        # Setup Parameters
        self.number_of_alpha_lvl = self.global_bounds_DataArray['number_of_alpha_levels'].size
        self.x_glob = []

        if kwargs.get('name') is None: self.name = 'Fuzzy Objective Value'
        else: self.name = kwargs.get('name')


    def main(self, search='quick', error=False, progressbar_disable = False, lsa=None, **kwargs):
        """
        Calculate Fuzzy Sensitivity of Fuzzy Variables

        :param search:                      Search Precision of Optimization Routine (quick / default / deep)
        :param error:                       include calculation of possible error margin
        :param progressbar_disable:         disable progressbar
        :param lsa:                         local search algorithm (instead of uniform fuzzy variables it uses
                                            deterministic variables to calculate the sensitivity at a certain Point)
        :param kwargs:                      kwargs
        :return:
        """

        if search == "quick":
            self.n = 15
            self.iters = 2
        elif search == "default":
            self.n = 30
            self.iters = 3
        elif search == 'deep':
            self.n = 60
            self.iters = 4


        if not error:
            self.sensis_total, self.sensis_l, self.sensis_r = \
                self._sensitivity_rotuine(progressbar_disable = progressbar_disable, **kwargs)
        else:
            sensi_total_dict = {}
            sensi_l_dict = {}
            sensi_r_dict = {}
            #with tqdm(total=7,desc='Average Sensitivity Analysis',leave=True, disable=True) as pbar_glob:
            for alpha_Level in range(3,9):
                sensis_total, sensis_l, sensis_r =\
                    self._sensitivity_rotuine(alpha_Level=alpha_Level, progressbar_disable = progressbar_disable, lsa=lsa,
                                      leave=False,**kwargs)
                sensi_total_dict[str(alpha_Level)] = sensis_total
                sensi_l_dict[str(alpha_Level)] = sensis_l
                sensi_r_dict[str(alpha_Level)] = sensis_r
                #pbar_glob.update()


            self.sensi_total_df = pd.DataFrame.from_dict(sensi_total_dict)
            self.sensi_l_df = pd.DataFrame.from_dict(sensi_l_dict)
            self.sensi_r_df = pd.DataFrame.from_dict(sensi_r_dict)

            self.sensis_total = self.sensi_total_df.mean(axis=1).values
            self.sensis_total_var = self.sensi_total_df.var(axis=1).values
            self.sensis_l = self.sensi_l_df.mean(axis=1).values
            self.sensis_l_var = self.sensi_l_df.var(axis=1).values
            self.sensis_r = self.sensi_r_df.mean(axis=1).values
            self.sensis_r_var = self.sensi_r_df.var(axis=1).values


    def barchart_plot(self, plot=False):

        y_pos = np.arange(self.global_bounds_DataArray['fuzzy_variable'].size)
        fig, ax = plt.subplots(nrows=1, ncols=2, sharex=False,figsize=(10,len(y_pos)))
        width = 0.6
        width2 = 0.3

        if hasattr(self,'sensis_total_var') == True:
            fig.suptitle('Average Fuzzy Sensitivity Analysis')
            ax[0].set_title('Average Total Relative Sensitivity')
            ax[0].set_yticks(y_pos)
            ax[0].set_yticklabels(self.global_bounds_DataArray['fuzzy_variable'].values)
            ax[0].invert_yaxis()  # labels read top-to-bottom
            ax[0].set_ylabel('Variables')
            ax[0].set_xlabel('Total Sensitivity')
            ax[0].set_axisbelow(True)
            ax[0].set_xlim(0,max(self.sensis_total)+0.1)
            ax[0].grid(b=True, which='major')
            ax[0].grid(b=True, which='minor')
            formatted_sensis = [round(elem, 3) for elem in self.sensis_total]
            for i, v in enumerate(formatted_sensis):
                ax[0].text(v+0.001, i-0.055, " "+str(v), color='black', va='center',fontweight='bold')
            ax[0].barh(y_pos, self.sensis_total, width, xerr=self.sensis_total_var, align='center', color='lightslategrey', ecolor='black')
            ax[1].set_title('Average Absolute Left / Right Sensitivity')
            ax[1].set_yticks(y_pos)
            ax[1].set_yticklabels(self.global_bounds_DataArray['fuzzy_variable'].values)
            ax[1].invert_yaxis()  # labels read top-to-bottom
            ax[1].set_ylabel('Variables')
            ax[1].set_xlabel('Relative Partial Sensitivity')
            ax[1].set_axisbelow(True)
            ax[1].set_xlim(0,max(max(self.sensis_l),max(self.sensis_r))+0.1)
            ax[1].grid(b=True, which='major')
            ax[1].grid(b=True, which='minor')
            formatted_sensis_l = [round(elem, 3) for elem in self.sensis_l]
            formatted_sensis_r = [round(elem, 3) for elem in self.sensis_r]
            for (i_l, v_l),(i_r, v_r) in zip(enumerate(formatted_sensis_l),enumerate(formatted_sensis_r)):
                ax[1].text(v_l+0.001, i_l-0.5*width2-0.055, " "+str(v_l), color='black', va='center',fontweight='bold')
                ax[1].text(v_r+0.001, i_r+0.5*width2+0.055, " "+str(v_r), color='black', va='center',fontweight='bold')
            ax[1].barh(y_pos-0.5*width2, self.sensis_l, width2, xerr=self.sensis_l_var, color='SkyBlue', label='Left', ecolor='black')
            ax[1].barh(y_pos+0.5*width2, self.sensis_r, width2, xerr=self.sensis_r_var, color='IndianRed', label='Right', ecolor='black')
            ax[1].legend(fancybox=True, framealpha=0.5)

        else:

            fig.suptitle('Fuzzy Sensitivity Analysis')
            ax[0].set_title('Total Relative Sensitivity')
            ax[0].set_yticks(y_pos)
            ax[0].set_yticklabels(self.global_bounds_DataArray['fuzzy_variable'].values)
            ax[0].invert_yaxis()  # labels read top-to-bottom
            ax[0].set_ylabel('Variables')
            ax[0].set_xlabel('Total Sensitivity')
            ax[0].set_axisbelow(True)
            ax[0].set_xlim(left=0,right=max(self.sensis_total)+0.1)
            ax[0].grid(b=True, which='major')
            ax[0].grid(b=True, which='minor')
            formatted_sensis = [round(elem, 3) for elem in self.sensis_total]
            for i, v in enumerate(formatted_sensis):
                ax[0].text(v+0.001, i-0.055, " "+str(v), color='black', va='center',fontweight='bold')
            ax[0].barh(y_pos, self.sensis_total, width, align='center', color='lightslategrey')
            ax[1].set_title('Absolute Left / Right Sensitivity')
            ax[1].set_yticks(y_pos)
            ax[1].set_yticklabels(self.global_bounds_DataArray['fuzzy_variable'].values)
            ax[1].invert_yaxis()  # labels read top-to-bottom
            ax[1].set_ylabel('Variables')
            ax[1].set_xlabel('Relative Partial Sensitivity')
            ax[1].set_axisbelow(True)
            ax[1].set_xlim(0,max(max(self.sensis_l),max(self.sensis_r))+0.1)
            ax[1].grid(b=True, which='major')
            ax[1].grid(b=True, which='minor')
            formatted_sensis_l = [round(elem, 3) for elem in self.sensis_l]
            formatted_sensis_r = [round(elem, 3) for elem in self.sensis_r]
            for (i_l, v_l),(i_r, v_r) in zip(enumerate(formatted_sensis_l),enumerate(formatted_sensis_r)):
                ax[1].text(v_l+0.001, i_l-0.5*width2-0.055, " "+str(v_l), color='black', va='center',fontweight='bold')
                ax[1].text(v_r+0.001, i_r+0.5*width2+0.055, " "+str(v_r), color='black', va='center',fontweight='bold')
            ax[1].barh(y_pos-0.5*width2, self.sensis_l, width2, color='SkyBlue', label='Left')
            ax[1].barh(y_pos+0.5*width2, self.sensis_r, width2, color='IndianRed', label='Right')
            ax[1].legend(fancybox=True, framealpha=0.5)


        if plot: plt.show()
        return fig, ax


    def _sensitivity_rotuine(self, alpha_Level=None, progressbar_disable = False, lsa=None, leave=False, **kwargs):
        res_total = []
        res_l = []
        res_r = []

        s_sum = 0
        l_sum = 0
        r_sum = 0

        if alpha_Level is not None:
            self.fuzzy_variables_dict= self._change_alpha_level(alpha_Level=alpha_Level)
            self.global_bounds_DataArray = Alpha_Level_Optimization._boundary_constraints(self.fuzzy_var_list)
            self.number_of_alpha_lvl = self.global_bounds_DataArray['number_of_alpha_levels'].size

        index = np.arange(len(self.global_bounds_DataArray))

        with tqdm(total=len(index), desc='Sensitivity Analysis', disable = progressbar_disable, leave=leave) as pbar_loc:

            for i in range(len(index)):
                name_i = self.global_bounds_DataArray['fuzzy_variable'].values[i]
                df_i = self.global_bounds_DataArray.data[i]
                fuzzy_var_list_new = []
                names_new = []

                for j in range(len(index)):
                    name_j = self.global_bounds_DataArray['fuzzy_variable'].values[j]
                    df_j = self.global_bounds_DataArray.data[j]

                    if name_i == name_j:
                        fuzzy_var_list_new.append(self.fuzzy_var_list[i])
                        names_new.append(name_i)
                    else:
                        if lsa is None:
                            y = phuzzy.Uniform(alpha0=[df_j[0][1], df_j[0][2]],
                                               number_of_alpha_levels=self.number_of_alpha_lvl)
                        else:
                            y = phuzzy.Uniform(alpha0=[lsa[j],lsa[j]], number_of_alpha_levels=self.number_of_alpha_lvl)

                        fuzzy_var_list_new.append(y)

                        #names_new.append(name_j)
                        #kwargs = {**self.input_dict , **fvars}
                        #kwargs = {self.input_dict ,fvars}

                    kwargs['fuzzy_variables'] = fuzzy_var_list_new

                z = Alpha_Level_Optimization(**kwargs)
                z.main(n=self.n,iters=self.iters, progressbar_disable=True, backup=False)

                E = 1. - (z.df['r']-z.df['l'])/(z.df['r'].iloc[0]-z.df['l'].iloc[0])
                full_length = abs(z.df['r']-z.df['l'])
                E_l = abs(((z.df['l'].iloc[-1]-z.df['l'])/full_length)/(len(z.df)-1))
                E_r = abs(((z.df['r']-z.df['r'].iloc[-1])/full_length)/(len(z.df)-1))

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


        return sensis_total, sensis_l, sensis_r


    def _change_alpha_level(self, alpha_Level):
        input_fuzzy_var_list = self.fuzzy_var_list
        for i, fuzzy_var in enumerate(self.fuzzy_var_list):
            fuzzy_var.convert_df(alpha_levels=alpha_Level)
            input_fuzzy_var_list[i] = fuzzy_var
        return input_fuzzy_var_list


if __name__ == "__main__":
    pass
