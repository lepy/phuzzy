import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import phuzzy.mpl as phm

from scipy import interpolate, optimize
from pandas._libs.testing import isna






class New_Data_Fitting(object):

    def __init__(self):
        pass

    def create_fuzzy_variable(self, input_link=None, type='Triangle', number_of_alpha_levels=5, alpha_level_1_method= 'maxVal',
                              alpha_level_0_method='direct', name=None, bin_calc_method='scott', custom_bins=25):

        if input_link is None: raise ValueError('PLEASE ADD INPUT LINK')
        if name is None: self.name = 'Input Data'

        input_df = self.read_input_csv(input_link=input_link)
        self.histogram_df = self.create_histogram(input_df=input_df, bin_calc_method=bin_calc_method, custom_bins=custom_bins)

        alpha_lvl_0 = self.set_alpha_level_0(histogram_df=self.histogram_df, alpha_level_0_method=alpha_level_0_method)
        alpha_lvl_1 = self.set_alpha_level_1(histogram_df=self.histogram_df,alpha_level_1_method=alpha_level_1_method)

        self.fuzzy_variable = self.fitting(type=type,number_of_alpha_levels=number_of_alpha_levels,
                                      alpha_lvl_0=alpha_lvl_0,alpha_lvl_1=alpha_lvl_1)


        #self.plot_data()
        #r = 1

    def plot_data(self):

        #phm.plot_bar(y=self.histogram_df['absBinFrequencyNorm'],x=self.histogram_df['center'])

        binsSize = self.histogram_df['widthR'].values[0]-self.histogram_df['widthL'].values[0]
        plt.bar(height=self.histogram_df['absBinFrequencyNorm'].values,x=self.histogram_df['center'].values,width=binsSize)
        self.fuzzy_variable.plot()

        plt.show()

        self.histogram_df['absBinFrequencyNorm'].plot.hist(grid=True, bins=20, rwidth=0.9,
                           color='#607c8e')
        bins = np.concatenate((self.histogram_df['widthL'].values,self.histogram_df['widthR'].values[-1]),axis=None)
        plt.hist2d(bins, self.histogram_df['absBinFrequencyNorm'], 'r--', linewidth=1)


    def read_input_csv(self, input_link):
        return pd.read_csv(input_link,sep=',',header=None)


    def minimization_dist_function(self,xx):
        if self.type == 'TruncGenNorm':
            func = phm.TruncGenNorm(alpha0=self.histo_class.alpha0, alpha1=self.histo_class.alpha1, number_of_alpha_levels=self.set_number_of_alpha_levels, beta=xx[0])
        elif self.type == 'Superellipse':
            func = phm.Superellipse(alpha0=self.histo_class.alpha0, alpha1=self.histo_class.alpha1, m= xx[0], n=xx[1], number_of_alpha_levels=self.set_number_of_alpha_levels)
        elif self.type == 'TruncGenSkewNorm':
            func = phm.TruncGenSkewNorm(alpha0=self.histo_class.alpha0, alpha1=self.histo_class.alpha1, number_of_alpha_levels=self.set_number_of_alpha_levels, a=xx[0])
        x_values = []
        alpha_values = []
        #for index_histo, row_histo in self.histo_class.histo_df.iterrows():
        for i in range(len(self.histo_class.histo_df)):
            #x = row_histo['center']
            x = self.histo_class.histo_df.iloc[i]["center"]
            x_values.append(x)
            loop = 0
            if x < func._df.iloc[-1]["l"]:
                for i in range(0,len(func._df)-1):
                    if x >= func._df.iloc[i]["l"] and x <= func._df.iloc[i+1]["l"] and loop == 0:
                        x_interpol = [func._df.iloc[i]["l"], func._df.iloc[i+1]["l"]]
                        y_interpol = [func._df.iloc[i]["alpha"], func._df.iloc[i+1]["alpha"]]
                        f = interpolate.interp1d(x_interpol, y_interpol)
                        alpha_x = f(x)
                        alpha_values.append(alpha_x.item(0))
                        loop = 1
            elif x >= func._df.iloc[-1]["r"]:
                for i in range(0,len(func._df)-1):
                    if x <= func._df.iloc[i]["r"] and x >= func._df.iloc[i+1]["r"] and loop == 0:
                        x_interpol = [func._df.iloc[i]["r"], func._df.iloc[i+1]["r"]]
                        y_interpol = [func._df.iloc[i]["alpha"], func._df.iloc[i+1]["alpha"]]
                        f = interpolate.interp1d(x_interpol, y_interpol)
                        alpha_x = f(x)
                        alpha_values.append(alpha_x.item(0))
                        loop = 1
            else:
                alpha_values.append(1.0)
        #calc dist
        alpha_values_hist = self.histo_class.histo_df['absBinFrequencyNorm'].values
        alpha_values_plot = alpha_values
        return sum(abs(alpha_values_plot-alpha_values_hist))


    def create_histogram(self,input_df,bin_calc_method,custom_bins=None):

        #Bins Size & Number of Bins Definition
        if bin_calc_method == 'scott':
            binSize = ((3.49 * input_df.values.std()) / (len(input_df))**(1/3))
            nbins = (int(round((input_df.values.max() - input_df.values.min()) / binSize)))
        elif bin_calc_method == 'freedman':
            binSize = ((2*((input_df.quantile(0.75) - input_df.quantile(0.25)).values)) / (len(input_df))**(1/3)).item(0)
            nbins = (int(round((input_df.values.max() - input_df.values.min()) / binSize)))
        elif bin_calc_method == 'custom':
            nbins = custom_bins
            binSize = (input_df.values.max() - input_df.values.min()) / nbins
        elif bin_calc_method == 'rice':
            nbins = int(round((2*len(input_df)**(1/3))))                      #Rice Rule
            binSize = (input_df.values.max() - input_df.values.min()) / nbins

        #Calculate Histogram Data
        values = np.ravel((input_df._convert(datetime=True)._get_numeric_data()))
        values = values[~isna(values)]
        abs_bin_frequency, binsIntval = np.histogram(values, bins=nbins)
        #abs_bin_frequency_norm = (abs_bin_frequency - abs_bin_frequency.min()) / (abs_bin_frequency.max() - abs_bin_frequency.min())
        abs_bin_frequency_norm = (abs_bin_frequency - 0.0) / (abs_bin_frequency.max() - 0.0)

        #Histo Array Pandas DataFrame creation
        histo_array = np.concatenate((abs_bin_frequency,abs_bin_frequency_norm,binsIntval[0:-1],binsIntval[1:],0.5*(binsIntval[1:]+binsIntval[0:-1])),axis=None)
        histo_array = histo_array.reshape(len(abs_bin_frequency),5).T
        histogram_df = pd.DataFrame(columns=["absBinFrequency", "absBinFrequencyNorm", "widthL", "widthR","center"],data=histo_array, dtype=np.float)
        return histogram_df


    def set_alpha_level_1(self,histogram_df,alpha_level_1_method):
        if alpha_level_1_method == 'maxVal':
            maxValRow = histogram_df.loc[histogram_df['absBinFrequency'].idxmax()]
            if histogram_df.loc[histogram_df['absBinFrequency'].idxmax()].name == 0:
                maxMeanVal = maxValRow[2]
                alpha1 = [maxMeanVal, maxMeanVal]
            elif histogram_df.loc[histogram_df['absBinFrequency'].idxmax()].name == (len(histogram_df.index)-1):
                maxMeanVal = maxValRow[3]
                alpha1 = [maxMeanVal, maxMeanVal]
            else:
                maxMeanVal = maxValRow[4]
                alpha1 = [maxMeanVal, maxMeanVal]
        elif alpha_level_1_method == 'mean':
            meanVal = histogram_df.mean()
            alpha1 = [meanVal.values.item(0), meanVal.values.item(0)]
        elif alpha_level_1_method == 'modus':
            modusVal = histogram_df.mode()
            if len(modusVal) == 1:
                alpha1 = [modusVal.values.item(0), modusVal.values.item(0)]
            else:
                alpha1 = [modusVal.values.min(), modusVal.values.max()]
        elif alpha_level_1_method == 'gnstd':
            meanVal = histogram_df.mean()
            stdVal = histogram_df.std()
            alpha1 = [meanVal.values.item(0)-(0.50*stdVal.values.item(0)), meanVal.values.item(0)+(0.50*stdVal.values.item(0))]
        return np.array(alpha1)


    def set_alpha_level_0(self,histogram_df,alpha_level_0_method):
        if alpha_level_0_method == 'direct':
            alpha0L = histogram_df['widthL'].min()
            alpha0R = histogram_df['widthR'].max()
        elif alpha_level_0_method == 'saftyfactor':
            binSize = histogram_df['widthR'][0]-histogram_df['widthL'][0]
            alpha0L = histogram_df['widthL'].min() - 0.5*binSize
            alpha0R = histogram_df['widthR'].max() + 0.5*binSize
        return np.array([alpha0L, alpha0R])


    def fitting(self,type,number_of_alpha_levels,alpha_lvl_0,alpha_lvl_1):

        # Optimization Call
        if type == 'TruncGenNorm':
            res = optimize.minimize(self.minimization_dist_function, [1.0], method='Nelder-Mead', tol=1e-03,options={'disp': True})
        elif type == 'Superellipse':
            res = optimize.minimize(self.minimization_dist_function, [1.0,1.0], method='Nelder-Mead', tol=1e-03,options={'disp': True})
        elif type == 'TruncGenSkewNorm':
            res = optimize.minimize(self.minimization_dist_function, [1.0], method='Nelder-Mead', tol=1e-03,options={'disp': True})

        # Fitting Functions & Plot
        if type == 'TruncGenNorm':
            fuzzy_var = phm.TruncGenNorm(alpha0=alpha_lvl_0, alpha1=alpha_lvl_1, number_of_alpha_levels=number_of_alpha_levels, beta=res.x[0])
            #tgn.plot(show=True, filepath="truncgennorm.png", title=True, input_df = self.histo_class)
        elif type == 'Superellipse':
            fuzzy_var = phm.Superellipse(alpha0=alpha_lvl_0, alpha1=alpha_lvl_1, m= res.x[0], n=res.x[1], number_of_alpha_levels=number_of_alpha_levels)
            #se.plot(show=True, filepath="truncgennorm.png", title=True, input_df = self.histo_class)
        elif type == 'TruncGenSkewNorm':
            fuzzy_var = phm.TruncGenSkewNorm(alpha0=alpha_lvl_0, alpha1=alpha_lvl_1, number_of_alpha_levels=number_of_alpha_levels, a=res.x[0])
            #tgsn.plot(show=True, filepath="truncgennorm.png", title=True, input_df = self.histo_class)
        elif type == 'Triangle':
            fuzzy_var = phm.Triangle(alpha0=alpha_lvl_0, alpha1=alpha_lvl_1, number_of_alpha_levels=number_of_alpha_levels)
            #tri.plot(show=True, filepath="truncgennorm.png", title=True, input_df = self.histo_class)
        elif type == 'Trapezoid':
            fuzzy_var = phm.Trapezoid(alpha0=alpha_lvl_0, alpha1=alpha_lvl_1, number_of_alpha_levels=number_of_alpha_levels)
            #tra.plot(show=True, filepath="truncgennorm.png", title=True, input_df = self.histo_class)
        elif type == 'TruncNorm':
            fuzzy_var = phm.TruncNorm(alpha0=alpha_lvl_0, number_of_alpha_levels=number_of_alpha_levels)
            #tn.plot(show=True, filepath="truncgennorm.png", title=True, input_df = self.histo_class)
        return fuzzy_var



































