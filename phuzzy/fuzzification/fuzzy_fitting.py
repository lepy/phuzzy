import numpy as np
import pandas as pd
import scipy.stats as st

import phuzzy.mpl as phm
import phuzzy.mpl.plots

import warnings
import matplotlib.pyplot as plt
plt.style.use('seaborn')




class Data_Fitting(object):

    def __init__(self):
        pass


    def best_fit_distribution(self, data, number_of_alpha_levels=6, bins=False, bootstrap=False, ax=True,
                              filepath=None):
        """
        Routine to determin automatically a suiting membership function to a given data set

        :param data:                        input array
        :param number_of_alpha_levels:      level of discritization of the membership function
        :param bins:                        number of histogram bins
        :param bootstrap:                   applying / not applying bootstrap algorithm
        :param ax:                          plot command
        :param filepath:                    safe plot

        :return:                            fuzzy parameter
        """

        """Model data by finding best fit distribution to data"""

        if bootstrap:
            data = self._bootstrap(data)

        # Get histogram of original data
        if bins:
            y, x = np.histogram(data, bins=bins, density=True)
        else:
            y, x = np.histogram(data, bins='fd', density=True)

        y, x = np.histogram(data, density=True)
        x = (x + np.roll(x, -1))[:-1] / 2.0
        DISTRIBUTIONS = [
            st.triang,
            st.norm,
            st.uniform
        ]
        # Best holders
        best_distribution = st.triang
        best_params = (0.0, 1.0)
        best_sse = np.inf
        dists = []
        # Estimate distribution parameters from data
        for distribution in DISTRIBUTIONS:
            # Try to fit the distribution
            try:
                # Ignore warnings from data that can't be fit
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore')
                    # fit dist to data
                    params = distribution.fit(data)
                    # Separate parts of parameters
                    arg = params[:-2]
                    loc = params[-2]
                    scale = params[-1]
                    # Calculate fitted PDF and error with fit in distribution
                    pdf = distribution.pdf(x, loc=loc, scale=scale, *arg)
                    sse = np.sum(np.power(y - pdf, 2.0))
                    dist = distribution(loc=loc, scale=scale, *arg)
                    dists.append([dist, loc, scale, arg])
                    # if axis pass in add to plot
                    #try:
                    #    if ax:
                    #        pd.Series(pdf, x).plot(ax=ax)
                    #except Exception:
                    #    pass
                    # identify if this distribution is better
                    if best_sse > sse > 0:
                        best_distribution = distribution.name
                        best_params = params
                        best_sse = sse
            except Exception:
                pass

        x = np.linspace(data.min(),data.max(), 500)

        if best_distribution == 'norm':
            dist, loc, scale, args = dists[0]
            a = max(abs(loc-data.min()), abs(loc-data.max()))
            fuzzy_var = phm.TruncNorm(alpha0=[loc-a, loc+a], alpha1=[data.mean()],
                                      number_of_alpha_levels=number_of_alpha_levels)
            #return (tgn, tgn.get_shape())

        elif best_distribution == 'triang':
            dist, loc, scale, args = dists[0]
            y = dist.pdf(x)
            y /= y.max()
            df = pd.DataFrame({"x":x, "y":y})
            loc = df.loc[df.y.idxmax()].x
            fuzzy_var = phm.Triangle(alpha0=[data.min(), data.max()], alpha1=[loc],
                                     number_of_alpha_levels=number_of_alpha_levels)
            #return (tria, tria.get_shape())

        elif best_distribution == 'uniform':
            dist, loc, scale, args = dists[1]
            fuzzy_var = phm.Uniform(alpha0=[data.min(),data.max()], alpha1=[1.00,1.00],
                                    number_of_alpha_levels=number_of_alpha_levels)
            #return (uni, uni.get_shape())
        else:
            raise ValueError


        if ax:
            fig, ax = plt.subplots(1,1, figsize=(8,5))
            x = np.linspace(data.min(),data.max(), 500)
            y = dist.pdf(x)
            y /= y.max()
            df = pd.DataFrame({"x":x, "y":y})
            ax.plot(x,y, label="hist fit", color="g", lw=2, alpha=.8, ls="--")
            ax.axvline(data.min(), dashes=[5,2,1,2], c="k", alpha=.5)
            ax.axvline(data.max(), dashes=[5,2,1,2], c="k", alpha=.5)
            ax.axvline(loc, dashes=[5,2,1,2], c="k", alpha=.5)
            ax.scatter(data, np.ones_like(data)*(-.1), color="g", alpha=.4, label="data")
            ax.set_ylabel(r"$\alpha$ [$-$]")
            fuzzy_var_shape = fuzzy_var.get_shape()
            ax.plot(fuzzy_var_shape.x, fuzzy_var_shape.alpha, label="Phuzzy Variable", alpha=.4, color="r")

            if bins:
                phuzzy.mpl.plots.plot_hist(data, bins=bins, ax=ax, normed=True, color="b",
                                           filled=True, alpha=.3, label='histo data')
            else:
                phuzzy.mpl.plots.plot_hist(data, ax=ax, bins='fd', normed=True, color="b",
                                           filled=True, alpha=.3, label='histo data')

            if best_distribution == 'norm': ax.set_title('TruncNorm')
            elif best_distribution == 'triang': ax.set_title('Triangle')
            elif best_distribution == 'uniform': ax.set_title('Uniform')

            ax.legend(fancybox=True, framealpha=0.5, loc=1)
            plt.show()

            if filepath:
                fig.savefig(filepath, dpi=360)


        return fuzzy_var


    def _bootstrap(self,data):
        """
        Bootstrapping Algorithm for stretching data

        :param data:        Input Data
        :return:            Boostrapped Data
        """
        xbar = np.zeros(shape=1000)
        for i in range(1000):
            sample = data[np.random.randint(0,len(data),size=len(data))]
            xbar[i] = np.mean(sample)
        np.append(xbar,data.min())
        np.append(xbar,data.max())
        return xbar


    def fit_plot(self,data,fuzzy_var,dist):

        fig, ax = plt.subplots(1,1, figsize=(10,5))

        x = np.linspace(data.min(),data.max(), 500)
        y = dist.pdf(x)
        y /= y.max()
        df = pd.DataFrame({"x":x, "y":y})
        # pdf = make_pdf(st.norm, [loc, scale]+args)
        ax.plot(x,y, label="hist fit", color="g", lw=2, alpha=.8, ls="--")
        dist, loc, scale, args = dist[0]

        ax.axvline(data.min(), dashes=[5,2,1,2], c="k", alpha=.5)
        ax.axvline(data.max(), dashes=[5,2,1,2], c="k", alpha=.5)
        ax.axvline(loc, dashes=[5,2,1,2], c="k", alpha=.5)
        ax.scatter(data, np.ones_like(data)*(-.1), color="g", alpha=.4, label="data")


