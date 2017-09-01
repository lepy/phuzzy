import logging
logger = logging.getLogger("phuzzy")
import matplotlib.pyplot as plt

import numpy as np
import phuzzy

class MPL_Mixin():

    def plot(self, filepath=None, show=False, range=None):
        """plots fuzzy number with mpl"""
        logging.debug("plots fuzzy number with mpl")
        df = self.df


        H = 170.  # mm
        B = 170.  # mm
        fig, ax = plt.subplots(dpi=90, facecolor='w', edgecolor='k', figsize=(B / 25.4, H / 25.4))
        ax.set_title("%s" % self.name)
        ax.set_xlabel('%s' % self.name)
        ax.set_ylabel(r'$\alpha$')
        ax.grid(c="gray", alpha=.5, lw=.5, dashes=[1, 3])


        xs = np.hstack([df["min"].values, df["max"].values[::-1]])
        ys = np.hstack([df["alpha"].values, df["alpha"].values[::-1]])
        ax.plot(xs, ys, lw=1, alpha=.7)
        ax.fill_between(xs, 0, ys, alpha=.2)

        ax.fill_betweenx(df["alpha"].values, df["min"].min(), df["min"].values, color="gray", alpha=.1)
        ax.fill_betweenx(df["alpha"].values, df["max"].values, df["max"].max(), color="gray", alpha=.1)
        # alphalines
        for i, row in df.iterrows():
            ax.plot([row["min"], row["max"]], [row.alpha, row.alpha], lw=.8, alpha=.7, ls="--", c="gray")

        a0 = self.alpha0
        ax.annotate('%.2f' % a0["min"], xy=(a0["min"], a0["alpha"]), xycoords='data',
                        xytext=(-2, 2), textcoords='offset points',
                        horizontalalignment='right', verticalalignment='bottom', alpha=.4)
        ax.annotate('%.2f' % a0["max"], xy=(a0["max"], a0["alpha"]), xycoords='data',
                        xytext=(2, 2), textcoords='offset points',
                        horizontalalignment='left', verticalalignment='bottom', alpha=.4)
        a1 = self.alpha1
        ax.annotate('%.2f' % a1["min"], xy=(a1["min"], a1["alpha"]), xycoords='data',
                        xytext=(-2, 2), textcoords='offset points',
                        horizontalalignment='right', verticalalignment='bottom', alpha=.4)
        ax.annotate('%.2f' % a1["max"], xy=(a1["max"], a1["alpha"]), xycoords='data',
                        xytext=(2, 2), textcoords='offset points',
                        horizontalalignment='left', verticalalignment='bottom', alpha=.4)
        dx = abs(self.alpha0["max"] - self.alpha0["min"])
        ax.set_xlim(self.alpha0["min"] - 0.2 * dx, self.alpha0["max"] + 0.2 * dx)
        ax.set_ylim(0, 1.1)
        if range is not None:
            ax.set_xlim(range)
        if filepath:
            fig.savefig(filepath, dpi=300)
        if show is True:
            print("show")
            plt.show()

class Triangle(phuzzy.Triangle, MPL_Mixin):
    def __init__(self, **kwargs):
        phuzzy.Triangle.__init__(self, **kwargs)

class Trapezoid(phuzzy.Trapezoid, MPL_Mixin):
    def __init__(self, **kwargs):
        phuzzy.Trapezoid.__init__(self, **kwargs)

class TruncNorm(phuzzy.TruncNorm, MPL_Mixin):
    def __init__(self, **kwargs):
        phuzzy.TruncNorm.__init__(self, **kwargs)