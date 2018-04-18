import logging
logger = logging.getLogger("phuzzy")
import os
import matplotlib
if os.environ.get('DISPLAY','') == '':
    print('no display found. Using non-interactive Agg backend')
    matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
import phuzzy

def extend_instance(obj, cls):
    """Apply mixins to a class instance after creation"""
    base_cls = obj.__class__
    base_cls_name = obj.__class__.__name__
    obj.__class__ = type(base_cls_name, (base_cls, cls),{})


def mix_mpl(obj):
    base_cls = obj.__class__
    base_cls_name = obj.__class__.__name__
    obj.__class__ = type(base_cls_name, (base_cls, MPL_Mixin),{})


class MPL_Mixin():

    def plot(self, ax=None, filepath=None, show=False, xlim=None, labels=True, title=False):
        """plots fuzzy number with mpl"""
        logging.debug("plots fuzzy number with mpl")
        df = self.df

        if ax is None:
            H = 100.  # mm
            B = 100.  # mm
            fig, ax = plt.subplots(dpi=90, facecolor='w', edgecolor='k', figsize=(B / 25.4, H / 25.4))

        if title is True:
            ax.set_title("%s" % self.__class__.__name__)
        if labels is True:
            ax.set_xlabel('%s' % self.name)
            ax.set_ylabel(r'$\alpha$')
            ax.grid(c="gray", alpha=.5, lw=.5, dashes=[1, 3])


        xs = np.hstack([df["l"].values, df["r"].values[::-1]])
        ys = np.hstack([df["alpha"].values, df["alpha"].values[::-1]])
        ax.plot(xs, ys, lw=1, alpha=.7)
        ax.fill_between(xs, 0, ys, alpha=.2)

        ax.fill_betweenx(df["alpha"].values, df["l"].min(), df["l"].values, color="gray", alpha=.1)
        ax.fill_betweenx(df["alpha"].values, df["r"].values, df["r"].max(), color="gray", alpha=.1)
        # alphalines
        for _, row in df.iterrows():
            ax.plot([row["l"], row["r"]], [row.alpha, row.alpha], lw=.8, alpha=.7, ls="--", c="gray")

        if labels is True:
            a0 = self.alpha0
            # ax.set_title("%s" % self.__class__.__name__)
            ax.set_xlabel('%s' % self.name)
            ax.set_ylabel(r'$\alpha$')
            ax.grid(c="gray", alpha=.5, lw=.5, dashes=[1, 3])
            ax.annotate('%.2f' % a0["l"], xy=(a0["l"], a0["alpha"]), xycoords='data',
                        xytext=(-2, 2), textcoords='offset points',
                        horizontalalignment='right', verticalalignment='bottom', alpha=.4)
            ax.annotate('%.2f' % a0["r"], xy=(a0["r"], a0["alpha"]), xycoords='data',
                        xytext=(2, 2), textcoords='offset points',
                        horizontalalignment='left', verticalalignment='bottom', alpha=.4)
            a1 = self.alpha1
            ax.annotate('%.2f' % a1["l"], xy=(a1["l"], a1["alpha"]), xycoords='data',
                        xytext=(-2, 2), textcoords='offset points',
                        horizontalalignment='right', verticalalignment='bottom', alpha=.4)
            ax.annotate('%.2f' % a1["r"], xy=(a1["r"], a1["alpha"]), xycoords='data',
                        xytext=(2, 2), textcoords='offset points',
                        horizontalalignment='left', verticalalignment='bottom', alpha=.4)
        dx = abs(self.alpha0["r"] - self.alpha0["l"])
        ax.set_xlim(self.alpha0["l"] - 0.2 * dx, self.alpha0["r"] + 0.2 * dx)
        ax.set_ylim(0, 1.1)
        if xlim is not None:
            ax.set_xlim(xlim)
        try:
            fig.tight_layout()
            if filepath:
                fig.savefig(filepath, dpi=90)
        except UnboundLocalError:
            pass

        if show is True:
            plt.show()

        return True

class Uniform(phuzzy.Uniform, MPL_Mixin):
    """Uniform fuzzy number with matplotlib mixin"""
    def __init__(self, **kwargs):
        phuzzy.Uniform.__init__(self, **kwargs)

class Triangle(phuzzy.Triangle, MPL_Mixin):
    """Triangle fuzzy number with matplotlib mixin"""
    def __init__(self, **kwargs):
        phuzzy.Triangle.__init__(self, **kwargs)

class Trapezoid(phuzzy.Trapezoid, MPL_Mixin):
    """Trapezoid fuzzy number with matplotlib mixin"""
    def __init__(self, **kwargs):
        phuzzy.Trapezoid.__init__(self, **kwargs)

class TruncNorm(phuzzy.TruncNorm, MPL_Mixin):
    """TruncNorm fuzzy number with matplotlib mixin"""
    def __init__(self, **kwargs):
        phuzzy.TruncNorm.__init__(self, **kwargs)

class TruncGenNorm(phuzzy.TruncGenNorm, MPL_Mixin):
    """Truncates general normal fuzzy number with matplotlib mixin"""
    def __init__(self, **kwargs):
        phuzzy.TruncGenNorm.__init__(self, **kwargs)

class Superellipse(phuzzy.Superellipse, MPL_Mixin):
    """Superellipse fuzzy number with matplotlib mixin"""
    def __init__(self, **kwargs):
        phuzzy.Superellipse.__init__(self, **kwargs)
