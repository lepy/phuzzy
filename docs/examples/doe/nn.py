print(__doc__)

# Author: Alexandre Gramfort <alexandre.gramfort@inria.fr>
#         Fabian Pedregosa <fabian.pedregosa@inria.fr>
#
# License: BSD 3 clause (C) INRIA

# -*- coding: utf-8 -*-
import phuzzy
from phuzzy.mpl import mix_mpl
import phuzzy.mpl.plots
import matplotlib.pyplot as plt
import phuzzy.approx.doe
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import phuzzy.mpl.plots

x = phuzzy.TruncNorm(alpha0=[1, 2], name="x")
y = phuzzy.TruncNorm(alpha0=[1, 2], name="y")
# y = phuzzy.Triangle(alpha0=[3, 6], alpha1=[4], name="y")

mix_mpl(x)
mix_mpl(y)

def f(x1, x2):
    return (x1-1)**2+(x2+.4)**2 + .1

z = f(x,y)

doe = phuzzy.approx.doe.DOE(designvars = [x,y], name="xy")
print(doe)

# #############################################################################
# Generate sample data
import numpy as np
import matplotlib.pyplot as plt
from sklearn import neighbors

# np.random.seed(0)
# X = np.sort(5 * np.random.rand(40, 1), axis=0)
# T = np.linspace(0, 5, 500)[:, np.newaxis]
# y = np.sin(X).ravel()
#
# # Add noise to targets
# y[::5] += 1 * (0.5 - np.random.rand(8))

samples = doe.sample_doe(n=20, method="cc").copy()
X = samples[["x", "y"]].values
y = f(samples.x, samples.y)
samples["res"] = y
# print("samples", samples)

#
z_a = doe.eval(f, [samples.x, samples.y], method="cc", n_samples=20, name="z_a")
# # print(z_a)
# mix_mpl(z_a)
# z_a.plot()

samplesm = doe.sample_doe(n=10000, method="meshgrid").copy()
T = samplesm[["x", "y"]].values

# #############################################################################
# Fit regression model
n_neighbors = 5
weights = "distance"

# for i, weights in enumerate(['uniform', 'distance']):
for i, n_neighbors in enumerate(range(3, 4)):
    print("o")
    knn = neighbors.KNeighborsRegressor(n_neighbors, weights=weights)
    y_ = knn.fit(X, y).predict(T)
    samplesm["res"] = y_

    print("a")
    z_b = phuzzy.FuzzyNumber.from_results(samplesm[["res", "alpha"]], name="z_b")
    z_b.convert_df(alpha_levels=11)
    # mix_mpl(z_b)
    # z_b.plot()
    print("b")
    fig, axs = phuzzy.mpl.plots.plot_xyz(z, z_a, z_b)
    mix_mpl(z)
    z.plot(ax=axs[1])
    z.plot(ax=axs[2])
    axs[2].set_title("%d - %s" % (n_neighbors, weights))

    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # # print("samplesm.alpha.values")
    # # print(samplesm.alpha.values.max())
    #
    # ax.scatter(samples.x.values, samples.y.values, samples.alpha.values, c='k', label='data', s=1)
    # # plt.plot(samplesm.x.values, samplesm.y.values, samplesm.alpha.values, c='g', label='prediction', lw=0, marker="o", ms=2)
    # # ax.scatter(samplesm.x.values, samplesm.y.values, samplesm.alpha.values, c='g', label='prediction', s=1)
    # ax.axis('tight')
    # ax.legend()
    # ax.title("KNeighborsRegressor (k = %i, weights = '%s')" % (n_neighbors, weights))
    print("c")
plt.show()
