# -*- coding: utf-8 -*-
import time
import sys
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

import matplotlib.pyplot as plt
from sklearn import neighbors
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV

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
samples = doe.sample_doe(n=20, method="cc").copy()
X = samples[["x", "y"]].values
y = f(samples.x, samples.y)
samples["res"] = y
print(doe)
print(doe.samples)
# print("samples", samples)

#
z_a = doe.eval(f, [samples.x, samples.y], method="cc", n_samples=20, name="z_a")
# # print(z_a)
# mix_mpl(z_a)
# z_a.plot()

samplesm = doe.sample_doe(n=1000, method="meshgrid").copy()
T = samplesm[["x", "y"]].values

## SVM

svr = GridSearchCV(SVR(kernel='rbf', gamma=.1), cv=5,
                       param_grid={"C": [1e0, 1e1, 1e2, 1e3, ],
                                   "gamma": np.logspace(-2, 2, num=5)})

train_size = int(len(samples)*.75)
print("train_size", train_size)
t0 = time.time()
svr.fit(X[:train_size], y[:train_size])
svr_fit = time.time() - t0
print("SVR complexity and bandwidth selected and model fitted in %.3f s"
      % svr_fit)

t0 = time.time()
y_svr = svr.predict(T)
svr_predict = time.time() - t0
print("SVR prediction for %d inputs in %.3f s"
      % (T.shape[0], svr_predict))


samplesm["res"] = y_svr

print("z_a.df_res",z_a.df_res)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
# print("samplesm.alpha.values")
# print(samplesm.alpha.values.max())

ax.scatter(samples.x.values, samples.y.values, samples.alpha.values, c='k', label='data', s=1)
# plt.plot(samplesm.x.values, samplesm.y.values, samplesm.alpha.values, c='g', label='prediction', lw=0, marker="o", ms=2)
ax.scatter(samplesm.x.values, samplesm.y.values, samplesm.alpha.values, c='g', label='prediction', s=1)
ax.axis('tight')
ax.legend()
# ax.title("KNeighborsRegressor (k = %i, weights = '%s')" % (n_neighbors, weights))


df_res = pd.concat([z_a.df_res, samplesm[["res", "alpha"]]])

z_b = phuzzy.FuzzyNumber.from_results(df_res, name="z_b")
z_b.convert_df(alpha_levels=11)
# mix_mpl(z_b)
# z_b.plot()
fig, axs = phuzzy.mpl.plots.plot_xyz(z, z_a, z_b)
mix_mpl(z)
z.plot(ax=axs[1], labels=False, title=False)
z.plot(ax=axs[2], labels=False, title=False)
axs[2].scatter(z_a.df_res.res, z_a.df_res.alpha)

plt.show()
