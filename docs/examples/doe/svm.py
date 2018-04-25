# -*- coding: utf-8 -*-
import time
import sys
import phuzzy
from phuzzy.mpl import mix_mpl
import phuzzy.mpl.plots
import phuzzy.approx.doe
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import phuzzy.mpl.plots
import numpy as np

import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV

x = phuzzy.TruncNorm(alpha0=[1, 2], name="x")
# y = phuzzy.TruncNorm(alpha0=[3, 6], name="y")
# y = phuzzy.Triangle(alpha0=[3, 6], alpha1=[4], name="y")
y = phuzzy.Trapezoid(alpha0=[3, 6], alpha1=[4, 5], name="y")

mix_mpl(x)
mix_mpl(y)

def f(x1, x2):
    return (x1-1)**2+(x2+.4)**2 + .1

z = f(x,y)

# create DoE
doe = phuzzy.approx.doe.DOE(designvars = [x,y], name="xy")
# sample parameter sets for (expensive) evaluation
doe.sample_doe(n=10, method="lhs").copy()
# run (expensive) evaluation
z_a = doe.eval(f, ["x", "y"], name="z_a")
doe.fit_model(model="svm")

t0 = time.time()
samplesm = doe.sample_doe(n=10000, method="meshgrid").copy()
T = samplesm[["x", "y"]].values
y_svr = z_a.model.predict(T)
svr_predict = time.time() - t0
print("SVR prediction for %d inputs in %.3f s"
      % (T.shape[0], svr_predict))


samplesm["res"] = y_svr

# print("z_a.df_res",z_a.df_res)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
# print("samplesm.alpha.values")
# print(samplesm.alpha.values.max())

ax.scatter(z_a.samples.x.values, z_a.samples.y.values, z_a.samples.alpha.values, c='b', label='data', s=5)
# plt.plot(samplesm.x.values, samplesm.y.values, samplesm.alpha.values, c='g', label='prediction', lw=0, marker="o", ms=2)
ax.scatter(samplesm.x.values, samplesm.y.values, samplesm.alpha.values, c='r', label='prediction', s=2)
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
axs[2].scatter(z_a.df_res.res, z_a.df_res.alpha, s=4, alpha=.5, c="b")

plt.show()
