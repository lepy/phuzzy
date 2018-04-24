# -*- coding: utf-8 -*-
import phuzzy
from phuzzy.mpl import mix_mpl
import phuzzy.mpl.plots
import matplotlib.pyplot as plt
import phuzzy.approx.doe
import numpy as np
import pandas as pd

x = phuzzy.TruncNorm(alpha0=[1, 2], name="x")
y = phuzzy.TruncNorm(alpha0=[1, 2], name="y")
# y = phuzzy.Triangle(alpha0=[3, 6], alpha1=[4], name="y")

mix_mpl(x)
mix_mpl(y)

doe = phuzzy.approx.doe.DOE(designvars = [x,y], name="xy")
doe.sample_doe(n=10, method="cc")
print(doe)
print(doe.samples)

fig, axs = phuzzy.mpl.plots.plot_xy_3d(x,y)

axx = axs[0,0]
axy = axs[1,1]
axxy = axs[1,0]
ax3d = axs[0,1]

axxy.scatter(doe.samples.x.values, doe.samples.y.values, c="r")
axx.scatter(doe.samples.x.values, np.zeros_like(doe.samples.x.values), c="r")
axy.scatter(np.zeros_like(doe.samples.x.values), doe.samples.y.values, c="r")
ax3d.scatter(doe.samples.x.values, doe.samples.y.values, np.zeros_like(doe.samples.x.values), c="r")

for i, row in doe.samples.iterrows():
    axx.axvline(row.x, alpha=.2, c="r")
    axy.axhline(row.y, alpha=.2, c="r")

def f(x1, x2):
    return (x1-1)**2+(x2+.4)**2 + .1

xs = np.linspace(-1, 2, 100)
ys = np.linspace(-1, 1, 100)

Xs, Ys = np.meshgrid(xs, ys)

Zs = f(Xs, Ys)

z = f(x,y)
z.name = "z"
print("z", z.df)
print("z", z.min(), z.max())

z_a = doe.eval(f, [x,y], method="cc")
print(z_a)

print("run function %d times." % len(z_a.samples))

fig, axs = phuzzy.mpl.plots.plot_xy(z, z_a)
axs[1].scatter(z_a.df_res.res, z_a.df_res.alpha)

plt.show()
