# -*- coding: utf-8 -*-

import phuzzy as ph
from phuzzy.mpl import mix_mpl
import matplotlib.pyplot as plt
import numpy as np
# import matplotlib.mlab as mlab
import matplotlib.colors
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection

def f(x1, x2):
    return (x1-1)**2+(x2+.4)**2 + .1

xs = np.linspace(-1, 2, 100)
ys = np.linspace(-1, 1, 100)

Xs, Ys = np.meshgrid(xs, ys)

Zs = f(Xs, Ys)

x = ph.Trapezoid(alpha0=[-1,2], alpha1=[-.5, .5])
y = ph.TruncNorm(alpha0=[-1,0.5])

z = f(x,y)
z.name = "z"
print(z.df)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, PathPatch,Rectangle
# register Axes3D class with matplotlib by importing Axes3D
from mpl_toolkits.mplot3d import Axes3D
import mpl_toolkits.mplot3d.art3d as art3d
from matplotlib.text import TextPath
from matplotlib.transforms import Affine2D

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
# p = Circle((5, 5), 3)
# ax.add_patch(p)
# # art3d.pathpatch_2d_to_3d(p, z=0, zdir="x")
# art3d.pathpatch_2d_to_3d(p, z=2)
# rectangle = Rectangle((4, 4), 5, 4, fc="r")
# ax.add_patch(rectangle)
# art3d.pathpatch_2d_to_3d(rectangle, z=1)
ax.set_xlim(-2, 3)
ax.set_ylim(-2, 1)
ax.set_zlim(0, 1)

for i, zi in z.df.iterrows():
    xi = x.df.loc[i]
    yi = y.df.loc[i]
    polygon = Polygon(np.vstack([[xi.l, xi.r, xi.r, xi.l, xi.l],
                           [yi.l, yi.l, yi.r, yi.r, yi.l]]).T,
                      alpha=.2)
    ax.add_patch(polygon)
    art3d.pathpatch_2d_to_3d(polygon, z=yi.alpha)

    ax.plot([xi.l, xi.r, xi.r, xi.l, xi.l],
            [yi.l, yi.l, yi.r, yi.r, yi.l],
            [yi.alpha, yi.alpha, yi.alpha, yi.alpha, yi.alpha],
             c="k", alpha=.5, lw=1.5)

ax.plot(x.df.l, y.df.l, y.df.alpha, c="k", alpha=.5, lw=1.5)
ax.plot(x.df.l, y.df.r, y.df.alpha, c="k", alpha=.5, lw=1.5)
ax.plot(x.df.r, y.df.l, y.df.alpha, c="k", alpha=.5, lw=1.5)
ax.plot(x.df.r, y.df.r, y.df.alpha, c="k", alpha=.5, lw=1.5)

ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel(r"$\alpha$")
fig.tight_layout()
plt.show()
