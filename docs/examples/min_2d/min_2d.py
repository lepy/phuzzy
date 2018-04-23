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


Hf = 200.  # mm
Bf = 300.  # mm
fig, axs = plt.subplots(2,2, dpi=90, facecolor='w', edgecolor='k', figsize=(Bf / 25.4, Hf / 25.4))

axl = axs[0,0]
axr = axs[0,1]
axc = axs[1,1]
axz = axs[1,0]
axl.set_title("z.l")
axr.set_title("z.r")
for ax in [axl, axr, axc]:
    ax.set_xlabel("x")
    ax.set_ylabel("y")

cmap = "autumn"


csc = axc.contourf(Xs, Ys, Zs, alpha=.2, cmap=cmap)
csc2 = axc.contour(Xs, Ys, Zs, color="k", cmap=cmap)
axc.clabel(csc2, fontsize=9, inline=1, alpha=1)

norm= matplotlib.colors.Normalize(vmin=csc.vmin, vmax=csc.vmax)
sm = plt.cm.ScalarMappable(norm=norm, cmap =csc.cmap)
sm.set_array([])
cbar = fig.colorbar(sm, ticks=csc.levels)
cbar.set_label("z")
x = ph.Trapezoid(alpha0=[-1,2], alpha1=[-.5, .5])
y = ph.TruncNorm(alpha0=[-1,0.5])

z = f(x,y)
z.name = "z"
print(z.df)

for i, zi in z.df.iterrows():
    xi = x.df.loc[i]
    yi = y.df.loc[i]
    axl.plot([xi.l, xi.r, xi.r, xi.l, xi.l],
            [yi.l, yi.l, yi.r, yi.r, yi.l],
             c=sm.to_rgba(zi.l))

    axr.plot([xi.l, xi.r, xi.r, xi.l, xi.l],
            [yi.l, yi.l, yi.r, yi.r, yi.l],
             c=sm.to_rgba(zi.r))

    polygon = Polygon(np.vstack([[xi.l, xi.r, xi.r, xi.l, xi.l],
                           [yi.l, yi.l, yi.r, yi.r, yi.l]]).T)
    pl = PatchCollection([polygon], alpha=.1, color=sm.to_rgba(zi.l))
    axl.add_collection(pl)

    pl = PatchCollection([polygon], alpha=.1, color=sm.to_rgba(zi.r))
    axr.add_collection(pl)

# cs = axl.contourf(Xs, Ys, Zs, alpha=.2, cmap="hot")
cs2 = axl.contour(Xs, Ys, Zs, cmap=cmap)
# cs2k = axl.contour(Xs, Ys, Zs, colors="k")
axl.clabel(cs2, fontsize=9, inline=1, alpha=1)

# csr = axr.contourf(Xs, Ys, Zs, alpha=.2, cmap="hot")
csr2 = axr.contour(Xs, Ys, Zs, color="k", cmap=cmap)
axr.clabel(csr2, fontsize=9, inline=1, alpha=1)



axl.scatter([1], [-.4])
axr.scatter([1], [-.4])
axc.scatter([1], [-.4])

mix_mpl(z)
z.plot(ax= axz, show=False)
axz.set_title(z)
axc.set_title("f(x,y)")
fig.tight_layout()
plt.show()
