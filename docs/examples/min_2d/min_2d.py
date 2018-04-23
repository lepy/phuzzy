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


csc = axc.contourf(Xs, Ys, Zs, alpha=1, cmap=cmap)
csc2 = axc.contour(Xs, Ys, Zs, colors="k")
axc.clabel(csc2, fontsize=9, inline=1, alpha=1)

print(dir(csc))

norm= matplotlib.colors.Normalize(vmin=csc.zmin, vmax=csc.zmax)
sm = plt.cm.ScalarMappable(norm=norm, cmap =csc.cmap)
sm.set_array([])
cbar = fig.colorbar(sm, ticks=csc.levels)
cbar.set_label("z")
x = ph.Trapezoid(alpha0=[-1,2], alpha1=[-.5, .5])
y = ph.TruncNorm(alpha0=[-1,0.5])

z = f(x,y)
z.name = "z"
print(z.df)


# left
# cs = axl.contourf(Xs, Ys, Zs, alpha=.2, cmap="hot")
norml = matplotlib.colors.Normalize(vmin=z.df.l.min(), vmax=z.df.l.max())
sml = plt.cm.ScalarMappable(norm=norml, cmap=cmap)
sml.set_array([])
csl2 = axl.contour(Xs, Ys, Zs, colors="k")
# cs2k = axl.contour(Xs, Ys, Zs, colors="k")
axl.clabel(csl2, fontsize=9, inline=1, alpha=1)
cbarl = fig.colorbar(sml, ticks=[z.df.l.min(), z.df.l.max()], ax=axl)
cbarl.set_label("z.l")

# right
# cs = axr.contourf(Xs, Ys, Zs, alpha=.2, cmap="hot")
normr = matplotlib.colors.Normalize(vmin=z.df.r.min(), vmax=z.df.r.max())
smr = plt.cm.ScalarMappable(norm=normr, cmap=cmap)
smr.set_array([])
csr2 = axr.contour(Xs, Ys, Zs, colors="k")
# cs2k = axl.contour(Xs, Ys, Zs, colors="k")
axr.clabel(csr2, fontsize=9, inline=1, alpha=1)
cbarr = fig.colorbar(smr, ticks=[z.df.r.min(), z.df.r.max()], ax=axr)
cbarr.set_label("z.r")


for i, zi in z.df.iterrows():
    xi = x.df.loc[i]
    yi = y.df.loc[i]
    polygon = Polygon(np.vstack([[xi.l, xi.r, xi.r, xi.l, xi.l],
                           [yi.l, yi.l, yi.r, yi.r, yi.l]]).T)

    pl = PatchCollection([polygon], alpha=1, color=sml.to_rgba(zi.l))
    axl.add_collection(pl)
    pr = PatchCollection([polygon], alpha=1, color=sm.to_rgba(zi.r))
    axr.add_collection(pr)

    axl.plot([xi.l, xi.r, xi.r, xi.l, xi.l],
            [yi.l, yi.l, yi.r, yi.r, yi.l],
             c="k", alpha=.5, lw=.5, dashes=[2,2])

    axr.plot([xi.l, xi.r, xi.r, xi.l, xi.l],
            [yi.l, yi.l, yi.r, yi.r, yi.l],
             c="k", alpha=.5, lw=.5, dashes=[2,2])

# # csr = axr.contourf(Xs, Ys, Zs, alpha=.2, cmap="hot")
# csr2 = axr.contour(Xs, Ys, Zs, color="k", cmap=cmap)
# axr.clabel(csr2, fontsize=9, inline=1, alpha=1)



axl.scatter([1], [-.4])
axr.scatter([1], [-.4])
axc.scatter([1], [-.4])

mix_mpl(z)
z.plot(ax= axz, show=False)
axz.set_title(z)
axc.set_title("f(x,y)")
fig.tight_layout()
plt.show()
