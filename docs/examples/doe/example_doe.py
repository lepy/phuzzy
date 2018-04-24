# -*- coding: utf-8 -*-
import phuzzy
from phuzzy.mpl import mix_mpl
import matplotlib.pyplot as plt
import phuzzy.approx.doe

x = phuzzy.TruncNorm(alpha0=[1, 2], name="x")
y = phuzzy.Triangle(alpha0=[3, 6], alpha1=[4], name="y")

mix_mpl(x)
mix_mpl(y)

doe = phuzzy.approx.doe.DOE(designvars = [x,y], name="xy")
doe.sample_doe(n=10, method="lhs")
print(doe)
print(doe.samples)

Hf = 200.  # mm
Bf = 200.  # mm
fig, axs = plt.subplots(2,2, dpi=90, facecolor='w', edgecolor='k', figsize=(Bf / 25.4, Hf / 25.4))

axx = axs[0,0]
axy = axs[1,1]
axxy = axs[1,0]

axxy.set_xlabel("x")
axxy.set_ylabel("y")

axx.get_shared_x_axes().join(axx, axxy)
axy.get_shared_y_axes().join(axy, axxy)

x.plot(ax=axx)
y.vplot(ax=axy)


for i, xi in x.df.iterrows():
    yi = y.df.loc[i]
    # polygon = Polygon(np.vstack([[xi.l, xi.r, xi.r, xi.l, xi.l],
    #                        [yi.l, yi.l, yi.r, yi.r, yi.l]]).T)
    #
    # pl = PatchCollection([polygon], alpha=1, color=sml.to_rgba(zi.l))
    # axl.add_collection(pl)
    # pr = PatchCollection([polygon], alpha=1, color=sm.to_rgba(zi.r))
    # axr.add_collection(pr)
    #
    # axl.plot([xi.l, xi.r, xi.r, xi.l, xi.l],
    #         [yi.l, yi.l, yi.r, yi.r, yi.l],
    #          c="k", alpha=.5, lw=.5, dashes=[2,2])

    axxy.plot([xi.l, xi.r, xi.r, xi.l, xi.l],
            [yi.l, yi.l, yi.r, yi.r, yi.l],
             c="k", alpha=.5, lw=.5, dashes=[2,2])

axxy.scatter(doe.samples.x.values, doe.samples.y.values, c="r")
for i, row in doe.samples.iterrows():
    axx.axvline(row.x, alpha=.2, c="r")
    axy.axhline(row.y, alpha=.2, c="r")

fig.tight_layout()
plt.show()
