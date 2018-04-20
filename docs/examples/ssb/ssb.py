# -*- coding: utf-8 -*-

import phuzzy as ph
number_of_alpha_levels = 31

# load P
P0 = 5000.  # N
dP = 0.01 * P0  # N
P = ph.Triangle(alpha0=[P0 - dP, P0 + dP], alpha1=[P0], name="P", number_of_alpha_levels=number_of_alpha_levels)

# dimensions L, W, H
W0 = 50  # mm
H0 = 100  # mm
L0 = 2000  # mm

dW = 0.01 * W0  # mm
dH = 0.01 * H0  # mm
dL = 0.01 * L0  # mm

L = ph.Triangle(alpha0=[L0 - dL, L0 + dL], alpha1=[L0], name="L", number_of_alpha_levels=number_of_alpha_levels)
W = ph.Triangle(alpha0=[W0 - dW, W0 + dW], alpha1=[W0], name="W", number_of_alpha_levels=number_of_alpha_levels)
H = ph.Triangle(alpha0=[H0 - dH, H0 + dH], alpha1=[H0], name="H", number_of_alpha_levels=number_of_alpha_levels)

# material

E0 = 30000.  # N/mm2
dE = 0.1 * E0  # N/mm2
E = ph.TruncNorm(alpha0=[E0 - dE, E0 + dE], alpha1=[E0], name="E", number_of_alpha_levels=number_of_alpha_levels)

I0 = W0 * H0 ** 3 / 12.
w0 = P0 * L0 ** 3 / (48 * E0 * I0)

print("I0 = {:.4g} mm^4".format(I0))
# I0 = 4.167e+06 mm^4
print("w0 = {:.4g} mm".format(w0))
# w0 = 6.667 mm

I = W * H** 3 / 12.
I.name = "I"
w = P * L ** 3 / (48 * E * I)
w.name = r"P L^3 / (48 EI)"

print("I = {} mm^4".format(I))
# I = FuzzyNumber(W*H^3/12.0:[[4002483.375, 4335850.041666667], [4166666.6666666665, 4166666.6666666665]]) mm^4

print("w = {} mm".format(w))
# w = FuzzyNumber(P*L^3/E*48*W*H^3/12.0:[[5.594629603627992, 8.024370049019725], [6.666666666666667, 6.666666666666667]]) mm

w_mean = w.mean()
dw_l = w_mean - w.min()
dw_r = w.max() - w_mean
print("w = {:.4g} mm (- {:.4g}|+ {:.4g})".format(w_mean, dw_l, dw_r))
# w = 6.703 mm (- 1.109|+ 1.321)
print("w = {:.4g} mm [{:.4g},{:.4g}]".format(w_mean, w.min(), w.max()))
# w = 6.703 mm [5.595,8.024]

from phuzzy.mpl import mix_mpl
import matplotlib.pyplot as plt

mix_mpl(I)
mix_mpl(w)

H_ = 100. # mm
B_ = 300. # mm

fig, axs = plt.subplots(1, 2, dpi=90, facecolor='w', edgecolor='k', figsize=(B_ / 25.4, H_ / 25.4))

axs[0].axvline(I0, lw=2, alpha=.4, c="r", label="$I_0$")
axs[1].axvline(w0, lw=2, alpha=.4, c="r", label="$w_0 = {:.4g}\,mm$".format(w0))
I.plot(ax=axs[0])
w.plot(ax=axs[1])

axs[0].set_title("area moment of inertia $I$")
axs[1].set_title("deflection $w$")

axs[0].set_xlabel(r"area moment of inertia $I=\frac{WH^3}{12}$")
axs[1].set_xlabel(r"deflection $w=\frac{PL^3}{48EI}$" + "$ = {:.4g}\,mm\,[{:.4g},{:.4g}]$".format(w_mean, w.min(), w.max()))

axs[0].legend()
axs[1].legend()
fig.tight_layout(pad=1.18, h_pad=1.1)
fig.savefig("ssb.png")

H_ = 250.  # mm
B_ = 300.  # mm

fig, axs = plt.subplots(3, 2, dpi=90, facecolor='w', edgecolor='k', figsize=(B_ / 25.4, H_ / 25.4))

A = W * H

ys = [P, L,
      W, H,
      E, A]

P.title = r"load $P$"
L.title = r"length $L$"
W.title = r"width $W$"
H.title = r"height $H$"
E.title = r"young's modulus $E$"
A.title = r"cross section area $A$"

for i, y in enumerate(ys):
    mix_mpl(y)
    ax = axs.ravel()[i]
    y.plot(ax=ax)
    if hasattr(y, "title"):
        ax.set_title(y.title)

fig.tight_layout()
fig.savefig("ssb_parameter.png")

plt.show()
