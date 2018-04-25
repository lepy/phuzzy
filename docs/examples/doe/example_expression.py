# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt

import phuzzy.approx.doe
import phuzzy.mpl.plots
from phuzzy.mpl import mix_mpl


def f(x1, x2):
    return (x1 - 1) ** 2 + (x2 + .4) ** 2 + .1


# define design variables
x = phuzzy.TruncNorm(alpha0=[1, 2], name="x")
# y = phuzzy.Triangle(alpha0=[3, 6], alpha1=[4], name="y")
y = phuzzy.TruncNorm(alpha0=[3, 6], name="y")

# exact reference solution
z = f(x, y)
z.name = "z"

# create Expression object
expr = phuzzy.approx.doe.Expression(designvars=[x, y],
                                    function=f,
                                    name="f(x,y)")

# expr.generate_training_doe(name="train", n=10, method="cc")
expr.generate_training_doe(name="train", n=10, method="lhs")
expr.eval()
z_a = expr.get_fuzzynumber_from_results()
print(z.df)
mix_mpl(z)

expr.fit_model()
print(expr.model)

# expr.generate_prediction_doe(name="prediction", n=10000, method="lhs")
expr.generate_prediction_doe(name="prediction", n=10000, method="meshgrid")
z_b = expr.predict(name="z_b")
print(z_b)
mix_mpl(z_b)
fig, axs = phuzzy.mpl.plots.plot_xy(z, z_b)
z.plot(axs[1], labels=False)
samples = expr.results_training
axs[1].scatter(samples.res, samples.alpha)
plt.show()
# z.plot(show=True)
