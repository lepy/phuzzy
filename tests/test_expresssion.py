# -*- coding: utf-8 -*-

import phuzzy
import phuzzy.approx.doe


def f(x1, x2):
    return (x1 - 1) ** 2 + (x2 + .4) ** 2 + .1


def test_expression():
    x = phuzzy.TruncNorm(alpha0=[1, 2], name="x")
    y = phuzzy.Triangle(alpha0=[3, 6], alpha1=[4], name="y")
    z = f(x, y)

    expr = phuzzy.approx.doe.Expression(designvars=[x, y],
                                        function=f,
                                        name="f(x,y)")

    expr.generate_training_doe(name="train", n=10, method="lhs")
    expr.eval()
    z = expr.get_fuzzynumber_from_results()
    print(z.df)

    expr.fit_model()
    print(expr.model)

    expr.generate_prediction_doe(name="prediction", n=10000, method="lhs")
    u = expr.predict(name="u")
    print(u)

    # doe = phuzzy.approx.doe.DOE(designvars = [x,y], name="xy")
    # doe.sample_doe(n=10, method="lhs")
    # print(doe)
    # print(doe.samples)
    # assert len(doe.samples)==10
    # assert x.min() <= doe.samples.iloc[:,0].min()
    # assert x.max() >= doe.samples.iloc[:,0].max()
    # assert y.min() <= doe.samples.iloc[:,1].min()
    # assert y.max() >= doe.samples.iloc[:,1].max()
