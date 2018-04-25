# -*- coding: utf-8 -*-

import phuzzy
import phuzzy.approx.doe

def test_project():
    p = phuzzy.approx.doe.Project(name="f")

    x = phuzzy.TruncNorm(alpha0=[1, 2], name="x")
    y = phuzzy.Triangle(alpha0=[3, 6], alpha1=[4], name="y")
    p.add_designvars([x, y])


    doe = phuzzy.approx.doe.DOE(designvars = [x,y], name="xy")
    doe.sample_doe(n=10, method="cc")
    print(doe)
    print(doe.samples)
    assert len(doe.samples)==19
    assert x.min() <= doe.samples.iloc[:,0].min()
    assert x.max() >= doe.samples.iloc[:,0].max()
    assert y.min() <= doe.samples.iloc[:,1].min()
    assert y.max() >= doe.samples.iloc[:,1].max()

