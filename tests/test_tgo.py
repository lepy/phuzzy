# -*- coding: utf-8 -*-

import phuzzy.contrib.tgo
import numpy as np

def f(x):
    x1 = x[0]
    x2 = x[1]

    return (x1-1)**2+(x2+.4)**2 + .1

def test_tgo():
    bounds = [(-6, 6), (-6, 6)]

    res = phuzzy.contrib.tgo.tgo(f, bounds, args=(), g_cons=None, g_args=(), n=50)
    print(res)
    assert np.allclose(res.x, [1, -.4])
