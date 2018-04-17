# -*- coding: utf-8 -*-

import phuzzy.mpl as phm

w = phm.Triangle(alpha0=[2.9, 3.1], alpha1=[3], name="width")
h = phm.Trapezoid(alpha0=[4.5, 5.5], alpha1=[4.8, 5.2], name="height")

A = w * h
A.name = "area"

phm.mix_mpl(A)
A.plot(show=True)
