# -*- coding: utf-8 -*-
import phuzzy
from phuzzy.mpl import MPL_Mixin


def extend_instance(obj, cls):
    """Apply mixins to a class instance after creation"""
    base_cls = obj.__class__
    base_cls_name = obj.__class__.__name__
    obj.__class__ = type(base_cls_name, (base_cls, cls), {})


def test_dynamic_mixin_on_instance():
    p = phuzzy.TruncNorm(alpha0=[1, 3], number_of_alpha_levels=15, name="y")
    print(p)
    print(p.df)

    extend_instance(p, MPL_Mixin)

    assert hasattr(p, "plot")

    # p.plot(show=True, filepath="truncnorm.png")
