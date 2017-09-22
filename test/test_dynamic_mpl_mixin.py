import phuzzy
from phuzzy.mpl import MPL_Mixin
import matplotlib.pyplot as plt


def extend_instance(obj, cls):
    """Apply mixins to a class instance after creation"""
    base_cls = obj.__class__
    base_cls_name = obj.__class__.__name__
    obj.__class__ = type(base_cls_name, (base_cls, cls),{})


def test_dynamic_mixin_on_instance():

    p = phuzzy.Triangle(alpha0=[1,3], alpha1=[2], number_of_alpha_levels=15)
    print(p)
    print(p.df)

    extend_instance(p, MPL_Mixin)

    p.plot(show=True)
