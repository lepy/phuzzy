# -*- coding: utf-8 -*-
import sys
is_py2 = sys.version_info.major == 2
import phuzzy
import phuzzy.analysis.alo

def test_alo():
    v1 = phuzzy.Triangle(alpha0=[0,4], alpha1=[1], number_of_alpha_levels=5)
    v2 = phuzzy.Superellipse(alpha0=[-1, 2.], alpha1=None, m=1.0, n=.5, number_of_alpha_levels=6)
    v3 = phuzzy.TruncGenNorm(alpha0=[1, 4], alpha1=[2, 3], number_of_alpha_levels=5, beta=3.)
    v4 = phuzzy.Trapezoid(alpha0=[0, 4], alpha1=[2, 3], number_of_alpha_levels=5)
    v5 = phuzzy.TruncNorm(alpha0=[1, 3], number_of_alpha_levels=5, name="y")
    v6 = phuzzy.Triangle(alpha0=[1,4], alpha1=[3], number_of_alpha_levels=5)

    obj_function = '-1*((x[0] - 1) ** 2 + (x[1] + .1) ** 2 + .1 - (x[2] + 2) ** 2 - (x[3] - 0.1) ** 2 - (x[4] * x[5]) ** 2)'
    name = 'Opti_Test'

    kwargs = {'var1': v1, 'var2': v2, 'var3': v3,
              'var4': v4,'var5': v5, 'var6': v6,
              'obj_function': obj_function, 'name': name}

    # alo = phuzzy.analysis.alo.Alpha_Level_Optimization(name="test", obj_function=obj_function, vars=[v1, v2, v3, v4, v5, v6])
    alo = phuzzy.analysis.alo.Alpha_Level_Optimization(**kwargs)
    alo.calculation()
    print(alo)

if __name__ == '__main__':
    test_alo()
