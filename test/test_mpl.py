import phuzzy

from phuzzy.mpl import MPL_Mixin, Trapezoid, Triangle, TruncNorm
import numpy as np

p = Trapezoid(alpha0=[1,4], alpha1=[2,3])
print(p)
print(p.df)
p.convert_df(5)
print(p.df)

# p.plot(show=True)



p = Triangle(alpha0=[1,4], alpha1=[2,3])
print(p)
print(p.df)
p.convert_df(5)
print(p.df)

# p.plot(show=True)


p = TruncNorm(alpha0=[1,4], alpha1=[2,3])
print(p)
print(p.df)
p.convert_df(5)
print(p.df)
print(p.distr.ppf([.05,.5, .95]))
p.plot(show=True)
