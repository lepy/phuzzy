import phuzzy

n = phuzzy.Fuzzy_Number()
print(n)

t = phuzzy.Triangle(alpha0=[1,3], alpha1=[2])
print(t)
print(t.df)
t.convert_df(5)
print(t.df)

p = phuzzy.Trapezoid(alpha0=[1,4], alpha1=[2,3])
print(p)
print(p.df)
p.convert_df(5)
print(p.df)
