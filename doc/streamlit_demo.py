import streamlit as st
import phuzzy
from phuzzy.mpl import mix_mpl
import matplotlib.pyplot as plt

add_selectbox = st.sidebar.selectbox(
    "phuzzy numbers",
    ("Triangle", "Uniform", "Trapezoid")
)

st.text('phuzzy')

p1_slot = st.empty()

uniform = phuzzy.Uniform(alpha0=[1, 2])
mix_mpl(uniform)

trapezoid = phuzzy.Trapezoid(alpha0=[1, 2], alpha1=[1.2, 1.5])
mix_mpl(trapezoid)

triangle = phuzzy.Triangle(alpha0=[1, 2],alpha1=[1.8])
mix_mpl(triangle)

p_dict= {"Triangle":triangle,
         "Uniform":uniform,
         "Trapezoid":trapezoid}
print(add_selectbox)
p = p_dict.get(add_selectbox, triangle)


p00 = st.sidebar.number_input('a0.l', value=p.alpha0.l, min_value=None, max_value=None)
p01 = st.sidebar.number_input('a0.r', value=p.alpha0.r)

p10 = st.sidebar.number_input('a1.l', value=p.alpha1.l)
p11 = st.sidebar.number_input('a1.r', value=p.alpha1.r)

st.sidebar.write('Phuzzy', p)

p.update(alpha0=[p00, p01], alpha1=[p10, p11])

st.cache()
def plot_phuzzy_number(p):

    H = 170.  #mm
    B = 270.  #mm
    fig, ax = plt.subplots(dpi=90, facecolor='w', edgecolor='k', figsize=(B / 25.4, H / 25.4))
    fig, ax = p.plot(ax)

    p1_slot.write(fig)

plot_phuzzy_number(p)

print(add_selectbox)




