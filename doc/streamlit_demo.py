import streamlit as st
import phuzzy
from phuzzy.mpl import mix_mpl
import matplotlib.pyplot as plt

add_selectbox = st.sidebar.selectbox(
    "phuzzy numbers",
    ("Triangle", "Uniform", "Trapezoid", "TruncNorm")
)

st.text('phuzzy')

p1_slot = st.empty()

uniform = phuzzy.Uniform(alpha0=[1, 2])
mix_mpl(uniform)

trapezoid = phuzzy.Trapezoid(alpha0=[1, 2], alpha1=[1.2, 1.5])
mix_mpl(trapezoid)

triangle = phuzzy.Triangle(alpha0=[1, 2],alpha1=[1.8])
mix_mpl(triangle)

truncnorm = phuzzy.TruncNorm(alpha0=[1, 2], alpha1=None, number_of_alpha_levels=7)
mix_mpl(truncnorm)


p_dict= {"Triangle":triangle,
         "Uniform":uniform,
         "Trapezoid":trapezoid,
         "TruncNorm":truncnorm}
print(add_selectbox)
p = p_dict.get(add_selectbox, triangle)

col1, col2 = st.sidebar.beta_columns(2)
with col1:
    p10 = st.number_input('a1.l', value=p.alpha1.l)
    p00 = st.number_input('a0.l', value=p.alpha0.l, min_value=None, max_value=None)
with col2:
    p11 = st.number_input('a1.r', value=p.alpha1.r)
    p01 = st.number_input('a0.r', value=p.alpha0.r)


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




