import streamlit as st
from scipy.integrate import odeint
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.animation import FuncAnimation
from seaborn.palettes import blend_palette

st.set_page_config(layout="wide")

st.title('TROCAL Simulator - Simulation of a Double Tube Heat Exchanger')
st.sidebar.title('About the Simulator:')
st.sidebar.write('This is a simulator of a double tube heat exchanger operating in parallel currents. When running the simulation, you will be able to observe the temperature profile of fluids 1 (cold) and 2 (hot) as time passes. You will also be able to view the temperature variation graph for fluids 1 and 2 when the exchanger reaches steady state.')
st.sidebar.write('Below is an image exemplifying this exchanger, created by myself.')
st.sidebar.image('Double Tube img #2.png', use_column_width=True)
st.sidebar.write('This type of exchanger is commonly used in the chemical, food, and oil and gas industries.')
st.sidebar.write('This simulator uses the following energy balance equations for cold and hot fluids, considering the principle of energy conservation:')
st.sidebar.image('Double Tube equations #2.jpg', use_column_width=True)

# Creating the figure for the steady-state graph
fig_permanente = plt.figure(figsize=(8, 6))

def run_simulation(L, r1, r2, n, m1, Cp1, rho1, m2, Cp2, rho2, T1i, T2i, T0, U, dx, t_final, dt):
    Ac1 = np.pi * r1**2
    Ac2 = np.pi * (r2**2-r1**2)

    x = np.linspace(dx/2, L-dx/2, n)
    T1 = np.ones(n) * T1i
    T2 = np.ones(n) * T2i
    t = np.arange(0, t_final, dt)

    # Function that defines the ODE for the temperature variation for Fluid 1
    def dT1dt_function(T1, t):
        dT1dt = np.zeros(n)
        dT1dt[1:n] = (m1 * Cp1 * (T1[0:n-1] - T1[1:n]) + U * 2 * np.pi * r1 * dx * (T2[1:n] - T1[1:n])) / (rho1 * Cp1 * dx * Ac1)
        dT1dt[0] = (m1 * Cp1 * (T1i - T1[0]) + U * 2 * np.pi * r1 * dx * (T2[0] - T1[0])) / (rho1 * Cp1 * dx * Ac1)
        return dT1dt

    # Function that defines the ODE for the temperature variation for Fluid 2
    def dT2dt_function(T2, t):
        dT2dt = np.zeros(n)
        dT2dt[1:n] = (m2 * Cp2 * (T2[0:n-1] - T2[1:n]) - U * 2 * np.pi * r1 * dx * (T2[1:n] - T1[1:n])) / (rho2 * Cp2 * dx * Ac2)
        dT2dt[0] = (m2 * Cp2 * (T2i - T2[0]) - U * 2 * np.pi * r1 * dx * (T2[0] - T1[0])) / (rho2 * Cp2 * dx * Ac2)
        return dT2dt

    T_out1 = odeint(dT1dt_function, T1, t)
    T_out1 = T_out1
    T_out2 = odeint(dT2dt_function, T2, t)
    T_out2 = T_out2

    # Creating the DataFrames
    df_Temp1 = pd.DataFrame(np.array(T_out1), columns=x)
    df_Temp2 = pd.DataFrame(np.array(T_out2), columns=x)

    # Creating color palettes for fluids 1 and 2
    paleta_calor = blend_palette(['blue', 'yellow', 'orange','red'], as_cmap=True, n_colors=100)

    # Function that updates the plot for Fluid 1
    def update_plot1(t):
        plt.clf()
        line = pd.DataFrame(df_Temp1.iloc[t, :]).T
        sns.heatmap(line, cmap=paleta_calor)
        plt.title(f'Tempo: {t} (s)')
        plt.gca().set_xticklabels(['{:.2f}'.format(val) for val in x])

    # Function that updates the plot for Fluid 2
    def update_plot2(t):
        plt.clf()
        line = pd.DataFrame(df_Temp2.iloc[t, :]).T
        sns.heatmap(line, cmap=paleta_calor)
        plt.title(f'Tempo: {t} (s)')
        plt.gca().set_xticklabels(['{:.2f}'.format(val) for val in x])

    # Creation and display of figure 1
    fig_ani1 = plt.figure(figsize=(8,6))
    ani1 = FuncAnimation(fig_ani1, update_plot1, frames=df_Temp1.shape[0], repeat=False)
    save1 = ani1.save('Temperature Variation - Fluido 1.gif', writer='pillow', fps=10)

    # Creation and display of figure 2
    fig_ani2 = plt.figure(figsize=(8,6))
    ani2 = FuncAnimation(fig_ani2, update_plot2, frames=df_Temp2.shape[0], repeat=False)
    save2 = ani2.save('Temperature Variation - Fluido 2.gif', writer='pillow', fps=10)

    # Displaying the simulation
    with st.expander("Visualization of the real-time Simulation of Fluid 1 (cold) (Click here to see)"):
        st.write('Variation in the temperature of the cold fluid over time and length.')
        st.write('Time represented above the GIF, in seconds. Temperatures in Kelvin on the y-axis. Length of the exchanger in meters on the x-axis of the GIF.')
        st.image('Temperature Variation - Fluido 1.gif')
    with st.expander("Visualization of the real-time Simulation of Fluid 2 (hot) (Click here to see)"):
        st.write('Variation in the temperature of the hot fluid over time and length.')
        st.write('Time represented above the GIF, in seconds. Temperatures in Kelvin on the y-axis. Length of the exchanger in meters on the x-axis of the GIF.')
        st.image('Temperature Variation - Fluido 2.gif')

    # Displaying the graph of temperature variation along length in steady-state for both fluids
    plt.figure(fig_permanente)
    plt.plot(x, df_Temp1.iloc[-1, :] , color='blue', label='Fluido frio')
    plt.plot(x, df_Temp2.iloc[-1, :], color='red', label='Fluido quente')
    plt.xlabel('Length (m)')
    plt.ylabel('Temperature (K)')
    plt.legend()
    plt.title('Temperatura dos fluidos frio e quente ao longo do comprimento do trocador no regime permanente.')
    plt.title('Temperature of the cold and hot fluids along the length of the exchanger at steady-state.')
    st.pyplot(plt)

col1, col2 = st.columns(2)

with col1:
  st.header('Parameters')
  st.write('ATTENTION: On the main page, you will find a button that runs the simulation with a pre-defined example ("Run standard example"). This example takes around 40 seconds to run, depending on your connection speed. If you want to use your own input values, use the "Run simulation" button. It is recommended to use a number of nodes between 10 and 30, depending on the specific example used. Furthermore, the cold fluid inlet temperature must be lower than the hot fluid inlet temperature, otherwise the model may not work correctly.')

  # Input Values
  L = st.number_input('Length of the tube (m)', min_value=0.0)
  r1 = st.number_input('Internal radius of the tube (m)', min_value=0.0)
  r2 = st.number_input('External radius of the tube (m)', min_value=0.0)
  n = st.number_input('Number of nodes for discretization', min_value=1)
  m1 = st.number_input('Mass flow of the cold fluid (kg/s)', min_value=0.0)
  Cp1 = st.number_input('Specific heat capacity of the cold fluid (J/kg.K)', min_value=0.0)
  rho1 = st.number_input('Specific mass of the cold fluid (kg/m³)', min_value=0.0)
  m2 = st.number_input('Mass flow of the hot fluid (kg/s)', min_value=0.0)
  Cp2 = st.number_input('Specific heat capacity of the hot fluid (J/kg.K)', min_value=0.0)
  rho2 = st.number_input('Specific mass of the hot fluid (kg/m³)', min_value=0.0)
  T1i = st.number_input('Inlet temperature of the cold fluid in the exchanger (K)')
  T2i = st.number_input('Inlet temperature of the hot fluid in the exchanger (K)')
  T0 = st.number_input('Initial temperature of the exchanger (K)')
  U = st.number_input('Overall heat transfer coefficient (W/m².K)', min_value=0.0)
  dx = L / n

  t_final = st.number_input('Simulation Time (s)', min_value=0.0)
  dt = st.number_input('Time step (s)', min_value=0.0)

with col2:
  st.header('Results')
  if st.button('Run simulation'):
      run_simulation(L, r1, r2, n, m1, Cp1, rho1, m2, Cp2, rho2, T1i, T2i, T0, U, dx, t_final, dt)
  elif st.button('Run standard example'):
      run_simulation(10, 0.1, 0.15, 10, 3, 4180, 995.61, 5, 4180, 995.61, 400, 800, 300, 1500, 10 / 10, 100, 1)
