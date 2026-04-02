# Heat Pump Optimization Script
# This code optimizes the performance of a heat pump by first determining the fluid type to be used based on user inputs,
# then maximizes the difference of powers between the pipe and boiler heat extraction and pump. 

import pandas as pd
from CoolProp.CoolProp import propsSI
import numpy as np
import matplotlib.pyplot as plt

# User Input Section
print("enter the required parameters for heat pump optimization:") 
COP_heating = float(input("Coefficient of Performance for Heating (COP_heating): "))
pipe_radius = float(input("Pipe Radius (m): "))
flue_height = float(input("Flue Height (m): "))
flue_material = input("Flue Material (e.g., 'Steel', 'Concrete'): ")
flue_thickness = float(input("Flue Thickness (m): "))
pipe_material = input("Pipe Material (e.g., 'Steel', 'Copper'): ")
pipe_thickness = float(input("Pipe Thickness (m): "))
boiler_thickness = float(input("Boiler Thickness (m): "))
boiler_long_side = float(input("Boiler Long Side Length (m): "))
boiler_short_side = float(input("Boiler Short Side Length (m): "))
flue_gas = input("Flue Gas Type (e.g., 'Air', 'CO2'): ")
flue_gas_temp_in = float(input("Flue Gas Inlet Temperature (K): "))
flue_radius = float(input("Flue Radius (m): "))



# Constant Parameters
g = 9.81  
boiler_liquid = 'water'

# To be optimized parameters (all values in SI units)
fluid_types = ['DowthermA', 'TherminolVP1', 'Syltherm800', 'Toluene',  'Santotherm100', 'MarlothermSH', 'ParathermNF']
max_operating_temp_K = [643.15, 673.15, 673.15, 393.15, 623.15, 673.15, 643.15]  
best_fluid = None
inlet_pressure = np.linspace(1.01325, 50.6625, 50 ) * 1e5 # Pressure of liquid just before contacting with flue
best_inlet_pressure = None
inlet_mass_flow_rate = np.linspace(0.1, 5, 50) # mass flow rate of the liquid in kg/s
best_mass_flow_rate = None
inlet_liquid_temp = np.linspace(323.15, 370.15, 50)
number_of_elbows_boiler = np.linspace(1, boiler_short_side/(2*pipe_radius), boiler_short_side/(2*pipe_radius)-1) # pipe thickness to be neglected


# Other Parameters (all values in SI units)


# Derived Parameters
pipe_length = 2*np.pi*(2*pipe_radius+flue_radius)*flue_height # pipe thickness is neglected
pipe_cross_sectional_area = np.pi * pipe_radius**2
flue_cross_sectional_area = np.pi * flue_radius**2
theta = np.arctan(1 / (2 * np.pi * (flue_radius + 2*pipe_radius))) # angle of inclination of the pipe with respect to horizontal
dz = pipe_length / 5000  # Discretization step along the pipe length

def friction_factor_gnielinski(Re,):
    """Friction factor using Gnielinski correlation (laminar and turbulent)."""
    if Re < 2300:
        f = 64 / Re
    elif Re <4000:
        f = 0.3164 * Re**(-0.25)
    elif Re < 5e6:
        f = (0.79 * np.log(Re) - 1.64)**(-2) * (1 + (Re/282000)**0.625)**0.8
    else:
        f = 0.184 * Re**(-0.2)
    return f

def h_internal_tube(mdot_l, D_i, fluid, T, P_local=101325):
    """Internal convective heat transfer in pipe."""
    rho = propsSI("D", "T", T, "P", P_local, fluid)
    mu = propsSI("V", "T", T, "P", P_local, fluid)
    k = propsSI("L", "T", T, "P", P_local, fluid)
    cp = propsSI("C", "T", T, "P", P_local, fluid)
    A = np.pi * (D_i**2)/4
    v = mdot_l / (rho * A)
    Re = max(rho * v * D_i / mu, 1e-9)
    Pr = max(cp * mu / k, 1e-9)
    if Re < 2300:
        Nu = 3.66
    else:
        Nu = 0.023 * Re**0.8 * Pr**0.4
    h = Nu * k / D_i
    return h, Re, Pr, cp, rho, mu, k, v

def h_external_crossflow_cylinder(U_g_bulk, D_o, fluid, T_g, P_local=101325):
    """External convective heat transfer (flue gas crossflow)."""
    rho_g = propsSI("D", "T", T_g, "P", P_local, fluid)
    mu_g = propsSI("V", "T", T_g, "P", P_local, fluid)
    k_g = propsSI("L", "T", T_g, "P", P_local, fluid)
    cp_g = propsSI("C", "T", T_g, "P", P_local, fluid)
    Re = max(rho_g * U_g_bulk * D_o / mu_g, 1e-9)
    Pr = max(cp_g * mu_g / k_g, 1e-9)
    # Churchill-Bernstein correlation
    term1 = 0.3
    term2 = (0.62 * Re**0.5 * Pr**(1/3)) / (1 + (0.4/Pr)**(2/3))**0.25
    term3 = (1 + (Re/282000)**(5/8))**(4/5)
    Nu = term1 + term2 * term3
    h = Nu * k_g / D_o
    return h, Re, Pr, cp_g, rho_g, mu_g, k_g

# -----------------------------
# --- Helical Pipe Simulation ---
# -----------------------------
def simulate_helical_pipe(fluid_pipe='Water',
                          T_pipe_in=323.15,
                          mdot_pipe=1.0,
                          D_i=0.02,
                          pipe_thickness=0.002,
                          pipe_k=16.2,
                          flue_gas='Air',
                          T_gas_in=873.15,
                          U_g_bulk=10.0,
                          flue_radius=0.25,
                          flue_height=5.0,
                          n_turns=4,
                          dz=None,
                          P_atm=101325):
    """Simulate temperatures, Re, and friction along a helical pipe."""
    r_o = D_i/2 + pipe_thickness
    D_o = 2*r_o
    theta = np.arctan(1 / (2*np.pi*(flue_radius + D_i)))
    pipe_length = 2*np.pi*(2*D_i + flue_radius) * flue_height * n_turns

    if dz is None:
        dz = pipe_length / 3000
    n_steps = int(np.ceil(pipe_length / dz))
    x = np.linspace(0, pipe_length, n_steps+1)

    Tg = np.zeros_like(x)
    Tl = np.zeros_like(x)
    Re_pipe = np.zeros_like(x)
    v_pipe = np.zeros_like(x)
    f_pipe = np.zeros_like(x)

    Tg[0] = T_gas_in
    Tl[0] = T_pipe_in

    A_flue = np.pi * flue_radius**2
    rho_g_in = propsSI("D", "T", T_gas_in, "P", P_atm, flue_gas)
    mdot_gas = rho_g_in * U_g_bulk * A_flue

    P_per_length = 2 * np.pi * r_o

    for i in range(n_steps):
        # internal
        h_l, Re_l, Pr_l, cp_l, rho_l, mu_l, k_l, v_l = h_internal_tube(mdot_pipe, D_i, fluid_pipe, Tl[i], P_atm)
        f = friction_factor_gnielinski(Re_l, Pr_l)
        f_pipe[i] = f
        Re_pipe[i] = Re_l
        v_pipe[i] = v_l

        # external
        h_g, Re_g, Pr_g, cp_g, rho_g, mu_g, k_g = h_external_crossflow_cylinder(U_g_bulk, D_o, flue_gas, Tg[i], P_atm)

        # resistances
        R_conv_g = 1 / (h_g * P_per_length)
        R_conv_l = 1 / (h_l * P_per_length)
        R_cond = np.log(r_o/(D_i/2)) / (2*np.pi*pipe_k)
        R_total = R_conv_g + R_conv_l + R_cond

        # heat transfer
        q_prime = (Tg[i] - Tl[i]) / R_total
        dTg = - q_prime * dz / (mdot_gas * cp_g)
        dTl =   q_prime * dz / (mdot_pipe * cp_l)

        Tg[i+1] = Tg[i] + dTg
        Tl[i+1] = Tl[i] + dTl

    Re_pipe[-1] = Re_l
    f_pipe[-1] = f

    return {'x': x, 'Tg': Tg, 'Tl': Tl, 'Re_pipe': Re_pipe, 'v_pipe': v_pipe,
            'f_pipe': f_pipe, 'pipe_length': pipe_length, 'mdot_gas': mdot_gas,
            'theta_rad': theta}

# -----------------------------
# Example usage
# -----------------------------
# sim = simulate_helical_pipe(fluid_pipe='Water', T_pipe_in=323.15,
#                             mdot_pipe=1.0, D_i=pipe_radius,
#                             pipe_thickness=pipe_thickness,
#                             pipe_k=16.2, flue_gas=flue_gas,
#                             T_gas_in=flue_gas_temp_in,
#                             U_g_bulk=10.0, flue_radius=flue_radius,
#                             flue_height=flue_height, n_turns=4)
# plt.plot(sim['x'], sim['Tg']-273.15, label='Flue Gas (°C)')
# plt.plot(sim['x'], sim['Tl']-273.15, label='Pipe Liquid (°C)')
# plt.xlabel('Distance along pipe (m)')
# plt.ylabel('Temperature (°C)')
# plt.legend()
# plt.grid(True)
# plt.title('Temperature profile along helical pipe')
# plt.show()

#reynolds_number_rotating = (propsSI('D', 'T', T_ambient, 'P', inlet_pressure[0], fluid_types[0]) * velocity * 2 * pipe_radius) / propsSI('V', 'T', T_ambient, 'P', inlet_pressure[0], fluid_types[0])
#prandtl_number_rotating = (propsSI('C', 'T', T_ambient, 'P', inlet_pressure[0], fluid_types[0]) * propsSI('V', 'T', T_ambient, 'P', inlet_pressure[0], fluid_types[0])) / propsSI('L', 'T', T_ambient, 'P', inlet_pressure[0], fluid_types[0])
#nusselt_number_rotating = 0.023 * reynolds_number_rotating**0.8 * prandtl_number_rotating**0.4
#h_inner_rotating = (nusselt_number_rotating * propsSI('L', 'T', T_ambient, 'P', inlet_pressure[0], fluid_types[0])) / (2 * pipe_radius)
#velocity = inlet_mass_flow_rate / (propsSI('D', 'T', T_ambient, 'P', inlet_pressure[0], fluid_types[0]) * pipe_cross_sectional_area)
