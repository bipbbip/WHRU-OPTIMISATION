# Tübitak 2204-A Lise Öğrencileri Arası Araştırma Projesi
# Made by Hüseyin Bilge Cengiz
# All values are in SI units
# Shell-and-Tube heat exchanger ve waste heat boilerdan oluşan WHRU sisteminde 
# pump-heat-exchanger optimizasyonuyla enerji verimliliğinin arttırılması


from scipy.optimize import fsolve
import pandas as pd
from CoolProp.CoolProp import PropsSI
import numpy as np
#import matplotlib.pyplot as plt
from thermo import Chemical

# User Input Section
print("enter the required parameters for heat pump optimization.\n for the pipe species Enter: ") 

pipe_radius = float(input("Pipe Radius (m): "))
pipe_material = int(input(
    "Pipe Material: carbon_steel(0), ss304(1), ss316(2), ss321(3), ss347(4), "
    "t11_alloy(5), t22_alloy(6), inconel600(7), inconel625(8), inconel718(9), "
    "hastelloyc276(10), titanium(11), copper(12), aluminum(13), brass(14), "
    "bronze(15), cast_iron(16), ductile_iron(17), galvanized_steel(18), "
    "lead(19), zinc(20), nickel(21), tungsten(22), molybdenum(23), "
    "platinum(24), silver(25), gold(26): \n"
))
pipe_thickness = float(input("Pipe Thickness (m): "))
small_pipe_radius = float(input("Small Pipe Radius (pipe inside the shell-and-tube heat exchanger) (m): "))
small_pipe_number = int(input("Number of Small Pipes in the shell-and-tube heat exchanger: "))

print("\nenter for the tube in the shell-and-tube heat exchanger\n")
# heat transfer between tube and surroundings is neglected

tube_radius = float(input("Tube Radius (m): "))
tube_length = float(input("Tube Height (m): "))
tube_material = int(input(
    "Tube Material: carbon_steel(0), ss304(1), ss316(2), ss321(3), ss347(4), "
    "t11_alloy(5), t22_alloy(6), inconel600(7), inconel625(8), inconel718(9), "
    "hastelloyc276(10), titanium(11), copper(12), aluminum(13), brass(14), "
    "bronze(15), cast_iron(16), ductile_iron(17), galvanized_steel(18), "
    "lead(19), zinc(20), nickel(21), tungsten(22), molybdenum(23), "
    "platinum(24), silver(25), gold(26): \n"
))

print("\nenter efficiencies:\n")
pump_efficiency = float(input("Pump efficiency (combined, e.g. 0.75): "))
boiler_efficiency = float(input("Boiler efficiency while gaining electrical energy (e.g. 0.5): "))

print("\nenter for boiler tank part:\n")
# boiler tank is supposed to be rectangular prism

boiler_tank_material = int(input(
    "Boiler Tank Material: carbon_steel(0), ss304(1), ss316(2), ss321(3), "
    "ss347(4), t11_alloy(5), t22_alloy(6), inconel600(7), inconel625(8), "
    "inconel718(9), hastelloyc276(10), titanium(11), copper(12), aluminum(13), "
    "brass(14), bronze(15), cast_iron(16), ductile_iron(17), "
    "galvanized_steel(18), lead(19), zinc(20), nickel(21), tungsten(22), "
    "molybdenum(23), platinum(24), silver(25), gold(26): \n"
))
boiler_tank_thickness = float(input("Boiler Tank Thickness (m): "))
boiler_liquid_tank_thickness = float(input("Thickness of the region where fluid flows under the boiler tank (m): "))
boiler_short_side = float(input("Boiler short side length (m): "))
boiler_long_side = float(input("Boiler long side length (m): "))

print("\nenter the properties of waste gas that enters the shell-and-tube heat exchanger:\n")

waste_gas_index = int(input(
    "air(0), nitrogen(1), carbondioxide(2), carbonmonoxide(3), oxygen(4), "
    "hydrogen(5), water(6), argon(7), helium(8), methane(9), ethane(10), "
    "propane(11), sulfurdioxide(12), ammonia(13): "
))
waste_gas_temp_in = float(input("Flue Gas Inlet Temperature (K): "))
waste_gas_pressure_in = float(input("Flue Gas Inlet Pressure (Pa): "))
waste_gas_mass_flow_rate = float(input("Flue Gas Mass Flow Rate (kg/s): "))
P_saturation = float(input("Boiling tank pressure (Pa): "))


#roughness and thermal conductivity data retrieval
roughness = [4.5e-05, 1.5e-05, 1.5e-05, 1.5e-05, 1.5e-05, 4.5e-05, 4.5e-05, 1.5e-05, 1.5e-05, 1.5e-05, 1.5e-05, 1.5e-05, 1.5e-06, 1.5e-06, 1.5e-05, 1.5e-05, 2.6e-04, 2.6e-04, 1.5e-04, 1.5e-05, 1.5e-05, 1.5e-05, 1.0e-05, 1.0e-05, 1.0e-05, 1.0e-06, 1.0e-06]
dk_dT = [-0.030, 0.015, 0.014, 0.015, 0.014, -0.025, -0.022, 0.012, 0.010, 0.011, 0.010, -0.020, -0.200, -0.100, -0.060, -0.045, -0.030, -0.028, -0.030, -0.030, -0.080, -0.060, -0.020, -0.020, -0.010, -0.250, -0.150]
k_0C = [54.0, 15.0, 14.8, 15.2, 15.0, 42.0, 38.0, 15.5, 10.5, 12.0, 11.0, 22.0, 420.0, 250.0, 115.0, 62.0, 58.0, 37.0, 47.0, 35.0, 120.0, 92.0, 180.0, 140.0, 72.0, 430.0, 320.0]

# Constant parameters
g = 9.81  
boiler_liquid = 'water'
N = 1000  # Number of segments along the tube length

# To be optimized parameters (all values are in SI units)
liquid_types = [
    "D5",  
    "D6",                
    "MD3M",            
    "n-Dodecane",   
    "MethylStearate",    
    "MethylLinoleate",   
    "MethylLinolenate",  
    "MethylOleate",  
    "MethylPalmitate"    
]

C_sf_list = [
    0.0130,  # carbon_steel
    0.0110,  # ss304
    0.0110,  # ss316
    0.0115,  # ss321
    0.0115,  # ss347
    0.0125,  # t11_alloy
    0.0125,  # t22_alloy
    0.0095,  # inconel600
    0.0095,  # inconel625
    0.0090,  # inconel718
    0.0090,  # hastelloyc276
    0.0060,  # titanium
    0.0130,  # copper
    0.0120,  # aluminum
    0.0125,  # brass
    0.0125,  # bronze
    0.0140,  # cast_iron
    0.0135,  # ductile_iron
    0.0130,  # galvanized_steel
    0.0150,  # lead
    0.0140,  # zinc
    0.0100,  # nickel
    0.0070,  # tungsten
    0.0080,  # molybdenum
    0.0090,  # platinum
    0.0105,  # silver
    0.0110   # gold
]

gas_list = [
    "Air",                 # 0
    "Nitrogen",            # 1
    "Oxygen",              # 2
    "CarbonDioxide",       # 3
    "CarbonMonoxide",      # 4
    "Hydrogen",            # 5
    "Water",               # 6 (steam)
    "Helium",              # 7
    "Neon",                # 8
    "Argon",               # 9
    "Krypton",             # 10
    "Xenon",               # 11
    "Methane",             # 12
    "Ethane",              # 13
    "Propane",             # 14
    "Isobutane",           # 15
    "n-Butane",            # 16
    "Isopentane",          # 17
    "n-Pentane",           # 18
    "Ammonia",             # 19
    "SulfurDioxide"        # 20
]

water = Chemical('water')
water_boiling_point_K = water.Tsat(P_saturation)

pressure = np.linspace(1.01325, 50.6625, 50 ) * 1e5 
best_pressure = None
mass_flow_rate = np.linspace(0.1, 5, 50) # mass flow rate of the liquid in kg/s
best_mass_flow_rate = None
inlet_liquid_temp = np.linspace(int(water_boiling_point_K), int(waste_gas_temp_in), int(waste_gas_temp_in-water_boiling_point_K)) # Temperature of the liquid is not known, estimations to be approved

# Derived parameters
pipe_cross_sectional_area = np.pi * pipe_radius**2
tube_cross_sectional_area = np.pi * tube_radius**2
N = 1000
dx_tube = tube_length / N # Discretization step along the pipe length (differential element)
dx_boiler = boiler_long_side / N 

print("Select the liquid")
for liquid in range(len(liquid_types)):
    print(f'({liquid}), {liquid_types[liquid]}')

liquid_type_index = int(input())   
liquid = liquid_types[liquid_type_index]

params_gas = {
    "Air": {"mu0": 1.85e-5, "T0": 300, "S": 110, "k0": 0.026, "Sk": 194},
    "Nitrogen": {"mu0": 1.76e-5, "T0": 300, "S": 111, "k0": 0.025, "Sk": 180},
    "Oxygen": {"mu0": 2.07e-5, "T0": 300, "S": 127, "k0": 0.027, "Sk": 200},
    "CarbonDioxide": {"mu0": 1.48e-5, "T0": 300, "S": 240, "k0": 0.016, "Sk": 250},
    "CarbonMonoxide": {"mu0": 1.81e-5, "T0": 300, "S": 118, "k0": 0.024, "Sk": 190},
    "Hydrogen": {"mu0": 8.9e-6, "T0": 300, "S": 72, "k0": 0.180, "Sk": 100},
    "Water": {"mu0": 9.0e-6, "T0": 300, "S": 350, "k0": 0.025, "Sk": 400},
    "Helium": {"mu0": 1.96e-5, "T0": 300, "S": 79, "k0": 0.151, "Sk": 120},
    "Neon": {"mu0": 3.1e-5, "T0": 300, "S": 120, "k0": 0.049, "Sk": 150},
    "Argon": {"mu0": 2.2e-5, "T0": 300, "S": 144, "k0": 0.017, "Sk": 200},
    "Krypton": {"mu0": 2.4e-5, "T0": 300, "S": 170, "k0": 0.009, "Sk": 220},
    "Xenon": {"mu0": 2.3e-5, "T0": 300, "S": 190, "k0": 0.005, "Sk": 240},
    "Methane": {"mu0": 1.1e-5, "T0": 300, "S": 160, "k0": 0.030, "Sk": 200},
    "Ethane": {"mu0": 9.2e-6, "T0": 300, "S": 170, "k0": 0.019, "Sk": 210},
    "Propane": {"mu0": 7.4e-6, "T0": 300, "S": 180, "k0": 0.017, "Sk": 220},
    "Isobutane": {"mu0": 7.0e-6, "T0": 300, "S": 185, "k0": 0.016, "Sk": 225},
    "n-Butane": {"mu0": 7.2e-6, "T0": 300, "S": 185, "k0": 0.016, "Sk": 225},
    "Isopentane": {"mu0": 6.5e-6, "T0": 300, "S": 190, "k0": 0.014, "Sk": 230},
    "n-Pentane": {"mu0": 6.6e-6, "T0": 300, "S": 190, "k0": 0.014, "Sk": 230},
    "Ammonia": {"mu0": 9.8e-6, "T0": 300, "S": 370, "k0": 0.025, "Sk": 400},
    "SulfurDioxide": {"mu0": 1.3e-5, "T0": 300, "S": 300, "k0": 0.013, "Sk": 350}
}

params_liq = {
    "D5": {"A": 1.2e-4, "B": 500, "beta": 1e-8, "P0": 1e5,
           "mu0": 1.8e-5, "T0": 300, "S": 110,
           "k0": 0.12, "a": 1e-3, "b": 1e-8, "Sk": 194},
    "D6": {"A": 1.3e-4, "B": 520, "beta": 1e-8, "P0": 1e5,
           "mu0": 1.9e-5, "T0": 300, "S": 120,
           "k0": 0.13, "a": 1e-3, "b": 1e-8, "Sk": 200},
    "MD3M": {"A": 1.1e-4, "B": 480, "beta": 1e-8, "P0": 1e5,
             "mu0": 1.7e-5, "T0": 300, "S": 100,
             "k0": 0.11, "a": 1e-3, "b": 1e-8, "Sk": 180},
    "n-Dodecane": {"A": 2.0e-4, "B": 600, "beta": 1e-8, "P0": 1e5,
                   "mu0": 2.0e-5, "T0": 300, "S": 150,
                   "k0": 0.14, "a": 1e-3, "b": 1e-8, "Sk": 210},
    "MethylStearate": {"A": 2.2e-4, "B": 650, "beta": 1e-8, "P0": 1e5,
                       "mu0": 2.1e-5, "T0": 300, "S": 160,
                       "k0": 0.15, "a": 1e-3, "b": 1e-8, "Sk": 220},
    "MethylLinoleate": {"A": 2.3e-4, "B": 670, "beta": 1e-8, "P0": 1e5,
                        "mu0": 2.2e-5, "T0": 300, "S": 165,
                        "k0": 0.16, "a": 1e-3, "b": 1e-8, "Sk": 225},
    "MethylLinolenate": {"A": 2.4e-4, "B": 690, "beta": 1e-8, "P0": 1e5,
                         "mu0": 2.3e-5, "T0": 300, "S": 170,
                         "k0": 0.17, "a": 1e-3, "b": 1e-8, "Sk": 230},
    "MethylOleate": {"A": 2.5e-4, "B": 710, "beta": 1e-8, "P0": 1e5,
                     "mu0": 2.4e-5, "T0": 300, "S": 175,
                     "k0": 0.18, "a": 1e-3, "b": 1e-8, "Sk": 235},
    "MethylPalmitate": {"A": 2.6e-4, "B": 730, "beta": 1e-8, "P0": 1e5,
                        "mu0": 2.5e-5, "T0": 300, "S": 180,
                        "k0": 0.19, "a": 1e-3, "b": 1e-8, "Sk": 240}
}


p_l = params_liq[liquid_types[liquid_type_index]]
p_g = params_gas[gas_list[waste_gas_index]]

def viscosity_liquid(T, P, A, B, beta, P0):
    return A * np.exp(B / T) * np.exp(beta * (P - P0))

def viscosity_gas(T, mu0, T0, S):
    # μ = μ0 * (T/T0)^(3/2) * (T0 + S)/(T + S)
    return mu0 * (T / T0) ** 1.5 * (T0 + S) / (T + S)

def thermal_conductivity_liquid(T, P, k0, a, b, T0, P0):
    # k = k0 * [1 + a(T - T0)] * [1 + b(P - P0)]
    return k0 * (1 + a * (T - T0)) * (1 + b * (P - P0))

def thermal_conductivity_gas(T, k0, T0, Sk):
    # k = k0 * (T/T0)^(3/2) * (T0 + Sk)/(T + Sk)
    return k0 * (T / T0) ** 1.5 * (T0 + Sk) / (T + Sk)

def k_material(material_index, T_celcius):
    return k_0C[material_index] + dk_dT[material_index] * T_celcius

def friction_factor_colebrook_white(f, Re, roughness, D): 
    return 1/np.sqrt(f) + 2*np.log10(roughness/(3.7*D) + 2.51/(Re*np.sqrt(f))) # Colebrook-White equation for turbulent flow

def mu_wall_correction(T_bulk, T_wall, pressure):
    mu_bulk = viscosity_liquid(T_bulk, pressure, p_l["A"], p_l["B"], p_l["beta"], p_l["P0"])
    mu_wall = viscosity_liquid(T_wall, pressure, p_l["A"], p_l["B"], p_l["beta"], p_l["P0"])
    return (mu_bulk / mu_wall)**0.14

def friction_factor_corrected(Re, T_in, T_out, fluid, pressure_val, roughness):
    if Re < 2300:
        f = 64 / Re # Laminar flow
    elif Re < 4000:
        f = (1.82*np.log10(Re) - 1.64)**(-2) # Transitional flow
    else:
        f_initial = 0.0015 # initial guess
        f = fsolve(friction_factor_colebrook_white, f_initial, args=(Re, roughness[pipe_material], 2*pipe_radius))[0]
    return f * mu_wall_correction(T_in, T_out, fluid, pressure_val)

def convective_heat_transfer_coefficient(Re, Pr, k, D, f):
    if Re < 2300:
        Nu = 3.66  # Laminar flow in pipe
    else:
        Nu = (f/2.0)*(Re - 1000.0)*Pr / (1.0 + 12.7*np.sqrt(f/2.0)*(Pr**(2.0/3.0)-1.0))  # Turbulent flow in pipe (Gnielinski correlation)
    h = Nu * k / D
    return h 

def reynolds_number(rho, v, D, mu):
    return rho * v * D / mu

def loss_coefficient_for_fitting_contraction(radius_big, radius_small, number): # K_contraction
    return 0.42 * (1-(number*radius_small**2)/(radius_big**2))**2 # Crane TP-410, section 3.4

def loss_coefficient_for_fitting_expansion(radius_big, radius_small, number): # K_expansion
    return 0.5 * (1-(number*radius_small**2)/(radius_big**2)) 

def props_liq(T, P, material):

    rho = PropsSI("D","T",T,"P",P,material)
    cp  = PropsSI("C","T",T,"P",P,material)

    mu_l = viscosity_liquid(T, P, p_l["A"], p_l["B"], p_l["beta"], p_l["P0"])
    k_l = thermal_conductivity_liquid(T, P, p_l["k0"], p_l["a"], p_l["b"], inlet_liquid_temp[i_T], p_l["P0"])    

    return rho, mu_l, cp, k_l

def props_gas(T, P, material):

    rho = PropsSI("D","T",T,"P",P,material)
    cp  = PropsSI("C","T",T,"P",P,material)

    mu_g = viscosity_gas(T, p_g["mu0"], p_g["T0"], p_g["S"])
    k_g = thermal_conductivity_gas(T, p_g["k0"], waste_gas_temp_in, p_g["Sk"])    

    return rho, mu_g, cp, k_g

def wall_conduction_resistance(k_pipe):
    r_outer = pipe_radius + pipe_thickness
    R_cond = np.log(r_outer / pipe_radius) / (2 * np.pi * k_pipe * tube_length)
    return R_cond

def thermal_resistance_overall(h_i, h_o, R_cond):
    R_conv_i = 1 / (h_i * 2 * np.pi * pipe_radius * dx_tube)
    R_conv_o = 1 / (h_o * 2 * np.pi * (pipe_radius + pipe_thickness) * dx_tube)
    R_total = R_conv_i + R_cond + R_conv_o
    return R_total

def thermal_resistance_overall_boiler(h_i, h_o, R_cond):
    R_conv_i = 1 / (h_i * boiler_short_side * dx_boiler)
    R_conv_o = 1 / (h_o * boiler_short_side * dx_boiler) 
    R_total = R_conv_i + R_cond + R_conv_o
    return R_total

def boiling_q_and_h(T_wall_water_side, C_sf):
    
    delta_T_water_wall = T_wall_water_side - water_boiling_point_K # temperature difference between wall and water
    if delta_T_water_wall <= 0:
        return 0.0, 0.0
    # water characteristic identities. Q = 0 means water is saturated but doesn't consist vapour. We assumed it like that becuase we consider here the layer where  
    # boiling water contacts with wall that heats the water(keeping it at the same temperature) 
    rho_w = PropsSI('D', 'P', P_saturation, 'Q', 0, "water")     
    mu_w = PropsSI('V', 'P', P_saturation, 'Q', 0, "water")
    k_w = PropsSI('L', 'P', P_saturation, 'Q', 0, "water")
    c_p_w = PropsSI('C', 'P', P_saturation, 'Q', 0, "water")

    h_v = PropsSI('H', 'P', P_saturation, 'Q', 1, "water") #vapour's enthalpy
    h_w = PropsSI('H', 'P', P_saturation, 'Q', 0, "water") #water's enthalpy
    h_latent = h_v - h_w #Latent heat
    sigma = PropsSI('SURFACE_TENSION', 'p', P_saturation, 'Q', 0, 'water')
    pr_w = (PropsSI('C', 'P', P_saturation, 'Q', 0, 'water') * mu_w ) / k_w
    rho_v = PropsSI('D', 'P', P_saturation, 'Q', 1, 'water')

    #we will split the equation of rohsenow equation
    numerator = c_p_w * delta_T_water_wall * (mu_w ** 0.33) * (h_latent ** 0.33) 
    denominator = C_sf * h_latent * (pr_w)
    
    sqrt_term = np.sqrt(sigma / (g * (rho_w - rho_v)))
    # r term at the equation is taken 1 for water to maintain simplicity
    heat_flux = (numerator * (sqrt_term ** 0.33)) / denominator
    
    h = heat_flux / delta_T_water_wall

    return heat_flux, h

# Pressure at the end of the tube
x = np.linspace(0, tube_length, N)

# Optimization results storage
results = []
margin_error = np.empty(N)  
for i_m in range(len(mass_flow_rate)):
    m_dot = mass_flow_rate[i_m]
    for i_p in range(len(pressure)):
        p_in = pressure[i_p]
        for i_T in range(len(inlet_liquid_temp)):
            T_in = inlet_liquid_temp[i_T]
            # creating empty arrays for gas and liquid temperatures, pressures, and mass flow rates along the tube length
            T_pipe_wall_inner = np.empty(N)
            T_pipe_wall_outer = np.empty(N)
            T_liquid = np.empty(N)
            T_gas = np.empty(N) 

            T_gas[0] = waste_gas_temp_in
            T_liquid[0] = inlet_liquid_temp[i_T]

            P_gas = np.empty(N)
            P_liquid = np.empty(N)

            P_gas[0] = waste_gas_pressure_in
            P_liquid[0] = pressure[i_p]

            D_hydraulic_tube = 4 * np.pi * (tube_radius ** 2 - small_pipe_number * small_pipe_radius ** 2 ) / ( np.pi * (small_pipe_number * 2 * small_pipe_radius + 2 * tube_radius))

            #since the roughness of tube and pipe is different, we take average
            roughness_avg_for_gas = (roughness[tube_material] * np.pi * 2 * tube_radius + roughness[pipe_material] * np.pi * small_pipe_number * 2 * small_pipe_radius) / (2 * np.pi * tube_radius + 2 * small_pipe_number * small_pipe_radius)

            def find_wall_T(outer, inner):
                T_pipe_wall_outer[0] = outer
                T_pipe_wall_inner[0] = inner

                q_cond = U_liquid * ((T_gas[j] - T_liquid[j]) * dx_tube)
                q_gas = h_gas * 2 * np.pi * small_pipe_radius * dx_tube * (T_gas[j] - T_pipe_wall_outer[j])
                q_liquid = h_liquid * 2 * np.pi * small_pipe_radius * (T_pipe_wall_inner[j] - T_liquid[j]) * dx_tube

                return q_liquid - q_cond, q_cond - q_gas
            
            #similar to euler method at integration
            for j in range(0, N-1):
                if j == 0:

                    T_pipe_wall_inner[0] = T_pipe_wall_outer[0] = (T_liquid[j]+T_gas[j])/2

                #because the temperature of tube and pipe outer wall is different, we take the average
                T_avg_tube_pipe = (T_gas[j] * np.pi * 2 * tube_radius + T_pipe_wall_outer[j] * np.pi * small_pipe_number * 2 * small_pipe_radius) / (2 * np.pi * tube_radius + 2 * small_pipe_number * small_pipe_radius)

                T_boil = PropsSI("T", "P", P_liquid[j], "Q", 0, liquid)
        
                # Gas properties
                rho_gas, mu_gas, cp_gas, k_gas = props_gas(T_gas[j], P_gas[j], gas_list[waste_gas_index-1])
                # Liquid properties
                rho_liquid, mu_liquid, cp_liquid, k_liquid = props_liq(T_liquid[j], P_liquid[j], liquid)

                # Velocities
                A_flow_gas = np.pi * (tube_radius**2 - small_pipe_number*pipe_radius**2)
                v_gas = waste_gas_mass_flow_rate / (rho_gas * A_flow_gas)
                A_flow_liquid = np.pi * (small_pipe_radius**2) * small_pipe_number
                v_liquid = m_dot / (rho_liquid * A_flow_liquid)

                # Reynolds numbers
                Re_gas = reynolds_number(rho_gas, v_gas, D_hydraulic_tube, mu_gas) 
                Re_liquid = reynolds_number(rho_liquid, v_liquid, 2*small_pipe_radius, mu_liquid)

                # Prandtl numbers
                Pr_gas = cp_gas * mu_gas / k_gas
                Pr_liquid = cp_liquid * mu_liquid / k_liquid
                
                # Friction factors
                f_gas = friction_factor_corrected(Re_gas, T_gas[j], T_pipe_wall_outer[j], gas_list[waste_gas_index-1], P_gas[j], roughness[pipe_material])
                f_liquid = friction_factor_corrected(Re_liquid, T_liquid[j], T_pipe_wall_inner[j], liquid, P_liquid[j], roughness[liquid_type_index])

                # Convective heat transfer coefficients
                h_gas = convective_heat_transfer_coefficient(Re_gas, Pr_gas, k_gas, D_hydraulic_tube, f_gas)
                h_liquid = convective_heat_transfer_coefficient(Re_liquid, Pr_liquid, k_liquid, 2*small_pipe_radius, f_liquid)

                # Overall heat transfer coefficient
                U_liquid = 1 / thermal_resistance_overall(convective_heat_transfer_coefficient(Re_liquid, Pr_liquid, k_liquid, 2*small_pipe_radius, f_liquid), 
                convective_heat_transfer_coefficient(Re_gas, Pr_gas, k_gas, D_hydraulic_tube, f_gas), wall_conduction_resistance(k_material(pipe_material, T_avg_tube_pipe - 273.15)))
                #because the temperature of tube and pipe outer wall is different, we take the average
                T_avg_tube_pipe = (T_gas[j] * np.pi * 2 * tube_radius + T_pipe_wall_outer[j] * np.pi * small_pipe_number * 2 * small_pipe_radius) / (2 * np.pi * tube_radius + 2 * small_pipe_number * small_pipe_radius)
                
                # Heat transfer rate
                dQ = U_liquid * np.pi * 2 * pipe_radius * ((T_gas[j] - T_liquid[j]) * dx_tube) # at steady state, this is the heat transferred from gas to whole system

                T_pipe_wall_inner[j], T_pipe_wall_outer[j] = find_wall_T([(T_liquid[0]+T_gas[0])/2, (T_liquid[0]+T_gas[0])/2])
                #This is for calculating the first wall temperatures
                if j ==  0:
                    
                    # Gas properties
                    rho_gas, mu_gas, cp_gas, k_gas = props_gas(T_gas[j], P_gas[j], gas_list[waste_gas_index])
                    # Liquid properties
                    rho_liquid, mu_liquid, cp_liquid, k_liquid = props_liq(T_liquid[j], P_liquid[j], liquid)

                    # Velocities
                    A_flow_gas = np.pi * (tube_radius**2 - small_pipe_number*pipe_radius**2)
                    v_gas = waste_gas_mass_flow_rate / (rho_gas * A_flow_gas)
                    A_flow_liquid = np.pi * (small_pipe_radius**2) * small_pipe_number
                    v_liquid = m_dot / (rho_liquid * A_flow_liquid)

                    # Reynolds numbers
                    Re_gas = reynolds_number(rho_gas, v_gas, D_hydraulic_tube, mu_gas) 
                    Re_liquid = reynolds_number(rho_liquid, v_liquid, 2*small_pipe_radius, mu_liquid)

                    # Prandtl numbers
                    Pr_gas = cp_gas * mu_gas / k_gas
                    Pr_liquid = cp_liquid * mu_liquid / k_liquid

                    # Friction factors
                    f_gas = friction_factor_corrected(Re_gas, T_gas[j], T_pipe_wall_outer[j], P_gas[j], gas_list[waste_gas_index-1], roughness[pipe_material])
                    f_liquid = friction_factor_corrected(Re_liquid, T_liquid[j], T_pipe_wall_inner[j], P_liquid[j], liquid, roughness[pipe_material])

                    # Convective heat transfer coefficients
                    h_gas = convective_heat_transfer_coefficient(Re_gas, Pr_gas, k_gas, D_hydraulic_tube)
                    h_liquid = convective_heat_transfer_coefficient(Re_liquid, Pr_liquid, k_liquid, 2*small_pipe_radius)

                    # Overall heat transfer coefficient
                    U_liquid = 1 / thermal_resistance_overall(convective_heat_transfer_coefficient(Re_liquid, Pr_liquid, k_liquid, 2*pipe_radius, f_liquid), 
                    convective_heat_transfer_coefficient(Re_gas, Pr_gas, k_gas, D_hydraulic_tube, f_gas), wall_conduction_resistance(k_material(pipe_material, T_avg_tube_pipe - 273.15)))

                    #because the temperature of tube and pipe outer wall is different, we take the average
                    T_avg_tube_pipe = (T_gas[j] * np.pi * 2 * tube_radius + T_pipe_wall_outer[j] * np.pi * small_pipe_number * 2 * small_pipe_radius) / (2 * np.pi * tube_radius + 2 * small_pipe_number * small_pipe_radius)

                    # Heat transfer rate
                    dQ = U_liquid * np.pi * 2 * pipe_radius * ((T_gas[j] - T_liquid[j]) * dx_tube) # at steady state, this is the heat transferred from gas to whole system

                # Update temperatures
                T_liquid[j+1] = T_liquid[j] + dQ / (m_dot * cp_liquid)
                T_gas[j+1] = T_gas[j] - dQ / (waste_gas_mass_flow_rate * cp_gas)
                T_pipe_wall_inner[j+1] = T_liquid[j+1] + dQ * 1 / (h_liquid * 2 * np.pi * pipe_radius * dx_tube)
                T_pipe_wall_outer[j+1] = T_gas[j+1] + dQ * 1 / (h_gas * 2 * np.pi * (pipe_radius + pipe_thickness) * dx_tube)

                # Update pressures
                P_liquid[j+1] = P_liquid[j] - (f_liquid * (dx_tube) / (2 * small_pipe_radius) * (rho_liquid * v_liquid**2) / 2)
                P_gas[j+1] = P_gas[j] - (f_gas * (dx_tube) / (2 * pipe_radius) * (rho_gas * v_gas**2) / 2)

                if T_liquid[N-1] >= water_boiling_point_K:
                    print("Error: this system cannot operate with 2 phase flow. Liquid reached its boiling point.")
                    exit(0)
            boiler_area = boiler_short_side*boiler_long_side
            dx_tube = boiler_long_side / N
            T_liquid_boiler = np.empty(N) #T_liquid_boiler corresponds to the liquid that passes under the waste heat boiler
            #assuming the temperature of the water to be constant everywhere for simplicity
            
            T_liquid_boiler = np.empty(N)
            T_liquid_boiler[0] = T_liquid[N-1]

            P_liquid_boiler = np.empty(N)
            P_liquid_boiler[0] = P_liquid
            
            T_wall_water_side = np.empty(N)
            T_wall_liquid_side = np.empty(N)
            T_wall_water_side[0] = (T_liquid_boiler + water_boiling_point_K ) / 2 # this is inital guess, exact value to be determined later

            def T_boiler_tank_wall(values): #at steady q values of liquid, gas, and wall must be equal

                T_wall_liquid_side, T_wall_water_side = values

                q_liquid =  h_liquid_boiler * (dx_boiler * boiler_short_side)*(T_liquid_boiler[0] - T_wall_water_side[0])
                q_boiler =  heat_flux #this is calculated before at rohsenow calculation
                q_cond = (boiler_tank_thickness) * (T_wall_liquid_side[0] - T_wall_water_side[0]) / (dx_boiler * boiler_short_side)

                return [
                    q_liquid - q_cond,
                    q_cond - q_boiler
                ]
            
            D_hydraulic = 4 * boiler_area / (2 * (boiler_short_side + boiler_long_side))
            T_avg_boiler_wall = (T_wall_water_side[i] + T_wall_liquid_side[i]) / 2 

            electrical_energy_gained = 0

            for i in range(0, N-1):

                #liquid properties that embrace change at every newcoming pressure and temperature
                rho_liquid_boiler, mu_liquid_boiler, cp_liquid_boiler, k_liquid_boiler = props_liq(T_liquid_boiler[i], P_liquid_boiler[i], liquid)
                rho_water, mu_water, cp_water, k_water = props_liq(water_boiling_point_K, P_saturation, "water")

                heat_flux, h_water_boiler = boiling_q_and_h(T_wall_water_side[i], C_sf_list[boiler_tank_material]) #rohsenow calculation

                #liquid of velocity 
                v_liquid_boiler = m_dot * boiler_liquid_tank_thickness * dx_boiler / (pipe_cross_sectional_area * rho_liquid_boiler)

                #for rectangular shapes, instead of writing diameter, hydraulic diameter in terms of edge lengths of boiler is written
                Re_liquid_boiler = reynolds_number(rho_liquid_boiler, v_liquid_boiler, D_hydraulic, mu_liquid)

                #prandt number
                Pr_liquid_boiler = cp_liquid_boiler * mu_liquid_boiler / k_liquid_boiler
    
                #heat transfer coefficient
                h_liquid_boiler = convective_heat_transfer_coefficient(Re_liquid_boiler, Pr_liquid_boiler, k_liquid_boiler, D_hydraulic)

                T_wall_water_side[i], T_wall_liquid_side[i] = fsolve(T_boiler_tank_wall, [(T_liquid+T_gas)/2, (T_liquid+T_gas)/2])

                if i == 0:

                    T_wall_water_side[0], T_wall_liquid_side[0] = fsolve(T_boiler_tank_wall, [(T_liquid+T_gas)/2, (T_liquid+T_gas)/2])

                    #liquid properties that embrace change at every newcoming pressure and temperature
                    rho_liquid_boiler, mu_liquid_boiler, cp_liquid_boiler, k_liquid_boiler = props_liq(T_liquid_boiler[i], P_liquid_boiler[i], liquid)
                    rho_water, mu_water, cp_water, k_water = props_liq(water_boiling_point_K, P_saturation, "water")

                    heat_flux, h_water_boiler = boiling_q_and_h(T_wall_water_side, C_sf_list[boiler_tank_material])  #rohsenow calculation

                    #liquid of velocity 
                    v_liquid_boiler = m_dot * boiler_liquid_tank_thickness * dx_boiler / (pipe_cross_sectional_area * rho_liquid_boiler)

                    #for rectangular shapes, instead of writing diameter, hydraulic diameter in terms of edge lengths of boiler is written
                    Re_liquid_boiler = reynolds_number(rho_liquid_boiler, v_liquid, D_hydraulic, mu_liquid)

                    #prandt number
                    Pr_liquid_boiler = cp_liquid_boiler * mu_liquid_boiler / k_liquid_boiler

                #friction factor 
                f_liquid_boiler = friction_factor_corrected(Re_liquid_boiler, T_liquid_boiler[i], T_wall_liquid_side[i], liquid, P_liquid_boiler[i], roughness[boiler_tank_material])
                    
                electrical_energy_gained += heat_flux * dx_boiler * boiler_short_side * boiler_efficiency

                T_liquid_boiler[i+1] = T_liquid_boiler[i] + heat_flux / (m_dot * cp_liquid_boiler)
                T_wall_liquid_side[i+1] = T_liquid_boiler[j] - heat_flux / (h_liquid_boiler * boiler_short_side * dx_tube)
                T_wall_water_side[i+1] = T_wall_liquid_side[i+1] - boiler_tank_thickness * heat_flux / k_material(boiler_tank_material, T_avg_boiler_wall - 273.15) #converting to celcius
                
                P_liquid_boiler[i+1] = P_liquid_boiler[i] - (f_liquid_boiler * (dx_tube) / (D_hydraulic) * (rho_liquid_boiler * v_liquid_boiler**2) / 2)

            t = boiler_long_side / v_liquid_boiler  


            H_pump = (pressure[i_p] - P_liquid_boiler[i]) / (rho_liquid_boiler * g)

            P_hydraulic = m_dot * (pressure[i_p] - P_liquid_boiler[i]) / rho_liquid_boiler

            P_shaft = P_hydraulic / pump_efficiency # to be input
            electrical_energy_consumed = t * P_shaft

            margin_error[i_T] = T_liquid_boiler[i] - inlet_liquid_temp[i_T]
            if i_T == 0:
                results.insert(i_p + (i_m)*len(pressure) + 1, electrical_energy_gained - electrical_energy_consumed) # we are inserting net energy gains

            elif margin_error[i_T] - margin_error[i_T - 1] < 0:
                margin_error_lowest = margin_error[i_T]
                results.insert(i_p + (i_m)*len(pressure) + 1, electrical_energy_gained - electrical_energy_consumed)
        
        if results[i_p + (i_m)*len(pressure) + 1] < results[i_p + (i_m)*len(pressure)]:
            best_pump_power = P_hydraulic
            best_mass_flow_rate = mass_flow_rate[i_m]
            best_pressure = pressure[i_p]
        
print(f"For the liquid{liquid}:\n Best pump power to be maintained:{P_hydraulic} \n Best mass flow rate:{mass_flow_rate:.3f}\n Best pressure = {best_pressure:.3f} \n net energy gained per time{t} is {electrical_energy_gained - electrical_energy_consumed} ")