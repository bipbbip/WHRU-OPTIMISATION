"""
import pandas as pd
# All physical properties are stored in CSV files for easy access and modification
# All values are in SI units

flue_df = pd.read_csv("flue_materials.csv")

def manual_input():
    thermal_conductivity = float(input("Enter thermal conductivity (W/mK): "))
    roughness = float(input("Enter roughness (m): "))
    return thermal_conductivity, roughness

row = None
flue_material = input("Enter the flue material: ").strip().lower()

for i in flue_df.index:
    if flue_df.loc[i, "material"] == flue_material:
        row = flue_df.loc[[i]]
        break

if row is not None:
    flue_thermal_conductivity = row["thermal_conductivity_W_mK"].values[0]
    flue_roughness = row["roughness_m"].values[0]

else:
    print("Flue material not found in CSV. Please enter values manually.")
    flue_thermal_conductivity, flue_roughness = manual_input()

print("Flue:", flue_material)
print("Thermal Conductivity:", flue_thermal_conductivity)
print("Roughness:", flue_roughness)

pipe_df = pd.read_csv("pipe_materials.csv")    
pipe_material = input("Enter the pipe material: ").strip().lower()

row = None

for i in pipe_df.index:
    if pipe_df.loc[i, "material"] == pipe_material:
        row = pipe_df.loc[[i]]
        break

if row is not None:
    pipe_thermal_conductivity = row["thermal_conductivity_W_mK"].values[0]
    pipe_roughness = row["roughness_m"].values[0]

else:
    print("Pipe material not found in CSV. Please enter values manually.")
    pipe_thermal_conductivity, pipe_roughness = manual_input()

print("Pipe:", pipe_material)
print("Thermal Conductivity:", pipe_thermal_conductivity)
print("Roughness:", pipe_roughness) 
"""
"""
CoolProp-based variable-property model for swirling annular flue flow.

Outputs temperature profile T(y) (K), local heat flux q''(y) (W/m^2) and total Q (W).

Author: ChatGPT (adapt to your units/needs)
"""

import numpy as np
from CoolProp.CoolProp import PropsSI

# -------------------------
# USER INPUTS (edit)
# -------------------------
fluid = "Air"              # CoolProp fluid string or mixture (e.g. "Air", "HEOS::Air", or "TdepMix:0.77N2&0.11CO2&0.09H2O&0.03O2")
T_top_C = 450.0            # top gas temperature (°C)
P_top = 101325.0           # top absolute pressure (Pa)
m_dot = 0.8                # mass flow rate through annulus (kg/s) -- constant mass flow
D_inner = 0.20             # outer diameter of inner flue (m)
D_outer = 0.40             # inner diameter of outer pipe (m)
L = 15.0                   # vertical length of the annulus (m)
omega = 10.0               # angular velocity, rad/s  (set to None if supplying v_theta_mean)
v_theta_mean_input = None  # optionally set mean tangential velocity (m/s) instead of omega

# wall/ambient
t_wall = 0.01              # wall thickness (m)
k_wall = 45.0              # wall thermal conductivity (W/m.K)
eps_gas = 0.0              # gas emissivity inside (if radiative exchange inside annulus is desired; 0->ignored)
eps_wall = 0.8             # wall emissivity to ambient for radiation
T_amb_C = 20.0             # ambient temperature (°C)
h_outer = 5.0              # outer convective coefficient W/m^2K (outside pipe)

# numerical
N = 500                    # number of axial slices (increase for accuracy)
include_radiation = True   # include simple radiation from inner wall to ambient (approx)
sigma = 5.670374419e-8     # Stefan-Boltzmann constant

# small safety
min_T = 50.0 + 273.15      # minimum allowed gas temperature (K)
max_T = 2000.0 + 273.15    # maximum allowed gas temperature (K)

# -------------------------
# Derived geometry
# -------------------------
R_i = D_inner / 2.0
R_o = D_outer / 2.0
A_ann = np.pi * (R_o**2 - R_i**2)        # cross-sectional area (m^2)
P_inner_surface = 2.0 * np.pi * R_o      # inner surface perimeter (m)  (wet perimeter of outer pipe)
Dh = (4.0 * A_ann) / (2*np.pi*(R_o+R_i)) # hydraulic diameter of annulus (exact formula)
dy = L / N

# convert temps to K
T_top = T_top_C + 273.15
T_amb = T_amb_C + 273.15

# prepare arrays
ys = np.linspace(0.0, L, N+1)  # y = 0 top, y = L bottom
T = np.empty_like(ys)          # gas temperature (K)
q_pp = np.empty_like(ys)       # local heat flux W/m^2
q_line = np.empty_like(ys)     # heat per meter along pipe (W/m)
rho_arr = np.empty_like(ys)
cp_arr = np.empty_like(ys)
mu_arr = np.empty_like(ys)
k_gas_arr = np.empty_like(ys)
h_inner_arr = np.empty_like(ys)
U_arr = np.empty_like(ys)
v_axial_arr = np.empty_like(ys)
v_theta_arr = np.empty_like(ys)
Re_arr = np.empty_like(ys)
Nu_arr = np.empty_like(ys)

# initialize top
T[0] = np.clip(T_top, min_T, max_T)

# helper: property function
def gas_props(Tk, p):
    """Return dict of properties at T (K) and p (Pa) using CoolProp."""
    # PropsSI: output in SI
    rho = PropsSI('D', 'T', Tk, 'P', p, fluid)          # kg/m3
    cp = PropsSI('Cpmass', 'T', Tk, 'P', p, fluid)      # J/kg.K
    mu = PropsSI('V', 'T', Tk, 'P', p, fluid)          # Pa.s (viscosity)
    k = PropsSI('L', 'T', Tk, 'P', p, fluid)           # W/m.K (conductivity)
    return dict(rho=rho, cp=cp, mu=mu, k=k)

# march downwards
p_local = P_top  # NOTE: can include pressure drop integration later; here assume constant pressure (or small drop)
for i in range(0, N):
    # get properties at current slice top of segment
    props = gas_props(T[i], p_local)
    rho = props['rho']; cp = props['cp']; mu = props['mu']; k_gas = props['k']
    rho_arr[i] = rho; cp_arr[i] = cp; mu_arr[i] = mu; k_gas_arr[i] = k_gas

    # velocities
    v_axial = m_dot / (rho * A_ann)
    if v_theta_mean_input is not None:
        v_theta_mean = v_theta_mean_input
    else:
        r_mean = 0.5 * (R_i + R_o)
        v_theta_mean = omega * r_mean
    v_eff = np.sqrt(v_axial**2 + v_theta_mean**2)

    v_axial_arr[i] = v_axial
    v_theta_arr[i] = v_theta_mean

    # Re, Pr
    Re = rho * v_eff * Dh / mu
    Pr = cp * mu / k_gas

    # choose Nu correlation (approx)
    if Re < 2300:
        # laminar annulus: use Nu ~ 3.66 (approx for fully developed), but swirl may alter it:
        Nu = 3.66
    else:
        # turbulent: Dittus-Boelter-like (engineering approximation)
        Nu = 0.023 * (Re**0.8) * (Pr**0.4)

    # inner convective coefficient (to outer pipe inner surface)
    h_inner = Nu * (k_gas / Dh)

    # overall U (includes conduction through wall and outer h)
    U = 1.0 / (1.0/h_inner + t_wall/k_wall + 1.0/h_outer)

    # radiation (simple): radiative heat transfer coefficient from wall to ambient
    if include_radiation:
        # approximate equivalent h_rad at wall temperature ~ film temp
        T_wall_est = (T[i] + T_amb) / 2.0  # crude
        h_rad = eps_wall * sigma * ( (T_wall_est**2 + T_amb**2) * (T_wall_est + T_amb) )
    else:
        h_rad = 0.0

    # combine outer convection and radiation for outer side (we already included outer convection inside U)
    # If including radiation separately from U, adjust accordingly; here we add h_rad to outer conv.
    # recompute U including h_rad on outside:
    if include_radiation and h_rad > 0:
        U = 1.0 / (1.0/h_inner + t_wall/k_wall + 1.0/(h_outer + h_rad))

    # fill arrays
    Re_arr[i] = Re; Nu_arr[i] = Nu
    h_inner_arr[i] = h_inner; U_arr[i] = U

    # energy ODE: dT/dy = - (U * P)/(m_dot * cp) * (T - T_amb)
    K_local = (U * P_inner_surface) / (m_dot * cp)
    dT = - K_local * (T[i] - T_amb) * dy
    T[i+1] = T[i] + dT

    # local heat flux to wall (based on h_inner and local wall/gas delta)
    # wall temperature unknown; approximate wall at T_amb + (T - T_amb)*(h_inner/(h_inner + h_outer + h_rad))
    # simpler: q'' = h_inner * (T_gas - T_wall), but we approximate T_wall ~= T_amb (for thin insulation) or solve iteratively
    T_wall_est = T_amb + (T[i] - T_amb) * (h_inner / (h_inner + h_outer + (h_rad if include_radiation else 0.0)))
    qpp_local = h_inner * (T[i] - T_wall_est)   # W/m^2 inner surface
    q_pp[i] = qpp_local
    q_line[i] = qpp_local * P_inner_surface

# final element props
props = gas_props(T[-1], p_local)
rho_arr[-1] = props['rho']; cp_arr[-1] = props['cp']; mu_arr[-1] = props['mu']; k_gas_arr[-1] = props['k']

# compute last heat flux using last h_inner estimate (copy previous)
h_inner_arr[-1] = h_inner_arr[-2]
U_arr[-1] = U_arr[-2]
q_pp[-1] = q_pp[-2]
q_line[-1] = q_line[-2]

# integrate total Q (W) along length by trapezoidal integration of q_line
Q_total = np.trapz(q_line, ys)

# -------------------------
# Results (print summary)
# -------------------------
print("=== Summary ===")
print(f"Fluid: {fluid}")
print(f"Top T = {T_top - 273.15:.2f} °C, Top p = {P_top:.1f} Pa, m_dot = {m_dot:.4f} kg/s")
print(f"Annulus area A = {A_ann:.4e} m^2, Dh = {Dh:.4e} m, perimeter = {P_inner_surface:.4f} m")
print(f"Total length L = {L:.2f} m, slices = {N}")
print(f"Estimated total heat transfer Q ≈ {Q_total:.1f} W")
print()
# print a short table of selected points
print("y (m)   T_gas (°C)   rho (kg/m3)   v_axial (m/s)   Re      Nu     h_inner (W/m2K)   q'' (W/m2)")
for idx in np.linspace(0, N, min(11, N+1), dtype=int):
    print(f"{ys[idx]:5.2f}   {T[idx]-273.15:9.2f}   {rho_arr[idx]:10.3f}   {v_axial_arr[idx]:13.3f}   {Re_arr[idx]:7.2e}   {Nu_arr[idx]:6.2f}   {h_inner_arr[idx]:14.2f}   {q_pp[idx]:10.1f}")

# Outputs for user use
# arrays: ys (m), T (K), q_pp (W/m2), q_line (W/m)
# Q_total (W)
