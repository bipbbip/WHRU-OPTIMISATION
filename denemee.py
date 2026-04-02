import numpy as np
import CoolProp.CoolProp as CP

# --- Define the function (omitted for brevity, assume the previous function is loaded) ---

def calculate_boiling_q_and_h_with_coolprop_by_T(
    T_wall,          # Inner surface temperature of the tank bottom (K)
    P_abs=101325,    # Absolute Pressure (Pa). Determines T_sat.
    fluid='Water',   # Fluid name
    C_sf=0.013,      
    r=0.33,          
    n=1.0            
):
    # --- Function body from previous response goes here ---
    T_sat = CP.PropsSI('T', 'P', P_abs, 'Q', 0, fluid)
    Delta_T_e = T_wall - T_sat
    
    if Delta_T_e <= 0:
        print("Warning: Wall is not hotter than saturation temperature.")
        return 0.0, 0.0
    #props func
    rho_l = CP.PropsSI('D', 'P', P_abs, 'Q', 0, fluid)     
    mu_l = CP.PropsSI('V', 'P', P_abs, 'Q', 0, fluid)
    k_l = CP.PropsSI('L', 'P', P_abs, 'Q', 0, fluid)
    c_p_l = CP.PropsSI('C', 'P', P_abs, 'Q', 0, fluid)
    #-
    h_v = CP.PropsSI('H', 'P', P_abs, 'Q', 1, fluid)
    h_l = CP.PropsSI('H', 'P', P_abs, 'Q', 0, fluid)
    h_fg = h_v - h_l                                       
    sigma = CP.PropsSI('SURFACE_TENSION', 'P', P_abs, 'Q', 0, fluid) 
    Pr_l = (CP.PropsSI('C', 'P', P_abs, 'Q', 0, fluid) * mu_l) / k_l
    g = 9.81
    rho_v = CP.PropsSI('D', 'P', P_abs, 'Q', 1, fluid)     
    
    
    LHS_Numerator = c_p_l * Delta_T_e
    LHS_Denominator = C_sf * h_fg * (Pr_l**n)
    LHS_Term = LHS_Numerator / LHS_Denominator
    
    Square_Root_Term = np.sqrt(sigma / (g * (rho_l - rho_v)))
    Pre_Factor = (mu_l * h_fg) / Square_Root_Term

    q_double_prime = Pre_Factor * (LHS_Term**(1/r))
    h = q_double_prime / Delta_T_e

    print(f"--- Boiling Analysis for {fluid} at {P_abs/1000:.1f} kPa ---")
    print(f"Saturation Temp (T_sat): {T_sat:.2f} K ({T_sat - 273.15:.1f} °C)")
    print(f"Wall Temp (T_w): {T_wall:.2f} K ({T_wall - 273.15:.1f} °C)")
    print(f"Wall Superheat (Delta T_e): {Delta_T_e:.2f} K")
    print("---------------------------------------")
    print(f"Calculated Heat Flux (q''): {q_double_prime:.0f} W/m^2")
    print(f"Heat Transfer Coefficient (h): {h:.2f} W/(m^2·K)")
    
    return q_double_prime, h

# --- Example Usage with New Pressure ---

P_sat_new = 150000 # Pa (1.5 bar)
T_wall_new = 388.0 # K (A wall temperature of 114.85 C, giving a Delta_T_e of 6.85 K)

q_flux, h_coeff = calculate_boiling_q_and_h_with_coolprop_by_T(
    T_wall=T_wall_new, 
    P_abs=P_sat_new, 
    C_sf=0.013 
)