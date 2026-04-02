# Copilot Instructions for TÜBITAK 2204-A WHRU Optimization Project

## Project Overview
This is a **Waste Heat Recovery Unit (WHRU) optimization system** combining shell-and-tube heat exchangers with heat pumps. The primary goal is to optimize energy efficiency by finding optimal fluid type, pressure, and mass flow rate combinations in the heat exchanger circuit.

## Architecture & Data Flow

### Key Components
1. **`2204A.py`** - Main optimization engine
   - User input collection for system parameters (pipe dimensions, materials, temperatures)
   - Iterates through fluid candidates (DowthermA, TherminolVP1, Syltherm800, etc.)
   - Solves coupled equations: heat transfer, fluid dynamics, thermal resistance
   - Goal: Find best fluid type, operating pressure, and mass flow rate

2. **`csvokuma.py`** & `try.py`** - Experimental/prototype scripts
   - May contain legacy models or alternative implementations
   - Treat as reference; don't assume they're active in main flow

3. **Material Property Files** (CSV-based lookup)
   - `pipe_materials.csv`: ~30 material types with thermal conductivity and roughness
   - `tube_materials.csv`: ~30+ material types (flue/boiler tubes)
   - Used for dynamic material property loading instead of hardcoding

### Data & Unit Conventions
- **All values in SI units** (K for temperature, Pa for pressure, m for dimensions, W for power)
- Temperature ranges: DowthermA/others max 643-673 K; Toluene restricted to 393 K
- Pressure sweep: 1.01325–50.6625 bar (101325–5066250 Pa)
- Mass flow rates tested: 0.1–5 kg/s

## Critical Patterns & Workflows

### 1. Material Property Resolution
**Pattern**: User input → CSV lookup → fallback manual input
```python
# Example from 2204A.py lines 57-74
for i in flue_df.index:
    if flue_df.loc[i, "material"] == pipe_material:
        row = flue_df.loc[[i]]
        break
if row is not None:
    tube_roughness = row["roughness_m"].values[0]
else:
    tube_roughness = float(input("Enter roughness (m): "))
```
- **Action**: When adding new materials, update both CSV files; don't hardcode properties
- **When enhancing**: Add thermal conductivity & roughness columns if missing

### 2. Thermophysical Property Queries
**Pattern**: Use CoolProp for real-time property lookup with fallback
```python
def props(T, P, material):
    rho = propsSI("D","T",T,"P",P,material)
    mu  = propsSI("V","T",T,"P",P,material)
    cp  = propsSI("C","T",T,"P",P,material)
    k   = propsSI("L","T",T,"P",P,material)
    return rho, mu, cp, k
```
- **Key functions**: `propsSI()` returns density, viscosity, specific heat, thermal conductivity
- **Failure mode**: Some fluid-temperature combos raise exceptions → wrap in try/except
- **Candidates for optimization**: DowthermA, TherminolVP1, Syltherm800, Toluene, Santotherm100, MarlothermSH, ParathermNF

### 3. Hydraulic Calculations (Turbulent Flow Model)
**Pattern**: Friction factor via Colebrook-White for Re > 4000, with laminar/transitional handling

Key functions:
- `friction_factor_colebrook_white()` – implicit Colebrook-White equation solved with fsolve
- `mu_wall_correction()` – Viscosity temperature correction: $(μ_{bulk}/μ_{wall})^{0.14}$
- `reynolds_number()` – Standard Re = ρvD/μ
- `loss_coefficient_for_fitting_contraction()` & `...expansion()` – Crane TP-410 correlations

**Discretization**: Tube divided into N=1000 segments (dx_tube = tube_length / N)

### 4. Heat Transfer Correlations
- **Convective coeff**: Gnielinski correlation for Nu in turbulent flow:
  $$Nu = \frac{(f/2)(Re-1000)Pr}{1+12.7\sqrt{f/2}(Pr^{2/3}-1)}$$
  Laminar: Nu = 3.66 (constant)
- **Thermal resistance**: Multi-layer model (convection inside pipe → conduction through wall → convection outside)
  ```python
  R_total = R_conv_i + R_cond + R_conv_o
  ```

### 5. Optimization Loop Structure
Expected iteration pattern (inferred from variable declarations):
```python
for fluid in fluid_types:
    for pressure in np.linspace(...):
        for mass_flow_rate in np.linspace(...):
            for inlet_liquid_temp in np.linspace(...):
                # Solve heat exchanger system
                # Evaluate efficiency metric
                # Track best_fluid, inlet_pressure, best_mass_flow_rate
```
- **Convergence**: Use `fsolve()` for implicit equations (friction factor, heat balance)
- **Bounds**: Respect max_operating_temp_K per fluid to avoid CoolProp domain errors

## Development & Testing Guidelines

### Running the Project
1. Install dependencies: `pip install scipy pandas CoolProp numpy matplotlib thermo`
2. Execute: `python 2204A.py` → Interactive prompt for all input parameters
3. Ensure CSV files (`pipe_materials.csv`, `tube_materials.csv`) are in working directory

### Extending the Code
- **Add new fluid**: Insert in `fluid_types` list and `max_operating_temp_K` parallel array
- **Add new material**: Append row to `pipe_materials.csv` or `tube_materials.csv` with thermal_conductivity_W_mK & roughness_m
- **Modify correlations**: Update correlation functions and test against validation data (track before/after efficiency values)
- **Debug output**: Leverage `print()` statements; no logging framework in place

### Common Pitfalls
- **CoolProp queries beyond fluid limits** → Wrap in try/except; fallback to constant properties or raise user-friendly error
- **CSV material name mismatches** → Case-sensitive after `.strip().lower()`; verify exact spelling
- **Division by zero in hydraulics** → Check Re > 2300 before applying turbulent correlations
- **Discretization artifacts** → Increase N (segments) if gradient fields show oscillations

## Key Files for Reference
- **Correlations & equations**: lines 120–145 in `2204A.py` (convection, friction factor)
- **Material lookup pattern**: lines 57–85 in `2204A.py` (reuse this template)
- **Property function template**: `try.py` lines 35–48 (fallback when CoolProp unavailable)
- **Material databases**: `pipe_materials.csv`, `tube_materials.csv` (expandable)

## Avoiding Common Mistakes
1. Don't hardcode thermal conductivity/roughness → always use CSV + fallback
2. Don't ignore CoolProp errors silently → log them for debugging
3. Don't forget SI units → temperature in K, pressure in Pa, length in m
4. Don't assume CSV header names match code expectations → validate on first file read
