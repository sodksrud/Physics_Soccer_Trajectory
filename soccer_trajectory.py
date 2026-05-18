import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# =====================================================================
# STEP 1: Load Data & Initialize Constants
# =====================================================================
# Make sure the CSV file name matches your local file.
df = pd.read_csv('data.csv', encoding='utf-8')
t = df['시간(s)'].values
x = df['X좌표(m)'].values
z = df['Y좌표(m)'].values

# Physical constants (based on the paper)
m = 0.424  # Mass of the soccer ball (kg)
A = 0.0375  # Cross-sectional area (m^2)
rho = 1.2  # Air density (kg/m^3)
beta = (rho * A) / (2 * m)
g = 9.8  # Acceleration due to gravity (m/s^2)
dt = 0.1  # Time interval between frames (s)

# =====================================================================
# STEP 2: Calculate Velocity & Acceleration (Central Difference)
# =====================================================================
vx = np.zeros_like(x);
vz = np.zeros_like(z)
ax = np.zeros_like(x);
az = np.zeros_like(z)

for i in range(1, len(t) - 1):
    vx[i] = (x[i + 1] - x[i - 1]) / (2 * dt)
    vz[i] = (z[i + 1] - z[i - 1]) / (2 * dt)
    ax[i] = (x[i + 1] - 2 * x[i] + x[i - 1]) / (dt ** 2)
    az[i] = (z[i + 1] - 2 * z[i] + z[i - 1]) / (dt ** 2)

# =====================================================================
# STEP 3: Calculate Drag (C_D) and Lift (C_L) Coefficients
# =====================================================================
valid_t = []
CD_list = []
CL_list = []

print("--- Frame-by-Frame Aerodynamic Coefficients ---")
for i in range(1, len(t) - 1):
    v2 = vx[i] ** 2 + vz[i] ** 2
    v = np.sqrt(v2)
    v3 = v ** 3

    if v3 > 0:
        # Equations derived from Newton's Second Law
        CD = -(((az[i] + g) * vz[i] + ax[i] * vx[i]) / (beta * v3))
        CL = (((az[i] + g) * vx[i] - ax[i] * vz[i]) / (beta * v3))

        # Filtering extreme noise caused by manual tracking
        if -2 < CD < 3 and -3 < CL < 3:
            valid_t.append(t[i])
            CD_list.append(CD)
            CL_list.append(CL)
            print(f"t = {t[i]:.1f}s | C_D: {CD:.3f} | C_L: {CL:.3f}")

avg_CD = np.mean(CD_list)
avg_CL = np.mean(CL_list)

print(f"\n=> Calculated Average C_D: {avg_CD:.3f}")
print(f"=> Calculated Average C_L: {avg_CL:.3f}\n")


# =====================================================================
# STEP 4: Numerical Simulation (ODE Integration)
# =====================================================================
def soccer_ball_equations(state, time, CD, CL, beta, g):
    pos_x, pos_z, vel_x, vel_z = state
    speed = np.sqrt(vel_x ** 2 + vel_z ** 2)

    dxdt = vel_x
    dzdt = vel_z
    dvxdt = -beta * speed * (CD * vel_x + CL * vel_z)
    dvzdt = beta * speed * (-CD * vel_z + CL * vel_x) - g

    return [dxdt, dzdt, dvxdt, dvzdt]


# Initial conditions for simulation (at t = 0.1s)
initial_state = [x[1], z[1], vx[1], vz[1]]
t_sim = np.linspace(t[1], t[-2], 100)

# Solve the differential equations
solution = odeint(soccer_ball_equations, initial_state, t_sim, args=(avg_CD, avg_CL, beta, g))
x_sim = solution[:, 0]
z_sim = solution[:, 1]

# =====================================================================
# STEP 5: Data Visualization (Plots)
# =====================================================================

# Plot 1: Raw Experimental Trajectory
plt.figure(figsize=(10, 5))
plt.plot(x, z, 'ro--', label='Tracker Data', markersize=6)
plt.title('Experimental Soccer Ball Trajectory')
plt.xlabel('Horizontal Distance (m)')
plt.ylabel('Vertical Height (m)')
plt.legend()
plt.grid(True)


# Plot 2: Simulation vs Experimental Trajectory
plt.figure(figsize=(10, 5))
plt.plot(x[1:-1], z[1:-1], 'ro', label='Experimental Data', markersize=6)
plt.plot(x_sim, z_sim, 'b-', linewidth=2, label=f'Simulation (Avg $C_D$={avg_CD:.2f}, $C_L$={avg_CL:.2f})')
plt.title('Comparison: Experiment vs. Numerical Simulation')
plt.xlabel('Horizontal Distance (m)')
plt.ylabel('Vertical Height (m)')
plt.legend()
plt.grid(True)


# Plot 3: Variations of Coefficients over Time
plt.figure(figsize=(10, 5))
plt.plot(valid_t, CD_list, 'g-o', label='$C_D$ (Drag)')
plt.plot(valid_t, CL_list, 'm-o', label='$C_L$ (Lift)')
plt.title('Aerodynamic Coefficients over Time')
plt.xlabel('Time (s)')
plt.ylabel('Coefficient Value')
plt.legend()
plt.grid(True)

plt.show()