import brainpy as bp
import brainpy.math as bm

bm.set_platform('cpu')
bm.set_dt(dt=0.02)
bm.enable_x64()

# Constants for the neuron model
C = 1  # Capacitance
E_L = -78  # Leak reversal potential, different from high-threshold model
g_L = 8  # Leak conductance
g_Na = 20  # Sodium conductance
g_K = 10  # Potassium conductance
E_K = -90  # Potassium reversal potential
E_Na = 60  # Sodium reversal potential
Vm_half = -20  # Half-maximal voltage for m_inf
k_m = 15  # Slope factor for m_inf
Vn_half = -45  # Half-maximal voltage for n_inf, different from high-threshold model
k_n = 5  # Slope factor for n_inf
tau = 1  # Time constant


# Function to integrate over V
@bp.odeint(method='exp_auto')
def int_V(V, t, n, Iext):
    m_inf = 1 / (1 + bm.exp((Vm_half - V) / k_m))  # Steady-state value for m
    I_leak = g_L * (V - E_L)  # Leak current
    I_Na = g_Na * m_inf * (V - E_Na)  # Sodium current
    I_K = g_K * n * (V - E_K)  # Potassium current
    dvdt = (-I_leak - I_Na - I_K + Iext) / C  # Change in voltage over time
    return dvdt

# Function to integrate over n
@bp.odeint(method='exp_auto')
def int_n(n, t, V):
    n_inf = 1 / (1 + bm.exp((Vn_half - V) / k_n))  # Steady-state value for n
    dndt = (n_inf - n) / tau  # Change in n over time
    return dndt

# Create a 2D phase plane analyzer with the neuron model
analyzer = bp.analysis.PhasePlane2D(
    model=[int_n, int_V],  # The neuron model functions
    target_vars={'n': [0., 1.], 'V': [-90, 20]},  # The range of variables n and V
    pars_update={'Iext': 50.},  # Update the external current parameter
    resolutions={'n': 0.01, 'V': 0.1}  # The resolution for n and V
)

# Plot the nullclines of the system
analyzer.plot_nullcline()

# Plot the vector field of the system
analyzer.plot_vector_field()

# Plot the fixed points of the system
analyzer.plot_fixed_point()

# Plot the trajectory of the system with initial conditions for n and V
analyzer.plot_trajectory({'n': [0.2, 0.4], 'V': [-10, -80]},
                         duration=100., show=True)

# Create a 2D bifurcation analyzer with the neuron model
analyzer = bp.analysis.Bifurcation2D(
  model=[int_V, int_n],  # The neuron model functions
  target_vars={"V": [-90., 20.], 'n': [0., 1.]},  # The range of variables V and n
  target_pars={'Iext': [0, 50.]},  # The range of parameter Iext
  resolutions={'Iext': 0.1},  # The resolution for Iext
)

# Plot the bifurcation diagram with the top 30 eigenvalues
analyzer.plot_bifurcation(num_rank=30)

# Plot the limit cycle by simulating the system for a given duration
analyzer.plot_limit_cycle_by_sim(duration=100., show=True)