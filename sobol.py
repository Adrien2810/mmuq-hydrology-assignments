"""Sobol sensitivity analysis"""
import numpy as np
import pandas as pd

from hmg import HBV001A

# Step 1: Generate matrices A and B
Bounds = [(0, 0), (-2, 3), (0, 3), (0, 10), (0, 100), (5, 700), (100, 700), (0.01, 10),
          (0, 20), (0, 100), (0, 1), (0, 1), (0, 200), (0.01, 1), (0, 1), (0, 1), (0, 1), (0, 1)]
lower_bounds = [b[0] for b in Bounds]
upper_bounds = [b[1] for b in Bounds]

for j in range(len(Bounds)):
    parameter_column_A = np.random.uniform(
        lower_bounds[j], upper_bounds[j], (100, 1))
    if j == 0:
        A = np.column_stack((np.empty((100, 0)), parameter_column_A))
    else:
        A = np.column_stack((A, parameter_column_A))

for j in range(len(Bounds)):
    parameter_column_B = np.random.uniform(
        lower_bounds[j], upper_bounds[j], (100, 1))
    if j == 0:
        B = np.column_stack((np.empty((100, 0)), parameter_column_B))
    else:
        B = np.column_stack((B, parameter_column_B))

# Step 2: Generate C matrices
C_matrices = np.zeros(shape=(len(Bounds), 100, len(Bounds)))
for j in range(len(Bounds)):
    C_j = B.copy()
    C_j[:, j] = A[:, j]
    C_matrices[j] = C_j

# Step 3: Get the mean  OFV for each matrix and each parameter set
data_dir = r"C:\MMUQ\Assignment 3"
time_series_file = pd.read_csv(
    data_dir + r"\time_series___24163005.csv", sep=";")
tems = time_series_file["tavg__ref"].values
ppts = time_series_file["pptn__ref"].values
pets = time_series_file["petn__ref"].values
diso = time_series_file["diso__ref"].values
cumm_area = float(pd.read_csv(
    data_dir + r"\area___24163005.csv", sep=";", index_col=0).values[0, 0])
dslr = cumm_area / (3600 * 1000)
tsps = len(tems)
model = HBV001A()
model.set_inputs(tems, ppts, pets)
model.set_outputs(tsps)
model.set_discharge_scaler(dslr)
model.set_optimization_flag(0)


def calc_ofv(obs: np.ndarray, sim: np.ndarray) -> float:
    """Compute OFV."""
    obs = np.asarray(obs, float)
    sim = np.asarray(sim, float)
    mask = np.isfinite(obs) & np.isfinite(sim)
    obs = obs[mask]
    sim = sim[mask]
    denom = np.sum((obs - obs.mean()) ** 2)
    if denom <= 0:
        return float("-inf")
    return 1-(1.0 - np.sum((obs - sim) ** 2) / denom)


ofv_A = np.zeros(100)
for j in range(100):
    prms0 = A[j]
    model.set_parameters(prms0)
    model.run_model()
    outputs = model.get_outputs()
    discharge = model.get_discharge()
    ofv_A[j] = calc_ofv(diso, discharge)

ofv_B = np.zeros(100)
for j in range(100):
    prms0 = B[j]
    model.set_parameters(prms0)
    model.run_model()
    outputs = model.get_outputs()
    discharge = model.get_discharge()
    ofv_B[j] = calc_ofv(diso, discharge)

ofv_C = np.zeros((len(Bounds), 100))
for i in range(len(Bounds)):
    for j in range(100):
        prms0 = C_matrices[i][j]
        model.set_parameters(prms0)
        model.run_model()
        outputs = model.get_outputs()
        discharge = model.get_discharge()
        ofv_C[i][j] = calc_ofv(diso, discharge)

# Step 4: Calculate Sobol indices
S_i = np.zeros(len(Bounds))
f0_sq = (ofv_A.mean()) * (ofv_B.mean())
S_i_denominator = np.mean(ofv_A**2) - f0_sq
S_i_numerator = np.zeros(len(Bounds))
for j in range(len(Bounds)):
    S_i_numerator[j] = np.mean(
        ofv_B * (ofv_C[j] - ofv_A))
    S_i[j] = S_i_numerator[j] / S_i_denominator
print("Sobol indices S_i:", S_i)
print("Sum of Sobol indices S_i:", np.sum(S_i))

S_Ti = np.zeros(len(Bounds))
S_Ti_numerator = np.zeros(len(Bounds))
for j in range(len(Bounds)):
    S_Ti_numerator[j] = np.mean(
        ofv_A * (ofv_A - ofv_C[j]))
    S_Ti[j] = S_Ti_numerator[j] / S_i_denominator
print("Sobol total indices S_Ti:", S_Ti)
print("Sum of Sobol total indices S_Ti:", np.sum(S_Ti))
