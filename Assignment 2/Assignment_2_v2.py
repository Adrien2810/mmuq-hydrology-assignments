# We import all posisble relevant modules here to avoid circular imports later
import os
import sys
import time
import timeit
import traceback as tb
from pathlib import Path
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt


sys.path.append(os.path.expanduser("~/.local/lib/python3.11/site-packages"))
from hmg import HBV001A  #impirt the model

#---------------------------------------------------------------------------------
# Optimal parameters from DE optimization
#---------------------------------------------------------------------------------

OptimalParameterDE = np.array([0.000000,
0.148940,
0.634415,
0.036731,
94.568525,
432.979200,
148.866130,
2.659653,
1.728010,
0.739939,
0.748241,
0.185272,
60.579322,
0.143289,
0.000401,
0.000147,
0.008535,
0.000024])

# Parameter names in model order 
PARAM_NAMES = [
    "snw_dth","snw_att","snw_pmf","snw_amf",
    "sl0_dth","sl0_pwp","sl0_fcy","sl0_bt0",
    "urr_dth","lrr_dth",
    "urr_wsr","urr_ulc","urr_tdh","urr_tdr","urr_ndr","urr_uct",
    "lrr_dre","lrr_lct"
]

#---------------------------------------------------------------------------------
# From Assignment 1 we take the following functions
#---------------------------------------------------------------------------------

#Function to compute NSE
def calc_nse(obs: np.ndarray, sim: np.ndarray) -> float:
    """Compute NSE."""
    obs = np.asarray(obs, float)
    sim = np.asarray(sim, float)
    mask = np.isfinite(obs) & np.isfinite(sim)
    obs = obs[mask]; sim = sim[mask]
    denom = np.sum((obs - obs.mean()) ** 2)
    if denom <= 0:
        return float("-inf")
    return 1.0 - np.sum((obs - sim) ** 2) / denom

# Initialize the model
def build_model(tems, ppts, pets, tsps, dslr) -> HBV001A:
    """Construct and initialize HBV001A for a run."""
    m = HBV001A()
    m.set_inputs(tems, ppts, pets) # set the inputs from the time series
    m.set_outputs(tsps)
    m.set_discharge_scaler(dslr)
    m.set_optimization_flag(0)  # why: we optimize externally
    return m

# Here we define the objective function for the DE optimization
def objective_function(params, model,diso):
    """Clamp params, run model, return (ofv=1-NSE, {'NSE': nse}, sim)."""
    p = np.asarray(params, float).copy()
    try:
        model.set_parameters(p) # set the model parameters to the initialized model
    except AssertionError:
        return 1e6, {"NSE": float("-inf")}, None  
    model.run_model()
    sim = model.get_discharge() # simulate the discharge
    nse = calc_nse(diso, sim) # calculate the NSE between observed and simulated discharge
    ofv = 1.0 - nse if np.isfinite(nse) else 1e6
    return ofv, {"NSE": float(nse)}, sim

#---------------------------------------------------------------------------------
# Before starting the loop, we need to initialize the model
#---------------------------------------------------------------------------------

data_dir = Path(r'C:\Users\adrie\Documents\Python\MMUQ Hydrology\mmuq-hydrology-assignments\Assignment 2\data') 
#data_dir = Path.home() / ... / ...
if not data_dir.exists():
    raise FileNotFoundError(f"Data folder not found: {data_dir}")
os.chdir(data_dir)
print(f"Using data from: {data_dir}")
# read time series and area files
df = pd.read_csv("time_series___24163005.csv", sep=";", index_col=0)
df.index = pd.to_datetime(df.index, format="%Y-%m-%d-%H")
cca = float(pd.read_csv("area___24163005.csv", sep=";", index_col=0).values[0, 0])

required_cols = ["tavg__ref", "pptn__ref", "petn__ref", "diso__ref"]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise KeyError(f"Missing required columns: {missing}")
# set the parameters
tems = df["tavg__ref"].values
ppts = df["pptn__ref"].values
pets = df["petn__ref"].values
diso = df["diso__ref"].values

tsps = len(tems)
dslr = cca / (3600 * 1000)
# Baseline
m0 = build_model(tems, ppts, pets, tsps, dslr)

#---------------------------------------------------------------------------------
# Sensitivity Analysis Loop
#---------------------------------------------------------------------------------
# Container initialization
results = {}
spaces = {}
# Now start the foor loop to run over each parameter set
for j in range(len(PARAM_NAMES)):
    index = j
    name = PARAM_NAMES[j]
    OptimalValue = OptimalParameterDE[j]
    # No relative but absolute change for TT
    if name != "snw_att":
       lower = OptimalValue * 0.8
       upper = OptimalValue * 1.2
    else: # absoulte change for TT
       lower = OptimalValue - 1
       upper = OptimalValue + 1

    # Loop over the relative changes
    parameterspace = np.linspace(lower, upper, num=100, endpoint=True)
    spaces[name] = parameterspace

    ofv_values = []  # Das wird nicht klappen
    for i in range(len(parameterspace)):
        parameter_value = parameterspace[i]
        # Set the parameter value in the model
        params = OptimalParameterDE.copy()
        params[index] = parameter_value
        # Evaluate the objective function
        ofv, metrics, sim = objective_function(params, m0, diso)
        ofv_values.append(ofv)
    
    results[name] = ofv_values

    
for name in PARAM_NAMES:
    plt.figure(figsize=(10, 6))
    plt.plot(spaces[name], results[name], label=f"Sensitivity of NSE to {name}")
    plt.vlines(OptimalParameterDE[PARAM_NAMES.index(name)], ymin=min(results[name]),
               ymax=max(results[name]), color="red", linestyle="--", label="Optimal Value")
    plt.xlabel(f"Parameter Value: {name}")
    plt.ylabel("Objective Function (1 - NSE)")
    plt.title(f"Sensitivity Analysis for {name}")
    plt.grid()
    plt.legend()
    plt.savefig(f"sensitivity_analysis_{name}.png")
    plt.show()

    
