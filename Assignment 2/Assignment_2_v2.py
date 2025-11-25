# We import all posisble relevant modules here to avoid circular imports later
import os
import sys
import time
import timeit
import traceback as tb
from pathlib import Path
import numpy as np
import pandas as pd
import tqdm
import math
import matplotlib.pyplot as plt


sys.path.append(os.path.expanduser("~/.local/lib/python3.11/site-packages"))
from hmg import HBV001A  #impirt the model

#---------------------------------------------------------------------------------
# Optimal parameters from DE optimization
#---------------------------------------------------------------------------------

# OptimalParameterDE = np.array([0.000000, #snw_dth
# 0.148940, #snw_att
# 0.634415, #snw_pmf
# 0.036731, #snw_amf
# 94.568525,#sl0_dth
# 432.979200, #sl0_pwp
# 148.866130, #sl0_fcy
# 2.659653, #sl0_bt0
# 1.728010, #urr_dth
# 0.739939, #lrr_dth
# 0.748241, #urr_wsr
# 0.185272, #urr_ulc
# 60.579322, #urr_tdh
# 0.143289, #urr_tdr
# 0.000401, #urr_ndr
# 0.000147, #urr_uct
# 0.008535, #lrr_dre
# 0.000024]) #lrr_lct
# Run where we applied everything once
# OptimalParameterDE = np.array([0.000000,
# -0.006516,
# 0.501091,
# 0.047384,
# 56.524960,
# 542.773370,
# 119.391360,
# 2.178531,
# 1.938387,
# 0.718311,
# 0.867841,
# 0.175833,
# 59.563114,
# 0.054822,
# 0.000035,
# 0.000165,
# 0.009305,
# 0.000003
# ])
#Run where we plugged as param0 the previous parameter
OptimalParameterDE = np.array([0.000000,
-0.005571,
0.521145,
0.046251,
99.167578,
576.309080,
164.676837,
2.953878,
2.363494,
0.304237,
0.996199,
0.176932,
30.970124,
0.182238,
0.000003, # urr_ndr
0.000038, #urr_uct
0.009186,
0.000000])
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
min_value = [] # init. min and max container for ranking later on
max_value = []
diff = []

# Define the bounds to check for
BOUNDS = [
    (0.00, 0.00),   # snw_dth  (fixed 0)
    (-2.0, 3.0),    # snw_att
    (0.00, 3.00),   # snw_pmf
    (0.00, 10.0),   # snw_amf

    (0.00, 100.0),  # sl0_dth
    (5.00, 700.0),  # sl0_pwp
    (100.0, 700.0), # sl0_fcy
    (0.01, 10.0),   # sl0_bt0

    (0.00, 20.0),   # urr_dth
    (0.00, 100.0),  # lrr_dth

    (0.00, 1.00),   # urr_wsr
    (0.00, 1.00),   # urr_ulc
    (0.00, 200.0),  # urr_tdh
    (0.01, 1.00),   # urr_tdr
    (0.00, 1.00),   # urr_ndr
    (0.00, 1.00),   # urr_uct

    (0.00, 1.00),   # lrr_dre
    (0.00, 1.00),   # lrr_lct
]

# Now start the foor loop to run over each parameter set
for j in tqdm.tqdm(range(len(PARAM_NAMES))):
    index = j
    name = PARAM_NAMES[j]
    OptimalValue = OptimalParameterDE[j]
    # No relative but absolute change for TT
    if name != "snw_att":
       lower = OptimalValue * 0.7
       upper = OptimalValue * 1.3
    else: # absoulte change for TT
       lower = OptimalValue - 1
       upper = OptimalValue + 1

    # Check if bounds are violated
    lowerbound, upperbound = BOUNDS[index]

    if upper > upperbound:
       upper = upperbound

    if lower < lowerbound:
       lower = lowerbound

    # Loop over the relative changes
    parameterspace = np.linspace(lower, upper, num=200, endpoint=True)
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
    mini = round(np.min(ofv_values),4)
    min_value.append(mini)
    maxi = round(np.max(ofv_values),4)
    max_value.append(maxi)
    diff.append(round(np.abs(np.max(ofv_values)- np.min(ofv_values)),4))
    
# Table outcome
fig, ax = plt.subplots()
fig.patch.set_visible(False)
ax.axis('off')
ax.axis('tight')

df_results = pd.DataFrame({
    'Parameter': PARAM_NAMES,
    'Min': min_value,
    'Max': max_value,
    'Diff': diff
})
table = ax.table(cellText=df_results.values, colLabels=df_results.columns, loc='center')
table.auto_set_font_size(False)
table.set_fontsize(12)

fig.tight_layout()
plt.savefig("Table_sensitivity_analysis.png")
plt.show()

# Now the barplot with ranks
fig_bar, ax_bar = plt.subplots(figsize=(15, 8))

# Calculate ranks (highest diff gets rank 1)
ranks = pd.Series(diff).rank(ascending=False, method='min').astype(int)

# Create the bar plot
bars = ax_bar.bar(PARAM_NAMES, diff)

# Add rank numbers on top of each bar
for bar, rank in zip(bars, ranks):
    yval = bar.get_height()
    ax_bar.text(bar.get_x() + bar.get_width()/2.0, yval + 0.001, rank, ha='center', va='bottom')

ax_bar.set_ylabel("Maximal absolute diff")
ax_bar.set_xlabel("Parameters")
ax_bar.set_title("Absolute Difference in OFV per Parameter")
ax_bar.grid(axis='y')
#plt.xticks(rotation=45, ha='right') # Rotate labels for better readability
fig_bar.tight_layout()
plt.savefig("absolute_diff.png")
plt.show()


for name in PARAM_NAMES:
    plt.figure(figsize=(10, 6))
    plt.scatter(spaces[name], results[name], label=f"Sensitivity of log(OFV) to {name}")
    plt.vlines(OptimalParameterDE[PARAM_NAMES.index(name)], ymin=min(results[name]),
            ymax=max(results[name]), color="red", linestyle="--", label="Optimal Value")
    plt.yscale("log")
    plt.xlabel(f"Parameter Value: {name}")
    plt.ylabel(" Objective Function (1 - NSE) on log scale")
    plt.title(f"Sensitivity Analysis for {name}")
    plt.grid()
    plt.legend()
    plt.savefig(f"sensitivity_analysis_{name}.png")
    #plt.show()
