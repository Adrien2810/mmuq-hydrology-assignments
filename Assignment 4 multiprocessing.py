import os
import sys
import time
import timeit
import traceback as tb
from pathlib import Path
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution, minimize
from multiprocessing import Pool

sys.path.append(os.path.expanduser("~/.local/lib/python3.11/site-packages"))
from hmg import HBV001A #import the model

#----------------------------------------------------------------------------------------------------------------------
# Define functions to calibrate Using Assignment 1
#----------------------------------------------------------------------------------------------------------------------

PARAM_NAMES = [
    "snw_dth","snw_att","snw_pmf","snw_amf",
    "sl0_dth","sl0_pwp","sl0_fcy","sl0_bt0",
    "urr_dth","lrr_dth",
    "urr_wsr","urr_ulc","urr_tdh","urr_tdr","urr_ndr","urr_uct",
    "lrr_dre","lrr_lct"
]
# Bounds for DE
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

def simple_ofv(params, model, diso):
    """Return only the objective function value for local search."""
    ofv, _, _ = objective_function(params, model, diso)
    return ofv

def plot_obs_sim(index, obs, sim, title, out_png):
    # index stands for the parameters
    fig = plt.figure(figsize=(6, 3), dpi=120)
    plt.plot(index, obs, label="OBS", alpha=0.85)
    plt.plot(index, sim, label="SIM", alpha=0.85)
    plt.grid(True); plt.legend()
    plt.xticks(rotation=45)
    plt.xlabel("Time [hr]"); plt.ylabel("Discharge [m³/s]")
    plt.title(title)
    plt.tight_layout(); plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.show(); plt.close(fig)

def CalibrateModel(precipitation, suffix=""):
    max_minutes = 5.0   # stop long DE runs by wall clock
    max_evals   = 60000  # cap total objective calls

    data_dir = Path(__file__).parent / 'data' 
    #data_dir = Path.home() / ... / ...
    if not data_dir.exists():
        raise FileNotFoundError(f"Data folder not found: {data_dir}")
    # os.chdir(data_dir)
    print(f"Using data from: {data_dir}")
    # read time series and area files
    df = pd.read_csv(data_dir / "time_series___24163005.csv", sep=";", index_col=0)
    df.index = pd.to_datetime(df.index, format="%Y-%m-%d-%H")
    cca = float(pd.read_csv(data_dir / "area___24163005.csv", sep=";", index_col=0).values[0, 0])

    required_cols = ["tavg__ref", "pptn__ref", "petn__ref", "diso__ref"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")
    # set the parameters
    tems = df["tavg__ref"].values
    ppts = precipitation
    pets = df["petn__ref"].values
    diso = df["diso__ref"].values

    tsps = len(tems)
    dslr = cca / (3600 * 1000)

    prms0 = np.array([
         0.00, 0.10, 0.01, 0.10,
         0.00, 300., 70.,  2.50,
         0.00, 0.00,
         1.00, 0.01, 30., 0.10, 0.10, 0.01,
         1e-3, 1e-5
     ], dtype=float)

    # Baseline
    m0 = build_model(tems, ppts, pets, tsps, dslr)
    m0.set_parameters(prms0)
    m0.run_model()
    sim0 = m0.get_discharge()

    # Logging containers 
    eval_params, eval_ofv = [], []
    eval_log = []                 # every evaluation row
    best_by_gen = []
    best_params_by_gen = []
    best_nse_by_gen = []

    # Best-so-far snapshot
    best_solution = {"params": None, "ofv": np.inf, "metrics": None}

    #  DE settings 
    popsize = 7
    n_params = len(BOUNDS)
    pop_n = popsize * n_params         # approx trials per generation
    eval_idx = 0
    start_time = time.time()

    def recorded_ofv(x):
        nonlocal eval_idx
        ofv, metrics, _ = objective_function(x, m0, diso)
        eval_params.append(np.array(x, float))
        eval_ofv.append(float(ofv))
        # use popsize (evaluations per generation) instead of popsize * n_params
        generation = eval_idx // popsize
        row = {
            "eval_idx": eval_idx,
            "generation": generation,
            "ofv": float(ofv),
            "nse": float(metrics["NSE"]),
        }
        row.update({name: float(val) for name, val in zip(PARAM_NAMES, x)})
        eval_log.append(row)
        eval_idx += 1
        return ofv

    def de_callback(xk, convergence):
        # Stop on time or eval budget 
        elapsed_min = (time.time() - start_time) / 60.0
        if elapsed_min >= max_minutes or eval_idx >= max_evals:
           print(f"\n Stopping early: elapsed={elapsed_min:.2f} min, evals={eval_idx}")
           return True  # stop DE

         # Use last recorded evaluation for current best (avoid re-eval)
        if eval_ofv:
            ofv = eval_ofv[-1]
            best_params_by_gen.append(np.array(xk, float))
            best_by_gen.append(ofv)
            # try to recover NSE from eval_log last row
            best_nse_by_gen.append(eval_log[-1].get("nse", float("-inf")))
        else:
            # fallback: do a single evaluation (rare at start)
            ofv, metrics, _ = objective_function(xk, m0, diso)
            best_by_gen.append(ofv)
            best_params_by_gen.append(np.array(xk, float))
            best_nse_by_gen.append(float(metrics["NSE"]))

        if ofv < best_solution["ofv"]:
            best_solution["params"] = np.array(xk, float)
            best_solution["ofv"] = ofv
            # metrics may be unavailable here if we didn't re-eval; keep None-safe
            best_solution["metrics"] = {"NSE": best_nse_by_gen[-1]}
            #print(f"\n New best! Gen {len(best_by_gen)}: OFV={ofv:.4f}, NSE={best_solution['metrics']['NSE']:.4f}")

        #print(f"Gen {len(best_by_gen):3d} | best (1-NSE) = {ofv:.6f}")
        return False

    res = differential_evolution(
        recorded_ofv,
        bounds=BOUNDS,
        strategy="best1bin", # A good default strategy
        maxiter=2500,                 # Increased number of generations
        popsize=popsize,
        tol=1e-3,
        mutation=(0.5, 1.3), # Increased upper bound for mutation to encourage more exploration
        recombination=0.8, # Increased to promote more exploration
        seed=42,
        callback=de_callback,
        #polish=True,
        updating="deferred",
        #workers=-1,
        atol = 1e-3,
        polish = False
    )

    # Use the result from DE as the starting point for a local search
    de_best_params = res.x
    
    # Run scipy.minimize to polish the result
    local_res = minimize(
        fun=simple_ofv,
        x0=de_best_params,
        args=(m0, diso),
        method='L-BFGS-B',
        bounds=BOUNDS,
        options={'ftol': 1e-9, 'gtol': 1e-7} # Stricter tolerance for refinement
    )

    # Compare the results from DE and the local search, and pick the best one
    best_params = local_res.x if local_res.fun < res.fun else de_best_params
    best_ofv, best_metrics, best_sim = objective_function(best_params, m0, diso)
    print("\nOverall optimization complete. Differnetial evolution and gradient based optimizer ran successfully. Best results:")

    # Prefer callback best; fallback to res.x
    save_params = best_solution["params"] if best_solution["params"] is not None else best_params
    save_ofv = best_solution["ofv"] if np.isfinite(best_solution["ofv"]) else best_ofv
    save_metrics = best_solution["metrics"] if best_solution["metrics"] is not None else best_metrics

    # Save best
    np.savetxt(f"best_params{suffix}.txt", save_params, fmt="%.6f")
    with open(f"best_metrics{suffix}.txt", "w") as f:
        f.write(f"Best OFV (1-NSE): {save_ofv:.6f}\n")
        f.write(f"NSE: {save_metrics['NSE']:.4f}\n")

    # Save ALL evaluations table
    pd.DataFrame(eval_log).to_csv(f"de_eval_log{suffix}.csv", index=False)

    # Save per-generation current-best table
    if best_params_by_gen:
        arr = np.vstack(best_params_by_gen)
        df_best = pd.DataFrame(arr, columns=PARAM_NAMES)
        df_best.insert(0, "generation", np.arange(len(best_params_by_gen)))
        df_best["best_ofv"] = best_by_gen
        df_best["best_nse"] = best_nse_by_gen
        df_best.to_csv(f"de_best_by_gen{suffix}.csv", index=False)

    print(f"Best objective (1 - NSE): {best_ofv:.6f}")
    print("Best-fit NSE:", round(best_metrics["NSE"], 4))

    return save_params, save_ofv, save_metrics, best_sim

#----------------------------------------------------------------------------------------------------------------------
# Now we perturb precipitation and let the model run 
#----------------------------------------------------------------------------------------------------------------------

Nsim = 20 #number of simulations
# Initialize  
CalibratedParameters_REF = np.zeros((Nsim, len(PARAM_NAMES))) #to store best parametres for each calibration run
OFVs_REF = []  # to store objective function values
SimulationResults_REF = []  # to store simulation results for boxplots etc.
CalibratedParameters = np.zeros((Nsim, len(PARAM_NAMES))) #to store best parametres for each calibration run
OFVs = []  # to store objective function values
SimulationResults = []  # to store simulation results for boxplots etc.
precipitation_perturbed_all = []
data_dir = Path(__file__).parent / 'data'
if not data_dir.exists():
    raise FileNotFoundError(f"Data folder not found: {data_dir}")

df = pd.read_csv(data_dir / "time_series___24163005.csv", sep=";", index_col=0)
pptn_ref = df["pptn__ref"].values

# Worker function for parallel calibration
def calibrate_worker(i):
    print(f"\n--- Calibration run {i+1} of {Nsim} ---")
    # Perturb precipitation with normal noise
    pptn = pptn_ref
    noise = np.random.normal(1, 0.05, size=pptn.shape)
    pptn_perturbed = np.maximum(noise * pptn, 0.0)  # ensure no negative precipitation
    
    # Calibrate model with perturbed precipitation
    best_params, best_ofv, best_metrics, best_sim = CalibrateModel(pptn_perturbed, f"_perturbed_{i}")
    best_params_REF, best_ofv_REF, best_metrics_REF, best_sim_REF = CalibrateModel(pptn_ref, f"_ref_{i}")
    
    return {
        'i': i,
        'pptn_perturbed': pptn_perturbed,
        'perturbed': {'params': best_params, 'ofv': best_ofv, 'metrics': best_metrics, 'sim': best_sim},
        'reference': {'params': best_params_REF, 'ofv': best_ofv_REF, 'metrics': best_metrics_REF, 'sim': best_sim_REF}
    }

#==========================================================================
# Queuing and multiprocessing.
#==========================================================================

mprg_pool_size = min(Nsim, 16)  # Adjust based on CPU cores

if mprg_pool_size == 1:
    mprg_pool = Pool(1)
else:
    mprg_pool = Pool(mprg_pool_size)

# Run in parallel
results_async = mprg_pool.map_async(calibrate_worker, range(Nsim))
results = results_async.get()  # Wait for all to complete

# Store results
for res in results:
    i = res['i']
    precipitation_perturbed_all.append(res['pptn_perturbed'])
    CalibratedParameters[i, :] = res['perturbed']['params']
    OFVs.append(res['perturbed']['ofv'])
    SimulationResults.append(res['perturbed']['sim'])

    CalibratedParameters_REF[i, :] = res['reference']['params']
    OFVs_REF.append(res['reference']['ofv'])
    SimulationResults_REF.append(res['reference']['sim'])

mprg_pool.close()
mprg_pool.join()
print(f"\n--- Calibration run {i+1} of {Nsim} succesfully completed ---")

Best_Parameter_Assignment1 = np.array([
    0.000000,
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
    0.000003,
    0.000038,
    0.009186,
    0.000000
])
print("\n Now we proceed to plotting the results")
# Now that we ran and calibrated everything, we go ahead and plot several results to interpret
# CDF plots of OFVs
def plot_cdf_ofvs(ofvs_ref, ofvs_perturbed, out_png):
    fig = plt.figure(figsize=(6, 4), dpi=120)
    sorted_ref = np.sort(ofvs_ref) # sort the ofvs which you round to then count the individual values
    sorted_perturbed = np.sort(ofvs_perturbed)
    p_ref = np.arange(1, len(sorted_ref) + 1) / len(sorted_ref)  # Plus 1 as python counts from zero. Now count cumulative probabilities (obs i / total obs)
    p_perturbed = np.arange(1, len(sorted_perturbed) + 1) / len(sorted_perturbed)
    plt.plot(sorted_ref, p_ref, label="Reference Precipitation", marker='o', linestyle='-', markersize=4)
    plt.plot(sorted_perturbed, p_perturbed, label="Perturbed Precipitation", marker='s', linestyle='--', markersize=4)
    plt.xlabel("Objective Function Value (1 - NSE)")
    plt.ylabel("Cumulative Probability")
    plt.title("CDF of Objective Function Values")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.show()
    plt.close(fig)

plot_cdf_ofvs(OFVs_REF, OFVs, "cdf_ofvs.png")

# CDF's of the parameters
def plot_cdf_params(params_ref, params_perturbed, out_png):
    fig, axs = plt.subplots(len(PARAM_NAMES), 1, figsize=(6, 3 * len(PARAM_NAMES)), dpi=120)
    for i, name in enumerate(PARAM_NAMES):
        sorted_ref = np.sort(params_ref[:, i])
        sorted_pertubed = np.sort(params_perturbed[:, i])
        p_ref = np.arange(1, len(sorted_ref) + 1) / len(sorted_ref)
        p_perturbed = np.arange(1, len(sorted_pertubed) + 1) / len(sorted_pertubed)
        axs[i].plot(sorted_ref, p_ref, label="Reference Precipitation", marker='o', linestyle='-', markersize=4)
        axs[i].plot(sorted_pertubed, p_perturbed, label="Perturbed Precipitation", marker='s', linestyle='--', markersize=4)
        axs[i].set_xlabel(f"Parameter: {name}")
        axs[i].set_ylabel("Cumulative Probability")
        axs[i].set_title(f"CDF of {name}")
        axs[i].grid(True)
        axs[i].legend() 
    plt.tight_layout()
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.show()
    plt.close(fig)

plot_cdf_params(CalibratedParameters_REF, CalibratedParameters, "cdf_parameters.png")

# Scatterplots of OFVS vs perturbed inputs
def plot_scatter_ofvs_precipitation(precip_ref, precip_perturbed, ofvs_perturbed, out_png):
    fig = plt.figure(figsize=(6, 4), dpi=120)
    avg_perturbed = np.mean(precip_perturbed, axis=1)
    plt.scatter(avg_perturbed, ofvs_perturbed, alpha=0.7)
    plt.xlabel("Average Perturbed Precipitation")
    plt.ylabel("Objective Function Value (1 - NSE)")
    plt.title("Scatterplot of OFV vs Average Perturbed Precipitation")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.show()
    plt.close(fig)
plot_scatter_ofvs_precipitation(pptn, precipitation_perturbed_all, OFVs, "scatter_ofvs_precipitation.png")

# Boxplots of the simulation results 
def boxplot_simulation_results(sim_results_ref, sim_results_perturbed, out_png):
    fig = plt.figure(figsize=(10, 6), dpi=120)
    data = [sim_results_ref[i] for i in range(len(sim_results_ref))] + [sim_results_perturbed[i] for i in range(len(sim_results_perturbed))]
    labels = [f"Ref {i+1}" for i in range(len(sim_results_ref))] + [f"Pert {i+1}" for i in range(len(sim_results_perturbed))]
    plt.boxplot(data, labels=labels, showfliers=False)
    plt.xticks(rotation=90)
    plt.xlabel("Simulation Runs")
    plt.ylabel("Discharge [m³/s]")
    plt.title("Boxplot of Simulation Results")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.show()
    plt.close(fig)

boxplot_simulation_results(SimulationResults_REF, SimulationResults, "boxplot_simulation_results.png")