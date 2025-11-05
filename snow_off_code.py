# File: hmg/test/aa_run_model.py

#---------------------------------------------------------------------------------
# Import the required packages
#---------------------------------------------------------------------------------
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

from scipy.optimize import differential_evolution #here you extract the differnetial evolution package
from scipy.stats import spearmanr # If we ant to test interdependence of parameters

#---------------------------------------------------------------------------------
# Intialization
#---------------------------------------------------------------------------------

# Parameter names in model order 
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
    (-100.0, -100.00),    # snw_att
    (0.00, 0.00),   # snw_pmf
    (0.00, 0.00),   # snw_amf

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

#---------------------------------------------------------------------------------
# Model functions
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
    for i, (lo, hi) in enumerate(BOUNDS):
        p[i] = np.clip(p[i], lo, hi) # Clamp to bounds
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
# Plotting functions
#---------------------------------------------------------------------------------

# Plot observed vs simulated discharge
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

# Function to plot the convergence of DE of the OFV
def plot_convergence(best_list, out_png="de_convergence.png"):
    if not best_list:
        return
    fig, ax = plt.subplots(figsize=(5, 3), dpi=120)
    ax.plot(best_list, lw=2)
    ax.set_xlabel("Number of Generation"); ax.set_ylabel("Objective (1 - NSE)")
    ax.set_title("DE Convergence"); ax.grid(True)
    fig.tight_layout(); fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.show(); plt.close(fig)

# Function to get the scatter plots of parameters vs OFV (convergence)
def plot_param_scatter(eval_params, eval_ofv, param_names, bounds, out_dir="."):
    eval_params = np.array(eval_params)
    eval_ofv = np.array(eval_ofv)
    n_evals = len(eval_ofv)
    
    # Filter out infinite OFVs for cleaner plots
    finite_mask = np.isfinite(eval_ofv)
    if not np.any(finite_mask):
         print("Warning: No finite objective values for scatter plots.")
         return
         
    eval_ofv_finite = eval_ofv[finite_mask]
    eval_params_finite = eval_params[finite_mask, :]
    
    # loop over each parameter and the bounded range
    for i, (name, (lo, hi)) in enumerate(zip(param_names, bounds)):
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Scatter plot colored black/dark gray to reproduce the density effect
        scatter = ax.scatter(eval_params_finite[:, i], eval_ofv_finite, 
                             color='black',
                             alpha=0.6,
                             s=50) # Increased size for better density
        
        # Enable Logarithmic Y-axis
        ax.set_yscale('log') 
        
        # Format title with bounds
        title = f"Bounds: {lo:.3E} - {hi:.3E}"
        ax.set_title(title)
        
        # Labels and axis limits
        ax.set_xlabel(f'{name} [-]')
        ax.set_ylabel('Objective function value [-]') # Adjusted label for clarity
        
        # Set parameter bounds with a small buffer
        ax.set_xlim(lo - 0.05*(hi-lo), hi + 0.05*(hi-lo))
        
        # Grid and layout
        ax.grid(True, which='both', ls=':')
        plt.tight_layout()
        
        # Save
        out_png = os.path.join(out_dir, f"param_scatter_{name.lower()}.png")
        plt.savefig(out_png, dpi=200, bbox_inches="tight")
        plt.close(fig) # Use plt.close(fig) to manage memory

# Function to plot parameter paths over generations
def plot_param_paths(best_params_by_gen, out_dir=".", prefix="param_path_"):
    """Plot step trajectories of current-best parameter vs generation."""
    if not best_params_by_gen:
        return
    arr = np.vstack(best_params_by_gen)  # (G, P)
    gens = np.arange(arr.shape[0])
    for j, name in enumerate(PARAM_NAMES):
        fig = plt.figure(figsize=(6, 6), dpi=120)
        plt.step(gens, arr[:, j], where="post")
        plt.grid(True)
        plt.xlabel("Generation number [-]")
        plt.ylabel(f"{name.upper()} [-]")  # why: generic units
        plt.title(f"{name.upper()} vs Generation")
        plt.tight_layout()
        out_png = os.path.join(out_dir, f"{prefix}{name.lower()}.png")
        plt.savefig(out_png, dpi=200, bbox_inches="tight")
        plt.show(); plt.close(fig)

# Function to plot internal model variables
def plot_internal_vars(index, df, otps, labels, title, out_png):
    fig, axs = plt.subplots(8, 1, figsize=(6, 9), dpi=120, sharex=True)
    (axs_tem, axs_ppt, axs_snw, axs_sl0, axs_etn, axs_rrr, axs_rnf, axs_bal) = axs

    axs_tem.plot(df["tavg__ref"], alpha=0.85); axs_tem.set_ylabel("TEM\n[°C]")
    axs_ppt.plot(df["pptn__ref"], alpha=0.85); axs_ppt.set_ylabel("PPT\n[mm]")

    axs_snw.plot(index, otps[:, labels["snw_dth"]], alpha=0.85); axs_snw.set_ylabel("SNW\n[mm]")
    axs_sl0.plot(index, otps[:, labels["sl0_dth"]], alpha=0.85); axs_sl0.set_ylabel("SL0\n[mm]")

    axs_etn.plot(index, df["petn__ref"], label="PET", alpha=0.85)
    axs_etn.plot(index, otps[:, labels["sl0_etn"]], label="ETN", alpha=0.85)
    axs_etn.legend(); axs_etn.set_ylabel("ETN\n[mm]")

    axs_rrr.plot(index, otps[:, labels["urr_dth"]], label="URR", alpha=0.85)
    axs_rrr.plot(index, otps[:, labels["lrr_dth"]], label="LRR", alpha=0.85)
    axs_rrr.legend(); axs_rrr.set_ylabel("DTH\n[mm]")

    axs_rnf.plot(index, otps[:, labels["chn_pow"]], label="SFC", alpha=0.85)
    axs_rnf.plot(index, otps[:, labels["urr_urf"]] + otps[:, labels["lrr_lrf"]], label="GND", alpha=0.85)
    axs_rnf.legend(); axs_rnf.set_ylabel("RNF\n[mm]")

    axs_bal.plot(index, otps[:, labels["mod_bal"]], alpha=0.85); axs_bal.set_ylabel("BAL\n[mm]")

    for ax in axs: ax.grid(True)
    axs[-1].set_xlabel("Time [hr]")
    plt.xticks(rotation=45)
    plt.suptitle(title)
    plt.tight_layout(); plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.show(); plt.close(fig)

# Run model with given params and return outputs
# Thiis we use to model with processes turned off
def run_with_params(params,m, diso):
    m.set_parameters(np.array(params, float))
    m.run_model()
    otps = m.get_outputs()
    sim  = m.get_discharge()
    labels = m.get_output_labels()
    nse = calc_nse(diso, sim)
    return otps, sim, labels, {"NSE": float(nse)}

# Turn off snow module
def turn_off_snow(p):
    q = np.array(p, float)
    q[0]=0.0; q[1]=0.0; q[2]=0.0; q[3]=0.0
    return q

""" def turn_off_upper_reservoir(p):
    q = np.array(p, float)
    q[8]=0.0; q[10]=0.0; q[11]=0.0; q[12]=0.0; q[13]=1.0; q[14]=0.0; q[15]=0.0
    return q

def turn_off_lower_reservoir(p):
    q = np.array(p, float)
    q[9]=0.0; q[16]=0.0; q[17]=0.0
    return q 

#turn off soil 
def turn_off_soil(p):
    q = np.array(p,float)
    q[7]=0.0; q[4]=0.000001; q[6]=0.000001; q[5]=1000000
    return q"""

def main():
    
    max_minutes = 30.0   # stop long DE runs by wall clock
    max_evals   = 20000  # cap total objective calls

    data_dir = Path(r'C:\Users\adrie\Documents\Python\MMUQ Hydrology\mmuq-hydrology-assignments\data\snow_off') 
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

    prms0 = np.array([
        0.00, -100.00, 0.00, 0.00,
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
    base_nse = calc_nse(diso, sim0)
    print("Baseline NSE:", round(base_nse, 4))
    plot_obs_sim(df.index, diso, sim0, "Observed vs Simulated (Baseline)", "baseline_run.png")

    print("\nStarting Differential Evolution calibration...\n")

    # Logging containers 
    eval_params, eval_ofv = [], []
    eval_log = []                 # every evaluation row
    best_by_gen = []
    best_params_by_gen = []
    best_nse_by_gen = []

    # Best-so-far snapshot
    best_solution = {"params": None, "ofv": np.inf, "metrics": None}

    #  DE settings 
    popsize = 6
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
            print(f"\n New best! Gen {len(best_by_gen)}: OFV={ofv:.4f}, NSE={best_solution['metrics']['NSE']:.4f}")

        print(f"Gen {len(best_by_gen):3d} | best (1-NSE) = {ofv:.6f}")
        return False

    res = differential_evolution(
        recorded_ofv,
        bounds=BOUNDS,
        strategy="best1bin",
        maxiter=1000,                 # high cap; budget guard stops earlier
        popsize=popsize,
        tol=1e-4,
        mutation=(0.5, 1.0),
        recombination=0.7,
        seed=42,
        callback=de_callback,
        #polish=True,
        updating="deferred",
        #workers=-1,
        atol = 1e-3,
        polish = False
    )

    # Final evaluation at DE-returned best
    best_params = res.x
    best_ofv, best_metrics, best_sim = objective_function(best_params, m0, diso)
    print("\nOptimization complete.")

    # Prefer callback best; fallback to res.x
    save_params = best_solution["params"] if best_solution["params"] is not None else best_params
    save_ofv = best_solution["ofv"] if np.isfinite(best_solution["ofv"]) else best_ofv
    save_metrics = best_solution["metrics"] if best_solution["metrics"] is not None else best_metrics

    # Save best
    np.savetxt("best_params.txt", save_params, fmt="%.6f")
    with open("best_metrics.txt", "w") as f:
        f.write(f"Best OFV (1-NSE): {save_ofv:.6f}\n")
        f.write(f"NSE: {save_metrics['NSE']:.4f}\n")

    # Save ALL evaluations table
    pd.DataFrame(eval_log).to_csv("de_eval_log.csv", index=False)

    # Save per-generation current-best table
    if best_params_by_gen:
        arr = np.vstack(best_params_by_gen)
        df_best = pd.DataFrame(arr, columns=PARAM_NAMES)
        df_best.insert(0, "generation", np.arange(len(best_params_by_gen)))
        df_best["best_ofv"] = best_by_gen
        df_best["best_nse"] = best_nse_by_gen
        df_best.to_csv("de_best_by_gen.csv", index=False)

    print("\nSaved:")
    print(" - best_params.txt")
    print(" - best_metrics.txt")
    print(" - de_eval_log.csv            (all evaluations)")
    print(" - de_best_by_gen.csv         (current best per generation)")

    print(f"Best objective (1 - NSE): {best_ofv:.6f}")
    print("Best-fit NSE:", round(best_metrics["NSE"], 4))

    # Plots
    plot_obs_sim(df.index, diso, best_sim, "Observed vs Simulated (Optimized)", "optimized_run.png")
    plot_convergence(best_by_gen, out_png="de_convergence.png")
    plot_param_paths(best_params_by_gen, out_dir=os.getcwd(), prefix="param_path_")
    plot_param_scatter(eval_params, eval_ofv, PARAM_NAMES, BOUNDS, out_dir=os.getcwd())
    # Internals + ablations
    """print("\n--- Internal variables with processes ON vs OFF ---")
    otps_on, sim_on, labels, met_on = run_with_params(best_params, m0, diso)
    print(f"All ON  | NSE = {met_on['NSE']:.3f}")
    plot_internal_vars(df.index, df, otps_on, labels,
                       "Internal Variables — All Processes ON", "internal_all_on.png")

    p_no_snow = turn_off_snow(best_params)
    otps_ns, sim_ns, _, met_ns = run_with_params(p_no_snow, m0, diso)
    print(f"Snow OFF| NSE = {met_ns['NSE']:.3f}")
    plot_internal_vars(df.index, df, otps_ns, labels,
                       "Internal Variables — Snow OFF", "internal_snow_off.png")

    p_no_lrr = turn_off_lower_reservoir(best_params)
    otps_nl, sim_nl, _, met_nl = run_with_params(p_no_lrr, m0, diso)
    print(f"LRR OFF | NSE = {met_nl['NSE']:.3f}")
    plot_internal_vars(df.index, df, otps_nl, labels,
                       "Internal Variables — Lower Reservoir OFF", "internal_lrr_off.png")

    p_no_urr = turn_off_upper_reservoir(best_params)
    otps_nu, sim_nu, _, met_nu = run_with_params(p_no_urr, m0, diso)
    print(f"URR OFF | NSE = {met_nu['NSE']:.3f}")
    plot_internal_vars(df.index, df, otps_nu, labels,
                       "Internal Variables — Upper Reservoir OFF", "internal_urr_off.png")

    p_no_soil = turn_off_soil(best_params)
    otps_no, sim_no, _, met_no = run_with_params(p_no_soil, m0, diso)
    print(f"URR OFF | NSE = {met_no['NSE']:.3f}")
    plot_internal_vars(df.index, df, otps_no, labels,
                       "Internal Variables — Soil OFF", "internal_soil_off.png")

    plot_obs_sim(df.index, diso, sim_on, "Hydrograph — All ON (optimized)", "hydro_all_on.png")
    plot_obs_sim(df.index, diso, sim_ns, "Hydrograph — Snow OFF", "hydro_snow_off.png")
    plot_obs_sim(df.index, diso, sim_nl, "Hydrograph — LRR OFF", "hydro_lrr_off.png")
    plot_obs_sim(df.index, diso, sim_nu, "Hydrograph — URR OFF", "hydro_urr_off.png")
    plot_obs_sim(df.index, diso, sim_no, "Hydrograph — Soil OFF", "soil_off.png")
 """
    print("\n All outputs saved in:", os.getcwd())



if __name__ == "__main__":
    print("Started on %s \n" % time.asctime())
    START = timeit.default_timer()
    try:
        main()
    except Exception:
        pre_stack = tb.format_stack()[:-1]
        err_tb = list(tb.TracebackException(*sys.exc_info()).format())
        lines = [err_tb[0]] + pre_stack + err_tb[2:]
        for line in lines:
            print(line, file=sys.stderr, end="")
        raise
    STOP = timeit.default_timer()
    print(f"\n Done on {time.asctime()}.\nTotal runtime ≈ {STOP - START:.3f} s")
