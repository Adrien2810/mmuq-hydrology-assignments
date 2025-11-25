# File: mse_optimization.py

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
from scipy.optimize import differential_evolution, minimize # Import both optimizers
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

#---------------------------------------------------------------------------------
# Model functions
#---------------------------------------------------------------------------------

# Function to compute Mean Squared Error (MSE)
def calc_mse(obs: np.ndarray, sim: np.ndarray) -> float:
    """Compute Mean Squared Error (MSE)."""
    obs = np.asarray(obs, float)
    sim = np.asarray(sim, float)
    mask = np.isfinite(obs) & np.isfinite(sim)
    obs = obs[mask]; sim = sim[mask]
    if len(obs) == 0: # Handle cases where no valid data points exist
        return float("inf") # Return a very high MSE
    return np.mean((obs - sim) ** 2)

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
def objective_function(params, model, diso):
    """Run model, return (ofv=MSE, {'MSE': mse}, sim)."""
    p = np.asarray(params, float).copy()
    try:
        model.set_parameters(p) # set the model parameters to the initialized model
    except AssertionError:
        return 1e6, {"MSE": float("inf")}, None  # Return a very high MSE for invalid parameters
    model.run_model()
    sim = model.get_discharge() # simulate the discharge
    mse = calc_mse(diso, sim) # calculate the MSE between observed and simulated discharge
    ofv = mse # For MSE, we directly minimize the value
    return ofv, {"MSE": float(mse)}, sim

# A simpler objective function for use with scipy.minimize
def simple_ofv(params, model, diso):
    """Return only the objective function value for local search."""
    ofv, _, _ = objective_function(params, model, diso)
    return ofv

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
    plt.yscale("log")
    ax.set_xlabel("Number of Generation"); ax.set_ylabel("Objective (MSE) on log scale") # Changed label
    ax.set_title("DE Convergence (MSE)"); ax.grid(True) # Changed title
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
        ax.set_ylabel('Objective function value (MSE) [-]') # Adjusted label for clarity
        
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
        out_png = os.path.join(out_dir, f"param_path_{name.lower()}.png")
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
# This we use to model with processes turned off
def run_with_params(params,m, diso):
    m.set_parameters(np.array(params, float))
    m.run_model()
    otps = m.get_outputs()
    sim  = m.get_discharge()
    labels = m.get_output_labels()
    mse = calc_mse(diso, sim) # Changed to calc_mse
    return otps, sim, labels, {"MSE": float(mse)} # Changed key to MSE

# Turn off snow module
# def turn_off_snow(p):
#     q = np.array(p, float)
#     q[0]=0.0; q[1]=0.0; q[2]=0.0; q[3]=0.0
#     return q

""" def turn_off_upper_reservoir(p):
    q = np.array(p, float)
    q[8]=0.0; q[10]=0.0; q[11]=0.0; q[12]=0.0; q[13]=1.0; q[14]=0.0; q[15]=0.0
    return q

def turn_off_lower_reservoir(p):
    q = np.array(p, float)
    q[9]=0.0; q[16]=0.0; q[17]=0.0
    return q """

def main():
    
    max_minutes = 40.0   # stop long DE runs by wall clock
    max_evals   = 60000  # cap total objective calls

    # Define base path for the assignment
    assignment_dir = Path(r'C:\Users\adrie\Documents\Python\MMUQ Hydrology\mmuq-hydrology-assignments\Assignment 1')
    data_dir = assignment_dir / 'data'
    output_dir = assignment_dir / 'mse'
    output_dir.mkdir(parents=True, exist_ok=True) # Create the output directory if it doesn't exist

    if not data_dir.exists():
        raise FileNotFoundError(f"Data folder not found: {data_dir}")
    print(f"Using data from: {data_dir}\nSaving output to: {output_dir}")
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
    ppts = df["pptn__ref"].values
    pets = df["petn__ref"].values
    diso = df["diso__ref"].values

    tsps = len(tems)
    dslr = cca / (3600 * 1000)

    prms0 = np.array([
        0.000000,
        -0.006516,
        0.501091,
        0.047384,
        56.524960,
        542.773370,
        119.391360,
        2.178531,
        1.938387,
        0.718311,
        0.867841,
        0.175833,
        59.563114,
        0.054822,
        0.000035,
        0.000165,
        0.009305,
        0.000003])

    # Baseline
    m0 = build_model(tems, ppts, pets, tsps, dslr)
    m0.set_parameters(prms0)
    m0.run_model()
    sim0 = m0.get_discharge()
    base_mse = calc_mse(diso, sim0) # Changed to calc_mse
    print("Baseline MSE:", round(base_mse, 4)) # Changed print
    plot_obs_sim(df.index, diso, sim0, "Observed vs Simulated (Baseline)", output_dir / "baseline_run.png")

    print("\nStarting Differential Evolution calibration...\n")

    # Logging containers 
    eval_params, eval_ofv = [], []
    eval_log = []                 # every evaluation row
    best_by_gen = []
    best_params_by_gen = []
    best_mse_by_gen = [] # Changed to best_mse_by_gen

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
            "mse": float(metrics["MSE"]), # Changed to mse
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
            # try to recover MSE from eval_log last row
            best_mse_by_gen.append(eval_log[-1].get("mse", float("inf"))) # Changed to mse
        else:
            # fallback: do a single evaluation (rare at start)
            ofv, metrics, _ = objective_function(xk, m0, diso)
            best_by_gen.append(ofv)
            best_params_by_gen.append(np.array(xk, float))
            best_mse_by_gen.append(float(metrics["MSE"])) # Changed to MSE

        if ofv < best_solution["ofv"]:
            best_solution["params"] = np.array(xk, float)
            best_solution["ofv"] = ofv
            # metrics may be unavailable here if we didn't re-eval; keep None-safe
            best_solution["metrics"] = {"MSE": best_mse_by_gen[-1]} # Changed to MSE
            print(f"\n New best! Gen {len(best_by_gen)}: OFV={ofv:.4f}, MSE={best_solution['metrics']['MSE']:.4f}") # Changed print

        print(f"Gen {len(best_by_gen):3d} | best MSE = {ofv:.6f}") # Changed print
        return False

    res = differential_evolution(
        recorded_ofv,
        bounds=BOUNDS,
        strategy="best1bin", # A good default strategy
        maxiter=2500,                 # Increased number of generations
        popsize=popsize,
        tol=1e-6,
        mutation=(0.5, 1.3), # Increased upper bound for mutation to encourage more exploration
        recombination=0.8, # Increased to promote more exploration
        seed=42,
        callback=de_callback,
        #polish=True,
        updating="deferred",
        #workers=-1,
        atol = 1e-6,
        polish = False
    )

    print("\nDifferential Evolution finished. Now refining with local search (minimize)...")

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

    print("Local search complete.")

    # Compare the results from DE and the local search, and pick the best one
    best_params = local_res.x if local_res.fun < res.fun else de_best_params
    best_ofv, best_metrics, best_sim = objective_function(best_params, m0, diso)
    print("\nOverall optimization complete.")

    # Prefer callback best; fallback to res.x
    save_params = best_solution["params"] if best_solution["params"] is not None else best_params
    save_ofv = best_solution["ofv"] if np.isfinite(best_solution["ofv"]) else best_ofv
    save_metrics = best_solution["metrics"] if best_solution["metrics"] is not None else best_metrics

    # Save best
    np.savetxt(output_dir / "best_params_mse.txt", save_params, fmt="%.6f") # Changed filename
    with open(output_dir / "best_metrics_mse.txt", "w") as f: # Changed filename
        f.write(f"Best OFV (MSE): {save_ofv:.6f}\n") # Changed print
        f.write(f"MSE: {save_metrics['MSE']:.4f}\n") # Changed print

    # Save ALL evaluations table
    pd.DataFrame(eval_log).to_csv(output_dir / "de_eval_log_mse.csv", index=False) # Changed filename

    # Save per-generation current-best table
    if best_params_by_gen:
        arr = np.vstack(best_params_by_gen)
        df_best = pd.DataFrame(arr, columns=PARAM_NAMES)
        df_best.insert(0, "generation", np.arange(len(best_params_by_gen)))
        df_best["best_ofv"] = best_by_gen
        df_best["best_mse"] = best_mse_by_gen # Changed column name
        df_best.to_csv(output_dir / "de_best_by_gen_mse.csv", index=False) # Changed filename

    print("\nSaved:")
    print(" - best_params_mse.txt")
    print(" - best_metrics_mse.txt")
    print(" - de_eval_log_mse.csv            (all evaluations)")
    print(" - de_best_by_gen_mse.csv         (current best per generation)")

    print(f"Best objective (MSE): {best_ofv:.6f}") # Changed print
    print("Best-fit MSE:", round(best_metrics["MSE"], 4)) # Changed print

    # Plots
    plot_obs_sim(df.index, diso, best_sim, "Observed vs Simulated (Optimized - MSE)", output_dir / "optimized_run_mse.png") # Changed title and filename
    plot_convergence(best_by_gen, out_png=output_dir / "de_convergence_mse.png") # Changed filename
    plot_param_paths(best_params_by_gen, out_dir=output_dir, prefix="param_path_mse_") # Changed prefix
    plot_param_scatter(eval_params, eval_ofv, PARAM_NAMES, BOUNDS, out_dir=output_dir)
    # Internals + ablations
    print("\n--- Internal variables with processes ON vs OFF ---")
    otps_on, sim_on, labels, met_on = run_with_params(best_params, m0, diso)
    print(f"All ON  | MSE = {met_on['MSE']:.3f}") # Changed print and key
    plot_internal_vars(df.index, df, otps_on, labels,
                       "Internal Variables — All Processes ON (MSE)", output_dir / "internal_all_on_mse.png") # Changed title and filename
    """"
    p_no_snow = turn_off_snow(best_params)
    otps_ns, sim_ns, _, met_ns = run_with_params(p_no_snow, m0, diso)
    print(f"Snow OFF| MSE = {met_ns['MSE']:.3f}")
    plot_internal_vars(df.index, df, otps_ns, labels,
                       "Internal Variables — Snow OFF (MSE)", "internal_snow_off_mse.png")

    p_no_lrr = turn_off_lower_reservoir(best_params)
    otps_nl, sim_nl, _, met_nl = run_with_params(p_no_lrr, m0, diso)
    print(f"LRR OFF | MSE = {met_nl['MSE']:.3f}")
    plot_internal_vars(df.index, df, otps_nl, labels,
                       "Internal Variables — Lower Reservoir OFF (MSE)", "internal_lrr_off_mse.png")

    p_no_urr = turn_off_upper_reservoir(best_params)
    otps_nu, sim_nu, _, met_nu = run_with_params(p_no_urr, m0, diso)
    print(f"URR OFF | MSE = {met_nu['MSE']:.3f}")
    plot_internal_vars(df.index, df, otps_nu, labels,
                       "Internal Variables — Upper Reservoir OFF (MSE)", "internal_urr_off_mse.png")

    plot_obs_sim(df.index, diso, sim_on, "Hydrograph — All ON (optimized - MSE)", "hydro_all_on_mse.png")
    plot_obs_sim(df.index, diso, sim_ns, "Hydrograph — Snow OFF (MSE)", "hydro_snow_off_mse.png")
    plot_obs_sim(df.index, diso, sim_nl, "Hydrograph — LRR OFF (MSE)", "hydro_lrr_off_mse.png")
    plot_obs_sim(df.index, diso, sim_nu, "Hydrograph — URR OFF (MSE)", "hydro_urr_off_mse.png")
 """
    print("\n All outputs saved in:", output_dir)



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
