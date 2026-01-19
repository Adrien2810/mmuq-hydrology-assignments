import os

# Numpy sneakily uses multiple threads sometimes. I don't want that.
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MPI_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'

import sys
import time
import timeit
import traceback as tb
from pathlib import Path
from multiprocessing import Manager, Pool
from multiprocessing.pool import ThreadPool
from queue import Queue

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.optimize import differential_evolution, minimize

sys.path.append(os.path.expanduser("~/.local/lib/python3.11/site-packages"))
from hmg import HBV001A

def main():
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
    ppts = df["pptn__ref"].values
    pets = df["petn__ref"].values
    diso = df["diso__ref"].values

    tsps = len(tems)
    dslr = cca / (3600 * 1000)
    #--------------------------------------------------------------------------------------
    # Use Nsim instead of SobolSampleSize
    Nsim = 2000
    # Use float32 as HBV001A does so as well and it is less expensive
    dtyp = np.float32
    mprg_pool_size = 8

    #==========================================================================
    #Verify if Nsim is compatible with your amount of workes. Adjust if necessary
    if (Nsim % mprg_pool_size):

        Nsim += (mprg_pool_size - (Nsim % mprg_pool_size))

        assert (Nsim % mprg_pool_size) == 0

        print('Adjusted Nsim to', Nsim)
    modl_objt = HBV001A()
    modl_objt.set_inputs(tems, ppts, pets)
    modl_objt.set_outputs(tsps)

    modl_objt.set_discharge_scaler(dslr)

    #Pass all the arguments as a class (?)
    #==========================================================================

    optn_args = OPTNARGS()

    optn_args.cntr = 0
    optn_args.oflg = 0
    optn_args.vbse = False
    optn_args.diso = diso
    optn_args.ppts = ppts
    optn_args.tems = tems
    optn_args.pets = pets
    optn_args.tsps = tsps
    optn_args.dslr = dslr
    optn_args.Best_Parameter_Assignment1 = Best_Parameter_Assignment1
    optn_args.PARAM_NAMES = PARAM_NAMES
    optn_args.BOUNDS = BOUNDS
    optn_args.modl_objt = modl_objt
    optn_args.take_idxs = np.isfinite(diso)

    modl_objt.set_optimization_flag(1)
    optn_args.Nsim = Nsim
    fast_model = FastHBVModel(str(data_dir))
    optn_args.fast_model = fast_model
    #==========================================================================
    # Copy from Faizan and adapt the function you want to parallize on
    #==========================================================================
    # Multiprocessing.
    #==========================================================================
    bgn_tmr = timeit.default_timer()
    
    # Manager Pool cannot work with a generator!
    mprg_args = [(
        i,
        optn_args,
        )
        for i in range(Nsim)]

    if mprg_pool_size == 1:
        mprg_pool = ThreadPool(1)

    else:
        mprg_pool = Pool(mprg_pool_size)
        wait_for_pool_init(mprg_pool, mprg_pool_size, 'RECALIBRATION', True)

    result = mprg_pool.map_async(multi, mprg_args).get()
    # Initilialize the lists and containers to store the results to proceed further
    CalibratedParameters = []
    OFV = []
    Metrics = []
    SimulationResults = []
    OFV_REF = []
    Sim_REF = []
    pptn_perturbed_all = []
    for j in range(Nsim):
        i, best_params, best_ofv, best_metrics, best_sim, best_ofv_REF, best_sim_REF, pptn_perturbed = result[j]
        CalibratedParameters.append(best_params)
        OFV.append(best_ofv)
        Metrics.append(best_metrics)
        SimulationResults.append(best_sim)
        OFV_REF.append(best_ofv_REF)
        Sim_REF.append(best_sim_REF)
        pptn_perturbed_all.append(pptn_perturbed)   

        # here all the appending can happen

    # Now plot all after collecting results
    plot_cdf_ofvs(OFV_REF, OFV, "cdf_ofvs.png")
    plot_cdf_params(Best_Parameter_Assignment1, np.array(CalibratedParameters), "cdf_parameters.png")
    # For precipitation CDF, perhaps plot reference vs one perturbed, e.g., the first
    if pptn_perturbed_all:
        plot_cdf_precipitation_log(optn_args.ppts.copy(), pptn_perturbed_all[0], "cdf_precipitation.png")
    plot_scatter_ofvs_precipitation(OFV, OFV_REF, "scatter_ofvs.png")
    improvement_count, min_ofv_perturbed = analyze_ofv_improvements(OFV_REF, OFV)
    print(f"\nNumber of improvements by recalibration: {improvement_count} out of {Nsim}")
    boxplot_parameters_results(PARAM_NAMES, CalibratedParameters, Best_Parameter_Assignment1, "boxplot_parameters_results.png")
    np.random.seed(42)  # for reproducibility
    random_indices = np.random.choice(len(SimulationResults), 6, replace=False)
    selected_sims = [SimulationResults[i] for i in random_indices]

    plot_all_sim_obs(df.index, diso, selected_sims, "Observed Discharge and 6 Random Simulated Discharges", "Random_6_Simulations_Discharge.png")
    plot_survival_log_vs_reference(pptn_perturbed_all, optn_args.ppts.copy(), "survival_log_vs_reference.png")

    print(f"Successfully processed {len(result)} simulations.")
    #===========================================================================
     #==========================================================================
    mprg_pool.close()

    end_tmr = timeit.default_timer()

    print(f'Took {end_tmr - bgn_tmr:0.2E} seconds to complete.')
    #==========================================================================


# Arguments are two thing: the idx and opt_args that has everything that we need
def multi(ags):
    i = ags[0]
    optn_args = ags[1]
    Nsim = optn_args.Nsim
    print(f"\n--- Calibration run {i+1} of {Nsim} ---")
    import sys
    sys.stdout.flush()
    # Perturb precipitation with normal noise
    pptn = optn_args.ppts.copy()
    tems = optn_args.tems.copy()
    pets = optn_args.pets.copy()
    tsps = optn_args.tsps
    dslr = optn_args.dslr
    Best_Parameter_Assignment1 = optn_args.Best_Parameter_Assignment1.copy()
    diso = optn_args.diso.copy()
    noise = np.random.normal(1, 0.05, size=pptn.shape)
    pptn_perturbed = np.maximum(noise * pptn, 0.0)  # ensure no negative precipitation
    # Calibrate model with perturbed precipitation -> have to rewrite it
    best_params, best_ofv, best_metrics, best_sim = CalibrateModel(pptn_perturbed, ags, optn_args.modl_objt) 
    # Calibrate model with reference precipitation
    q_fixed = optn_args.fast_model.run_model(pptn_perturbed, Best_Parameter_Assignment1)
    nse_fixed_raw = float(optn_args.fast_model.nse_raw(q_fixed))
    ofv_fixed = 1.0 - nse_fixed_raw
    # model = build_model(tems,pptn_perturbed, pets, tsps, dslr)
    # best_ofv_REF, _, best_sim_REF = objective_function(Best_Parameter_Assignment1, model,diso)
    #appending not right here anymor
    #print(f"\n--- Calibration run {i+1} of {Nsim} succesfully completed ---")
    return i, best_params, best_ofv, best_metrics, best_sim, ofv_fixed, q_fixed, pptn_perturbed

def wait_for_pool_init(mp_pool, pol_sze, pol_nme, vb):

    trs = 0
    tmr_bgn = timeit.default_timer()
    while True:
        pol_res = list(mp_pool.map(get_pid, range(pol_sze), chunksize=1))

        time.sleep(0.2)

        trs += 1

        if len(pol_res) == len(set(pol_res)): break

    tmr_end = timeit.default_timer()

    if vb: print(
        f'{pol_nme}', '|',
        os.getppid(), '|',
        pol_res, '|',
        trs, '|',
        f'{tmr_end - tmr_bgn:0.2f} secs')
    return


def get_pid(args):

    _ = args

    return os.getpid()


class OPTNARGS: pass

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
    m.set_optimization_flag(1)  # why: we optimize externally
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

def CalibrateModel(precipitation, ags, model):
    max_minutes = 20   # stop long DE runs by wall clock
    max_evals   = 60000  # cap total objective calls
    
    i = ags[0]
    optn_args = ags[1]
    # Perturb precipitation with normal noise
    pptn = optn_args.ppts.copy()
    tems = optn_args.tems.copy()
    pets = optn_args.pets.copy()
    tsps = optn_args.tsps
    dslr = optn_args.dslr
    diso = optn_args.diso.copy()
    BOUNDS = optn_args.BOUNDS.copy()
    PARAM_NAMES = optn_args.PARAM_NAMES.copy()
    prms0 = np.array([
         0.00, 0.10, 0.01, 0.10,
         0.00, 300., 70.,  2.50,
         0.00, 0.00,
         1.00, 0.01, 30., 0.10, 0.10, 0.01,
         1e-3, 1e-5
     ], dtype=float)
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
    
    m0 = build_model(tems, precipitation, pets, tsps, dslr)
    m0.set_parameters(prms0)
    m0.run_model()
    sim0 = m0.get_discharge()

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
        maxiter=500,                 # Reduced for variation
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
        options={'ftol': 1e-9, 'gtol': 1e-4} # Stricter tolerance for refinement
    )

    # Compare the results from DE and the local search, and pick the best one
    if local_res.fun < res.fun:
        de_best_params = local_res.x

    best_params = de_best_params
    best_ofv, best_metrics, best_sim = objective_function(best_params, m0, diso)
    #print("\nOverall optimization complete. Differnetial evolution and gradient based optimizer ran successfully. Best results:")

    # Prefer callback best; fallback to res.x
    save_params = best_solution["params"] if best_solution["params"] is not None else best_params
    save_ofv = local_res.fun#best_solution["ofv"] if np.isfinite(best_solution["ofv"]) else best_ofv
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

    #print(f"Best objective (1 - NSE): {best_ofv:.6f}")
    #print("Best-fit NSE:", round(best_metrics["NSE"], 4))

    return save_params, save_ofv, save_metrics, best_sim

#=======================================================================================
# Plotting functions
#=======================================================================================

def plot_cdf_ofvs(ofvs_ref, ofvs_perturbed, out_png):
    fig = plt.figure(figsize=(6, 4), dpi=120)
    sorted_ref = np.sort(ofvs_ref) # sort the ofvs which you round to then count the individual values
    sorted_perturbed = np.sort(ofvs_perturbed)
    p_ref = np.arange(1, len(sorted_ref) + 1) / len(sorted_ref)  # Plus 1 as python counts from zero. Now count cumulative probabilities (obs i / total obs)
    p_perturbed = np.arange(1, len(sorted_perturbed) + 1) / (len(sorted_perturbed)+1) # verify if we have an integer division or not
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

# CDF's of the parameters
def plot_cdf_params(params_ref, params_perturbed, out_png):
    PARAM_NAMES = [
    "snw_dth","snw_att","snw_pmf","snw_amf",
    "sl0_dth","sl0_pwp","sl0_fcy","sl0_bt0",
    "urr_dth","lrr_dth",
    "urr_wsr","urr_ulc","urr_tdh","urr_tdr","urr_ndr","urr_uct",
    "lrr_dre","lrr_lct"
    ]
    fig, axs = plt.subplots(len(PARAM_NAMES), 1, figsize=(6, 3 * len(PARAM_NAMES)), dpi=120)
    for i, name in enumerate(PARAM_NAMES):
        #sorted_ref = np.sort(params_ref[:, i])
        sorted_pertubed = np.sort(params_perturbed[:, i])
        #p_ref = np.arange(1, len(sorted_ref) + 1) / len(sorted_ref)
        p_perturbed = np.arange(1, len(sorted_pertubed) + 1) / (1+len(sorted_pertubed))
        axs[i].axvline(x=params_ref[i], color='red', linestyle='--', label='Reference Parameter Value')
        #axs[i].plot(sorted_ref, p_ref, label="Reference Precipitation", marker='o', linestyle='-', markersize=4)
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

def plot_cdf_precipitation_log(precip_ref, precip_perturbed, out_png):
    fig = plt.figure(figsize=(6, 4), dpi=120)
    sorted_ref = np.sort(precip_ref)
    sorted_perturbed = np.sort(precip_perturbed)
    p_ref = np.arange(1, len(sorted_ref) + 1) / len(sorted_ref)
    p_perturbed = np.arange(1, len(sorted_perturbed) + 1) / len(sorted_perturbed)
    plt.plot(sorted_ref, p_ref, label="Reference Precipitation", marker='o', linestyle='-', markersize=4)
    plt.plot(sorted_perturbed, p_perturbed, label="Perturbed Precipitation", marker='s', linestyle='--', markersize=4)
    plt.xlabel("Precipitation [mm/hr]")
    plt.ylabel("Cumulative Probability")
    plt.title("CDF of Precipitation")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.show()
    plt.close(fig)

# Scatterplots of OFVS vs perturbed inputs
def plot_scatter_ofvs_precipitation(OFVs, OFVs_REF, out_png):
    fig = plt.figure(figsize=(6, 4), dpi=120)
    plt.scatter(OFVs, OFVs_REF, alpha=0.7)
    # Add the angle bisector (1:1 line)
    min_val = min(min(OFVs), min(OFVs_REF))
    max_val = max(max(OFVs), max(OFVs_REF))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='1:1 line (angle bisector)')
    plt.xlabel("OFV when recalibrating after perturbation")
    plt.ylabel("OFV when using parameters from assignment 1")
    plt.title("Scatterplot of OFVs")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.show()
    plt.close(fig)


# Function to count how often we got better by chance when recalibratingg and what the min OFV was.
def analyze_ofv_improvements(ofvs_ref, ofvs_perturbed):
    improvements = [1 if ofvs_perturbed[i] < ofvs_ref[i] else 0 for i in range(len(ofvs_ref))]
    improvement_count = sum(improvements)
    min_ofv_perturbed = min(ofvs_perturbed)
    print(f"\nNumber of times recalibration led to better OFV: {improvement_count} out of {len(ofvs_ref)}")
    print(f"Minimum OFV achieved after recalibration: {min_ofv_perturbed:.6f}")
    return improvement_count, min_ofv_perturbed


# Boxplots of the simulation results for the individul parameter
def boxplot_parameters_results(parameter_name, parameter_values, ref_parameter, out_png):
    for j in range(len(parameter_name)):
        fig = plt.figure(figsize=(10, 6), dpi=120)
        plt.boxplot([parameter_values[i][j] for i in range(len(parameter_values))], labels=[parameter_name[j]], showfliers=False)
        plt.scatter(1, ref_parameter[j], color='red', label='Reference Parameter', zorder=5)
        plt.xticks(rotation=90)
        plt.xlabel("Parameters")
        plt.ylabel("Parameter Value")
        plt.title(f"Boxplot of Parameter: {parameter_name[j]}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_png.replace(".png", f"_{parameter_name[j]}.png"), dpi=200, bbox_inches="tight")
        plt.show()
        plt.close(fig)
#Now plot some of the recalibrated results against observed discharge diso
def plot_all_sim_obs(index, obs, sim_results, title, out_png):
    fig = plt.figure(figsize=(10, 6), dpi=120)
    plt.plot(index, obs, label="Observed Discharge", linewidth=3, color='black')
    for j, sim in enumerate(sim_results):
        plt.plot(index, sim, label=f"Simulated {j+1}", alpha=0.7)
    plt.grid(True); plt.legend()
    plt.xticks(rotation=45)
    plt.xlabel("Time [hr]"); plt.ylabel("Discharge [m³/s]")
    plt.title(title)
    plt.tight_layout(); plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.show(); plt.close(fig)


# Function to plot 1-F on log scale against reference precipitation
def plot_survival_log_vs_reference(precipitation_perturbed_all, reference_precipitation, out_png):
    """
    For each perturbed precipitation simulation, compute its CDF F.
    Plot 1-F on log scale against the sorted reference precipitation in red.
    This should show affine functions with negative slopes in the tail for each simulation.
    """
    # Sort the reference precipitation
    sorted_ref = np.sort(reference_precipitation)
    
    fig = plt.figure(figsize=(8, 6), dpi=120)
    
    # First, plot the reference precipitation survival in red
    ref_flat = reference_precipitation.flatten()
    sorted_ref_values = np.sort(ref_flat)
    n_ref = len(sorted_ref_values)
    ref_survival = []
    for x in sorted_ref:
        prop_le = np.searchsorted(sorted_ref_values, x, side='right') / n_ref
        survival = 1 - prop_le
        ref_survival.append(survival)
    ref_survival = np.maximum(np.array(ref_survival), 1e-10)
    plt.plot(sorted_ref, ref_survival, color='red', linewidth=2, label='Reference')
    
    # Plot for each perturbed simulation
    #colors = plt.cm.viridis(np.linspace(0, 1, len(precipitation_perturbed_all)))
    for i, pert in enumerate(precipitation_perturbed_all):
        pert_flat = pert.flatten()
        sorted_pert = np.sort(pert_flat)
        n = len(sorted_pert)
        
        # For each value in sorted_ref, compute 1 - F(x) where F is CDF of this perturbed
        survival_values = []
        for x in sorted_ref:
            # Proportion of this perturbed <= x
            prop_le = np.searchsorted(sorted_pert, x, side='right') / n
            survival = 1 - prop_le
            survival_values.append(survival)
        
        survival_values = np.maximum(np.array(survival_values), 1e-10)
        
        # Plot with black color
        plt.plot(sorted_ref, survival_values, color='black', alpha=0.7)
    
    plt.xlabel("Reference Precipitation [mm/hr]")
    plt.ylabel("1 - F")
    plt.yscale('log')
    plt.title("Survival Function of Perturbed Precipitation on Log Scale vs Reference")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.show()
    plt.close(fig)


# Scenario A: Fixed
class FastHBVModel:
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.load_data()
        
        self.model = HBV001A()
        self.model.set_inputs(self.tems, self.ppts_orig, self.pets)
        self.model.set_outputs(self.tsps)
        self.model.set_discharge_scaler(self.dslr)
        self.model.set_optimization_flag(0)
        
        self.warmup = 150
        self.q_obs_warmup = self.diso[self.warmup:]
        self.q_obs_mean = float(np.mean(self.q_obs_warmup))
        self.denominator = float(np.sum((self.q_obs_warmup - self.q_obs_mean) ** 2))
    
    def load_data(self):
        ts_csv = self.data_dir / "time_series___24163005.csv"
        ar_csv = self.data_dir / "area___24163005.csv"
        
        df = pd.read_csv(ts_csv, sep=";", index_col=0)
        self.tems = df["tavg__ref"].values.astype(np.float32)
        self.ppts_orig = df["pptn__ref"].values.astype(np.float32)
        self.pets = df["petn__ref"].values.astype(np.float32)
        self.diso = df["diso__ref"].values.astype(np.float32)
        self.tsps = len(self.tems)
        
        cca_df = pd.read_csv(ar_csv, sep=";", index_col=0)
        ccaa = float(cca_df.values[0, 0])
        self.dslr = ccaa / (3600.0 * 1000.0)
    
    def run_model(self, ppt, params):
        self.model.set_inputs(self.tems, ppt, self.pets)
        self.model.set_parameters(np.asarray(params, dtype=np.float32))
        self.model.run_model()
        return self.model.get_discharge()
    
    def nse_raw(self, q_sim):
        q_sim_w = q_sim[self.warmup:]
        if not np.all(np.isfinite(q_sim_w)): return -999.0
        valid = np.isfinite(q_sim_w)
        if int(np.sum(valid)) < 100: return -999.0
        
        q_s = q_sim_w[valid]
        q_o = self.q_obs_warmup[valid]
        num = float(np.sum((q_o - q_s) ** 2))
        if self.denominator <= 0: return -999.0
        return 1.0 - num / self.denominator
    
    def objective(self, ppt, params):
        try:
            q = self.run_model(ppt, params)
            nse = self.nse_raw(q)
            if not np.isfinite(nse) or nse <= -900: return 1e6
            return 1.0 - nse
        except: return 1e6

if __name__ == "__main__":
    main()