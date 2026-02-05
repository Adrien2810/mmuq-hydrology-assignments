from hmg import HBV001A
from scipy.optimize import differential_evolution, minimize
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from queue import Queue
from multiprocessing.pool import ThreadPool
from multiprocessing import Manager, Pool
from pathlib import Path
import traceback as tb
import timeit
import time
import sys
import os
# TODO: Ratingcurves plotting, plotting the requirements, plotting the perturbed series, analysis
# Numpy sneakily uses multiple threads sometimes. I don't want that.
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MPI_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'


sys.path.append(os.path.expanduser("~/.local/lib/python3.11/site-packages"))


def main():
    PARAM_NAMES = [
        "snw_dth", "snw_att", "snw_pmf", "snw_amf",
        "sl0_dth", "sl0_pwp", "sl0_fcy", "sl0_bt0",
        "urr_dth", "lrr_dth",
        "urr_wsr", "urr_ulc", "urr_tdh", "urr_tdr", "urr_ndr", "urr_uct",
        "lrr_dre", "lrr_lct"
    ]
    # Bounds for DE
    BOUNDS = [
        (0.00, 0.00),   # snw_dth  (fixed 0)
        (-2.0, 3.0),    # snw_att
        (0.00, 3.00),   # snw_pmf
        (0.00, 10.0),   # snw_amf

        (0.00, 100.0),  # sl0_dth
        (5.00, 700.0),  # sl0_pwp
        (100.0, 700.0),  # sl0_fcy
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
    # data_dir = Path.home() / ... / ...
    if not data_dir.exists():
        raise FileNotFoundError(f"Data folder not found: {data_dir}")
    # os.chdir(data_dir)
    print(f"Using data from: {data_dir}")
    # read time series and area files
    df = pd.read_csv(data_dir / "time_series___24163005.csv",
                     sep=";", index_col=0)
    df.index = pd.to_datetime(df.index, format="%Y-%m-%d-%H")
    cca = float(pd.read_csv(data_dir / "area___24163005.csv",
                sep=";", index_col=0).values[0, 0])

    required_cols = ["tavg__ref", "pptn__ref", "petn__ref", "diso__ref"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")
    # set the parameters
    tems = df["tavg__ref"].values
    ppts = df["pptn__ref"].values
    pets = df["petn__ref"].values
    diso = df["diso__ref"].values
    ddho = df["ddho__ref"].values
    tsps = len(tems)
    dslr = cca / (3600 * 1000)
    # --------------------------------------------------------------------------------------
    # Use Nsim instead of SobolSampleSize
    Nsim = 1400
    # Use float32 as HBV001A does so as well and it is less expensive
    dtyp = np.float32
    mprg_pool_size = 4
    #====
    # Build interpolating function for diso from ddho using 3-segment power-law fit
    fit_first, fit_second, fit_third = build_diso_from_ddho(df)
    diso_interpolated = get_discharge_from_ddho(df["ddho__ref"].values, fit_first, fit_second, fit_third)

    # ==========================================================================
    # Verify if Nsim is compatible with your amount of workes. Adjust if necessary
    if (Nsim % mprg_pool_size):

        Nsim += (mprg_pool_size - (Nsim % mprg_pool_size))

        assert (Nsim % mprg_pool_size) == 0

        print('Adjusted Nsim to', Nsim)
    modl_objt = HBV001A()
    modl_objt.set_inputs(tems, ppts, pets)
    modl_objt.set_outputs(tsps)

    modl_objt.set_discharge_scaler(dslr)

    # Pass all the arguments as a class (?)
    # ==========================================================================

    optn_args = OPTNARGS()

    optn_args.cntr = 0
    optn_args.oflg = 0
    optn_args.vbse = False
    optn_args.df = df
    optn_args.diso = diso
    optn_args.ddho = ddho
    optn_args.ppts = ppts
    optn_args.tems = tems
    optn_args.pets = pets
    optn_args.tsps = tsps
    optn_args.fit_first = fit_first
    optn_args.fit_second = fit_second
    optn_args.fit_third = fit_third
    optn_args.dslr = dslr
    optn_args.diso_interpolated = diso_interpolated
    optn_args.Best_Parameter_Assignment1 = Best_Parameter_Assignment1
    optn_args.PARAM_NAMES = PARAM_NAMES
    optn_args.BOUNDS = BOUNDS
    optn_args.take_idxs = np.isfinite(diso)

    modl_objt.set_optimization_flag(1)
    optn_args.Nsim = Nsim
    # ==========================================================================
    # Copy from Faizan and adapt the function you want to parallize on
    # ==========================================================================
    # Multiprocessing.
    # ==========================================================================
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
    Perturbed_discharge = []
    Perturbed_depth = []
    OFV_REF = []
    for j in range(Nsim):
        i, best_params, best_ofv, best_metrics, best_sim, ddho_new, diso_new, ofv_old = result[
            j]
        CalibratedParameters.append(best_params)
        OFV.append(best_ofv)
        Metrics.append(best_metrics)
        SimulationResults.append(best_sim)
        Perturbed_discharge.append(diso_new)
        Perturbed_depth.append(ddho_new)
        OFV_REF.append(ofv_old)

        # here all the appending can happen
    # Analyze the error the interpolation gives compared to the original data
    ofv_orig,_, _ = objective_function(
        Best_Parameter_Assignment1, modl_objt, diso)
    _, ofv_interpol, _ , _= CalibrateModel(get_discharge_from_ddho(df["ddho__ref"].values, fit_first, fit_second, fit_third), (0, optn_args), modl_objt)
    print(f"OFV original discharge: {ofv_orig:.6f}")
    print(f"OFV interpolated discharge: {ofv_interpol:.6f}")
    # Now plot all after collecting results
    plot_cdf_ofvs(OFV_REF, OFV, "cdf_ofvs.png")
    plot_cdf_params(Best_Parameter_Assignment1, np.array(
        CalibratedParameters), "cdf_parameters.png")
    # For precipitation CDF, perhaps plot reference vs one perturbed, e.g., the first
    if Perturbed_discharge:
        plot_cdf_variable(optn_args.diso.copy(
        ), Perturbed_discharge, "Discharge", "m³/s", "cdf_discharge.png")
        plot_cdf_variable(optn_args.ddho.copy(),
                          Perturbed_depth, "Depth", "m", "cdf_depth.png")

    #plot_rating_curve(diso, ddho, "rating_curve_reference.png")
    #plot_rating_curve(
        #Perturbed_discharge[0], Perturbed_depth[0], "rating_curve_perturbed.png")
    plot_perturbation_curve_dep(
        df.index, Perturbed_depth[0], "rating_curve_time_series_perturbed.png")
    plot_perturbation_curve_dis(
        df.index, Perturbed_discharge[0], "rating_curve_time_series_perturbed.png")
    plot_scatter_ofvs_precipitation(OFV, OFV_REF, "scatter_ofvs.png")
    improvement_count, min_ofv_perturbed = analyze_ofv_improvements(
        OFV_REF, OFV)
    print(
        f"\nNumber of improvements by recalibration: {improvement_count} out of {Nsim}")
    # boxplot_parameters_results(PARAM_NAMES, CalibratedParameters, Best_Parameter_Assignment1, "boxplot_parameters_results.png")
    np.random.seed(42)  # for reproducibility
    random_indices = np.random.choice(len(SimulationResults), 2, replace=False)
    selected_sims = [SimulationResults[i] for i in random_indices]
    selected_discharges = [Perturbed_discharge[i] for i in random_indices]
    plot_sims_curves(df.index, selected_discharges, selected_sims,
                     "Reference Discharge and 3 Random Simulated Discharges", "Random_3_Simulations_Discharge.png")
    # plot_survival_log_vs_reference(pptn_perturbed_all, optn_args.ppts.copy(), "survival_log_vs_reference.png")

    print(f"Successfully processed {len(result)} simulations.")
    # ===========================================================================
    # ==========================================================================
    # Save all results to CSV
    data = []
    for j in range(Nsim):
        row = {
            'Simulation_ID': j,
            'OFV': OFV[j],
            'OFV_REF': OFV_REF[j],
            'NSE': Metrics[j]['NSE'],
        }
        for k, name in enumerate(PARAM_NAMES):
            row[name] = CalibratedParameters[j][k]
            row[f'{name}_ref'] = Best_Parameter_Assignment1[k]
        data.append(row)
    df = pd.DataFrame(data)
    df.to_csv('all_results.csv', index=False)
    df.to_excel('all_results.xlsx', index=False)
    print(f"Results saved to {os.getcwd()}/all_results.csv")
    print("Results saved to all_results.csv")
    # ===========================================================================
    mprg_pool.close()

    end_tmr = timeit.default_timer()

    print(f'Took {end_tmr - bgn_tmr:0.2E} seconds to complete.')
    # ==========================================================================

# Arguments are two thing: the idx and opt_args that has everything that we need


def multi(ags):
    i = ags[0]
    optn_args = ags[1]
    Nsim = optn_args.Nsim
    print(f"\n--- Calibration run {i+1} of {Nsim} ---")
    import sys
    sys.stdout.flush()
    df = optn_args.df.copy()
    Best_Parameter_Assignment1 = optn_args.Best_Parameter_Assignment1.copy()
    # Build original function
    diso_from_ddho = optn_args.diso_interpolated.copy()
    # ddho perturbieren
    ddho_new = perturbed_ddho(df["ddho__ref"].values)
    # Now get new discharge profile dependent on the new perturbed depth
    diso_new = get_discharge_from_ddho(ddho_new, optn_args.fit_first, optn_args.fit_second, optn_args.fit_third)
    # Calibrate model with changed discharge -> have to rewrite it
    best_params, best_ofv, best_metrics, best_sim = CalibrateModel(
        diso_new, ags, None)
    ofv_old, _, _ = objective_function(
        Best_Parameter_Assignment1, build_model(optn_args.tems, optn_args.ppts, optn_args.pets, optn_args.tsps, optn_args.dslr), diso_new)
    # appending not right here anymore
    # print(f"\n--- Calibration run {i+1} of {Nsim} succesfully completed ---")
    return i, best_params, best_ofv, best_metrics, best_sim, ddho_new, diso_new, ofv_old


def wait_for_pool_init(mp_pool, pol_sze, pol_nme, vb):

    trs = 0
    tmr_bgn = timeit.default_timer()
    while True:
        pol_res = list(mp_pool.map(get_pid, range(pol_sze), chunksize=1))

        time.sleep(0.2)

        trs += 1

        if len(pol_res) == len(set(pol_res)):
            break

    tmr_end = timeit.default_timer()

    if vb:
        print(
            f'{pol_nme}', '|',
            os.getppid(), '|',
            pol_res, '|',
            trs, '|',
            f'{tmr_end - tmr_bgn:0.2f} secs')
    return


def get_pid(args):

    _ = args

    return os.getpid()


class OPTNARGS:
    pass

# ======================================================================================
# Define functions to perform perturbation
# ======================================================================================


def perturbed_ddho(ddho):
    #values = np.linspace(-5, 5, 11)
    perturbation = np.random.uniform(-5, 5, size=len(ddho))
    ddho_perturbed = ddho + perturbation
    return ddho_perturbed

def get_discharge_from_ddho(ddho, fit_first, fit_second, fit_third, c=None):
    """Predict discharge from height using 3-segment piecewise power-law."""
    ddho = np.asarray(ddho, dtype=float)
    
    # Extract parameters
    a1, b1, c1 = fit_first['a'], fit_first['b'], fit_first['c']
    h_break1 = fit_first['h_break']
    
    a2, b2, c2 = fit_second['a'], fit_second['b'], fit_second['c']
    h_break2 = fit_second['h_break']
    
    a3, b3, c3 = fit_third['a'], fit_third['b'], fit_third['c']
    
    # Compute Q = a*(h-c)^b for each segment
    delta1 = np.maximum(ddho - c1, 1e-10)
    delta2 = np.maximum(ddho - c2, 1e-10)
    delta3 = np.maximum(ddho - c3, 1e-10)
    
    q1 = a1 * (delta1 ** b1)
    q2 = a2 * (delta2 ** b2)
    q3 = a3 * (delta3 ** b3)
    
    # Select appropriate segment based on height
    result = np.zeros_like(ddho)
    result[ddho <= h_break1] = q1[ddho <= h_break1]
    result[(ddho > h_break1) & (ddho < h_break2)] = q2[(ddho > h_break1) & (ddho < h_break2)]
    result[ddho >= h_break2] = q3[ddho >= h_break2]
    
    return result

def build_diso_from_ddho(df):
    """Fit piecewise power-law rating curve using LinearRegression in log-space with THREE segments."""
    from sklearn.linear_model import LinearRegression
    
    # Hard-coded parameters from optimized fit
    c_hardcoded = 95.0
    h_break1 = 200.0
    h_break2 = 440.0
    
    df = df.copy()
    df.sort_values(by='ddho__ref', inplace=True)
    
    # Segment 1: heights <= h_break1
    seg1_mask = df['ddho__ref'] <= h_break1
    h_seg1 = df[seg1_mask]['ddho__ref'].values
    q_seg1 = df[seg1_mask]['diso__ref'].values
    
    delta_h1 = h_seg1 - c_hardcoded
    if np.any(delta_h1 <= 0):
        delta_h1 = np.maximum(delta_h1, 1e-6)
    
    log_h1 = np.log(delta_h1)
    log_q1 = np.log(q_seg1)
    
    X1 = log_h1.reshape(-1, 1)
    model1 = LinearRegression()
    model1.fit(X1, log_q1)
    
    ln_a1 = model1.intercept_
    b1 = model1.coef_[0]
    a1 = np.exp(ln_a1)
    
    # Segment 2: h_break1 < heights < h_break2
    seg2_mask = (df['ddho__ref'] > h_break1) & (df['ddho__ref'] < h_break2)
    h_seg2 = df[seg2_mask]['ddho__ref'].values
    q_seg2 = df[seg2_mask]['diso__ref'].values
    
    delta_h2 = h_seg2 - c_hardcoded
    if np.any(delta_h2 <= 0):
        delta_h2 = np.maximum(delta_h2, 1e-6)
    
    log_h2 = np.log(delta_h2)
    log_q2 = np.log(q_seg2)
    
    X2 = log_h2.reshape(-1, 1)
    model2 = LinearRegression()
    model2.fit(X2, log_q2)
    
    ln_a2 = model2.intercept_
    b2 = model2.coef_[0]
    a2 = np.exp(ln_a2)
    
    # Segment 3: heights >= h_break2
    seg3_mask = df['ddho__ref'] >= h_break2
    h_seg3 = df[seg3_mask]['ddho__ref'].values
    q_seg3 = df[seg3_mask]['diso__ref'].values
    
    delta_h3 = h_seg3 - c_hardcoded
    if np.any(delta_h3 <= 0):
        delta_h3 = np.maximum(delta_h3, 1e-6)
    
    log_h3 = np.log(delta_h3)
    log_q3 = np.log(q_seg3)
    
    X3 = log_h3.reshape(-1, 1)
    model3 = LinearRegression()
    model3.fit(X3, log_q3)
    
    ln_a3 = model3.intercept_
    b3 = model3.coef_[0]
    a3 = np.exp(ln_a3)
    
    # Return segments with power-law parameters
    fit_first = {'a': a1, 'b': b1, 'c': c_hardcoded, 'h_break': h_break1}
    fit_second = {'a': a2, 'b': b2, 'c': c_hardcoded, 'h_break': h_break2}
    fit_third = {'a': a3, 'b': b3, 'c': c_hardcoded}
    
    print(f"Built 3-segment rating curve:")
    print(f"  Segment 1 (h≤{h_break1}): Q = {a1:.4f}*(h-{c_hardcoded})^{b1:.4f}")
    print(f"  Segment 2 ({h_break1}<h<{h_break2}): Q = {a2:.4f}*(h-{c_hardcoded})^{b2:.4f}")
    print(f"  Segment 3 (h≥{h_break2}): Q = {a3:.4f}*(h-{c_hardcoded})^{b3:.4f}")
    
    return fit_first, fit_second, fit_third
# ========================================================================================
# Function to compute NSE


def calc_nse(obs: np.ndarray, sim: np.ndarray) -> float:
    """Compute NSE."""
    obs = np.asarray(obs, float)
    sim = np.asarray(sim, float)
    mask = np.isfinite(obs) & np.isfinite(sim)
    obs = obs[mask]
    sim = sim[mask]
    denom = np.sum((obs - obs.mean()) ** 2)
    if denom <= 0:
        return float("-inf")
    return 1.0 - np.sum((obs - sim) ** 2) / denom

# Initialize the model


def build_model(tems, ppts, pets, tsps, dslr) -> HBV001A:
    """Construct and initialize HBV001A for a run."""
    m = HBV001A()
    m.set_inputs(tems, ppts, pets)  # set the inputs from the time series
    m.set_outputs(tsps)
    m.set_discharge_scaler(dslr)
    m.set_optimization_flag(1)  # why: we optimize externally
    return m

# Here we define the objective function for the DE optimization


def objective_function(params, model, diso):
    """Clamp params, run model, return (ofv=1-NSE, {'NSE': nse}, sim)."""
    p = np.asarray(params, float).copy()
    try:
        # set the model parameters to the initialized model
        model.set_parameters(p)
    except AssertionError:
        return 1e6, {"NSE": float("-inf")}, None
    model.run_model()
    sim = model.get_discharge()  # simulate the discharge
    # calculate the NSE between observed and simulated discharge
    nse = calc_nse(diso, sim)
    ofv = 1.0 - nse if np.isfinite(nse) else 1e6
    return ofv, {"NSE": float(nse)}, sim


def simple_ofv(params, model, diso):
    """Return only the objective function value for local search."""
    ofv, _, _ = objective_function(params, model, diso)
    return ofv


def CalibrateModel(ref_diso, ags, model):
    max_minutes = 2   # stop long DE runs by wall clock
    max_evals = 150000  # cap total objective calls

    i = ags[0]
    optn_args = ags[1]
    # Change the reference discharge
    pptn = optn_args.ppts.copy()
    tems = optn_args.tems.copy()
    pets = optn_args.pets.copy()
    tsps = optn_args.tsps
    dslr = optn_args.dslr
    diso = ref_diso
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
    popsize = 10
    n_params = len(BOUNDS)
    pop_n = popsize * n_params         # approx trials per generation
    eval_idx = 0
    start_time = time.time()

    m0 = build_model(tems, pptn, pets, tsps, dslr)
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
            print(
                f"\n Stopping early: elapsed={elapsed_min:.2f} min, evals={eval_idx}")
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
            # print(f"\n New best! Gen {len(best_by_gen)}: OFV={ofv:.4f}, NSE={best_solution['metrics']['NSE']:.4f}")

        # print(f"Gen {len(best_by_gen):3d} | best (1-NSE) = {ofv:.6f}")
        return False

    res = differential_evolution(
        recorded_ofv,
        bounds=BOUNDS,
        strategy="best1bin",  # A good default strategy
        maxiter=1200,                 # Reduced for variation
        popsize=popsize,
        tol=1e-4,
        # Increased upper bound for mutation to encourage more exploration
        mutation=(0.5, 1.0),
        recombination=0.7,  # Increased to promote more exploration
        seed=42,
        callback=de_callback,
        # polish=True,
        updating="deferred",
        # workers=-1,
        atol=1e-4,
        polish=False
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
        # Stricter tolerance for refinement
        options={'ftol': 1e-9, 'gtol': 1e-4}
    )

    # Compare the results from DE and the local search, and pick the best one
    if local_res.fun < res.fun:
        de_best_params = local_res.x

    best_params = de_best_params
    best_ofv, best_metrics, best_sim = objective_function(
        best_params, m0, diso)
    # print("\nOverall optimization complete. Differnetial evolution and gradient based optimizer ran successfully. Best results:")

    # Prefer callback best; fallback to res.x
    save_params = best_solution["params"] if best_solution["params"] is not None else best_params
    # best_solution["ofv"] if np.isfinite(best_solution["ofv"]) else best_ofv
    save_ofv = min(local_res.fun, res.fun)
    save_metrics = best_solution["metrics"] if best_solution["metrics"] is not None else best_metrics

    return save_params, save_ofv, save_metrics, best_sim

# =======================================================================================
# Plotting functions
# =======================================================================================


def plot_sims_curves(time_index, observed, simulations, title, out_png):
    fig = plt.figure(figsize=(10, 6), dpi=120)
    for i, (obs, sim) in enumerate(zip(observed, simulations)):
        plt.plot(time_index, obs, label=f"Target {i+1}",
                 color=f'C{i}', linestyle='--', linewidth=1.5, alpha=0.8)
        plt.plot(time_index, sim,
                 label=f"Sim {i+1}", color=f'C{i}', linestyle='-', linewidth=1.0, alpha=0.8)
    plt.xlabel("Time")
    plt.ylabel("Discharge [m³/s]")
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.show()
    plt.close(fig)


def plot_perturbation_curve_dis(depth, discharge, out_png):
    fig = plt.figure(figsize=(10, 4), dpi=120)
    plt.plot(depth, discharge)
    plt.ylabel("Discharge [m³/s]")
    plt.xlabel("Time")
    plt.title("Perturbation Curve of Discharge")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.show()
    plt.close(fig)

def plot_perturbation_curve_dep(depth, discharge, out_png):
    fig = plt.figure(figsize=(10, 4), dpi=120)
    plt.plot(depth, discharge)
    plt.ylabel("Depth [cm]")
    plt.xlabel("Time")
    plt.title("Perturbation Curve of Depth")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.show()
    plt.close(fig)


def plot_cdf_ofvs(ofvs_ref, ofvs_perturbed, out_png):
    fig = plt.figure(figsize=(6, 4), dpi=120)
    # sort the ofvs which you round to then count the individual values
    sorted_ref = np.sort(ofvs_ref)
    sorted_perturbed = np.sort(ofvs_perturbed)
    # Plus 1 as python counts from zero. Now count cumulative probabilities (obs i / total obs)
    p_ref = np.arange(1, len(sorted_ref) + 1) / len(sorted_ref)
    # verify if we have an integer division or not
    p_perturbed = np.arange(1, len(sorted_perturbed) +
                            1) / (len(sorted_perturbed)+1)
    plt.plot(sorted_ref, p_ref, label="Reference Discharge",
             marker='o', linestyle='-', markersize=4)
    plt.plot(sorted_perturbed, p_perturbed, label="Perturbed Discharge",
             marker='s', linestyle='--', markersize=4)
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
        "snw_dth", "snw_att", "snw_pmf", "snw_amf",
        "sl0_dth", "sl0_pwp", "sl0_fcy", "sl0_bt0",
        "urr_dth", "lrr_dth",
        "urr_wsr", "urr_ulc", "urr_tdh", "urr_tdr", "urr_ndr", "urr_uct",
        "lrr_dre", "lrr_lct"
    ]
    fig, axs = plt.subplots(len(PARAM_NAMES), 1, figsize=(
        6, 3 * len(PARAM_NAMES)), dpi=120)
    for i, name in enumerate(PARAM_NAMES):
        # sorted_ref = np.sort(params_ref[:, i])
        sorted_pertubed = np.sort(params_perturbed[:, i])
        # p_ref = np.arange(1, len(sorted_ref) + 1) / len(sorted_ref)
        p_perturbed = np.arange(
            1, len(sorted_pertubed) + 1) / (1+len(sorted_pertubed))
        axs[i].axvline(x=params_ref[i], color='red',
                       linestyle='--', label='Reference Parameter Value')
        # axs[i].plot(sorted_ref, p_ref, label="Reference Precipitation", marker='o', linestyle='-', markersize=4)
        axs[i].plot(sorted_pertubed, p_perturbed, label="Perturbed Precipitation",
                    marker='s', linestyle='--', markersize=4)
        axs[i].set_xlabel(f"Parameter: {name}")
        axs[i].set_ylabel("Cumulative Probability")
        axs[i].set_title(f"CDF of {name}")
        axs[i].grid(True)
        axs[i].legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.show()
    plt.close(fig)

# Do a foor loop over the different sims
def plot_cdf_variable(ref_data, perturbed_data, var_name, unit, out_png):
    # Plot all perturbed CDFs in thin gray lines and the reference in red
    fig = plt.figure(figsize=(6, 4), dpi=120)
    sorted_ref = np.sort(ref_data)
    p_ref = np.arange(1, len(sorted_ref) + 1) / len(sorted_ref)

    # Debug: report how many perturbed series we will plot
    try:
        n_pert = len(perturbed_data)
    except Exception:
        n_pert = 0
    print(f"plot_cdf_variable: plotting {n_pert} perturbed CDF(s)")

    # Plot each perturbed CDF as a thin gray line
    for i in range(len(perturbed_data)):
        sorted_perturbed = np.sort(perturbed_data[i])
        p_perturbed = np.arange(1, len(sorted_perturbed) + 1) / len(sorted_perturbed)
        plt.plot(sorted_perturbed, p_perturbed,
                 color='gray', linewidth=1.5, alpha=0.6)

    # Overlay the reference CDF in red with marker
    plt.plot(sorted_ref, p_ref,
             color='red', label=f"Reference {var_name}", marker='o', linestyle='-', markersize=1)

    plt.xlabel(f"{var_name} [{unit}]")
    plt.ylabel("Cumulative Probability")
    plt.title(f"CDF of {var_name}")
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
    plt.plot([min_val, max_val], [min_val, max_val],
             'r--', label='1:1 line (angle bisector)')
    plt.xlabel(
        "OFV when recalibrating after perturbation of depth and hence discharge")
    plt.ylabel(
        "OFV when using parameters from assignment 1 and modified discharge")
    plt.title("Scatterplot of OFVs")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.show()
    plt.close(fig)

# Function to count how often we got better by chance when recalibratingg and what the min OFV was.


def analyze_ofv_improvements(ofvs_ref, ofvs_perturbed):
    improvements = [1 if ofvs_perturbed[i] < ofvs_ref[i]
                    else 0 for i in range(len(ofvs_ref))]
    improvement_count = sum(improvements)
    min_ofv_perturbed = min(ofvs_perturbed)
    print(
        f"\nNumber of times recalibration led to better OFV: {improvement_count} out of {len(ofvs_ref)}")
    print(f"Minimum OFV achieved after recalibration: {min_ofv_perturbed:.6f}")
    return improvement_count, min_ofv_perturbed

# Boxplots of the simulation results for the individul parameter


def boxplot_parameters_results(parameter_name, parameter_values, ref_parameter, out_png):
    for j in range(len(parameter_name)):
        fig = plt.figure(figsize=(10, 6), dpi=120)
        plt.boxplot([parameter_values[i][j] for i in range(
            len(parameter_values))], labels=[parameter_name[j]], showfliers=False)
        plt.scatter(1, ref_parameter[j], color='red',
                    label='Reference Parameter', zorder=5)
        plt.xticks(rotation=90)
        plt.xlabel("Parameters")
        plt.ylabel("Parameter Value")
        plt.title(f"Boxplot of Parameter: {parameter_name[j]}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_png.replace(
            ".png", f"_{parameter_name[j]}.png"), dpi=200, bbox_inches="tight")
        # plt.show()
        plt.close(fig)


if __name__ == "__main__":
    main()