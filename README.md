# mmuq-hydrology-assignments
Group work of Nafisa, Vincent and Adrien for the MMUQ Hydrology assignments

Assignment – 1, Tasks

• Implement various efficiency metrics
− NSE, LnNSE, PBias…
Nash–Sutcliffe Efficiency:
    dnm = np.sum((obs - obs.mean())**2)
    nse = 1.0 - np.sum((obs - sim)**2) / dnm
    
Log-NSE: 
    obsl = np.log1p(obs)
    siml = np.log1p(sim)
    dnml = np.sum((obsl - obsl.mean())**2)
    lognse = 1 - np.sum((obsl - siml)**2) / dnml
    
Percentage Bias: 
    pbias = 100.0 * (np.sum(sim - obs) / np.sum(obs)) if np.sum(obs) != 0 else np.nan
    
• Implement the objective function (OF)
− Accepts model parameters 
− Accepts model and related data
− Runs model
− Computes efficiency metrics
− Returns objective function value (OFV)




• Implement the optimization scheme

• Obtain best performing parameter vector
Estimate the least number of processes required
− Similar overall efficiency
− Makes physical sense

• Plot internal model variables
− With all processes turned on
− With some turned off

• Plot convergence of the OFVs
• Plot parameter vs. OFVs (each)
− Which ones are sensitive?

