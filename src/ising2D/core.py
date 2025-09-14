#!/usr/bin/env python
import numpy as np
from numba import njit
from joblib import Parallel, delayed
from collections import defaultdict # to compute pdf
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d

# Maximum number of steps to keep for truncated output.
MAX_STEPS = 200000
EPSILON = 1e-6 # to avoid log(p) getting too large due to random fluctuations
DENO_EPS = 0.5 # to avoid division by small number causing info-geometric form to blow up


#--------------------------------------------------------------------------
# Low-level (Numba-accelerated) functions
#--------------------------------------------------------------------------

@njit(nogil=True)
def initialise(L, bias):
    """
    Initialize an LxL lattice with values in {1, -1}.
    Each site is set to 1 with probability `bias`, otherwise -1.
    Uses vectorized operations.
    """
    rand_matrix = np.random.rand(L, L)
    lattice = np.where(rand_matrix < bias, 1, -1)
    return lattice

@njit(nogil=True)
def get_mu(lattice):
    """
    Compute the net interaction energy (mu) using periodic boundary conditions.
    """
    L = lattice.shape[0]
    mu = 0.0
    for i in range(L):
        for j in range(L):
            down  = lattice[(i + 1) % L, j]
            right = lattice[i, (j + 1) % L]
            mu += -lattice[i, j] * (down + right)
    return mu

@njit(nogil=True)
def metropolis(lattice, time, J, temperature, mu, E0, h, truncate):
    """
    Run the Metropolis algorithm for a given number of time steps in non-equilibrium ising model.
    
    Parameters:
      lattice : 2D array of int64
          The initial lattice.
      time : int
          Total number of time steps.
      J : float
          Coupling strength.
      temperature : float
          Temperature of the system.
      mu : float
          Initial net interaction energy.
      E0 : 1D array of float
          Additional energy offset for each time step. If constant pass np.full(time, E0).
      h : 1D array of float
            External field at each time step.
      truncate : bool
          If True, only the last min(time, MAX_STEPS) observations are kept.
    
    Returns:
      magnetisations, mus
      Each is a 1D array of floats containing lattice magnetisation and net energy (mu) respectively.
    """
    if temperature <= 0:
        raise ValueError("Temperature must be greater than 0.")
    if time <= 0:
        raise ValueError("Time must be greater than 0.")
    beta = 1.0 / temperature
    L = lattice.shape[0]
    if E0 is None:
        E0 = np.zeros(time, dtype=np.float64)
    if h is None:
        h = np.zeros(time, dtype=np.float64)
    
    if truncate:
        # truncate=True returns the last min(time, MAX_STEPS) observations
        sample = min(time, MAX_STEPS)
        magnetisations = np.zeros(sample)
        mus = np.zeros(sample)
    else:
        # keep results for every time step.
        magnetisations = np.zeros(time)
        mus = np.zeros(time)
    
    for t in range(time):
        x = np.random.randint(L)
        y = np.random.randint(L)
        spin_i = lattice[x, y]
        spin_f = -spin_i
        # Compute local energy contribution at the chosen site.
        mu_i = -spin_i * (lattice[(x - 1) % L, y] +
                            lattice[(x + 1) % L, y] +
                            lattice[x, (y - 1) % L] +
                            lattice[x, (y + 1) % L])
        mu_f = -spin_f * (lattice[(x - 1) % L, y] +
                            lattice[(x + 1) % L, y] +
                            lattice[x, (y - 1) % L] +
                            lattice[x, (y + 1) % L])
        # H(s) = -J*sum(si*sj) - h*si
        # dE = H(s_f) - H(s_i)
        dE = J * (mu_f - mu_i) + h[t] * (-spin_f + spin_i) #negative sign included in mu_i and mu_f
        dE_eff = dE + E0[t]
        if dE_eff < 0 or np.random.rand() < np.exp(-beta * dE_eff):
            lattice[x, y] = spin_f
            mu += (mu_f - mu_i)
        if truncate and t >= time - sample:
            idx = t - (time - sample)
            mus[idx] = mu
            magnetisations[idx] = lattice.sum() / lattice.size
        elif truncate and t < time - sample:
            # do not store the first time - sample steps
            pass
        else:
            mus[t] = mu
            magnetisations[t] = lattice.sum() / lattice.size
    return magnetisations, mus
    

@njit(nogil=True)
def glauber(lattice, time, J, temperature, mu, E0, h, truncate):
    """
    Run the Glauber algorithm for a given number of time steps in non-equilibrium ising model.
    
    Parameters:
      lattice : 2D array of int64
          The initial lattice.
      time : int
          Total number of time steps.
      J : float
          Coupling strength.
      temperature : float
          Temperature of the system.
      mu : float
          Initial net interaction energy.
      E0 : 1D array of float
          Additional energy offset for each time step. If constant pass np.full(time, E0).
      h : 1D array of float
            External field at each time step.
      truncate : bool
          If True, only the last min(time, MAX_STEPS) observations are kept.
    
    Returns:
      magnetisations, mus
      Each is a 1D array of floats containing lattice magnetisation and net energy (mu) respectively.
    """
    if temperature <= 0:
        raise ValueError("Temperature must be greater than 0.")
    if time <= 0:
        raise ValueError("Time must be greater than 0.")
    beta = 1.0 / temperature
    L = lattice.shape[0]
    if E0 is None:
        E0 = np.zeros(time, dtype=np.float64)
    if h is None:
        h = np.zeros(time, dtype=np.float64)
    
    if truncate:
        # truncate=True returns the last min(time, MAX_STEPS) observations
        sample = min(time, MAX_STEPS)
        magnetisations = np.zeros(sample)
        mus = np.zeros(sample)
    else:
        # keep results for every time step.
        magnetisations = np.zeros(time)
        mus = np.zeros(time)
    
    for t in range(time):
        x = np.random.randint(L)
        y = np.random.randint(L)
        spin_i = lattice[x, y]
        spin_f = -spin_i
        # Compute local energy contribution at the chosen site.
        mu_i = -spin_i * (lattice[(x - 1) % L, y] +
                            lattice[(x + 1) % L, y] +
                            lattice[x, (y - 1) % L] +
                            lattice[x, (y + 1) % L])
        mu_f = -spin_f * (lattice[(x - 1) % L, y] +
                            lattice[(x + 1) % L, y] +
                            lattice[x, (y - 1) % L] +
                            lattice[x, (y + 1) % L])
        dE = J * (mu_f - mu_i) + h[t] * (-spin_f + spin_i) #negative sign included in mu_i and mu_f
        dE_eff = dE + E0[t]
        if np.random.rand() < 1.0/(1.0 + np.exp(beta * dE_eff)):
            lattice[x, y] = spin_f
            mu += (mu_f - mu_i)
        if truncate and t >= time - sample:
            idx = t - (time - sample)
            mus[idx] = mu
            magnetisations[idx] = lattice.sum() / lattice.size
        elif truncate and t < time - sample:
            # do not store the first time - sample steps
            pass
        else:
            mus[t] = mu
            magnetisations[t] = lattice.sum() / lattice.size
    return magnetisations, mus


#--------------------------------------------------------------------------
# High-level simulation driver functions
#--------------------------------------------------------------------------

def run_single_simulation(L, steps, J, bias=0.5, temperature=1.0, E0=None, h=None, algorithm='Metropolis'):
    """ 
    Run a single simulation of the chosen algorithm.
    
    Parameters:
      L (int): Lattice size (LxL).
      steps (int): Number of time steps to run the simulation.
      J (float): Coupling strength.
      sampleSize (int): Number of final steps to return.
      bias (float): Bias for lattice initialization.
      temperature (float): Temperature of the simulation.
      E0 (1D array of float): Energy offset per time step.
      algorithm (str): 'Metropolis' or 'Glauber'.
    
    Returns:
      tuple: (S_last, A_last, SNext_last, magnetisations_last, lattice)
             Each array is of length sampleSize.
    """
    lattice = initialise(L, bias=bias)
    mu_initial = get_mu(lattice)
    if E0 is not None:
        if np.isscalar(E0):
            E0_arr = np.full(steps, E0, dtype=np.float64)
        elif len(E0) != steps:
            print("Warning: E0 array length does not match the number of steps. Using constant E0[0].")
            E0_arr = np.full(steps, E0[0], dtype=np.float64)
        else:
            E0_arr = np.asarray(E0, dtype=np.float64)
    else:
        E0_arr = E0
    if h is not None:
        if np.isscalar(h):
            h_arr = np.full(steps, h, dtype=np.float64)
        elif len(h) != steps:
            print("Warning: h array length does not match the number of steps. Using constant h[0].")
            h_arr = np.full(steps, h[0], dtype=np.float64)
        else:
            h_arr = np.asarray(h, dtype=np.float64)
    else:
        h_arr = h
    if algorithm == 'Glauber':
        magnetisations, mus = glauber(lattice, steps, J, temperature, mu_initial, E0_arr, h_arr, truncate=True)
    else:
        magnetisations, mus = metropolis(lattice, steps, J, temperature, mu_initial, E0_arr, h_arr, truncate=True)
    return magnetisations, mus, lattice

def run_multi_simulation(L, steps, J, numSims, bias=0.5, temperature=1.0, E0=None, h=None, algorithm='Metropolis', n_jobs=-1):
    """
    Run multiple single simulations in parallel using joblib.
    
    Parameters:
      L (int): Lattice size.
      steps (int): Number of time steps per simulation.
      J (float): Coupling strength.
      numSims (int): Number of simulations.
      bias (float): Lattice initialization bias.
      temperature (float): Temperature of the simulation.
      E0 (1D array of float): Energy offset per time step.
      algorithm (str): 'Metropolis' or 'Glauber'.
      n_jobs (int): Number of parallel jobs (-1 uses all cores).
      
    Returns:
      List of numSims tuples (magnetisation, sum_interaction, lattice), each tuple contains the results of a single simulation. magnetisation and sum_interaction are numpy arrays of length MAX_STEPS (if truncate=True) or time (if truncate=False), lattice is a numpy array of shape (L, L).
    """
    results = Parallel(n_jobs=n_jobs)(
        delayed(run_single_simulation)(L, steps, J, bias, temperature, E0, h, algorithm) for _ in range(numSims)
    )
    return results

#--------------------------------------------------------------------------
# High-level analysis functions
#--------------------------------------------------------------------------

# Thermodynamic efficiency ====================================================
def compute_probability_distribution(lattice, n, m):
    """
    Compute the probability distribution of configurations in a given nxm area
    in a wrapped around square lattice.
    
    Parameters:
    lattice (np.ndarray): A 10x10 numpy array with values -1 or 1.
    n (int): The number of rows in the area.
    m (int): The number of columns in the area.
    
    Returns:
    dict: A dictionary with configurations as keys and their probabilities as values.
    """
    L = lattice.shape[0]
    count_dict = defaultdict(int)
    
    for i in range(L):
        for j in range(L):
            # Extract the nxm sub-lattice starting at (i, j)
            sub_lattice = tuple(tuple(lattice[(i + x) % L, (j + y) % L] for y in range(m)) for x in range(n))
            count_dict[sub_lattice] += 1

    # Compute the probability distribution
    total_counts = sum(count_dict.values())
    prob_distribution = {k: v / total_counts for k, v in count_dict.items()}
    
    return prob_distribution

def compute_entropy(prob_distribution):
    """
    Compute the entropy of a given probability distribution.
    
    Parameters:
    prob_distribution (dict): A dictionary with probabilities.
    
    Returns:
    float: The entropy value.
    """
    entropy = -sum(p * np.log2(p) for p in prob_distribution.values() if p > EPSILON)
    return entropy

def get_entropy_kikuchi(lattice):
    """ 
    Compute configuration entropy using kikuchi approximation S = S1-2*S2+S4.

    Parameters:
    lattice (np.array): A 2D integer array of the LxL squre lattice. Values +/-1.
    
    Returns:
    float: The entropy value.
    """
    # compute kikuchi approx 
    entp1 = compute_entropy(compute_probability_distribution(lattice, 1, 1))
    entp2 = compute_entropy(compute_probability_distribution(lattice, 1, 2))
    entp4 = compute_entropy(compute_probability_distribution(lattice, 2, 2))
    return entp1 - 2 * entp2 + entp4

def get_entropy_meanfield(pdf):
    """ 
    Compute configuration entropy using meanfield approximation S = sum(-1,1) -p*log(p).

    Parameters:
    pdf (dict): Proability distribution {spin:p(spin)}. This can be an average distribution computed over 
                a period of time and (or) mutltiple simulations.
    
    Returns:
    float: The entropy value.
    """
    # compute mean-field approx 
    return -sum(p * np.log2(p) for p in pdf.values() if p > EPSILON)
    
def get_fisher(pdf, method = 'sqrt'):
    """
    Compute fisher information using three different methods. Default is square-root approximation because it is more stable when p is small.

    Parameters:
    pdf (dict): A dictionary {theta: f(x)}, where f(x) is also a dictionary ({x:p(x)}). Density function takes the form f(x;theta), 
                and theta is the parameter with respect to which we compute the fisher information. Assume theta is uniformly spaced. 
                For continuous distribution make sure samples are binned and normalised.
    
    Returns:
    dict: Fisher information of different Js.
    """
    thetas = np.array(list(pdf.keys()))
    fisher = {}
    # dTheta = thetas[1] - thetas[0] # theta is uniformly spaced

    # Convert pdf dictionaries to arrays
    x_values = np.array(list(pdf[thetas[0]].keys()))
    p_values = np.array([[pdf[theta].get(x, 0) for x in x_values] for theta in thetas])

    if method == 'sqrt':
        # Compute the square root of p_values
        sqrt_p_values = np.sqrt(p_values)
    
        # Compute the gradient of sqrt_p_values with respect to theta
        dsqrtp_dtheta = np.gradient(sqrt_p_values, thetas, axis=0)
    
        # Compute Fisher Information for each theta
        for i, theta in enumerate(thetas):
            fisher[theta] = 4* np.sum(dsqrtp_dtheta[i]**2)
    elif method == 'inverse_p': 
        # using original sum_x {(dp_dtheta)^2 / p}
        # Compute the gradient of p_values with respect to theta. Use dp_dtheta is more stable than dlogp_dtheta when p is small
        dp_dtheta = np.gradient(p_values, thetas, axis=0)
        
        # Compute Fisher Information for each theta
        for i, theta in enumerate(thetas):
            # Avoid division by zero, sum over x
            valid_mask = p_values[i] > 0
            fisher[theta] = np.sum((dp_dtheta[i][valid_mask]**2) / p_values[i][valid_mask])
    else:
        # use E(log-likelihood derivative)
        # I_ij = sum_x( np.gradient(log_p(x;t),t_i) * np.gradient(log_p(x;t),t_j) * p(x;t) )
        log_p_values = np.log(p_values + EPSILON)  # add small epsilon to avoid log(0)
        dlogp_dtheta = np.gradient(log_p_values, thetas, axis=0)
        for i, theta in enumerate(thetas):
            fisher[theta] = np.sum(dlogp_dtheta[i]**2 * p_values[i])
    return fisher
    
def get_eta(configEntropy, fisherInfo, entropyFilt=False, derivativeFilt=False, fisherFilt=False, integralFilt=False, window=14, theta_star=None, threshold=None):
    """
    Compute thermodynamic efficiency. Interpolate from where fisherInfo < threshold to theta_star. Change return eta to np.array.
    
    Parameters:
    configEntropy (dict): A dictionary with parameter and configuration entropy.
    fisherInfo (dict): A dictionary with parameter and Fisher information.
    entropyFilt, derivativeFilt, fisherFilt, integralFilt (bool): switches for filters (Savitzky-Golay filter).
    window (int): window size for the filter.
    
    Returns:
    np.array: thermodynamic efficiency eta;
    np.array: configuration entropy values extracted from configEntropy, for checking;
    np.array: numerators for each eta, for checking;
    np.array: fisher information values extracted from fisherInfo, for checking;
    np.array: denominators for each eta, for checking.
    """
    theta = np.array(list(configEntropy.keys()))
    fisher = np.array(list(fisherInfo.values()))
    hx = np.array(list(configEntropy.values()))
    denominator = np.empty(len(hx)) # values to keep

    # Check if sum to theta_star is on:
    if (theta_star != None) and (theta_star > theta.max()):
        theta = np.append(theta, theta_star)
        fisher = np.append(fisher, 0)

        # interpolate between fi_small and fi_thetastar
        if (threshold != None) and (threshold > 0):
            # find the last index where fisher > threshold
            indices = np.where(fisher > threshold)[0]
            if indices.size == 0:
                # No value exceeds the threshold.
                start_index = 0  # or handle it in a way that suits your analysis
            else:
                start_index = indices[-1]

            x = np.append(theta[start_index], theta[-1]) # define starting and end point for interpolation
            y = np.append(fisher[start_index], fisher[-1]) # define starting and end point for interpolation
            interp_func = interp1d(x, y, kind='linear') # Create the interpolation function
            x_interp = theta[start_index:] # Define the range of x values to interpolate over
            y_interp = interp_func(x_interp) # Perform the interpolation
            fisher = np.append(fisher[:start_index], y_interp)

    # apply filters: entropy, fisher
    if entropyFilt:
        hx = savgol_filter(hx, window_length=window, polyorder=1)
    if fisherFilt:
        fisher = savgol_filter(fisher, window_length=window, polyorder=1)
    
    # compute numerator, denominator
    numerator = np.gradient(hx,theta[:len(hx)])
    for i in range(len(hx)):
        denominator[i] = np.trapezoid(fisher[i:], theta[i:]) # integrate from theta0 to point to zero-response 10
    
    # apply filters: derivative, integral
    if derivativeFilt:
        numerator = savgol_filter(numerator, window_length=window, polyorder=1)
    if integralFilt:
        denominator = savgol_filter(denominator, window_length=window, polyorder=1)
    # compute eta
    mask = (abs(denominator) < EPSILON) | (abs(numerator) < EPSILON)
    eta = np.zeros_like(numerator)
    eta[~mask] = -numerator[~mask] / denominator[~mask]
    return eta, hx, numerator, fisher, denominator

# Compute all intrinsic utilities ==========================================
def unpack_results(results):
    magnetisations_array = np.array([result[0] for result in results])
    interactions_array = np.array([result[1] for result in results])
    lattice_array = np.array([result[2].astype(int) for result in results])
    return magnetisations_array, interactions_array, lattice_array

def prep_data(results, subSample, method='kikuchi', absolute_m=True):
    """
    Compute all intrinsic utilities for a given j value.
    
    Parameters:
    results (list): A list of simulation results, where each result is a tuple containing the arrays of magnetisations (time series), sum_interactions (time series) and lattice (last snapshot of the simulation). 
    method (str): The method to use for computing configuration entropy (thermodynamic efficiency). Defaults to 'kikuchi'.
    absolute_m (bool): Whether to take the absolute value of magnetisation for Fisher Information (thermodynamic efficiency). Defaults to True.
    
    Returns:
    tuple: A tuple containing the mean values of configuration entropy, and average magnetisation across all simulations.
    """
    magnetisations_array, _, lattice_array = unpack_results(results)
    numSims = len(magnetisations_array) # each element in the array is a list of samples collected for one simulation. Length of the array is the number of simulations.
    hx = np.zeros(numSims) # configuration entropy for each simulation
    mm = np.zeros(numSims) # average magnetisation for each simulation
        
    for i in range(numSims):
        # average magnetisation each simulation
        # magnetisation is sampled every subSample time steps because it is highly correlated
        if absolute_m:
            mm[i] = np.mean(abs(magnetisations_array[i][::subSample]))
        else:
            mm[i] = np.mean(magnetisations_array[i][::subSample])
        # configuration entropy each simulation
        if method == 'kikuchi':
            hx[i] = get_entropy_kikuchi(lattice_array[i])
        else:   
            pdf = {1:(1 + mm[i]) / 2, -1:(1 - mm[i]) / 2}
            hx[i] = get_entropy_meanfield(pdf)
    
    return np.mean(hx), np.mean(mm)

def get_mean_var(results, L, subSample, absolute_m=True):
    magnetisations_array, interactions_array, _= unpack_results(results)
    numSims = len(magnetisations_array) # each element in the array is a list of samples collected for one simulation. Length of the array is the number of simulations.
    pooled_sum_s = []
    pooled_sum_ss = []
        
    for i in range(numSims):
        # average magnetisation each simulation
        # magnetisation is sampled every subSample time steps because it is highly correlated
        if absolute_m:
            pooled_sum_s.append(abs(magnetisations_array[i][::subSample]))
        else:
            pooled_sum_s.append(magnetisations_array[i][::subSample])
        # interaction energy
        pooled_sum_ss.append(interactions_array[i][::subSample])
    pooled_sum_s = np.array(pooled_sum_s).flatten()
    pooled_sum_ss = np.array(pooled_sum_ss).flatten()
    pooled_sum_s = pooled_sum_s * L * L # convert to total magnetisation
    pooled_sum_ss = -pooled_sum_ss # interaction terms includes negative sign from computing Hamiltonian, remove it here

    # compute mean and variance
    mean_sum_s = np.mean(pooled_sum_s)
    mean_sum_ss = np.mean(pooled_sum_ss)
    var_sum_s = np.var(pooled_sum_s)
    var_sum_ss = np.var(pooled_sum_ss)
    cov_s_ss = np.cov(pooled_sum_s, pooled_sum_ss)[0, 1]
    return mean_sum_s, mean_sum_ss, var_sum_s, var_sum_ss, cov_s_ss


#--------------------------------------------------------------------------
# High-level analysis functions (new in v1.1)
#--------------------------------------------------------------------------

def get_fisher_coarse(theta, avrg_magt, subSample=1):
    """
    Coarsen the data by a factor of subSample and re-compute the Fisher information.
    """
    theta_coarse = theta[::subSample]
    avrg_magt_coarse = avrg_magt[::subSample]
    pdf_dict = {}
    for j, m in zip(theta_coarse, avrg_magt_coarse):
        pdf_dict[j] = {1:(1 + m) / 2, -1:(1 - m) / 2} # p(+1) = (1 + M)/2, p(-1) = (1 - M)/2
    print(f'd_lambda={theta_coarse[1]-theta_coarse[0]:.3f}')
    fisher_info_dict = get_fisher(pdf_dict)
    return np.array(list(fisher_info_dict.keys())), np.array(list(fisher_info_dict.values()))

def get_eta_cov_form(thetas, cov_matrix, mean_x):
    """
    Calculate the thermodynamic efficiency using the covariance form. There are K parameters, each with N samples. 
    E.g. for Ising model with external field, thetas is an array of shape (N, K) where K=2 (Js and h0). First row of thetas is the parameter of interest. Similarly, first row of cov_matrix is the variance of the observable conjugate to the first parameter.
    Value of eta is smoothed by avoiding division near singularities. Similar treatment is used in the derivative form of eta.
    Parameters:
        thetas (np.ndarray): Array of shape (N, K) containing the parameters.
        cov_matrix (np.ndarray): Array of shape (N, K) containing the corresponding variance or covariance values.
        mean_x (float): Mean value of the observable conjugate to the first parameter.
    Returns:
        eta (np.ndarray): Array of thermodynamic efficiency values.
    """
    numerator = np.sum(thetas * cov_matrix, axis=1)
    denominator = mean_x
    # Avoid division near singularities
    mask = (abs(denominator) < EPSILON) | (abs(numerator) < EPSILON)
    eta = np.zeros_like(numerator)
    eta[~mask] = numerator[~mask] / denominator[~mask]
    
    # Avoid division near singularities
    mask = (abs(denominator) < EPSILON) | (abs(numerator) < EPSILON)
    eta = np.zeros_like(numerator)
    eta[~mask] = numerator[~mask] / denominator[~mask]
    return eta

# Compute the curve using cov form from data set
def compute_eta_cov(data, variable='J', subSample=1):
    """
    Compute the thermodynamic efficiency curve using the covariance form. Multi-parameter case with one parameter fixed.
    """
    if variable == 'J':
        h = data['h']
        Js = np.array(data['Js'])[::subSample]
        cov = np.array(data['cov_s_ss'])[::subSample]
        var = np.array(data['var_sum_ss'])[::subSample]
        mean = np.array(data['mean_sum_ss'])[::subSample]
        thetas = np.column_stack((Js, np.full_like(Js, h)))
    elif variable == 'h':
        J = data['J']
        hs = np.array(data['hs'])[::subSample]
        cov = np.array(data['cov_s_ss'])[::subSample]
        var = np.array(data['var_sum_s'])[::subSample]
        mean = np.array(data['mean_sum_s'])[::subSample]
        thetas = np.column_stack((hs, np.full_like(hs, J)))
    else:
        raise ValueError("Variable must be either 'h' or 'J'")
    cov_matrix = np.column_stack((var, cov))  # shape (N, K)
    eta = get_eta_cov_form(thetas, cov_matrix, mean)
    return eta

# Compute the curve using fisher form from data set
def compute_eta_fisher(data, variable='J', subSample=1):
    """
    Compute the thermodynamic efficiency curve using the fisher form. Multi-parameter case with one parameter fixed.
    """
    if variable == 'J':
        vars = np.array(data['Js'])
    elif variable == 'h':
        vars = np.array(data['hs'])
    else:
        raise ValueError("Variable must be either 'h' or 'J'")
    if subSample > 1:
        vars = vars[::subSample]
        avrg_magt = np.array(data['avrg_magt'])
        _, fisher_info = get_fisher_coarse(vars, avrg_magt, subSample=subSample)
        denominator = np.array(data['result_den'])[::subSample]
    else:
        fisher_info = np.array(data['result_fi'])[:-1]
        denominator = np.array(data['result_den'])

    eta = np.zeros_like(vars)
    mask = abs(denominator) < EPSILON
    eta[~mask] = vars[~mask]*fisher_info[~mask] / denominator[~mask]
    return eta


#--------------------------------------------------------------------------
# High-level analysis functions (new in v2)
#--------------------------------------------------------------------------
def get_fisher_matrix(p, d_theta, method = 'sqrt'):
    """
    Compute the Fisher information matrix for a given probability distribution p and its parameterization theta. Three method options. Partial derivatives wrt theta: dlog_p(xn;theta)/dtheta_k = np.gradient(np.log(p[:,:,n]), d_theta[k,:], axis=k). Assume uniform spacing for all parameters.

    Args:
        p (np.ndarray): Probability distribution (shape=(M1, M2, ..., Mk, N)). K is the number of parameters. Mk is the number of samples for kth parameter. N is the number of x values.
        d_theta (np.ndarray): Spacing of parameters ((K, ) array for uniformed spacing).
        method (str): Method for computing Fisher information ('sqrt', 'inverse_p', or 'log_likelihood').
        edge_order (int): Order of the finite difference approximation (default is 1).

    Returns:
        np.ndarray: Fisher information matrix (2D array).
    """
    k = int(d_theta.shape[0])
    if p.ndim < k:
        raise ValueError(f"p has {p.ndim} dims but d_theta implies {k} parameter axes.")
    axes = tuple(range(k))  # take gradient along the first k axes only
    fim_shape = p.shape[:-1] + (k, k) # for a given set of parameter, FIM is shape K*K. There are (M1, M2, ..., Mk) combinations of parameter values 
    fisher_info = np.zeros(fim_shape)  # Initialize the Fisher information matrix

    if method == 'sqrt':
        # I_ij = sum_x(2*np.gradient(sqrt_p(x;t),t_i) * 2*np.gradient(sqrt_p(x;t),t_j))
        sqrt_p = np.sqrt(p)
        grads_sqrt_p = np.gradient(sqrt_p, *d_theta.tolist(), axis=axes)
        # Normalize to a list when k==1 (np.gradient returns a single ndarray)
        if not isinstance(grads_sqrt_p, (list, tuple)):
            grads_sqrt_p = [grads_sqrt_p]  # length 1
        for i in range(k):
            for j in range(i, k):
                fisher_info[..., i, j] = 4 * np.sum(grads_sqrt_p[i] * grads_sqrt_p[j], axis=-1) # sum across all x, which is the last dim, shape (M1, M2, ..., Mk), i.e. the I_ij element for all different parameter combinations
                if i != j:
                    fisher_info[..., j, i] = fisher_info[..., i, j]  # FIM is symmetric
    elif method == 'inverse_p':
        # I_ij = sum_x( np.gradient(p(x;t),t_i) * np.gradient(p(x;t),t_j) * 1/p(x;t))
        grads_p = np.gradient(p, *d_theta.tolist(), axis=axes)
        # Normalize to a list when k==1 (np.gradient returns a single ndarray)
        if not isinstance(grads_p, (list, tuple)):
            grads_p = [grads_p]  # length 1
        for i in range(k):
            for j in range(i, k):
                fisher_info[..., i, j] = np.sum(grads_p[i] * grads_p[j] * 1/(p+EPSILON), axis=-1) # sum across all x, which is the last dim, shape (M1, M2, ..., Mk), i.e. the I_ij element for all different parameter combinations
                if i != j:
                    fisher_info[..., j, i] = fisher_info[..., i, j]  # FIM is symmetric
    else:
        # use E(log-likelihood derivative)
        # I_ij = sum_x( np.gradient(log_p(x;t),t_i) * np.gradient(log_p(x;t),t_j) * p(x;t) )
        log_p = np.log(p + EPSILON)  # add small epsilon to avoid log(0)
        grads_log_p = np.gradient(log_p, *d_theta.tolist(), axis=axes) # tuple of len=K, each element is an array shape=(M1, M2, ..., Mk, N)), representing dlogp/dtheta_i
        # Normalize to a list when k==1 (np.gradient returns a single ndarray)
        if not isinstance(grads_log_p, (list, tuple)):
            grads_log_p = [grads_log_p]  # length 1
        for i in range(k):
            for j in range(i,k):
                fisher_info[..., i, j] = np.sum(grads_log_p[i] * grads_log_p[j] * p, axis=-1) # sum across all x, which is the last dim, shape (M1, M2, ..., Mk), i.e. the I_ij element for all different parameter combinations
                if i != j:
                    fisher_info[..., j, i] = fisher_info[..., i, j]  # FIM is symmetric
    return fisher_info

def compute_fisher_integral(fim, hs, Js):
    """
    Compute the integral of Fisher Information Matrix (FIM) elements over h (dim=0) and J(dim=1). Use raw fisher information matrix without interpolation. 

    arguments:
        fim: Fisher Information Matrix (4D numpy array shape (M1, M2, 2, 2))
        hs: array of h values (len M1)
        Js: array of J values (len M2)

    returns:
        intg_I_hh: integrated I_{h,h} (2D numpy array)
        intg_I_JJ: integrated I_{J,J} (2D numpy array)
    """
    I_hh = fim[:, :, 0, 0]  # I_{h,h}
    I_JJ = fim[:, :, 1, 1]  # I_{J,J}
    intg_I_hh = np.full((len(hs), len(Js)), np.nan, dtype=float)
    intg_I_JJ = np.full((len(hs), len(Js)), np.nan, dtype=float)
    for i in range(len(hs)):
        for k in range(len(Js)):
            if abs(hs[0]) < abs(hs[-1]): # apply positive field, point of no work is on -1 end
                # print('positive')
                intg_I_hh[i, k] = np.trapezoid(I_hh[i:, k], hs[i:])  # integrate I_hh wrt h
                intg_I_JJ[i, k] = np.trapezoid(I_JJ[i, k:], Js[k:])  # integrate I_JJ wrt J
            else: # apply positive field, point of no work is on 0 end
                # print('negative')
                intg_I_hh[i, k] = np.trapezoid(I_hh[:i+1, k], hs[:i+1])  # integrate I_hh wrt h
                intg_I_JJ[i, k] = np.trapezoid(I_JJ[i, k:], Js[k:])  # integrate I_JJ wrt J
    return intg_I_hh, intg_I_JJ

def fisher_interpolate(fisher, thetas, theta_star=10, threshold=0.05):
    if theta_star < thetas[0]: #left most is point of no work (theta_star < 0)
        # find indices where fisher >= threshold, start interpolation from there to left most
        above = np.nonzero(fisher >= threshold)[0]
        if above.size != 0:
            # take the smallest index
            idx_bound = int(above[0]) + 1 # the new x,y has one extra point to the left
        else:
            idx_bound = -1 # always below threshold, start interpolate from right most
        x_vals = np.concatenate(([theta_star], thetas)) # add point of no work to the left
        y_vals = np.concatenate(([0.0], fisher)) # correspond to fisher=0

        x = np.append(x_vals[0], x_vals[idx_bound]) # define starting and end point for interpolation
        y = np.append(y_vals[0], y_vals[idx_bound]) # define starting and end point for interpolation
        interp_func = interp1d(x, y, kind='linear') # Create the interpolation function

        x_interp = x_vals[:idx_bound+1] # Define the range of x values to interpolate over
        y_interp = interp_func(x_interp) # Perform the interpolation
        fisher_smoothed = np.append(y_interp[1:], y_vals[idx_bound+1:]) # remove theta_star
    else: #right most is point of no work (theta_star > 0)
        above = np.nonzero(fisher >= threshold)[0]
        if above.size != 0:
            # take the largest index
            idx_bound = int(above[-1])
        else:
            idx_bound = 0 # always below threshold, start interpolate from left most
        x_vals = np.concatenate((thetas, [theta_star])) # add point of no work to the right
        y_vals = np.concatenate((fisher, [0.0])) # correspond to fisher=0
        x = np.append(x_vals[idx_bound], x_vals[-1]) # define starting and end point for interpolation
        y = np.append(y_vals[idx_bound], y_vals[-1]) # define starting and end point for interpolation
        interp_func = interp1d(x, y, kind='linear') # Create the interpolation function
        
        x_interp = x_vals[idx_bound:]
        y_interp = interp_func(x_interp) # Perform the interpolation
        fisher_smoothed = np.append(y_vals[:idx_bound+1], y_interp[1:-1]) # remove theta_star
    return fisher_smoothed

def compute_fisher_integral_smooth(fim, hs, Js, theta_star, threshold):
    I_hh = fim[:, :, 0, 0] # h is dim=0
    I_JJ = fim[:, :, 1, 1] # J is dim=1

    # check if theta_star is scalar
    if np.isscalar(theta_star):
        theta_star = np.array([theta_star, theta_star])
    if np.isscalar(threshold):
        threshold = np.array([threshold, threshold])

    I_hh_smoothed = np.full((len(hs), len(Js)), np.nan, dtype=float)
    I_JJ_smoothed = np.full((len(hs), len(Js)), np.nan, dtype=float)
    intg_I_hh = np.full((len(hs), len(Js)), np.nan, dtype=float)
    intg_I_JJ = np.full((len(hs), len(Js)), np.nan, dtype=float)

    # compute integral for h 
    for k in range(len(Js)):
        fisher = I_hh[:, k]  # take the k-th column
        thetas = hs
        # print(f'k={k}\n')
        if abs(hs[0]) < abs(hs[-1]):
            # apply positive field, point of no work is on -1 end
            fisher_smoothed = fisher_interpolate(fisher, thetas, theta_star=theta_star[0], threshold=threshold[0])
            for i in range(len(hs)): # integrate I_hh wrt h
                intg_I_hh[i, k] = np.trapezoid(np.append(fisher_smoothed[i:], 0), 
                                               np.append(thetas[i:], theta_star[0]))
        else: # apply positive field, point of no work is on 0 end
            fisher_smoothed = fisher_interpolate(fisher, thetas, theta_star=theta_star[0], threshold=threshold[0])
            for i in range(len(hs)): # integrate I_hh wrt h
                intg_I_hh[i, k] = np.trapezoid(np.append(0, fisher_smoothed[:i+1]), 
                                               np.append(theta_star[0], thetas[:i+1]))
        I_hh_smoothed[:, k] = fisher_smoothed

    # compute integral for J
    for i in range(len(hs)):
        fisher = I_JJ[i, :]  # take the i-th row
        thetas = Js
        # print(f'i={i}\n')
        if abs(Js[0]) < abs(Js[-1]):
            # apply positive coupling, point of no work is on -1 end
            fisher_smoothed = fisher_interpolate(fisher, thetas, theta_star=theta_star[1], threshold=threshold[1])
            for k in range(len(Js)): # integrate I_JJ wrt J
                intg_I_JJ[i, k] = np.trapezoid(np.append(fisher_smoothed[k:], 0), 
                                               np.append(thetas[k:], theta_star[1]))
        else: # apply negative coupling, point of no work is on 0 end
            fisher_smoothed = fisher_interpolate(fisher, thetas, theta_star=theta_star[1], threshold=threshold[1])
            for k in range(len(Js)): # integrate I_JJ wrt J
                intg_I_JJ[i, k] = np.trapezoid(np.append(0, fisher_smoothed[:k+1]), 
                                               np.append(theta_star[1], thetas[:k+1]))
        I_JJ_smoothed[i, :] = fisher_smoothed
    return intg_I_hh, intg_I_JJ, I_hh_smoothed, I_JJ_smoothed

def compute_eta2D_cov(data, subsample=[1,1]):
    hs = np.array(data['hs'])[::subsample[0]] # dimension 0
    Js = np.array(data['Js'])[::subsample[1]] # dimension 1
    mean_sum_s = np.array(data['mean_sum_s'])[::subsample[0], ::subsample[1]] # shape (len(hs), len(Js)), negative sign incorporated
    mean_sum_ss = np.array(data['mean_sum_ss'])[::subsample[0], ::subsample[1]] # shape (len(hs), len(Js)), negative sign incorporated
    var_sum_s = np.array(data['var_sum_s'])[::subsample[0], ::subsample[1]] # shape (len(hs), len(Js))
    var_sum_ss = np.array(data['var_sum_ss'])[::subsample[0], ::subsample[1]] # shape (len(hs), len(Js))
    cov_s_ss = np.array(data['cov_s_ss'])[::subsample[0], ::subsample[1]] # shape (len(hs), len(Js))
    eta_cov_h = (hs[:, None]*var_sum_s + Js[None, :]*cov_s_ss) / mean_sum_s
    eta_cov_J = (hs[:, None]*cov_s_ss + Js[None, :]*var_sum_ss) / mean_sum_ss
    return eta_cov_h, eta_cov_J

def compute_eta2D_derv(data, subsample=[1,1], interpolation=False, method='sqrt', theta_star=None, threshold=None):
    hs = np.array(data['hs'])[::subsample[0]] # dimension 0
    Js = np.array(data['Js'])[::subsample[1]] # dimension 1
    avrg_magt = np.array(data['avrg_magt'])[::subsample[0], ::subsample[1]] # shape (len(hs), len(Js))
    cnfg_entp = np.array(data['cnfg_entp'])[::subsample[0], ::subsample[1]] # shape (len(hs), len(Js))

    dh = hs[1]-hs[0]
    dJ = Js[1]-Js[0]
    dS = np.gradient(cnfg_entp, dh, dJ, axis=(0, 1))  # shape (len(hs), len(Js), 2)
    dS_h = dS[0]  # shape (len(hs), len(Js))
    dS_J = dS[1]  # shape (len(hs), len(Js))

    # fisher matrix
    p = np.zeros((len(hs), len(Js), 2))  # shape (M1, M2, 2)
    for i in range(len(hs)):
        for j in range(len(Js)):
            p[i, j, 0] = (1 + avrg_magt[i, j]) / 2 # p(+)
            p[i, j, 1] = (1 - avrg_magt[i, j]) / 2 # p(-)
    d_theta = np.array([dh, dJ])  # uniform spacing
    fim = get_fisher_matrix(p, d_theta, method=method)

    # fisher integral
    if interpolation:
        intg_I_hh, intg_I_JJ, _, _ = compute_fisher_integral_smooth(fim, hs, Js, theta_star, threshold)
        eta_derv_h = -dS_h / (intg_I_hh + EPSILON)
        eta_derv_J = -dS_J / (intg_I_JJ + EPSILON)
    else:
        intg_I_hh, intg_I_JJ = compute_fisher_integral(fim, hs, Js)
        eta_derv_h = -dS_h / (intg_I_hh + DENO_EPS)
        eta_derv_J = -dS_J / (intg_I_JJ + DENO_EPS)
    return eta_derv_h, eta_derv_J

def compute_eta2D_fisher(data, subsample, method='sqrt', interpolate=False, theta_star=None, threshold=None):
    hs = np.array(data['hs'])[::subsample[0]] # dimension 0
    Js = np.array(data['Js'])[::subsample[1]] # dimension 1
    avrg_magt = np.array(data['avrg_magt'])[::subsample[0], ::subsample[1]] # shape (len(hs), len(Js))
    dh = hs[1]-hs[0]
    dJ = Js[1]-Js[0]
    d_theta = np.array([dh, dJ])  # uniform spacing

    # fisher matrix (numerator)
    p = np.zeros((len(hs), len(Js), 2))  # shape (M1, M2, 2)
    for i in range(len(hs)):
        for j in range(len(Js)):
            p[i, j, 0] = (1 + avrg_magt[i, j]) / 2 # p(+)
            p[i, j, 1] = (1 - avrg_magt[i, j]) / 2 # p(-)
    fim = get_fisher_matrix(p, d_theta, method=method)

    I_hh = fim[:, :, 0, 0]  # I_{h,h}
    I_hJ = fim[:, :, 0, 1]  # I_{h,J}
    I_JJ = fim[:, :, 1, 1]  # I_{J,J}

    # integral fisher (denominator)
    if interpolate:
        intg_I_hh, intg_I_JJ, I_hh_smoothed, I_JJ_smoothed = compute_fisher_integral_smooth(fim, hs, Js, theta_star, threshold)
        # thermodynamic efficiency
        eta_fisher_h = (hs[:,None]*I_hh_smoothed + Js[None,:]*I_hJ) / (intg_I_hh + EPSILON)
        eta_fisher_J = (hs[:,None]*I_hJ + Js[None,:]*I_JJ_smoothed) / (intg_I_JJ + EPSILON)
    else:
        intg_I_hh, intg_I_JJ = compute_fisher_integral(fim, hs, Js)
        # thermodynamic efficiency
        eta_fisher_h = (hs[:,None]*I_hh + Js[None,:]*I_hJ) / (intg_I_hh + DENO_EPS)
        eta_fisher_J = (hs[:,None]*I_hJ + Js[None,:]*I_JJ) / (intg_I_JJ + DENO_EPS)
    return eta_fisher_h, eta_fisher_J

