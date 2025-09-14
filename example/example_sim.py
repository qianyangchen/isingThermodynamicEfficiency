from ising2D.core import run_multi_simulation, prep_data, get_mean_var
import numpy as np
import os

# Parallel processing allocation: 
## HPC utilises 480 cpus (WORLD_SIZE = 10 nodes). Each node opens one Python process (in PBS script, "mpirun -np ${NODES} --map-by ppr:1:node").
## Each node gets an equal share of the (h,J) values. Since each HPC node has 48 cpus, use n_jobs=48 such that each cpu deals with one worker. For a given pair of (h,J), run run_multi_simulation spawns the job (i.e. 500 simulations) to 48 workers and they run in parallel.
## The 10 nodes also run in parallel.

if __name__ == '__main__':
    # for parallel processing
    def shard(lst, w, r):
        n = len(lst)
        base, extra = divmod(n, w)
        start = r * base + min(r, extra)
        end   = start + base + (1 if r < extra else 0)
        return lst[start:end]
    
    # parameters for simulation
    L = 50 # 50
    time = 40_400_000 # perform 40.4mil flips
    Js = np.linspace(0.01,1,20) # np.linspace(0.01,1,20)
    E0 = None # no energy bias 0.0
    hs = np.linspace(-0.6, 0.0,20)  # np.linspace(-0.6,0.0,20)
    temperature = 1
    n_jobs = 48 # 48
    algorithm = 'Metropolis'
    sampleSize = 200_000 # take 200k observations to compute eta, defined in noneq_core.py MAX_STEPS
    subSample = L*L # sample every sweep for magnetisation
    numSims = 500 # number of simulations 500
    bias = 1
    filterWindow = 5

    # parameters for computing intrinsic utilities
    absolute_m = True # use absolute value of magnetisation for fisher information
    theta_star = 10 # theta_star for thermodynamic efficiency upper bound
    threshold = 5e-2 # threshold for interpolation in fisher information integral

    # CHANGE THIS SETTINGS ACCORDINGLY: parallel processing scaffolding
    cells = [(hi, kj, h, J) for hi, h in enumerate(hs) for kj, J in enumerate(Js)]
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank       = int(os.environ.get("RANK", "0"))
    my_cells = shard(cells, world_size, rank)

    # quantities to collect
    H, K = len(hs), len(Js)
    avrg_magt_2D    = np.full((H, K), np.nan, dtype=float) # average magnetisation
    cnfg_entp_2D    = np.full((H, K), np.nan, dtype=float) # configuration entropy
    mean_sum_s_2D   = np.full((H, K), np.nan, dtype=float) # mean sum of si
    mean_sum_ss_2D  = np.full((H, K), np.nan, dtype=float) # mean sum of si*sj
    var_sum_s_2D    = np.full((H, K), np.nan, dtype=float) # variance of sum of si
    var_sum_ss_2D   = np.full((H, K), np.nan, dtype=float) # variance of sum of si*sj
    cov_s_ss_2D     = np.full((H, K), np.nan, dtype=float) # covariance of sum of si and sum of si*sj

    # run simulations for different J and h values, save result on a 2D grid Jxh
    for hi, kj, h, J in my_cells:
            # run simulations for each J
            results = run_multi_simulation(L, time, J, numSims, bias=bias, temperature=temperature, E0=E0, h=h, algorithm=algorithm, n_jobs=n_jobs)
            # for derivative or Fisher form
            cnfg_entp_2D[hi, kj], avrg_magt_2D[hi, kj] = prep_data(results, subSample, method='kikuchi', absolute_m=absolute_m)
            # for covariance form
            mean_sum_s_2D[hi, kj], mean_sum_ss_2D[hi, kj], var_sum_s_2D[hi, kj], var_sum_ss_2D[hi, kj], cov_s_ss_2D[hi, kj] = get_mean_var(results, L, subSample, absolute_m=True)

    # CHANGE THIS SETTINGS ACCORDINGLY: Save to a shared path ($PBS_O_WORKDIR, which #PBS -l wd sets as .). Don’t save to $PBS_JOBFS unless you copy back. Stitch together results via merge_shards.py
    out_dir = os.environ.get("OUT_DIR", os.environ.get("PBS_O_WORKDIR", "."))
    os.makedirs(out_dir, exist_ok=True)

    np.savez_compressed(
        os.path.join(out_dir, f"grid_rank{rank:03d}.npz"),
        # payload
        avrg_magt_2D=avrg_magt_2D,
        cnfg_entp_2D=cnfg_entp_2D,
        mean_sum_s_2D=mean_sum_s_2D,
        mean_sum_ss_2D=mean_sum_ss_2D,
        var_sum_s_2D=var_sum_s_2D,
        var_sum_ss_2D=var_sum_ss_2D,
        cov_s_ss_2D=cov_s_ss_2D,
        # helpful metadata (redundant but useful for checks)
        hs=np.asarray(hs),
        Js=np.asarray(Js),
        L=np.int64(L),
        time=np.int64(time),
        numSims=np.int64(numSims),
        temperature=float(temperature),
        bias=float(bias),
        algorithm=str(algorithm),
        subSample=np.int64(subSample),
    )