"""
fit.py
Simple grid-search fitting for simulation parameters against experimental DACF and MSD

This script provides a minimal interface to fit simulation parameters (e.g., persistence,
chemotaxis_strength) by running short simulations and comparing DACF/MSD to experimental
measurements via reduced chi-squared.

"""

import numpy as np
import pandas as pd
from simulation import Simulation
from analysis import calculate_autocorrelation, calculate_msd, compute_msd_dacf_per_movie


def reduced_chi_squared(obs, sim, err=None):
    """
    Compute reduced chi-squared statistic between observed and simulated arrays.

    This measures the goodness-of-fit between simulation and experimental data,
    weighted by observational uncertainties. For DACF/MSD comparisons, we use
    the standard error of the mean (SEM) as it is statistically rigorous for averaged data.

    Parameters:
    -----------
    obs : array-like
        Observed (experimental) values
    sim : array-like
        Simulated values (same length as obs)
    err : array-like, optional
        Observational uncertainties (same length). If None, uses sqrt(obs) as heuristic.

    Returns:
    --------
    float : Reduced chi-squared value (chi^2 / degrees_of_freedom)
    """
    obs = np.asarray(obs)
    sim = np.asarray(sim)
    if err is None:
        # Simple heuristic: assume Poisson-like errors on obs
        err = np.sqrt(np.abs(obs) + 1e-6)
    chi2 = np.sum(((obs - sim) / err) ** 2)
    dof = len(obs) - 1 # degrees of freedom
    return chi2 / max(dof, 1)


def fit_grid(exp_df, simulation_mode, param_grid, chi_weight=0.5, velocity_dist_params=None, vonmises_params=None, n_steps=100, n_cells=30, seed=None):
    """
    Perform grid search to find best-fit simulation parameters against experimental data.

    This function systematically tests all combinations of parameters in the provided grid,
    running short simulations for each and comparing DACF and MSD to experimental data
    using reduced chi-squared statistics.

    Parameters:
    -----------
    exp_df : DataFrame
        Experimental trajectory data with columns: track_id, step, t, x, y, v_x, v_y, file
    simulation_mode : str
        Movement mode for simulation (e.g., 'persistent', 'exp_memory')
    param_grid : dict
        Dictionary mapping parameter names to lists of values to test
        e.g., {'persistence': [0.0, 0.5, 0.9], 'chemotaxis_strength': [0.0, 0.3]}
    chi_weight : float
        Weight for combining chi-squared values: score = chi_weight*chi2_dacf + (1-chi_weight)*chi2_msd
    velocity_dist_params : dict, optional
        Log-normal velocity parameters: {'shape': , 'loc': , 'scale': }
    vonmises_params : dict, optional
        Von Mises mixture parameters: {'kappa1': , 'kappa2': , 'W1': }
    n_steps : int
        Number of simulation steps per trial (default: 100)
    n_cells : int
        Number of cells to simulate (default: 30)
    seed : int, optional
        Random seed for reproducibility

    Returns:
    --------
    tuple : (best_params_dict, results_dataframe)
        - best_params_dict: Dictionary of best-fit parameters with scores
        - results_dataframe: DataFrame with all tested combinations sorted by score
    """
    if seed is not None:
        np.random.seed(seed)

    # Calculate experimental statistics per position
    exp_msd, exp_dacf = compute_msd_dacf_per_movie(exp_df, max_lag=None)
    
    # cut dacf at the first negative value to avoid noise domination
    ival = np.where(exp_dacf['dacf_mean'].values < 0)[0]
    if len(ival) > 0:
        exp_dacf = exp_dacf.iloc[:ival[0]]

    import itertools
    keys = list(param_grid.keys())
    combos = list(itertools.product(*[param_grid[k] for k in keys]))

    results = []
    for combo in combos:
        params = dict(zip(keys, combo))

        sim = Simulation(n_cells=n_cells,
                         time_step=5,
                         stadium_L=800,
                         stadium_R=200,
                         source_length=400,
                         chemotaxis_strength=params.get('chemotaxis_strength', 0.0),
                         repulsion_strength=params.get('repulsion_strength', 0.0),
                         interaction_radius=params.get('interaction_radius', 0.0),
                         velocity_params=velocity_dist_params,
                         markov_chain=params.get('markov_chain', None),
                         persistence=params.get('persistence', 0.0),
                         starting_positions=params.get('starting_positions', 'uniform'),
                         memory_window=params.get('memory_window', None),
                         memory_exp_lambda=params.get('memory_exp_lambda', 0.1),
                         memory_power_alpha=params.get('memory_power_alpha', 2.0),
                         mode=simulation_mode,
                         vonmises_params=vonmises_params)

        sim.run(n_steps=n_steps, verbose=False)
        sim_df = sim.get_dataframe()

        sim_dacf = calculate_autocorrelation(sim_df, max_lag=None, directional=True)
        sim_msd = calculate_msd(sim_df, max_lag=None)

        # Use SEM from position-to-position variability
        l = min(len(exp_dacf), len(sim_dacf))
        chi2_dacf = reduced_chi_squared(
            exp_dacf['dacf_mean'].values[1:l], 
            sim_dacf['dacf'].values[1:l],
            exp_dacf['dacf_sem'].values[1:l]  # Use SEM
        )

        l2 = min(len(exp_msd), len(sim_msd))
        chi2_msd = reduced_chi_squared(
            exp_msd['msd_mean'].values[:l2], 
            sim_msd['msd'].values[:l2],
            exp_msd['msd_sem'].values[:l2]  # Use SEM 
        )

        score = chi_weight * chi2_dacf + (1-chi_weight) * chi2_msd

        results.append({**params, 'chi2_dacf': chi2_dacf, 'chi2_msd': chi2_msd, 'score': score})

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('score')

    best = results_df.iloc[0].to_dict()
    return best, results_df


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Fit simulation to experimental data (simple grid-search)')
    parser.add_argument('--data', required=True, help='Path to experimental CSV')
    parser.add_argument('--n_steps', type=int, default=100)
    args = parser.parse_args()

    exp_df = pd.read_csv(args.data)
    grid = {
        'persistence': [0.0, 0.2, 0.5],
        'chemotaxis_strength': [0.0, 0.1, 0.3],
        'repulsion_strength': [0.0]
    }

    best, table = fit_grid(exp_df, grid, n_steps=args.n_steps, n_cells=30, seed=123)
    print('Best parameters:')
    print(best)
    table.to_csv('fit_results.csv', index=False)
    print('Results saved to fit_results.csv')
