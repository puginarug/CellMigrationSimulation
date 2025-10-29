import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import i0
from scipy.optimize import minimize


def calculate_autocorrelation(df, max_lag=None, directional=True):
    """
    Calculate velocity autocorrelation function.
    
    Parameters:
    -----------
    df : DataFrame
        Trajectory data with columns: track_id, step, v_x, v_y
    directional : bool
        If True, compute directional autocorrelation (DACF)
        If False, compute velocity autocorrelation (VACF)
    
    Returns:
    --------
    DataFrame with autocorrelation values
    """
    # Pivot to wide format
    df_vx = df.pivot(index='track_id', columns='step', values='v_x')
    df_vy = df.pivot(index='track_id', columns='step', values='v_y')
    Vx = df_vx.values
    Vy = df_vy.values

    n_particles, n_steps = Vx.shape
    acorr_vals = np.empty(n_steps, dtype=float)
    acorr_stds = np.empty(n_steps, dtype=float)
    acorr_sems = np.empty(n_steps, dtype=float)  # Add SEM
    n_samples = np.empty(n_steps, dtype=int)      # Add sample count

    for dt in range(n_steps):
        v1x = Vx[:, :n_steps-dt]
        v2x = Vx[:, dt:]
        v1y = Vy[:, :n_steps-dt]
        v2y = Vy[:, dt:]

        dot = v1x*v2x + v1y*v2y
        if max_lag is not None and dt >= max_lag:
            break
        if directional:
            # Normalize by magnitudes for DACF
            mag1 = np.sqrt(v1x**2 + v1y**2)
            mag2 = np.sqrt(v2x**2 + v2y**2)
            valid_mask = (mag1 > 0) & (mag2 > 0)
            norm_dot = np.full(dot.shape, np.nan)
            norm_dot[valid_mask] = dot[valid_mask] / (mag1[valid_mask] * mag2[valid_mask])
            
            # Count valid samples
            n_valid = np.sum(~np.isnan(norm_dot))
            n_samples[dt] = n_valid
            
            acorr_vals[dt] = np.nanmean(norm_dot)
            acorr_stds[dt] = np.nanstd(norm_dot)
            acorr_sems[dt] = acorr_stds[dt] / np.sqrt(n_valid) if n_valid > 0 else np.nan
            column_name = 'dacf'
        else:
            # VACF without normalization
            n_valid = np.sum(~np.isnan(dot))
            n_samples[dt] = n_valid
            
            acorr_vals[dt] = np.nanmean(dot)
            acorr_stds[dt] = np.nanstd(dot)
            acorr_sems[dt] = acorr_stds[dt] / np.sqrt(n_valid) if n_valid > 0 else np.nan
            column_name = 'vacf'

    # Only keep up to max_lag if specified
    if max_lag is not None:
        acorr_vals = acorr_vals[:max_lag]
        acorr_stds = acorr_stds[:max_lag]
        acorr_sems = acorr_sems[:max_lag]
        n_samples = n_samples[:max_lag]
        lags = np.arange(max_lag)
    else:
        lags = np.arange(n_steps)

    acorr_df = pd.DataFrame({
        column_name: acorr_vals,
        f'{column_name}_std': acorr_stds,
        f'{column_name}_sem': acorr_sems,      # Add SEM column
        'n': n_samples,                 # Add sample count per lag
        'lag': lags
    })
    
    # Calculate time resolution from the actual data
    time_steps = df.groupby('step')['t'].first().sort_index()
    time_res = time_steps.iloc[1] - time_steps.iloc[0]

    acorr_df['dt'] = acorr_df['lag'] * time_res  # Convert to minutes

    return acorr_df


def calculate_msd(df, x_col='x', y_col='y', max_lag=None):
    """
    Calculate Mean Squared Displacement.
    
    Parameters:
    -----------
    df : DataFrame
        Trajectory data with columns: track_id, step, x, y
    max_lag : int, optional
        Maximum lag to compute MSD for
    
    Returns:
    --------
    DataFrame with MSD values
    """

    tracks = df.groupby('track_id')
    n_steps = df.groupby('track_id').size().max()
    
    if max_lag is None:
        max_lag = n_steps - 1
    else:
        max_lag = min(max_lag, n_steps - 1)
    
    msd_values = [[] for _ in range(max_lag)]
    
    for track_id, track in tracks:
        x = track[x_col].values
        y = track[y_col].values
        n = len(x)
        
        for lag in range(1, min(n, max_lag + 1)):
            dx = x[lag:] - x[:-lag]
            dy = y[lag:] - y[:-lag]
            squared_disp = dx**2 + dy**2
            msd_values[lag-1].extend(squared_disp)
    
    msd_mean = np.array([np.mean(vals) if len(vals) > 0 else np.nan for vals in msd_values])
    msd_std = np.array([np.std(vals) if len(vals) > 0 else np.nan for vals in msd_values])
    msd_sem = np.array([np.std(vals)/np.sqrt(len(vals)) if len(vals) > 0 else np.nan 
                        for vals in msd_values])
    n_samples = np.array([len(vals) for vals in msd_values])
    
    time_steps = df.groupby('step')['t'].first().sort_index()
    time_res = time_steps.iloc[1] - time_steps.iloc[0]

    msd_df = pd.DataFrame({
        'msd': msd_mean,
        'msd_std': msd_std,
        'msd_sem': msd_sem,  # Add SEM
        'n': n_samples,  # Add sample size per lag
        'lag': np.arange(1, max_lag + 1),
        'dt': np.arange(1, max_lag + 1) * time_res
    })
    
    return msd_df


def plot_dacf(dacf_df, title='Directional Autocorrelation Function'):
    """Plot DACF."""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    ax.plot(dacf_df['lag'], dacf_df['dacf'], 'o-', linewidth=2, markersize=6)
    ax.fill_between(dacf_df['lag'], dacf_df['dacf'] - dacf_df['dacf_std'], dacf_df['dacf'] + dacf_df['dacf_std'], alpha=0.2)
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax.set_xlabel('Time Lag (steps)', fontsize=12)
    ax.set_ylabel('DACF', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.2, 1.0)
    
    plt.tight_layout()
    plt.show()
    
    return fig, ax


def plot_msd(msd_df, title='Mean Squared Displacement'):
    """Plot MSD with reference lines."""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot MSD
    ax.plot(msd_df['lag'], msd_df['msd'], 'o-', linewidth=2, markersize=6, label='MSD')
    ax.fill_between(msd_df['lag'], msd_df['msd'] - msd_df['msd_std'], msd_df['msd'] + msd_df['msd_std'], alpha=0.2)
    
    # Add reference lines
    lags = msd_df['lag'].values[1:]  # Skip lag=0
    if len(lags) > 0:
        # Ballistic motion (∝ t²)
        ballistic = lags[0]**2 * (msd_df['msd'].iloc[1] / lags[0]**2)
        ax.plot(lags, ballistic * (lags / lags[0])**2, 'r--', alpha=0.5, label='Ballistic (∝t²)')
        
        # Diffusive motion (∝ t)
        diffusive = lags[0] * (msd_df['msd'].iloc[1] / lags[0])
        ax.plot(lags, diffusive * (lags / lags[0]), 'g--', alpha=0.5, label='Diffusive (∝t)')
    
    ax.set_xlabel('Time Lag (steps)', fontsize=12)
    ax.set_ylabel('MSD (μm²)', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Log scale can be helpful
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    plt.tight_layout()
    plt.show()
    
    return fig, ax


def compare_simulations(sim_df, exp_df=None, max_lag=30):
    """
    Compare simulation with experimental data (if provided).
    
    Parameters:
    -----------
    sim_df : DataFrame
        Simulation trajectory data
    exp_df : DataFrame
        Experimental trajectory data (optional)
    max_lag : int
        Maximum lag for analysis
    """
    # Calculate metrics for simulation
    sim_dacf = calculate_autocorrelation(sim_df, max_lag, directional=True)
    sim_msd = calculate_msd(sim_df, max_lag)
    
    if exp_df is not None:
        # Calculate metrics for experimental data
        exp_dacf = calculate_autocorrelation(exp_df, max_lag, directional=True)
        exp_msd = calculate_msd(exp_df, max_lag)
        
        # Create comparison plots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # DACF comparison (scatter plots)
        ax1.scatter(exp_dacf['lag'], exp_dacf['dacf'], marker='o', label='Experimental', alpha=0.7)
        ax1.scatter(sim_dacf['lag'], sim_dacf['dacf'], marker='s', label='Simulation', alpha=0.7)
        ax1.set_xscale('log')
        ax1.set_yscale('log')
        ax1.set_xlabel('Time Lag (steps)')
        ax1.set_ylabel('DACF')
        ax1.set_title('Directional Autocorrelation Comparison')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # MSD comparison (scatter plots)
        ax2.scatter(exp_msd['lag'][1:], exp_msd['msd'][1:], marker='o', label='Experimental', alpha=0.7)
        ax2.scatter(sim_msd['lag'][1:], sim_msd['msd'][1:], marker='s', label='Simulation', alpha=0.7)
        ax2.set_xscale('log')
        ax2.set_yscale('log')
        ax2.set_xlabel('Time Lag (steps)')
        ax2.set_ylabel('MSD (μm²)')
        ax2.set_title('Mean Squared Displacement Comparison')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        return {
            'sim_dacf': sim_dacf,
            'sim_msd': sim_msd,
            'exp_dacf': exp_dacf,
            'exp_msd': exp_msd
        }
    else:
        # Plot simulation results
        plot_dacf(sim_dacf, 'Simulation DACF')
        plot_msd(sim_msd, 'Simulation MSD')
        
        return {
            'sim_dacf': sim_dacf,
            'sim_msd': sim_msd
        }   
    

def compute_msd_dacf_per_movie(df, x_col='x', y_col='y', max_lag=None):
    """
    Compute MSD and DACF per movie, then aggregate across movies with n_obs weighting.
    The input DataFrame must have a 'file' column representing the movie ID.
    """

    msd_results = []
    dacf_results = []

    # --- Loop over each movie ---
    for movie_id, df_movie in df.groupby("file"):
        try:
            # Calculate MSD for this movie
            msd_df = calculate_msd(df_movie, x_col=x_col, y_col=y_col, max_lag=max_lag)
            msd_df["file"] = movie_id
            msd_results.append(msd_df)

            # Calculate DACF for this movie
            dacf_df = calculate_autocorrelation(df_movie, max_lag=max_lag, directional=True)
            dacf_df["file"] = movie_id
            dacf_results.append(dacf_df)
        except Exception as e:
            print(f"⚠️ Skipped movie {movie_id} due to error: {e}")

    # --- Combine results ---
    msd_all = pd.concat(msd_results, ignore_index=True)
    dacf_all = pd.concat(dacf_results, ignore_index=True)

    # --- Weighted average by number of observations per lag ---
    def weighted_stats(group):
        # Check if 'n' column exists and has valid weights
        if 'n' not in group.columns or group['n'].isna().all() or group['n'].sum() == 0:
            # Fallback to unweighted if 'n' column not available or all zeros
            return pd.Series({
                'mean': group['value'].mean(),
                'std': group['value'].std(),
                'sem': group['value'].std() / np.sqrt(len(group)),
                'dt': group['dt'].iloc[0],
                'n_total': len(group)
            })
        
        # Filter out invalid weights
        valid_mask = (group['n'] > 0) & (~group['n'].isna())
        if not valid_mask.any():
            # No valid weights, use unweighted
            return pd.Series({
                'mean': group['value'].mean(),
                'std': group['value'].std(),
                'sem': group['value'].std() / np.sqrt(len(group)),
                'dt': group['dt'].iloc[0],
                'n_total': 0
            })
        
        weights = group.loc[valid_mask, 'n'].values
        values = group.loc[valid_mask, 'value'].values
        
        # Weighted mean
        weighted_mean = np.average(values, weights=weights)
        
        # Weighted variance
        weighted_var = np.average((values - weighted_mean)**2, weights=weights)
        weighted_std = np.sqrt(weighted_var)
        
        # Weighted SEM (using effective sample size)
        n_eff = weights.sum()**2 / (weights**2).sum()
        weighted_sem = weighted_std / np.sqrt(n_eff)
        
        return pd.Series({
            'mean': weighted_mean,
            'std': weighted_std,
            'sem': weighted_sem,
            'dt': group['dt'].iloc[0],
            'n_total': weights.sum()
        })
    
    # Apply weighted aggregation
    msd_all_renamed = msd_all.rename(columns={'msd': 'value'})
    msd_summary = msd_all_renamed.groupby("lag").apply(weighted_stats).reset_index()
    msd_summary.columns = ['lag', 'msd_mean', 'msd_std', 'msd_sem', 'dt', 'n_total']
    
    dacf_all_renamed = dacf_all.rename(columns={'dacf': 'value'})
    dacf_summary = dacf_all_renamed.groupby("lag").apply(weighted_stats).reset_index()
    dacf_summary.columns = ['lag', 'dacf_mean', 'dacf_std', 'dacf_sem', 'dt', 'n_total']

    return msd_summary, dacf_summary


def compute_turning_angles(df, track_col='track_id', x_col='x_microns', y_col='y_microns', step_col='step', lag=1):
    """
    Compute signed turning angles (radians) between consecutive movement vectors for each track.
    Angles are in range (-pi, pi), where positive means counterclockwise turn.
    """
    if not {track_col, x_col, y_col, step_col}.issubset(df.columns):
        raise ValueError(f"Input DataFrame must contain columns: {track_col}, {x_col}, {y_col}, {step_col}")

    df_sorted = df.sort_values([track_col, step_col]).copy()
    all_angles = []

    for _, track_df in df_sorted.groupby(track_col):
        x = track_df[x_col].values
        y = track_df[y_col].values

        dx = np.diff(x)
        dy = np.diff(y)

        for i in range(lag, len(dx)):
            v1 = np.array([dx[i - lag], dy[i - lag]])
            v2 = np.array([dx[i], dy[i]])
            norm1 = np.linalg.norm(v1)
            norm2 = np.linalg.norm(v2)
            if norm1 == 0 or norm2 == 0:
                continue

            # Normalize
            v1 /= norm1
            v2 /= norm2

            # Compute signed angle using atan2 of cross and dot products
            cross = v1[0]*v2[1] - v1[1]*v2[0]
            dot = np.dot(v1, v2)
            angle = np.arctan2(cross, dot)
            all_angles.append(angle)

    return all_angles

def von_mises_mixture_no_amplitudes(x, W1, kappa1, kappa2):
    """
    Mixture of two von Mises distributions centered at 0 without amplitude parameters.
    W1: weight of the first component (0..1)
    kappa1, kappa2: concentration parameters (>=0)
    """
    vm1 = np.exp(kappa1 * np.cos(x)) / (2 * np.pi * i0(kappa1))
    vm2 = np.exp(kappa2 * np.cos(x)) / (2 * np.pi * i0(kappa2))
    return W1 * vm1 + (1.0 - W1) * vm2


def fit_von_mises_mixture_mle(x_data, initial_guess=(0.5, 1.0, 5.0)):
    """
    Fit a mixture of two von Mises distributions (centered at 0) to raw angular data via MLE.

    Parameters
    ----------
    x_data : array_like
        Sample of angles (in radians).
    initial_guess : tuple, optional
        Initial guess for parameters (W1, kappa1, kappa2).

    Returns
    -------
    result : dict
        Dictionary containing fitted parameters, optimization success, and message.
    """

    # Ensure numpy array
    x_data = np.asarray(x_data)
    # Normalize to [-pi, pi)
    x_data = np.mod(x_data + np.pi, 2*np.pi) - np.pi

    # Define the negative log-likelihood
    def neg_log_likelihood(params):
        W1, kappa1, kappa2 = params
        # Enforce constraints
        if not (0 < W1 < 1 and kappa1 >= 0 and kappa2 >= 0):
            return np.inf
        
        # Compute mixture PDF
        pdf_vals = von_mises_mixture_no_amplitudes(x_data, W1, kappa1, kappa2)
        
        # Avoid log(0)
        pdf_vals = np.clip(pdf_vals, 1e-12, None)
        
        # Negative log-likelihood
        return -np.sum(np.log(pdf_vals))

    # Optimize using bounded minimization
    bounds = [(1e-3, 1-1e-3), (0, None), (0, None)]
    res = minimize(neg_log_likelihood, x0=initial_guess, bounds=bounds)

    # Collect results
    fit_params = {"W1": res.x[0], "kappa1": res.x[1], "kappa2": res.x[2]}
    
    return {
        "params": fit_params,
        "success": res.success,
        "message": res.message,
        "nll": res.fun,
    }


# Example usage
if __name__ == "__main__":
    np.random.seed(0)
    # Generate synthetic mixture data
    n = 2000
    W_true, k1_true, k2_true = 0.3, 2.0, 10.0
    comps = np.random.rand(n) < W_true
    x_data = np.where(
        comps,
        np.random.vonmises(0, k1_true, size=n),
        np.random.vonmises(0, k2_true, size=n)
    )

    result = fit_von_mises_mixture_mle(x_data)
    print(result)