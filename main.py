"""
run_simulation.py
Main script to run the simplified cell migration simulation
"""

import numpy as np
import pandas as pd
from scipy.stats import lognorm
import matplotlib.pyplot as plt

from cell import Cell
from stadium import VerticalStadium
from simulation import Simulation
from analysis import calculate_dacf, calculate_msd, compare_simulations
from visualization import plot_trajectories, create_animation, plot_cell_statistics


def load_experimental_data(filepath):
    """
    Load and preprocess experimental trajectory data from CSV file.

    Parameters:
    -----------
    filepath : str
        Path to CSV file containing experimental data

    Returns:
    --------
    tuple : (DataFrame, list of tracks)
        - DataFrame with columns: track_id, step, t, x, y, v_x, v_y
        - List of numpy arrays containing (x, y) coordinates for each track
    """
    print(f"Loading experimental data from {filepath}...")
    
    data = pd.read_csv(filepath)
    
    # Rename columns if needed
    if 'normalized_time' in data.columns:
        data = data.rename(columns={
            'normalized_time': 'step',
            'x_microns': 'x',
            'y_microns': 'y'
        })
    
    # Filter short tracks
    data = data.groupby('track_id').filter(lambda g: len(g) >= 10)
    
    # Calculate velocities
    data['v_x'] = data.groupby('track_id')['x'].diff()
    data['v_y'] = data.groupby('track_id')['y'].diff()
    
    # Extract tracks
    tracks = [group[['x', 'y']].values for _, group in data.groupby('track_id')]
    
    print(f"Loaded {len(tracks)} tracks")
    
    return data, tracks


def fit_velocity_distribution(data):
    """
    Fit log-normal distribution to experimental velocity magnitudes.

    Parameters:
    -----------
    data : DataFrame
        Trajectory data with v_x and v_y columns

    Returns:
    --------
    dict : Log-normal parameters {'shape': , 'loc': , 'scale': }
    """
    velocities = np.sqrt(data['v_x']**2 + data['v_y']**2)
    velocities = velocities.dropna()
    velocities = velocities[velocities > 0]
    
    shape, loc, scale = lognorm.fit(velocities, floc=0)
    
    print(f"Velocity parameters: shape={shape:.3f}, loc={loc:.3f}, scale={scale:.3f}")
    
    return {'shape': shape, 'loc': loc, 'scale': scale}


def run_example(experimental_data_path=None):
    """
    Run a complete example simulation with visualization and analysis.

    This function demonstrates the full workflow: loading experimental data (optional),
    fitting velocity distributions, running simulation, analyzing results (DACF, MSD),
    creating visualizations and animations, and saving outputs.

    Parameters:
    -----------
    experimental_data_path : str, optional
        Path to experimental CSV data for comparison

    Returns:
    --------
    tuple : (Simulation object, results dictionary)
    """
    
    print("="*60)
    print("SIMPLIFIED CELL MIGRATION SIMULATION")
    print("Vertical Stadium with Line Source Gradient")
    print("="*60)
    
    # Load experimental data if provided
    velocity_params = {'shape': 0.5, 'loc': 0, 'scale': 1.0}
    exp_data = None
    
    if experimental_data_path:
        try:
            exp_data, exp_tracks = load_experimental_data(experimental_data_path)
            
            # Fit velocity distribution
            velocity_params = fit_velocity_distribution(exp_data)
              
        except Exception as e:
            print(f"Could not load experimental data: {e}")
            print("Using default parameters...")
    
    # Create and run simulation
    print("\nInitializing simulation...")
    print("Stadium parameters:")
    print(f"  L (wall length): 60 μm")
    print(f"  R (semicircle radius): 20 μm")
    print(f"  Source length: 40 μm")
    
    sim = Simulation(
        n_cells=30,
        stadium_L=60,
        stadium_R=20,
        source_length=40,
        chemotaxis_strength=0.3,
        repulsion_strength=0.2,
        interaction_radius=10.0,
        velocity_params=velocity_params,
    )
    
    # Visualize initial state
    print("\nVisualizing initial configuration...")
    sim.stadium.visualize(sim.cells)
    
    # Run simulation
    print("\nRunning simulation...")
    sim.run(n_steps=100, verbose=True)
    
    # Get results
    df = sim.get_dataframe()
    
    # Analysis
    print("\nAnalyzing results...")
    results = compare_simulations(df, exp_data, max_lag=30)
    
    # Visualizations
    print("\nCreating visualizations...")
    
    # Plot trajectories
    plot_trajectories(sim, show_gradient=True)
    
    # Plot statistics
    plot_cell_statistics(sim)
    
    # Create animation
    print("\nCreating animation (this may take a moment)...")
    anim = create_animation(sim, interval=100, save_path='migration.gif')
    
    # Save data
    print("\nSaving results...")
    sim.save_trajectories('simulation_trajectories.csv')
    
    # Save analysis results
    results['sim_dacf'].to_csv('dacf_results.csv', index=False)
    results['sim_msd'].to_csv('msd_results.csv', index=False)
    
    print("\n" + "="*60)
    print("SIMULATION COMPLETE!")
    print("="*60)
    print("\nFiles saved:")
    print("  - simulation_trajectories.csv")
    print("  - dacf_results.csv")
    print("  - msd_results.csv")
    print("  - migration.gif")
    
    return sim, results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run simplified cell migration simulation')
    parser.add_argument('--data', type=str, default=None,
                       help='Path to experimental data CSV')
    parser.add_argument('--n_cells', type=int, default=30,
                       help='Number of cells (default: 30)')
    parser.add_argument('--n_steps', type=int, default=100,
                       help='Number of simulation steps (default: 100)')
    parser.add_argument('--L', type=float, default=60,
                       help='Stadium wall length (default: 60)')
    parser.add_argument('--R', type=float, default=20,
                       help='Stadium semicircle radius (default: 20)')
    parser.add_argument('--source_length', type=float, default=40,
                       help='Gradient source line length (default: 40)')
    parser.add_argument('--chemotaxis', type=float, default=0.3,
                       help='Chemotaxis strength 0-1 (default: 0.3)')
    parser.add_argument('--repulsion', type=float, default=0.2,
                       help='Repulsion strength 0-1 (default: 0.2)')
    
    args = parser.parse_args()
    
    # Run with command line parameters
    if args.data:
        sim, results = run_example(args.data)
    else:
        # Run basic example
        print("Running with default parameters...")
        print("To use experimental data, run with: --data your_data.csv")
        sim, results = run_example()