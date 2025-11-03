# Cell Migration Simulation

A Python-based simulation framework for modeling cell migration in a vertical stadium-shaped domain with gradient fields. This project provides tools for simulating, analyzing, and visualizing cell motility with chemotaxis and cell-cell interactions.

## Overview

This simulation models cell migration behavior in confined geometries, specifically a vertical stadium (capsule) shape with a line-source gradient field. The framework supports multiple motion models including random walks, persistent random walks, and memory-based motion, making it suitable for studying collective cell behavior and chemotaxis.

## Features

### Core Simulation
- **Multiple Motion Models:**
  - Random walk
  - Persistent random walk
  - Biased persistent walk (with chemotaxis)
  - Memory-based models (uniform, exponential, power-law memory kernels)

- **Cell Behaviors:**
  - Chemotaxis response to gradient fields
  - Cell-cell repulsion with configurable interaction radius
  - Log-normal velocity distribution matching experimental data

- **Domain Geometry:**
  - Vertical stadium (capsule) shaped boundary
  - Configurable dimensions (wall length L, radius R)
  - Line-source gradient field with polynomial decay

### Analysis Tools
- **Directional Autocorrelation Function (DACF):** Measures persistence of cell movement direction
- **Mean Squared Displacement (MSD):** Characterizes cell motility and diffusion properties
- **Velocity Distribution Analysis:** Fits log-normal distributions to cell velocities
- **Turning Angle Analysis:** Von Mises mixture model fitting for angular distributions
- **Per-movie/position statistics:** Weighted aggregation across multiple experimental datasets

### Visualization
- Trajectory plotting with gradient field overlay
- Animated cell migration (GIF export)
- Statistical plots (MSD, DACF, velocity distributions)
- Initial and final position distributions

### Parameter Fitting
- Grid-search optimization to match experimental data
- Chi-squared minimization for DACF and MSD
- Support for loading experimental trajectory data
- Automatic parameter estimation from data

## Installation

### Requirements
- Python 3.7+
- NumPy
- Pandas
- Matplotlib
- SciPy

### Setup
```bash
git clone <repository-url>
cd CellMigrationSimulation
pip install numpy pandas matplotlib scipy
```

## Usage

### Basic Simulation

Run a simulation with default parameters:

```bash
python main.py
```

### With Experimental Data

Load experimental data and fit velocity distributions:

```bash
python main.py --data path/to/experimental_data.csv
```

### Custom Parameters

Configure simulation parameters:

```bash
python main.py \
    --n_cells 50 \
    --n_steps 200 \
    --L 800 \
    --R 200 \
    --source_length 400 \
    --chemotaxis 0.3 \
    --repulsion 0.2
```

### Parameter Fitting

Fit simulation parameters to experimental data:

```bash
python fit.py --data path/to/experimental_data.csv --n_steps 100
```

## Project Structure

```
CellMigrationSimulation/
├── main.py              # Main simulation script with CLI
├── cell.py              # Cell agent class with movement logic
├── stadium.py           # Stadium geometry and gradient field
├── simulation.py        # Simulation engine and orchestration
├── analysis.py          # Statistical analysis functions (DACF, MSD)
├── visualization.py     # Plotting and animation tools
├── fit.py              # Parameter fitting via grid search
└── running_sim_in_nb.ipynb  # Jupyter notebook for interactive use
```

## Key Components

### Cell Agent (`cell.py`)
The `Cell` class represents individual cell agents with:
- Position tracking (x, y coordinates)
- Orientation (theta angle)
- Movement history
- Gradient sensing capabilities
- Repulsion calculation from neighboring cells

### Stadium Domain (`stadium.py`)
The `VerticalStadium` class defines:
- Capsule-shaped boundary geometry
- Line-source gradient field with polynomial decay
- Boundary conditions (reflecting walls)
- Initial position sampling (uniform, perimeter, bottom)
- Gradient visualization

### Simulation Engine (`simulation.py`)
The `Simulation` class orchestrates:
- Multiple cell agents
- Time stepping and integration
- Motion model selection (random, persistent, memory-based)
- Cell-cell and cell-gradient interactions
- Data export to CSV and DataFrame formats

### Analysis Module (`analysis.py`)
Statistical analysis functions:
- `calculate_autocorrelation()`: Computes DACF or VACF
- `calculate_msd()`: Computes mean squared displacement
- `compute_msd_dacf_per_movie()`: Per-position weighted statistics
- `compute_turning_angles()`: Extracts turning angle distributions
- `fit_von_mises_mixture_mle()`: Fits von Mises mixture models

### Visualization (`visualization.py`)
Plotting utilities:
- `plot_trajectories()`: Static trajectory plots with gradient field
- `create_animation()`: Animated cell migration (GIF export)
- `plot_cell_statistics()`: Multi-panel statistical plots

## Motion Models

### Random Walk
Cells change direction uniformly at each time step.

### Persistent Random Walk
Cells maintain their previous direction with probability `persistence`:
```python
sim = Simulation(mode='persistent', persistence=0.8, ...)
```

### Biased Persistent Walk
Adds chemotaxis bias toward gradient:
```python
sim = Simulation(
    mode='biased_persistent',
    persistence=0.8,
    chemotaxis_strength=0.3,
    ...
)
```

### Memory-Based Models
Direction determined by weighted average of past headings:
- **Uniform memory:** Equal weight to all past directions
- **Exponential memory:** Recent directions weighted more (decay rate `lambda`)
- **Power-law memory:** Power-law decay with exponent `alpha`

```python
sim = Simulation(
    mode='exp_memory',
    memory_window=10,
    memory_exp_lambda=0.1,
    vonmises_params={'kappa1': 2.0, 'kappa2': 10.0, 'W1': 0.3},
    ...
)
```

## Experimental Data Format

Expected CSV format for experimental trajectories:
```csv
track_id,step,normalized_time,x_microns,y_microns
0,0,0.0,10.5,20.3
0,1,5.0,11.2,21.1
...
```

Or with generic column names:
```csv
track_id,step,t,x,y
0,0,0.0,10.5,20.3
...
```

For per-position analysis, include a `file` column:
```csv
file,track_id,step,t,x_microns,y_microns
movie1,0,0,0.0,10.5,20.3
...
```

## Output Files

Running the simulation generates:
- `simulation_trajectories.csv`: Raw trajectory data
- `dacf_results.csv`: Directional autocorrelation function
- `msd_results.csv`: Mean squared displacement
- `migration.gif`: Animation of cell migration
- `fit_results.csv`: Parameter fitting results (from `fit.py`)

## Examples

### Example 1: Basic Simulation
```python
from simulation import Simulation

sim = Simulation(
    n_cells=30,
    stadium_L=60,
    stadium_R=20,
    source_length=40,
    chemotaxis_strength=0.3
)

sim.run(n_steps=100)
sim.save_trajectories('output.csv')
```

### Example 2: Analysis
```python
from analysis import calculate_msd, calculate_dacf
import pandas as pd

df = pd.read_csv('simulation_trajectories.csv')
msd = calculate_msd(df, max_lag=50)
dacf = calculate_autocorrelation(df, max_lag=50, directional=True)
```

### Example 3: Visualization
```python
from visualization import plot_trajectories, create_animation

plot_trajectories(sim, show_gradient=True)
anim = create_animation(sim, interval=100, save_path='migration.gif')
```

### Example 4: Interactive Notebook
Open `running_sim_in_nb.ipynb` in Jupyter for interactive exploration and parameter tuning.

## Parameters Reference

### Simulation Parameters
- `n_cells`: Number of cell agents (default: 30)
- `time_step`: Time step in minutes (default: 5.0)
- `stadium_L`: Length of straight walls in microns (default: 800)
- `stadium_R`: Radius of semicircles in microns (default: 200)
- `source_length`: Length of gradient line source in microns (default: 400)
- `chemotaxis_strength`: Chemotaxis weight 0-1 (default: 0.0)
- `repulsion_strength`: Cell-cell repulsion weight 0-1 (default: 0.0)
- `interaction_radius`: Range for cell interactions in microns (default: 0.0)
- `persistence`: Probability of maintaining direction (default: 0.0)
- `starting_positions`: Initial distribution ('uniform', 'perimeter', 'bottom')
- `mode`: Motion model ('random', 'persistent', 'biased_persistent', 'uniform_memory', 'exp_memory', 'power_memory')

### Motion Model Parameters
- `velocity_params`: Log-normal parameters `{'shape': 0.5, 'loc': 0, 'scale': 1.0}`
- `memory_window`: Number of past steps to consider (memory models)
- `memory_exp_lambda`: Decay rate for exponential memory (default: 0.1)
- `memory_power_alpha`: Exponent for power-law memory (default: 0.5)
- `vonmises_params`: Von Mises mixture parameters `{'kappa1': 1.0, 'kappa2': 1.0, 'W1': 0.5}`

## Citation

If you use this code in your research, please cite appropriately and refer to the associated publication(s).

## License

See LICENSE file for details.

## Contributing

Contributions are welcome! Please open issues for bugs or feature requests, and submit pull requests for improvements.

## Contact

For questions or feedback, please open an issue on the GitHub repository.

