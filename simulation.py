"""
simulation.py
Simplified simulation engine for cell migration in vertical stadium
"""

import numpy as np
import pandas as pd
from cell import Cell
from stadium import VerticalStadium

class Simulation:
    """
    Cell migration simulation in a stadium-shaped domain with chemotactic gradient.

    This class orchestrates the simulation of multiple cells moving in a vertical stadium
    with various behavioral modes including random walk, persistent random walk, biased
    persistent walk, and memory-based movement with von Mises distributed turning angles.
    """
    
    def __init__(self, 
                 n_cells=30,
                 time_step=5.0, # in minutes (to match experimental data)
                 stadium_L=800,
                 stadium_R=200,
                 source_length=400,
                 chemotaxis_strength=0.0,
                 repulsion_strength=0.0, # the default is no cell-cell repulsion
                 interaction_radius=0.0, 
                 velocity_params=None,
                 persistence=0.0,
                 starting_positions='perimeter',
                 mode='biased_persistent',
                 # Memory parameters (used for memory modes)
                 memory_window=None,
                 memory_exp_lambda=0.1,
                 memory_power_alpha=0.5,
                 vonmises_params=None):
        """
        Initialize simulation.
        
        Parameters:
        -----------
        n_cells : int
            Number of cells
        time_step : float
            Time step in minutes
        stadium_L : float
            Length of straight walls in stadium
        stadium_R : float
            Radius of semicircles
        source_length : float
            Length of vertical line source for gradient
        chemotaxis_strength : float
            Strength of chemotactic response (0-1)
        repulsion_strength : float
            Strength of cell-cell repulsion (0-1)
        interaction_radius : float
            Range for cell-cell interactions
        velocity_params : dict
            Parameters for velocity distribution: {'shape': , 'loc': , 'scale': }
        persistence : float
            Persistence parameter for persistent random walk (0-1), ignored if mode uses memory
        starting_positions : str
            Initial positions distribution for cells: 'perimeter' or 'uniform'
        mode : str
            Movement mode: 'random', 'persistent', 'biased_persistent',
            'uniform_memory', 'exp_memory', or 'power_memory'
        memory_window : int, optional
            Number of past time steps to consider for memory-based modes
        memory_exp_lambda : float
            Exponential decay rate for 'exp_memory' mode (default: 0.1)
        memory_power_alpha : float
            Power law exponent for 'power_memory' mode (default: 0.5)
        vonmises_params : dict, optional
            Parameters for von Mises mixture: {'kappa1': , 'kappa2': , 'W1': }
        """
        self.n_cells = n_cells
        self.chemotaxis_strength = chemotaxis_strength
        self.repulsion_strength = repulsion_strength
        self.interaction_radius = interaction_radius
        self.persistence = persistence

        # Default velocity parameters
        if velocity_params is None:
            velocity_params = {'shape': 0.5, 'loc': 0, 'scale': 1.0}
        self.velocity_params = velocity_params
        
        # Initialize stadium
        self.stadium = VerticalStadium(L=stadium_L, R=stadium_R, 
                                      center=(0, 0), source_length=source_length)
        
        # Initialize cells
        positions = self.stadium.sample_initial_positions(n_cells, distribution=starting_positions)
        self.cells = []
        for i, (x, y) in enumerate(positions):
            self.cells.append(Cell(i, x, y, velocity_params))

        # Simulation mode: 'random', 'persistent', 'biased_persistent',
        # 'uniform_memory', 'exp_memory', 'power_memory'
        self.mode = mode

        # Memory kernel params
        self.memory_window = memory_window
        self.memory_exp_lambda = memory_exp_lambda
        self.memory_power_alpha = memory_power_alpha

        # von Mises mixture params for turning-angle distribution
        self.vonmises_kappa1 = vonmises_params['kappa1'] if vonmises_params else 1.0
        self.vonmises_kappa2 = vonmises_params['kappa2'] if vonmises_params else 1.0
        self.vonmises_mixture_alpha = vonmises_params['W1'] if vonmises_params else 0.5

        # Time tracking
        self.time = 0
        self.time_step = time_step
        
    def step(self):
        """
        Perform one simulation step for all cells.

        Each step involves:
        1. Determining angle change based on movement mode
        2. Calculating chemotaxis bias from gradient field
        3. Calculating repulsion from nearby cells
        4. Updating cell position with weighted combination of factors
        5. Applying boundary conditions
        6. Recording new position in history
        """
        
        for cell in self.cells:
            # 1. Get angle change according to selected mode
            angle_change = 0.0

            if self.mode == 'random':
                angle_change = np.random.uniform(-np.pi, np.pi)
            elif self.mode in ('persistent', 'biased_persistent'):
                # persistent behaviour: keep previous heading with probability persistence
                if len(cell.theta_history) > 0 and np.random.rand() < self.persistence:
                    # previous heading stored in history is an absolute heading; compute change
                    prev_theta = cell.theta_history[-1]
                    angle_change = np.arctan2(np.sin(prev_theta - cell.theta), np.cos(prev_theta - cell.theta))
                else:
                    angle_change = np.random.uniform(-np.pi, np.pi)
            elif self.mode in ('uniform_memory', 'exp_memory', 'power_memory'):
                # Build reference angle from past headings using selected kernel
                history = np.array(cell.theta_history)
                if history.size == 0:
                    ref_angle = cell.theta
                else:
                    # compute ages so that most recent has age 0
                    indices = np.arange(history.size)
                    ages = (history.size - 1) - indices

                    # Optionally restrict to last memory_window entries
                    if self.memory_window is not None:
                        m = int(self.memory_window)
                        if m <= 0:
                            # treat non-positive window as no memory
                            history_trim = np.array([])
                            ages_trim = np.array([])
                        else:
                            history_trim = history[-m:]
                            ages_trim = ages[-m:]
                    else:
                        history_trim = history
                        ages_trim = ages

                    # If after trimming there's no history, fallback to random sampling later
                    if history_trim.size == 0:
                        use_random = True
                    else:
                        use_random = False

                    if not use_random:
                        if self.mode == 'uniform_memory':
                            weights = np.ones_like(history_trim)
                        elif self.mode == 'exp_memory':
                            weights = np.exp(-self.memory_exp_lambda * ages_trim)
                        else:  # power_memory
                            weights = (ages_trim + 1.0) ** (-self.memory_power_alpha)

                        # Normalize
                        wsum = np.sum(weights)
                        if wsum <= 1e-12:
                            # numerically degenerate: fallback to using the most recent heading
                            ref_angle = history_trim[-1]
                        else:
                            weights = weights / wsum
                            # Circular weighted mean
                            c = np.sum(weights * np.cos(history_trim))
                            s = np.sum(weights * np.sin(history_trim))
                            ref_angle = np.arctan2(s, c)
                    else:
                        # No memory available -> force random behavior
                        ref_angle = None

                # Sample from mixture of two von Mises centered at ref_angle (or random if ref_angle is None)
                if ref_angle is None:
                    # No memory available -> choose a random heading
                    angle_change = np.random.uniform(-np.pi, np.pi)
                else:
                    alpha = float(self.vonmises_mixture_alpha)
                    if np.random.rand() < alpha:
                        kappa = float(self.vonmises_kappa1)
                    else:
                        kappa = float(self.vonmises_kappa2)

                    # numpy's vonmises takes (mu, kappa)
                    sampled_theta = np.random.vonmises(ref_angle, kappa)
                    angle_change = np.arctan2(np.sin(sampled_theta - cell.theta), np.cos(sampled_theta - cell.theta))
            else:
                # fallback: random
                angle_change = np.random.uniform(-np.pi, np.pi)

            # 2. Calculate chemotaxis bias
            grad_strength, grad_direction = self.stadium.get_gradient(cell.x, cell.y)
            chemotaxis_bias = cell.sense_gradient(grad_strength, grad_direction)
            
            # 3. Calculate repulsion bias
            if self.repulsion_strength > 0 and self.interaction_radius > 0:
                repulsion_bias = cell.calculate_repulsion(self.cells, self.interaction_radius)
            else:
                repulsion_bias = 0.0

            # 4. Update cell position (returns new position before boundary check)
            new_x, new_y = cell.update_position(
                self.time_step,
                angle_change,
                chemotaxis_bias,
                repulsion_bias,
                self.chemotaxis_strength,
                self.repulsion_strength
            )
            
            # 5. Apply boundary conditions
            bounded_x, bounded_y = self.stadium.apply_boundary(new_x, new_y)
            
            # 6. Set the final position
            cell.set_position(bounded_x, bounded_y)
        
        self.time += self.time_step
    
    def run(self, n_steps, verbose=True):
        """
        Run simulation for n steps.
        
        Parameters:
        -----------
        n_steps : int
            Number of steps to simulate
        verbose : bool
            Print progress
        """
        for i in range(n_steps):
            self.step()
            
            if verbose and (i + 1) % 25 == 0:
                print(f"Step {i+1}/{n_steps} completed")
        
        if verbose:
            print(f"Simulation completed: {n_steps} steps")
    
    def get_dataframe(self):
        """
        Convert simulation results to DataFrame.
        
        Returns:
        --------
        DataFrame with columns: track_id, step, x, y, v_x, v_y
        """
        data_list = []
        
        for cell in self.cells:
            n_points = len(cell.x_history)
            
            # Calculate velocities
            x_array = np.array(cell.x_history)
            y_array = np.array(cell.y_history)
            
            v_x = np.zeros(n_points)
            v_y = np.zeros(n_points)
            
            v_x[1:] = np.diff(x_array)
            v_y[1:] = np.diff(y_array)
            
            # Create DataFrame for this cell
            cell_data = pd.DataFrame({
                'track_id': cell.id,
                'step': range(n_points),
                't': np.arange(n_points) * self.time_step,
                'x': x_array,
                'y': y_array,
                'v_x': v_x,
                'v_y': v_y
            })
            
            data_list.append(cell_data)
        
        return pd.concat(data_list, ignore_index=True)
    
    def save_trajectories(self, filename='trajectories.csv'):
        """Save trajectories to CSV file."""
        df = self.get_dataframe()
        df.to_csv(filename, index=False)
        print(f"Trajectories saved to {filename}")
        return df