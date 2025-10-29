"""
cell.py
Simplified cell agent for vertical stadium migration
"""

import numpy as np
from scipy.stats import lognorm


class Cell:
    """
    A cell agent that moves in a 2D environment with chemotaxis and cell-cell interactions.

    The cell maintains its position, orientation (theta), and movement history. It can sense
    chemical gradients, respond to nearby cells through repulsion, and move according to a
    log-normal velocity distribution.
    """
    
    def __init__(self, cell_id, x, y, velocity_params):
        """
        Initialize a cell.
        
        Parameters:
        -----------
        cell_id : int
            Unique identifier
        x, y : float
            Initial position
        velocity_params : dict
            Log-normal distribution parameters (shape, loc, scale)
        """
        self.id = cell_id
        self.x = x
        self.y = y
        self.theta = np.random.uniform(-np.pi, np.pi)
        self.velocity_params = velocity_params
        
        # History tracking
        self.x_history = [x]
        self.y_history = [y]
        self.theta_history = [self.theta]
        
    def sense_gradient(self, gradient_strength, gradient_direction):
        """
        Calculate response to gradient.
        
        Parameters:
        -----------
        gradient_strength : float
            Strength of gradient at current position
        gradient_direction : float
            Direction of gradient (angle in radians)
            
        Returns:
        --------
        float : Angle bias toward gradient
        """
        if gradient_strength <= 0:
            return 0
            
        # Calculate angle difference to gradient
        angle_diff = gradient_direction - self.theta
        angle_diff = np.arctan2(np.sin(angle_diff), np.cos(angle_diff))
        
        # Saturation function (simple exponential)
        saturation = 1 - np.exp(-gradient_strength)
        
        return angle_diff * saturation
    
    def calculate_repulsion(self, cells, interaction_radius=10.0):
        """
        Calculate repulsion from nearby cells.
        
        Parameters:
        -----------
        cells : list of Cell
            All cells in simulation
        interaction_radius : float
            Maximum distance for repulsion
            
        Returns:
        --------
        float : Angle bias away from neighbors
        """
        repulse_x = 0
        repulse_y = 0
        
        for other in cells:
            if other.id == self.id:
                continue
                
            dx = self.x - other.x
            dy = self.y - other.y
            dist = np.sqrt(dx**2 + dy**2)
            
            if 0 < dist < interaction_radius:
                # Repulsion strength decreases with distance
                force = (interaction_radius - dist) / interaction_radius
                repulse_x += (dx / dist) * force
                repulse_y += (dy / dist) * force
        
        if repulse_x == 0 and repulse_y == 0:
            return 0
            
        # Convert to angle
        repulsion_angle = np.arctan2(repulse_y, repulse_x)
        angle_diff = repulsion_angle - self.theta
        angle_diff = np.arctan2(np.sin(angle_diff), np.cos(angle_diff))
        
        return angle_diff * min(np.sqrt(repulse_x**2 + repulse_y**2), 1.0)
    
    def update_position(self, time_step, angle_change, chemotaxis_bias, repulsion_bias, 
                       chemotaxis_strength=0.3, repulsion_strength=0.2):
        """
        Update cell position based on various inputs.
        This is the main method that performs one step of cell movement.
        
        Parameters:
        -----------
        time_step : float
            Time step for the simulation
        angle_change : float
            Base angle change
        chemotaxis_bias : float
            Bias toward chemoattractant
        repulsion_bias : float
            Bias away from other cells
        chemotaxis_strength : float
            Weight for chemotaxis (0-1)
        repulsion_strength : float
            Weight for repulsion (0-1)
            
        Returns:
        --------
        tuple : (new_x, new_y) - the new position before boundary checking
        """
        # Compute suggested absolute directions from each contribution
        theta_rand = self.theta + angle_change
        theta_chemo = self.theta + chemotaxis_bias
        theta_rep = self.theta + repulsion_bias

        # Decide weights. If chemotaxis + repulsion < 1, give remaining weight to randomness.
        # Otherwise normalize all three so they form a convex combination.
        w_c = float(chemotaxis_strength)
        w_r = float(repulsion_strength)
        w_rand = max(0.0, 1.0 - (w_c + w_r))

        total = w_c + w_r + w_rand
        if total == 0.0:
            # fallback: give equal weight to each direction
            w_c = w_r = w_rand = 1.0/3.0
        else:
            w_c /= total
            w_r /= total
            w_rand /= total

        # Vector (cos,sin) averaging is robust to angle wrap-around
        vx = (w_rand * np.cos(theta_rand) +
              w_c * np.cos(theta_chemo) +
              w_r * np.cos(theta_rep))
        vy = (w_rand * np.sin(theta_rand) +
              w_c * np.sin(theta_chemo) +
              w_r * np.sin(theta_rep))

        # New orientation is the angle of the weighted mean vector
        self.theta = np.arctan2(vy, vx)
        
        # Sample velocity from log-normal distribution
        velocity = lognorm.rvs(
            self.velocity_params['shape'],
            self.velocity_params['loc'],
            self.velocity_params['scale']
        )
        
        # Calculate new position
        dx = velocity * np.cos(self.theta) * time_step
        dy = velocity * np.sin(self.theta) * time_step
        
        new_x = self.x + dx
        new_y = self.y + dy
        
        # Return the new position (will be checked for boundaries in simulation)
        return new_x, new_y
    
    def set_position(self, x, y):
        """
        Set the cell's position after boundary checking and record history.
        
        Parameters:
        -----------
        x, y : float
            Final position after boundary checking
        """
        self.x = x
        self.y = y
        
        # Store history
        self.x_history.append(self.x)
        self.y_history.append(self.y)
        self.theta_history.append(self.theta)