"""
stadium.py
Vertical stadium domain with gradient field
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon


class VerticalStadium:
    """
    Vertical stadium-shaped domain (rectangle with semicircular caps) with line source gradient.

    The stadium consists of two vertical straight walls of length L connected by two semicircles
    of radius R. A vertical line source at the center creates a chemotactic gradient field that
    decays with distance from the source according to a fitted polynomial.
    """
    
    def __init__(self, L=60, R=20, center=(0, 0), source_length=40):
        """
        Initialize vertical stadium.
        
        Parameters:
        -----------
        L : float
            Length of straight walls (vertical extent)
        R : float
            Radius of semicircles at top and bottom
        center : tuple
            (x, y) center position
        source_length : float
            Length of vertical line source for gradient
        """
        self.L = L  # Length of straight walls
        self.R = R  # Radius of semicircles
        self.center = center
        self.source_length = source_length
        
        # Total height = L + 2*R (straight part + two semicircles)
        self.total_height = L + 2 * R
        
        # Gradient parameters
        self.gradient_strength = 1.0
        self.gradient_decay = 0.05  # Decay rate from line source
        
    def is_inside(self, x, y):
        """
        Check if point (x, y) is inside the stadium boundary.

        Parameters:
        -----------
        x, y : float
            Coordinates to check

        Returns:
        --------
        bool : True if point is inside stadium, False otherwise
        """
        cx, cy = self.center
        
        # Check if in straight wall region
        if -self.R <= x <= self.R and -self.L/2 <= y <= self.L/2:
            return True
        
        # Check top semicircle
        if y > self.L/2:
            dy = y - self.L/2
            if x**2 + dy**2 <= self.R**2:
                return True
        
        # Check bottom semicircle
        if y < -self.L/2:
            dy = y + self.L/2
            if x**2 + dy**2 <= self.R**2:
                return True
        
        return False
    
    def apply_boundary(self, x, y):
        """
        Apply boundary condition - cells stop at boundary.
        
        Parameters:
        -----------
        x, y : float
            Proposed position
            
        Returns:
        --------
        tuple : (x, y) constrained to be inside stadium
        """
        # If inside, return as is
        if self.is_inside(x, y):
            return x, y
        
        # Find closest point on boundary
        # Check which section we're closest to
        
        # Straight wall region
        if -self.L/2 <= y <= self.L/2:
            # Closest to left or right wall
            if x < 0:
                return -self.R, y
            else:
                return self.R, y
        
        # Top semicircle
        elif y > self.L/2:
            center_y = self.L/2
            dy = y - center_y
            dist = np.sqrt(x**2 + dy**2)
            if dist > 0:
                # Project to circle boundary
                return x/dist * self.R, center_y + dy/dist * self.R
            else:
                return 0, center_y + self.R
        
        # Bottom semicircle
        else:
            center_y = -self.L/2
            dy = y - center_y
            dist = np.sqrt(x**2 + dy**2)
            if dist > 0:
                # Project to circle boundary
                return x/dist * self.R, center_y + dy/dist * self.R
            else:
                return 0, center_y - self.R
    
    def get_gradient(self, x, y):
        """
        Get gradient at position from vertical line source.
        
        The gradient emanates from a vertical line of length source_length
        centered at the origin. The field decays with distance from the line.
        
        Returns:
        --------
        tuple : (gradient_strength, gradient_direction)
        """
        if not self.is_inside(x, y):
            return 0, 0
        
        # Vertical line source from -source_length/2 to source_length/2
        line_bottom = -self.source_length/2
        line_top = self.source_length/2
        
        # Find closest point on line source
        if y < line_bottom:
            # Below line source
            closest_y = line_bottom
            dx = x
            dy = y - closest_y
        elif y > line_top:
            # Above line source
            closest_y = line_top
            dx = x
            dy = y - closest_y
        else:
            # Beside line source (y is within line bounds)
            closest_y = y
            dx = x
            dy = 0
        
        # Distance to line source
        dist = np.sqrt(dx**2 + dy**2)
        
        if dist < 0.1:  # Avoid division by zero
            return 0, 0

        # normalize distance to [-1, 1]
        norm_dist = dist / (self.R + self.L/2)
        norm_dist = np.clip(norm_dist, -1, 1)

        # Evaluate the 4th-degree polynomial
        polynomial_coefficients = [9.77156799e-01, -1.27160505e-16, -1.93457460e+00, 
                                -2.25204704e-17, 9.88800428e-01]
        polynomial_value = np.polyval(polynomial_coefficients, norm_dist)

        # Gradient strength decreases with distance
        strength = self.gradient_strength * polynomial_value

        # Direction points towards line source
        direction = np.arctan2(-dy, -dx)

        return strength, direction
    
    def sample_initial_positions(self, n_cells, distribution='bottom'):
        """
        Sample initial cell positions.
        
        Parameters:
        -----------
        n_cells : int
            Number of cells
        distribution : str
            'bottom' - start in bottom semicircle
            'uniform' - uniform throughout stadium
            'perimeter' - along the perimeter
            
        Returns:
        --------
        list : [(x, y)] positions
        """
        positions = []
        
        if distribution == 'bottom':
            # Start cells in bottom semicircle
            for _ in range(n_cells):
                # Random point in bottom semicircle
                angle = np.random.uniform(np.pi, 2*np.pi)  # Bottom half
                r = np.sqrt(np.random.uniform(0, 1)) * self.R * 0.8
                x = r * np.cos(angle)
                y = -self.L/2 + r * np.sin(angle)
                positions.append((x, y))
                
        elif distribution == 'uniform':
            # Uniform distribution throughout stadium
            attempts = 0
            while len(positions) < n_cells and attempts < n_cells * 100:
                x = np.random.uniform(-self.R, self.R)
                y = np.random.uniform(-self.L/2 - self.R, self.L/2 + self.R)
                if self.is_inside(x, y):
                    positions.append((x, y))
                attempts += 1
                
        elif distribution == 'perimeter':
            # Distribute along perimeter
            # Total perimeter = 2*L + 2*π*R
            perimeter = 2 * self.L + 2 * np.pi * self.R
            
            for i in range(n_cells):
                s = (i / n_cells) * perimeter
                
                if s < self.L:
                    # Right wall going up
                    x = self.R
                    y = -self.L/2 + s
                elif s < self.L + np.pi * self.R:
                    # Top semicircle
                    angle = (s - self.L) / self.R
                    x = self.R * np.cos(angle)
                    y = self.L/2 + self.R * np.sin(angle)
                elif s < 2 * self.L + np.pi * self.R:
                    # Left wall going down
                    x = -self.R
                    y = self.L/2 - (s - self.L - np.pi * self.R)
                else:
                    # Bottom semicircle
                    angle = np.pi + (s - 2 * self.L - np.pi * self.R) / self.R
                    x = self.R * np.cos(angle)
                    y = -self.L/2 + self.R * np.sin(angle)
                
                positions.append((x, y))
        
        return positions
    
    def visualize(self, cells=None):
        """Visualize the stadium and optional cell positions."""
        fig, ax = plt.subplots(figsize=(6, 10))
        
        # Draw stadium as a single closed polygon
        verts = []
        
        # 1) Right wall from bottom to top
        verts.append((self.R, -self.L/2))
        verts.append((self.R, self.L/2))
        
        # 2) Top semicircle (θ from 0 → π)
        theta = np.linspace(0, np.pi, 50)
        for t in theta:
            x = self.R * np.cos(t)
            y = self.L/2 + self.R * np.sin(t)
            verts.append((x, y))
        
        # 3) Left wall from top to bottom
        verts.append((-self.R, self.L/2))
        verts.append((-self.R, -self.L/2))
        
        # 4) Bottom semicircle (θ from π → 2π)
        theta = np.linspace(np.pi, 2*np.pi, 50)
        for t in theta:
            x = self.R * np.cos(t)
            y = -self.L/2 + self.R * np.sin(t)
            verts.append((x, y))
        
        # Create and add the polygon
        stadium = Polygon(verts, closed=True, fill=False, 
                         edgecolor='black', linewidth=2)
        ax.add_patch(stadium)
        
        # Draw the gradient line source
        ax.plot([0, 0], [-self.source_length/2, self.source_length/2], 
                'r-', linewidth=3, label=f'Gradient source (L={self.source_length})')
        
        # Show gradient field with arrows
        x_grid = np.linspace(-self.R*0.8, self.R*0.8, 7)
        y_grid = np.linspace(-self.L/2, self.L/2, 10)
        
        for xi in x_grid:
            for yi in y_grid:
                if self.is_inside(xi, yi):
                    strength, direction = self.get_gradient(xi, yi)
                    if strength > 0.1:  # Only show significant gradients
                        dx = strength * np.cos(direction) * 3
                        dy = strength * np.sin(direction) * 3
                        ax.arrow(xi, yi, dx, dy, 
                                head_width=1, head_length=0.5,
                                fc='blue', ec='blue', alpha=0.3)
        
        # Plot cells if provided
        if cells is not None:
            for cell in cells:
                ax.scatter(cell.x, cell.y, s=50, c='green', alpha=0.7, zorder=5)
        
        ax.set_xlim(-self.R * 1.3, self.R * 1.3)
        ax.set_ylim(-self.L/2 - self.R * 1.2, self.L/2 + self.R * 1.2)
        ax.set_aspect('equal')
        ax.set_xlabel('X (μm)')
        ax.set_ylabel('Y (μm)')
        ax.set_title(f'Vertical Stadium (L={self.L}, R={self.R})')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()