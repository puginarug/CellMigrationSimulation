"""
visualization.py
Simple visualization for cell migration in vertical stadium
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Polygon

from analysis import calculate_autocorrelation, calculate_msd


def plot_trajectories(simulation, show_gradient=True):
    """
    Plot all cell trajectories in the stadium.
    
    Parameters:
    -----------
    simulation : Simulation object
        Completed simulation
    show_gradient : bool
        Show gradient field arrows
    """
    fig, ax = plt.subplots(figsize=(6, 10))
    
    stadium = simulation.stadium
    
    # Draw stadium as a single closed polygon
    verts = []
    
    # Right wall from bottom to top
    verts.append((stadium.R, -stadium.L/2))
    verts.append((stadium.R, stadium.L/2))
    
    # Top semicircle
    theta = np.linspace(0, np.pi, 50)
    for t in theta:
        x = stadium.R * np.cos(t)
        y = stadium.L/2 + stadium.R * np.sin(t)
        verts.append((x, y))
    
    # Left wall from top to bottom
    verts.append((-stadium.R, stadium.L/2))
    verts.append((-stadium.R, -stadium.L/2))
    
    # Bottom semicircle
    theta = np.linspace(np.pi, 2*np.pi, 50)
    for t in theta:
        x = stadium.R * np.cos(t)
        y = -stadium.L/2 + stadium.R * np.sin(t)
        verts.append((x, y))
    
    # Create and add the polygon
    stadium_patch = Polygon(verts, closed=True, fill=False, 
                          edgecolor='black', linewidth=2)
    ax.add_patch(stadium_patch)
    
    # Show gradient source line
    ax.plot([0, 0], [-stadium.source_length/2, stadium.source_length/2], 
            'r-', linewidth=3, alpha=0.5, label='Gradient source')
    
    # Show gradient field arrows if requested
    if show_gradient:
        x_grid = np.linspace(-stadium.R*0.7, stadium.R*0.7, 5)
        y_grid = np.linspace(-stadium.L/2, stadium.L/2, 8)
        
        for xi in x_grid:
            for yi in y_grid:
                if stadium.is_inside(xi, yi):
                    strength, direction = stadium.get_gradient(xi, yi)
                    if strength > 0.1:
                        dx = strength * np.cos(direction) * 2
                        dy = strength * np.sin(direction) * 2
                        ax.arrow(xi, yi, dx, dy, 
                                head_width=0.8, head_length=0.4,
                                fc='blue', ec='blue', alpha=0.2)
    
    # Plot trajectories
    for cell in simulation.cells:
        # Trajectory
        ax.plot(cell.x_history, cell.y_history, alpha=0.5, linewidth=1)
        
        # Start position (green circle)
        ax.scatter(cell.x_history[0], cell.y_history[0], 
                  s=30, c='green', marker='o', alpha=0.7, zorder=5)
        
        # End position (red square)
        ax.scatter(cell.x_history[-1], cell.y_history[-1], 
                  s=30, c='red', marker='s', alpha=0.7, zorder=5)
    
    # Add legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='g', 
              markersize=8, label='Start'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='r', 
              markersize=8, label='End'),
        Line2D([0], [0], color='red', alpha=0.5, linewidth=3, 
              label='Gradient source')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    ax.set_xlim(-stadium.R * 1.3, stadium.R * 1.3)
    ax.set_ylim(-stadium.L/2 - stadium.R * 1.2, stadium.L/2 + stadium.R * 1.2)
    ax.set_aspect('equal')
    ax.set_xlabel('X (μm)', fontsize=12)
    ax.set_ylabel('Y (μm)', fontsize=12)
    ax.set_title(f'Cell Migration Trajectories (L={stadium.L}, R={stadium.R})', fontsize=14)
    ax.grid(False)
    
    plt.tight_layout()
    plt.show()
    
    return fig, ax


def create_animation(simulation, interval=100, save_path=None):
    """
    Create animation of cell migration.
    
    Parameters:
    -----------
    simulation : Simulation object
        Completed simulation
    interval : int
        Milliseconds between frames
    save_path : str
        Path to save animation (e.g., 'migration.gif')
    """
    fig, ax = plt.subplots(figsize=(6, 10))
    
    stadium = simulation.stadium
    
    # Draw stadium boundary
    verts = []
    verts.append((stadium.R, -stadium.L/2))
    verts.append((stadium.R, stadium.L/2))
    
    theta = np.linspace(0, np.pi, 50)
    for t in theta:
        x = stadium.R * np.cos(t)
        y = stadium.L/2 + stadium.R * np.sin(t)
        verts.append((x, y))
    
    verts.append((-stadium.R, stadium.L/2))
    verts.append((-stadium.R, -stadium.L/2))
    
    theta = np.linspace(np.pi, 2*np.pi, 50)
    for t in theta:
        x = stadium.R * np.cos(t)
        y = -stadium.L/2 + stadium.R * np.sin(t)
        verts.append((x, y))
    
    stadium_patch = Polygon(verts, closed=True, fill=False, 
                          edgecolor='black', linewidth=2)
    ax.add_patch(stadium_patch)
    
    # Gradient source line
    ax.plot([0, 0], [-stadium.source_length/2, stadium.source_length/2], 
            'r-', linewidth=3, alpha=0.5)
    
    # Gradient arrows (static)
    x_grid = np.linspace(-stadium.R*0.7, stadium.R*0.7, 5)
    y_grid = np.linspace(-stadium.L/2, stadium.L/2, 8)
    
    for xi in x_grid:
        for yi in y_grid:
            if stadium.is_inside(xi, yi):
                strength, direction = stadium.get_gradient(xi, yi)
                if strength > 0.1:
                    dx = strength * np.cos(direction) * 2
                    dy = strength * np.sin(direction) * 2
                    ax.arrow(xi, yi, dx, dy, 
                            head_width=0.8, head_length=0.4,
                            fc='blue', ec='blue', alpha=0.1)
    
    # Initialize cell points
    cell_points = ax.scatter([], [], s=50, c='blue', alpha=0.7)
    
    # Initialize trails
    trails = []
    for _ in simulation.cells:
        trail, = ax.plot([], [], 'b-', alpha=0.3, linewidth=0.5)
        trails.append(trail)
    
    # Time text
    time_text = ax.text(0.02, 0.98, '', transform=ax.transAxes,
                        fontsize=12, verticalalignment='top')
    
    ax.set_xlim(-stadium.R * 1.3, stadium.R * 1.3)
    ax.set_ylim(-stadium.L/2 - stadium.R * 1.2, stadium.L/2 + stadium.R * 1.2)
    ax.set_aspect('equal')
    ax.set_xlabel('X (μm)', fontsize=12)
    ax.set_ylabel('Y (μm)', fontsize=12)
    ax.set_title('Cell Migration Animation', fontsize=14)
    
    def init():
        cell_points.set_offsets(np.empty((0, 2)))
        for trail in trails:
            trail.set_data([], [])
        time_text.set_text('')
        return [cell_points] + trails + [time_text]
    
    def update(frame):
        # Update cell positions
        positions = []
        for i, cell in enumerate(simulation.cells):
            if frame < len(cell.x_history):
                positions.append([cell.x_history[frame], cell.y_history[frame]])
                
                # Update trail
                if frame > 0:
                    trails[i].set_data(cell.x_history[:frame+1], 
                                     cell.y_history[:frame+1])
        
        if positions:
            cell_points.set_offsets(np.array(positions))
        
        time_text.set_text(f'Step: {frame}')
        return [cell_points] + trails + [time_text]
    
    # Determine number of frames
    max_frames = max(len(cell.x_history) for cell in simulation.cells)
    
    anim = animation.FuncAnimation(fig, update, init_func=init,
                                 frames=max_frames, interval=interval,
                                 blit=True, repeat=True)
    
    if save_path:
        try:
            anim.save(save_path, writer='pillow', fps=10)
            print(f"Animation saved to {save_path}")
        except Exception as e:
            print(f"Could not save animation: {e}")
    
    plt.show()
    
    return anim


def plot_cell_statistics(simulation):
    """
    Plot comprehensive statistics of the simulation in a 2x2 grid.

    Creates four subplots showing:
    1. Mean Squared Displacement (log-log)
    2. Velocity distribution histogram
    3. Directional Autocorrelation Function (log-log)
    4. Initial vs final cell positions

    Parameters:
    -----------
    simulation : Simulation object
        Completed simulation with cell trajectories
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Get DataFrame
    df = simulation.get_dataframe()
    stadium = simulation.stadium
    
    # 1. MSD (scatter plot)
    msd = calculate_msd(df, max_lag=None)
    ax = axes[0, 0]
    ax.scatter(msd['lag'], msd['msd'], s=20)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Time Lag (steps)')
    ax.set_ylabel('MSD (μm²)')
    ax.set_title('Mean Squared Displacement')
    ax.grid(True, alpha=0.3)

    # 2. Velocity distribution
    ax = axes[0, 1]
    velocities = np.sqrt(df['v_x']**2 + df['v_y']**2)
    velocities = velocities[velocities > 0]
    ax.hist(np.log(velocities), bins=30, alpha=0.7, edgecolor='black')
    ax.set_xlabel('Velocity (μm/step)')
    ax.set_ylabel('Count')
    ax.set_title('Velocity Distribution')
    ax.grid(True, alpha=0.3)
    
    # 3. DACF (scatter plot)
    dacf = calculate_autocorrelation(df, max_lag=None, directional=True)
    ax = axes[1, 0]
    ax.scatter(dacf['lag'], dacf['dacf'], s=20)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Time Lag')
    ax.set_ylabel('DACF')
    ax.set_title('Directional Autocorrelation Function')
    ax.grid(True, alpha=0.3)

    # 4. Initial and Final positions
    ax = axes[1, 1]
    for cell in simulation.cells:
        ax.scatter(cell.x_history[0], cell.y_history[0], 
                  s=50, alpha=0.6, color='blue', label='Initial' if cell.id == 0 else "")
        ax.scatter(cell.x_history[-1], cell.y_history[-1], 
                  s=50, alpha=0.6, color='red', label='Final' if cell.id == 0 else "")
    ax.set_aspect('equal')
    ax.legend()