import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from typing import List, Dict, Optional, Tuple
from ..models.celestial_body import CelestialBody
import plotly.express as px

class GravityVisualizer:
    """Creates interactive visualizations for gravity simulations"""
    
    def __init__(self):
        self.default_layout = {
            'template': 'plotly_dark',
            'height': 600,
            'showlegend': True,
            'hovermode': 'closest'
        }
        
        # Color palette for celestial bodies
        self.colors = {
            'Sun': '#FDB813',
            'Earth': '#4169E1',
            'Mars': '#CD5C5C',
            'Jupiter': '#DAA520',
            'Saturn': '#F4A460',
            'Moon': '#C0C0C0',
            'Asteroid': '#A0522D',
            'Comet': '#00CED1',
            'Spacecraft': '#32CD32',
            'default': '#9370DB'
        }
    
    def create_orbit_plot(self, trajectories: Dict[str, np.ndarray], 
                         bodies: List[CelestialBody]) -> go.Figure:
        """Create interactive orbital trajectory plot"""
        fig = go.Figure()
        
        # Plot trajectories
        for body_name, trajectory in trajectories.items():
            color = self._get_color(body_name)
            
            # Main trajectory
            fig.add_trace(go.Scatter(
                x=trajectory[:, 0] / 1.496e11,  # Convert to AU
                y=trajectory[:, 1] / 1.496e11,
                mode='lines',
                name=f'{body_name} path',
                line=dict(color=color, width=2),
                hovertemplate='%{text}<br>X: %{x:.3f} AU<br>Y: %{y:.3f} AU',
                text=[f'{body_name} at step {i}' for i in range(len(trajectory))]
            ))
            
            # Current position
            fig.add_trace(go.Scatter(
                x=[trajectory[-1, 0] / 1.496e11],
                y=[trajectory[-1, 1] / 1.496e11],
                mode='markers',
                name=body_name,
                marker=dict(
                    size=15,
                    color=color,
                    symbol='circle',
                    line=dict(color='white', width=2)
                ),
                hovertemplate=f'{body_name}<br>X: %{{x:.3f}} AU<br>Y: %{{y:.3f}} AU'
            ))
        
        # Update layout
        fig.update_layout(
            title='Orbital Trajectories',
            xaxis=dict(
                title='X Position (AU)',
                zeroline=True,
                gridcolor='rgba(255,255,255,0.1)'
            ),
            yaxis=dict(
                title='Y Position (AU)',
                zeroline=True,
                gridcolor='rgba(255,255,255,0.1)',
                scaleanchor='x',
                scaleratio=1
            ),
            **self.default_layout
        )
        
        return fig
    
    def create_gravity_field_3d(self, bodies: List[CelestialBody], 
                               grid_size: int = 50) -> go.Figure:
        """Create 3D gravitational potential field visualization"""
        # Create grid
        bound = max([np.linalg.norm(body.position) for body in bodies]) * 2
        x = np.linspace(-bound, bound, grid_size)
        y = np.linspace(-bound, bound, grid_size)
        X, Y = np.meshgrid(x, y)
        
        # Calculate potential field
        potential = np.zeros_like(X)
        G = 6.67430e-11
        
        for body in bodies:
            for i in range(grid_size):
                for j in range(grid_size):
                    r = np.sqrt((X[i, j] - body.position[0])**2 + 
                              (Y[i, j] - body.position[1])**2)
                    if r > body.radius:
                        potential[i, j] -= G * body.mass / r
        
        # Normalize for visualization
        potential_normalized = potential / np.abs(np.min(potential))
        
        fig = go.Figure(data=[
            go.Surface(
                x=X / 1.496e11,  # Convert to AU
                y=Y / 1.496e11,
                z=potential_normalized,
                colorscale='Viridis',
                name='Gravitational Potential',
                hovertemplate='X: %{x:.2f} AU<br>Y: %{y:.2f} AU<br>Potential: %{z:.3f}'
            )
        ])
        
        # Add body positions
        for body in bodies:
            fig.add_trace(go.Scatter3d(
                x=[body.position[0] / 1.496e11],
                y=[body.position[1] / 1.496e11],
                z=[0],
                mode='markers+text',
                name=body.name,
                marker=dict(size=10, color=self._get_color(body.name)),
                text=[body.name],
                textposition='top center'
            ))
        
        fig.update_layout(
            title='3D Gravitational Potential Field',
            scene=dict(
                xaxis_title='X (AU)',
                yaxis_title='Y (AU)',
                zaxis_title='Normalized Potential',
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
            ),
            **self.default_layout
        )
        
        return fig
    
    def create_energy_plot(self, trajectories: Dict[str, np.ndarray], 
                          bodies: List[CelestialBody], dt: float) -> go.Figure:
        """Plot system energy over time"""
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=('Kinetic Energy', 'Potential Energy', 'Total Energy'),
            shared_xaxes=True,
            vertical_spacing=0.1
        )
        
        # Time array
        steps = len(list(trajectories.values())[0])
        time = np.arange(steps) * dt / 3600  # Convert to hours
        
        # Calculate energies
        kinetic = np.zeros(steps)
        potential = np.zeros(steps)
        
        # This is simplified - in reality would need velocities at each step
        for step in range(steps):
            # Kinetic energy (simplified)
            for body in bodies:
                kinetic[step] += 0.5 * body.mass * np.sum(body.velocity**2)
            
            # Potential energy
            for i, body1 in enumerate(bodies):
                for j, body2 in enumerate(bodies[i+1:], i+1):
                    r = np.linalg.norm(
                        trajectories[body1.name][step] - trajectories[body2.name][step]
                    )
                    if r > 0:
                        potential[step] -= 6.67430e-11 * body1.mass * body2.mass / r
        
        total = kinetic + potential
        
        # Plot
        fig.add_trace(go.Scatter(
            x=time, y=kinetic / 1e30,
            name='Kinetic', line=dict(color='red')
        ), row=1, col=1)
        
        fig.add_trace(go.Scatter(
            x=time, y=potential / 1e30,
            name='Potential', line=dict(color='blue')
        ), row=2, col=1)
        
        fig.add_trace(go.Scatter(
            x=time, y=total / 1e30,
            name='Total', line=dict(color='green')
        ), row=3, col=1)
        
        # Update layout
        fig.update_xaxes(title_text="Time (hours)", row=3, col=1)
        fig.update_yaxes(title_text="Energy (×10³⁰ J)", row=1, col=1)
        fig.update_yaxes(title_text="Energy (×10³⁰ J)", row=2, col=1)
        fig.update_yaxes(title_text="Energy (×10³⁰ J)", row=3, col=1)
        
        fig.update_layout(
            title='System Energy Conservation',
            height=800,
            **{k: v for k, v in self.default_layout.items() if k != 'height'}
        )
        
        return fig
    
    def create_phase_space_plot(self, body_name: str, trajectory: np.ndarray, 
                               velocity: np.ndarray) -> go.Figure:
        """Create phase space diagram (position vs velocity)"""
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('X Phase Space', 'Y Phase Space')
        )
        
        # X phase space
        fig.add_trace(go.Scatter(
            x=trajectory[:, 0] / 1.496e11,  # Position in AU
            y=velocity[:, 0] / 1000,  # Velocity in km/s
            mode='lines',
            name='X trajectory',
            line=dict(color='blue')
        ), row=1, col=1)
        
        # Y phase space
        fig.add_trace(go.Scatter(
            x=trajectory[:, 1] / 1.496e11,
            y=velocity[:, 1] / 1000,
            mode='lines',
            name='Y trajectory',
            line=dict(color='red')
        ), row=1, col=2)
        
        # Update axes
        fig.update_xaxes(title_text="Position (AU)", row=1, col=1)
        fig.update_xaxes(title_text="Position (AU)", row=1, col=2)
        fig.update_yaxes(title_text="Velocity (km/s)", row=1, col=1)
        fig.update_yaxes(title_text="Velocity (km/s)", row=1, col=2)
        
        fig.update_layout(
            title=f'Phase Space Diagram - {body_name}',
            **self.default_layout
        )
        
        return fig
    
    def create_slingshot_visualization(self, planet_name: str, 
                                     initial_velocity: float, 
                                     approach_angle: float) -> go.Figure:
        """Visualize gravitational slingshot maneuver"""
        # This would use pre-computed data in real implementation
        fig = go.Figure()
        
        # Planet
        fig.add_trace(go.Scatter(
            x=[0], y=[0],
            mode='markers',
            name=planet_name,
            marker=dict(size=30, color=self._get_color(planet_name))
        ))
        
        # Approach trajectory (simplified)
        t = np.linspace(-10, 10, 100)
        x_approach = t * np.cos(np.radians(approach_angle))
        y_approach = t * np.sin(np.radians(approach_angle)) + 5 * np.exp(-0.1 * np.abs(t))
        
        fig.add_trace(go.Scatter(
            x=x_approach,
            y=y_approach,
            mode='lines',
            name='Spacecraft trajectory',
            line=dict(color='green', width=3)
        ))
        
        # Velocity vectors
        fig.add_annotation(
            x=x_approach[0], y=y_approach[0],
            ax=x_approach[0] - 2, ay=y_approach[0],
            xref='x', yref='y',
            axref='x', ayref='y',
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=2,
            arrowcolor='red',
            text=f'v_in = {initial_velocity} km/s'
        )
        
        fig.update_layout(
            title=f'Gravitational Slingshot - {planet_name}',
            xaxis=dict(title='Distance (planetary radii)', scaleanchor='y'),
            yaxis=dict(title='Distance (planetary radii)'),
            **self.default_layout
        )
        
        return fig
    
    def create_sandbox_visualization(self, bodies: List[CelestialBody]) -> go.Figure:
        """Create real-time sandbox visualization"""
        fig = go.Figure()
        
        # Plot each body
        for body in bodies:
            color = self._get_color(body.name)
            
            # Body position
            fig.add_trace(go.Scatter(
                x=[body.position[0] / 1.496e11],
                y=[body.position[1] / 1.496e11],
                mode='markers',
                name=body.name,
                marker=dict(
                    size=self._get_marker_size(body.mass),
                    color=color,
                    line=dict(color='white', width=2)
                ),
                hovertemplate=(
                    f'<b>{body.name}</b><br>' +
                    'Position: (%{x:.3f}, %{y:.3f}) AU<br>' +
                    f'Mass: {body.mass:.2e} kg<br>' +
                    '<extra></extra>'
                )
            ))
            
            # Velocity vector
            if np.linalg.norm(body.velocity) > 0:
                fig.add_annotation(
                    x=body.position[0] / 1.496e11,
                    y=body.position[1] / 1.496e11,
                    ax=body.position[0] / 1.496e11 + body.velocity[0] / 3e7,
                    ay=body.position[1] / 1.496e11 + body.velocity[1] / 3e7,
                    xref='x', yref='y',
                    axref='x', ayref='y',
                    showarrow=True,
                    arrowhead=2,
                    arrowsize=1,
                    arrowwidth=2,
                    arrowcolor=color,
                    opacity=0.7
                )
        
        # Update layout
        max_dist = max([np.linalg.norm(b.position) for b in bodies]) / 1.496e11 * 1.5
        fig.update_layout(
            title='Gravity Sandbox - Current State',
            xaxis=dict(
                title='X Position (AU)',
                range=[-max_dist, max_dist],
                zeroline=True,
                gridcolor='rgba(255,255,255,0.1)'
            ),
            yaxis=dict(
                title='Y Position (AU)',
                range=[-max_dist, max_dist],
                zeroline=True,
                gridcolor='rgba(255,255,255,0.1)',
                scaleanchor='x',
                scaleratio=1
            ),
            **self.default_layout
        )
        
        return fig
    
    def create_tidal_force_visualization(self, primary: CelestialBody, 
                                       satellite: CelestialBody) -> go.Figure:
        """Visualize tidal forces on a body"""
        fig = go.Figure()
        
        # Calculate tidal force field around satellite
        n_points = 20
        angles = np.linspace(0, 2*np.pi, n_points)
        
        # Satellite surface points
        surface_x = satellite.position[0] + satellite.radius * np.cos(angles)
        surface_y = satellite.position[1] + satellite.radius * np.sin(angles)
        
        # Calculate tidal force at each point
        tidal_forces = []
        for x, y in zip(surface_x, surface_y):
            r_to_primary = np.sqrt(
                (x - primary.position[0])**2 + (y - primary.position[1])**2
            )
            r_to_satellite = satellite.radius
            
            # Simplified tidal force
            F_tidal = 2 * 6.67430e-11 * primary.mass * satellite.mass * r_to_satellite / r_to_primary**3
            tidal_forces.append(F_tidal)
        
        # Plot
        fig.add_trace(go.Scatterpolar(
            r=tidal_forces,
            theta=np.degrees(angles),
            mode='lines',
            name='Tidal Force',
            fill='toself',
            fillcolor='rgba(255,0,0,0.3)',
            line=dict(color='red')
        ))
        
        fig.update_layout(
            title=f'Tidal Forces on {satellite.name} from {primary.name}',
            polar=dict(
                radialaxis=dict(title='Tidal Force (N)')
            ),
            **self.default_layout
        )
        
        return fig
    
    def _get_color(self, body_name: str) -> str:
        """Get color for a celestial body"""
        for key in self.colors:
            if key.lower() in body_name.lower():
                return self.colors[key]
        return self.colors['default']
    
    def _get_marker_size(self, mass: float) -> float:
        """Calculate marker size based on mass"""
        # Logarithmic scaling
        earth_mass = 5.972e24
        base_size = 15
        
        if mass <= 0:
            return base_size
        
        size_factor = np.log10(mass / earth_mass) + 1
        return max(5, min(50, base_size * size_factor))