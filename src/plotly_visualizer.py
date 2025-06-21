"""
Plotly Physics Visualizer

Advanced Plotly-based visualization module for physics simulations in the
Gravity Yonder Over educational platform. Provides interactive 3D visualizations,
animated trajectories, field visualizations, and educational overlays.
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
import colorcet as cc
from scipy.ndimage import gaussian_filter


class PhysicsVisualizer:
    """
    Advanced Plotly visualizer for physics simulations.
    
    Features:
    - 3D gravitational field visualizations
    - Animated orbital trajectories
    - Potential energy surfaces
    - Vector field plots
    - Interactive educational annotations
    - Multi-scenario comparison plots
    """
    
    def __init__(self):
        # Color schemes for different physics types
        self.color_schemes = {
            "newtonian": {
                "primary": "#1f77b4",
                "secondary": "#ff7f0e",
                "field": "Viridis",
                "trajectory": ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
            },
            "relativistic": {
                "primary": "#d62728",
                "secondary": "#9467bd",
                "field": "Plasma",
                "trajectory": ["#d62728", "#9467bd", "#8c564b", "#e377c2"]
            },
            "orbital": {
                "primary": "#2ca02c",
                "secondary": "#ff7f0e",
                "field": "Cividis",
                "trajectory": ["#2ca02c", "#ff7f0e", "#1f77b4", "#d62728", "#9467bd"]
            }
        }
        
        # Default layout settings
        self.default_layout = {
            "scene": {
                "bgcolor": "rgba(0,0,0,0.1)",
                "xaxis": {"showgrid": True, "gridcolor": "rgba(255,255,255,0.3)"},
                "yaxis": {"showgrid": True, "gridcolor": "rgba(255,255,255,0.3)"},
                "zaxis": {"showgrid": True, "gridcolor": "rgba(255,255,255,0.3)"},
                "camera": {
                    "eye": {"x": 1.5, "y": 1.5, "z": 1.5}
                }
            },
            "margin": {"l": 0, "r": 0, "t": 30, "b": 0},
            "font": {"family": "Arial, sans-serif", "size": 12}
        }
    
    def create_3d_field_visualization(
        self, 
        simulation_data: Dict[str, Any], 
        visualization_type: str = "potential",
        slice_plane: str = "xy",
        slice_position: float = 0.0,
        show_vectors: bool = True,
        vector_scale: float = 1.0
    ) -> go.Figure:
        """
        Create 3D gravitational field visualization.
        
        Args:
            simulation_data: Simulation results dictionary
            visualization_type: 'potential', 'field_magnitude', 'curvature'
            slice_plane: 'xy', 'xz', 'yz' for 2D slice visualization
            slice_position: Position of the slice plane
            show_vectors: Whether to show vector field arrows
            vector_scale: Scaling factor for vector arrows
            
        Returns:
            Plotly Figure object
        """
        coordinates = simulation_data["coordinates"]
        X, Y, Z = coordinates["X"], coordinates["Y"], coordinates["Z"]
        physics_type = simulation_data.get("physics_type", "newtonian")
        colors = self.color_schemes[physics_type]
        
        fig = make_subplots(
            rows=1, cols=1,
            specs=[[{"type": "scene"}]],
            subplot_titles=[f"{visualization_type.title()} Field Visualization"]
        )
        
        # Extract slice data
        if slice_plane == "xy":
            # Find closest z index to slice_position
            z_idx = np.argmin(np.abs(Z[0, 0, :] - slice_position))
            x_slice = X[:, :, z_idx]
            y_slice = Y[:, :, z_idx]
            z_slice = np.full_like(x_slice, slice_position)
        elif slice_plane == "xz":
            # Find closest y index to slice_position
            y_idx = np.argmin(np.abs(Y[0, :, 0] - slice_position))
            x_slice = X[:, y_idx, :]
            y_slice = np.full_like(x_slice, slice_position)
            z_slice = Z[:, y_idx, :]
        else:  # yz
            # Find closest x index to slice_position
            x_idx = np.argmin(np.abs(X[:, 0, 0] - slice_position))
            x_slice = np.full_like(Y[x_idx, :, :], slice_position)
            y_slice = Y[x_idx, :, :]
            z_slice = Z[x_idx, :, :]
        
        # Get field values for visualization
        if visualization_type == "potential":
            field_values = self._extract_slice_values(simulation_data["potential"], slice_plane, slice_position, X, Y, Z)
            colorbar_title = "Gravitational Potential (J/kg)"
        elif visualization_type == "field_magnitude":
            fx = simulation_data["field_x"]
            fy = simulation_data["field_y"]
            fz = simulation_data["field_z"]
            field_magnitude = np.sqrt(fx**2 + fy**2 + fz**2)
            field_values = self._extract_slice_values(field_magnitude, slice_plane, slice_position, X, Y, Z)
            colorbar_title = "Field Strength (m/s²)"
        else:  # curvature approximation
            potential = simulation_data["potential"]
            # Simple curvature approximation using Laplacian
            laplacian = self._compute_laplacian(potential)
            field_values = self._extract_slice_values(laplacian, slice_plane, slice_position, X, Y, Z)
            colorbar_title = "Curvature (1/m²)"
        
        # Create surface plot
        field_values_smooth = gaussian_filter(field_values, sigma=1.0)
        
        surface = go.Surface(
            x=x_slice,
            y=y_slice,
            z=z_slice,
            surfacecolor=field_values_smooth,
            colorscale=colors["field"],
            opacity=0.8,
            colorbar=dict(title=colorbar_title, x=1.02),
            name=f"{visualization_type.title()} Field"
        )
        
        fig.add_trace(surface)
        
        # Add vector field if requested
        if show_vectors and visualization_type != "curvature":
            vector_traces = self._create_vector_field(
                simulation_data, slice_plane, slice_position, 
                X, Y, Z, vector_scale, colors["primary"]
            )
            for trace in vector_traces:
                fig.add_trace(trace)
        
        # Add special physics features
        if physics_type == "relativistic" and "schwarzschild_radius" in simulation_data:
            rs = simulation_data["schwarzschild_radius"]
            fig.add_trace(self._create_event_horizon(rs))
          # Update layout
        layout_dict = dict(self.default_layout)
        layout_dict["title"] = f"3D {visualization_type.title()} Field - {physics_type.title()} Physics"
        layout_dict["scene"]["aspectmode"] = "cube"
        
        fig.update_layout(layout_dict)
        
        return fig
    
    def create_trajectory_animation(
        self, 
        simulation_data: Dict[str, Any], 
        animation_speed: float = 50,
        show_trails: bool = True,
        trail_length: int = 100
    ) -> go.Figure:
        """
        Create animated trajectory visualization.
        
        Args:
            simulation_data: Simulation results with trajectory data
            animation_speed: Animation speed in ms per frame
            show_trails: Whether to show particle trails
            trail_length: Number of points in trails
            
        Returns:
            Animated Plotly Figure
        """
        physics_type = simulation_data.get("physics_type", "newtonian")
        colors = self.color_schemes[physics_type]["trajectory"]
        
        fig = go.Figure()
        
        # Handle different trajectory data structures
        if "trajectories" in simulation_data:
            # Binary system trajectories
            time = simulation_data["trajectories"]["time"]
            obj1_traj = simulation_data["trajectories"]["object1"]
            obj2_traj = simulation_data["trajectories"]["object2"]
            
            trajectories = [
                {"name": "Object 1", "data": obj1_traj, "color": colors[0]},
                {"name": "Object 2", "data": obj2_traj, "color": colors[1]}
            ]
            
        elif "planet_trajectories" in simulation_data:
            # Planetary system trajectories
            time = simulation_data["time"]
            trajectories = []
            
            for i, planet_data in enumerate(simulation_data["planet_trajectories"]):
                trajectories.append({
                    "name": f"Planet {i+1}",
                    "data": planet_data["trajectory"],
                    "color": colors[i % len(colors)]
                })
        
        elif "geodesics" in simulation_data:
            # Relativistic geodesics
            trajectories = []
            for i, geodesic in enumerate(simulation_data["geodesics"][:5]):  # Limit to 5 for clarity
                trajectories.append({
                    "name": f"Test Particle {i+1}",
                    "data": geodesic,
                    "color": colors[i % len(colors)]
                })
            time = np.linspace(0, 1000, len(trajectories[0]["data"]))
        
        else:
            # Default case - create simple test trajectory
            time = np.linspace(0, 100, 1000)
            radius = 1e11
            x = radius * np.cos(2 * np.pi * time / 100)
            y = radius * np.sin(2 * np.pi * time / 100)
            z = np.zeros_like(time)
            
            trajectories = [{
                "name": "Test Object",
                "data": np.column_stack([x, y, z]),
                "color": colors[0]
            }]
        
        # Create animation frames
        n_frames = min(200, len(time))  # Limit frames for performance
        frame_indices = np.linspace(0, len(time)-1, n_frames, dtype=int)
        
        frames = []
        for frame_idx, time_idx in enumerate(frame_indices):
            frame_data = []
            
            for traj in trajectories:
                traj_data = traj["data"]
                
                # Current position
                current_pos = traj_data[time_idx]
                
                # Trail
                if show_trails and time_idx > 0:
                    trail_start = max(0, time_idx - trail_length)
                    trail_x = traj_data[trail_start:time_idx+1, 0]
                    trail_y = traj_data[trail_start:time_idx+1, 1]
                    trail_z = traj_data[trail_start:time_idx+1, 2]
                    
                    # Trail trace
                    frame_data.append(                        go.Scatter3d(
                            x=trail_x,
                            y=trail_y,
                            z=trail_z,
                            mode='lines',
                            line=dict(color=traj["color"], width=2),
                            opacity=0.6,
                            name=f"{traj['name']} Trail",
                            showlegend=frame_idx == 0
                        )
                    )
                
                # Current position marker
                frame_data.append(
                    go.Scatter3d(
                        x=[current_pos[0]],
                        y=[current_pos[1]],
                        z=[current_pos[2]],
                        mode='markers',
                        marker=dict(
                            size=8,
                            color=traj["color"],
                            symbol='circle'
                        ),
                        name=traj["name"],
                        showlegend=frame_idx == 0
                    )
                )
            
            frames.append(go.Frame(data=frame_data, name=str(frame_idx)))
        
        # Initial frame
        fig.add_traces(frames[0].data)
        fig.frames = frames
        
        # Add animation controls
        fig.update_layout(
            **self.default_layout,
            title=f"Trajectory Animation - {physics_type.title()} Physics",
            updatemenus=[{
                "type": "buttons",
                "showactive": False,
                "buttons": [
                    {
                        "label": "Play",
                        "method": "animate",
                        "args": [None, {
                            "frame": {"duration": animation_speed, "redraw": True},
                            "fromcurrent": True,
                            "transition": {"duration": 0}
                        }]
                    },
                    {
                        "label": "Pause",
                        "method": "animate",
                        "args": [[None], {
                            "frame": {"duration": 0, "redraw": False},
                            "mode": "immediate",
                            "transition": {"duration": 0}
                        }]
                    }
                ],
                "x": 0.1,
                "y": 0.1
            }],
            sliders=[{
                "steps": [
                    {
                        "args": [[str(i)], {
                            "frame": {"duration": 0, "redraw": True},
                            "mode": "immediate"
                        }],
                        "label": f"Frame {i}",
                        "method": "animate"
                    }
                    for i in range(len(frames))
                ],
                "active": 0,
                "x": 0.1,
                "len": 0.9,
                "y": 0.02
            }]
        )
        
        return fig
    
    def create_educational_comparison(
        self, 
        simulation_data_list: List[Dict[str, Any]], 
        scenario_names: List[str],
        comparison_type: str = "potential"
    ) -> go.Figure:
        """
        Create educational comparison visualization.
        
        Args:
            simulation_data_list: List of simulation results to compare
            scenario_names: Names for each scenario
            comparison_type: Type of comparison ('potential', 'trajectories', 'fields')
            
        Returns:
            Comparison plot figure
        """
        if comparison_type == "potential":
            return self._create_potential_comparison(simulation_data_list, scenario_names)
        elif comparison_type == "trajectories":
            return self._create_trajectory_comparison(simulation_data_list, scenario_names)
        elif comparison_type == "fields":
            return self._create_field_comparison(simulation_data_list, scenario_names)
        else:
            raise ValueError(f"Unknown comparison type: {comparison_type}")
    
    def create_interactive_dashboard(
        self, 
        simulation_data: Dict[str, Any]
    ) -> Dict[str, go.Figure]:
        """
        Create comprehensive interactive dashboard.
        
        Args:
            simulation_data: Complete simulation results
            
        Returns:
            Dictionary of figures for dashboard display
        """
        physics_type = simulation_data.get("physics_type", "newtonian")
        
        dashboard = {}
        
        # 1. Main 3D visualization
        dashboard["main_3d"] = self.create_3d_field_visualization(
            simulation_data, 
            visualization_type="potential"
        )
        
        # 2. Field magnitude plot
        dashboard["field_magnitude"] = self.create_3d_field_visualization(
            simulation_data, 
            visualization_type="field_magnitude"
        )
        
        # 3. Trajectory animation (if available)
        if any(key in simulation_data for key in ["trajectories", "planet_trajectories", "geodesics"]):
            dashboard["trajectories"] = self.create_trajectory_animation(simulation_data)
        
        # 4. Cross-section plots
        dashboard["cross_sections"] = self._create_cross_section_plots(simulation_data)
        
        # 5. Physics-specific visualizations
        if physics_type == "relativistic":
            dashboard["relativistic_effects"] = self._create_relativistic_effects_plot(simulation_data)
        elif physics_type == "orbital":
            dashboard["orbital_analysis"] = self._create_orbital_analysis_plot(simulation_data)
        
        # 6. Educational annotations
        dashboard["educational_info"] = self._create_educational_info_plot(simulation_data)
        
        return dashboard
    
    def _extract_slice_values(self, field_3d, slice_plane, slice_position, X, Y, Z):
        """Extract 2D slice from 3D field data"""
        if slice_plane == "xy":
            z_idx = np.argmin(np.abs(Z[0, 0, :] - slice_position))
            return field_3d[:, :, z_idx]
        elif slice_plane == "xz":
            y_idx = np.argmin(np.abs(Y[0, :, 0] - slice_position))
            return field_3d[:, y_idx, :]
        else:  # yz
            x_idx = np.argmin(np.abs(X[:, 0, 0] - slice_position))
            return field_3d[x_idx, :, :]
    
    def _create_vector_field(self, simulation_data, slice_plane, slice_position, X, Y, Z, vector_scale, color):
        """Create vector field visualization"""
        fx = simulation_data["field_x"]
        fy = simulation_data["field_y"]
        fz = simulation_data["field_z"]
        
        # Extract slice coordinates and vectors
        if slice_plane == "xy":
            z_idx = np.argmin(np.abs(Z[0, 0, :] - slice_position))
            x_coords = X[:, :, z_idx]
            y_coords = Y[:, :, z_idx]
            z_coords = np.full_like(x_coords, slice_position)
            u = fx[:, :, z_idx]
            v = fy[:, :, z_idx]
            w = fz[:, :, z_idx]
        elif slice_plane == "xz":
            y_idx = np.argmin(np.abs(Y[0, :, 0] - slice_position))
            x_coords = X[:, y_idx, :]
            y_coords = np.full_like(x_coords, slice_position)
            z_coords = Z[:, y_idx, :]
            u = fx[:, y_idx, :]
            v = fy[:, y_idx, :]
            w = fz[:, y_idx, :]
        else:  # yz
            x_idx = np.argmin(np.abs(X[:, 0, 0] - slice_position))
            x_coords = np.full_like(Y[x_idx, :, :], slice_position)
            y_coords = Y[x_idx, :, :]
            z_coords = Z[x_idx, :, :]
            u = fx[x_idx, :, :]
            v = fy[x_idx, :, :]
            w = fz[x_idx, :, :]
        
        # Subsample for clarity
        step = max(1, min(x_coords.shape) // 20)
        x_sub = x_coords[::step, ::step].flatten()
        y_sub = y_coords[::step, ::step].flatten()
        z_sub = z_coords[::step, ::step].flatten()
        u_sub = u[::step, ::step].flatten() * vector_scale
        v_sub = v[::step, ::step].flatten() * vector_scale
        w_sub = w[::step, ::step].flatten() * vector_scale
        
        # Create vector traces
        traces = []
        for i in range(len(x_sub)):
            if np.sqrt(u_sub[i]**2 + v_sub[i]**2 + w_sub[i]**2) > 1e-20:  # Skip zero vectors
                traces.append(
                    go.Scatter3d(
                        x=[x_sub[i], x_sub[i] + u_sub[i]],
                        y=[y_sub[i], y_sub[i] + v_sub[i]],
                        z=[z_sub[i], z_sub[i] + w_sub[i]],
                        mode='lines',
                        line=dict(color=color, width=2),
                        showlegend=False,
                        hoverinfo='skip'
                    )
                )
        
        return traces
    
    def _compute_laplacian(self, field):
        """Compute Laplacian for curvature approximation"""
        # Simple finite difference Laplacian
        laplacian = np.zeros_like(field)
        
        # Interior points
        laplacian[1:-1, 1:-1, 1:-1] = (
            field[2:, 1:-1, 1:-1] + field[:-2, 1:-1, 1:-1] +
            field[1:-1, 2:, 1:-1] + field[1:-1, :-2, 1:-1] +
            field[1:-1, 1:-1, 2:] + field[1:-1, 1:-1, :-2] -
            6 * field[1:-1, 1:-1, 1:-1]
        )
        
        return laplacian
    
    def _create_event_horizon(self, schwarzschild_radius):
        """Create event horizon visualization for black holes"""
        # Create sphere at Schwarzschild radius
        theta = np.linspace(0, 2*np.pi, 30)
        phi = np.linspace(0, np.pi, 20)
        
        theta_grid, phi_grid = np.meshgrid(theta, phi)
        
        x = schwarzschild_radius * np.sin(phi_grid) * np.cos(theta_grid)
        y = schwarzschild_radius * np.sin(phi_grid) * np.sin(theta_grid)
        z = schwarzschild_radius * np.cos(phi_grid)
        
        return go.Surface(
            x=x, y=y, z=z,
            colorscale=[[0, 'black'], [1, 'black']],
            showscale=False,
            opacity=0.8,
            name="Event Horizon"
        )
    
    def _create_potential_comparison(self, simulation_data_list, scenario_names):
        """Create potential energy comparison plot"""
        fig = make_subplots(
            rows=1, cols=len(simulation_data_list),
            subplot_titles=scenario_names,
            specs=[[{"type": "scene"}] * len(simulation_data_list)]
        )
        
        for i, (data, name) in enumerate(zip(simulation_data_list, scenario_names)):
            # Extract central slice
            coords = data["coordinates"]
            X, Y, Z = coords["X"], coords["Y"], coords["Z"]
            z_center_idx = Z.shape[2] // 2
            
            x_slice = X[:, :, z_center_idx]
            y_slice = Y[:, :, z_center_idx]
            potential_slice = data["potential"][:, :, z_center_idx]
            
            surface = go.Surface(
                x=x_slice,
                y=y_slice,
                z=potential_slice,
                colorscale="Viridis",
                name=name
            )
            
            fig.add_trace(surface, row=1, col=i+1)
        
        fig.update_layout(
            title="Gravitational Potential Comparison",
            **self.default_layout
        )
        
        return fig
    
    def _create_trajectory_comparison(self, simulation_data_list, scenario_names):
        """Create trajectory comparison plot"""
        fig = go.Figure()
        
        colors = px.colors.qualitative.Set1
        
        for i, (data, name) in enumerate(zip(simulation_data_list, scenario_names)):
            color = colors[i % len(colors)]
            
            if "trajectories" in data:
                # Binary system
                obj1 = data["trajectories"]["object1"]
                obj2 = data["trajectories"]["object2"]
                
                fig.add_trace(go.Scatter3d(
                    x=obj1[:, 0], y=obj1[:, 1], z=obj1[:, 2],
                    mode='lines',
                    name=f"{name} - Object 1",
                    line=dict(color=color, width=3)
                ))
                
                fig.add_trace(go.Scatter3d(
                    x=obj2[:, 0], y=obj2[:, 1], z=obj2[:, 2],
                    mode='lines',
                    name=f"{name} - Object 2",
                    line=dict(color=color, width=3, dash='dash')
                ))
            
            elif "planet_trajectories" in data:
                # Planetary system
                for j, planet in enumerate(data["planet_trajectories"]):
                    traj = planet["trajectory"]
                    fig.add_trace(go.Scatter3d(
                        x=traj[:, 0], y=traj[:, 1], z=traj[:, 2],
                        mode='lines',
                        name=f"{name} - Planet {j+1}",
                        line=dict(color=color, width=2)
                    ))
        
        fig.update_layout(
            **self.default_layout,
            title="Trajectory Comparison"
        )
        
        return fig
    
    def _create_field_comparison(self, simulation_data_list, scenario_names):
        """Create field strength comparison plot"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=["Field Magnitude", "X Component", "Y Component", "Z Component"],
            specs=[[{"type": "heatmap"}, {"type": "heatmap"}],
                   [{"type": "heatmap"}, {"type": "heatmap"}]]
        )
        
        # Use first scenario for layout, compare central slices
        data = simulation_data_list[0]
        coords = data["coordinates"]
        X, Y, Z = coords["X"], coords["Y"], coords["Z"]
        z_center = Z.shape[2] // 2
        
        x_2d = X[:, :, z_center]
        y_2d = Y[:, :, z_center]
        
        # Calculate average fields across all scenarios
        for i, (data, name) in enumerate(zip(simulation_data_list, scenario_names)):
            fx = data["field_x"][:, :, z_center]
            fy = data["field_y"][:, :, z_center]
            fz = data["field_z"][:, :, z_center]
            field_mag = np.sqrt(fx**2 + fy**2 + fz**2)
            
            opacity = 0.7 if i > 0 else 1.0
            
            # Field magnitude
            fig.add_trace(go.Heatmap(
                x=x_2d[0, :], y=y_2d[:, 0], z=field_mag,
                colorscale="Viridis", name=f"{name} Magnitude",
                opacity=opacity
            ), row=1, col=1)
            
            # Components
            fig.add_trace(go.Heatmap(
                x=x_2d[0, :], y=y_2d[:, 0], z=fx,
                colorscale="RdBu", name=f"{name} Fx",
                opacity=opacity
            ), row=1, col=2)
            
            fig.add_trace(go.Heatmap(
                x=x_2d[0, :], y=y_2d[:, 0], z=fy,
                colorscale="RdBu", name=f"{name} Fy",
                opacity=opacity
            ), row=2, col=1)
            
            fig.add_trace(go.Heatmap(
                x=x_2d[0, :], y=y_2d[:, 0], z=fz,
                colorscale="RdBu", name=f"{name} Fz",
                opacity=opacity
            ), row=2, col=2)
        
        fig.update_layout(
            title="Gravitational Field Comparison",
            height=800
        )
        
        return fig
    
    def _create_cross_section_plots(self, simulation_data):
        """Create cross-section analysis plots"""
        coords = simulation_data["coordinates"]
        X, Y, Z = coords["X"], coords["Y"], coords["Z"]
        potential = simulation_data["potential"]
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=["XY Plane (z=0)", "XZ Plane (y=0)", "YZ Plane (x=0)", "Radial Profile"],
            specs=[[{"type": "heatmap"}, {"type": "heatmap"}],
                   [{"type": "heatmap"}, {"type": "xy"}]]
        )
        
        # XY plane
        z_center = Z.shape[2] // 2
        fig.add_trace(go.Heatmap(
            x=X[0, :, z_center], y=Y[:, 0, z_center], z=potential[:, :, z_center],
            colorscale="Viridis", name="XY Potential"
        ), row=1, col=1)
        
        # XZ plane
        y_center = Y.shape[1] // 2
        fig.add_trace(go.Heatmap(
            x=X[:, y_center, 0], y=Z[:, y_center, :], z=potential[:, y_center, :],
            colorscale="Viridis", name="XZ Potential"
        ), row=1, col=2)
        
        # YZ plane
        x_center = X.shape[0] // 2
        fig.add_trace(go.Heatmap(
            x=Y[x_center, :, 0], y=Z[x_center, :, :], z=potential[x_center, :, :],
            colorscale="Viridis", name="YZ Potential"
        ), row=2, col=1)
        
        # Radial profile
        center_idx = [s//2 for s in X.shape]
        max_radius = min(X.shape) // 2
        radii = []
        potential_radial = []
        
        for r in range(1, max_radius):
            # Sample points at radius r
            theta = np.linspace(0, 2*np.pi, 20)
            phi = np.linspace(0, np.pi, 10)
            
            pot_samples = []
            for t in theta[:5]:  # Sample subset for efficiency
                for p in phi[:3]:
                    i = center_idx[0] + int(r * np.sin(p) * np.cos(t))
                    j = center_idx[1] + int(r * np.sin(p) * np.sin(t))
                    k = center_idx[2] + int(r * np.cos(p))
                    
                    if 0 <= i < X.shape[0] and 0 <= j < X.shape[1] and 0 <= k < X.shape[2]:
                        pot_samples.append(potential[i, j, k])
            
            if pot_samples:
                radii.append(r * np.abs(X[1,0,0] - X[0,0,0]))  # Convert to physical units
                potential_radial.append(np.mean(pot_samples))
        
        fig.add_trace(go.Scatter(
            x=radii, y=potential_radial,
            mode='lines+markers',
            name="Radial Profile"
        ), row=2, col=2)
        
        fig.update_layout(
            title="Cross-Section Analysis",
            height=800
        )
        
        return fig
    
    def _create_relativistic_effects_plot(self, simulation_data):
        """Create relativistic effects visualization"""
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=["Frame Dragging Effect", "Geodesic Trajectories"],
            specs=[[{"type": "heatmap"}, {"type": "scene"}]]
        )
        
        # Frame dragging
        if "frame_dragging" in simulation_data:
            coords = simulation_data["coordinates"]
            X, Y, Z = coords["X"], coords["Y"], coords["Z"]
            z_center = Z.shape[2] // 2
            
            fig.add_trace(go.Heatmap(
                x=X[0, :, z_center], 
                y=Y[:, 0, z_center], 
                z=simulation_data["frame_dragging"][:, :, z_center],
                colorscale="Plasma", 
                name="Frame Dragging"
            ), row=1, col=1)
        
        # Geodesics
        if "geodesics" in simulation_data:
            colors = px.colors.qualitative.Set1
            for i, geodesic in enumerate(simulation_data["geodesics"][:5]):
                color = colors[i % len(colors)]
                fig.add_trace(go.Scatter3d(
                    x=geodesic[:, 0],
                    y=geodesic[:, 1],
                    z=geodesic[:, 2],
                    mode='lines',
                    line=dict(color=color, width=3),
                    name=f"Geodesic {i+1}"
                ), row=1, col=2)
        
        # Add event horizon if present
        if "schwarzschild_radius" in simulation_data:
            rs = simulation_data["schwarzschild_radius"]
            fig.add_trace(self._create_event_horizon(rs), row=1, col=2)
        
        fig.update_layout(
            title="Relativistic Effects",
            height=600
        )
        
        return fig
    
    def _create_orbital_analysis_plot(self, simulation_data):
        """Create orbital mechanics analysis"""
        if "planet_trajectories" not in simulation_data:
            return go.Figure()
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=["Orbital Radii", "Orbital Periods", "Energy Analysis", "3D Orbits"],
            specs=[[{"type": "xy"}, {"type": "xy"}],
                   [{"type": "xy"}, {"type": "scene"}]]
        )
        
        planet_data = simulation_data["planet_trajectories"]
        colors = px.colors.qualitative.Set1
        
        # Extract orbital parameters
        radii = [planet["orbital_radius"] for planet in planet_data]
        periods = [planet["period"] for planet in planet_data]
        masses = [planet["mass"] for planet in planet_data]
        
        # Orbital radii
        fig.add_trace(go.Bar(
            x=[f"Planet {i+1}" for i in range(len(radii))],
            y=[r/1.5e11 for r in radii],  # Convert to AU
            name="Orbital Radius (AU)",
            marker_color=colors[0]
        ), row=1, col=1)
        
        # Orbital periods
        fig.add_trace(go.Bar(
            x=[f"Planet {i+1}" for i in range(len(periods))],
            y=[p/(365.25*24*3600) for p in periods],  # Convert to years
            name="Orbital Period (years)",
            marker_color=colors[1]
        ), row=1, col=2)
        
        # Kepler's 3rd law verification
        theoretical_periods = [2*np.pi*np.sqrt(r**3/(6.67430e-11*simulation_data["central_mass"])) for r in radii]
        
        fig.add_trace(go.Scatter(
            x=[r/1.5e11 for r in radii],
            y=[p/(365.25*24*3600) for p in periods],
            mode='markers',
            name="Observed",
            marker=dict(color=colors[2], size=10)
        ), row=2, col=1)
        
        fig.add_trace(go.Scatter(
            x=[r/1.5e11 for r in radii],
            y=[p/(365.25*24*3600) for p in theoretical_periods],
            mode='lines',
            name="Kepler's 3rd Law",
            line=dict(color=colors[3], dash='dash')
        ), row=2, col=1)
        
        # 3D orbits
        for i, planet in enumerate(planet_data):
            traj = planet["trajectory"]
            color = colors[i % len(colors)]
            
            fig.add_trace(go.Scatter3d(
                x=traj[:, 0]/1.5e11,  # Convert to AU
                y=traj[:, 1]/1.5e11,
                z=traj[:, 2]/1.5e11,
                mode='lines',
                name=f"Planet {i+1}",
                line=dict(color=color, width=3)
            ), row=2, col=2)
        
        # Add central star
        fig.add_trace(go.Scatter3d(
            x=[0], y=[0], z=[0],
            mode='markers',
            marker=dict(color='yellow', size=15, symbol='circle'),
            name="Central Star"
        ), row=2, col=2)
        
        fig.update_layout(
            title="Orbital Mechanics Analysis",
            height=800
        )
        
        return fig
    
    def _create_educational_info_plot(self, simulation_data):
        """Create educational information display"""
        physics_type = simulation_data.get("physics_type", "newtonian")
        
        # Create text annotations with key physics concepts
        info_text = self._generate_physics_explanation(simulation_data)
        
        fig = go.Figure()
        
        fig.add_annotation(
            text=info_text,
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            xanchor="center", yanchor="middle",
            showarrow=False,
            font=dict(size=14, family="Arial"),
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="rgba(0,0,0,0.2)",
            borderwidth=1
        )
        
        fig.update_layout(
            title=f"{physics_type.title()} Physics - Key Concepts",
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            height=400,
            margin=dict(l=20, r=20, t=50, b=20)
        )
        
        return fig
    
    def _generate_physics_explanation(self, simulation_data):
        """Generate educational explanation text"""
        physics_type = simulation_data.get("physics_type", "newtonian")
        
        if physics_type == "newtonian":
            return """
<b>Newtonian Gravity</b><br><br>
• <b>Inverse Square Law:</b> F = G·m₁·m₂/r²<br>
• <b>Superposition:</b> Total field is sum of individual fields<br>
• <b>Conservative Force:</b> Work is path-independent<br>
• <b>Orbital Motion:</b> Balance of gravity and centrifugal force<br><br>
<b>Key Insights:</b><br>
- Gravitational potential energy decreases as 1/r<br>
- Circular orbits require specific velocity for each radius<br>
- Escape velocity: v = √(2GM/r)
"""
        elif physics_type == "relativistic":
            return """
<b>General Relativity</b><br><br>
• <b>Spacetime Curvature:</b> Mass-energy curves spacetime<br>
• <b>Geodesics:</b> Particles follow straightest paths in curved space<br>
• <b>Event Horizon:</b> Point of no return around black holes<br>
• <b>Frame Dragging:</b> Rotating masses drag spacetime<br><br>
<b>Key Effects:</b><br>
- Time dilation near massive objects<br>
- Light bending around massive bodies<br>
- Gravitational waves from accelerating masses
"""
        elif physics_type == "orbital":
            return """
<b>Orbital Mechanics</b><br><br>
• <b>Kepler's Laws:</b> Planetary motion principles<br>
• <b>Orbital Elements:</b> Six parameters define orbit<br>
• <b>Energy Conservation:</b> E = K + U = constant<br>
• <b>Angular Momentum:</b> L = r × mv conserved<br><br>
<b>Orbit Types:</b><br>
- Circular: e = 0, constant radius<br>
- Elliptical: 0 < e < 1, varying radius<br>
- Parabolic: e = 1, escape trajectory<br>
- Hyperbolic: e > 1, unbound motion
"""
        else:
            return "Physics simulation visualization"
