"""
Interactive Video Simulation Engine

This module integrates multiple simulation libraries to create rich, interactive
physics simulations with video output capabilities.
"""

import numpy as np
import pygame
import cv2
import imageio
from typing import Dict, List, Tuple, Optional, Any
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pybullet as p
import pymunk
import streamlit as st
from pathlib import Path
import tempfile
import base64


class InteractiveSimulationEngine:
    """
    Advanced simulation engine combining multiple libraries for interactive physics simulations
    """
    
    def __init__(self):
        self.pygame_initialized = False
        self.pybullet_initialized = False
        self.pymunk_initialized = False
        self.temp_dir = Path(tempfile.mkdtemp())
        
    def initialize_pygame(self, width: int = 800, height: int = 600):
        """Initialize pygame for 2D interactive simulations"""
        if not self.pygame_initialized:
            pygame.init()
            self.pygame_screen = pygame.display.set_mode((width, height))
            pygame.display.set_caption("Gravity Simulation")
            self.pygame_clock = pygame.time.Clock()
            self.pygame_initialized = True
    
    def initialize_pybullet(self):
        """Initialize PyBullet for 3D physics simulations"""
        if not self.pybullet_initialized:
            p.connect(p.DIRECT)  # Use DIRECT mode for headless operation
            p.setGravity(0, 0, -9.81)
            self.pybullet_initialized = True
    
    def initialize_pymunk(self):
        """Initialize Pymunk for 2D physics simulations"""
        if not self.pymunk_initialized:
            self.pymunk_space = pymunk.Space()
            self.pymunk_space.gravity = (0, -981)  # Gravity in pixels
            self.pymunk_initialized = True
    
    def create_pygame_orbital_simulation(self, 
                                       mass1: float, 
                                       mass2: float, 
                                       separation: float,
                                       duration: int = 10) -> str:
        """
        Create an interactive orbital simulation using pygame
        
        Returns:
            Path to generated video file
        """
        self.initialize_pygame()
        
        # Simulation parameters
        G = 6.67430e-11  # Gravitational constant
        dt = 0.016  # Time step (60 FPS)
        frames = int(duration * 60)  # Duration in seconds * 60 FPS
        
        # Scale factors for visualization
        scale = 1e-9  # Scale from meters to pixels
        center_x, center_y = 400, 300
        
        # Initial positions and velocities
        m1_pos = np.array([-separation/3 * scale + center_x, center_y])
        m2_pos = np.array([separation*2/3 * scale + center_x, center_y])
        
        # Calculate orbital velocity
        v_orbit = np.sqrt(G * (mass1 + mass2) / separation)
        m1_vel = np.array([0, -v_orbit * mass2/(mass1 + mass2) * scale])
        m2_vel = np.array([0, v_orbit * mass1/(mass1 + mass2) * scale])
        
        # Store frames for video
        frames_list = []
        
        # Simulation loop
        for frame in range(frames):
            # Clear screen
            self.pygame_screen.fill((0, 0, 0))
            
            # Calculate gravitational force
            r_vec = m2_pos - m1_pos
            r_mag = np.linalg.norm(r_vec)
            r_hat = r_vec / r_mag
            
            # Force magnitude
            F_mag = G * mass1 * mass2 / (r_mag / scale)**2
            F_vec = F_mag * r_hat
            
            # Update velocities
            m1_vel += F_vec / mass1 * dt * scale
            m2_vel -= F_vec / mass2 * dt * scale
            
            # Update positions
            m1_pos += m1_vel * dt
            m2_pos += m2_vel * dt
            
            # Draw objects
            pygame.draw.circle(self.pygame_screen, (255, 255, 0), 
                             (int(m1_pos[0]), int(m1_pos[1])), 
                             max(5, int(np.log10(mass1/1e24))))
            pygame.draw.circle(self.pygame_screen, (100, 100, 255), 
                             (int(m2_pos[0]), int(m2_pos[1])), 
                             max(3, int(np.log10(mass2/1e24))))
            
            # Add trajectory trails
            if frame > 0:
                trail_length = min(50, frame)
                for i in range(trail_length):
                    alpha = i / trail_length
                    # This is a simplified trail - in a real implementation,
                    # you'd store position history
            
            # Convert pygame surface to numpy array
            frame_array = pygame.surfarray.array3d(self.pygame_screen)
            frame_array = np.rot90(frame_array, -1)
            frame_array = np.flipud(frame_array)
            frames_list.append(frame_array)
            
            pygame.display.flip()
            self.pygame_clock.tick(60)
        
        # Save as video
        video_path = self.temp_dir / f"orbital_simulation_{int(mass1/1e30)}_{int(mass2/1e24)}.mp4"
        imageio.mimsave(str(video_path), frames_list, fps=60, quality=8)
        
        return str(video_path)
    
    def create_pybullet_3d_simulation(self, 
                                     objects: List[Dict],
                                     duration: int = 10) -> str:
        """
        Create a 3D physics simulation using PyBullet
        
        Args:
            objects: List of objects with mass, position, etc.
            duration: Simulation duration in seconds
            
        Returns:
            Path to generated video file
        """
        self.initialize_pybullet()
        
        # Create objects in PyBullet
        object_ids = []
        for obj in objects:
            # Create sphere
            sphere_radius = max(0.1, obj.get('radius', 1.0))
            sphere_mass = obj.get('mass', 1.0)
            
            # Create collision shape
            collision_shape = p.createCollisionShape(p.GEOM_SPHERE, radius=sphere_radius)
            visual_shape = p.createVisualShape(p.GEOM_SPHERE, radius=sphere_radius,
                                             rgbaColor=obj.get('color', [1, 1, 0, 1]))
            
            # Create multibody
            body_id = p.createMultiBody(baseMass=sphere_mass,
                                       baseCollisionShapeIndex=collision_shape,
                                       baseVisualShapeIndex=visual_shape,
                                       basePosition=obj.get('position', [0, 0, 0]))
            object_ids.append(body_id)
        
        # Simulation parameters
        dt = 1./240.  # Time step
        frames = int(duration * 60)  # 60 FPS output
        
        # Store frames
        frames_list = []
        
        for frame in range(frames):
            # Step simulation
            p.stepSimulation()
            
            # Get camera image
            width, height = 640, 480
            view_matrix = p.computeViewMatrix(
                cameraEyePosition=[3, 3, 3],
                cameraTargetPosition=[0, 0, 0],
                cameraUpVector=[0, 0, 1]
            )
            proj_matrix = p.computeProjectionMatrixFOV(
                fov=60, aspect=width/height, nearVal=0.1, farVal=100.0
            )
            
            # Render
            _, _, rgba, _, _ = p.getCameraImage(
                width=width, height=height,
                viewMatrix=view_matrix,
                projectionMatrix=proj_matrix
            )
            
            # Convert to RGB
            rgb_array = np.array(rgba).reshape(height, width, 4)[:, :, :3]
            frames_list.append(rgb_array)
        
        # Save video
        video_path = self.temp_dir / f"pybullet_simulation_{len(objects)}_objects.mp4"
        imageio.mimsave(str(video_path), frames_list, fps=60, quality=8)
        
        return str(video_path)
    
    def create_pymunk_2d_simulation(self, 
                                   masses: List[float],
                                   positions: List[Tuple[float, float]],
                                   duration: int = 10) -> str:
        """
        Create a 2D physics simulation using Pymunk
        
        Returns:
            Path to generated video file
        """
        self.initialize_pymunk()
        self.initialize_pygame(800, 600)
        
        # Create bodies
        bodies = []
        for mass, pos in zip(masses, positions):
            body = pymunk.Body(mass, pymunk.moment_for_circle(mass, 0, 20))
            body.position = pos
            shape = pymunk.Circle(body, 20)
            shape.friction = 0.7
            self.pymunk_space.add(body, shape)
            bodies.append((body, shape))
        
        # Simulation parameters
        dt = 1/60.0
        frames = int(duration * 60)
        frames_list = []
        
        for frame in range(frames):
            # Step physics
            self.pymunk_space.step(dt)
            
            # Clear screen
            self.pygame_screen.fill((0, 0, 0))
            
            # Draw bodies
            for i, (body, shape) in enumerate(bodies):
                pos = int(body.position.x), int(body.position.y)
                color = [(255, 255, 0), (100, 100, 255), (255, 100, 100)][i % 3]
                pygame.draw.circle(self.pygame_screen, color, pos, int(shape.radius))
            
            # Convert to array
            frame_array = pygame.surfarray.array3d(self.pygame_screen)
            frame_array = np.rot90(frame_array, -1)
            frame_array = np.flipud(frame_array)
            frames_list.append(frame_array)
            
            pygame.display.flip()
        
        # Save video
        video_path = self.temp_dir / f"pymunk_simulation_{len(masses)}_bodies.mp4"
        imageio.mimsave(str(video_path), frames_list, fps=60, quality=8)
        
        return str(video_path)
    
    def create_plotly_animated_simulation(self, 
                                        simulation_data: Dict[str, Any]) -> go.Figure:
        """
        Create an animated Plotly visualization
        
        Args:
            simulation_data: Simulation data with trajectories
            
        Returns:
            Plotly animated figure
        """
        if 'trajectories' not in simulation_data:
            return go.Figure()
        
        trajectories = simulation_data['trajectories']
        time_points = trajectories.get('time', np.linspace(0, 10, 100))
        
        # Create frames for animation
        frames = []
        
        # Get trajectory data
        if 'object1' in trajectories and 'object2' in trajectories:
            traj1 = trajectories['object1']
            traj2 = trajectories['object2']
            
            # Create frames
            for i in range(0, len(time_points), max(1, len(time_points)//50)):
                frame_data = []
                
                # Object 1 trajectory
                frame_data.append(go.Scatter3d(
                    x=traj1[:i+1, 0],
                    y=traj1[:i+1, 1],
                    z=traj1[:i+1, 2],
                    mode='lines+markers',
                    name='Object 1',
                    line=dict(color='yellow', width=3),
                    marker=dict(size=8, color='yellow')
                ))
                
                # Object 2 trajectory
                frame_data.append(go.Scatter3d(
                    x=traj2[:i+1, 0],
                    y=traj2[:i+1, 1],
                    z=traj2[:i+1, 2],
                    mode='lines+markers',
                    name='Object 2',
                    line=dict(color='blue', width=3),
                    marker=dict(size=6, color='blue')
                ))
                
                frames.append(go.Frame(data=frame_data, name=f"frame_{i}"))
        
        # Create initial figure
        fig = go.Figure(
            data=frames[0].data if frames else [],
            frames=frames
        )
        
        # Add play button
        fig.update_layout(
            title="Animated Orbital Simulation",
            scene=dict(
                xaxis_title="X (m)",
                yaxis_title="Y (m)",
                zaxis_title="Z (m)",
                aspectmode="cube"
            ),
            updatemenus=[{
                'type': 'buttons',
                'buttons': [
                    {
                        'label': 'Play',
                        'method': 'animate',
                        'args': [None, {
                            'frame': {'duration': 100, 'redraw': True},
                            'fromcurrent': True,
                            'transition': {'duration': 50}
                        }]
                    },
                    {
                        'label': 'Pause',
                        'method': 'animate',
                        'args': [[None], {
                            'frame': {'duration': 0, 'redraw': False},
                            'mode': 'immediate',
                            'transition': {'duration': 0}
                        }]
                    }
                ],
                'direction': 'left',
                'pad': {'r': 10, 't': 87},
                'showactive': False,
                'x': 0.1,
                'xanchor': 'right',
                'y': 0,
                'yanchor': 'top'
            }]
        )
        
        return fig
    
    def create_video_from_plotly(self, fig: go.Figure, filename: str = "plotly_animation.mp4") -> str:
        """
        Convert Plotly animation to video file
        
        Args:
            fig: Plotly figure with frames
            filename: Output filename
            
        Returns:
            Path to video file
        """
        # This is a simplified version - in practice, you'd need to render each frame
        # and combine them into a video
        video_path = self.temp_dir / filename
        
        # For now, we'll save the HTML and return the path
        # In a full implementation, you'd use a headless browser to render frames
        html_path = str(video_path).replace('.mp4', '.html')
        fig.write_html(html_path)
        
        return html_path
    
    def get_video_base64(self, video_path: str) -> str:
        """
        Convert video file to base64 for embedding in Streamlit
        
        Args:
            video_path: Path to video file
            
        Returns:
            Base64 encoded video
        """
        try:
            with open(video_path, "rb") as video_file:
                video_bytes = video_file.read()
                video_base64 = base64.b64encode(video_bytes).decode()
                return video_base64
        except Exception as e:
            st.error(f"Error encoding video: {e}")
            return ""
    
    def cleanup(self):
        """Clean up resources"""
        if self.pygame_initialized:
            pygame.quit()
        
        if self.pybullet_initialized:
            p.disconnect()
        
        # Clean up temporary files
        import shutil
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)


# Global instance
interactive_engine = InteractiveSimulationEngine()
