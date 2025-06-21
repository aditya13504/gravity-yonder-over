"""
Streamlit Game Runner for Educational Physics Games
Handles pygame integration with Streamlit web interface
"""

import streamlit as st
import base64
import time
import threading
from typing import Dict, Any, Optional
import json
from src.educational_games import GameManager
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class StreamlitGameRunner:
    """
    Runs educational games within Streamlit interface
    """
    
    def __init__(self):
        self.game_manager = GameManager()
        self.initialize_session_state()
    
    def initialize_session_state(self):
        """Initialize Streamlit session state for games"""
        if 'games_state' not in st.session_state:
            st.session_state.games_state = {}
        
        if 'current_game' not in st.session_state:
            st.session_state.current_game = None
        
        if 'game_running' not in st.session_state:
            st.session_state.game_running = False
    
    def run_gravity_drop_game(self):
        """Run the Gravity Drop Game interface"""
        st.subheader("🎯 Gravity Drop Challenge")
        
        # Game info
        st.info("""
        **Objective:** Launch projectiles to hit all targets by adjusting gravity and launch parameters!
        
        **Physics Concepts:**
        - Projectile motion under different gravity conditions
        - Trajectory prediction and physics intuition
        - Effect of initial velocity and launch angle
        """)
        
        # Controls
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Launch Parameters")
            launch_angle = st.slider("Launch Angle (degrees)", 0, 90, 45)
            launch_speed = st.slider("Launch Speed (m/s)", 10, 200, 100)
            gravity = st.slider("Gravity (m/s²)", 1.0, 20.0, 9.81, 0.1)
            
            # Convert angle and speed to velocity components
            angle_rad = np.radians(launch_angle)
            vx = launch_speed * np.cos(angle_rad)
            vy = -launch_speed * np.sin(angle_rad)  # Negative for upward
            
            if st.button("🚀 Launch Ball"):
                action = {
                    'launch_ball': True,
                    'x': 50, 'y': 500,  # Start position
                    'vx': vx, 'vy': vy,
                    'change_gravity': True,
                    'gravity': gravity
                }
                
                # Run game step
                game_state = self.game_manager.run_game_step('gravity_drop', action)
                if game_state:
                    st.session_state.games_state['gravity_drop'] = game_state
        
        with col2:
            st.subheader("Game Display")
            
            # Display game state if available
            if 'gravity_drop' in st.session_state.games_state:
                game_img = st.session_state.games_state['gravity_drop']
                st.image(f"data:image/png;base64,{game_img}", 
                        caption="Gravity Drop Game", use_column_width=True)
            else:
                # Show trajectory prediction
                self.show_trajectory_prediction(launch_angle, launch_speed, gravity)
        
        # Educational content
        st.subheader("📚 Learn More About Projectile Motion")
        
        with st.expander("Physics Explanation"):
            st.write("""
            **Projectile Motion Equations:**
            
            - Horizontal position: x = v₀ₓ × t
            - Vertical position: y = v₀ᵧ × t - ½ × g × t²
            - Horizontal velocity: vₓ = v₀ₓ (constant)
            - Vertical velocity: vᵧ = v₀ᵧ - g × t
            
            **Key Insights:**
            - Horizontal and vertical motions are independent
            - Gravity only affects vertical motion
            - Maximum range occurs at 45° launch angle (in vacuum)
            - Time of flight depends on initial vertical velocity and gravity
            """)
        
        # Interactive trajectory plotter
        if st.checkbox("Show Trajectory Calculator"):
            self.interactive_trajectory_plotter()
    
    def show_trajectory_prediction(self, angle: float, speed: float, gravity: float):
        """Show predicted trajectory using matplotlib"""
        # Calculate trajectory
        angle_rad = np.radians(angle)
        vx = speed * np.cos(angle_rad)
        vy = speed * np.sin(angle_rad)
        
        # Time of flight
        t_flight = 2 * vy / gravity if vy > 0 else 1
        t = np.linspace(0, t_flight, 100)
        
        # Position
        x = vx * t
        y = vy * t - 0.5 * gravity * t**2
        
        # Create plot
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(x, y, 'b-', linewidth=2, label='Predicted Trajectory')
        ax.set_xlabel('Horizontal Distance (m)')
        ax.set_ylabel('Height (m)')
        ax.set_title(f'Projectile Motion: {angle}° angle, {speed} m/s, g={gravity} m/s²')
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_ylim(bottom=0)
        
        # Add targets (example positions)
        target_x = [50, 100, 150]
        target_y = [30, 20, 40]
        ax.scatter(target_x, target_y, c='red', s=100, marker='o', label='Targets')
        
        st.pyplot(fig)
        plt.close()
    
    def interactive_trajectory_plotter(self):
        """Interactive trajectory comparison tool"""
        st.subheader("🎯 Trajectory Comparison Tool")
        
        col1, col2 = st.columns(2)
        
        with col1:
            angles = st.multiselect("Launch Angles (degrees)", 
                                  options=[15, 30, 45, 60, 75], 
                                  default=[30, 45, 60])
            speed = st.slider("Launch Speed", 20, 150, 80)
            gravity = st.slider("Gravity", 1.0, 15.0, 9.81)
        
        with col2:
            if angles:
                fig = go.Figure()
                
                for angle in angles:
                    angle_rad = np.radians(angle)
                    vx = speed * np.cos(angle_rad)
                    vy = speed * np.sin(angle_rad)
                    
                    t_flight = 2 * vy / gravity if vy > 0 else 1
                    t = np.linspace(0, t_flight, 100)
                    
                    x = vx * t
                    y = vy * t - 0.5 * gravity * t**2
                    y = np.maximum(y, 0)  # Ground level
                    
                    fig.add_trace(go.Scatter(x=x, y=y, mode='lines',
                                           name=f'{angle}°',
                                           line=dict(width=3)))
                
                fig.update_layout(
                    title="Trajectory Comparison",
                    xaxis_title="Distance (m)",
                    yaxis_title="Height (m)",
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig, use_container_width=True)
    
    def run_orbit_design_game(self):
        """Run the Orbital Design Game interface"""
        st.subheader("🛰️ Orbital Designer")
        
        st.info("""
        **Objective:** Design stable circular orbits at target radii around the central body!
        
        **Physics Concepts:**
        - Gravitational force and orbital mechanics
        - Relationship between orbital velocity and radius
        - Stable vs unstable orbital configurations
        """)
        
        # Controls
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Satellite Parameters")
            orbit_radius = st.slider("Desired Orbit Radius", 100, 400, 200)
            
            # Calculate orbital velocity for circular orbit
            # v = sqrt(GM/r) - simplified
            G_scaled = 6.67e-11 * 1e6  # From game
            M = 1e6  # Central body mass
            orbital_velocity = np.sqrt(G_scaled * M / orbit_radius)
            
            st.write(f"**Theoretical Orbital Velocity:** {orbital_velocity:.1f} units/s")
            
            velocity_factor = st.slider("Velocity Factor", 0.5, 2.0, 1.0, 0.1)
            actual_velocity = orbital_velocity * velocity_factor
            
            st.write(f"**Actual Velocity:** {actual_velocity:.1f} units/s")
            
            if st.button("🛰️ Launch Satellite"):
                # Calculate position and velocity for orbit
                x = 400 + orbit_radius  # Center + radius
                y = 300  # Center height
                vx = 0
                vy = actual_velocity
                
                action = {
                    'add_satellite': True,
                    'x': x, 'y': y, 'vx': vx, 'vy': vy
                }
                
                game_state = self.game_manager.run_game_step('orbit_design', action)
                if game_state:
                    st.session_state.games_state['orbit_design'] = game_state
        
        with col2:
            st.subheader("Orbital System")
            
            if 'orbit_design' in st.session_state.games_state:
                game_img = st.session_state.games_state['orbit_design']
                st.image(f"data:image/png;base64,{game_img}", 
                        caption="Orbital Design Game", use_column_width=True)
            else:
                # Show orbital velocity relationship
                self.show_orbital_relationships()
        
        # Educational content
        with st.expander("Orbital Mechanics Fundamentals"):
            st.write("""
            **Kepler's Laws:**
            1. Orbits are ellipses with the central body at one focus
            2. A line joining the satellite and central body sweeps equal areas in equal times
            3. The square of orbital period is proportional to the cube of the semi-major axis
            
            **Circular Orbit Velocity:**
            v = √(GM/r)
            
            Where:
            - G is the gravitational constant
            - M is the mass of the central body
            - r is the orbital radius
            
            **Escape Velocity:**
            v_escape = √(2GM/r) = √2 × v_orbital
            """)
    
    def show_orbital_relationships(self):
        """Show orbital velocity vs radius relationship"""
        radii = np.linspace(100, 400, 100)
        G_scaled = 6.67e-11 * 1e6
        M = 1e6
        
        velocities = np.sqrt(G_scaled * M / radii)
        periods = 2 * np.pi * radii / velocities
        
        fig = make_subplots(rows=2, cols=1,
                           subplot_titles=('Orbital Velocity vs Radius', 
                                         'Orbital Period vs Radius'))
        
        fig.add_trace(go.Scatter(x=radii, y=velocities, mode='lines',
                               name='Orbital Velocity'), row=1, col=1)
        
        fig.add_trace(go.Scatter(x=radii, y=periods, mode='lines',
                               name='Orbital Period'), row=2, col=1)
        
        fig.update_xaxes(title_text="Radius (units)", row=2, col=1)
        fig.update_yaxes(title_text="Velocity (units/s)", row=1, col=1)
        fig.update_yaxes(title_text="Period (s)", row=2, col=1)
        
        st.plotly_chart(fig, use_container_width=True)
    
    def run_black_hole_game(self):
        """Run the Black Hole Escape Game interface"""
        st.subheader("🕳️ Black Hole Escape")
        
        st.info("""
        **Objective:** Navigate around the black hole, collect items, and avoid the event horizon!
        
        **Physics Concepts:**
        - Event horizons and Schwarzschild radius
        - Tidal forces and spaghettification
        - Relativistic effects near massive objects
        """)
        
        # Controls
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Spacecraft Controls")
            thrust_power = st.slider("Thrust Power", 0.1, 2.0, 1.0)
            
            # Direction controls
            thrust_direction = st.selectbox("Thrust Direction", 
                                          ["Up", "Down", "Left", "Right", 
                                           "Up-Left", "Up-Right", "Down-Left", "Down-Right"])
            
            direction_map = {
                "Up": (0, -1), "Down": (0, 1), "Left": (-1, 0), "Right": (1, 0),
                "Up-Left": (-0.7, -0.7), "Up-Right": (0.7, -0.7),
                "Down-Left": (-0.7, 0.7), "Down-Right": (0.7, 0.7)
            }
            
            if st.button("🚀 Apply Thrust"):
                tx, ty = direction_map[thrust_direction]
                action = {
                    'thrust': True,
                    'thrust_x': tx * thrust_power,
                    'thrust_y': ty * thrust_power
                }
                
                game_state = self.game_manager.run_game_step('black_hole_escape', action)
                if game_state:
                    st.session_state.games_state['black_hole_escape'] = game_state
        
        with col2:
            st.subheader("Black Hole System")
            
            if 'black_hole_escape' in st.session_state.games_state:
                game_img = st.session_state.games_state['black_hole_escape']
                st.image(f"data:image/png;base64,{game_img}", 
                        caption="Black Hole Escape Game", use_column_width=True)
            else:
                # Show black hole structure
                self.show_black_hole_structure()
        
        # Educational content
        with st.expander("Black Hole Physics"):
            st.write("""
            **Event Horizon:**
            The boundary around a black hole beyond which nothing can escape.
            
            **Schwarzschild Radius:**
            r = 2GM/c²
            
            **Tidal Forces:**
            Objects approaching a black hole experience extreme tidal forces due to 
            the difference in gravitational pull between their near and far sides.
            
            **Spaghettification:**
            The stretching of objects into long, thin shapes due to tidal forces.
            
            **Time Dilation:**
            Time passes more slowly in stronger gravitational fields.
            """)
    
    def show_black_hole_structure(self):
        """Show black hole structure visualization"""
        # Create visualization of black hole regions
        theta = np.linspace(0, 2*np.pi, 100)
        
        fig = go.Figure()
        
        # Event horizon
        r_eh = 25
        x_eh = r_eh * np.cos(theta)
        y_eh = r_eh * np.sin(theta)
        fig.add_trace(go.Scatter(x=x_eh, y=y_eh, mode='lines',
                               name='Event Horizon', line=dict(color='red', width=3)))
        
        # Photon sphere
        r_ps = 37.5  # 1.5 * Schwarzschild radius
        x_ps = r_ps * np.cos(theta)
        y_ps = r_ps * np.sin(theta)
        fig.add_trace(go.Scatter(x=x_ps, y=y_ps, mode='lines',
                               name='Photon Sphere', line=dict(color='orange', dash='dash')))
        
        # Stable orbit regions
        for r in [75, 100, 125, 150]:
            x_orbit = r * np.cos(theta)
            y_orbit = r * np.sin(theta)
            fig.add_trace(go.Scatter(x=x_orbit, y=y_orbit, mode='lines',
                                   name=f'Stable Orbit {r}', 
                                   line=dict(color='green', dash='dot'),
                                   showlegend=False))
        
        fig.update_layout(
            title="Black Hole Structure",
            xaxis_title="Distance",
            yaxis_title="Distance",
            showlegend=True,
            width=500, height=500
        )
        
        fig.update_xaxes(scaleanchor="y", scaleratio=1)
        
        st.plotly_chart(fig, use_container_width=True)
    
    def run_gravitational_waves_game(self):
        """Run the Gravitational Waves Game interface"""
        st.subheader("🌊 Gravitational Wave Detective")
        
        st.info("""
        **Objective:** Detect gravitational waves by analyzing signals from multiple detectors!
        
        **Physics Concepts:**
        - Gravitational wave sources (binary mergers)
        - Signal detection and analysis
        - Coincidence detection methods
        """)
        
        # Controls
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Detector Control")
            
            detectors = ["LIGO-1", "LIGO-2", "LIGO-3"]
            for i, detector in enumerate(detectors):
                if st.button(f"Toggle {detector}"):
                    action = {'activate_detector': i}
                    game_state = self.game_manager.run_game_step('gravitational_waves', action)
                    if game_state:
                        st.session_state.games_state['gravitational_waves'] = game_state
            
            # Auto-update option
            if st.checkbox("Auto-update detectors"):
                if st.button("Update Detection"):
                    game_state = self.game_manager.run_game_step('gravitational_waves', {})
                    if game_state:
                        st.session_state.games_state['gravitational_waves'] = game_state
        
        with col2:
            st.subheader("Detection Interface")
            
            if 'gravitational_waves' in st.session_state.games_state:
                game_img = st.session_state.games_state['gravitational_waves']
                st.image(f"data:image/png;base64,{game_img}", 
                        caption="Gravitational Wave Detection", use_column_width=True)
            else:
                # Show example gravitational wave
                self.show_gravitational_wave_example()
        
        # Educational content
        with st.expander("Gravitational Wave Science"):
            st.write("""
            **What are Gravitational Waves?**
            Ripples in spacetime caused by accelerating massive objects.
            
            **Detection Principle:**
            LIGO uses laser interferometry to detect tiny changes in distance 
            (smaller than 1/10,000th the width of a proton!).
            
            **Binary Merger Chirp:**
            As two massive objects spiral inward, the frequency of gravitational 
            waves increases, creating a characteristic "chirp" signal.
            
            **Coincidence Detection:**
            Multiple detectors must see the same signal to confirm a detection 
            and rule out local noise.
            
            **Famous Detections:**
            - GW150914: First direct detection (2015)
            - GW170817: Binary neutron star merger with electromagnetic counterpart
            """)
    
    def show_gravitational_wave_example(self):
        """Show example gravitational wave signal"""
        t = np.linspace(0, 2, 1000)
        
        # Simulate chirp signal
        f0 = 50  # Starting frequency
        chirp_rate = 1.5
        frequency = f0 * (chirp_rate ** t)
        
        # Amplitude decreases as objects get closer
        amplitude = np.exp(-t/2)
        
        # Generate strain signal
        strain = amplitude * np.sin(2 * np.pi * frequency * t)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=t, y=strain, mode='lines',
                               name='Gravitational Wave Strain',
                               line=dict(color='blue', width=2)))
        
        fig.update_layout(
            title="Example: Binary Merger Gravitational Wave",
            xaxis_title="Time (seconds)",
            yaxis_title="Strain (dimensionless)",
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Frequency evolution
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=t, y=frequency, mode='lines',
                                name='Frequency Evolution',
                                line=dict(color='red', width=2)))
        
        fig2.update_layout(
            title="Chirp: Frequency vs Time",
            xaxis_title="Time (seconds)",
            yaxis_title="Frequency (Hz)"
        )
        
        st.plotly_chart(fig2, use_container_width=True)
    
    def display_game_menu(self):
        """Display the main game selection menu"""
        st.title("🎮 Educational Physics Games")
        
        st.write("""
        Welcome to the interactive physics games! Each game teaches important physics concepts
        through hands-on experimentation and exploration.
        """)
        
        # Game selection
        games = {
            "Gravity Drop Challenge": "gravity_drop",
            "Orbital Designer": "orbit_design", 
            "Black Hole Escape": "black_hole_escape",
            "Gravitational Wave Detective": "gravitational_waves"
        }
        
        # Display game options
        for game_name, game_key in games.items():
            with st.expander(f"🎯 {game_name}"):
                info = self.game_manager.get_game_info(game_key)
                st.write(f"**Description:** {info.get('description', 'Educational physics game')}")
                st.write(f"**Controls:** {info.get('controls', 'Interactive controls available')}")
                
                if st.button(f"Play {game_name}", key=f"play_{game_key}"):
                    st.session_state.current_game = game_key
                    st.experimental_rerun()
        
        # If a game is selected, run it
        if st.session_state.current_game:
            st.write("---")
            
            if st.button("🔙 Back to Game Menu"):
                st.session_state.current_game = None
                st.experimental_rerun()
            
            if st.session_state.current_game == "gravity_drop":
                self.run_gravity_drop_game()
            elif st.session_state.current_game == "orbit_design":
                self.run_orbit_design_game()
            elif st.session_state.current_game == "black_hole_escape":
                self.run_black_hole_game()
            elif st.session_state.current_game == "gravitational_waves":
                self.run_gravitational_waves_game()


def run_educational_games():
    """Main function to run educational games in Streamlit"""
    runner = StreamlitGameRunner()
    runner.display_game_menu()


if __name__ == "__main__":
    run_educational_games()
