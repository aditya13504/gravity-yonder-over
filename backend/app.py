import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from datetime import datetime
import json
import os
import time
import math
from typing import Dict, List, Tuple, Optional
from simulations.gravity_solver import GravitySolver
from simulations.precompute import PrecomputedSimulations
from visualizations.plotly_graphs import GravityVisualizer
from models.celestial_body import CelestialBody
import streamlit.components.v1 as components

# Try to import CUDA and Modulus components (optional)
try:
    from simulations.cuda_accelerator import CUDAGravityAccelerator
    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False
    st.warning("CUDA acceleration not available. Using CPU fallback.")

try:
    from simulations.modulus_wrapper import ModulusGravitySimulator
    MODULUS_AVAILABLE = True
except ImportError:
    MODULUS_AVAILABLE = False
    st.info("NVIDIA Modulus not available. Using standard physics engine.")

# Try to import ML components
try:
    from ml.models.pinn_gravity import GravityPINN, GravityPINNTrainer
    from ml.models.trajectory_predictor import TrajectoryPredictor
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    st.info("ML models not available. Using classical physics only.")

# Page configuration
st.set_page_config(
    page_title="Gravity Yonder Over",
    page_icon="🌌",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main {
        padding: 0rem 1rem;
    }
    .stButton>button {
        background-color: #1e3d59;
        color: white;
        border-radius: 20px;
        border: none;
        padding: 0.5rem 1rem;
        font-weight: bold;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: #2e5266;
        transform: scale(1.05);
    }
    .game-container {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        border-radius: 15px;
        padding: 2rem;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
    }
    h1, h2, h3 {
        color: #f5f5f5;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'current_game' not in st.session_state:
    st.session_state.current_game = None
if 'score' not in st.session_state:
    st.session_state.score = 0
if 'level' not in st.session_state:
    st.session_state.level = 1
if 'sandbox_bodies' not in st.session_state:
    st.session_state.sandbox_bodies = []

# Header
st.markdown("""
<div style='text-align: center;'>
    <h1>🌌 Gravity Yonder Over</h1>
    <p style='font-size: 1.2rem; color: #a8dadc;'>From falling apples to orbital slingshots — learn gravity the cosmic way.</p>
</div>
""", unsafe_allow_html=True)

# Sidebar navigation
with st.sidebar:
    st.markdown("## 🎮 Navigation")
    mode = st.radio(
        "Choose Mode:",
        ["🏠 Home", "🎯 Mini Games", "🔬 Sandbox", "📚 Learn", "📊 Visualizations"]
    )
    
    st.markdown("---")
    st.markdown("### 📈 Progress")
    st.progress(st.session_state.score / 1000)
    st.markdown(f"**Score:** {st.session_state.score}")
    st.markdown(f"**Level:** {st.session_state.level}")

# Main content area
if mode == "🏠 Home":
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        <div class='game-container'>
            <h2>Welcome to Gravity Yonder Over! 🚀</h2>
            <p>Embark on an interactive journey through the cosmos and master the fundamental force that shapes our universe.</p>
            
            <h3>What You'll Learn:</h3>
            <ul>
                <li>🍎 Newton's Law of Universal Gravitation</li>
                <li>🌍 Orbital Mechanics and Kepler's Laws</li>
                <li>🚀 Escape Velocity and Space Travel</li>
                <li>⚫ Black Holes and Extreme Gravity</li>
                <li>🌌 Einstein's General Relativity Basics</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class='game-container' style='text-align: center;'>
            <h3>Quick Stats</h3>
            <p>🎮 5 Mini Games</p>
            <p>🔬 Interactive Sandbox</p>
            <p>📊 Real-time Simulations</p>
            <p>💻 GPU-Optional</p>
            <p>📱 Mobile Friendly</p>
        </div>
        """, unsafe_allow_html=True)

elif mode == "🎯 Mini Games":
    st.markdown("<h2>🎮 Choose Your Game</h2>", unsafe_allow_html=True)
    
    games = {
        "Apple Drop": {
            "icon": "🍎",
            "description": "Experience Newton's discovery! Drop apples and observe gravity.",
            "difficulty": "Beginner"
        },
        "Orbital Slingshot": {
            "icon": "🛸",
            "description": "Master gravitational assists to reach distant planets.",
            "difficulty": "Intermediate"
        },
        "Escape Velocity": {
            "icon": "🚀",
            "description": "Launch rockets and escape planetary gravity wells.",
            "difficulty": "Intermediate"
        },
        "Black Hole Navigator": {
            "icon": "⚫",
            "description": "Navigate near black holes without crossing the event horizon.",
            "difficulty": "Advanced"
        },
        "Lagrange Explorer": {
            "icon": "🌐",
            "description": "Find and utilize gravitational equilibrium points.",
            "difficulty": "Expert"
        }
    }
    
    cols = st.columns(3)
    for idx, (game_name, game_info) in enumerate(games.items()):
        with cols[idx % 3]:
            if st.button(
                f"{game_info['icon']} {game_name}\n{game_info['difficulty']}", 
                key=game_name,
                use_container_width=True
            ):
                st.session_state.current_game = game_name
                st.rerun()
    
    if st.session_state.current_game:
        st.markdown(f"<h3>Playing: {st.session_state.current_game}</h3>", unsafe_allow_html=True)
        
        if st.session_state.current_game == "Apple Drop":
            render_apple_drop_game()
        elif st.session_state.current_game == "Orbital Slingshot":
            render_orbital_slingshot_game()
        # Add other games...

elif mode == "🔬 Sandbox":
    st.markdown("<h2>🔬 Gravity Sandbox</h2>", unsafe_allow_html=True)
    render_sandbox_mode()

elif mode == "📚 Learn":
    render_learning_section()

elif mode == "📊 Visualizations":
    render_visualizations()

# Game Functions
def render_apple_drop_game():
    """Apple Drop Game - Learn basic gravity"""
    col1, col2 = st.columns([2, 1])
    
    with col2:
        st.markdown("### Game Controls")
        height = st.slider("Drop Height (m)", 1, 100, 10)
        mass = st.slider("Apple Mass (kg)", 0.1, 1.0, 0.2)
        gravity = st.slider("Gravity (m/s²)", 1.0, 20.0, 9.81)
        
        if st.button("Drop Apple! 🍎"):
            # Calculate physics
            time_to_fall = np.sqrt(2 * height / gravity)
            final_velocity = gravity * time_to_fall
            
            st.success(f"Time to fall: {time_to_fall:.2f} seconds")
            st.info(f"Final velocity: {final_velocity:.2f} m/s")
            
            # Update score
            st.session_state.score += 10
    
    with col1:
        # Create animation
        solver = GravitySolver()
        t = np.linspace(0, np.sqrt(2 * height / gravity), 100)
        y = height - 0.5 * gravity * t**2
        
        fig = go.Figure()
        
        # Add ground
        fig.add_shape(
            type="rect",
            x0=-5, x1=5, y0=-1, y1=0,
            fillcolor="brown",
            line=dict(width=0)
        )
        
        # Add apple trajectory
        fig.add_trace(go.Scatter(
            x=[0] * len(t),
            y=y,
            mode='markers',
            marker=dict(size=20, color='red', symbol='circle'),
            name='Apple'
        ))
        
        fig.update_layout(
            title="Apple Drop Simulation",
            xaxis=dict(range=[-5, 5], title="Position (m)"),
            yaxis=dict(range=[-1, height + 5], title="Height (m)"),
            showlegend=False,
            height=500,
            template="plotly_dark"
        )
        
        st.plotly_chart(fig, use_container_width=True)

def render_orbital_slingshot_game():
    """Orbital Slingshot Game - Learn gravitational assists"""
    st.markdown("### 🛸 Orbital Slingshot Challenge")
    
    col1, col2 = st.columns([3, 1])
    
    with col2:
        st.markdown("### Mission Parameters")
        initial_velocity = st.slider("Initial Velocity (km/s)", 10, 50, 30)
        approach_angle = st.slider("Approach Angle (degrees)", 0, 90, 45)
        planet_mass = st.select_slider(
            "Planet Type",
            options=["Earth", "Jupiter", "Saturn"],
            value="Jupiter"
        )
        
        masses = {"Earth": 5.972e24, "Jupiter": 1.898e27, "Saturn": 5.683e26}
        
        if st.button("Launch Spacecraft! 🚀"):
            # Simulate gravitational assist
            solver = GravitySolver()
            planet = CelestialBody(
                name=planet_mass,
                mass=masses[planet_mass],
                position=np.array([0.0, 0.0]),
                velocity=np.array([0.0, 0.0])
            )
            
            # Calculate trajectory
            trajectory = solver.calculate_slingshot(
                planet, 
                initial_velocity * 1000,  # Convert to m/s
                np.radians(approach_angle)
            )
            
            st.session_state.score += 25
    
    with col1:
        # Visualization placeholder
        visualizer = GravityVisualizer()
        fig = visualizer.create_slingshot_visualization(
            planet_mass, 
            initial_velocity, 
            approach_angle
        )
        st.plotly_chart(fig, use_container_width=True)

def render_sandbox_mode():
    """Interactive Gravity Sandbox"""
    col1, col2 = st.columns([3, 1])
    
    with col2:
        st.markdown("### Add Celestial Body")
        name = st.text_input("Name", "Planet")
        mass = st.number_input("Mass (Earth masses)", 0.1, 1000.0, 1.0)
        x = st.number_input("X Position (AU)", -10.0, 10.0, 0.0)
        y = st.number_input("Y Position (AU)", -10.0, 10.0, 0.0)
        vx = st.number_input("X Velocity (km/s)", -50.0, 50.0, 0.0)
        vy = st.number_input("Y Velocity (km/s)", -50.0, 50.0, 0.0)
        
        if st.button("Add Body"):
            body = CelestialBody(
                name=name,
                mass=mass * 5.972e24,  # Convert to kg
                position=np.array([x * 1.496e11, y * 1.496e11]),  # Convert to meters
                velocity=np.array([vx * 1000, vy * 1000])  # Convert to m/s
            )
            st.session_state.sandbox_bodies.append(body)
        
        if st.button("Clear All"):
            st.session_state.sandbox_bodies = []
        
        if st.button("Run Simulation"):
            run_sandbox_simulation()
    
    with col1:
        if len(st.session_state.sandbox_bodies) > 0:
            visualizer = GravityVisualizer()
            fig = visualizer.create_sandbox_visualization(st.session_state.sandbox_bodies)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Add celestial bodies to begin the simulation!")

def render_learning_section():
    """Educational content section"""
    st.markdown("<h2>📚 Learn About Gravity</h2>", unsafe_allow_html=True)
    
    topics = {
        "Newton's Law": {
            "content": """
            ### Newton's Law of Universal Gravitation
            
            Every particle attracts every other particle with a force proportional to the product 
            of their masses and inversely proportional to the square of the distance between them.
            
            **Formula:** F = G × (m₁ × m₂) / r²
            
            Where:
            - F = gravitational force
            - G = gravitational constant (6.674 × 10⁻¹¹ N⋅m²/kg²)
            - m₁, m₂ = masses of the objects
            - r = distance between centers
            """,
            "demo": "apple_drop"
        },
        "Orbital Mechanics": {
            "content": """
            ### Orbital Mechanics
            
            Objects in orbit are in constant free fall, but their horizontal velocity prevents 
            them from hitting the surface.
            
            **Orbital Velocity:** v = √(GM/r)
            
            Where:
            - v = orbital velocity
            - G = gravitational constant
            - M = mass of the central body
            - r = orbital radius
            """,
            "demo": "orbit_demo"
        }
    }
    
    selected_topic = st.selectbox("Choose a topic:", list(topics.keys()))
    
    st.markdown(topics[selected_topic]["content"])
    
    if st.button(f"Try Interactive Demo: {topics[selected_topic]['demo']}"):
        st.session_state.current_game = topics[selected_topic]["demo"]

def render_visualizations():
    """Advanced visualizations section"""
    st.markdown("<h2>📊 Gravity Visualizations</h2>", unsafe_allow_html=True)
    
    viz_type = st.selectbox(
        "Choose Visualization:",
        ["Gravity Field", "Orbital Paths", "Schwarzschild Radius", "Lagrange Points"]
    )
    
    if viz_type == "Gravity Field":
        render_gravity_field_visualization()
    elif viz_type == "Orbital Paths":
        render_orbital_paths_visualization()
    # Add other visualizations...

def render_gravity_field_visualization():
    """Render gravity field visualization"""
    mass = st.slider("Central Mass (Solar masses)", 0.1, 10.0, 1.0)
    
    # Create meshgrid
    x = np.linspace(-5, 5, 50)
    y = np.linspace(-5, 5, 50)
    X, Y = np.meshgrid(x, y)
    
    # Calculate gravitational potential
    R = np.sqrt(X**2 + Y**2)
    R[R == 0] = 0.1  # Avoid division by zero
    potential = -mass / R
    
    fig = go.Figure(data=[
        go.Surface(
            x=X,
            y=Y,
            z=potential,
            colorscale='Viridis',
            name='Gravitational Potential'
        )
    ])
    
    fig.update_layout(
        title=f"Gravitational Field - {mass} Solar Masses",
        scene=dict(
            xaxis_title="X (AU)",
            yaxis_title="Y (AU)",
            zaxis_title="Potential Energy"
        ),
        height=600,
        template="plotly_dark"
    )
    
    st.plotly_chart(fig, use_container_width=True)

def run_sandbox_simulation():
    """Run the sandbox simulation"""
    if len(st.session_state.sandbox_bodies) < 2:
        st.warning("Add at least 2 bodies to run simulation!")
        return
    
    solver = GravitySolver()
    dt = 3600  # 1 hour time step
    steps = 1000
    
    with st.spinner("Running simulation..."):
        trajectories = solver.simulate_system(
            st.session_state.sandbox_bodies,
            dt,
            steps
        )
        
        # Store results for visualization
        st.session_state.simulation_results = trajectories
        st.success("Simulation complete!")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #888;'>
    <p>Built with ❤️ for space enthusiasts | 
    <a href='https://github.com/aditya13504/gravity-yonder-over.git' target='_blank'>GitHub</a> | 
    <a href='https://gravityyonder.streamlit.app' target='_blank'>Live Demo</a>
    </p>
</div>
""", unsafe_allow_html=True)