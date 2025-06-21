"""
🌌 Gravity Yonder Over - Streamlit Main Application
From falling apples to orbital slingshots — learn gravity the cosmic way.

This is the main entry point for the Streamlit web application.
"""

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

# Configure Streamlit page
st.set_page_config(
    page_title="🌌 Gravity Yonder Over",
    page_icon="🌌",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/aditya13504/gravity-yonder-over',
        'Report a bug': 'https://github.com/aditya13504/gravity-yonder-over/issues',
        'About': """
        # Gravity Yonder Over
        An interactive educational platform for learning gravity and orbital mechanics 
        through gamification and AI-powered simulations.
        
        Built with ❤️ for space enthusiasts and physics learners.
        """
    }
)

# Import backend modules
try:
    from backend.simulations.gravity_solver import GravitySolver
    from backend.simulations.precompute import PrecomputedSimulations
    from backend.visualizations.plotly_graphs import GravityVisualizer
    from backend.models.celestial_body import CelestialBody
except ImportError:
    # Fallback for development
    st.error("Backend modules not found. Please ensure the project structure is correct.")
    st.stop()

# Try to import optional components
try:
    from backend.simulations.cuda_accelerator import CUDAGravityAccelerator
    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False

try:
    from backend.simulations.modulus_wrapper import ModulusGravitySimulator
    MODULUS_AVAILABLE = True
except ImportError:
    MODULUS_AVAILABLE = False

try:
    from backend.ml.models.pinn_gravity import GravityPINN, GravityPINNTrainer
    from backend.ml.models.trajectory_predictor import TrajectoryPredictor
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

# Custom CSS for better styling
st.markdown("""
<style>
    .main {
        padding: 0rem 1rem;
    }
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 20px;
        border: none;
        padding: 0.5rem 1rem;
        font-weight: bold;
        transition: all 0.3s;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    .stButton>button:hover {
        transform: scale(1.05);
        box-shadow: 0 6px 20px rgba(0,0,0,0.3);
    }
    .game-container {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        border-radius: 15px;
        padding: 2rem;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        text-align: center;
    }
    .physics-equation {
        background: #f0f0f0;
        padding: 1rem;
        border-radius: 5px;
        font-family: 'Courier New', monospace;
        border-left: 4px solid #667eea;
    }
    h1, h2, h3 {
        color: #2c3e50;
        font-family: 'Arial', sans-serif;
    }
    .success-message {
        background: linear-gradient(135deg, #4CAF50, #45a049);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .info-box {
        background: linear-gradient(135deg, #e3f2fd, #bbdefb);
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #2196F3;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
def initialize_session_state():
    """Initialize all session state variables"""
    defaults = {
        'score': 0,
        'level': 1,
        'current_game': None,
        'sandbox_bodies': [],
        'physics_engine': 'standard',
        'tutorial_completed': False,
        'achievements': [],
        'game_progress': {},
        'user_preferences': {
            'physics_level': 'intermediate',
            'visualization_quality': 'high',
            'sound_enabled': True
        }
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def render_header():
    """Render the main header with title and navigation"""
    col1, col2, col3 = st.columns([2, 3, 2])
    
    with col2:
        st.markdown("""
        <div style='text-align: center; padding: 2rem 0;'>
            <h1 style='font-size: 3rem; margin-bottom: 0.5rem;'>🌌 Gravity Yonder Over</h1>
            <p style='font-size: 1.2rem; color: #7f8c8d; font-style: italic;'>
                From falling apples to orbital slingshots — learn gravity the cosmic way
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    # Feature availability indicators
    with col3:
        st.markdown("### System Status")
        st.success("✅ Core Physics Engine")
        if CUDA_AVAILABLE:
            st.success("✅ CUDA Acceleration")
        else:
            st.info("ℹ️ CPU Mode (CUDA N/A)")
        if MODULUS_AVAILABLE:
            st.success("✅ NVIDIA Modulus")
        else:
            st.info("ℹ️ Standard Physics")
        if ML_AVAILABLE:
            st.success("✅ AI/ML Models")
        else:
            st.info("ℹ️ Classical Physics Only")

def render_sidebar():
    """Render the sidebar with navigation and user stats"""
    with st.sidebar:
        st.markdown("## 🎮 Navigation")
        
        # Mode selection
        mode = st.radio(
            "Choose Your Learning Adventure:",
            ["🏠 Home Dashboard", "🎯 Interactive Games", "🔬 Physics Sandbox", 
             "📚 Learn & Explore", "📊 Data Visualizations", "🏆 Achievements"],
            help="Select different sections of the learning platform"
        )
        
        st.markdown("---")
        
        # User progress
        st.markdown("### 📈 Your Progress")
        progress_col1, progress_col2 = st.columns(2)
        
        with progress_col1:
            st.metric("Level", st.session_state.level, delta=None)
            st.metric("Score", st.session_state.score, delta=None)
        
        with progress_col2:
            st.metric("Games", len(st.session_state.game_progress), delta=None)
            st.metric("Achievements", len(st.session_state.achievements), delta=None)
        
        # Progress bar
        progress_value = min(st.session_state.score / 1000, 1.0)
        st.progress(progress_value)
        st.caption(f"Progress to next level: {progress_value*100:.1f}%")
        
        st.markdown("---")
        
        # Physics engine selection
        st.markdown("### ⚙️ Physics Engine")
        physics_options = ["standard"]
        if CUDA_AVAILABLE:
            physics_options.append("cuda")
        if MODULUS_AVAILABLE:
            physics_options.append("modulus")
        
        st.session_state.physics_engine = st.selectbox(
            "Engine Type:",
            physics_options,
            help="Choose the physics computation backend"
        )
        
        # Quick stats
        st.markdown("### 📊 Quick Stats")
        st.info(f"🌍 Bodies Simulated: {len(st.session_state.sandbox_bodies)}")
        st.info(f"🚀 Current Game: {st.session_state.current_game or 'None'}")
        
        # Help and links
        st.markdown("---")
        st.markdown("### 🔗 Quick Links")
        st.markdown("- [GitHub Repository](https://github.com/aditya13504/gravity-yonder-over)")
        st.markdown("- [Live Demo](https://gravityyonder.streamlit.app)")
        st.markdown("- [Physics Documentation](https://gravityyonder.streamlit.app)")
        
        return mode

def render_home_dashboard():
    """Render the main home dashboard"""
    st.markdown("## 🏠 Welcome to Your Cosmic Journey!")
    
    # Quick overview cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>🎮 5 Mini Games</h3>
            <p>From apple drops to black holes</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>🔬 Physics Sandbox</h3>
            <p>Create your own solar systems</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3>📚 Interactive Lessons</h3>
            <p>Newton to Einstein</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="metric-card">
            <h3>🏆 Achievement System</h3>
            <p>Track your progress</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Featured content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 🚀 Start Your Journey")
        
        if st.button("🍎 Begin with Apple Drop", help="Learn basic gravity with Newton's famous experiment"):
            st.session_state.current_game = "Apple Drop"
            st.experimental_rerun()
        
        if st.button("🔬 Explore Physics Sandbox", help="Create and simulate your own gravitational systems"):
            st.experimental_rerun()
        
        if st.button("📚 Browse Learning Modules", help="Dive into structured physics lessons"):
            st.experimental_rerun()
    
    with col2:
        st.markdown("### 🎯 Today's Challenge")
        st.info("🌙 Calculate the Moon's orbital velocity around Earth!")
        if st.button("Accept Challenge"):
            st.balloons()
            st.success("Challenge accepted! Head to the Physics Sandbox to get started.")

def render_interactive_games():
    """Render the interactive games section"""
    st.markdown("## 🎯 Interactive Physics Games")
    
    games = [
        {
            "name": "Apple Drop",
            "icon": "🍎",
            "description": "Experience Newton's discovery! Drop apples and observe gravity.",
            "difficulty": "Beginner",
            "estimated_time": "5-10 minutes",
            "physics_concepts": ["Gravity", "Free Fall", "Acceleration"]
        },
        {
            "name": "Orbital Slingshot", 
            "icon": "🛸",
            "description": "Master gravitational assists to accelerate spacecraft.",
            "difficulty": "Intermediate",
            "estimated_time": "10-15 minutes",
            "physics_concepts": ["Orbital Mechanics", "Energy Conservation", "Vector Addition"]
        },
        {
            "name": "Escape Velocity",
            "icon": "🚀", 
            "description": "Launch rockets and calculate escape speeds.",
            "difficulty": "Intermediate",
            "estimated_time": "10-15 minutes",
            "physics_concepts": ["Escape Velocity", "Energy Conservation", "Planetary Physics"]
        },
        {
            "name": "Black Hole Navigator",
            "icon": "⚫",
            "description": "Navigate near black holes without crossing event horizon.",
            "difficulty": "Advanced",
            "estimated_time": "15-20 minutes", 
            "physics_concepts": ["General Relativity", "Event Horizons", "Time Dilation"]
        },
        {
            "name": "Lagrange Explorer",
            "icon": "🌐",
            "description": "Find gravitational equilibrium points.",
            "difficulty": "Expert",
            "estimated_time": "20-25 minutes",
            "physics_concepts": ["Lagrange Points", "Three-Body Problem", "Stability"]
        },
        {
            "name": "Wormhole Navigator",
            "icon": "🌌",
            "description": "Traverse Einstein-Rosen bridges through spacetime.",
            "difficulty": "Master",
            "estimated_time": "25-30 minutes",
            "physics_concepts": ["Wormholes", "Exotic Matter", "Spacetime Geometry"]
        }
    ]
    
    # Game selection
    col1, col2 = st.columns([2, 1])
    
    with col1:
        selected_game = st.selectbox("Choose a game:", [game["name"] for game in games])
        game_info = next(game for game in games if game["name"] == selected_game)
        
        # Game details
        st.markdown(f"### {game_info['icon']} {game_info['name']}")
        st.write(game_info["description"])
        
        col_a, col_b = st.columns(2)
        with col_a:
            st.info(f"**Difficulty:** {game_info['difficulty']}")
        with col_b:
            st.info(f"**Time:** {game_info['estimated_time']}")
        
        st.markdown("**Physics Concepts:**")
        for concept in game_info["physics_concepts"]:
            st.markdown(f"- {concept}")
        
        if st.button(f"Play {selected_game}", key=f"play_{selected_game}"):
            st.session_state.current_game = selected_game
            render_game_interface(selected_game)
    
    with col2:
        st.markdown("### 🏆 Your Progress")
        for game in games:
            progress = st.session_state.game_progress.get(game["name"], {})
            completed = progress.get("completed", False)
            score = progress.get("best_score", 0)
            
            if completed:
                st.success(f"{game['icon']} {game['name']}: ✅ ({score} pts)")
            else:
                st.info(f"{game['icon']} {game['name']}: ⏳")

def render_game_interface(game_name):
    """Render the specific game interface"""
    st.markdown(f"## 🎮 Playing: {game_name}")
    
    # Game-specific rendering
    if game_name == "Apple Drop":
        render_apple_drop_game()
    elif game_name == "Orbital Slingshot":
        render_orbital_slingshot_game()
    elif game_name == "Escape Velocity":
        render_escape_velocity_game()
    elif game_name == "Black Hole Navigator":
        render_black_hole_navigator_game()
    elif game_name == "Lagrange Explorer":
        render_lagrange_explorer_game()
    elif game_name == "Wormhole Navigator":
        render_wormhole_navigator_game()

def render_apple_drop_game():
    """Apple Drop Game Implementation"""
    st.markdown("### 🍎 Apple Drop Experiment")
    st.write("Experience Newton's discovery of gravity through interactive apple dropping!")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Parameters
        height = st.slider("Drop Height (meters)", 1.0, 100.0, 10.0, 0.5)
        gravity = st.slider("Gravitational Acceleration (m/s²)", 1.0, 20.0, 9.81, 0.1)
        
        # Physics calculations
        time_to_fall = math.sqrt(2 * height / gravity)
        final_velocity = gravity * time_to_fall
        
        if st.button("🍎 Drop the Apple!"):
            # Simulation
            with st.spinner("Apple falling..."):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for i in range(101):
                    progress = i / 100
                    current_time = time_to_fall * progress
                    current_height = height - 0.5 * gravity * current_time**2
                    current_velocity = gravity * current_time
                    
                    status_text.text(f"Time: {current_time:.2f}s | Height: {max(0, current_height):.2f}m | Velocity: {current_velocity:.2f}m/s")
                    progress_bar.progress(progress)
                    time.sleep(0.05)
                
                # Results
                st.success("🍎 Apple has landed!")
                st.balloons()
                
                # Score calculation
                accuracy = 100 - abs(gravity - 9.81) * 10
                score = int(max(10, accuracy))
                st.session_state.score += score
                
                st.info(f"You earned {score} points!")
    
    with col2:
        st.markdown("### 📊 Physics Analysis")
        st.markdown(f"""
        <div class="physics-equation">
        <strong>Calculations:</strong><br>
        Height: {height} m<br>
        Gravity: {gravity} m/s²<br>
        Time to fall: {time_to_fall:.2f} s<br>
        Final velocity: {final_velocity:.2f} m/s
        </div>
        """, unsafe_allow_html=True)
        
        # Create trajectory plot
        times = np.linspace(0, time_to_fall, 100)
        heights = height - 0.5 * gravity * times**2
        velocities = gravity * times
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=times, y=heights, mode='lines', name='Height vs Time'))
        fig.update_layout(title="Apple Drop Trajectory", xaxis_title="Time (s)", yaxis_title="Height (m)")
        st.plotly_chart(fig, use_container_width=True)

def render_physics_sandbox():
    """Render the physics sandbox"""
    st.markdown("## 🔬 Physics Sandbox")
    st.write("Create your own gravitational systems and watch them evolve!")
    
    # Simulation setup
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### Add Celestial Bodies")
        
        body_name = st.text_input("Body Name", f"Body_{len(st.session_state.sandbox_bodies)+1}")
        mass = st.number_input("Mass (kg)", min_value=1e10, max_value=1e40, value=5.972e24, format="%.2e")
        
        col_x, col_y = st.columns(2)
        with col_x:
            x_pos = st.number_input("X Position (m)", value=0.0, format="%.2e")
            x_vel = st.number_input("X Velocity (m/s)", value=0.0)
        with col_y:
            y_pos = st.number_input("Y Position (m)", value=0.0, format="%.2e")
            y_vel = st.number_input("Y Velocity (m/s)", value=0.0)
        
        if st.button("Add Body"):
            new_body = {
                "name": body_name,
                "mass": mass,
                "position": [x_pos, y_pos, 0],
                "velocity": [x_vel, y_vel, 0]
            }
            st.session_state.sandbox_bodies.append(new_body)
            st.success(f"Added {body_name}!")
        
        if st.button("Clear All"):
            st.session_state.sandbox_bodies = []
            st.success("Cleared all bodies!")
        
        # Preset systems
        st.markdown("### 🌟 Preset Systems")
        if st.button("Earth-Moon System"):
            st.session_state.sandbox_bodies = [
                {"name": "Earth", "mass": 5.972e24, "position": [0, 0, 0], "velocity": [0, 0, 0]},
                {"name": "Moon", "mass": 7.342e22, "position": [3.844e8, 0, 0], "velocity": [0, 1022, 0]}
            ]
            st.success("Loaded Earth-Moon system!")
        
        if st.button("Solar System (Inner)"):
            st.session_state.sandbox_bodies = [
                {"name": "Sun", "mass": 1.989e30, "position": [0, 0, 0], "velocity": [0, 0, 0]},
                {"name": "Mercury", "mass": 3.301e23, "position": [5.79e10, 0, 0], "velocity": [0, 47870, 0]},
                {"name": "Venus", "mass": 4.867e24, "position": [1.082e11, 0, 0], "velocity": [0, 35020, 0]},
                {"name": "Earth", "mass": 5.972e24, "position": [1.496e11, 0, 0], "velocity": [0, 29780, 0]}
            ]
            st.success("Loaded inner solar system!")
    
    with col2:
        st.markdown("### 🌌 Gravitational Simulation")
        
        if len(st.session_state.sandbox_bodies) > 0:
            # Display current bodies
            st.markdown("**Current Bodies:**")
            for i, body in enumerate(st.session_state.sandbox_bodies):
                col_a, col_b = st.columns([3, 1])
                with col_a:
                    st.write(f"{body['name']}: {body['mass']:.2e} kg")
                with col_b:
                    if st.button("Remove", key=f"remove_{i}"):
                        st.session_state.sandbox_bodies.pop(i)
                        st.experimental_rerun()
            
            # Simulation controls
            if st.button("▶️ Run Simulation"):
                simulate_sandbox_system()
        else:
            st.info("Add celestial bodies to begin simulation!")

def simulate_sandbox_system():
    """Run the sandbox gravitational simulation"""
    if len(st.session_state.sandbox_bodies) < 2:
        st.warning("Need at least 2 bodies for simulation!")
        return
    
    # Initialize physics engine based on selection
    if st.session_state.physics_engine == "cuda" and CUDA_AVAILABLE:
        try:
            solver = CUDAGravityAccelerator()
            st.info("Using CUDA acceleration")
        except:
            solver = GravitySolver()
            st.info("CUDA failed, using standard solver")
    else:
        solver = GravitySolver()
        st.info("Using standard physics engine")
    
    # Convert to CelestialBody objects
    bodies = []
    for body_data in st.session_state.sandbox_bodies:
        body = CelestialBody(
            name=body_data["name"],
            mass=body_data["mass"],
            position=np.array(body_data["position"][:2]),  # 2D for simplicity
            velocity=np.array(body_data["velocity"][:2])
        )
        bodies.append(body)
    
    # Run simulation
    with st.spinner("Running gravitational simulation..."):
        try:
            # Simulate for 1 year with daily time steps
            dt = 24 * 3600  # 1 day in seconds
            steps = 365  # 1 year
            
            trajectories = solver.simulate_system(bodies, dt, steps)
            
            # Create visualization
            visualizer = GravityVisualizer()
            fig = visualizer.create_sandbox_visualization(trajectories, bodies)
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Analysis
            st.markdown("### 📊 Simulation Analysis")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Simulation Time", "1 Year")
            with col2:
                st.metric("Time Step", "1 Day")
            with col3:
                st.metric("Total Steps", steps)
            
            # Calculate system properties
            total_mass = sum(body.mass for body in bodies)
            st.success(f"Simulation completed! Total system mass: {total_mass:.2e} kg")
            
            # Award points
            points = len(bodies) * 50
            st.session_state.score += points
            st.info(f"Earned {points} points for simulation!")
            
        except Exception as e:
            st.error(f"Simulation failed: {str(e)}")
            st.info("Try using simpler initial conditions or fewer bodies.")

def render_learning_modules():
    """Render educational content"""
    st.markdown("## 📚 Learn & Explore")
    
    # Learning path
    learning_modules = [
        {"title": "Newton's Laws of Motion", "level": "Beginner", "duration": "30 min", "status": "available"},
        {"title": "Gravitational Forces", "level": "Beginner", "duration": "45 min", "status": "available"},
        {"title": "Orbital Mechanics", "level": "Intermediate", "duration": "60 min", "status": "available"},
        {"title": "Black Holes & Event Horizons", "level": "Advanced", "duration": "75 min", "status": "available"},
        {"title": "Wormholes & Exotic Matter", "level": "Expert", "duration": "90 min", "status": "available"},
        {"title": "General Relativity", "level": "Expert", "duration": "120 min", "status": "coming_soon"}
    ]
    
    st.markdown("### 📖 Learning Path")
    
    for i, module in enumerate(learning_modules):
        col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
        
        with col1:
            if module["status"] == "available":
                st.markdown(f"**{i+1}. {module['title']}**")
            else:
                st.markdown(f"**{i+1}. {module['title']}** ⏳")
        
        with col2:
            st.caption(module["level"])
        
        with col3:
            st.caption(module["duration"])
        
        with col4:
            if module["status"] == "available":
                if st.button("Start", key=f"learn_{i}"):
                    render_learning_content(module["title"])
            else:
                st.caption("Coming Soon")

def render_learning_content(module_title):
    """Render specific learning module content"""
    st.markdown(f"## 📖 {module_title}")
    
    if module_title == "Newton's Laws of Motion":
        st.markdown("""
        ### Isaac Newton's Three Laws of Motion
        
        #### First Law: Law of Inertia
        *An object at rest stays at rest, and an object in motion stays in motion, 
        unless acted upon by an external force.*
        
        **Example:** A spacecraft in deep space will continue moving in a straight line 
        at constant velocity unless its engines fire or it encounters a gravitational field.
        """)
        
        # Interactive demo
        st.markdown("#### 🎮 Interactive Demo")
        if st.button("Launch Virtual Spacecraft"):
            st.info("🚀 Spacecraft launched! Notice how it maintains constant velocity in empty space.")
            
    elif module_title == "Black Holes & Event Horizons":
        st.markdown("""
        ### Understanding Black Holes
        
        Black holes are regions of spacetime where gravity is so strong that nothing—not even light—can escape.
        
        #### Key Concepts:
        - **Event Horizon**: The boundary beyond which escape is impossible
        - **Schwarzschild Radius**: The radius of the event horizon
        - **Singularity**: The center where matter is compressed to infinite density
        """)
        
        # Schwarzschild radius calculator
        st.markdown("#### 🧮 Schwarzschild Radius Calculator")
        mass = st.number_input("Enter mass (kg):", min_value=1e10, max_value=1e40, value=5.972e24, format="%.2e")
        
        G = 6.67430e-11
        c = 299792458
        rs = 2 * G * mass / (c**2)
        
        st.markdown(f"""
        <div class="physics-equation">
        For mass = {mass:.2e} kg<br>
        Schwarzschild radius = {rs:.2e} meters<br>
        = {rs/1000:.2f} kilometers
        </div>
        """, unsafe_allow_html=True)

def render_visualizations():
    """Render data visualizations section"""
    st.markdown("## 📊 Data Visualizations")
    
    viz_type = st.selectbox("Choose visualization:", 
                           ["Gravitational Field", "Orbital Trajectories", "Force Vectors", "Energy Analysis"])
    
    if viz_type == "Gravitational Field":
        st.markdown("### 🌍 Gravitational Field Visualization")
        
        # Parameters
        col1, col2 = st.columns(2)
        with col1:
            mass = st.slider("Central Mass (Earth masses)", 0.1, 100.0, 1.0)
        with col2:
            grid_size = st.slider("Grid Resolution", 20, 100, 50)
        
        # Generate field visualization
        if st.button("Generate Field Visualization"):
            create_gravity_field_plot(mass, grid_size)

def create_gravity_field_plot(mass_earth_units, grid_size):
    """Create gravitational field visualization"""
    # Constants
    G = 6.67430e-11
    M_earth = 5.972e24
    mass = mass_earth_units * M_earth
    
    # Create grid
    x = np.linspace(-10, 10, grid_size) * 6.371e6  # Earth radii
    y = np.linspace(-10, 10, grid_size) * 6.371e6
    X, Y = np.meshgrid(x, y)
    
    # Calculate gravitational field strength
    R = np.sqrt(X**2 + Y**2)
    R[R == 0] = 1e-10  # Avoid division by zero
    
    g = G * mass / (R**2)
    
    # Create field direction
    gx = -g * X / R
    gy = -g * Y / R
    
    # Create plot
    fig = go.Figure()
    
    # Add field strength contours
    fig.add_trace(go.Contour(
        x=x/6.371e6, y=y/6.371e6, z=np.log10(g),
        colorscale='Viridis',
        name='log₁₀(Field Strength)',
        showscale=True
    ))
    
    # Add field vectors (subsample for clarity)
    skip = max(1, grid_size // 20)
    fig.add_trace(go.Scatter(
        x=X[::skip, ::skip].flatten()/6.371e6,
        y=Y[::skip, ::skip].flatten()/6.371e6,
        mode='markers',
        marker=dict(size=2, color='white'),
        name='Field Vectors',
        showlegend=False
    ))
    
    fig.update_layout(
        title=f"Gravitational Field for {mass_earth_units}× Earth Mass",
        xaxis_title="Distance (Earth Radii)",
        yaxis_title="Distance (Earth Radii)",
        width=600,
        height=600
    )
    
    st.plotly_chart(fig, use_container_width=True)

def render_achievements():
    """Render achievements and progress tracking"""
    st.markdown("## 🏆 Achievements & Progress")
    
    # Define achievements
    achievements = [
        {"name": "First Steps", "description": "Complete your first game", "icon": "👶", "unlocked": len(st.session_state.game_progress) > 0},
        {"name": "Apple Collector", "description": "Master the Apple Drop game", "icon": "🍎", "unlocked": False},
        {"name": "Rocket Scientist", "description": "Calculate escape velocity correctly", "icon": "🚀", "unlocked": False},
        {"name": "Black Hole Explorer", "description": "Navigate near a black hole safely", "icon": "⚫", "unlocked": False},
        {"name": "Lagrange Master", "description": "Find all 5 Lagrange points", "icon": "🌐", "unlocked": False},
        {"name": "Spacetime Navigator", "description": "Successfully traverse a wormhole", "icon": "🌌", "unlocked": False},
        {"name": "Physics Scholar", "description": "Complete all learning modules", "icon": "🎓", "unlocked": False},
        {"name": "Sandbox Creator", "description": "Create a stable 3-body system", "icon": "🔬", "unlocked": False}
    ]
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 🏅 Achievement Gallery")
        
        for achievement in achievements:
            if achievement["unlocked"]:
                st.success(f"{achievement['icon']} **{achievement['name']}** - {achievement['description']}")
            else:
                st.info(f"{achievement['icon']} **{achievement['name']}** - {achievement['description']} (Locked)")
    
    with col2:
        st.markdown("### 📈 Statistics")
        st.metric("Total Score", st.session_state.score)
        st.metric("Current Level", st.session_state.level)
        st.metric("Games Played", len(st.session_state.game_progress))
        st.metric("Achievements", sum(1 for a in achievements if a["unlocked"]))
        
        # Progress to next level
        next_level_score = st.session_state.level * 1000
        progress = min(st.session_state.score / next_level_score, 1.0)
        st.progress(progress)
        st.caption(f"Progress to Level {st.session_state.level + 1}")

def render_footer():
    """Render footer with links and credits"""
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #7f8c8d; padding: 2rem 0;'>
        <p><strong>🌌 Gravity Yonder Over</strong></p>
        <p>Built with ❤️ for space enthusiasts and physics learners</p>
        <p>
            <a href='https://github.com/aditya13504/gravity-yonder-over' target='_blank'>GitHub</a> | 
            <a href='https://gravityyonder.streamlit.app' target='_blank'>Live Demo</a> | 
            <a href='mailto:contact@gravityyonder.com'>Contact</a>
        </p>
        <p><em>From falling apples to orbital slingshots — learn gravity the cosmic way</em></p>
    </div>
    """, unsafe_allow_html=True)

# Main application logic
def main():
    """Main application entry point"""
    initialize_session_state()
    render_header()
    
    # Sidebar navigation
    mode = render_sidebar()
    
    # Main content area
    if mode == "🏠 Home Dashboard":
        render_home_dashboard()
    elif mode == "🎯 Interactive Games":
        render_interactive_games()
    elif mode == "🔬 Physics Sandbox":
        render_physics_sandbox()
    elif mode == "📚 Learn & Explore":
        render_learning_modules()
    elif mode == "📊 Data Visualizations":
        render_visualizations()
    elif mode == "🏆 Achievements":
        render_achievements()
    
    render_footer()

# Run the application
if __name__ == "__main__":
    main()
