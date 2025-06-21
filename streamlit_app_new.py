"""
🌌 Gravity Yonder Over - Educational Physics Platform
Built with Streamlit + CPU-based Physics Models + Pandas + Plotly

A comprehensive educational web application for interactive gravity and physics simulations
using pre-trained CPU-based models and classical physics computations.
"""

import streamlit as st
import streamlit.components.v1
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import time
import uuid
import random
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# Import our custom modules
try:
    from src.cpu_physics_engine import ModulusGravityEngine
    from src.cpu_data_processor import CuDFDataProcessor
    from src.simulation_datasets import PreGeneratedSimulations
    from src.plotly_visualizer import PhysicsVisualizer
    from src.educational_content import EducationalContentManager
    from src.interactive_simulation_engine import interactive_engine
    from src.streamlit_game_runner import StreamlitGameRunner
except ImportError as e:
    st.error(f"Module import error: {e}")
    st.info("Please ensure all required modules are available.")

# Configure Streamlit page
st.set_page_config(
    page_title="Gravity Yonder Over - Physics Education Platform",
    page_icon="🌌",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1f1c2c 0%, #928dab 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .game-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
    .physics-insight {
        background: linear-gradient(135deg, #ff9a9e 0%, #fad0c4 100%);
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .metric-card {
        background: rgba(255, 255, 255, 0.1);
        padding: 0.5rem;
        border-radius: 5px;
        text-align: center;
        margin: 0.2rem 0;
    }
</style>
""", unsafe_allow_html=True)


def initialize_session_state():
    """Initialize session state variables and system components"""
    if 'initialized' not in st.session_state:
        try:
            # Initialize core components
            st.session_state.modulus_engine = ModulusGravityEngine()
            st.session_state.cudf_processor = CuDFDataProcessor()
            st.session_state.simulations = PreGeneratedSimulations()
            st.session_state.visualizer = PhysicsVisualizer()
            st.session_state.content_manager = EducationalContentManager()
            
            # User session
            if 'user_id' not in st.session_state:
                st.session_state.user_id = str(uuid.uuid4())
            if 'session_id' not in st.session_state:
                st.session_state.session_id = str(uuid.uuid4())
            
            st.session_state.initialized = True
        except Exception as e:
            st.error(f"Failed to initialize components: {e}")
            st.session_state.initialized = False


def main():
    """Main application entry point"""
    
    # Initialize components
    initialize_session_state()
    
    if not st.session_state.get('initialized', False):
        st.error("System initialization failed. Please refresh the page.")
        return
    
    # Main header
    st.markdown("""
    <div class="main-header">
        <h1>🌌 Gravity Yonder Over</h1>
        <h3>Interactive Physics Education with CPU-based Models</h3>
    </div>
    """, unsafe_allow_html=True)
      # Sidebar navigation
    st.sidebar.markdown("## 🎮 Learning Adventures")
    
    # Initialize page selection in session state
    if 'page_selection' not in st.session_state:
        st.session_state.page_selection = "🏠 Home"
    
    # Use selectbox but sync with session state
    game_selection = st.sidebar.selectbox(
        "Choose your physics adventure:",
        [
            "🏠 Home",
            "📚 Learning Path", 
            "🍎 Gravity Basics",
            "🚀 Orbital Mechanics", 
            "⚫ Black Hole Physics",
            "🌌 Gravitational Waves",
            "📊 Progress Dashboard",
            "🔬 Physics Sandbox"
        ],
        index=0 if st.session_state.page_selection not in [
            "🏠 Home", "📚 Learning Path", "🍎 Gravity Basics", "🚀 Orbital Mechanics", 
            "⚫ Black Hole Physics", "🌌 Gravitational Waves", "📊 Progress Dashboard", "🔬 Physics Sandbox"
        ] else [
            "🏠 Home", "📚 Learning Path", "🍎 Gravity Basics", "🚀 Orbital Mechanics", 
            "⚫ Black Hole Physics", "🌌 Gravitational Waves", "📊 Progress Dashboard", "🔬 Physics Sandbox"
        ].index(st.session_state.page_selection)
    )
    
    # Update session state when selectbox changes
    if game_selection != st.session_state.page_selection:
        st.session_state.page_selection = game_selection
      # System status in sidebar
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🔧 System Status")
    
    # Check CPU Physics Engine availability
    modulus_status = st.session_state.modulus_engine.get_status()
    if modulus_status['status'] == 'initialized':
        st.sidebar.success("✅ CPU Physics Engine: Active")
        st.sidebar.info(f"Framework: {modulus_status['framework']}")
    else:
        st.sidebar.warning("⚠️ CPU Physics Engine: Error")
      # Check CPU Data Processor availability
    cudf_status = st.session_state.cudf_processor.get_performance_metrics()
    st.sidebar.success("✅ Data Processor: CPU Optimized")
    st.sidebar.info("📊 Backend: Pandas + NumPy")
    
    # Performance metrics
    st.sidebar.markdown("### 📊 Performance")
    available_scenarios = st.session_state.simulations.get_scenario_list()
    col1, col2 = st.sidebar.columns(2)
    with col1:
        st.metric("Scenarios", len(available_scenarios))
    with col2:
        precomputed_count = sum(1 for s in available_scenarios if s['status'] == 'pre-computed')
        st.metric("Pre-computed", precomputed_count)
      # Route to appropriate page based on session state
    current_page = st.session_state.page_selection
    if current_page == "🏠 Home":
        show_home_page()
    elif "Learning Path" in current_page:
        show_learning_path()
    elif "Gravity Basics" in current_page:
        show_gravity_basics()
    elif "Orbital Mechanics" in current_page:
        show_orbital_mechanics()
    elif "Black Hole" in current_page:
        show_black_hole_physics()
    elif "Gravitational Waves" in current_page:
        show_gravitational_waves()
    elif "Progress Dashboard" in current_page:
        show_progress_dashboard()
    elif "Physics Sandbox" in current_page:
        show_physics_sandbox()


def show_home_page():
    """Display the home page with overview"""
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("## 🎯 Welcome to Physics Education")
        st.markdown("""
        **Gravity Yonder Over** is an advanced educational platform that combines **theoretical physics** 
        with **interactive simulations** to provide a comprehensive learning experience in gravitational physics.
        
        ### 🌟 Learning Journey Overview
        
        #### 🍎 **Gravity Basics** - Newton's Foundation
        Start your journey with **Sir Isaac Newton's** revolutionary insights into universal gravitation.
        
        **What You'll Learn:**
        - The inverse square law and its implications
        - Gravitational fields and potential energy
        - Real-world applications from tides to satellites
        - Mathematical problem-solving techniques
        
        **Key Concepts:** Force, acceleration, energy conservation, escape velocity
        
        #### 🚀 **Orbital Mechanics** - Kepler's Laws
        Discover how **Johannes Kepler's** three laws govern all orbital motion in the universe.
        
        **What You'll Learn:**
        - Elliptical orbits and why circles are special cases
        - How planets sweep equal areas in equal times
        - The relationship between orbital period and distance
        - Spacecraft trajectory design and mission planning
        
        **Key Concepts:** Ellipses, angular momentum, energy, transfer orbits
        
        #### ⚫ **Black Hole Physics** - Einstein's Relativity
        Explore the most extreme objects in the universe through **Einstein's General Relativity**.
        
        **What You'll Learn:**
        - How mass curves spacetime itself
        - Event horizons and the point of no return
        - Time dilation and gravitational redshift
        - Hawking radiation and black hole thermodynamics
        
        **Key Concepts:** Spacetime curvature, geodesics, singularities, information paradox
        
        #### 🌊 **Gravitational Waves** - Ripples in Spacetime
        Understand the newest window into the universe: **gravitational wave astronomy**.
        
        **What You'll Learn:**
        - How accelerating masses create spacetime ripples
        - The incredible precision of LIGO detection
        - Binary mergers and cosmic catastrophes
        - Multi-messenger astronomy and future discoveries
        
        **Key Concepts:** Wave polarization, interferometry, chirp signals, standard sirens
        
        ### 🧪 **Interactive Learning Features**
        
        **🎮 Real-time Simulations:**
        - Adjust parameters and see immediate results
        - Multiple visualization modes and perspectives
        - Video animations for complex phenomena
        
        **📊 Data Analysis:**
        - Explore how scientists extract physics from observations
        - Interactive plots and parameter studies
        - Statistical analysis of simulation results
        
        **🎯 Adaptive Quizzes:**
        - Questions tailored to your progress level
        - Immediate feedback with detailed explanations
        - Track your understanding across all topics
        
        ### 🎓 **Educational Philosophy**
        
        Our approach combines:
        - **Conceptual Understanding**: Deep explanations of physical principles
        - **Mathematical Rigor**: Proper equations and derivations
        - **Computational Tools**: Modern simulation and analysis methods
        - **Historical Context**: How our understanding evolved over time
        - **Current Research**: Connection to ongoing scientific discoveries
        
        ### 🌌 **Why Gravity Matters**
        
        Gravity is:
        - The **weakest** fundamental force (10³⁹ times weaker than electromagnetic)
        - The **most important** for large-scale structure (dominates at astronomical scales)
        - The **most mysterious** (we still don't understand quantum gravity)
        - The **most beautiful** (connects geometry to physics through Einstein's equations)
        
        Understanding gravity means understanding:
        - How stars and planets form
        - Why the universe has the structure it does
        - How black holes work and what they tell us about physics
        - Whether we might detect ripples from the Big Bang itself
        """)
        
        # Add a "Did You Know?" section
        st.markdown("### 🤯 Did You Know?")
        
        facts = [
            "A black hole with the mass of Earth would have an event horizon smaller than a marble!",
            "The gravitational waves from colliding black holes carry more power than all the stars in the visible universe combined.",
            "If you could compress the Sun to a black hole, its event horizon would be about 6 kilometers across.",
            "The Moon's gravity creates two tidal bulges on Earth - one facing the Moon and one on the opposite side.",
            "A satellite in geostationary orbit travels at exactly the right speed to stay above the same point on Earth.",
            "Time runs about 38 microseconds per day faster for GPS satellites due to weaker gravity at their altitude.",
            "The escape velocity from Earth (11.2 km/s) is independent of the mass of the escaping object.",
            "LIGO can detect changes in arm length smaller than 1/10,000th the width of a proton."
        ]
        
        import random
        random_fact = random.choice(facts)
        st.info(f"💡 **Random Physics Fact:** {random_fact}")
        
        # Show available scenarios
        st.markdown("### 🎮 Available Simulations")
        scenarios = st.session_state.simulations.get_scenario_list()
        
        for scenario in scenarios:
            status_icon = "✅" if scenario['status'] == 'pre-computed' else "🔄"
            st.markdown(f"**{status_icon} {scenario['name']}** - *{scenario['description']}*")
    
    with col2:
        st.markdown("### 🏆 Your Progress")
        
        # Get user insights
        user_insights = st.session_state.content_manager.get_user_insights(st.session_state.user_id)
        
        if "overall_progress" in user_insights:
            progress = user_insights["overall_progress"]
            st.metric("Lessons Completed", progress["completed_lessons"])
            st.metric("Average Mastery", f"{progress['avg_mastery']:.1%}")
            st.metric("Time Spent", f"{progress['total_time_hours']:.1f} hrs")
        else:
            st.info("Start learning to track your progress!")
          # Learning recommendations
        st.markdown("### 💡 Recommendations")
        if "recommendations" in user_insights:
            for rec in user_insights["recommendations"][:3]:
                st.markdown(f"• {rec}")
        else:
            st.markdown("• Begin with Gravity Basics")
            st.markdown("• Explore orbital mechanics")
            st.markdown("• Try the physics sandbox")
        
        # Main Start Learning Button
        st.markdown("---")
        if st.button("🚀 Start Learning Journey", type="primary", use_container_width=True):
            st.session_state.page_selection = "📚 Learning Path"
            st.rerun()


def show_learning_path():
    """Display adaptive learning path"""
    st.markdown("## 📚 Your Learning Path")
    
    # Get user's current level and path
    user_insights = st.session_state.content_manager.get_user_insights(st.session_state.user_id)
    avg_mastery = user_insights.get("overall_progress", {}).get("avg_mastery", 0)
    user_level = min(5, max(1, int(avg_mastery * 5) + 1))
    
    learning_path = st.session_state.content_manager.get_learning_path(user_level)
    
    st.markdown(f"**Current Level:** {user_level}/5 | **Mastery:** {avg_mastery:.1%}")
      # Display learning objectives
    for i, objective in enumerate(learning_path):
        with st.expander(f"📖 {objective['title']} (Level {objective['difficulty']})"):
            st.markdown(objective['description'])
            st.markdown(f"**Estimated Time:** {objective['estimated_time']} minutes")
            st.markdown(f"**Concepts:** {', '.join(objective['concepts'])}")
            
            if st.button(f"Start Learning: {objective['title']}", key=f"start_{objective['id']}"):
                st.session_state.current_lesson = objective['id']
                # Navigate to the appropriate lesson page based on the objective
                if 'gravity' in objective['title'].lower():
                    st.session_state.page_selection = "🍎 Gravity Basics"
                elif 'orbital' in objective['title'].lower():
                    st.session_state.page_selection = "🚀 Orbital Mechanics"
                elif 'black hole' in objective['title'].lower():
                    st.session_state.page_selection = "⚫ Black Hole Physics"
                elif 'gravitational wave' in objective['title'].lower():
                    st.session_state.page_selection = "🌌 Gravitational Waves"
                else:
                    st.session_state.page_selection = "🍎 Gravity Basics"  # Default
                st.success(f"Starting lesson: {objective['title']}")
                st.rerun()


def show_gravity_basics():
    """Gravity basics simulation and learning"""
    st.markdown("## 🍎 Gravity Basics")
    
    # Comprehensive Educational content
    with st.expander("📖 Learn: Newton's Law of Universal Gravitation", expanded=True):
        st.markdown("""
        ### 🌟 The Foundation of Gravitational Physics
        
        **Newton's Law of Universal Gravitation** is one of the most fundamental laws in physics, 
        describing how every particle in the universe attracts every other particle.
        
        ### 📐 The Mathematical Formula
        
        **F = G·m₁·m₂/r²**
        
        Where:
        - **F** = gravitational force (measured in Newtons, N)
        - **G** = gravitational constant = 6.674 × 10⁻¹¹ m³/(kg·s²)
        - **m₁, m₂** = masses of the two objects (kg)
        - **r** = distance between the centers of mass (m)
        
        ### 🔍 Understanding Each Component
        
        **The Gravitational Constant (G):**
        - Discovered by Henry Cavendish in 1798
        - Represents the intrinsic strength of gravity
        - Extremely small value shows gravity is the weakest fundamental force
        - Universal: same value everywhere in the universe
        
        **The Inverse Square Law:**
        - Force decreases with the square of distance
        - Double the distance → Force becomes 1/4
        - Triple the distance → Force becomes 1/9
        - This relationship is crucial for planetary orbits
        
        ### 🌍 Real-World Applications
        
        1. **Planetary Motion**: Explains why planets orbit the Sun in elliptical paths
        2. **Ocean Tides**: Moon's gravity creates high and low tides
        3. **Satellite Orbits**: Determines the speed needed for stable orbits
        4. **Galaxy Formation**: Gravity shapes the large-scale structure of the universe
        
        ### 🧮 Mathematical Insights
        
        **Gravitational Field Strength:**
        - g = GM/r² (acceleration due to gravity)
        - On Earth's surface: g ≈ 9.81 m/s²
        - Varies with altitude and planetary mass
        
        **Escape Velocity:**
        - v_escape = √(2GM/r)
        - Minimum speed to escape gravitational pull
        - Earth: 11.2 km/s, Moon: 2.4 km/s
        
        ### 🌌 Historical Context
        
        - **Johannes Kepler (1609)**: Discovered planetary motion laws
        - **Isaac Newton (1687)**: Formulated universal gravitation
        - **Albert Einstein (1915)**: Refined with General Relativity
        - **Modern Era**: Verified with extreme precision using laser interferometry
        """)
    
    # Additional theoretical content
    with st.expander("🔬 Deep Dive: Gravitational Fields and Potential Energy"):
        st.markdown("""
        ### 🌊 Gravitational Fields
        
        Think of gravitational fields as invisible "hills and valleys" in space around massive objects.
        
        **Field Strength (g):**
        - Measures how strong gravity is at any point
        - Vector quantity (has direction toward the mass)
        - Units: m/s² or N/kg (equivalent)
        
        **Field Lines:**
        - Imaginary lines showing direction of gravitational force
        - Always point toward the center of mass
        - Closer lines = stronger field
        
        ### ⚡ Gravitational Potential Energy
        
        **U = -GM₁m₂/r**
        
        - Negative because gravity is attractive
        - Zero potential energy at infinite distance
        - Lower potential = stronger gravitational binding
        
        **Energy Conservation:**
        - Total energy = Kinetic + Potential
        - E = ½mv² - GMm/r
        - Explains orbital velocities and escape conditions
        
        ### 🎯 Practical Examples
        
        **Satellite Orbital Speed:**
        - Circular orbit: v = √(GM/r)
        - Higher orbits = slower speeds
        - Geostationary orbit: 35,786 km altitude
        
        **Tidal Forces:**
        - Difference in gravitational pull across an object
        - F_tidal ∝ M/r³ (depends on distance cubed!)
        - Creates ocean tides, shapes moons and rings
        """)
    
    with st.expander("🎓 Quiz Yourself: Test Your Understanding"):
        st.markdown("""
        ### 🤔 Conceptual Questions
        
        1. **Why don't you feel the gravitational pull of nearby objects?**
           - Hint: Consider the masses involved and the gravitational constant
        
        2. **If Earth's mass doubled but radius stayed the same, how would your weight change?**
           - Think about the relationship between mass and gravitational force
        
        3. **Why do astronauts float in the International Space Station?**
           - Consider the concept of free fall and orbital motion
        
        ### 🧮 Calculation Practice
        
        **Problem 1:** Calculate the gravitational force between Earth (5.97×10²⁴ kg) 
        and the Moon (7.35×10²² kg) separated by 384,400 km.
        
        **Problem 2:** At what altitude would your weight be half of what it is on Earth's surface?
        
        **Problem 3:** What is the escape velocity from the surface of Mars? 
        (Mass: 6.42×10²³ kg, Radius: 3,390 km)
        
        *Try solving these and check your understanding with the interactive simulations below!*
        """)
    
    st.markdown("---")
      # Interactive simulation
    st.markdown("### 🔬 Interactive Simulation")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("**Simulation Parameters:**")
        
        # Interactive controls
        mass1 = st.slider("Mass 1 (in solar masses)", 0.1, 10.0, 1.0, 0.1)
        mass2 = st.slider("Mass 2 (in Earth masses)", 0.1, 100.0, 1.0, 0.1)
        separation = st.slider("Separation (in AU)", 0.1, 5.0, 1.0, 0.1)
        grid_resolution = st.selectbox("Grid Resolution", [32, 64, 96, 128], index=1)
        
        # Convert units
        mass1_kg = mass1 * 1.989e30  # Solar mass to kg
        mass2_kg = mass2 * 5.97e24   # Earth mass to kg
        separation_m = separation * 1.496e11  # AU to meters
        
        # Display calculated values
        st.markdown("**Calculated Values:**")
        orbital_period_days = 2 * np.pi * np.sqrt(separation_m**3 / (6.67e-11 * (mass1_kg + mass2_kg))) / (24 * 3600)
        st.metric("Orbital Period", f"{orbital_period_days:.1f} days")
        
        # Load and run simulation
        if st.button("🚀 Run Custom Simulation"):
            with st.spinner("Generating physics simulation..."):
                # Create custom parameters
                custom_params = {
                    "mass1": mass1_kg,
                    "mass2": mass2_kg,
                    "separation": separation_m,
                    "domain_size": separation_m * 3,
                    "time_span": orbital_period_days * 24 * 3600 * 2  # 2 orbital periods
                }
                
                # Override default scenario temporarily
                original_params = st.session_state.simulations.default_scenarios["binary_orbit"]["parameters"]
                st.session_state.simulations.default_scenarios["binary_orbit"]["parameters"] = custom_params
                
                # Generate simulation
                simulation_data = st.session_state.simulations.load_simulation(
                    "binary_orbit", 
                    (grid_resolution, grid_resolution, grid_resolution//2)
                )
                
                # Restore original parameters
                st.session_state.simulations.default_scenarios["binary_orbit"]["parameters"] = original_params
                
                # Create visualization
                fig = st.session_state.visualizer.create_3d_field_visualization(
                    simulation_data, 
                    visualization_type="potential"
                )
                
                st.session_state.current_simulation = simulation_data
                st.session_state.current_figure = fig
        
        # Preset simulations
        st.markdown("---")
        st.markdown("**Quick Presets:**")
        
        col_preset1, col_preset2 = st.columns(2)
        with col_preset1:
            if st.button("🌍 Earth-Moon"):
                st.session_state.preset_params = {
                    "mass1": 1.0, "mass2": 0.012, "separation": 0.00257, "resolution": 64
                }
        with col_preset2:
            if st.button("⭐ Binary Star"):
                st.session_state.preset_params = {
                    "mass1": 2.0, "mass2": 1.5, "separation": 2.0, "resolution": 96
                }
        
        # Apply preset if selected
        if 'preset_params' in st.session_state:
            params = st.session_state.preset_params
            # Update sliders via session state (this is a workaround)
            st.info(f"Preset applied! Adjust sliders to: Mass1={params['mass1']}, Mass2={params['mass2']}, Sep={params['separation']}")
            del st.session_state.preset_params
    
    with col2:
        if 'current_figure' in st.session_state:
            st.plotly_chart(st.session_state.current_figure, use_container_width=True)
        else:
            st.info("Click 'Run Simulation' to see the gravitational field visualization")
      # Animation section
    if 'current_simulation' in st.session_state:
        st.markdown("### 🎬 Interactive Video Simulations")
        
        col_anim1, col_anim2, col_anim3 = st.columns(3)
        
        with col_anim1:
            if st.button("▶️ Plotly Animation"):
                with st.spinner("Creating interactive animation..."):
                    animation_fig = interactive_engine.create_plotly_animated_simulation(
                        st.session_state.current_simulation
                    )
                    st.plotly_chart(animation_fig, use_container_width=True)
        
        with col_anim2:
            if st.button("🎮 Pygame Simulation"):
                with st.spinner("Generating video simulation..."):
                    try:
                        # Get simulation parameters
                        masses = st.session_state.current_simulation.get('masses', {})
                        m1 = masses.get('m1', 1.989e30)
                        m2 = masses.get('m2', 5.97e24)
                        
                        # Create pygame video
                        video_path = interactive_engine.create_pygame_orbital_simulation(
                            m1, m2, 1.5e11, duration=5
                        )
                        
                        # Display video
                        if video_path.endswith('.mp4'):
                            video_base64 = interactive_engine.get_video_base64(video_path)
                            if video_base64:
                                video_html = f"""
                                <video width="100%" height="300" controls autoplay loop>
                                    <source src="data:video/mp4;base64,{video_base64}" type="video/mp4">
                                    Your browser does not support the video tag.
                                </video>
                                """
                                st.markdown(video_html, unsafe_allow_html=True)
                        else:
                            # If it's HTML, display as iframe
                            with open(video_path, 'r') as f:
                                html_content = f.read()
                            st.components.v1.html(html_content, height=400)
                    except Exception as e:
                        st.error(f"Failed to create pygame simulation: {e}")
                        st.info("Using fallback Plotly animation...")
                        animation_fig = st.session_state.visualizer.create_trajectory_animation(
                            st.session_state.current_simulation
                        )
                        st.plotly_chart(animation_fig, use_container_width=True)
        
        with col_anim3:
            if st.button("🎲 Physics Engine"):
                with st.spinner("Running 3D physics simulation..."):
                    try:
                        # Create 3D simulation objects
                        masses = st.session_state.current_simulation.get('masses', {})
                        m1 = masses.get('m1', 1.989e30)
                        m2 = masses.get('m2', 5.97e24)
                        
                        objects = [
                            {
                                'mass': m1/1e30,  # Scale down for PyBullet
                                'radius': 0.5,
                                'position': [-2, 0, 0],
                                'color': [1, 1, 0, 1]  # Yellow
                            },
                            {
                                'mass': m2/1e30,  # Scale down for PyBullet
                                'radius': 0.3,
                                'position': [2, 0, 0],
                                'color': [0.4, 0.4, 1, 1]  # Blue
                            }
                        ]
                        
                        video_path = interactive_engine.create_pybullet_3d_simulation(
                            objects, duration=5
                        )
                        
                        # Display video
                        video_base64 = interactive_engine.get_video_base64(video_path)
                        if video_base64:
                            video_html = f"""
                            <video width="100%" height="300" controls autoplay loop>
                                <source src="data:video/mp4;base64,{video_base64}" type="video/mp4">
                                Your browser does not support the video tag.
                            </video>
                            """
                            st.markdown(video_html, unsafe_allow_html=True)
                        
                    except Exception as e:
                        st.error(f"Failed to create 3D physics simulation: {e}")
                        st.info("This feature requires additional setup. Using standard animation...")
                        animation_fig = st.session_state.visualizer.create_trajectory_animation(
                            st.session_state.current_simulation
                        )
                        st.plotly_chart(animation_fig, use_container_width=True)
    
    # Educational Game Section
    st.markdown("---")
    st.markdown("### 🎮 Interactive Physics Game: Gravity Drop Challenge")
    
    # Initialize game runner if not exists
    if 'game_runner' not in st.session_state:
        st.session_state.game_runner = StreamlitGameRunner()
    
    with st.container():
        st.markdown("""
        **🎯 Game Objective:** Master projectile motion by launching balls to hit targets!
        Learn how gravity affects trajectory and develop physics intuition through gameplay.
        """)
        
        # Game interface
        game_col1, game_col2 = st.columns([1, 2])
        
        with game_col1:
            st.markdown("**🚀 Launch Controls**")
            
            # Game parameters
            launch_angle = st.slider("Launch Angle (°)", 0, 90, 45, key="gravity_game_angle")
            launch_speed = st.slider("Launch Speed (m/s)", 10, 200, 100, key="gravity_game_speed")
            gravity_strength = st.slider("Gravity (m/s²)", 1.0, 20.0, 9.81, 0.1, key="gravity_game_gravity")
            
            # Calculate velocity components
            angle_rad = np.radians(launch_angle)
            vx = launch_speed * np.cos(angle_rad)
            vy = -launch_speed * np.sin(angle_rad)  # Negative for upward
            
            # Game instructions
            st.markdown("""
            **📋 Instructions:**
            1. Adjust launch angle and speed
            2. Try different gravity settings
            3. Click "Launch Ball" to fire
            4. Hit all targets to advance!
            
            **🧠 Physics Learning:**
            - See how gravity affects trajectory
            - Understand projectile motion
            - Develop intuition for launch parameters
            """)
            
            # Launch button
            if st.button("🚀 Launch Ball", key="gravity_launch_btn"):
                action = {
                    'launch_ball': True,
                    'x': 50, 'y': 500,  # Start position
                    'vx': vx, 'vy': vy,
                    'change_gravity': True,
                    'gravity': gravity_strength
                }
                
                # Run game step
                game_state = st.session_state.game_runner.run_game_step('gravity_drop', action)
                if game_state:
                    st.session_state.gravity_game_state = game_state
            
            # Reset game
            if st.button("🔄 Reset Game", key="gravity_reset_btn"):
                st.session_state.game_runner.games['gravity_drop'] = st.session_state.game_runner.games['gravity_drop'].__class__()
                if 'gravity_game_state' in st.session_state:
                    del st.session_state.gravity_game_state
        
        with game_col2:
            st.markdown("**🎯 Game Arena**")
            
            # Display game state
            if 'gravity_game_state' in st.session_state:
                st.image(f"data:image/png;base64,{st.session_state.gravity_game_state}", 
                        caption="Gravity Drop Challenge - Hit all targets!", 
                        use_column_width=True)
            else:
                # Show trajectory prediction when no game active
                st.markdown("**📈 Trajectory Prediction**")
                
                # Calculate and display predicted trajectory
                t_flight = 2 * abs(vy) / gravity_strength if vy < 0 else 1
                t = np.linspace(0, t_flight, 100)
                
                x_traj = vx * t
                y_traj = 500 + vy * t + 0.5 * gravity_strength * t**2  # Start at y=500
                
                # Remove points below ground
                valid_points = y_traj >= 0
                x_traj = x_traj[valid_points]
                y_traj = y_traj[valid_points]
                
                if len(x_traj) > 0:
                    trajectory_fig = go.Figure()
                    trajectory_fig.add_trace(go.Scatter(
                        x=x_traj, y=y_traj,
                        mode='lines',
                        name=f'Trajectory (g={gravity_strength} m/s²)',
                        line=dict(color='blue', width=3)
                    ))
                    
                    # Add example targets
                    target_x = [150, 250, 350]
                    target_y = [300, 200, 250]
                    trajectory_fig.add_trace(go.Scatter(
                        x=target_x, y=target_y,
                        mode='markers',
                        name='Targets',
                        marker=dict(color='red', size=15, symbol='circle')
                    ))
                    
                    trajectory_fig.update_layout(
                        title=f"Predicted Trajectory: {launch_angle}° angle, {launch_speed} m/s",
                        xaxis_title="Horizontal Distance (pixels)",
                        yaxis_title="Height (pixels)",
                        yaxis=dict(autorange='reversed'),  # Flip Y axis for game coordinates
                        showlegend=True,
                        height=400
                    )
                    
                    st.plotly_chart(trajectory_fig, use_container_width=True)
                
                st.info("🎮 Click 'Launch Ball' to start the interactive game!")
        
        # Game tips and learning outcomes
        with st.expander("🎓 Learning Outcomes & Tips"):
            st.markdown("""
            **🎯 What You'll Learn:**
            - How launch angle affects projectile range and height
            - The relationship between gravity and trajectory curvature
            - Optimal angles for different target positions
            - Physics principles behind ballistics and projectile motion
            
            **💡 Pro Tips:**
            - 45° gives maximum range in a vacuum (no air resistance)
            - Lower gravity = flatter, longer trajectories
            - Higher gravity = steeper, shorter trajectories
            - Experiment with different combinations!
            
            **🔬 Real-World Applications:**
            - Artillery and ballistics calculations
            - Sports physics (basketball, golf, etc.)
            - Space mission trajectory planning
            - Video game physics engines
            """)
    
    # Quiz section
    show_quiz_section("gravity_basics")


def show_orbital_mechanics():
    """Orbital mechanics simulation and learning"""
    st.markdown("## 🚀 Orbital Mechanics")
    
    # Comprehensive Educational content
    with st.expander("📖 Learn: Kepler's Laws of Planetary Motion", expanded=True):
        st.markdown("""
        ### 🪐 Johannes Kepler's Revolutionary Discovery
        
        In the early 1600s, Johannes Kepler analyzed Tycho Brahe's precise observations of Mars 
        and discovered three fundamental laws that govern planetary motion.
        
        ### 📐 Kepler's Three Laws
        
        #### **First Law: The Law of Ellipses (1609)**
        *"Planets orbit the Sun in elliptical paths with the Sun at one focus"*
        
        - **Ellipse Properties:**
          - Two foci (plural of focus)
          - Semi-major axis (a): half the longest diameter
          - Semi-minor axis (b): half the shortest diameter
          - Eccentricity (e): how "stretched" the ellipse is
          - e = 0 → perfect circle, e → 1 → very elongated
        
        - **Real Examples:**
          - Earth's orbit: e = 0.0167 (nearly circular)
          - Mercury's orbit: e = 0.206 (more elliptical)
          - Halley's Comet: e = 0.967 (highly elliptical)
        
        #### **Second Law: The Law of Equal Areas (1609)**
        *"A line connecting a planet to the Sun sweeps equal areas in equal time intervals"*
        
        - **Physical Meaning:**
          - Planets move faster when closer to the Sun
          - Planets move slower when farther from the Sun
          - Conservation of angular momentum: L = mvr = constant
        
        - **Practical Applications:**
          - Earth moves fastest in January (perihelion)
          - Earth moves slowest in July (aphelion)
          - Explains seasonal variations in solar energy
        
        #### **Third Law: The Harmonic Law (1619)**
        *"The square of orbital period is proportional to the cube of the semi-major axis"*
        
        - **Mathematical Form:** T² ∝ a³
        - **Complete Equation:** T² = (4π²/GM) × a³
        - **Implications:**
          - Outer planets have much longer years
          - Mars: 1.88 Earth years, Jupiter: 11.86 Earth years
          - Used to determine masses of distant objects
        
        ### 🎯 Modern Applications
        
        **Spacecraft Mission Design:**
        - Hohmann transfer orbits for efficient travel
        - Gravity assists to save fuel
        - Orbital rendezvous and docking
        
        **Satellite Operations:**
        - Geostationary orbits for communication
        - Sun-synchronous orbits for Earth observation
        - Lagrange points for space telescopes
        
        ### 🧮 Mathematical Framework
        
        **Orbital Velocity:**
        - Circular orbit: v = √(GM/r)
        - Elliptical orbit: v = √(GM(2/r - 1/a))
        
        **Orbital Energy:**
        - Total energy: E = -GMm/(2a)
        - Always negative for bound orbits
        - Higher energy = larger, faster orbits
        """)
    
    with st.expander("🌌 Advanced Topics: Perturbations and N-Body Problems"):
        st.markdown("""
        ### 🌊 Orbital Perturbations
        
        Real orbits aren't perfect ellipses due to various disturbing forces:
        
        **Gravitational Perturbations:**
        - Other planets' gravitational pull
        - Non-spherical mass distribution
        - Tidal forces from moons
        
        **Non-Gravitational Forces:**
        - Atmospheric drag (low Earth orbit)
        - Solar radiation pressure
        - Magnetic field interactions
        
        ### 🎯 The Three-Body Problem
        
        **Classical Problem:**
        - Three masses interacting gravitationally
        - Generally no analytical solution
        - Chaotic behavior possible
        
        **Special Solutions:**
        - Lagrange points (L1-L5)
        - Trojan asteroids at L4 and L5
        - James Webb Space Telescope at L2
        
        **Restricted Three-Body Problem:**
        - Two large masses, one small test mass
        - Circular Restricted Three-Body Problem (CR3BP)
        - Foundation for spacecraft trajectory design
        
        ### 🛰️ Spacecraft Dynamics
        
        **Orbital Maneuvers:**
        - Hohmann transfers: efficient but slow
        - Bi-elliptic transfers: for large changes
        - Plane changes: most expensive maneuvers
        
        **Station-Keeping:**
        - Maintaining precise orbits
        - Compensating for perturbations
        - Fuel budget considerations
        
        ### 🔢 Numerical Methods
        
        **Integration Techniques:**
        - Runge-Kutta methods for precision
        - Symplectic integrators for long-term stability
        - Adaptive step-size control
        
        **Modern Applications:**
        - Galaxy simulations with millions of stars
        - Asteroid deflection mission planning
        - Space debris tracking and avoidance
        """)
    
    with st.expander("🎓 Problem-Solving Strategies"):
        st.markdown("""
        ### 🧮 Step-by-Step Approach
        
        **1. Identify the System:**
        - What objects are involved?
        - What forces are significant?
        - What approximations are valid?
        
        **2. Choose Coordinate System:**
        - Cartesian (x, y, z) for general problems
        - Polar (r, θ) for circular motion
        - Orbital elements for spacecraft
        
        **3. Apply Conservation Laws:**
        - Energy conservation: E = K + U = constant
        - Angular momentum: L = r × mv = constant
        - Linear momentum (if no external forces)
        
        **4. Use Kepler's Laws:**
        - Determine orbital shape and size
        - Calculate periods and velocities
        - Predict future positions
        
        ### 🎯 Example Problems
        
        **Problem 1: Satellite Orbit Design**
        Design a circular orbit for Earth observation with 90-minute period.
        - Given: T = 90 min = 5400 s
        - Find: orbital radius and altitude
        - Use: T² = (4π²/GM) × r³
        
        **Problem 2: Interplanetary Transfer**
        Calculate the time to travel from Earth to Mars using Hohmann transfer.
        - Earth orbit: 1 AU, Mars orbit: 1.52 AU
        - Transfer orbit semi-major axis: (1 + 1.52)/2 = 1.26 AU
        - Transfer time: ½ × T_transfer
        
        **Problem 3: Escape Velocity**
        What velocity does a spacecraft need to escape Earth's gravity?
        - Use energy conservation: ½mv² - GMm/R = 0
        - Solve for v: v_escape = √(2GM/R) = 11.2 km/s
        
        ### 💡 Useful Tips
        
        - Always check units and dimensional analysis
        - Sketch the problem to visualize geometry
        - Use energy methods when possible (often simpler)
        - Remember that gravity always does negative work when moving away
        - Conservation laws are your best friends!
        """)
    
    st.markdown("---")
      # Interactive simulation
    st.markdown("### 🪐 Interactive Solar System Simulation")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("**System Configuration:**")
        scenario_type = st.selectbox(
            "Choose simulation type:",
            ["planetary_system", "binary_orbit", "custom_system"],
            help="Select the type of gravitational system to simulate"
        )
        
        if scenario_type == "custom_system":
            st.markdown("**Custom System Parameters:**")
            central_mass = st.slider("Central Star Mass (solar masses)", 0.5, 5.0, 1.0, 0.1)
            num_planets = st.slider("Number of Planets", 1, 6, 3)
            
            planet_configs = []
            for i in range(num_planets):
                with st.expander(f"Planet {i+1} Configuration"):
                    mass = st.slider(f"Planet {i+1} Mass (Earth masses)", 0.1, 20.0, 1.0, 0.1, key=f"mass_{i}")
                    distance = st.slider(f"Planet {i+1} Distance (AU)", 0.3, 10.0, i+1.0, 0.1, key=f"dist_{i}")
                    planet_configs.append({"mass": mass * 5.97e24, "distance": distance * 1.496e11})
        
        # Visualization options
        st.markdown("**Visualization Options:**")
        show_trajectories = st.checkbox("Show Orbital Trajectories", True)
        show_field_lines = st.checkbox("Show Gravitational Field", False)
        grid_resolution = st.slider("Simulation Resolution", 32, 128, 64, step=16)
        
        # Animation controls
        time_scale = st.slider("Animation Speed", 0.1, 5.0, 1.0, 0.1)
        
        # Run simulation button
        if st.button("🌍 Run Interactive Simulation", type="primary"):
            with st.spinner("Computing orbital mechanics..."):
                if scenario_type == "custom_system":
                    # Create custom parameters
                    custom_params = {
                        "central_mass": central_mass * 1.989e30,
                        "planet_masses": [p["mass"] for p in planet_configs],
                        "planet_distances": [p["distance"] for p in planet_configs],
                        "domain_size": max([p["distance"] for p in planet_configs]) * 2,
                        "time_span": 2 * 365.25 * 24 * 3600
                    }
                    
                    # Override default scenario
                    original_params = st.session_state.simulations.default_scenarios["planetary_system"]["parameters"]
                    st.session_state.simulations.default_scenarios["planetary_system"]["parameters"] = custom_params
                    
                    simulation_data = st.session_state.simulations.load_simulation(
                        "planetary_system", 
                        (grid_resolution, grid_resolution, grid_resolution//2)
                    )
                    
                    # Restore original
                    st.session_state.simulations.default_scenarios["planetary_system"]["parameters"] = original_params
                else:
                    simulation_data = st.session_state.simulations.load_simulation(
                        scenario_type, 
                        (grid_resolution, grid_resolution, grid_resolution//2)
                    )
                
                # Store visualization preferences
                simulation_data["viz_options"] = {
                    "show_trajectories": show_trajectories,
                    "show_field_lines": show_field_lines,
                    "time_scale": time_scale
                }
                
                # Create dashboard
                dashboard = st.session_state.visualizer.create_interactive_dashboard(simulation_data)
                st.session_state.current_dashboard = dashboard
                st.session_state.current_orbital_data = simulation_data
    
    with col2:
        if 'current_dashboard' in st.session_state:
            dashboard = st.session_state.current_dashboard
            
            # Main 3D visualization
            if "main_3d" in dashboard:
                st.plotly_chart(dashboard["main_3d"], use_container_width=True)
        else:
            st.info("Select parameters and run simulation to see orbital mechanics")
      # Additional visualizations
    if 'current_dashboard' in st.session_state:
        dashboard = st.session_state.current_dashboard
        st.markdown("### 📊 Orbital Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if "trajectories" in dashboard:
                st.plotly_chart(dashboard["trajectories"], use_container_width=True)
        
        with col2:
            if "orbital_analysis" in dashboard:
                st.plotly_chart(dashboard["orbital_analysis"], use_container_width=True)
    
    # Educational Game Section
    st.markdown("---")
    st.markdown("### 🎮 Interactive Physics Game: Orbital Designer")
    
    # Initialize game runner if not exists
    if 'game_runner' not in st.session_state:
        st.session_state.game_runner = StreamlitGameRunner()
    
    with st.container():
        st.markdown("""
        **🛰️ Game Objective:** Design stable circular orbits at different target radii!
        Master the relationship between orbital velocity and radius through interactive gameplay.
        """)
        
        # Game interface
        orbit_col1, orbit_col2 = st.columns([1, 2])
        
        with orbit_col1:
            st.markdown("**🛰️ Satellite Launch Parameters**")
            
            # Orbital design parameters
            target_radius = st.slider("Target Orbit Radius", 100, 400, 200, key="orbit_radius")
            
            # Calculate theoretical orbital velocity
            G_scaled = 6.67e-11 * 1e6  # Scaled gravitational constant
            M_central = 1e6  # Central body mass
            theoretical_velocity = np.sqrt(G_scaled * M_central / target_radius)
            
            st.metric("Theoretical Orbital Velocity", f"{theoretical_velocity:.1f} units/s")
            
            # Velocity adjustment
            velocity_factor = st.slider("Velocity Factor", 0.5, 2.0, 1.0, 0.1, key="orbit_velocity_factor")
            actual_velocity = theoretical_velocity * velocity_factor
            
            st.metric("Actual Launch Velocity", f"{actual_velocity:.1f} units/s")
            
            # Orbit prediction
            if velocity_factor < 0.9:
                orbit_prediction = "❌ Insufficient velocity - will crash!"
            elif velocity_factor > 1.1:
                orbit_prediction = "🚀 Excess velocity - will escape!"
            else:
                orbit_prediction = "✅ Good velocity for stable orbit"
            
            st.markdown(f"**Orbit Prediction:** {orbit_prediction}")
            
            # Game instructions
            st.markdown("""
            **📋 Instructions:**
            1. Choose your target orbital radius
            2. Adjust velocity factor
            3. Launch satellite and observe orbit
            4. Try to match the green target circles!
            
            **🧠 Physics Learning:**
            - v = √(GM/r) for circular orbits
            - Higher orbits need slower speeds
            - Velocity too high = escape trajectory
            - Velocity too low = crash trajectory
            """)
            
            # Launch button
            if st.button("🛰️ Launch Satellite", key="orbit_launch_btn"):
                # Calculate launch position and velocity
                launch_x = 400 + target_radius  # Center x + radius
                launch_y = 300  # Center y
                launch_vx = 0  # Tangential velocity
                launch_vy = actual_velocity
                
                action = {
                    'add_satellite': True,
                    'x': launch_x, 'y': launch_y, 
                    'vx': launch_vx, 'vy': launch_vy
                }
                
                # Run game step
                game_state = st.session_state.game_runner.run_game_step('orbit_design', action)
                if game_state:
                    st.session_state.orbit_game_state = game_state
            
            # Reset game
            if st.button("🔄 Reset Orbital System", key="orbit_reset_btn"):
                st.session_state.game_runner.games['orbit_design'] = st.session_state.game_runner.games['orbit_design'].__class__()
                if 'orbit_game_state' in st.session_state:
                    del st.session_state.orbit_game_state
        
        with orbit_col2:
            st.markdown("**🌌 Orbital System Viewer**")
            
            # Display game state
            if 'orbit_game_state' in st.session_state:
                st.image(f"data:image/png;base64,{st.session_state.orbit_game_state}", 
                        caption="Orbital Designer - Create stable orbits!", 
                        use_column_width=True)
            else:
                # Show orbital velocity relationship
                st.markdown("**📈 Orbital Velocity vs Radius**")
                
                # Create orbital mechanics visualization
                radii = np.linspace(100, 400, 100)
                velocities = np.sqrt(G_scaled * M_central / radii)
                periods = 2 * np.pi * radii / velocities
                
                orbital_fig = go.Figure()
                
                # Velocity curve
                orbital_fig.add_trace(go.Scatter(
                    x=radii, y=velocities,
                    mode='lines',
                    name='Orbital Velocity',
                    line=dict(color='blue', width=3)
                ))
                
                # Highlight current target
                target_vel = np.sqrt(G_scaled * M_central / target_radius)
                orbital_fig.add_trace(go.Scatter(
                    x=[target_radius], y=[target_vel],
                    mode='markers',
                    name=f'Target ({target_radius}, {target_vel:.1f})',
                    marker=dict(color='red', size=12, symbol='diamond')
                ))
                
                orbital_fig.update_layout(
                    title="Orbital Velocity vs Radius Relationship",
                    xaxis_title="Orbital Radius (units)",
                    yaxis_title="Orbital Velocity (units/s)",
                    showlegend=True,
                    height=400
                )
                
                st.plotly_chart(orbital_fig, use_container_width=True)
                
                st.info("🎮 Click 'Launch Satellite' to start the orbital design game!")
        
        # Orbital mechanics learning section
        with st.expander("🎓 Orbital Mechanics Learning"):
            st.markdown("""
            **🚀 Key Orbital Concepts:**
            
            **Circular Orbital Velocity:**
            - Formula: v = √(GM/r)
            - Closer orbits require higher velocity
            - Farther orbits require lower velocity
            
            **Escape Velocity:**
            - v_escape = √(2GM/r) = √2 × v_orbital
            - Minimum speed to escape gravitational pull
            
            **Orbital Period (Kepler's 3rd Law):**
            - T = 2π√(r³/GM)
            - Period increases with orbital radius
            - Higher orbits take longer to complete
            
            **Real-World Applications:**
            - Satellite deployment strategies
            - Interplanetary mission planning
            - Space station orbital adjustments
            - Geostationary satellite positioning
            
            **🎯 Game Strategy Tips:**
            - Start with velocity factor close to 1.0
            - Watch how small changes affect orbit stability
            - Try different target radii to see the pattern
            - Observe how satellites with wrong velocities behave
            """)
    
    # Quiz section
    show_quiz_section("orbital_mechanics")


def show_black_hole_physics():
    """Black hole physics simulation and learning"""
    st.markdown("## ⚫ Black Hole Physics")
    
    # Comprehensive Educational content
    with st.expander("📖 Learn: General Relativity and Black Holes", expanded=True):
        st.markdown("""
        ### 🌌 Einstein's Revolutionary Theory
        
        **General Relativity (1915)** fundamentally changed our understanding of gravity, 
        space, and time. Einstein showed that gravity isn't a force—it's the curvature of spacetime itself.
        
        ### 🧮 Key Concepts
        
        #### **Spacetime Curvature**
        *"Matter tells spacetime how to curve, and curved spacetime tells matter how to move"* - John Wheeler
        
        - **Spacetime**: The unified fabric of space and time (4 dimensions)
        - **Curvature**: Mass and energy bend spacetime like a heavy ball on a stretched rubber sheet
        - **Geodesics**: Objects follow the "straightest" paths in curved spacetime
        
        #### **The Einstein Field Equations**
        **Gμν + Λgμν = (8πG/c⁴)Tμν**
        
        - **Left side**: Describes spacetime curvature
        - **Right side**: Describes matter and energy distribution
        - **Most beautiful equation in physics**: Links geometry to physics
        
        ### ⚫ Black Hole Formation
        
        **Stellar Evolution Path:**
        1. **Main Sequence Star** (hydrogen fusion)
        2. **Red Giant/Supergiant** (heavier element fusion)
        3. **Supernova Explosion** (core collapse)
        4. **Black Hole** (if core mass > 3 solar masses)
        
        **Critical Mass Limits:**
        - **White Dwarf**: < 1.4 solar masses (Chandrasekhar limit)
        - **Neutron Star**: 1.4 - 3 solar masses (Tolman-Oppenheimer-Volkoff limit)
        - **Black Hole**: > 3 solar masses (no known upper limit)
        
        ### 📐 The Schwarzschild Solution
        
        **Schwarzschild Radius (Event Horizon):**
        **Rs = 2GM/c²**
        
        - **Physical Meaning**: The "point of no return"
        - **For the Sun**: Rs ≈ 3 km (Sun would need to be compressed to this size)
        - **For Earth**: Rs ≈ 9 mm (smaller than a marble!)
        - **Supermassive Black Holes**: Rs can be larger than our solar system
        
        **Properties of the Event Horizon:**
        - Nothing can escape once inside (not even light)
        - One-way membrane: you can fall in, but never get out
        - Surface area never decreases (Hawking's Area Theorem)
        - Information paradox: what happens to information that falls in?
        
        ### ⏰ Relativistic Effects
        
        #### **Time Dilation**
        **Δt = Δt₀/√(1 - Rs/r)**
        
        - Time slows down in strong gravitational fields
        - At the event horizon, time appears to stop for outside observers
        - Falling observer experiences normal time until... well, we don't know!
        
        #### **Gravitational Redshift**
        **λ_observed = λ_emitted × √(1 - Rs/r)**
        
        - Light loses energy climbing out of gravitational wells
        - Frequency decreases (wavelength increases)
        - Light from near black holes appears very red
        
        #### **Frame Dragging (Kerr Black Holes)**
        - Rotating black holes drag spacetime around them
        - Creates "ergosphere" outside event horizon
        - Penrose process: can extract energy from rotation
        
        ### 🌊 Tidal Forces
        
        **"Spaghettification":**
        - Difference in gravitational pull from head to toe
        - Tidal force ∝ M/r³ (varies as distance cubed!)
        - Stretches objects radially, compresses them laterally
        - Stellar-mass black holes: deadly before reaching horizon
        - Supermassive black holes: can cross horizon relatively safely
        
        ### 💫 Hawking Radiation
        
        **Stephen Hawking's 1974 Discovery:**
        - Quantum effects near event horizon create particle pairs
        - One particle falls in, one escapes
        - Black hole slowly evaporates!
        
        **Hawking Temperature:**
        **T = ħc³/(8πGMkB) ≈ 6 × 10⁻⁸ K × (M☉/M)**
        
        - Smaller black holes are hotter
        - Stellar-mass black holes: colder than cosmic microwave background
        - Primordial black holes: could be very hot and detectable
        
        ### 🔭 Observational Evidence
        
        **Direct Detection:**
        - **Event Horizon Telescope (2019)**: First image of M87* black hole
        - **Sagittarius A* (2022)**: Black hole at center of Milky Way
        - **LIGO/Virgo**: Gravitational waves from black hole mergers
        
        **Indirect Evidence:**
        - X-ray emissions from accretion disks
        - Stellar orbits around Sagittarius A*
        - Gravitational lensing effects
        - Jets and outflows from active galactic nuclei
        """)
    
    with st.expander("🌌 Advanced Black Hole Physics"):
        st.markdown("""
        ### 🔄 Types of Black Holes
        
        #### **Schwarzschild Black Holes (Non-rotating)**
        - Spherically symmetric
        - Described by mass only
        - Simplest case, but probably rare in nature
        
        #### **Kerr Black Holes (Rotating)**
        - Described by mass and angular momentum
        - Most realistic for astrophysical black holes
        - Creates ergosphere and frame-dragging effects
        
        #### **Reissner-Nordström Black Holes (Charged)**
        - Described by mass and electric charge
        - Probably rare (quickly neutralized in space)
        - Can have two event horizons
        
        #### **Kerr-Newman Black Holes (Rotating + Charged)**
        - Most general case
        - Described by mass, charge, and angular momentum
        - Theoretical completeness
        
        ### 🌪️ Black Hole Thermodynamics
        
        **The Four Laws:**
        
        **0th Law**: Surface gravity is constant on the event horizon
        **1st Law**: dM = (κ/8πG)dA + ΩdJ + ΦdQ
        **2nd Law**: Event horizon area never decreases (classically)
        **3rd Law**: Cannot reach zero surface gravity in finite steps
        
        **Bekenstein-Hawking Entropy:**
        **S = Ac³/(4Gℏ) = A/(4 × Planck area)**
        
        - Entropy proportional to surface area (not volume!)
        - Fundamental limit on information storage
        - Connection between gravity and thermodynamics
        
        ### 🌊 Gravitational Waves from Black Holes
        
        **Binary Black Hole Mergers:**
        - Inspiraling phase: chirp signal
        - Merger phase: most energy released
        - Ringdown phase: final black hole settles
        
        **LIGO/Virgo Discoveries:**
        - GW150914: First detection (36 + 29 → 62 solar masses)
        - GW170817: Neutron star merger with electromagnetic counterpart
        - 90+ confirmed detections and counting!
        
        ### 🧩 Unsolved Mysteries
        
        **Information Paradox:**
        - Hawking radiation appears thermal (no information)
        - Conflicts with quantum mechanics (information conservation)
        - Proposed solutions: firewalls, complementarity, holography
        
        **Singularities:**
        - Classical physics predicts infinite density
        - Quantum gravity needed for complete description
        - May be resolved by string theory or loop quantum gravity
        
        **Dark Matter Connection:**
        - Primordial black holes as dark matter candidates?
        - Intermediate-mass black hole population
        - Supermassive black hole formation mechanisms
        """)
    
    with st.expander("🎓 Mathematical Tools and Problem Solving"):
        st.markdown("""
        ### 📊 Key Equations Summary
        
        **Schwarzschild Metric:**
        ds² = -(1-Rs/r)c²dt² + dr²/(1-Rs/r) + r²(dθ² + sin²θdφ²)
        
        **Orbital Frequency (Circular Orbits):**
        Ω = √(GM/r³) × (1 + 3Rs/(2r))  [Relativistic correction]
        
        **Photon Sphere:**
        r = 3Rs/2 (unstable circular orbits for light)
        
        **Innermost Stable Circular Orbit (ISCO):**
        r = 6Rs (for Schwarzschild), r = Rs (for maximally rotating Kerr)
        
        ### 🎯 Problem-Solving Approach
        
        **1. Identify the Black Hole Type:**
        - Schwarzschild (mass only)
        - Kerr (mass + rotation)
        - Consider realistic astrophysical parameters
        
        **2. Choose Appropriate Coordinates:**
        - Schwarzschild coordinates for distant observers
        - Eddington-Finkelstein for infalling observers
        - Boyer-Lindquist for Kerr black holes
        
        **3. Apply Conservation Laws:**
        - Energy conservation
        - Angular momentum conservation
        - Carter constant (for Kerr black holes)
        
        **4. Use Relativistic Corrections:**
        - Time dilation effects
        - Precession of orbits
        - Frame-dragging effects
        
        ### 💡 Practical Examples
        
        **Example 1: Schwarzschild Radius**
        Calculate the Schwarzschild radius for a 10 solar mass black hole.
        Rs = 2GM/c² = 2 × 6.67×10⁻¹¹ × 10×1.99×10³⁰ / (3×10⁸)² ≈ 30 km
        
        **Example 2: Time Dilation**
        How much does time slow down at distance r = 3Rs?
        Δt/Δt₀ = 1/√(1-Rs/r) = 1/√(1-1/3) = 1/√(2/3) ≈ 1.22
        Time runs 22% slower compared to infinity.
        
        **Example 3: Hawking Temperature**
        What's the temperature of a solar-mass black hole?
        T = 6×10⁻⁸ K × (M☉/M☉) = 6×10⁻⁸ K
        Extremely cold—much colder than the cosmic microwave background (2.7 K)!
        """)
    
    st.markdown("---")
    
    # Interactive simulation
    st.markdown("### 🕳️ Black Hole Simulation")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        black_hole_mass = st.slider(
            "Black Hole Mass (solar masses)", 
            1, 50, 10
        )
        
        visualization_type = st.selectbox(
            "Visualization Type:",
            ["potential", "field_magnitude", "curvature"]
        )
        
        if st.button("🌌 Generate Black Hole"):
            with st.spinner("Solving Einstein's equations..."):
                # Update simulation parameters
                simulation_data = st.session_state.simulations.load_simulation("black_hole_accretion")
                
                # Create relativistic visualization
                fig = st.session_state.visualizer.create_3d_field_visualization(
                    simulation_data,
                    visualization_type=visualization_type
                )
                
                st.session_state.current_blackhole = simulation_data
                st.session_state.current_bh_figure = fig
    
    with col2:
        if 'current_bh_figure' in st.session_state:
            st.plotly_chart(st.session_state.current_bh_figure, use_container_width=True)
        else:
            st.info("Configure black hole parameters and generate simulation")
      # Relativistic effects
    if 'current_blackhole' in st.session_state:
        st.markdown("### ⚡ Relativistic Effects")
        
        dashboard = st.session_state.visualizer.create_interactive_dashboard(
            st.session_state.current_blackhole
        )
        
        if "relativistic_effects" in dashboard:
            st.plotly_chart(dashboard["relativistic_effects"], use_container_width=True)
    
    # Educational Game Section
    st.markdown("---")
    st.markdown("### 🎮 Interactive Physics Game: Black Hole Escape")
    
    # Initialize game runner if not exists
    if 'game_runner' not in st.session_state:
        st.session_state.game_runner = StreamlitGameRunner()
    
    with st.container():
        st.markdown("""
        **🕳️ Game Objective:** Navigate your spacecraft around the black hole, collect items, and avoid the event horizon!
        Learn about extreme gravity, tidal forces, and relativistic effects through gameplay.
        """)
        
        # Game interface
        bh_col1, bh_col2 = st.columns([1, 2])
        
        with bh_col1:
            st.markdown("**🚀 Spacecraft Controls**")
            
            # Thrust controls
            thrust_power = st.slider("Thrust Power", 0.1, 3.0, 1.0, key="bh_thrust_power")
            
            # Direction selection
            thrust_direction = st.selectbox("Thrust Direction", 
                                          ["⬆️ Up", "⬇️ Down", "⬅️ Left", "➡️ Right", 
                                           "↖️ Up-Left", "↗️ Up-Right", "↙️ Down-Left", "↘️ Down-Right"],
                                          key="bh_thrust_direction")
            
            # Direction mapping
            direction_map = {
                "⬆️ Up": (0, -1), "⬇️ Down": (0, 1), "⬅️ Left": (-1, 0), "➡️ Right": (1, 0),
                "↖️ Up-Left": (-0.7, -0.7), "↗️ Up-Right": (0.7, -0.7),
                "↙️ Down-Left": (-0.7, 0.7), "↘️ Down-Right": (0.7, 0.7)
            }
            
            # Game status and warnings
            if 'blackhole_game_status' in st.session_state:
                status = st.session_state.blackhole_game_status
                if status.get('near_horizon', False):
                    st.error("⚠️ DANGER: Approaching Event Horizon!")
                elif status.get('tidal_forces', False):
                    st.warning("🌊 Strong Tidal Forces Detected!")
                else:
                    st.success("✅ Safe Distance from Black Hole")
            
            # Game instructions
            st.markdown("""
            **📋 Instructions:**
            1. Use thrust controls to navigate spacecraft
            2. Collect yellow items for points and fuel
            3. Stay outside the red event horizon!
            4. Manage fuel carefully - it's limited!
            
            **🧠 Physics Learning:**
            - Experience extreme gravitational pull
            - Understand event horizons
            - Feel the effects of tidal forces
            - Learn about relativistic effects
            """)
            
            # Control buttons
            col_thrust, col_reset = st.columns(2)
            with col_thrust:
                if st.button("🚀 Apply Thrust", key="bh_thrust_btn"):
                    tx, ty = direction_map[thrust_direction]
                    action = {
                        'thrust': True,
                        'thrust_x': tx * thrust_power,
                        'thrust_y': ty * thrust_power
                    }
                    
                    # Run game step
                    game_state = st.session_state.game_runner.run_game_step('black_hole_escape', action)
                    if game_state:
                        st.session_state.blackhole_game_state = game_state
            
            with col_reset:
                if st.button("🔄 Reset Mission", key="bh_reset_btn"):
                    st.session_state.game_runner.games['black_hole_escape'] = st.session_state.game_runner.games['black_hole_escape'].__class__()
                    if 'blackhole_game_state' in st.session_state:
                        del st.session_state.blackhole_game_state
        
        with bh_col2:
            st.markdown("**🕳️ Black Hole System**")
            
            # Display game state
            if 'blackhole_game_state' in st.session_state:
                st.image(f"data:image/png;base64,{st.session_state.blackhole_game_state}", 
                        caption="Black Hole Escape - Avoid the Event Horizon!", 
                        use_column_width=True)
            else:
                # Show black hole structure diagram
                st.markdown("**📊 Black Hole Structure**")
                
                # Create black hole visualization
                theta = np.linspace(0, 2*np.pi, 100)
                
                bh_fig = go.Figure()
                
                # Event horizon
                r_eh = 25
                x_eh = r_eh * np.cos(theta)
                y_eh = r_eh * np.sin(theta)
                bh_fig.add_trace(go.Scatter(
                    x=x_eh, y=y_eh, mode='lines', fill='toself',
                    name='Event Horizon', 
                    line=dict(color='red', width=3),
                    fillcolor='rgba(255, 0, 0, 0.3)'
                ))
                
                # Photon sphere
                r_ps = 37.5  # 1.5 * Schwarzschild radius
                x_ps = r_ps * np.cos(theta)
                y_ps = r_ps * np.sin(theta)
                bh_fig.add_trace(go.Scatter(
                    x=x_ps, y=y_ps, mode='lines',
                    name='Photon Sphere', 
                    line=dict(color='orange', dash='dash', width=2)
                ))
                
                # Safe orbital zones
                for i, r in enumerate([75, 100, 125, 150]):
                    x_orbit = r * np.cos(theta)
                    y_orbit = r * np.sin(theta)
                    bh_fig.add_trace(go.Scatter(
                        x=x_orbit, y=y_orbit, mode='lines',
                        name=f'Safe Zone {i+1}' if i == 0 else None,
                        showlegend=i == 0,
                        line=dict(color='green', dash='dot', width=1),
                        opacity=0.7 - i*0.1
                    ))
                
                # Example spacecraft position
                spacecraft_x, spacecraft_y = 100, 0
                bh_fig.add_trace(go.Scatter(
                    x=[spacecraft_x], y=[spacecraft_y],
                    mode='markers',
                    name='Starting Position',
                    marker=dict(color='blue', size=12, symbol='triangle-up')
                ))
                
                bh_fig.update_layout(
                    title="Black Hole Regions and Safe Zones",
                    xaxis_title="Distance (units)",
                    yaxis_title="Distance (units)",
                    showlegend=True,
                    width=600, height=500,
                    xaxis=dict(scaleanchor="y", scaleratio=1)
                )
                
                st.plotly_chart(bh_fig, use_container_width=True)
                
                st.info("🎮 Click 'Apply Thrust' to start navigating around the black hole!")
        
        # Black hole physics learning section
        with st.expander("🎓 Black Hole Physics Learning"):
            st.markdown("""
            **🕳️ Key Black Hole Concepts:**
            
            **Event Horizon:**
            - The "point of no return" around a black hole
            - Schwarzschild radius: r = 2GM/c²
            - Nothing can escape once inside (not even light!)
            
            **Tidal Forces:**
            - Difference in gravitational pull across an object
            - Creates stretching ("spaghettification")
            - Stronger for smaller black holes at same distance
            
            **Relativistic Effects:**
            - Time dilation: time slows in strong gravity
            - Gravitational redshift: light loses energy
            - Frame dragging: rotating black holes drag spacetime
            
            **Photon Sphere:**
            - Unstable orbit for light at r = 1.5 × Schwarzschild radius
            - Photons can orbit the black hole multiple times
            
            **🎯 Survival Strategy:**
            - Maintain thrust to counteract gravitational pull
            - Use orbital mechanics for fuel efficiency
            - Stay well outside the event horizon
            - Collect fuel items to extend mission time
            
            **Real-World Relevance:**
            - Understanding extreme environments in space
            - Navigation near massive objects
            - Theoretical physics concepts made tangible
            - Future space mission considerations
            """)
    
    # Quiz section
    show_quiz_section("black_holes")


def show_gravitational_waves():
    """Gravitational waves simulation and learning"""
    st.markdown("## 🌊 Gravitational Waves")
    
    # Comprehensive Educational content
    with st.expander("📖 Learn: Ripples in Spacetime", expanded=True):
        st.markdown("""
        ### 🌌 Einstein's Final Prediction
        
        In 1916, just one year after formulating General Relativity, Einstein predicted that 
        accelerating masses would create **ripples in spacetime itself**—gravitational waves.
        It took exactly 100 years to detect them!
        
        ### 🌊 What Are Gravitational Waves?
        
        **Physical Nature:**
        - Disturbances in the curvature of spacetime
        - Travel at the speed of light (c = 299,792,458 m/s)
        - Carry energy and angular momentum away from their source
        - Stretch and squeeze space itself as they pass
        
        **Mathematical Description:**
        The metric perturbation: **gμν = ημν + hμν**
        - ημν: flat spacetime (special relativity)
        - hμν: small perturbation (gravitational wave)
        - |hμν| << 1 (waves are tiny distortions)
        
        ### 📐 Wave Properties
        
        #### **Polarization States**
        Gravitational waves have two polarization modes:
        
        **Plus Polarization (+):**
        - Stretches space in x-direction
        - Compresses space in y-direction
        - Pattern rotates every half period
        
        **Cross Polarization (×):**
        - Rotated 45° from plus polarization
        - Same stretching/compressing pattern
        - Independent of plus mode
        
        #### **Strain Amplitude**
        **h = ΔL/L** (fractional change in length)
        
        - Typical values: h ~ 10⁻²¹ to 10⁻²³
        - For LIGO (4 km arms): ΔL ~ 10⁻¹⁸ m
        - Smaller than 1/10,000th the width of a proton!
        
        ### 🌟 Sources of Gravitational Waves
        
        #### **Binary Black Hole Mergers**
        **Three Phases:**
        
        1. **Inspiral Phase** (minutes to hours):
           - Two black holes spiral inward
           - Frequency increases ("chirp" signal)
           - Follows post-Newtonian dynamics
        
        2. **Merger Phase** (milliseconds):
           - Black holes touch and merge
           - Nonlinear, chaotic dynamics
           - Most energy radiated here
        
        3. **Ringdown Phase** (milliseconds):
           - Final black hole "rings" like a bell
           - Exponentially decaying oscillations
           - Reveals final mass and spin
        
        #### **Binary Neutron Star Mergers**
        - Similar inspiral and merger
        - Can produce electromagnetic counterparts
        - Kilonova explosions create heavy elements
        - GW170817: first multi-messenger detection
        
        #### **Other Sources**
        - **Supernovae**: Asymmetric core collapse
        - **Pulsar Glitches**: Sudden spin changes
        - **Cosmic Strings**: Hypothetical 1D defects
        - **Primordial Waves**: From cosmic inflation
        
        ### 📊 Energy and Power
        
        **Quadrupole Formula** (Einstein, 1918):
        **P = (G/5c⁵) × <d³Qᵢⱼ/dt³>²**
        
        - Power scales as frequency⁶
        - Higher frequencies radiate much more power
        - Explains why final merger is so bright
        
        **Binary Inspiral Power:**
        **P ∝ (GM)⁵/³ × f¹⁰/³ / c⁵**
        
        - More massive systems radiate more power
        - Higher frequencies radiate exponentially more
        - Causes orbital decay and "death spiral"
        
        ### 🔬 Detection Methods
        
        #### **Laser Interferometry (LIGO/Virgo)**
        **Principle:**
        - Michelson interferometer with 4 km arms
        - Laser light bounces between mirrors
        - Gravitational waves change arm lengths
        - Interference pattern reveals strain
        
        **Sensitivity Challenges:**
        - Seismic vibrations (low frequency)
        - Thermal noise in mirrors (mid frequency)
        - Shot noise from laser photons (high frequency)
        - Quantum noise at the detection limit
        
        #### **Pulsar Timing Arrays**
        - Use pulsars as cosmic clocks
        - Gravitational waves change pulse arrival times
        - Sensitive to nanohertz frequencies
        - Can detect supermassive black hole binaries
        
        #### **Space-Based Detectors**
        **LISA (Launch ~2034):**
        - Three spacecraft in triangular formation
        - 2.5 million km arm length
        - Millihertz frequency band
        - Can observe for years, not seconds
        
        ### 🏆 Historic Detections
        
        **GW150914 (September 14, 2015):**
        - First direct detection of gravitational waves
        - Binary black hole merger: 36 + 29 → 62 solar masses
        - 3 solar masses converted to gravitational wave energy
        - Peak power: 3.6 × 10⁴⁹ watts (more than all stars in observable universe!)
        
        **GW170817 (August 17, 2017):**
        - First neutron star merger detection
        - Electromagnetic counterpart observed across spectrum
        - Confirmed that GW and light travel at same speed
        - Source of heavy elements (gold, platinum, uranium)
        
        **Current Status:**
        - 90+ confirmed detections
        - Regular observations of black hole mergers
        - Population studies revealing formation channels
        - Tests of general relativity in strong field regime
        """)
    
    with st.expander("🧮 Mathematical Framework"):
        st.markdown("""
        ### 📐 Linearized General Relativity
        
        **Einstein Field Equations (Linearized):**
        □hμν = -(16πG/c⁴) × Tμν
        
        Where □ is the d'Alembertian operator: ∂²/∂t² - ∇²
        
        **Wave Solutions:**
        In the TT (transverse-traceless) gauge:
        - hμν = h₊(t-z/c) × e₊μν + h×(t-z/c) × e×μν
        - e₊ and e× are polarization tensors
        - Waves propagate in z-direction
        
        ### 🌊 Waveform Modeling
        
        #### **Post-Newtonian Expansion**
        For inspiral phase, expand in v/c where v is orbital velocity:
        
        **Frequency Evolution:**
        df/dt = (96π/5) × (πGMf/c³)⁵/³
        
        **Phase Evolution:**
        Ψ(f) = 2πft₀ + φ₀ + Σₙ ψₙ(πMf)⁽ⁿ⁻⁵⁾/³
        
        #### **Numerical Relativity**
        - Solve full Einstein equations on supercomputers
        - Required for merger and ringdown phases
        - Provides templates for data analysis
        - Breakthrough achievement of 2005
        
        ### 📊 Data Analysis
        
        #### **Matched Filtering**
        Signal-to-noise ratio: **ρ = 4Re∫[s̃(f)h̃*(f)/Sₙ(f)]df**
        
        - s̃(f): Fourier transform of detector data
        - h̃(f): Template waveform
        - Sₙ(f): Noise power spectral density
        - Optimal filter for known signal shapes
        
        #### **Parameter Estimation**
        **Bayes' Theorem:**
        P(θ|data) ∝ P(data|θ) × P(θ)
        
        - θ: source parameters (masses, spins, distance, etc.)
        - P(data|θ): Likelihood function
        - P(θ): Prior probability distribution
        - Use Markov Chain Monte Carlo sampling
        
        ### 🎯 Astrophysical Applications
        
        #### **Standard Sirens**
        - Gravitational waves measure luminosity distance
        - Electromagnetic counterpart gives redshift
        - Independent measure of Hubble constant
        - H₀ = 70 ± 8 km/s/Mpc (from GW170817)
        
        #### **Black Hole Population Studies**
        - Mass distribution reveals formation channels
        - Spin measurements test formation scenarios
        - Merger rates constrain stellar evolution
        - Most massive black holes challenge theory
        
        #### **Tests of General Relativity**
        - Propagation speed: v = c ± 2 × 10⁻¹⁵
        - Polarization: consistent with tensor modes
        - Dispersion: no frequency-dependent delays
        - Strong-field regime tests pass all checks
        """)
    
    with st.expander("🔮 Future Prospects"):
        st.markdown("""
        ### 🚀 Next-Generation Detectors
        
        #### **Cosmic Explorer**
        - 40 km arm length (10× longer than LIGO)
        - 10× better sensitivity across all frequencies
        - Could detect stellar-mass binaries across universe
        - Underground construction reduces noise
        
        #### **Einstein Telescope**
        - Underground triangular detector in Europe
        - Three 10 km arms per vertex
        - Cryogenic mirrors reduce thermal noise
        - Broader frequency band (3 Hz - 10 kHz)
        
        #### **LISA Constellation**
        - Three spacecraft in heliocentric orbit
        - 2.5 million km baseline
        - Millihertz band (10⁻⁴ - 1 Hz)
        - Years-long observations of single sources
        
        ### 🌌 Scientific Goals
        
        #### **Precision Cosmology**
        - Dark energy equation of state
        - Modified gravity theories
        - Primordial gravitational waves from inflation
        - Phase transitions in early universe
        
        #### **Fundamental Physics**
        - Black hole no-hair theorem tests
        - Quantum gravity phenomenology
        - Extra dimension signatures
        - Dark matter interactions
        
        #### **Multi-Messenger Astronomy**
        - Joint GW + electromagnetic observations
        - GW + neutrino coincidences
        - Complete picture of cosmic catastrophes
        - Population III star formation
        
        ### 🧩 Open Questions
        
        **Formation Mysteries:**
        - How do supermassive black holes form so early?
        - What creates the "pair-instability gap"?
        - Are there intermediate-mass black holes?
        - How common are primordial black holes?
        
        **Theoretical Puzzles:**
        - Do black holes have hair?
        - What happens at quantum gravity scales?
        - Are there extra dimensions?
        - How does inflation end?
        
        **Observational Challenges:**
        - Detecting continuous waves from pulsars        - Resolving stochastic background components
        - Finding electromagnetic counterparts
        - Measuring individual pulsar distances
        """)
    
    st.markdown("---")
    
    # Interactive simulation
    st.markdown("### 🌊 Binary Merger Simulation")
    
    if st.button("🔄 Generate Gravitational Wave Simulation"):
        with st.spinner("Computing spacetime dynamics..."):
            simulation_data = st.session_state.simulations.load_simulation("gravitational_waves")
            
            # Create wave visualization
            fig = st.session_state.visualizer.create_trajectory_animation(
                simulation_data,
                animation_speed=100,
                show_trails=True
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    # Educational Game Section
    st.markdown("---")
    st.markdown("### 🎮 Interactive Physics Game: Gravitational Wave Detective")
    
    # Initialize game runner if not exists
    if 'game_runner' not in st.session_state:
        st.session_state.game_runner = StreamlitGameRunner()
    
    with st.container():
        st.markdown("""
        **🌊 Game Objective:** Detect gravitational waves from merging black holes by operating multiple detectors!
        Learn about coincidence detection, signal analysis, and the incredible precision of LIGO.
        """)
        
        # Game interface
        gw_col1, gw_col2 = st.columns([1, 2])
        
        with gw_col1:
            st.markdown("**📡 Detector Control Center**")
            
            # Detector controls
            st.markdown("**LIGO Detector Network:**")
            detectors = ["LIGO Hanford", "LIGO Livingston", "Virgo (Europe)"]
            
            for i, detector in enumerate(detectors):
                detector_key = f"detector_{i}"
                
                # Detector status
                if detector_key not in st.session_state:
                    st.session_state[detector_key] = False
                
                status = "🟢 ACTIVE" if st.session_state[detector_key] else "🔴 INACTIVE"
                st.markdown(f"**{detector}:** {status}")
                
                if st.button(f"Toggle {detector}", key=f"toggle_detector_{i}"):
                    st.session_state[detector_key] = not st.session_state[detector_key]
                    
                    action = {'activate_detector': i}
                    game_state = st.session_state.game_runner.run_game_step('gravitational_waves', action)
                    if game_state:
                        st.session_state.gw_game_state = game_state
            
            # Detection requirements
            active_count = sum([st.session_state.get(f"detector_{i}", False) for i in range(3)])
            
            if active_count >= 2:
                st.success(f"✅ {active_count}/3 detectors active - Ready for coincidence detection!")
            else:
                st.warning(f"⚠️ {active_count}/3 detectors active - Need at least 2 for detection")
            
            # Game instructions
            st.markdown("""
            **📋 Instructions:**
            1. Activate at least 2 detectors
            2. Watch for coincident signals
            3. Look for chirp patterns in the data
            4. Score points for successful detections!
            
            **🧠 Physics Learning:**
            - Understand gravitational wave sources
            - Learn about detector sensitivity
            - Experience coincidence detection
            - Analyze real-world detection methods
            """)
            
            # Control buttons
            col_update, col_reset = st.columns(2)
            with col_update:
                if st.button("📊 Update Detection", key="gw_update_btn"):
                    # Run game step to update detectors
                    game_state = st.session_state.game_runner.run_game_step('gravitational_waves', {})
                    if game_state:
                        st.session_state.gw_game_state = game_state
            
            with col_reset:
                if st.button("🔄 Reset Network", key="gw_reset_btn"):
                    st.session_state.game_runner.games['gravitational_waves'] = st.session_state.game_runner.games['gravitational_waves'].__class__()
                    if 'gw_game_state' in st.session_state:
                        del st.session_state.gw_game_state
                    # Reset detector states
                    for i in range(3):
                        st.session_state[f"detector_{i}"] = False
        
        with gw_col2:
            st.markdown("**🌊 Gravitational Wave Detection Interface**")
            
            # Display game state
            if 'gw_game_state' in st.session_state:
                st.image(f"data:image/png;base64,{st.session_state.gw_game_state}", 
                        caption="Gravitational Wave Detective - Find the Signals!", 
                        use_column_width=True)
            else:
                # Show example gravitational wave
                st.markdown("**📈 Example: Binary Merger Gravitational Wave**")
                
                # Generate example chirp signal
                t = np.linspace(0, 2, 1000)
                f0 = 50  # Starting frequency
                chirp_rate = 1.5
                frequency = f0 * (chirp_rate ** t)
                
                # Amplitude envelope
                amplitude = np.exp(-t/2)
                
                # Generate strain signal
                strain = amplitude * np.sin(2 * np.pi * frequency * t)
                
                # Create visualization
                gw_fig = go.Figure()
                
                # Main chirp signal
                gw_fig.add_trace(go.Scatter(
                    x=t, y=strain,
                    mode='lines',
                    name='Gravitational Wave Strain',
                    line=dict(color='blue', width=2)
                ))
                
                # Frequency evolution
                gw_fig.add_trace(go.Scatter(
                    x=t, y=frequency/100,  # Scale for visibility
                    mode='lines',
                    name='Frequency Evolution (×100)',
                    line=dict(color='red', width=2),
                    yaxis='y2'
                ))
                
                gw_fig.update_layout(
                    title="Binary Black Hole Merger: Gravitational Wave Chirp",
                    xaxis_title="Time (seconds)",
                    yaxis_title="Strain (dimensionless)",
                    yaxis2=dict(
                        title="Frequency (Hz)",
                        overlaying='y',
                        side='right'
                    ),
                    showlegend=True,
                    height=400
                )
                
                st.plotly_chart(gw_fig, use_container_width=True)
                
                st.info("🎮 Activate detectors and click 'Update Detection' to start hunting for gravitational waves!")
        
        # Gravitational wave science learning section
        with st.expander("🎓 Gravitational Wave Science Learning"):
            st.markdown("""
            **🌊 What Are Gravitational Waves?**
            
            **Einstein's Prediction (1915):**
            - Ripples in spacetime fabric itself
            - Generated by accelerating massive objects
            - Travel at the speed of light
            - Extremely weak by the time they reach Earth
            
            **Detection Principle:**
            - Laser interferometry (LIGO/Virgo)
            - Measures tiny changes in arm length
            - Sensitivity: 1/10,000th the width of a proton!
            - Requires multiple detectors for confirmation
            
            **Binary Merger Signatures:**
            - Inspiral phase: frequency increases slowly
            - Merger phase: rapid frequency sweep
            - Ringdown: exponential decay
            - Characteristic "chirp" pattern
            
            **Famous Discoveries:**
            - GW150914 (2015): First direct detection
            - Binary black hole: 36 + 29 → 62 solar masses
            - GW170817 (2017): Neutron star merger
            - Multi-messenger astronomy breakthrough
            
            **🎯 Detection Strategy:**
            - Coincidence requirement: 2+ detectors
            - Signal matching: template-based search
            - Statistical significance: > 5σ confidence
            - Follow-up observations across electromagnetic spectrum
            
            **Future Prospects:**
            - Space-based detectors (LISA)
            - Next-generation ground detectors
            - Neutron star physics insights
            - Cosmology and dark energy studies
            """)
    
    # Quiz section
    show_quiz_section("gravitational_waves")


def show_progress_dashboard():
    """Show user progress and analytics"""
    st.markdown("## 📊 Your Learning Progress")
    
    # Get comprehensive user insights
    user_insights = st.session_state.content_manager.get_user_insights(st.session_state.user_id)
    
    if "overall_progress" in user_insights:
        progress = user_insights["overall_progress"]
        
        # Overall metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Lessons Completed", progress["completed_lessons"])
        with col2:
            st.metric("Average Mastery", f"{progress['avg_mastery']:.1%}")
        with col3:
            st.metric("Time Spent", f"{progress['total_time_hours']:.1f} hrs")
        with col4:
            completion_rate = progress.get("completion_rate", 0)
            st.metric("Completion Rate", f"{completion_rate:.1%}")
        
        # Concept mastery
        if "concept_mastery" in user_insights:
            st.markdown("### 🧠 Concept Mastery")
            
            concept_mastery = user_insights["concept_mastery"]
            concepts_df = pd.DataFrame([
                {"Concept": concept.replace("_", " ").title(), "Mastery": mastery}
                for concept, mastery in concept_mastery.items()
            ])
            
            if not concepts_df.empty:
                fig = px.bar(
                    concepts_df, 
                    x="Concept", 
                    y="Mastery",
                    title="Physics Concept Mastery",
                    color="Mastery",
                    color_continuous_scale="viridis"
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # Quiz performance
        if "quiz_performance" in user_insights:
            st.markdown("### 🧪 Quiz Performance")
            quiz_perf = user_insights["quiz_performance"]
            
            if "overall_accuracy" in quiz_perf:
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Overall Accuracy", f"{quiz_perf['overall_accuracy']:.1%}")
                with col2:
                    st.metric("Total Attempts", quiz_perf.get("total_attempts", 0))
        
        # Learning streaks
        if "learning_streaks" in user_insights:
            st.markdown("### 🔥 Learning Streaks")
            streaks = user_insights["learning_streaks"]
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Current Streak", f"{streaks['current_streak']} days")
            with col2:
                st.metric("Longest Streak", f"{streaks['longest_streak']} days")
            with col3:
                st.metric("Total Active Days", streaks['total_active_days'])
        
        # Recommendations
        st.markdown("### 💡 Personalized Recommendations")
        for rec in user_insights.get("recommendations", []):
            st.markdown(f"• {rec}")
    
    else:
        st.info("Start learning to see your progress dashboard!")
        st.markdown("**Get started with:**")
        st.markdown("• Gravity Basics")
        st.markdown("• Take some quizzes")
        st.markdown("• Explore simulations")


def show_physics_sandbox():
    """Physics sandbox for experimentation"""
    st.markdown("## 🔬 Physics Sandbox")
    st.markdown("*Experiment with different physics scenarios and parameters*")
    
    # Add theoretical background
    with st.expander("📚 Understanding Computational Physics", expanded=False):
        st.markdown("""
        ### 🖥️ Numerical Methods in Physics
        
        **Why Simulations?**
        - Most physics problems don't have analytical solutions
        - Complex systems require computational approaches
        - Simulations allow exploration of extreme conditions
        - Visualization helps develop physical intuition
        
        ### 🧮 Key Numerical Methods
        
        #### **N-Body Simulations**
        **Direct N-Body:**
        - Calculate forces between all particle pairs
        - Computational cost: O(N²) operations
        - Exact for point masses
        - Limited to ~10⁴ particles
        
        **Tree Methods (Barnes-Hut):**
        - Group distant particles into larger masses
        - Computational cost: O(N log N)
        - Controllable accuracy parameter θ
        - Enables simulations with millions of particles
        
        **Particle Mesh (PM) Methods:**
        - Solve Poisson equation on grid
        - Fast Fourier Transforms for efficiency
        - Good for large-scale structure
        - Less accurate for close encounters
        
        #### **Time Integration Schemes**
        
        **Euler Method (1st order):**
        - xₙ₊₁ = xₙ + h×f(xₙ)
        - Simple but inaccurate
        - Energy not conserved
        - Only for demonstration
        
        **Runge-Kutta Methods (4th order):**
        - More accurate but computationally expensive
        - Good energy conservation
        - Standard for many applications
        - Adaptive step size possible
        
        **Symplectic Integrators:**
        - Preserve phase space volume
        - Excellent long-term stability
        - Ideal for planetary dynamics
        - Leapfrog and Verlet methods
        
        ### 🎯 Accuracy Considerations
        
        **Sources of Error:**
        - Finite precision arithmetic
        - Time step discretization
        - Spatial resolution limits
        - Approximations in physics
        
        **Convergence Testing:**
        - Reduce time step by factor of 2
        - Increase spatial resolution
        - Compare different methods
        - Check conservation laws
        
        ### 🌌 Applications in Astrophysics
        
        **Galaxy Formation:**
        - Dark matter halos collapse under gravity
        - Gas dynamics and star formation
        - Feedback from supernovae and black holes
        - Cosmological context with expansion
        
        **Planetary System Evolution:**
        - Migration of giant planets
        - Asteroid and comet dynamics
        - Tidal interactions with stars
        - Long-term stability analysis
        
        **Stellar Dynamics:**
        - Globular cluster evolution
        - Binary star interactions
        - Galactic center dynamics
        - Black hole interactions
        """)
    
    # Add tabbed interface for different experiment types
    tab1, tab2, tab3 = st.tabs(["🎯 Quick Experiments", "🔧 Custom Builder", "📊 Parameter Study"])
    
    with tab1:
        st.markdown("### ⚡ Quick Physics Experiments")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("� Earth-Moon System", use_container_width=True):
                params = {
                    "mass1": 5.97e24, "mass2": 7.34e22, "separation": 3.84e8,
                    "domain_size": 8e8, "time_span": 30*24*3600
                }
                run_quick_experiment("binary_orbit", params, "Earth-Moon orbital dynamics")
        
        with col2:
            if st.button("⭐ Alpha Centauri", use_container_width=True):
                params = {
                    "mass1": 2.2e30, "mass2": 1.8e30, "separation": 3.5e12,
                    "domain_size": 7e12, "time_span": 80*365.25*24*3600
                }
                run_quick_experiment("binary_orbit", params, "Binary star system")
                
        with col3:
            if st.button("🕳️ Black Hole", use_container_width=True):
                params = {
                    "black_hole_mass": 10*1.989e30, "domain_size": 1e12, "time_span": 1000
                }
                run_quick_experiment("black_hole_accretion", params, "Black hole gravitational field")
        
        # Show results if available
        if 'quick_experiment_result' in st.session_state:
            st.markdown("### 📈 Experiment Results")
            result = st.session_state.quick_experiment_result
            
            col_vis1, col_vis2 = st.columns(2)
            with col_vis1:
                st.plotly_chart(result['visualization'], use_container_width=True)
            with col_vis2:
                st.markdown("**Physics Insights:**")
                for insight in result['insights']:
                    st.markdown(f"• {insight}")
    
    with tab2:
        st.markdown("### 🛠️ Custom System Builder")
        
        # Scenario selection
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("#### 🎛️ System Configuration")
            
            scenario = st.selectbox(
                "Base Scenario:",
                ["binary_orbit", "planetary_system", "black_hole_accretion", "gravitational_waves"],
                help="Choose the base physics scenario to customize"
            )
            
            physics_type = st.selectbox(
                "Physics Model:",
                ["newtonian", "relativistic", "orbital"],
                help="Select the physics model to use"
            )
            
            # Dynamic parameter controls based on scenario
            if scenario == "binary_orbit":
                st.markdown("**Binary System Parameters:**")
                mass1 = st.number_input("Primary Mass (kg)", value=1.989e30, format="%.2e")
                mass2 = st.number_input("Secondary Mass (kg)", value=5.97e24, format="%.2e")
                separation = st.number_input("Separation (m)", value=1.5e11, format="%.2e")
                
                custom_params = {
                    "mass1": mass1, "mass2": mass2, "separation": separation,
                    "domain_size": separation * 3, "time_span": 365.25*24*3600
                }
                
            elif scenario == "black_hole_accretion":
                st.markdown("**Black Hole Parameters:**")
                bh_mass = st.number_input("Black Hole Mass (solar masses)", value=10.0, min_value=1.0)
                domain_size = st.number_input("Domain Size (m)", value=1e12, format="%.2e")
                
                custom_params = {
                    "black_hole_mass": bh_mass * 1.989e30,
                    "domain_size": domain_size,
                    "time_span": 1000
                }
            
            # Simulation settings
            st.markdown("**Simulation Settings:**")
            grid_res = st.slider("Grid Resolution", 32, 128, 64, step=16)
            time_resolution = st.slider("Time Steps", 100, 2000, 1000, step=100)
            
            # Visualization options
            st.markdown("**Visualization Options:**")
            viz_3d = st.checkbox("3D Field Visualization", True)
            viz_traj = st.checkbox("Trajectory Animation", True)
            viz_cross = st.checkbox("Cross-sectional Views", False)
            viz_dashboard = st.checkbox("Interactive Dashboard", True)
            
            if st.button("🚀 Build & Run Custom System", type="primary"):
                build_custom_system(scenario, custom_params, grid_res, {
                    "3d": viz_3d, "trajectories": viz_traj, 
                    "cross_sections": viz_cross, "dashboard": viz_dashboard
                })
        
        with col2:
            if 'custom_system_result' in st.session_state:
                result = st.session_state.custom_system_result
                st.markdown("#### 🎬 Custom System Visualization")
                
                # Display selected visualizations
                if result.get('viz_3d'):
                    st.plotly_chart(result['viz_3d'], use_container_width=True)
                
                if result.get('viz_dashboard'):
                    dashboard = result['viz_dashboard']
                    if "main_3d" in dashboard:
                        st.plotly_chart(dashboard["main_3d"], use_container_width=True)
                  # Show system properties
                st.markdown("**System Properties:**")
                props = result.get('properties', {})
                for key, value in props.items():
                    st.metric(key.replace('_', ' ').title(), value)
            else:
                st.info("👈 Configure and run a custom system to see visualizations here")
    
    with tab3:
        st.markdown("### 📊 Parameter Study Mode")
        st.markdown("*Study how changing parameters affects the physics*")
        
        study_param = st.selectbox(
            "Parameter to Study:",
            ["mass_ratio", "separation", "orbital_period", "black_hole_mass"],
            help="Choose which parameter to vary in the study"
        )
        
        param_range = st.slider(
            f"Parameter Range (multiplier)",
            0.1, 10.0, (0.5, 2.0),
            help="Range of parameter values to study"
        )
        
        n_samples = st.slider("Number of Samples", 5, 20, 10)
        
        if st.button("📈 Run Parameter Study"):
            run_parameter_study(study_param, param_range, n_samples)


# Helper functions for the enhanced physics sandbox
def run_quick_experiment(scenario_id, params, description):
    """Run a quick physics experiment with predefined parameters"""
    try:
        with st.spinner(f"Running {description}..."):
            # Override scenario parameters
            original_params = st.session_state.simulations.default_scenarios[scenario_id]["parameters"]
            st.session_state.simulations.default_scenarios[scenario_id]["parameters"] = params
            
            # Generate simulation
            simulation_data = st.session_state.simulations.load_simulation(scenario_id, (64, 64, 32))
            
            # Restore original parameters
            st.session_state.simulations.default_scenarios[scenario_id]["parameters"] = original_params
            
            # Create visualization
            fig = st.session_state.visualizer.create_3d_field_visualization(
                simulation_data, visualization_type="potential"
            )
            
            # Generate insights
            insights = generate_physics_insights(simulation_data, scenario_id)
            
            st.session_state.quick_experiment_result = {
                'visualization': fig,
                'insights': insights,
                'data': simulation_data
            }
            
            st.success(f"✅ {description} complete!")
            
    except Exception as e:
        st.error(f"Experiment failed: {e}")


def build_custom_system(scenario, params, grid_res, viz_options):
    """Build and run a custom gravitational system"""
    try:
        with st.spinner("Building custom system..."):
            # Override scenario parameters
            original_params = st.session_state.simulations.default_scenarios[scenario]["parameters"]
            st.session_state.simulations.default_scenarios[scenario]["parameters"] = params
            
            # Generate simulation
            simulation_data = st.session_state.simulations.load_simulation(
                scenario, (grid_res, grid_res, grid_res//2)
            )
            
            # Restore original parameters
            st.session_state.simulations.default_scenarios[scenario]["parameters"] = original_params
            
            # Create visualizations based on options
            result = {'properties': {}}
            
            if viz_options.get('3d'):
                result['viz_3d'] = st.session_state.visualizer.create_3d_field_visualization(
                    simulation_data, visualization_type="potential"
                )
            
            if viz_options.get('dashboard'):
                result['viz_dashboard'] = st.session_state.visualizer.create_interactive_dashboard(simulation_data)
            
            # Calculate system properties
            if 'orbital_period' in simulation_data:
                result['properties']['Orbital Period'] = f"{simulation_data['orbital_period']/(24*3600):.1f} days"
            
            if 'masses' in simulation_data:
                masses = simulation_data['masses']
                if 'm1' in masses and 'm2' in masses:
                    result['properties']['Mass Ratio'] = f"{masses['m1']/masses['m2']:.2f}"
            
            st.session_state.custom_system_result = result
            st.success("Custom system built successfully!")
            
    except Exception as e:
        st.error(f"Failed to build custom system: {e}")


def run_parameter_study(param_name, param_range, n_samples):
    """Run a parameter study"""
    try:
        with st.spinner("Running parameter study..."):
            # Create parameter values
            param_values = np.linspace(param_range[0], param_range[1], n_samples)
            results = []
            
            # Run simulations for each parameter value
            for value in param_values:
                # This is a simplified version - in practice you'd vary the specific parameter
                simulation_data = st.session_state.simulations.load_simulation("binary_orbit", (32, 32, 16))
                results.append({
                    'param_value': value,
                    'orbital_period': simulation_data.get('orbital_period', 0),
                    'simulation_data': simulation_data
                })
            
            # Create parameter study visualization
            import plotly.express as px
            df = pd.DataFrame([{
                'Parameter Value': r['param_value'],
                'Orbital Period (days)': r['orbital_period']/(24*3600) if r['orbital_period'] else 0
            } for r in results])
            
            fig = px.line(df, x='Parameter Value', y='Orbital Period (days)', 
                         title=f'Parameter Study: {param_name.replace("_", " ").title()}')
            
            st.plotly_chart(fig, use_container_width=True)
            st.success("Parameter study complete!")
            
    except Exception as e:
        st.error(f"Parameter study failed: {e}")


def generate_physics_insights(simulation_data, scenario_id):
    """Generate physics insights from simulation data"""
    insights = []
    
    if scenario_id == "binary_orbit" and 'orbital_period' in simulation_data:
        period_days = simulation_data['orbital_period'] / (24 * 3600)
        insights.append(f"Orbital period: {period_days:.1f} days")
        
        if 'masses' in simulation_data:
            masses = simulation_data['masses']
            mass_ratio = masses['m1'] / masses['m2']
            insights.append(f"Mass ratio: {mass_ratio:.2f}")
            
        insights.append("System follows Kepler's laws of planetary motion")
    
    elif scenario_id == "black_hole_accretion" and 'schwarzschild_radius' in simulation_data:
        rs = simulation_data['schwarzschild_radius']
        insights.append(f"Schwarzschild radius: {rs/1000:.1f} km")
        insights.append("Strong gravitational lensing effects near event horizon")
        insights.append("Time dilation becomes significant near the black hole")
    
    return insights


def show_quiz_section(topic: str):
    """Show quiz section for a topic"""
    st.markdown("### 🧪 Test Your Knowledge")
    
    if st.button(f"📝 Take {topic.replace('_', ' ').title()} Quiz", key=f"quiz_{topic}"):
        # Generate adaptive quiz
        quiz_questions = st.session_state.content_manager.generate_quiz(topic, num_questions=3)
        
        if quiz_questions:
            st.session_state.current_quiz = quiz_questions
            st.session_state.quiz_topic = topic
            st.session_state.quiz_answers = {}
            st.session_state.quiz_started = True
    
    # Display quiz if started
    if st.session_state.get('quiz_started', False) and 'current_quiz' in st.session_state:
        quiz_questions = st.session_state.current_quiz
        
        for i, question in enumerate(quiz_questions):
            st.markdown(f"**Question {i+1}:** {question.question}")
            
            if question.question_type == "multiple_choice":
                answer = st.radio(
                    "Select your answer:",
                    question.options,
                    key=f"q_{i}_{question.id}"
                )
                st.session_state.quiz_answers[question.id] = answer
            elif question.question_type == "numerical":
                answer = st.number_input(
                    "Enter your numerical answer:",
                    key=f"q_{i}_{question.id}"
                )
                st.session_state.quiz_answers[question.id] = str(answer)
        
        if st.button("Submit Quiz"):
            # Score quiz
            correct_answers = 0
            total_questions = len(quiz_questions)
            
            for question in quiz_questions:
                user_answer = st.session_state.quiz_answers.get(question.id, "")
                is_correct = st.session_state.content_manager.record_quiz_attempt(
                    st.session_state.user_id,
                    st.session_state.quiz_topic,
                    question.id,
                    user_answer,
                    question.correct_answer,
                    0  # response time not tracked yet
                )
                
                if is_correct:
                    correct_answers += 1
                    st.success(f"✅ Question {quiz_questions.index(question)+1}: Correct!")
                else:
                    st.error(f"❌ Question {quiz_questions.index(question)+1}: Incorrect")
                    st.info(f"**Explanation:** {question.explanation}")
            
            # Show results
            score = correct_answers / total_questions
            st.markdown(f"### 📊 Quiz Results: {correct_answers}/{total_questions} ({score:.1%})")
            
            if score >= 0.8:
                st.balloons()
                st.success("Excellent work! You've mastered this topic!")
            elif score >= 0.6:
                st.success("Good job! Review the explanations to improve.")
            else:
                st.warning("Keep studying! Review the material and try again.")
            
            # Update progress
            st.session_state.content_manager.update_user_progress(
                st.session_state.user_id,
                st.session_state.quiz_topic,
                {
                    "quiz_scores": [score],
                    "completion_percentage": min(100, score * 100),
                    "mastery_level": score
                }
            )
            
            # Reset quiz
            st.session_state.quiz_started = False


if __name__ == "__main__":
    main()
