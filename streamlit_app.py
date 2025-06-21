"""
🌌 Gravity Yonder Over - Educational Physics Platform
Built with Streamlit + NVIDIA Modulus + cuDF + Plotly

A comprehensive educational web application for interactive gravity and physics simulations
using pre-generated NVIDIA Modulus computations served as visual plots.
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import time
import uuid
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# Import our custom modules
try:
    from src.modulus_physics import ModulusGravityEngine
    from src.cudf_processor import CuDFDataProcessor
    from src.simulation_datasets import PreGeneratedSimulations
    from src.plotly_visualizer import PhysicsVisualizer
    from src.educational_content import EducationalContentManager
    
    # Try to import new NVIDIA tools, but don't fail if they're not available
    try:
        from backend.simulations.cuquantum_engine import CuQuantumGravityEngine
    except ImportError:
        CuQuantumGravityEngine = None
    
    try:
        from backend.simulations.morpheus_analyzer import MorpheusPhysicsAnalyzer
    except ImportError:
        MorpheusPhysicsAnalyzer = None
    
    try:
        from backend.simulations.physicsnemo_engine import PhysicsNeMoEngine
    except ImportError:
        PhysicsNeMoEngine = None
    
    try:
        from nvidia_tools_showcase import show_nvidia_tools_showcase
    except ImportError:
        show_nvidia_tools_showcase = None
        
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
            
            # Initialize new NVIDIA tools (if available)
            if CuQuantumGravityEngine is not None:
                st.session_state.cuquantum_engine = CuQuantumGravityEngine()
            else:
                st.session_state.cuquantum_engine = None
                
            if MorpheusPhysicsAnalyzer is not None:
                st.session_state.morpheus_analyzer = MorpheusPhysicsAnalyzer()
            else:
                st.session_state.morpheus_analyzer = None
                
            if PhysicsNeMoEngine is not None:
                st.session_state.physicsnemo_engine = PhysicsNeMoEngine()
            else:
                st.session_state.physicsnemo_engine = None
            
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
        <h3>Interactive Physics Education with NVIDIA Modulus</h3>
    </div>
    """, unsafe_allow_html=True)    # Sidebar navigation
    st.sidebar.markdown("## 🎮 Learning Adventures")
    
    # Build navigation options based on available features
    nav_options = [
        "🏠 Home",
        "📚 Learning Path", 
        "🍎 Gravity Basics",
        "🚀 Orbital Mechanics", 
        "⚫ Black Hole Physics",
        "🌌 Gravitational Waves",
        "📊 Progress Dashboard",
        "🔬 Physics Sandbox",
        "⚡ Relativity Missions",
        "🌍 Lagrange Point Missions"
    ]
    
    # Add NVIDIA showcase if available
    if show_nvidia_tools_showcase is not None:
        nav_options.append("🚀 NVIDIA Tools Showcase")
    
    game_selection = st.sidebar.selectbox(
        "Choose your physics adventure:",
        nav_options
    )
    
    # System status in sidebar
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🔧 System Status")
      # Check NVIDIA Modulus availability
    modulus_status = st.session_state.modulus_engine.get_status()
    if modulus_status['available']:
        st.sidebar.success("✅ NVIDIA Modulus: Active")
        st.sidebar.info(f"GPU: {modulus_status['gpu_count']} devices")
    else:
        st.sidebar.warning("⚠️ NVIDIA Modulus: CPU Fallback")
    
    # Check cuDF availability
    cudf_status = st.session_state.cudf_processor.get_status()
    if cudf_status['available']:
        st.sidebar.success("✅ cuDF: GPU Acceleration")
    else:
        st.sidebar.info("📊 cuDF: Pandas Fallback")
    
    # Check optional NVIDIA tools
    if st.session_state.cuquantum_engine is not None:
        cuquantum_status = st.session_state.cuquantum_engine.get_status()
        if cuquantum_status['available']:
            st.sidebar.success("✅ cuQuantum: Quantum Gravity")
        else:
            st.sidebar.info("⚛️ cuQuantum: CPU Fallback")    else:
        st.sidebar.info("⚛️ cuQuantum: Not Available")
    
    if st.session_state.morpheus_analyzer is not None:
        morpheus_status = st.session_state.morpheus_analyzer.get_status()
        if morpheus_status['available']:
            st.sidebar.success("✅ Morpheus: Real-time Analysis")
        else:
            st.sidebar.info("🔍 Morpheus: Basic Mode")
    else:
        st.sidebar.info("🔍 Morpheus: Not Available")
    
    if st.session_state.physicsnemo_engine is not None:
        physicsnemo_status = st.session_state.physicsnemo_engine.get_status()
        if physicsnemo_status['available']:
            st.sidebar.success("✅ PhysicsNeMo: AI Models")
        else:
            st.sidebar.info("🧠 PhysicsNeMo: PyTorch Mode")
    else:
        st.sidebar.info("🧠 PhysicsNeMo: Not Available")
    
    # Performance metrics
    st.sidebar.markdown("### 📊 Performance")
    available_scenarios = st.session_state.simulations.get_scenario_list()
    col1, col2 = st.sidebar.columns(2)
    with col1:
        st.metric("Scenarios", len(available_scenarios))
    with col2:
        precomputed_count = sum(1 for s in available_scenarios if s['status'] == 'pre-computed')
        st.metric("Pre-computed", precomputed_count)    # Route to appropriate page
    if game_selection == "🏠 Home":
        show_home_page()
    elif "Learning Path" in game_selection:
        show_learning_path()
    elif "Gravity Basics" in game_selection:
        show_gravity_basics()
    elif "Orbital Mechanics" in game_selection:
        show_orbital_mechanics()
    elif "Black Hole" in game_selection:
        show_black_hole_physics()
    elif "Gravitational Waves" in game_selection:
        show_gravitational_waves()
    elif "Progress Dashboard" in game_selection:
        show_progress_dashboard()
    elif "Physics Sandbox" in game_selection:
        show_physics_sandbox()
    elif "Relativity Missions" in game_selection:
        show_relativity_missions()
    elif "Lagrange Point Missions" in game_selection:
        show_lagrange_missions()
    elif "NVIDIA Tools Showcase" in game_selection:
        if show_nvidia_tools_showcase is not None:
            show_nvidia_tools_showcase()
        else:
            st.error("NVIDIA Tools Showcase not available - some dependencies missing")
            st.info("The platform works with CPU fallbacks, but the full showcase requires GPU libraries.")


def show_home_page():
    """Display the home page with overview"""
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("## 🎯 Welcome to Physics Education")
        st.markdown("""
        **Gravity Yonder Over** is an advanced educational platform that uses **NVIDIA Modulus** 
        to solve complex physics PDEs and serves pre-generated simulation results through 
        interactive **Plotly** visualizations.
        
        ### 🔬 Physics Concepts Covered:
        - **Classical Mechanics**: Gravity, motion, and forces
        - **Orbital Mechanics**: Planetary motion and escape velocity
        - **Relativistic Physics**: Black holes and spacetime curvature
        - **Gravitational Waves**: Ripples in spacetime
        - **Binary Systems**: Multi-body gravitational interactions
        - **Educational Insights**: Interactive learning with real physics
        """)
        
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
                st.success(f"Starting lesson: {objective['title']}")


def show_gravity_basics():
    """Gravity basics simulation and learning"""
    st.markdown("## 🍎 Gravity Basics")
    
    # Educational content
    with st.expander("📖 Learn: Newton's Law of Universal Gravitation"):
        st.markdown("""
        **Newton's Law of Universal Gravitation:**
        
        Every particle attracts every other particle with a force proportional to the product 
        of their masses and inversely proportional to the square of the distance between them.
        
        **Formula:** F = G·m₁·m₂/r²
        
        Where:
        - F = gravitational force
        - G = gravitational constant (6.674 × 10⁻¹¹ m³/kg·s²)
        - m₁, m₂ = masses of the objects
        - r = distance between centers of masses
        """)
    
    # Interactive simulation
    st.markdown("### 🔬 Interactive Simulation")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("**Simulation Parameters:**")
        
        # Load and run simulation
        if st.button("🚀 Run Binary Orbit Simulation"):
            with st.spinner("Generating physics simulation..."):
                simulation_data = st.session_state.simulations.load_simulation("binary_orbit")
                
                # Create visualization
                fig = st.session_state.visualizer.create_3d_field_visualization(
                    simulation_data, 
                    visualization_type="potential"
                )
                
                st.session_state.current_simulation = simulation_data
                st.session_state.current_figure = fig
    
    with col2:
        if 'current_figure' in st.session_state:
            st.plotly_chart(st.session_state.current_figure, use_container_width=True)
        else:
            st.info("Click 'Run Simulation' to see the gravitational field visualization")
    
    # Animation section
    if 'current_simulation' in st.session_state:
        st.markdown("### 🎬 Trajectory Animation")
        
        if st.button("▶️ Show Orbital Animation"):
            with st.spinner("Creating animation..."):
                animation_fig = st.session_state.visualizer.create_trajectory_animation(
                    st.session_state.current_simulation
                )
                st.plotly_chart(animation_fig, use_container_width=True)
    
    # Quiz section
    show_quiz_section("gravity_basics")


def show_orbital_mechanics():
    """Orbital mechanics simulation and learning"""
    st.markdown("## 🚀 Orbital Mechanics")
    
    # Educational content
    with st.expander("📖 Learn: Kepler's Laws of Planetary Motion"):
        st.markdown("""
        **Kepler's Three Laws:**
        
        1. **First Law (Elliptical Orbits):** Planets orbit in ellipses with the Sun at one focus
        2. **Second Law (Equal Areas):** A line from planet to Sun sweeps equal areas in equal times
        3. **Third Law (Harmonic Law):** T² ∝ r³ (orbital period squared is proportional to semi-major axis cubed)
        
        **Applications:**
        - Satellite orbital calculations
        - Planetary motion prediction
        - Space mission planning
        """)
    
    # Interactive simulation
    st.markdown("### 🪐 Solar System Simulation")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        scenario_type = st.selectbox(
            "Choose simulation:",
            ["planetary_system", "binary_orbit"]
        )
        
        grid_resolution = st.slider("Grid Resolution", 32, 128, 64, step=16)
        
        if st.button("🌍 Run Orbital Simulation"):
            with st.spinner("Computing orbital mechanics..."):
                simulation_data = st.session_state.simulations.load_simulation(
                    scenario_type, 
                    (grid_resolution, grid_resolution, grid_resolution//2)
                )
                
                # Create dashboard
                dashboard = st.session_state.visualizer.create_interactive_dashboard(simulation_data)
                st.session_state.current_dashboard = dashboard
    
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
    
    # Quiz section
    show_quiz_section("orbital_mechanics")


def show_black_hole_physics():
    """Black hole physics simulation and learning"""
    st.markdown("## ⚫ Black Hole Physics")
    
    # Educational content
    with st.expander("📖 Learn: General Relativity and Black Holes"):
        st.markdown("""
        **Einstein's General Relativity:**
        
        Gravity is not a force but the curvature of spacetime caused by mass and energy.
        
        **Key Concepts:**
        - **Event Horizon:** The boundary beyond which nothing can escape
        - **Schwarzschild Radius:** Rs = 2GM/c²
        - **Time Dilation:** Time slows down in strong gravitational fields
        - **Geodesics:** Particles follow the straightest paths in curved spacetime
        """)
    
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
    
    # Quiz section
    show_quiz_section("black_holes")


def show_gravitational_waves():
    """Gravitational waves simulation and learning"""
    st.markdown("## 🌊 Gravitational Waves")
    
    # Educational content
    with st.expander("📖 Learn: Ripples in Spacetime"):
        st.markdown("""
        **Gravitational Waves:**
        
        Accelerating masses create ripples in the fabric of spacetime that propagate 
        at the speed of light.
        
        **Sources:**
        - Binary black hole mergers
        - Binary neutron star mergers
        - Asymmetric supernovae
        - The Big Bang itself
        
        **Detection:** LIGO and Virgo interferometers can detect changes in distance 
        smaller than 1/10,000th the width of a proton!
        """)
    
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
    
    # Scenario selection
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### 🎛️ Simulation Controls")
        
        scenario = st.selectbox(
            "Choose Scenario:",
            ["binary_orbit", "planetary_system", "black_hole_accretion", "gravitational_waves"]
        )
        
        physics_type = st.selectbox(
            "Physics Type:",
            ["newtonian", "relativistic", "orbital"]
        )
        
        grid_res = st.slider("Grid Resolution", 32, 128, 64, step=16)
        
        visualization_options = st.multiselect(
            "Visualizations:",
            ["3D Field", "Trajectories", "Cross Sections", "Dashboard"],
            default=["3D Field"]
        )
        
        if st.button("🧪 Run Custom Simulation"):
            with st.spinner("Running custom physics simulation..."):
                # Load simulation
                simulation_data = st.session_state.simulations.load_simulation(
                    scenario, 
                    (grid_res, grid_res, grid_res//2)
                )
                
                st.session_state.sandbox_simulation = simulation_data
                st.session_state.sandbox_visualizations = visualization_options
                
                st.success("Simulation complete!")
    
    with col2:
        if 'sandbox_simulation' in st.session_state:
            simulation_data = st.session_state.sandbox_simulation
            viz_options = st.session_state.sandbox_visualizations
            
            # Create requested visualizations
            if "3D Field" in viz_options:
                st.markdown("#### 🌐 3D Field Visualization")
                fig = st.session_state.visualizer.create_3d_field_visualization(
                    simulation_data,
                    visualization_type="potential"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            if "Trajectories" in viz_options and any(key in simulation_data for key in ["trajectories", "planet_trajectories", "geodesics"]):
                st.markdown("#### 🎯 Trajectory Animation")
                fig = st.session_state.visualizer.create_trajectory_animation(simulation_data)
                st.plotly_chart(fig, use_container_width=True)
            
            if "Cross Sections" in viz_options:
                st.markdown("#### 📊 Cross-Section Analysis")
                dashboard = st.session_state.visualizer.create_interactive_dashboard(simulation_data)
                if "cross_sections" in dashboard:
                    st.plotly_chart(dashboard["cross_sections"], use_container_width=True)
            
            if "Dashboard" in viz_options:
                st.markdown("#### 📈 Complete Dashboard")
                dashboard = st.session_state.visualizer.create_interactive_dashboard(simulation_data)
                
                # Show multiple visualizations
                for key, fig in dashboard.items():
                    if key != "cross_sections":  # Already shown above
                        st.markdown(f"**{key.replace('_', ' ').title()}**")
                        st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Configure simulation parameters and run to see results")


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


def show_relativity_missions():
    """Enhanced special and general relativity missions"""
    st.markdown("## ⚡ Relativity Missions")
    
    # Educational content
    with st.expander("📖 Learn: Einstein's Relativity"):
        st.markdown("""
        **Special Relativity (1905):**
        - Time dilation: Moving clocks run slower
        - Length contraction: Objects shrink in direction of motion
        - Mass-energy equivalence: E = mc²
        
        **General Relativity (1915):**
        - Gravity is curved spacetime
        - Time runs slower in stronger gravity
        - Light bends around massive objects
        """)
    
    mission_type = st.selectbox(
        "Choose Relativity Mission:",
        ["GPS Satellite Correction", "Interstellar Travel", "Black Hole Orbit", "Gravitational Lensing"]
    )
    
    if mission_type == "GPS Satellite Correction":
        st.markdown("### 🛰️ GPS Relativity Mission")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            orbit_altitude = st.slider("Satellite Altitude (km)", 15000, 25000, 20200)
            mission_duration = st.slider("Mission Duration (days)", 1, 365, 30)
            
            # Calculate relativistic effects
            earth_radius = 6371  # km
            r = earth_radius + orbit_altitude
            
            # Special relativity: velocity effect
            v = np.sqrt(3.986e5 / r)  # orbital velocity
            gamma_sr = 1 / np.sqrt(1 - (v/299792.458)**2)
            time_loss_sr = (1 - 1/gamma_sr) * mission_duration * 24 * 3600 * 1e6  # microseconds
            
            # General relativity: gravitational effect
            time_gain_gr = 6.95e-10 * mission_duration * 24 * 3600 * 1e6  # microseconds
            
            net_effect = time_gain_gr - time_loss_sr
            position_error = abs(net_effect) * 0.3 / 1000  # meters (light travels 0.3m per microsecond)
            
            st.metric("Special Relativity Effect", f"-{time_loss_sr:.1f} μs")
            st.metric("General Relativity Effect", f"+{time_gain_gr:.1f} μs")
            st.metric("Net Time Drift", f"{net_effect:+.1f} μs")
            st.metric("Position Error", f"{position_error:.0f} m")
            
        with col2:
            # Create visualization
            fig = create_gps_relativity_viz(orbit_altitude, time_loss_sr, time_gain_gr)
            st.plotly_chart(fig, use_container_width=True)
            
        if abs(net_effect) > 1000:
            st.error("⚠️ GPS would be highly inaccurate without relativistic corrections!")
        else:
            st.success("✅ GPS accuracy maintained with proper corrections")
    
    elif mission_type == "Interstellar Travel":
        st.markdown("### 🚀 Relativistic Starship Mission")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            target_star = st.selectbox("Destination:", 
                ["Proxima Centauri (4.24 ly)", "Wolf 359 (7.86 ly)", "Sirius (8.6 ly)"])
            velocity_fraction = st.slider("Velocity (fraction of c)", 0.1, 0.99, 0.8)
            
            distances = {"Proxima Centauri (4.24 ly)": 4.24, 
                        "Wolf 359 (7.86 ly)": 7.86, 
                        "Sirius (8.6 ly)": 8.6}
            distance = distances[target_star]
            
            # Calculate relativistic effects
            gamma = 1 / np.sqrt(1 - velocity_fraction**2)
            earth_time = distance / velocity_fraction  # years
            ship_time = earth_time / gamma
            
            st.metric("Earth Time", f"{earth_time:.1f} years")
            st.metric("Ship Time", f"{ship_time:.1f} years")
            st.metric("Time Dilation Factor", f"γ = {gamma:.2f}")
            st.metric("Age Difference", f"{earth_time - ship_time:.1f} years")
            
        with col2:
            fig = create_interstellar_travel_viz(velocity_fraction, earth_time, ship_time)
            st.plotly_chart(fig, use_container_width=True)

def show_lagrange_missions():
    """Enhanced Lagrange point missions and simulations"""
    st.markdown("## 🌍 Lagrange Point Missions")
    
    # Educational content
    with st.expander("📖 Learn: Lagrange Points"):
        st.markdown("""
        **Lagrange Points** are special locations where gravitational forces balance:
        
        - **L1**: Between Earth and Sun - Solar observatories
        - **L2**: Beyond Earth from Sun - Space telescopes (James Webb)
        - **L3**: Opposite Earth's orbit - Rarely used
        - **L4/L5**: 60° ahead/behind Earth - Stable points for colonies
        """)
    
    system = st.selectbox(
        "Choose Gravitational System:",
        ["Earth-Sun", "Earth-Moon", "Jupiter-Sun", "Saturn-Titan"]
    )
    
    mission_type = st.selectbox(
        "Mission Type:",
        ["Space Telescope Deployment", "Solar Observatory", "Asteroid Mining", "Space Colony"]
    )
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        if mission_type == "Space Telescope Deployment":
            st.markdown("### 🔭 Deploy Space Telescope at L2")
            
            telescope_mass = st.slider("Telescope Mass (kg)", 100, 10000, 6200)  # JWST mass
            fuel_budget = st.slider("Fuel Budget (m/s Δv)", 50, 500, 150)
            
            # Mission parameters
            l2_distance = 1.5e6  # km from Earth
            station_keeping = 2.0  # m/s per year
            mission_years = st.slider("Mission Duration (years)", 5, 20, 10)
            
            total_fuel_needed = station_keeping * mission_years
            
            st.metric("L2 Distance", f"{l2_distance/1000:.0f} km")
            st.metric("Station-keeping", f"{station_keeping:.1f} m/s/year")
            st.metric("Total Fuel Needed", f"{total_fuel_needed:.1f} m/s")
            
            if fuel_budget >= total_fuel_needed:
                st.success(f"✅ Mission feasible! Excess fuel: {fuel_budget - total_fuel_needed:.1f} m/s")
            else:
                st.error(f"❌ Insufficient fuel! Need {total_fuel_needed - fuel_budget:.1f} m/s more")
                
        elif mission_type == "Asteroid Mining":
            st.markdown("### ⛏️ L4/L5 Trojan Asteroid Mining")
            
            asteroid_count = st.slider("Target Asteroids", 1, 10, 3)
            mining_duration = st.slider("Mining Duration (years)", 5, 25, 15)
            
            # Estimate resources
            avg_asteroid_mass = 1e15  # kg (rough estimate)
            valuable_fraction = 0.001  # 0.1% valuable materials
            total_resources = asteroid_count * avg_asteroid_mass * valuable_fraction
            
            st.metric("Total Asteroid Mass", f"{asteroid_count * avg_asteroid_mass:.2e} kg")
            st.metric("Valuable Materials", f"{total_resources:.2e} kg")
            st.metric("Earth Distance", "150 million km")
            st.metric("Transport Time", "6-9 months")
    
    with col2:
        # Create Lagrange point visualization
        fig = create_lagrange_point_viz(system, mission_type)
        st.plotly_chart(fig, use_container_width=True)
        
        # Mission simulation
        if st.button("🚀 Run Mission Simulation"):
            with st.spinner("Simulating mission trajectory..."):
                time.sleep(2)  # Simulate computation
                success_rate = simulate_lagrange_mission(system, mission_type)
                
                if success_rate > 0.8:
                    st.success(f"Mission Success! Success rate: {success_rate:.1%}")
                elif success_rate > 0.5:
                    st.warning(f"Mission Challenging. Success rate: {success_rate:.1%}")
                else:
                    st.error(f"Mission High Risk. Success rate: {success_rate:.1%}")

def create_gps_relativity_viz(altitude, sr_effect, gr_effect):
    """Create GPS relativity effects visualization"""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Orbital Configuration', 'Time Effects', 'Position Error Growth', 'Correction Necessity'),
        specs=[[{'type': 'scatter'}, {'type': 'bar'}],
               [{'type': 'scatter'}, {'type': 'indicator'}]]
    )
    
    # Orbital visualization
    earth_radius = 6371
    angles = np.linspace(0, 2*np.pi, 100)
    earth_x = earth_radius * np.cos(angles)
    earth_y = earth_radius * np.sin(angles)
    orbit_r = earth_radius + altitude
    orbit_x = orbit_r * np.cos(angles)
    orbit_y = orbit_r * np.sin(angles)
    
    fig.add_trace(go.Scatter(x=earth_x, y=earth_y, fill='toself', name='Earth',
                            fillcolor='blue', line=dict(color='blue')), row=1, col=1)
    fig.add_trace(go.Scatter(x=orbit_x, y=orbit_y, name='GPS Orbit',
                            line=dict(color='red', dash='dash')), row=1, col=1)
    
    # Time effects
    fig.add_trace(go.Bar(x=['Special Relativity', 'General Relativity', 'Net Effect'],
                        y=[-sr_effect, gr_effect, gr_effect - sr_effect],
                        name='Time Effects (μs)'), row=1, col=2)
    
    # Position error growth
    days = np.arange(1, 31)
    cumulative_error = np.abs(gr_effect - sr_effect) * days * 0.3 / 1000  # meters
    fig.add_trace(go.Scatter(x=days, y=cumulative_error, name='Position Error (m)',
                            line=dict(color='red')), row=2, col=1)
    
    # Correction necessity indicator
    net_effect_daily = abs(gr_effect - sr_effect)
    correction_needed = net_effect_daily > 100  # threshold
    
    fig.add_trace(go.Indicator(
        mode = "gauge+number",
        value = net_effect_daily,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Daily Drift (μs)"},
        gauge = {'axis': {'range': [None, 500]},
                'bar': {'color': "red" if correction_needed else "green"},
                'steps': [{'range': [0, 100], 'color': "lightgreen"},
                         {'range': [100, 500], 'color': "lightcoral"}],
                'threshold': {'line': {'color': "red", 'width': 4},
                             'thickness': 0.75, 'value': 100}}), row=2, col=2)
    
    fig.update_layout(height=600, showlegend=False, title="GPS Relativity Effects")
    return fig

def create_interstellar_travel_viz(velocity_fraction, earth_time, ship_time):
    """Create interstellar travel time dilation visualization"""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Velocity vs Time Dilation', 'Mission Timeline', 'Energy Requirements', 'Distance-Time Diagram')
    )
    
    # Velocity vs time dilation curve
    v_range = np.linspace(0.1, 0.99, 100)
    gamma_range = 1 / np.sqrt(1 - v_range**2)
    
    fig.add_trace(go.Scatter(x=v_range, y=gamma_range, name='Time Dilation Factor',
                            line=dict(color='blue')), row=1, col=1)
    fig.add_trace(go.Scatter(x=[velocity_fraction], y=[1/np.sqrt(1-velocity_fraction**2)],
                            mode='markers', marker=dict(size=12, color='red'),
                            name='Current Mission'), row=1, col=1)
    
    # Mission timeline comparison
    fig.add_trace(go.Bar(x=['Earth Observers', 'Ship Crew'],
                        y=[earth_time, ship_time],
                        name='Mission Duration (years)',
                        marker_color=['lightblue', 'orange']), row=1, col=2)
    
    # Energy requirements (simplified)
    kinetic_energy = (1/np.sqrt(1-velocity_fraction**2) - 1) * 1000  # relative to 1000 kg ship
    fig.add_trace(go.Bar(x=['Kinetic Energy'], y=[kinetic_energy],
                        name='Energy (relative units)',
                        marker_color='red'), row=2, col=1)
    
    # Distance-time diagram
    earth_timeline = np.linspace(0, earth_time, 100)
    ship_timeline = np.linspace(0, ship_time, 100)
    
    fig.add_trace(go.Scatter(x=earth_timeline, y=earth_timeline*velocity_fraction,
                            name='Earth Perspective', line=dict(color='blue')), row=2, col=2)
    fig.add_trace(go.Scatter(x=ship_timeline, y=ship_timeline*velocity_fraction*np.sqrt(1-velocity_fraction**2),
                            name='Ship Perspective', line=dict(color='orange')), row=2, col=2)
    
    fig.update_layout(height=600, title="Relativistic Interstellar Travel")
    return fig

def create_lagrange_point_viz(system, mission_type):
    """Create Lagrange point visualization"""
    fig = go.Figure()
    
    # System parameters
    systems = {
        "Earth-Sun": {"m1": 1.99e30, "m2": 5.97e24, "a": 1.496e8, "names": ["Sun", "Earth"]},
        "Earth-Moon": {"m1": 5.97e24, "m2": 7.35e22, "a": 3.844e5, "names": ["Earth", "Moon"]},
        "Jupiter-Sun": {"m1": 1.99e30, "m2": 1.90e27, "a": 7.78e8, "names": ["Sun", "Jupiter"]},
        "Saturn-Titan": {"m1": 5.68e26, "m2": 1.35e23, "a": 1.22e6, "names": ["Saturn", "Titan"]}
    }
    
    sys_params = systems[system]
    m1, m2, a = sys_params["m1"], sys_params["m2"], sys_params["a"]
    mu = m2 / (m1 + m2)
    
    # Calculate Lagrange point positions (simplified)
    l1_dist = a * (mu/3)**(1/3)
    l2_dist = a * (mu/3)**(1/3)
    
    # Primary masses
    fig.add_trace(go.Scatter(x=[-a*mu], y=[0], mode='markers', 
                            marker=dict(size=20, color='yellow'),
                            name=sys_params["names"][0]))
    fig.add_trace(go.Scatter(x=[a*(1-mu)], y=[0], mode='markers',
                            marker=dict(size=15, color='blue'),
                            name=sys_params["names"][1]))
    
    # Lagrange points
    l_points = {
        'L1': [a*(1-mu) - l1_dist, 0],
        'L2': [a*(1-mu) + l2_dist, 0],
        'L3': [-a*mu - a, 0],
        'L4': [a*(1-mu) - a/2, a*np.sqrt(3)/2],
        'L5': [a*(1-mu) - a/2, -a*np.sqrt(3)/2]
    }
    
    colors = {'L1': 'red', 'L2': 'green', 'L3': 'orange', 'L4': 'purple', 'L5': 'purple'}
    
    for point, pos in l_points.items():
        fig.add_trace(go.Scatter(x=[pos[0]], y=[pos[1]], mode='markers+text',
                                marker=dict(size=8, color=colors[point]),
                                text=[point], textposition="top center",
                                name=point))
    
    # Highlight mission target
    if mission_type == "Space Telescope Deployment":
        target = 'L2'
    elif mission_type == "Solar Observatory":
        target = 'L1'
    elif mission_type in ["Asteroid Mining", "Space Colony"]:
        target = 'L4'
    else:
        target = 'L2'
    
    target_pos = l_points[target]
    fig.add_trace(go.Scatter(x=[target_pos[0]], y=[target_pos[1]], mode='markers',
                            marker=dict(size=15, color='red', symbol='star'),
                            name=f'Mission Target: {target}'))
    
    fig.update_layout(
        title=f"{system} System - {mission_type}",
        xaxis_title="Distance (km)",
        yaxis_title="Distance (km)",
        aspectratio=dict(x=1, y=1),
        height=500
    )
    
    return fig

def simulate_lagrange_mission(system, mission_type):
    """Simulate mission success rate"""
    # Simple success rate calculation based on mission complexity
    base_rates = {
        "Space Telescope Deployment": 0.85,
        "Solar Observatory": 0.90,
        "Asteroid Mining": 0.65,
        "Space Colony": 0.45
    }
    
    system_difficulty = {
        "Earth-Sun": 0.0,
        "Earth-Moon": 0.05,
        "Jupiter-Sun": -0.1,
        "Saturn-Titan": 0.1
    }
    
    base_rate = base_rates.get(mission_type, 0.7)
    difficulty_modifier = system_difficulty.get(system, 0.0)
    return max(0.1, min(0.95, base_rate + difficulty_modifier))

if __name__ == "__main__":
    main()
