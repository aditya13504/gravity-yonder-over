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
        <h3>Interactive Physics Education with NVIDIA Modulus</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar navigation
    st.sidebar.markdown("## 🎮 Learning Adventures")
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
        ]
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
    
    # Performance metrics
    st.sidebar.markdown("### 📊 Performance")
    available_scenarios = st.session_state.simulations.get_scenario_list()
    col1, col2 = st.sidebar.columns(2)
    with col1:
        st.metric("Scenarios", len(available_scenarios))
    with col2:
        precomputed_count = sum(1 for s in available_scenarios if s['status'] == 'pre-computed')
        st.metric("Pre-computed", precomputed_count)
    
    # Route to appropriate page
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


if __name__ == "__main__":
    main()
