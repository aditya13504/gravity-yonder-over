"""
Debug version of the main Streamlit app
"""

import streamlit as st
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

st.set_page_config(
    page_title="Gravity Yonder Over",
    page_icon="🌌",
    layout="wide",
    initial_sidebar_state="expanded"
)

def debug_main():
    """Debug version of main function"""
    st.write("Debug: Starting main function")
    
    try:
        st.write("Debug: Testing imports...")
        
        from src.cpu_physics_engine import ModulusGravityEngine
        from src.cpu_data_processor import CuDFDataProcessor
        from src.simulation_datasets import PreGeneratedSimulations
        from src.plotly_visualizer import PhysicsVisualizer
        from src.educational_content import EducationalContentManager
        
        st.write("Debug: All imports successful")
        
        st.write("Debug: Initializing components...")
        
        # Initialize components one by one with debug info
        st.write("Initializing physics engine...")
        physics_engine = ModulusGravityEngine()
        st.success("✅ Physics engine initialized")
        
        st.write("Initializing data processor...")
        data_processor = CuDFDataProcessor()
        st.success("✅ Data processor initialized")
        
        st.write("Initializing simulations...")
        simulations = PreGeneratedSimulations()
        st.success("✅ Simulations initialized")
        
        st.write("Initializing visualizer...")
        visualizer = PhysicsVisualizer()
        st.success("✅ Visualizer initialized")
        
        st.write("Initializing content manager...")
        content_manager = EducationalContentManager()
        st.success("✅ Content manager initialized")
        
        # Now show the actual app
        st.title("🌌 Gravity Yonder Over")
        st.subheader("Interactive Physics Education with CPU-based Models")
        
        st.sidebar.title("🎮 Learning Adventures")
        
        # Simple navigation
        page = st.sidebar.selectbox(
            "Choose your adventure:",
            ["Home", "Gravity Basics", "Orbital Mechanics", "Physics Sandbox"]
        )
        
        if page == "Home":
            st.write("## Welcome to Gravity Yonder Over!")
            st.write("This is a physics education platform powered by CPU-based machine learning models.")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("### System Status")
                engine_status = physics_engine.get_status()
                st.json(engine_status)
            
            with col2:
                st.write("### Available Scenarios")
                scenarios = simulations.get_scenario_list()
                for scenario in scenarios[:3]:  # Show first 3
                    st.write(f"- {scenario['name']} ({scenario['status']})")
        
        elif page == "Gravity Basics":
            st.write("## 🍎 Gravity Basics")
            st.write("Learn about gravitational forces and how they work!")
            
            # Simple gravity simulation
            if st.button("Run Simple Gravity Demo"):
                import numpy as np
                import plotly.graph_objects as go
                
                # Create simple gravity visualization
                x = np.linspace(-10, 10, 20)
                y = np.linspace(-10, 10, 20)
                X, Y = np.meshgrid(x, y)
                Z = -1 / (np.sqrt(X**2 + Y**2) + 0.1)  # Simple potential
                
                fig = go.Figure(data=go.Surface(z=Z, x=X, y=Y))
                fig.update_layout(title="Gravitational Potential")
                st.plotly_chart(fig)
        
        elif page == "Orbital Mechanics":
            st.write("## 🚀 Orbital Mechanics")
            st.write("Explore how objects move in gravitational fields!")
            
        elif page == "Physics Sandbox":
            st.write("## 🔬 Physics Sandbox")
            st.write("Experiment with different physics scenarios!")
            
            # Simple parameter controls
            mass = st.slider("Central Mass (in Earth masses)", 1, 100, 10)
            distance = st.slider("Orbital Distance (in Earth radii)", 1, 50, 10)
            
            st.write(f"Simulation with {mass} Earth masses at {distance} Earth radii")
        
    except Exception as e:
        st.error(f"Error in main function: {e}")
        import traceback
        st.code(traceback.format_exc())

if __name__ == "__main__":
    debug_main()
