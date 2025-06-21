"""
Simple test version of the Streamlit app to debug issues
"""

import streamlit as st
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

st.set_page_config(
    page_title="Gravity Yonder Over - Test",
    page_icon="🌌",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🌌 Gravity Yonder Over - Debug Test")

try:
    st.write("Testing imports...")
    
    # Test imports one by one
    st.write("1. Testing CPU Physics Engine...")
    from src.cpu_physics_engine import ModulusGravityEngine
    st.success("✅ CPU Physics Engine imported successfully")
    
    st.write("2. Testing CPU Data Processor...")
    from src.cpu_data_processor import CuDFDataProcessor
    st.success("✅ CPU Data Processor imported successfully")
    
    st.write("3. Testing Simulation Datasets...")
    from src.simulation_datasets import PreGeneratedSimulations
    st.success("✅ Simulation Datasets imported successfully")
    
    st.write("4. Testing Plotly Visualizer...")
    from src.plotly_visualizer import PhysicsVisualizer
    st.success("✅ Plotly Visualizer imported successfully")
    
    st.write("5. Testing Educational Content...")
    from src.educational_content import EducationalContentManager
    st.success("✅ Educational Content imported successfully")
    
    st.write("6. Testing component initialization...")
    
    # Test initialization
    physics_engine = ModulusGravityEngine()
    st.success("✅ Physics engine initialized")
    
    data_processor = CuDFDataProcessor()
    st.success("✅ Data processor initialized")
    
    simulations = PreGeneratedSimulations()
    st.success("✅ Simulations manager initialized")
    
    visualizer = PhysicsVisualizer()
    st.success("✅ Visualizer initialized")
    
    content_manager = EducationalContentManager()
    st.success("✅ Content manager initialized")
    
    st.write("## System Status")
    st.write("All components loaded successfully!")
    
    # Test basic functionality
    st.write("## Testing Basic Functionality")
    
    # Test physics engine status
    engine_status = physics_engine.get_status()
    st.write("Physics Engine Status:", engine_status)
    
    # Test data processor metrics
    processor_metrics = data_processor.get_performance_metrics()
    st.write("Data Processor Metrics:", processor_metrics)
    
    # Test scenarios list
    scenarios = simulations.get_scenario_list()
    st.write(f"Available scenarios: {len(scenarios)}")
    for scenario in scenarios:
        st.write(f"- {scenario['name']} ({scenario['status']})")
    
except Exception as e:
    st.error(f"Error during testing: {e}")
    st.write("Error details:")
    import traceback
    st.code(traceback.format_exc())
