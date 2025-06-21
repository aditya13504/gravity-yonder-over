"""
Gravity Yonder Over - Educational Physics Platform
Clean version with NVIDIA tools integration and proper fallbacks
"""

import streamlit as st
import sys
import uuid
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# Import core modules
try:
    from src.modulus_physics import ModulusGravityEngine
    from src.cudf_processor import CuDFDataProcessor
    from src.simulation_datasets import PreGeneratedSimulations
    from src.plotly_visualizer import PhysicsVisualizer
    from src.educational_content import EducationalContentManager
except ImportError as e:
    st.error(f"Core module import error: {e}")
    st.stop()

# Try to import optional NVIDIA tools
CuQuantumGravityEngine = None
MorpheusPhysicsAnalyzer = None
PhysicsNeMoEngine = None
show_nvidia_tools_showcase = None

try:
    from backend.simulations.cuquantum_engine import CuQuantumGravityEngine
except ImportError:
    pass

try:
    from backend.simulations.morpheus_analyzer import MorpheusPhysicsAnalyzer
except ImportError:
    pass

try:
    from backend.simulations.physicsnemo_engine import PhysicsNeMoEngine
except ImportError:
    pass

try:
    from nvidia_tools_showcase import show_nvidia_tools_showcase
except ImportError:
    pass

# Configure Streamlit page
st.set_page_config(
    page_title="Gravity Yonder Over - Physics Education Platform",
    page_icon="🌌",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1f1c2c 0%, #928dab 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 1rem;
    }
    .stMetric {
        background: rgba(255, 255, 255, 0.1);
        padding: 0.5rem;
        border-radius: 5px;
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
            
            # Initialize optional NVIDIA tools
            if CuQuantumGravityEngine is not None:
                try:
                    st.session_state.cuquantum_engine = CuQuantumGravityEngine()
                except Exception:
                    st.session_state.cuquantum_engine = None
            else:
                st.session_state.cuquantum_engine = None
                
            if MorpheusPhysicsAnalyzer is not None:
                try:
                    st.session_state.morpheus_analyzer = MorpheusPhysicsAnalyzer()
                except Exception:
                    st.session_state.morpheus_analyzer = None
            else:
                st.session_state.morpheus_analyzer = None
                
            if PhysicsNeMoEngine is not None:
                try:
                    st.session_state.physicsnemo_engine = PhysicsNeMoEngine()
                except Exception:
                    st.session_state.physicsnemo_engine = None
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

def show_system_status():
    """Display system status in sidebar"""
    st.sidebar.markdown("### 🔧 System Status")
    
    # Core NVIDIA tools status
    modulus_status = st.session_state.modulus_engine.get_status()
    if modulus_status['available']:
        st.sidebar.success("✅ NVIDIA Modulus: Active")
    else:
        st.sidebar.warning("⚠️ NVIDIA Modulus: CPU Fallback")
    
    cudf_status = st.session_state.cudf_processor.get_status()
    if cudf_status['available']:
        st.sidebar.success("✅ cuDF: GPU Acceleration")
    else:
        st.sidebar.info("📊 cuDF: Pandas Fallback")
    
    # Optional NVIDIA tools status
    if st.session_state.cuquantum_engine is not None:
        try:
            cuquantum_status = st.session_state.cuquantum_engine.get_status()
            if cuquantum_status['available']:
                st.sidebar.success("✅ cuQuantum: Quantum Gravity")
            else:
                st.sidebar.info("⚛️ cuQuantum: CPU Fallback")
        except Exception:
            st.sidebar.info("⚛️ cuQuantum: Error")
    else:
        st.sidebar.info("⚛️ cuQuantum: Not Available")
    
    if st.session_state.morpheus_analyzer is not None:
        try:
            morpheus_status = st.session_state.morpheus_analyzer.get_status()
            if morpheus_status['available']:
                st.sidebar.success("✅ Morpheus: Real-time Analysis")
            else:
                st.sidebar.info("🔍 Morpheus: Basic Mode")
        except Exception:
            st.sidebar.info("🔍 Morpheus: Error")
    else:
        st.sidebar.info("🔍 Morpheus: Not Available")
    
    if st.session_state.physicsnemo_engine is not None:
        try:
            physicsnemo_status = st.session_state.physicsnemo_engine.get_status()
            if physicsnemo_status['available']:
                st.sidebar.success("✅ PhysicsNeMo: AI Models")
            else:
                st.sidebar.info("🧠 PhysicsNeMo: PyTorch Mode")
        except Exception:
            st.sidebar.info("🧠 PhysicsNeMo: Error")
    else:
        st.sidebar.info("🧠 PhysicsNeMo: Not Available")

def show_home_page():
    """Display the home page with overview"""
    st.markdown("""
    ## 🌌 Welcome to Gravity Yonder Over!
    
    **An Interactive Physics Education Platform powered by NVIDIA AI Tools**
    
    ### 🎯 What You Can Explore:
    
    - **🍎 Gravity Basics**: Start with fundamental concepts
    - **🚀 Orbital Mechanics**: Explore planetary motion and spacecraft trajectories  
    - **⚫ Black Hole Physics**: Dive into extreme gravity and spacetime
    - **🌌 Gravitational Waves**: Detect ripples in spacetime
    - **⚡ Relativity Missions**: Experience Einstein's theories
    - **🌍 Lagrange Point Missions**: Multi-body orbital mechanics
    
    ### 🔧 NVIDIA Technologies Integrated:
    
    ✅ **NVIDIA Modulus** - Physics-informed neural networks  
    ✅ **cuDF** - GPU-accelerated data processing  
    ✅ **cuQuantum** - Quantum computing acceleration (if available)  
    ✅ **Morpheus** - Real-time AI analytics (if available)  
    ✅ **PhysicsNeMo** - AI physics modeling (if available)  
    
    ### 🎓 Educational Levels:
    
    - **Beginner**: Interactive experiments and visual learning
    - **Intermediate**: Physics simulations and problem solving
    - **Advanced**: Research-level modeling and AI-assisted discovery
    
    Choose an adventure from the sidebar to begin your journey through the universe! 🚀
    """)
    
    # Show some basic metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Available Scenarios", "25+")
    
    with col2:
        gpu_available = st.session_state.modulus_engine.get_status()['available']
        st.metric("GPU Acceleration", "✅" if gpu_available else "⚠️ CPU")
    
    with col3:
        nvidia_tools = sum([
            st.session_state.cuquantum_engine is not None,
            st.session_state.morpheus_analyzer is not None,
            st.session_state.physicsnemo_engine is not None
        ]) + 2  # Always have Modulus and cuDF
        st.metric("NVIDIA Tools", f"{nvidia_tools}/5")

def show_nvidia_tools_status():
    """Show detailed NVIDIA tools status page"""
    st.header("🚀 NVIDIA Tools Integration Status")
    
    st.markdown("""
    This page shows the status of all NVIDIA AI and physics tools integrated into the platform.
    """)
    
    # Create status table
    tools_data = []
    
    # Core tools
    modulus_status = st.session_state.modulus_engine.get_status()
    tools_data.append({
        "Tool": "NVIDIA Modulus",
        "Status": "✅ Available" if modulus_status['available'] else "⚠️ CPU Fallback",
        "Function": "Physics-informed neural networks",
        "GPU Acceleration": "Yes" if modulus_status['available'] else "No"
    })
    
    cudf_status = st.session_state.cudf_processor.get_status()
    tools_data.append({
        "Tool": "cuDF",
        "Status": "✅ Available" if cudf_status['available'] else "⚠️ Pandas Fallback",
        "Function": "GPU data processing",
        "GPU Acceleration": "Yes" if cudf_status['available'] else "No"
    })
    
    # Optional tools
    if st.session_state.cuquantum_engine is not None:
        try:
            cuquantum_status = st.session_state.cuquantum_engine.get_status()
            tools_data.append({
                "Tool": "cuQuantum",
                "Status": "✅ Available" if cuquantum_status['available'] else "⚠️ CPU Fallback",
                "Function": "Quantum computing acceleration",
                "GPU Acceleration": "Yes" if cuquantum_status['available'] else "No"
            })
        except Exception:
            tools_data.append({
                "Tool": "cuQuantum",
                "Status": "❌ Error",
                "Function": "Quantum computing acceleration",
                "GPU Acceleration": "No"
            })
    else:
        tools_data.append({
            "Tool": "cuQuantum",
            "Status": "❌ Not Available",
            "Function": "Quantum computing acceleration",
            "GPU Acceleration": "No"
        })
    
    if st.session_state.morpheus_analyzer is not None:
        try:
            morpheus_status = st.session_state.morpheus_analyzer.get_status()
            tools_data.append({
                "Tool": "Morpheus",
                "Status": "✅ Available" if morpheus_status['available'] else "⚠️ Basic Mode",
                "Function": "Real-time AI analytics",
                "GPU Acceleration": "Yes" if morpheus_status['available'] else "No"
            })
        except Exception:
            tools_data.append({
                "Tool": "Morpheus",
                "Status": "❌ Error",
                "Function": "Real-time AI analytics",
                "GPU Acceleration": "No"
            })
    else:
        tools_data.append({
            "Tool": "Morpheus",
            "Status": "❌ Not Available",
            "Function": "Real-time AI analytics",
            "GPU Acceleration": "No"
        })
    
    if st.session_state.physicsnemo_engine is not None:
        try:
            physicsnemo_status = st.session_state.physicsnemo_engine.get_status()
            tools_data.append({
                "Tool": "PhysicsNeMo",
                "Status": "✅ Available" if physicsnemo_status['available'] else "⚠️ PyTorch Mode",
                "Function": "AI physics modeling",
                "GPU Acceleration": "Yes" if physicsnemo_status['available'] else "No"
            })
        except Exception:
            tools_data.append({
                "Tool": "PhysicsNeMo",
                "Status": "❌ Error",
                "Function": "AI physics modeling",
                "GPU Acceleration": "No"
            })
    else:
        tools_data.append({
            "Tool": "PhysicsNeMo",
            "Status": "❌ Not Available",
            "Function": "AI physics modeling",
            "GPU Acceleration": "No"
        })
    
    # Display the table
    import pandas as pd
    df = pd.DataFrame(tools_data)
    st.dataframe(df, use_container_width=True)
    
    # Summary
    available_count = sum(1 for tool in tools_data if "✅" in tool["Status"])
    total_count = len(tools_data)
    
    st.markdown(f"""
    ### 📊 Summary
    - **Available Tools**: {available_count}/{total_count}
    - **Platform Status**: {'🟢 Fully Operational' if available_count >= 2 else '🟡 Basic Mode'}
    - **GPU Acceleration**: {'🟢 Available' if any('Yes' == tool['GPU Acceleration'] for tool in tools_data) else '🔴 CPU Only'}
    
    The platform is designed to work with any combination of available tools, 
    automatically falling back to CPU implementations when GPU tools are not available.
    """)

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
        <h3>Interactive Physics Education with NVIDIA AI Tools</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar navigation
    st.sidebar.markdown("## 🎮 Learning Adventures")
    
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
        "🌍 Lagrange Point Missions",
        "🚀 NVIDIA Tools Status"
    ]
    
    # Add showcase if available
    if show_nvidia_tools_showcase is not None:
        nav_options.append("🎯 NVIDIA Tools Showcase")
    
    game_selection = st.sidebar.selectbox(
        "Choose your physics adventure:",
        nav_options
    )
    
    # Show system status
    show_system_status()
    
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
    elif "NVIDIA Tools Status" in game_selection:
        show_nvidia_tools_status()
    elif "NVIDIA Tools Showcase" in game_selection and show_nvidia_tools_showcase is not None:
        show_nvidia_tools_showcase()
    else:
        st.info(f"Page '{game_selection}' is available in the full application.")
        st.markdown("""
        ### 🔧 This is a Clean Demo Version
        
        This version demonstrates:
        - ✅ Core NVIDIA tools integration (Modulus, cuDF)
        - ✅ Optional NVIDIA tools detection (cuQuantum, Morpheus, PhysicsNeMo) 
        - ✅ Robust fallback mechanisms
        - ✅ Real-time status monitoring
        
        All tools are properly integrated with graceful fallbacks when GPU libraries are not available.
        The platform works on any system from mobile devices to high-end GPU clusters.
        """)

if __name__ == "__main__":
    main()
