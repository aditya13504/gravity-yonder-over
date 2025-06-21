"""
Demo Interactive Simulations
Quick test script to demonstrate the interactive simulation features
"""

import streamlit as st
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from src.interactive_simulation_engine import interactive_engine
from src.simulation_datasets import PreGeneratedSimulations
import numpy as np

st.set_page_config(
    page_title="Interactive Simulation Demo",
    page_icon="🎮",
    layout="wide"
)

st.title("🎮 Interactive Physics Simulation Demo")
st.markdown("Testing the new interactive simulation capabilities")

# Initialize simulation data
@st.cache_resource
def get_demo_simulation():
    """Create demo simulation data"""
    simulations = PreGeneratedSimulations()
    return simulations.load_simulation("binary_orbit", (32, 32, 16))

# Demo controls
demo_type = st.selectbox(
    "Choose Demo Type:",
    ["Plotly Animation", "Pygame Video", "PyBullet 3D", "All Demos"]
)

if st.button("🚀 Run Demo"):
    simulation_data = get_demo_simulation()
    
    if demo_type == "Plotly Animation" or demo_type == "All Demos":
        st.markdown("### 📊 Plotly Animated Simulation")
        with st.spinner("Creating Plotly animation..."):
            try:
                fig = interactive_engine.create_plotly_animated_simulation(simulation_data)
                st.plotly_chart(fig, use_container_width=True)
                st.success("✅ Plotly animation created successfully!")
            except Exception as e:
                st.error(f"Plotly animation failed: {e}")
    
    if demo_type == "Pygame Video" or demo_type == "All Demos":
        st.markdown("### 🎮 Pygame Video Simulation")
        with st.spinner("Creating Pygame video simulation..."):
            try:
                # Use simplified parameters for demo
                video_path = interactive_engine.create_pygame_orbital_simulation(
                    mass1=1.989e30,  # Solar mass
                    mass2=5.97e24,   # Earth mass  
                    separation=1.5e11,  # 1 AU
                    duration=3  # 3 seconds
                )
                
                st.info(f"Video created at: {video_path}")
                
                # Try to display the video
                if video_path.endswith('.mp4'):
                    try:
                        video_base64 = interactive_engine.get_video_base64(video_path)
                        if video_base64:
                            video_html = f"""
                            <video width="100%" height="400" controls autoplay loop>
                                <source src="data:video/mp4;base64,{video_base64}" type="video/mp4">
                                Your browser does not support the video tag.
                            </video>
                            """
                            st.markdown(video_html, unsafe_allow_html=True)
                            st.success("✅ Pygame video simulation created successfully!")
                        else:
                            st.warning("Video created but couldn't be displayed")
                    except Exception as e:
                        st.error(f"Error displaying video: {e}")
                        st.info("Video was created but couldn't be embedded")
                        
            except Exception as e:
                st.error(f"Pygame simulation failed: {e}")
                st.info("This might be due to headless environment limitations")
    
    if demo_type == "PyBullet 3D" or demo_type == "All Demos":
        st.markdown("### 🎲 PyBullet 3D Physics Simulation")
        with st.spinner("Creating 3D physics simulation..."):
            try:
                # Create demo objects
                objects = [
                    {
                        'mass': 2.0,
                        'radius': 0.5,
                        'position': [-2, 0, 2],
                        'color': [1, 1, 0, 1]  # Yellow
                    },
                    {
                        'mass': 1.0,
                        'radius': 0.3,
                        'position': [2, 0, 2],
                        'color': [0.4, 0.4, 1, 1]  # Blue
                    }
                ]
                
                video_path = interactive_engine.create_pybullet_3d_simulation(
                    objects, duration=3
                )
                
                st.info(f"3D simulation created at: {video_path}")
                
                # Try to display the video
                try:
                    video_base64 = interactive_engine.get_video_base64(video_path)
                    if video_base64:
                        video_html = f"""
                        <video width="100%" height="400" controls autoplay loop>
                            <source src="data:video/mp4;base64,{video_base64}" type="video/mp4">
                            Your browser does not support the video tag.
                        </video>
                        """
                        st.markdown(video_html, unsafe_allow_html=True)
                        st.success("✅ PyBullet 3D simulation created successfully!")
                    else:
                        st.warning("3D simulation created but couldn't be displayed")
                except Exception as e:
                    st.error(f"Error displaying 3D video: {e}")
                    st.info("3D simulation was created but couldn't be embedded")
                    
            except Exception as e:
                st.error(f"PyBullet simulation failed: {e}")
                st.info("This might be due to graphics/display limitations")

# Show system info
st.markdown("---")
st.markdown("### 🔧 System Information")

col1, col2, col3 = st.columns(3)

with col1:
    try:
        import pygame
        pygame_version = pygame.version.ver
        st.success(f"✅ Pygame: {pygame_version}")
    except:
        st.error("❌ Pygame not available")

with col2:
    try:
        import pybullet as p
        st.success("✅ PyBullet: Available")
    except:
        st.error("❌ PyBullet not available")

with col3:
    try:
        import plotly
        plotly_version = plotly.__version__
        st.success(f"✅ Plotly: {plotly_version}")
    except:
        st.error("❌ Plotly not available")

# Cleanup button
if st.button("🧹 Cleanup Resources"):
    interactive_engine.cleanup()
    st.success("Resources cleaned up successfully!")
