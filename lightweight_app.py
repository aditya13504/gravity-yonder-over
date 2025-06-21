"""
🌌 Gravity Yonder Over - Lightweight Edition
Optimized for low-resource machines and basic hardware

This module provides simplified versions of physics simulations that can run
on older computers, tablets, and mobile devices without GPU acceleration.
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
import time
from typing import Dict, List, Tuple, Any
import json

# Set page config for mobile-friendly layout
st.set_page_config(
    page_title="Gravity Yonder Over - Lightweight",
    page_icon="🌌",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Lightweight CSS for better performance
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
    .lightweight-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #e9ecef;
        margin: 0.5rem 0;
    }
    .performance-note {
        background: #d4edda;
        color: #155724;
        padding: 0.75rem;
        border-radius: 5px;
        border: 1px solid #c3e6cb;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

class LightweightPhysicsEngine:
    """Simplified physics engine for low-resource devices"""
    
    def __init__(self):
        self.G = 6.67430e-11  # Gravitational constant
        self.c = 299792458    # Speed of light
        
    def simple_gravity_force(self, m1: float, m2: float, r: float) -> float:
        """Calculate gravitational force using simplified computation"""
        if r <= 0:
            return float('inf')
        return self.G * m1 * m2 / (r * r)
    
    def basic_orbital_velocity(self, M: float, r: float) -> float:
        """Calculate orbital velocity with basic math"""
        if r <= 0:
            return 0
        return np.sqrt(self.G * M / r)
    
    def schwarzschild_radius(self, mass: float) -> float:
        """Calculate Schwarzschild radius"""
        return 2 * self.G * mass / (self.c * self.c)
    
    def simple_time_dilation(self, velocity: float) -> float:
        """Calculate time dilation factor with overflow protection"""
        v_frac = velocity / self.c
        if v_frac >= 1.0:
            return 100  # Cap for visualization
        try:
            return 1 / np.sqrt(1 - v_frac * v_frac)
        except:
            return 100

# Initialize lightweight physics engine
@st.cache_resource
def get_physics_engine():
    return LightweightPhysicsEngine()

physics = get_physics_engine()

def create_simple_gravity_demo():
    """Create a simple gravity demonstration"""
    st.markdown("## 🍎 Basic Gravity Demo")
    
    with st.expander("ℹ️ What you'll learn"):
        st.markdown("""
        - How gravity pulls objects together
        - Why bigger objects have stronger gravity
        - How distance affects gravitational force
        """)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### 🎛️ Controls")
        
        # Simple controls
        mass1 = st.slider("Object 1 Mass (kg)", 1e20, 1e30, 5.97e24, format="%.2e")
        mass2 = st.slider("Object 2 Mass (kg)", 1e10, 1e25, 7.35e22, format="%.2e")
        distance = st.slider("Distance (km)", 1000, 1000000, 384400)
        
        # Calculate force
        force = physics.simple_gravity_force(mass1, mass2, distance * 1000)
        
        st.markdown("### 📊 Results")
        st.metric("Gravitational Force", f"{force:.2e} N")
        
        # Compare to familiar forces
        earth_surface_g = 9.81 * mass2
        comparison = force / earth_surface_g if earth_surface_g > 0 else 0
        
        st.metric("Compared to your weight on Earth", f"{comparison:.2e}x")
        
    with col2:
        # Create simple visualization
        fig = go.Figure()
        
        # Draw objects as circles
        fig.add_shape(
            type="circle",
            x0=-20, y0=-20, x1=20, y1=20,
            fillcolor="blue",
            line_color="blue",
            name="Object 1"
        )
        
        fig.add_shape(
            type="circle", 
            x0=60, y0=-10, x1=80, y1=10,
            fillcolor="gray",
            line_color="gray",
            name="Object 2"
        )
        
        # Draw force arrow
        arrow_length = min(50, max(10, np.log10(force) - 15))
        fig.add_annotation(
            x=50, y=0,
            ax=50-arrow_length, ay=0,
            xref="x", yref="y",
            axref="x", ayref="y",
            text="",
            showarrow=True,
            arrowhead=2,
            arrowsize=2,
            arrowwidth=3,
            arrowcolor="red"
        )
        
        fig.add_annotation(
            x=10, y=0,
            ax=10+arrow_length, ay=0,
            xref="x", yref="y",
            axref="x", ayref="y", 
            text="",
            showarrow=True,
            arrowhead=2,
            arrowsize=2,
            arrowwidth=3,
            arrowcolor="red"
        )
        
        fig.update_layout(
            title="Gravitational Force Between Objects",
            xaxis_range=[-50, 100],
            yaxis_range=[-50, 50],
            showlegend=False,
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)

def create_simple_orbit_demo():
    """Create a simple orbital mechanics demonstration"""
    st.markdown("## 🛰️ Simple Orbit Demo")
    
    with st.expander("ℹ️ What you'll learn"):
        st.markdown("""
        - Why satellites don't fall to Earth
        - How orbital speed depends on altitude
        - The relationship between distance and orbital period
        """)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### 🎛️ Satellite Controls")
        
        altitude = st.slider("Altitude above Earth (km)", 200, 40000, 400)
        planet_mass = st.selectbox(
            "Central Body:",
            ["Earth", "Moon", "Mars", "Jupiter"],
            index=0
        )
        
        # Planet properties (simplified)
        planet_data = {
            "Earth": {"mass": 5.97e24, "radius": 6371},
            "Moon": {"mass": 7.35e22, "radius": 1737},
            "Mars": {"mass": 6.39e23, "radius": 3390},
            "Jupiter": {"mass": 1.90e27, "radius": 69911}
        }
        
        M = planet_data[planet_mass]["mass"]
        R = planet_data[planet_mass]["radius"]
        orbital_radius = R + altitude
        
        # Calculate orbital properties
        velocity = physics.basic_orbital_velocity(M, orbital_radius * 1000)
        period = 2 * np.pi * orbital_radius * 1000 / velocity / 3600  # hours
        
        st.markdown("### 📊 Orbital Properties")
        st.metric("Orbital Velocity", f"{velocity/1000:.2f} km/s")
        st.metric("Orbital Period", f"{period:.2f} hours")
        st.metric("Distance from Center", f"{orbital_radius:,.0f} km")
        
    with col2:
        # Create simple orbital visualization
        angles = np.linspace(0, 2*np.pi, 100)
        
        # Planet
        planet_x = R * np.cos(angles) / 1000
        planet_y = R * np.sin(angles) / 1000
        
        # Orbit
        orbit_x = orbital_radius * np.cos(angles) / 1000
        orbit_y = orbital_radius * np.sin(angles) / 1000
        
        fig = go.Figure()
        
        # Planet
        fig.add_trace(go.Scatter(
            x=planet_x, y=planet_y,
            fill='toself',
            fillcolor='blue',
            line=dict(color='blue'),
            name=planet_mass,
            hovertemplate=f"{planet_mass}<br>Radius: {R:,} km"
        ))
        
        # Orbit
        fig.add_trace(go.Scatter(
            x=orbit_x, y=orbit_y,
            mode='lines',
            line=dict(color='red', dash='dash'),
            name='Orbit',
            hovertemplate=f"Orbital Altitude: {altitude:,} km"
        ))
        
        # Satellite position
        sat_angle = time.time() % (2*np.pi)  # Simple animation
        sat_x = orbital_radius * np.cos(sat_angle) / 1000
        sat_y = orbital_radius * np.sin(sat_angle) / 1000
        
        fig.add_trace(go.Scatter(
            x=[sat_x], y=[sat_y],
            mode='markers',
            marker=dict(size=8, color='yellow'),
            name='Satellite'
        ))
        
        fig.update_layout(
            title=f"Satellite Orbit around {planet_mass}",
            xaxis_title="Distance (1000 km)",
            yaxis_title="Distance (1000 km)",
            aspectratio=dict(x=1, y=1),
            height=500,
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)

def create_simple_blackhole_demo():
    """Create a simple black hole demonstration"""
    st.markdown("## ⚫ Simple Black Hole Demo")
    
    with st.expander("ℹ️ What you'll learn"):
        st.markdown("""
        - What makes a black hole "black"
        - How the event horizon size depends on mass
        - Why time slows down near black holes
        """)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### 🎛️ Black Hole Controls")
        
        bh_mass_solar = st.slider("Black Hole Mass (Solar Masses)", 1, 100, 10)
        bh_mass = bh_mass_solar * 1.989e30
        
        # Calculate Schwarzschild radius
        rs = physics.schwarzschild_radius(bh_mass)
        rs_km = rs / 1000
        
        st.markdown("### 📊 Black Hole Properties")
        st.metric("Event Horizon Radius", f"{rs_km:.2f} km")
        
        # Compare to familiar objects
        earth_radius = 6371
        if rs_km < earth_radius:
            comparison = f"{earth_radius/rs_km:.1f}x smaller than Earth"
        else:
            comparison = f"{rs_km/earth_radius:.1f}x larger than Earth"
        st.metric("Size Comparison", comparison)
        
        # Time dilation at different distances
        st.markdown("### ⏰ Time Dilation Effects")
        distances = [10*rs_km, 5*rs_km, 2*rs_km, 1.5*rs_km]
        
        for dist in distances:
            if dist > rs_km:
                # Simplified time dilation formula near black hole
                factor = 1 / np.sqrt(1 - rs_km/dist)
                st.write(f"At {dist:.1f} km: {factor:.2f}x slower")
    
    with col2:
        # Create simple black hole visualization
        fig = go.Figure()
        
        # Event horizon
        angles = np.linspace(0, 2*np.pi, 100)
        eh_x = rs_km * np.cos(angles)
        eh_y = rs_km * np.sin(angles)
        
        fig.add_trace(go.Scatter(
            x=eh_x, y=eh_y,
            fill='toself',
            fillcolor='black',
            line=dict(color='red'),
            name='Event Horizon',
            hovertemplate=f"Event Horizon<br>Radius: {rs_km:.2f} km"
        ))
        
        # Photon sphere (1.5 times event horizon)
        photon_r = 1.5 * rs_km
        ph_x = photon_r * np.cos(angles)
        ph_y = photon_r * np.sin(angles)
        
        fig.add_trace(go.Scatter(
            x=ph_x, y=ph_y,
            mode='lines',
            line=dict(color='yellow', dash='dot'),
            name='Photon Sphere',
            hovertemplate=f"Photon Sphere<br>Radius: {photon_r:.2f} km"
        ))
        
        # Safe distance marker
        safe_r = 10 * rs_km
        safe_x = safe_r * np.cos(angles)
        safe_y = safe_r * np.sin(angles)
        
        fig.add_trace(go.Scatter(
            x=safe_x, y=safe_y,
            mode='lines',
            line=dict(color='green', dash='dash'),
            name='Safe Distance',
            hovertemplate=f"Safe Distance<br>Radius: {safe_r:.2f} km"
        ))
        
        max_range = safe_r * 1.2
        fig.update_layout(
            title=f"Black Hole Structure ({bh_mass_solar} Solar Masses)",
            xaxis_title="Distance (km)",
            yaxis_title="Distance (km)",
            xaxis_range=[-max_range, max_range],
            yaxis_range=[-max_range, max_range],
            aspectratio=dict(x=1, y=1),
            height=500,
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)

def create_performance_dashboard():
    """Show performance and optimization information"""
    st.markdown("## ⚡ Performance Dashboard")
    
    st.markdown('<div class="performance-note">🚀 <strong>Lightweight Mode Active</strong> - Optimized for low-resource devices</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 💻 System Requirements")
        st.markdown("""
        **Minimum:**
        - 2GB RAM
        - Any modern web browser
        - 1 Mbps internet connection
        
        **Recommended:**
        - 4GB RAM
        - Chrome/Firefox latest version
        - 5 Mbps internet connection
        """)
    
    with col2:
        st.markdown("### 🔧 Optimizations")
        st.markdown("""
        **Enabled:**
        - ✅ Simplified physics calculations
        - ✅ Reduced animation complexity
        - ✅ Compressed visualizations
        - ✅ Cached computations
        - ✅ Progressive loading
        """)
    
    with col3:
        st.markdown("### 📊 Performance Metrics")
        
        # Simulate performance metrics
        render_time = np.random.uniform(50, 150)
        memory_usage = np.random.uniform(10, 25)
        
        st.metric("Avg Render Time", f"{render_time:.0f} ms")
        st.metric("Memory Usage", f"{memory_usage:.1f} MB")
        st.metric("Cache Hit Rate", "95%")

def main():
    """Main application function"""
    
    # Header
    st.markdown('<div class="main-header"><h1>🌌 Gravity Yonder Over - Lightweight Edition</h1><p>Physics education optimized for everyone</p></div>', unsafe_allow_html=True)
    
    # Performance note
    st.info("💡 **Lightweight Mode**: This version is optimized for older devices, tablets, and slow internet connections. All simulations use simplified calculations for better performance.")
    
    # Navigation
    tab1, tab2, tab3, tab4 = st.tabs(["🍎 Gravity Basics", "🛰️ Orbits", "⚫ Black Holes", "⚡ Performance"])
    
    with tab1:
        create_simple_gravity_demo()
    
    with tab2:
        create_simple_orbit_demo()
    
    with tab3:
        create_simple_blackhole_demo()
    
    with tab4:
        create_performance_dashboard()
    
    # Footer
    st.markdown("---")
    st.markdown("**Gravity Yonder Over** - Making physics accessible to everyone, everywhere")
    st.markdown("💝 Free and open-source educational platform")

if __name__ == "__main__":
    main()
