"""
🌌 Gravity Yonder Over - Demo Application
A simplified version showcasing the main features
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import math
import time

# Page configuration
st.set_page_config(
    page_title="🌌 Gravity Yonder Over - Demo",
    page_icon="🌌",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main {
        padding: 1rem;
    }
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 20px;
        border: none;
        padding: 0.5rem 1rem;
        font-weight: bold;
        transition: all 0.3s;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .game-card {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'score' not in st.session_state:
    st.session_state.score = 0
if 'level' not in st.session_state:
    st.session_state.level = 1
if 'apple_drops' not in st.session_state:
    st.session_state.apple_drops = 0

# Sidebar
with st.sidebar:
    st.markdown("## 🎮 Navigation")
    mode = st.radio(
        "Choose Mode:",
        ["🏠 Home", "🍎 Apple Drop Game", "🌍 Orbital Mechanics", "🔬 Gravity Sandbox", "📊 Visualizations"]
    )
    
    st.markdown("---")
    st.markdown("### 📈 Progress")
    progress = min(st.session_state.score / 1000, 1.0)
    st.progress(progress)
    st.markdown(f"**Score:** {st.session_state.score}")
    st.markdown(f"**Level:** {st.session_state.level}")
    st.markdown(f"**Apple Drops:** {st.session_state.apple_drops}")

# Main content
if mode == "🏠 Home":
    # Hero Section
    st.markdown("""
    <div style='text-align: center; padding: 2rem; background: linear-gradient(135deg, #0f0c29, #302b63, #24243e); border-radius: 20px; margin-bottom: 2rem;'>
        <h1 style='color: white; font-size: 3rem; margin-bottom: 1rem;'>🌌 Gravity Yonder Over</h1>
        <h3 style='color: #94a3b8; margin-bottom: 2rem;'>From falling apples to orbital slingshots — learn gravity the cosmic way</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Feature cards
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class='game-card'>
            <h3>🎮 Interactive Games</h3>
            <p>5 engaging mini-games that teach gravity concepts through hands-on experimentation.</p>
            <ul>
                <li>🍎 Apple Drop Physics</li>
                <li>🚀 Orbital Mechanics</li>
                <li>⚫ Black Hole Navigation</li>
                <li>🌌 Gravity Wells</li>
                <li>🛸 Escape Velocity Challenge</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class='game-card'>
            <h3>🔬 Physics Sandbox</h3>
            <p>Experiment with gravitational parameters in real-time simulations.</p>
            <ul>
                <li>🌍 Multi-body simulations</li>
                <li>⚙️ Adjustable parameters</li>
                <li>📊 Real-time visualization</li>
                <li>🎯 Educational scenarios</li>
                <li>💫 Custom orbits</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class='game-card'>
            <h3>🤖 AI-Powered Learning</h3>
            <p>Advanced simulations using Physics-Informed Neural Networks.</p>
            <ul>
                <li>🧠 PINN gravity models</li>
                <li>🎯 Trajectory prediction</li>
                <li>⚡ GPU acceleration</li>
                <li>📈 Adaptive learning</li>
                <li>🔍 Pattern recognition</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

elif mode == "🍎 Apple Drop Game":
    st.markdown("# 🍎 Apple Drop Game")
    st.markdown("Experience Newton's gravity law by dropping apples from different heights!")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Game parameters
        height = st.slider("Drop Height (meters)", 1.0, 100.0, 10.0, 0.5)
        gravity = st.slider("Gravity (m/s²)", 1.0, 20.0, 9.81, 0.1)
        
        if st.button("🍎 Drop Apple!", key="drop_apple"):
            # Calculate fall time and impact velocity
            fall_time = math.sqrt(2 * height / gravity)
            impact_velocity = gravity * fall_time
            
            # Animation simulation
            st.markdown("### 📈 Fall Analysis")
            
            # Create time series for animation
            t = np.linspace(0, fall_time, 100)
            positions = height - 0.5 * gravity * t**2
            velocities = gravity * t
            
            # Position vs time plot
            fig_pos = go.Figure()
            fig_pos.add_trace(go.Scatter(
                x=t, y=positions,
                mode='lines+markers',
                name='Apple Position',
                line=dict(color='red', width=3)
            ))
            fig_pos.update_layout(
                title="Apple Position vs Time",
                xaxis_title="Time (seconds)",
                yaxis_title="Height (meters)",
                template="plotly_dark"
            )
            st.plotly_chart(fig_pos, use_container_width=True)
            
            # Velocity vs time plot
            fig_vel = go.Figure()
            fig_vel.add_trace(go.Scatter(
                x=t, y=velocities,
                mode='lines+markers',
                name='Apple Velocity',
                line=dict(color='blue', width=3)
            ))
            fig_vel.update_layout(
                title="Apple Velocity vs Time",
                xaxis_title="Time (seconds)",
                yaxis_title="Velocity (m/s)",
                template="plotly_dark"
            )
            st.plotly_chart(fig_vel, use_container_width=True)
            
            # Update game state
            st.session_state.apple_drops += 1
            st.session_state.score += int(10 * height)
            
            st.success(f"🎯 Apple hit the ground in {fall_time:.2f} seconds at {impact_velocity:.2f} m/s!")
            st.balloons()
    
    with col2:
        st.markdown("### 🎯 Game Stats")
        st.markdown(f"""
        <div class='metric-card'>
            <h4>Current Drop</h4>
            <p><strong>Height:</strong> {height:.1f} m</p>
            <p><strong>Gravity:</strong> {gravity:.2f} m/s²</p>
            <p><strong>Expected Time:</strong> {math.sqrt(2 * height / gravity):.2f} s</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### 📚 Learning Points")
        st.markdown("""
        - **Newton's Law:** F = ma
        - **Free Fall:** h = ½gt²
        - **Impact Velocity:** v = √(2gh)
        - **Acceleration:** Always 9.81 m/s² on Earth
        """)

elif mode == "🌍 Orbital Mechanics":
    st.markdown("# 🌍 Orbital Mechanics Simulator")
    st.markdown("Explore how objects orbit around celestial bodies!")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### ⚙️ Orbital Parameters")
        
        # Orbital parameters
        central_mass = st.slider("Central Body Mass (Earth masses)", 0.1, 10.0, 1.0, 0.1)
        orbital_radius = st.slider("Orbital Radius (Earth radii)", 1.1, 10.0, 2.0, 0.1)
        eccentricity = st.slider("Eccentricity", 0.0, 0.9, 0.0, 0.05)
        
        # Constants
        earth_mass = 5.97e24  # kg
        earth_radius = 6.37e6  # m
        G = 6.674e-11  # gravitational constant
        
        # Calculate orbital parameters
        mass = central_mass * earth_mass
        radius = orbital_radius * earth_radius
        
        # Circular orbital velocity
        v_circular = math.sqrt(G * mass / radius)
        
        # Orbital period
        period = 2 * math.pi * math.sqrt(radius**3 / (G * mass))
        
        st.markdown(f"""
        ### 📊 Calculated Values
        - **Orbital Velocity:** {v_circular/1000:.2f} km/s
        - **Orbital Period:** {period/3600:.2f} hours
        - **Surface Gravity:** {G * mass / (earth_radius)**2:.2f} m/s²
        """)
    
    with col2:
        # Create orbital visualization
        theta = np.linspace(0, 2*np.pi, 100)
        
        # Semi-major axis
        a = radius
        # Semi-minor axis  
        b = a * math.sqrt(1 - eccentricity**2)
        
        # Elliptical orbit
        x_orbit = a * np.cos(theta)
        y_orbit = b * np.sin(theta)
        
        # Central body (at one focus for ellipse)
        focus_offset = a * eccentricity
        
        fig = go.Figure()
        
        # Central body
        fig.add_trace(go.Scatter(
            x=[focus_offset], y=[0],
            mode='markers',
            marker=dict(size=20, color='blue'),
            name='Central Body'
        ))
        
        # Orbit
        fig.add_trace(go.Scatter(
            x=x_orbit + focus_offset, y=y_orbit,
            mode='lines',
            line=dict(color='white', width=2),
            name='Orbit'
        ))
        
        # Orbiting body (at current position)
        current_pos = [x_orbit[0] + focus_offset, y_orbit[0]]
        fig.add_trace(go.Scatter(
            x=[current_pos[0]], y=[current_pos[1]],
            mode='markers',
            marker=dict(size=10, color='red'),
            name='Orbiting Body'
        ))
        
        fig.update_layout(
            title="Orbital Simulation",
            xaxis_title="Distance (m)",
            yaxis_title="Distance (m)",
            template="plotly_dark",
            showlegend=True,
            aspectratio=dict(x=1, y=1)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Animation button
        if st.button("▶️ Animate Orbit"):
            placeholder = st.empty()
            
            for i in range(0, len(theta), 2):
                fig_anim = go.Figure()
                
                # Central body
                fig_anim.add_trace(go.Scatter(
                    x=[focus_offset], y=[0],
                    mode='markers',
                    marker=dict(size=20, color='blue'),
                    name='Central Body'
                ))
                
                # Full orbit path
                fig_anim.add_trace(go.Scatter(
                    x=x_orbit + focus_offset, y=y_orbit,
                    mode='lines',
                    line=dict(color='gray', width=1),
                    name='Orbit Path'
                ))
                
                # Orbiting body at current position
                current_x = x_orbit[i] + focus_offset
                current_y = y_orbit[i]
                fig_anim.add_trace(go.Scatter(
                    x=[current_x], y=[current_y],
                    mode='markers',
                    marker=dict(size=12, color='red'),
                    name='Satellite'
                ))
                
                # Velocity vector
                if i < len(theta) - 1:
                    dx = x_orbit[i+1] - x_orbit[i]
                    dy = y_orbit[i+1] - y_orbit[i]
                    scale = radius * 0.3
                    fig_anim.add_trace(go.Scatter(
                        x=[current_x, current_x + dx*scale],
                        y=[current_y, current_y + dy*scale],
                        mode='lines',
                        line=dict(color='yellow', width=3),
                        name='Velocity'
                    ))
                
                fig_anim.update_layout(
                    title=f"Orbital Animation - Step {i+1}",
                    xaxis_title="Distance (m)",
                    yaxis_title="Distance (m)", 
                    template="plotly_dark",
                    showlegend=False,
                    aspectratio=dict(x=1, y=1)
                )
                
                placeholder.plotly_chart(fig_anim, use_container_width=True)
                time.sleep(0.1)

elif mode == "🔬 Gravity Sandbox":
    st.markdown("# 🔬 Gravity Sandbox")
    st.markdown("Experiment with gravitational forces and multi-body systems!")
    
    # Body configuration
    num_bodies = st.slider("Number of Bodies", 2, 5, 3)
    
    bodies = []
    cols = st.columns(num_bodies)
    
    for i, col in enumerate(cols):
        with col:
            st.markdown(f"### Body {i+1}")
            mass = st.number_input(f"Mass {i+1} (kg)", value=1e24, format="%.2e", key=f"mass_{i}")
            x = st.slider(f"X Position {i+1}", -10.0, 10.0, float(i*2-2), key=f"x_{i}")
            y = st.slider(f"Y Position {i+1}", -10.0, 10.0, 0.0, key=f"y_{i}")
            
            bodies.append({
                'mass': mass,
                'x': x,
                'y': y,
                'name': f'Body {i+1}'
            })
    
    # Visualization
    fig = go.Figure()
    
    for i, body in enumerate(bodies):
        # Body size proportional to mass
        size = max(10, min(50, math.log10(body['mass']) * 2))
        
        fig.add_trace(go.Scatter(
            x=[body['x']], 
            y=[body['y']],
            mode='markers+text',
            marker=dict(size=size, color=px.colors.qualitative.Set1[i % len(px.colors.qualitative.Set1)]),
            text=[body['name']],
            textposition="top center",
            name=body['name']
        ))
        
        # Draw gravitational field lines (simplified)
        if len(bodies) > 1:
            for j, other_body in enumerate(bodies):
                if i != j:
                    # Force vector
                    dx = other_body['x'] - body['x']
                    dy = other_body['y'] - body['y']
                    distance = math.sqrt(dx**2 + dy**2)
                    
                    if distance > 0:
                        # Gravitational force direction
                        force_scale = 0.5
                        fx = dx / distance * force_scale
                        fy = dy / distance * force_scale
                        
                        fig.add_trace(go.Scatter(
                            x=[body['x'], body['x'] + fx],
                            y=[body['y'], body['y'] + fy],
                            mode='lines',
                            line=dict(color='white', width=1, dash='dash'),
                            showlegend=False
                        ))
    
    fig.update_layout(
        title="Gravitational System",
        xaxis_title="X Position (AU)",
        yaxis_title="Y Position (AU)",
        template="plotly_dark",
        aspectratio=dict(x=1, y=1)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Calculate total gravitational potential energy
    total_pe = 0
    G = 6.674e-11
    
    for i in range(len(bodies)):
        for j in range(i+1, len(bodies)):
            dx = bodies[i]['x'] - bodies[j]['x']
            dy = bodies[i]['y'] - bodies[j]['y']
            distance = math.sqrt(dx**2 + dy**2) * 1.5e11  # Convert AU to meters
            
            if distance > 0:
                pe = -G * bodies[i]['mass'] * bodies[j]['mass'] / distance
                total_pe += pe
    
    st.markdown(f"### ⚡ System Energy")
    st.markdown(f"**Total Gravitational Potential Energy:** {total_pe:.2e} J")

elif mode == "📊 Visualizations":
    st.markdown("# 📊 Advanced Visualizations")
    
    tab1, tab2, tab3 = st.tabs(["🌌 Gravity Fields", "📈 Orbital Dynamics", "⚫ Black Holes"])
    
    with tab1:
        st.markdown("### 🌌 Gravitational Field Visualization")
        
        # Create a gravitational field plot
        x = np.linspace(-10, 10, 20)
        y = np.linspace(-10, 10, 20)
        X, Y = np.meshgrid(x, y)
        
        # Central mass
        mass = 1e24
        G = 6.674e-11
        
        # Calculate gravitational field
        R = np.sqrt(X**2 + Y**2) + 0.1  # Avoid division by zero
        gx = -G * mass * X / R**3
        gy = -G * mass * Y / R**3
        
        fig = go.Figure()
        
        # Field vectors
        fig.add_trace(go.Scatter(
            x=X.flatten(), 
            y=Y.flatten(),
            mode='markers',
            marker=dict(size=2, color='white'),
            showlegend=False
        ))
        
        # Add quiver plot manually
        for i in range(0, len(x), 2):
            for j in range(0, len(y), 2):
                fig.add_trace(go.Scatter(
                    x=[X[i,j], X[i,j] + gx[i,j]*1e-22],
                    y=[Y[i,j], Y[i,j] + gy[i,j]*1e-22],
                    mode='lines',
                    line=dict(color='cyan', width=1),
                    showlegend=False
                ))
        
        # Central mass
        fig.add_trace(go.Scatter(
            x=[0], y=[0],
            mode='markers',
            marker=dict(size=20, color='yellow'),
            name='Central Mass'
        ))
        
        fig.update_layout(
            title="Gravitational Field Lines",
            xaxis_title="X Position",
            yaxis_title="Y Position",
            template="plotly_dark"
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.markdown("### 📈 Orbital Dynamics Analysis")
        
        # Kepler's laws demonstration
        st.markdown("#### Kepler's Third Law: T² ∝ a³")
        
        # Calculate periods for different orbital radii
        radii = np.linspace(1, 10, 50)
        earth_mass = 5.97e24
        earth_radius = 6.37e6
        G = 6.674e-11
        
        periods = []
        for r in radii:
            radius_m = r * earth_radius
            period = 2 * np.pi * np.sqrt(radius_m**3 / (G * earth_mass))
            periods.append(period / 3600)  # Convert to hours
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=radii, y=periods,
            mode='lines+markers',
            name='Orbital Period',
            line=dict(color='orange', width=3)
        ))
        
        fig.update_layout(
            title="Kepler's Third Law: Orbital Period vs Radius",
            xaxis_title="Orbital Radius (Earth radii)",
            yaxis_title="Period (hours)",
            template="plotly_dark"
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.markdown("### ⚫ Black Hole Visualization")
        
        # Schwarzschild radius calculator
        mass_bh = st.slider("Black Hole Mass (Solar masses)", 1.0, 100.0, 10.0)
        
        # Constants
        solar_mass = 1.989e30  # kg
        c = 299792458  # m/s
        G = 6.674e-11
        
        # Schwarzschild radius
        rs = 2 * G * (mass_bh * solar_mass) / c**2
        
        st.markdown(f"**Schwarzschild Radius:** {rs/1000:.2f} km")
        
        # Create visualization
        r = np.linspace(rs, rs*10, 100)
        
        # Gravitational time dilation
        time_dilation = 1 / np.sqrt(1 - rs/r)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=r/rs, y=time_dilation,
            mode='lines',
            name='Time Dilation Factor',
            line=dict(color='red', width=3)
        ))
        
        fig.update_layout(
            title="Gravitational Time Dilation near Black Hole",
            xaxis_title="Distance (Schwarzschild radii)",
            yaxis_title="Time Dilation Factor",
            template="plotly_dark"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("""
        **Note:** Time dilation approaches infinity as you approach the event horizon!
        At the Schwarzschild radius, time appears to stop for outside observers.
        """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #888; padding: 2rem;'>
    <p>🌌 <strong>Gravity Yonder Over</strong> - From falling apples to orbital slingshots</p>
    <p>Built with ❤️ for space enthusiasts and physics learners</p>
    <p>
        <a href='https://github.com/aditya13504/gravity-yonder-over' target='_blank'>GitHub</a> | 
        <a href='https://gravityyonder.streamlit.app' target='_blank'>Live Demo</a>
    </p>
</div>
""", unsafe_allow_html=True)
