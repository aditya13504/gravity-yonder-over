"""
🌌 Gravity Yonder Over - Beginner-Friendly Interactive Features
Making physics accessible and fun for all ages and skill levels
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd
from typing import Dict, List, Any
import time
import random

def create_beginner_dashboard():
    """Create a beginner-friendly main dashboard"""
    st.markdown("# 🌟 Welcome to Gravity Adventures!")
    st.markdown("*Discover the invisible force that shapes our universe*")
    
    # Progress tracking
    if 'user_progress' not in st.session_state:
        st.session_state.user_progress = {
            'completed_activities': [],
            'current_level': 1,
            'points': 0,
            'badges': []
        }
    
    # Display progress
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("🎯 Level", st.session_state.user_progress['current_level'])
    with col2:
        st.metric("⭐ Points", st.session_state.user_progress['points'])
    with col3:
        st.metric("🏆 Badges", len(st.session_state.user_progress['badges']))
    with col4:
        st.metric("✅ Completed", len(st.session_state.user_progress['completed_activities']))
    
    # Learning path selection
    st.markdown("## 🚀 Choose Your Learning Adventure!")
    
    learning_paths = {
        "🍎 Gravity Basics": {
            "description": "Start here! Learn what gravity is and why things fall down.",
            "time": "15-20 minutes",
            "difficulty": "Easy",
            "activities": ["Apple Drop", "Weight on Different Planets", "Gravity Quiz"]
        },
        "🛰️ Space Exploration": {
            "description": "Discover how satellites stay in orbit and explore the solar system.",
            "time": "25-30 minutes", 
            "difficulty": "Medium",
            "activities": ["Satellite Designer", "Planet Hopper", "Orbit Game"]
        },
        "⚫ Black Hole Adventure": {
            "description": "Journey to the most extreme places in the universe!",
            "time": "30-35 minutes",
            "difficulty": "Medium-Hard",
            "activities": ["Black Hole Explorer", "Time Traveler", "Escape Challenge"]
        },
        "🌌 Einstein's Universe": {
            "description": "Explore the weird and wonderful world of relativity.",
            "time": "35-40 minutes",
            "difficulty": "Hard",
            "activities": ["Time Dilation Lab", "GPS Mission", "Twin Paradox"]
        }
    }
    
    for path_name, path_info in learning_paths.items():
        with st.expander(f"{path_name} - {path_info['difficulty']}", expanded=False):
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.write(path_info['description'])
                st.write(f"⏱️ **Time needed:** {path_info['time']}")
                st.write(f"🎯 **Activities:** {', '.join(path_info['activities'])}")
            
            with col2:
                if st.button(f"Start {path_name}!", key=f"start_{path_name}"):
                    st.session_state.current_path = path_name
                    st.rerun()

def create_interactive_apple_drop():
    """Create an interactive apple drop experiment"""
    st.markdown("# 🍎 The Great Apple Drop Experiment")
    st.markdown("*Discover gravity just like Isaac Newton did!*")
    
    # Educational introduction
    with st.expander("📖 The Story Behind the Apple", expanded=True):
        st.markdown("""
        **Did you know?** Isaac Newton probably never got hit by an apple, but he did observe 
        apples falling from trees. This simple observation led to one of the most important 
        discoveries in science!
        
        **What you'll discover:**
        - Why do things fall down instead of up?
        - Do heavy things fall faster than light things?
        - How does gravity work on different planets?
        """)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### 🎛️ Experiment Controls")
        
        # Object selection
        objects = {
            "🍎 Apple": {"mass": 0.18, "emoji": "🍎"},
            "🽃 Feather": {"mass": 0.001, "emoji": "🪶"},
            "⚽ Soccer Ball": {"mass": 0.43, "emoji": "⚽"},
            "🔨 Hammer": {"mass": 0.5, "emoji": "🔨"},
            "🐘 Elephant": {"mass": 5000, "emoji": "🐘"}
        }
        
        selected_object = st.selectbox("Choose what to drop:", list(objects.keys()))
        
        # Planet selection
        planets = {
            "🌍 Earth": {"gravity": 9.81, "emoji": "🌍"},
            "🌙 Moon": {"gravity": 1.62, "emoji": "🌙"},
            "♀️ Venus": {"gravity": 8.87, "emoji": "♀️"},
            "♂️ Mars": {"gravity": 3.71, "emoji": "♂️"},
            "♃ Jupiter": {"gravity": 24.79, "emoji": "♃"}
        }
        
        selected_planet = st.selectbox("Choose a planet:", list(planets.keys()))
        
        # Drop height
        height = st.slider("Drop height (meters)", 1, 100, 10)
        
        # Calculate physics
        g = planets[selected_planet]["gravity"]
        mass = objects[selected_object]["mass"]
        
        # Time to fall (ignoring air resistance)
        time_to_fall = np.sqrt(2 * height / g)
        final_velocity = g * time_to_fall
        
        st.markdown("### 📊 Predictions")
        st.metric("Time to fall", f"{time_to_fall:.2f} seconds")
        st.metric("Final speed", f"{final_velocity:.1f} m/s")
        st.metric("Weight", f"{mass * g:.1f} N")
        
        if st.button("🚀 DROP IT!", key="drop_button"):
            st.session_state.dropping = True
            st.session_state.drop_start_time = time.time()
    
    with col2:
        # Create animation
        if 'dropping' in st.session_state and st.session_state.dropping:
            create_drop_animation(height, time_to_fall, objects[selected_object], planets[selected_planet])
        else:
            create_static_drop_setup(height, objects[selected_object], planets[selected_planet])
    
    # Fun facts and learning points
    st.markdown("---")
    st.markdown("## 🤔 Did You Know?")
    
    facts = [
        "🪶 On the Moon, a feather and a hammer fall at exactly the same speed!",
        "🌍 Everything falls at the same rate on Earth (ignoring air resistance).",
        "⚖️ Your weight changes on different planets, but your mass stays the same.",
        "🚀 Astronauts float in space because they're constantly falling around Earth!"
    ]
    
    for fact in facts:
        st.info(fact)

def create_drop_animation(height, fall_time, obj_info, planet_info):
    """Create animated visualization of dropping object"""
    current_time = time.time()
    
    if 'drop_start_time' not in st.session_state:
        st.session_state.drop_start_time = current_time
    
    elapsed = current_time - st.session_state.drop_start_time
    
    if elapsed < fall_time:
        # Object is still falling
        g = planet_info["gravity"]
        current_height = height - 0.5 * g * elapsed**2
        current_height = max(0, current_height)
        
        # Create visualization
        fig = go.Figure()
        
        # Ground
        fig.add_shape(
            type="rect",
            x0=-2, y0=-1, x1=2, y1=0,
            fillcolor="brown",
            line=dict(color="brown")
        )
        
        # Object
        fig.add_trace(go.Scatter(
            x=[0], y=[current_height],
            mode='markers+text',
            marker=dict(size=30, color="red"),
            text=[obj_info["emoji"]],
            textfont=dict(size=20),
            showlegend=False
        ))
        
        # Velocity arrow
        velocity = g * elapsed
        if velocity > 0:
            fig.add_annotation(
                x=0.5, y=current_height,
                ax=0.5, ay=current_height - velocity/10,
                xref="x", yref="y",
                axref="x", ayref="y",
                text=f"v = {velocity:.1f} m/s",
                showarrow=True,
                arrowhead=2,
                arrowcolor="blue"
            )
        
        fig.update_layout(
            title=f"Dropping {obj_info['emoji']} on {planet_info['emoji']}",
            xaxis_range=[-3, 3],
            yaxis_range=[-2, height+5],
            height=500,
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Auto-refresh
        time.sleep(0.1)
        st.rerun()
        
    else:
        # Object has landed
        st.session_state.dropping = False
        st.success(f"🎯 LANDED! The {obj_info['emoji']} took {fall_time:.2f} seconds to fall {height} meters!")
        
        # Award points
        if 'apple_drop_completed' not in st.session_state.user_progress['completed_activities']:
            st.session_state.user_progress['completed_activities'].append('apple_drop_completed')
            st.session_state.user_progress['points'] += 10
            st.balloons()

def create_static_drop_setup(height, obj_info, planet_info):
    """Create static visualization before dropping"""
    fig = go.Figure()
    
    # Ground
    fig.add_shape(
        type="rect",
        x0=-2, y0=-1, x1=2, y1=0,
        fillcolor="brown",
        line=dict(color="brown")
    )
    
    # Object at starting position
    fig.add_trace(go.Scatter(
        x=[0], y=[height],
        mode='markers+text',
        marker=dict(size=30, color="red"),
        text=[obj_info["emoji"]],
        textfont=dict(size=20),
        showlegend=False
    ))
    
    # Height indicator
    fig.add_annotation(
        x=-1, y=height/2,
        text=f"{height} m",
        showarrow=True,
        arrowhead=2,
        arrowcolor="green"
    )
    
    fig.update_layout(
        title=f"Ready to drop {obj_info['emoji']} on {planet_info['emoji']}",
        xaxis_range=[-3, 3],
        yaxis_range=[-2, height+5],
        height=500,
        showlegend=False
    )
    
    st.plotly_chart(fig, use_container_width=True)

def create_planet_weight_calculator():
    """Create interactive planet weight calculator"""
    st.markdown("# ⚖️ Your Weight Across the Solar System")
    st.markdown("*Ever wondered how much you'd weigh on Mars?*")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### 👤 About You")
        weight_earth = st.number_input("Your weight on Earth (kg)", 10, 200, 70)
        
        st.markdown("### 🌍 Planetary Data")
        
        planets = {
            "Mercury": {"gravity": 3.7, "emoji": "☿️", "color": "gray"},
            "Venus": {"gravity": 8.87, "emoji": "♀️", "color": "orange"},
            "Earth": {"gravity": 9.81, "emoji": "🌍", "color": "blue"},
            "Mars": {"gravity": 3.71, "emoji": "♂️", "color": "red"},
            "Jupiter": {"gravity": 24.79, "emoji": "♃", "color": "brown"},
            "Saturn": {"gravity": 10.44, "emoji": "🪐", "color": "gold"},
            "Uranus": {"gravity": 8.69, "emoji": "🌌", "color": "cyan"},
            "Neptune": {"gravity": 11.15, "emoji": "🔵", "color": "darkblue"},
            "Moon": {"gravity": 1.62, "emoji": "🌙", "color": "lightgray"},
            "Sun": {"gravity": 274.0, "emoji": "☀️", "color": "yellow"}
        }
        
        selected_planet = st.selectbox("Choose a celestial body:", list(planets.keys()))
        
        # Calculate weight
        earth_gravity = 9.81
        mass = weight_earth  # kg
        new_weight = mass * planets[selected_planet]["gravity"]
        
        st.markdown("### 📊 Results")
        st.metric("Your weight", f"{new_weight:.1f} kg")
        
        weight_ratio = new_weight / weight_earth
        if weight_ratio > 1:
            st.metric("Weight change", f"{weight_ratio:.1f}x HEAVIER")
        else:
            st.metric("Weight change", f"{1/weight_ratio:.1f}x LIGHTER")
    
    with col2:
        # Create comparison visualization
        planet_names = list(planets.keys())
        weights = [mass * planets[planet]["gravity"] for planet in planet_names]
        colors = [planets[planet]["color"] for planet in planet_names]
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=planet_names,
            y=weights,
            marker_color=colors,
            text=[f"{w:.1f} kg" for w in weights],
            textposition='auto'
        ))
        
        fig.update_layout(
            title="Your Weight Across the Solar System",
            xaxis_title="Celestial Body",
            yaxis_title="Weight (kg)",
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Fun facts
    st.markdown("---")
    st.markdown("## 🚀 Amazing Weight Facts!")
    
    facts = [
        f"🌙 On the Moon, you'd weigh only {mass * 1.62:.1f} kg - you could jump 6 times higher!",
        f"♃ On Jupiter, you'd weigh {mass * 24.79:.1f} kg - it would be hard to even stand up!",
        f"☀️ On the Sun, you'd weigh {mass * 274:.0f} kg - that's like carrying {int(mass * 274 / 70)} people!",
        f"♂️ On Mars, you'd weigh {mass * 3.71:.1f} kg - perfect for exploring the red planet!"
    ]
    
    for fact in facts:
        st.info(fact)

def create_simple_quiz_system():
    """Create beginner-friendly quiz system"""
    st.markdown("# 🧠 Gravity Knowledge Check")
    st.markdown("*Test what you've learned about gravity!*")
    
    # Quiz questions with different difficulty levels
    quiz_questions = {
        "easy": [
            {
                "question": "What makes things fall down to Earth?",
                "options": ["Gravity", "Magnetism", "Wind", "Electricity"],
                "correct": 0,
                "explanation": "Gravity is the force that pulls all objects with mass toward each other!"
            },
            {
                "question": "Where is gravity stronger?",
                "options": ["On the Moon", "On Earth", "In space", "They're all the same"],
                "correct": 1,
                "explanation": "Earth has more mass than the Moon, so its gravity is stronger!"
            },
            {
                "question": "What would happen if you dropped a feather and a hammer on the Moon?",
                "options": ["Feather falls first", "Hammer falls first", "They fall together", "They float"],
                "correct": 2,
                "explanation": "Without air resistance, all objects fall at the same rate!"
            }
        ],
        "medium": [
            {
                "question": "Why don't satellites fall to Earth?",
                "options": ["No gravity in space", "They're moving too fast sideways", "They're too light", "Rockets push them up"],
                "correct": 1,
                "explanation": "Satellites are constantly falling, but they're moving so fast sideways that they keep missing Earth!"
            },
            {
                "question": "What happens to your weight on different planets?",
                "options": ["It stays the same", "It changes based on planet's gravity", "It always increases", "It disappears"],
                "correct": 1,
                "explanation": "Weight = mass × gravity, so different planet gravity means different weight!"
            }
        ]
    }
    
    # Quiz state management
    if 'quiz_started' not in st.session_state:
        st.session_state.quiz_started = False
        st.session_state.current_question = 0
        st.session_state.score = 0
        st.session_state.answers = []
    
    # Difficulty selection
    if not st.session_state.quiz_started:
        st.markdown("## 🎯 Choose Your Challenge Level")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🟢 Easy Quiz", key="easy_quiz"):
                st.session_state.quiz_questions = quiz_questions["easy"]
                st.session_state.quiz_started = True
                st.rerun()
        
        with col2:
            if st.button("🟡 Medium Quiz", key="medium_quiz"):
                st.session_state.quiz_questions = quiz_questions["medium"]
                st.session_state.quiz_started = True
                st.rerun()
    
    else:
        # Quiz in progress
        questions = st.session_state.quiz_questions
        current_q = st.session_state.current_question
        
        if current_q < len(questions):
            question = questions[current_q]
            
            st.markdown(f"## Question {current_q + 1} of {len(questions)}")
            st.markdown(f"### {question['question']}")
            
            # Answer options
            selected_answer = st.radio(
                "Choose your answer:",
                options=question['options'],
                key=f"q_{current_q}"
            )
            
            if st.button("Submit Answer", key=f"submit_{current_q}"):
                selected_index = question['options'].index(selected_answer)
                st.session_state.answers.append(selected_index)
                
                if selected_index == question['correct']:
                    st.success("✅ Correct!")
                    st.session_state.score += 1
                else:
                    st.error("❌ Not quite right.")
                
                st.info(f"💡 **Explanation:** {question['explanation']}")
                
                st.session_state.current_question += 1
                
                if st.session_state.current_question < len(questions):
                    if st.button("Next Question"):
                        st.rerun()
                else:
                    if st.button("See Results"):
                        st.rerun()
        
        else:
            # Quiz completed
            st.markdown("## 🎉 Quiz Complete!")
            
            score_percentage = (st.session_state.score / len(questions)) * 100
            
            st.metric("Your Score", f"{st.session_state.score}/{len(questions)}")
            st.metric("Percentage", f"{score_percentage:.0f}%")
            
            if score_percentage >= 80:
                st.success("🌟 Excellent! You're a gravity expert!")
                st.balloons()
            elif score_percentage >= 60:
                st.info("👍 Good job! You understand the basics of gravity.")
            else:
                st.warning("🤔 Keep learning! Try the activities again to improve your understanding.")
            
            # Award points and badges
            if score_percentage >= 80 and 'quiz_master' not in st.session_state.user_progress['badges']:
                st.session_state.user_progress['badges'].append('quiz_master')
                st.session_state.user_progress['points'] += 25
                st.success("🏆 New Badge: Quiz Master!")
            
            if st.button("Take Quiz Again"):
                st.session_state.quiz_started = False
                st.session_state.current_question = 0
                st.session_state.score = 0
                st.session_state.answers = []
                st.rerun()

def main():
    """Main function for beginner-friendly features"""
    st.set_page_config(page_title="Gravity Adventures", page_icon="🌟", layout="wide")
    
    # Navigation
    page = st.sidebar.selectbox("Choose Your Adventure:", [
        "🏠 Home Dashboard",
        "🍎 Apple Drop Experiment", 
        "⚖️ Planet Weight Calculator",
        "🧠 Gravity Quiz"
    ])
    
    if page == "🏠 Home Dashboard":
        create_beginner_dashboard()
    elif page == "🍎 Apple Drop Experiment":
        create_interactive_apple_drop()
    elif page == "⚖️ Planet Weight Calculator":
        create_planet_weight_calculator()
    elif page == "🧠 Gravity Quiz":
        create_simple_quiz_system()

if __name__ == "__main__":
    main()
