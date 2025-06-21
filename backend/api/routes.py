from fastapi import APIRouter, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import numpy as np
import json
from datetime import datetime

from ..simulations.gravity_solver import GravitySolver
from ..simulations.modulus_physics_engine import ModulusPhysicsEngine
from ..models.celestial_body import CelestialBody
from ..visualizations.plotly_graphs import GravityVisualizer

# Create API router
api_router = APIRouter(prefix="/api", tags=["gravity"])

# Initialize the NVIDIA Modulus Physics Engine
physics_engine = ModulusPhysicsEngine()

# Pydantic models for request/response validation
class SimulationRequest(BaseModel):
    bodies: List[Dict[str, Any]]
    duration: float
    time_step: float
    solver_type: str = "runge_kutta"
    include_relativistic: bool = False

class SimulationResponse(BaseModel):
    simulation_id: str
    status: str
    results: List[Dict[str, Any]]
    metadata: Dict[str, Any]

class TrajectoryRequest(BaseModel):
    mass_central: float
    mass_orbiting: float
    initial_distance: float
    initial_velocity: float
    eccentricity: Optional[float] = None

class GravityFieldRequest(BaseModel):
    bodies: List[Dict[str, Any]]
    grid_size: int = 50
    field_range: Dict[str, float]

class OrbitCalculationRequest(BaseModel):
    central_mass: float
    orbiting_mass: float
    semi_major_axis: float
    eccentricity: float

class RelativisticRequest(BaseModel):
    mass: float
    velocity: float
    distance: float
    effect_type: str  # "time_dilation", "redshift", "precession"

# Educational Game Request/Response Models
class AppleDropRequest(BaseModel):
    height: float
    gravity: float = 9.81
    time_steps: int = 100

class OrbitalSlingshotRequest(BaseModel):
    planet_mass: float
    planet_radius: float
    approach_velocity: float
    approach_angle: float
    closest_approach_factor: float = 2.0

class LagrangePointRequest(BaseModel):
    m1_mass: float
    m2_mass: float
    separation: float
    test_mass_pos: List[float]
    time_span: float = 86400

class EscapeVelocityRequest(BaseModel):
    planet_mass: float
    planet_radius: float
    launch_angle: float
    launch_velocity: float

class BlackHoleNavigationRequest(BaseModel):
    black_hole_mass: float
    approach_trajectory: List[List[float]]
    navigation_commands: List[Dict[str, float]]

class GameResponse(BaseModel):
    success: bool
    data: Dict[str, Any]
    educational_notes: List[str]
    score: int

# Initialize physics components
gravity_solver = GravitySolver()
visualizer = GravityVisualizer()

@api_router.post("/simulate", response_model=SimulationResponse)
async def run_simulation(request: SimulationRequest):
    """
    Run a gravity simulation with specified parameters
    """
    try:
        # Convert request data to CelestialBody objects
        bodies = []
        for body_data in request.bodies:
            body = CelestialBody(
                name=body_data.get("name", "Unnamed"),
                mass=body_data["mass"],
                position=body_data["position"],
                velocity=body_data["velocity"]
            )
            bodies.append(body)
        
        # Run the simulation
        if request.include_relativistic:
            results = gravity_solver.simulate_relativistic_n_body(
                bodies, request.duration, request.time_step
            )
        else:
            results = gravity_solver.simulate_n_body(
                bodies, request.duration, request.time_step
            )
        
        # Generate simulation ID
        simulation_id = f"sim_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Prepare response
        response = SimulationResponse(
            simulation_id=simulation_id,
            status="completed",
            results=results,
            metadata={
                "duration": request.duration,
                "time_step": request.time_step,
                "num_bodies": len(bodies),
                "solver_type": request.solver_type,
                "relativistic": request.include_relativistic
            }
        )
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Simulation failed: {str(e)}")

@api_router.get("/trajectories/{scenario_id}")
async def get_trajectory(scenario_id: str):
    """
    Get pre-computed trajectory data for a specific scenario
    """
    try:
        # This would load from pre-computed data storage
        # For now, return a placeholder
        trajectory_data = {
            "scenario_id": scenario_id,
            "trajectory": [
                {"t": i * 0.1, "x": np.cos(i * 0.1), "y": np.sin(i * 0.1)}
                for i in range(100)
            ],
            "metadata": {
                "orbital_period": 2 * np.pi,
                "eccentricity": 0.0,
                "semi_major_axis": 1.0
            }
        }
        
        return trajectory_data
        
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Trajectory not found: {str(e)}")

@api_router.get("/gravity-field")
async def get_gravity_field(request: GravityFieldRequest):
    """
    Calculate gravitational field strength at grid points
    """
    try:
        # Create grid
        x_range = np.linspace(
            request.field_range["x_min"], 
            request.field_range["x_max"], 
            request.grid_size
        )
        y_range = np.linspace(
            request.field_range["y_min"], 
            request.field_range["y_max"], 
            request.grid_size
        )
        
        # Calculate field at each grid point
        field_data = {
            "x_range": x_range.tolist(),
            "y_range": y_range.tolist(),
            "fx": [],
            "fy": [],
            "magnitude": []
        }
        
        for body_data in request.bodies:
            body = CelestialBody(
                name=body_data["name"],
                mass=body_data["mass"],
                position=body_data["position"],
                velocity=[0, 0, 0]
            )
        
        # Calculate field (simplified implementation)
        G = 6.67430e-11
        fx_grid = np.zeros((request.grid_size, request.grid_size))
        fy_grid = np.zeros((request.grid_size, request.grid_size))
        
        for i, x in enumerate(x_range):
            for j, y in enumerate(y_range):
                fx_total = 0
                fy_total = 0
                
                for body_data in request.bodies:
                    dx = x - body_data["position"][0]
                    dy = y - body_data["position"][1]
                    r = np.sqrt(dx**2 + dy**2)
                    
                    if r > 0:
                        force_mag = G * body_data["mass"] / r**2
                        fx_total -= force_mag * dx / r
                        fy_total -= force_mag * dy / r
                
                fx_grid[j, i] = fx_total
                fy_grid[j, i] = fy_total
        
        field_data["fx"] = fx_grid.tolist()
        field_data["fy"] = fy_grid.tolist()
        field_data["magnitude"] = np.sqrt(fx_grid**2 + fy_grid**2).tolist()
        
        return field_data
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Field calculation failed: {str(e)}")

@api_router.post("/orbit-calculation")
async def calculate_orbit(request: OrbitCalculationRequest):
    """
    Calculate orbital parameters and characteristics
    """
    try:
        G = 6.67430e-11
        
        # Calculate orbital velocity
        orbital_velocity = np.sqrt(G * request.central_mass / request.semi_major_axis)
        
        # Calculate orbital period
        orbital_period = 2 * np.pi * np.sqrt(
            request.semi_major_axis**3 / (G * request.central_mass)
        )
        
        # Calculate escape velocity
        escape_velocity = np.sqrt(2 * G * request.central_mass / request.semi_major_axis)
        
        # Calculate aphelion and perihelion
        aphelion = request.semi_major_axis * (1 + request.eccentricity)
        perihelion = request.semi_major_axis * (1 - request.eccentricity)
        
        orbit_data = {
            "orbital_velocity": orbital_velocity,
            "orbital_period": orbital_period,
            "escape_velocity": escape_velocity,
            "aphelion": aphelion,
            "perihelion": perihelion,
            "semi_major_axis": request.semi_major_axis,
            "eccentricity": request.eccentricity,
            "is_bound": orbital_velocity < escape_velocity
        }
        
        return orbit_data
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Orbit calculation failed: {str(e)}")

@api_router.post("/relativistic-effects")
async def get_relativistic_effects(request: RelativisticRequest):
    """
    Calculate relativistic effects for given parameters
    """
    try:
        c = 299792458  # Speed of light
        G = 6.67430e-11  # Gravitational constant
        
        effects = {}
        
        if request.effect_type == "time_dilation":
            # Special relativistic time dilation
            gamma = 1 / np.sqrt(1 - (request.velocity / c)**2)
            effects["lorentz_factor"] = gamma
            effects["time_dilation_factor"] = 1 / gamma
            
            # Gravitational time dilation
            rs = 2 * G * request.mass / c**2  # Schwarzschild radius
            gravitational_factor = np.sqrt(1 - rs / request.distance)
            effects["gravitational_time_dilation"] = gravitational_factor
            
        elif request.effect_type == "redshift":
            rs = 2 * G * request.mass / c**2
            redshift = np.sqrt(rs / request.distance)
            effects["gravitational_redshift"] = redshift
            
        elif request.effect_type == "precession":
            # Orbital precession (simplified)
            rs = 2 * G * request.mass / c**2
            precession_per_orbit = 6 * np.pi * rs / request.distance
            effects["precession_per_orbit"] = precession_per_orbit
            effects["precession_degrees"] = np.degrees(precession_per_orbit)
        
        return effects
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Relativistic calculation failed: {str(e)}")

@api_router.post("/validate-parameters")
async def validate_parameters(parameters: Dict[str, Any]):
    """
    Validate physics parameters for simulation
    """
    try:
        validation_result = {
            "is_valid": True,
            "errors": [],
            "warnings": []
        }
        
        # Validate masses
        for key, value in parameters.items():
            if "mass" in key.lower():
                if value <= 0:
                    validation_result["is_valid"] = False
                    validation_result["errors"].append(f"{key} must be positive")
                elif value > 1e50:
                    validation_result["warnings"].append(f"{key} is extremely large")
            
            elif "velocity" in key.lower():
                c = 299792458
                if abs(value) >= c:
                    validation_result["is_valid"] = False
                    validation_result["errors"].append(f"{key} cannot exceed speed of light")
                elif abs(value) > 0.1 * c:
                    validation_result["warnings"].append(f"{key} is relativistic")
            
            elif "distance" in key.lower():
                if value <= 0:
                    validation_result["is_valid"] = False
                    validation_result["errors"].append(f"{key} must be positive")
        
        return validation_result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Validation failed: {str(e)}")

@api_router.get("/education/{topic_id}")
async def get_educational_content(topic_id: str):
    """
    Get educational content for a specific physics topic
    """
    try:
        # This would load from a content database
        content = {
            "topic_id": topic_id,
            "title": f"Physics Topic: {topic_id}",
            "content": "Educational content would be loaded here",
            "examples": [],
            "exercises": []
        }
        
        return content
        
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Educational content not found: {str(e)}")

@api_router.post("/scores")
async def submit_score(score_data: Dict[str, Any]):
    """
    Submit game score and achievements
    """
    try:
        # This would save to a database
        response = {
            "status": "success",
            "score_id": f"score_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "rank": 42  # Placeholder rank
        }
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Score submission failed: {str(e)}")

@api_router.get("/leaderboard/{game_id}")
async def get_leaderboard(game_id: str):
    """
    Get leaderboard for a specific game
    """
    try:
        # This would load from a database
        leaderboard = {
            "game_id": game_id,
            "top_scores": [
                {"rank": 1, "player": "Player1", "score": 1000},
                {"rank": 2, "player": "Player2", "score": 950},
                {"rank": 3, "player": "Player3", "score": 900}
            ]
        }
        
        return leaderboard
        
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Leaderboard not found: {str(e)}")

@api_router.get("/presets")
async def get_simulation_presets():
    """
    Get available simulation presets
    """
    try:
        presets = {
            "two_body": {
                "name": "Earth-Moon System",
                "description": "Simulate Earth-Moon gravitational interaction",
                "parameters": {
                    "bodies": [
                        {
                            "name": "Earth",
                            "mass": 5.972e24,
                            "position": [0, 0, 0],
                            "velocity": [0, 0, 0]
                        },
                        {
                            "name": "Moon",
                            "mass": 7.342e22,
                            "position": [3.844e8, 0, 0],
                            "velocity": [0, 1022, 0]
                        }
                    ]
                }
            },
            "solar_system": {
                "name": "Inner Solar System",
                "description": "Simulate inner planets around the Sun",
                "parameters": {
                    "bodies": [
                        # Sun, Mercury, Venus, Earth, Mars data would go here
                    ]
                }
            }
        }
        
        return presets
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load presets: {str(e)}")

# =============================================================================
# EDUCATIONAL GAME ENDPOINTS - REAL-TIME NVIDIA MODULUS SIMULATIONS
# =============================================================================

@api_router.post("/games/apple-drop", response_model=GameResponse)
async def simulate_apple_drop_game(request: AppleDropRequest):
    """
    🍎 Real-time apple drop simulation for educational physics game
    Uses NVIDIA Modulus for accurate gravitational acceleration
    """
    try:
        result = physics_engine.simulate_apple_drop_game(
            height=request.height,
            gravity=request.gravity,
            time_steps=request.time_steps
        )
        
        # Educational insights
        educational_notes = [
            f"At {request.height}m height, the apple takes {result['times'][-1]:.2f} seconds to fall",
            f"Final velocity: {result['velocities'][-1]:.2f} m/s",
            f"Kinetic energy at impact: {0.5 * 0.2 * result['velocities'][-1]**2:.2f} J",
            "This follows the kinematic equation: h = ½gt²"
        ]
        
        return GameResponse(
            success=True,
            data=result,
            educational_notes=educational_notes,
            score=max(10, int(request.height))
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Apple drop simulation failed: {str(e)}")

@api_router.post("/games/orbital-slingshot", response_model=GameResponse)
async def orbital_slingshot_game(request: OrbitalSlingshotRequest):
    """🚀 Orbital Slingshot Game - Use gravity assists to gain velocity"""
    try:
        result = physics_engine.simulate_orbital_slingshot(
            request.planet_mass,
            request.planet_radius,
            request.approach_velocity,
            request.approach_angle,
            request.closest_approach_factor
        )
        
        educational_notes = [
            "Gravity assists use a planet's gravity to change spacecraft velocity without fuel",
            "The Voyager missions used multiple gravity assists to reach the outer solar system",
            "Timing and approach angle are critical for successful gravity assists"
        ]
        
        return GameResponse(
            success=True,
            data=result,
            educational_notes=educational_notes,
            score=max(0, int(result.get("velocity_gain", 0) / 1000))
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@api_router.post("/games/lagrange-points", response_model=GameResponse)
async def lagrange_points_game(request: LagrangePointRequest):
    """🌍 Lagrange Points Game - Explore gravitational equilibrium points"""
    try:
        result = physics_engine.simulate_lagrange_points(
            request.m1_mass,
            request.m2_mass,
            request.separation,
            request.test_mass_pos,
            request.time_span
        )
        
        educational_notes = [
            "Lagrange points are positions where gravitational forces balance",
            "The James Webb Space Telescope operates at the L2 Lagrange point",
            "There are 5 Lagrange points in any two-body gravitational system"
        ]
        
        return GameResponse(
            success=True,
            data=result,
            educational_notes=educational_notes,
            score=result.get("stability_score", 50)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@api_router.post("/games/escape-velocity", response_model=GameResponse)
async def escape_velocity_game(request: EscapeVelocityRequest):
    """🚀 Escape Velocity Game - Launch rockets to escape planetary gravity"""
    try:
        result = physics_engine.simulate_escape_velocity(
            request.planet_mass,
            request.planet_radius,
            request.launch_angle,
            request.launch_velocity
        )
        
        educational_notes = [
            "Escape velocity depends on the planet's mass and radius",
            "It's the minimum velocity needed to escape gravitational attraction",
            "Real rockets use multiple stages to achieve escape velocity efficiently"
        ]
        
        return GameResponse(
            success=True,
            data=result,
            educational_notes=educational_notes,
            score=100 if result.get("escaped", False) else 0
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@api_router.post("/games/black-hole-navigation", response_model=GameResponse)
async def black_hole_navigation_game(request: BlackHoleNavigationRequest):
    """⚫ Black Hole Navigation Game - Navigate through curved spacetime"""
    try:
        result = physics_engine.simulate_black_hole_navigation(
            request.black_hole_mass,
            request.approach_trajectory,
            request.navigation_commands
        )
        
        educational_notes = [
            "Black holes curve spacetime, affecting the path of light and matter",
            "Time dilation near black holes means time passes differently",
            "The event horizon is the point of no return for black holes"
        ]
        
        return GameResponse(
            success=True,
            data=result,
            educational_notes=educational_notes,
            score=result.get("navigation_score", 0)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/models/info")
async def get_models_info():
    """Get information about available ML models and physics engine status"""
    try:
        return physics_engine.get_available_models()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Health check endpoint
@api_router.get("/health")
async def health_check():
    """Health check endpoint for monitoring"""
    return {
        "status": "healthy",
        "message": "Gravity Yonder Over API is running",
        "timestamp": datetime.now().isoformat(),
        "gpu_available": physics_engine.device.type == "cuda" if hasattr(physics_engine, 'device') else False
    }
