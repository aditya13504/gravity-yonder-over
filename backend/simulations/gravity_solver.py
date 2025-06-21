import numpy as np
from typing import List, Tuple, Dict, Optional
from ..models.celestial_body import CelestialBody, CelestialSystem
import scipy.integrate as integrate
from numba import jit, njit
import json

class GravitySolver:
    """Core physics engine for gravitational simulations"""
    
    def __init__(self, G: float = 6.67430e-11):
        self.G = G
        self.c = 299792458  # Speed of light for relativistic corrections
        
    def calculate_acceleration(self, body: CelestialBody, others: List[CelestialBody]) -> np.ndarray:
        """Calculate net gravitational acceleration on a body"""
        acceleration = np.zeros(2)
        
        for other in others:
            if other.name != body.name:
                force = body.gravitational_force_from(other, self.G)
                acceleration += force / body.mass
        
        return acceleration
    
    @staticmethod
    @njit
    def _calculate_forces_vectorized(positions: np.ndarray, masses: np.ndarray, G: float) -> np.ndarray:
        """Optimized force calculation using NumPy"""
        n = len(masses)
        forces = np.zeros((n, 2))
        
        for i in range(n):
            for j in range(i + 1, n):
                r_vec = positions[j] - positions[i]
                r = np.sqrt(r_vec[0]**2 + r_vec[1]**2)
                
                if r > 0:
                    force_mag = G * masses[i] * masses[j] / r**2
                    force_dir = r_vec / r
                    force = force_mag * force_dir
                    
                    forces[i] += force
                    forces[j] -= force
        
        return forces
    
    def simulate_system(self, bodies: List[CelestialBody], dt: float, steps: int) -> Dict[str, np.ndarray]:
        """Simulate the system for a given number of steps"""
        n_bodies = len(bodies)
        trajectories = {body.name: np.zeros((steps, 2)) for body in bodies}
        velocities = {body.name: np.zeros((steps, 2)) for body in bodies}
        
        # Convert to arrays for vectorized operations
        positions = np.array([body.position for body in bodies])
        velocities_arr = np.array([body.velocity for body in bodies])
        masses = np.array([body.mass for body in bodies])
        
        for step in range(steps):
            # Store current positions
            for i, body in enumerate(bodies):
                trajectories[body.name][step] = positions[i].copy()
                velocities[body.name][step] = velocities_arr[i].copy()
            
            # Calculate forces and accelerations
            forces = self._calculate_forces_vectorized(positions, masses, self.G)
            accelerations = forces / masses[:, np.newaxis]
            
            # Update velocities and positions (Verlet integration)
            velocities_arr += accelerations * dt
            positions += velocities_arr * dt
        
        return trajectories
    
    def calculate_orbit_parameters(self, body: CelestialBody, central_body: CelestialBody) -> Dict[str, float]:
        """Calculate orbital parameters for a body around a central body"""
        # Relative position and velocity
        r_vec = body.position - central_body.position
        v_vec = body.velocity - central_body.velocity
        r = np.linalg.norm(r_vec)
        v = np.linalg.norm(v_vec)
        
        # Standard gravitational parameter
        mu = self.G * (body.mass + central_body.mass)
        
        # Specific orbital energy
        energy = v**2 / 2 - mu / r
        
        # Semi-major axis
        a = -mu / (2 * energy) if energy < 0 else float('inf')
        
        # Eccentricity vector
        h = np.cross(r_vec, v_vec)  # Specific angular momentum
        e_vec = np.cross(v_vec, h) / mu - r_vec / r
        e = np.linalg.norm(e_vec)
        
        # Period (if bound orbit)
        period = 2 * np.pi * np.sqrt(a**3 / mu) if a > 0 and a != float('inf') else float('inf')
        
        # Periapsis and apoapsis
        periapsis = a * (1 - e) if a != float('inf') else r
        apoapsis = a * (1 + e) if a != float('inf') else float('inf')
        
        return {
            'semi_major_axis': a,
            'eccentricity': e,
            'period': period,
            'periapsis': periapsis,
            'apoapsis': apoapsis,
            'specific_energy': energy,
            'angular_momentum': abs(h) if isinstance(h, (int, float)) else np.linalg.norm(h)
        }
    
    def calculate_slingshot(self, planet: CelestialBody, v_initial: float, approach_angle: float) -> np.ndarray:
        """Calculate gravitational slingshot trajectory"""
        # Simplified 2D slingshot calculation
        # Set up initial conditions
        r_initial = planet.radius * 10  # Start far from planet
        x0 = r_initial * np.cos(approach_angle)
        y0 = r_initial * np.sin(approach_angle)
        
        # Initial velocity perpendicular to position
        vx0 = -v_initial * np.sin(approach_angle)
        vy0 = v_initial * np.cos(approach_angle)
        
        # Define ODE system
        def dynamics(t, state):
            x, y, vx, vy = state
            r = np.sqrt(x**2 + y**2)
            
            if r < planet.radius:
                return [0, 0, 0, 0]  # Collision
            
            ax = -self.G * planet.mass * x / r**3
            ay = -self.G * planet.mass * y / r**3
            
            return [vx, vy, ax, ay]
        
        # Integrate
        t_span = (0, 1000)
        t_eval = np.linspace(0, 1000, 1000)
        sol = integrate.solve_ivp(
            dynamics, t_span, [x0, y0, vx0, vy0], 
            t_eval=t_eval, method='RK45'
        )
        
        return np.column_stack((sol.y[0], sol.y[1]))
    
    def calculate_roche_limit(self, primary: CelestialBody, satellite: CelestialBody, 
                            fluid: bool = True) -> float:
        """Calculate Roche limit for tidal disruption"""
        if fluid:
            # Fluid Roche limit
            return 2.456 * primary.radius * (primary.mass / satellite.mass) ** (1/3)
        else:
            # Rigid body Roche limit
            return 2.455 * primary.radius * (primary.mass / satellite.mass) ** (1/3)
    
    def calculate_hill_sphere(self, body: CelestialBody, primary: CelestialBody, 
                            semi_major_axis: float) -> float:
        """Calculate Hill sphere radius"""
        return semi_major_axis * (body.mass / (3 * primary.mass)) ** (1/3)
    
    def calculate_tidal_force(self, body1: CelestialBody, body2: CelestialBody, 
                            radius: float) -> float:
        """Calculate tidal force at a given radius"""
        r = body1.distance_to(body2)
        return 2 * self.G * body1.mass * body2.mass * radius / r**3
    
    def relativistic_correction(self, body: CelestialBody, central_body: CelestialBody) -> np.ndarray:
        """Calculate relativistic correction to acceleration (GR effects)"""
        r_vec = body.position - central_body.position
        v_vec = body.velocity - central_body.velocity
        r = np.linalg.norm(r_vec)
        v = np.linalg.norm(v_vec)
        
        # Schwarzschild metric correction (simplified)
        rs = central_body.schwarzschild_radius(self.c)
        
        # Post-Newtonian correction
        correction_factor = 1 + 3 * rs / (2 * r) + (v / self.c)**2
        
        # Classical acceleration
        classical_accel = -self.G * central_body.mass * r_vec / r**3
        
        return classical_accel * correction_factor
    
    def three_body_problem(self, bodies: List[CelestialBody], dt: float, 
                          steps: int) -> Dict[str, np.ndarray]:
        """Solve restricted three-body problem"""
        if len(bodies) != 3:
            raise ValueError("Exactly 3 bodies required")
        
        # Use more sophisticated integration for chaotic system
        def dynamics(t, state):
            # Unpack state vector
            positions = state[:6].reshape(3, 2)
            velocities = state[6:].reshape(3, 2)
            
            # Calculate accelerations
            accelerations = np.zeros((3, 2))
            for i in range(3):
                for j in range(3):
                    if i != j:
                        r_vec = positions[j] - positions[i]
                        r = np.linalg.norm(r_vec)
                        if r > 0:
                            accelerations[i] += self.G * bodies[j].mass * r_vec / r**3
            
            # Pack derivatives
            return np.concatenate([velocities.flatten(), accelerations.flatten()])
        
        # Initial conditions
        y0 = np.concatenate([
            np.array([b.position for b in bodies]).flatten(),
            np.array([b.velocity for b in bodies]).flatten()
        ])
        
        # Integrate
        t_span = (0, steps * dt)
        t_eval = np.linspace(0, steps * dt, steps)
        sol = integrate.solve_ivp(dynamics, t_span, y0, t_eval=t_eval, 
                                method='DOP853', rtol=1e-10)
        
        # Extract trajectories
        trajectories = {}
        for i, body in enumerate(bodies):
            trajectories[body.name] = sol.y[2*i:2*i+2].T
        
        return trajectories