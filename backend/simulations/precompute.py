import numpy as np
import json
import pickle
from typing import Dict, List, Tuple
from ..models.celestial_body import CelestialBody
from .gravity_solver import GravitySolver
import os
from pathlib import Path

class PrecomputedSimulations:
    """Handles pre-computation and caching of complex simulations"""
    
    def __init__(self, cache_dir: str = "data/precomputed"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.solver = GravitySolver()
        
    def generate_standard_orbits(self):
        """Pre-compute standard orbital scenarios"""
        scenarios = {
            'circular_orbit': self._generate_circular_orbit(),
            'elliptical_orbit': self._generate_elliptical_orbit(),
            'parabolic_trajectory': self._generate_parabolic_trajectory(),
            'hyperbolic_trajectory': self._generate_hyperbolic_trajectory(),
            'binary_system': self._generate_binary_system(),
            'figure_eight': self._generate_figure_eight(),
            'lagrange_points': self._generate_lagrange_demonstration()
        }
        
        for name, data in scenarios.items():
            self._save_scenario(name, data)
        
        return scenarios
    
    def _generate_circular_orbit(self) -> Dict:
        """Generate perfect circular orbit data"""
        # Earth-like planet
        central = CelestialBody(
            name="Star",
            mass=1.989e30,  # Solar mass
            position=np.array([0.0, 0.0]),
            velocity=np.array([0.0, 0.0])
        )
        
        # Calculate circular orbital velocity
        r = 1.496e11  # 1 AU
        v_circular = np.sqrt(self.solver.G * central.mass / r)
        
        satellite = CelestialBody(
            name="Planet",
            mass=5.972e24,  # Earth mass
            position=np.array([r, 0.0]),
            velocity=np.array([0.0, v_circular])
        )
        
        # Simulate for one year
        dt = 3600  # 1 hour
        steps = 365 * 24  # One year
        
        trajectories = self.solver.simulate_system([central, satellite], dt, steps)
        
        return {
            'bodies': [self._body_to_dict(central), self._body_to_dict(satellite)],
            'trajectories': {k: v.tolist() for k, v in trajectories.items()},
            'parameters': {
                'dt': dt,
                'steps': steps,
                'orbital_radius': r,
                'orbital_velocity': v_circular,
                'period': 2 * np.pi * r / v_circular
            }
        }
    
    def _generate_elliptical_orbit(self) -> Dict:
        """Generate elliptical orbit with eccentricity 0.5"""
        central = CelestialBody(
            name="Star",
            mass=1.989e30,
            position=np.array([0.0, 0.0]),
            velocity=np.array([0.0, 0.0])
        )
        
        # Elliptical orbit parameters
        a = 1.496e11  # Semi-major axis (1 AU)
        e = 0.5  # Eccentricity
        periapsis = a * (1 - e)
        
        # Velocity at periapsis
        v_periapsis = np.sqrt(self.solver.G * central.mass * (2/periapsis - 1/a))
        
        satellite = CelestialBody(
            name="Comet",
            mass=1e15,  # Small mass
            position=np.array([periapsis, 0.0]),
            velocity=np.array([0.0, v_periapsis])
        )
        
        dt = 3600
        steps = 365 * 24 * 2  # Two years
        
        trajectories = self.solver.simulate_system([central, satellite], dt, steps)
        
        return {
            'bodies': [self._body_to_dict(central), self._body_to_dict(satellite)],
            'trajectories': {k: v.tolist() for k, v in trajectories.items()},
            'parameters': {
                'dt': dt,
                'steps': steps,
                'semi_major_axis': a,
                'eccentricity': e,
                'periapsis': periapsis,
                'apoapsis': a * (1 + e)
            }
        }
    
    def _generate_parabolic_trajectory(self) -> Dict:
        """Generate parabolic escape trajectory"""
        central = CelestialBody(
            name="Planet",
            mass=5.972e24,  # Earth mass
            position=np.array([0.0, 0.0]),
            velocity=np.array([0.0, 0.0])
        )
        
        # Starting position
        r0 = 6.371e6 + 1e6  # 1000 km above surface
        
        # Parabolic velocity (escape velocity)
        v_escape = np.sqrt(2 * self.solver.G * central.mass / r0)
        
        projectile = CelestialBody(
            name="Spacecraft",
            mass=1000,  # 1 ton
            position=np.array([r0, 0.0]),
            velocity=np.array([0.0, v_escape])
        )
        
        dt = 60  # 1 minute
        steps = 24 * 60  # One day
        
        trajectories = self.solver.simulate_system([central, projectile], dt, steps)
        
        return {
            'bodies': [self._body_to_dict(central), self._body_to_dict(projectile)],
            'trajectories': {k: v.tolist() for k, v in trajectories.items()},
            'parameters': {
                'dt': dt,
                'steps': steps,
                'escape_velocity': v_escape,
                'initial_radius': r0
            }
        }
    
    def _generate_hyperbolic_trajectory(self) -> Dict:
        """Generate hyperbolic flyby trajectory"""
        planet = CelestialBody(
            name="Jupiter",
            mass=1.898e27,  # Jupiter mass
            position=np.array([0.0, 0.0]),
            velocity=np.array([0.0, 0.0])
        )
        
        # Hyperbolic excess velocity
        v_infinity = 20000  # 20 km/s
        impact_parameter = 1e8  # 100,000 km
        
        # Starting position far away
        r0 = 1e9  # 1 million km
        
        spacecraft = CelestialBody(
            name="Voyager",
            mass=825,
            position=np.array([r0, impact_parameter]),
            velocity=np.array([-v_infinity, 0.0])
        )
        
        dt = 600  # 10 minutes
        steps = 24 * 6 * 10  # 10 days
        
        trajectories = self.solver.simulate_system([planet, spacecraft], dt, steps)
        
        return {
            'bodies': [self._body_to_dict(planet), self._body_to_dict(spacecraft)],
            'trajectories': {k: v.tolist() for k, v in trajectories.items()},
            'parameters': {
                'dt': dt,
                'steps': steps,
                'v_infinity': v_infinity,
                'impact_parameter': impact_parameter
            }
        }
    
    def _generate_binary_system(self) -> Dict:
        """Generate binary star system"""
        # Equal mass binary
        m = 1.989e30  # Solar mass each
        separation = 1.496e11  # 1 AU
        
        # Calculate orbital velocity for circular orbit
        v = np.sqrt(self.solver.G * m / (2 * separation))
        
        star1 = CelestialBody(
            name="Star A",
            mass=m,
            position=np.array([separation/2, 0.0]),
            velocity=np.array([0.0, v])
        )
        
        star2 = CelestialBody(
            name="Star B",
            mass=m,
            position=np.array([-separation/2, 0.0]),
            velocity=np.array([0.0, -v])
        )
        
        dt = 3600  # 1 hour
        steps = 365 * 24  # One year
        
        trajectories = self.solver.simulate_system([star1, star2], dt, steps)
        
        return {
            'bodies': [self._body_to_dict(star1), self._body_to_dict(star2)],
            'trajectories': {k: v.tolist() for k, v in trajectories.items()},
            'parameters': {
                'dt': dt,
                'steps': steps,
                'separation': separation,
                'orbital_velocity': v,
                'period': np.pi * separation / v
            }
        }
    
    def _generate_figure_eight(self) -> Dict:
        """Generate the famous figure-8 three-body solution"""
        # Specific initial conditions for figure-8
        m = 1.0  # Unit mass
        
        body1 = CelestialBody(
            name="Body 1",
            mass=m,
            position=np.array([0.97000436, -0.24308753]),
            velocity=np.array([0.466203685, 0.43236573])
        )
        
        body2 = CelestialBody(
            name="Body 2",
            mass=m,
            position=np.array([-0.97000436, 0.24308753]),
            velocity=np.array([0.466203685, 0.43236573])
        )
        
        body3 = CelestialBody(
            name="Body 3",
            mass=m,
            position=np.array([0.0, 0.0]),
            velocity=np.array([-0.93240737, -0.86473146])
        )
        
        dt = 0.001
        steps = 6000
        
        trajectories = self.solver.three_body_problem([body1, body2, body3], dt, steps)
        
        return {
            'bodies': [self._body_to_dict(body1), self._body_to_dict(body2), 
                      self._body_to_dict(body3)],
            'trajectories': {k: v.tolist() for k, v in trajectories.items()},
            'parameters': {
                'dt': dt,
                'steps': steps,
                'pattern': 'figure-eight'
            }
        }
    
    def _generate_lagrange_demonstration(self) -> Dict:
        """Generate Lagrange points demonstration"""
        # Sun-Earth system
        sun = CelestialBody(
            name="Sun",
            mass=1.989e30,
            position=np.array([0.0, 0.0]),
            velocity=np.array([0.0, 0.0])
        )
        
        # Earth in circular orbit
        r_earth = 1.496e11
        v_earth = np.sqrt(self.solver.G * sun.mass / r_earth)
        
        earth = CelestialBody(
            name="Earth",
            mass=5.972e24,
            position=np.array([r_earth, 0.0]),
            velocity=np.array([0.0, v_earth])
        )
        
        # Calculate L4 position (60 degrees ahead)
        angle = np.pi / 3
        l4_pos = np.array([
            r_earth * np.cos(angle),
            r_earth * np.sin(angle)
        ])
        
        # Velocity for L4 (same angular velocity as Earth)
        l4_vel = np.array([
            -v_earth * np.sin(angle),
            v_earth * np.cos(angle)
        ])
        
        trojan = CelestialBody(
            name="Trojan",
            mass=1e20,  # Asteroid mass
            position=l4_pos,
            velocity=l4_vel
        )
        
        dt = 3600 * 24  # 1 day
        steps = 365  # One year
        
        trajectories = self.solver.simulate_system([sun, earth, trojan], dt, steps)
        
        return {
            'bodies': [self._body_to_dict(sun), self._body_to_dict(earth), 
                      self._body_to_dict(trojan)],
            'trajectories': {k: v.tolist() for k, v in trajectories.items()},
            'parameters': {
                'dt': dt,
                'steps': steps,
                'lagrange_point': 'L4',
                'stability': 'stable'
            }
        }
    
    def _body_to_dict(self, body: CelestialBody) -> Dict:
        """Convert CelestialBody to dictionary"""
        return {
            'name': body.name,
            'mass': body.mass,
            'position': body.position.tolist(),
            'velocity': body.velocity.tolist(),
            'radius': body.radius,
            'color': body.color
        }
    
    def _save_scenario(self, name: str, data: Dict):
        """Save scenario to disk"""
        filepath = self.cache_dir / f"{name}.json"
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load_scenario(self, name: str) -> Dict:
        """Load pre-computed scenario"""
        filepath = self.cache_dir / f"{name}.json"
        if filepath.exists():
            with open(filepath, 'r') as f:
                return json.load(f)
        else:
            raise FileNotFoundError(f"Scenario '{name}' not found")
    
    def list_scenarios(self) -> List[str]:
        """List available pre-computed scenarios"""
        return [f.stem for f in self.cache_dir.glob("*.json")]