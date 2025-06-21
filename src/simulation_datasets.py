"""
Pre-generated Simulation Datasets Manager

This module manages pre-computed NVIDIA Modulus simulation results for efficient serving
in the Streamlit educational platform. It handles dataset caching, loading, and
real-time generation when pre-computed data is not available.
"""

import numpy as np
import pandas as pd
import json
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import time
import hashlib
from dataclasses import dataclass
from .cpu_physics_engine import ModulusGravityEngine
from .cpu_data_processor import CuDFDataProcessor


@dataclass
class SimulationMetadata:
    """Metadata for simulation datasets"""
    scenario_id: str
    physics_type: str  # 'newtonian', 'relativistic', 'orbital'
    parameters: Dict[str, Any]
    grid_resolution: Tuple[int, int, int]
    time_steps: int
    computation_time: float
    file_path: str
    created_at: str


class PreGeneratedSimulations:
    """
    Manager for pre-generated NVIDIA Modulus simulation datasets.
    
    Handles:
    - Loading pre-computed simulation results
    - Caching and serving simulation data efficiently
    - Real-time generation when pre-computed data unavailable
    - Dataset metadata management
    """
    
    def __init__(self, data_dir: str = "data/precomputed"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize physics engine and data processor
        self.physics_engine = ModulusGravityEngine()
        self.data_processor = CuDFDataProcessor()
        
        # Cache for loaded datasets
        self._cache = {}
        self._metadata_cache = {}
        
        # Load metadata index
        self.metadata_file = self.data_dir / "simulation_metadata.json"
        self._load_metadata_index()
        
        # Default scenarios
        self._init_default_scenarios()
    
    def _load_metadata_index(self):
        """Load simulation metadata index"""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    metadata_dict = json.load(f)
                    self._metadata_cache = {
                        k: SimulationMetadata(**v) for k, v in metadata_dict.items()
                    }
            except Exception as e:
                print(f"Warning: Could not load metadata index: {e}")
                self._metadata_cache = {}
        else:
            self._metadata_cache = {}
    
    def _save_metadata_index(self):
        """Save simulation metadata index"""
        try:
            metadata_dict = {
                k: v.__dict__ for k, v in self._metadata_cache.items()
            }
            with open(self.metadata_file, 'w') as f:
                json.dump(metadata_dict, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save metadata index: {e}")
    
    def _init_default_scenarios(self):
        """Initialize default physics scenarios"""
        self.default_scenarios = {
            "binary_orbit": {
                "physics_type": "newtonian",
                "parameters": {
                    "mass1": 1e30,  # kg (solar mass)
                    "mass2": 5.97e24,  # kg (earth mass)
                    "separation": 1.5e11,  # m (1 AU)
                    "domain_size": 3e11,
                    "time_span": 365.25 * 24 * 3600  # 1 year
                },
                "description": "Binary orbital system simulation"
            },
            "black_hole_accretion": {
                "physics_type": "relativistic",
                "parameters": {
                    "black_hole_mass": 10 * 1.989e30,  # 10 solar masses
                    "domain_size": 1e12,
                    "time_span": 1000
                },
                "description": "Black hole gravitational field simulation"
            },
            "planetary_system": {
                "physics_type": "orbital",
                "parameters": {
                    "central_mass": 1.989e30,  # solar mass
                    "planet_masses": [3.3e23, 4.87e24, 5.97e24, 6.39e23],  # Mercury, Venus, Earth, Mars
                    "planet_distances": [5.8e10, 1.08e11, 1.5e11, 2.28e11],  # AU
                    "domain_size": 5e11,
                    "time_span": 2 * 365.25 * 24 * 3600  # 2 years
                },
                "description": "Multi-planet orbital mechanics"
            },
            "gravitational_waves": {
                "physics_type": "relativistic",
                "parameters": {
                    "mass1": 30 * 1.989e30,  # 30 solar masses
                    "mass2": 20 * 1.989e30,  # 20 solar masses
                    "initial_separation": 1000e3,  # 1000 km
                    "domain_size": 1e6,
                    "time_span": 1.0
                },
                "description": "Gravitational wave generation from binary merger"
            }
        }
    
    def get_scenario_list(self) -> List[Dict[str, str]]:
        """Get list of available scenarios"""
        scenarios = []
        
        # Add pre-computed scenarios
        for scenario_id, metadata in self._metadata_cache.items():
            scenarios.append({
                "id": scenario_id,
                "name": scenario_id.replace("_", " ").title(),
                "type": metadata.physics_type,
                "status": "pre-computed",
                "description": f"Pre-computed {metadata.physics_type} simulation"
            })
        
        # Add default scenarios
        for scenario_id, config in self.default_scenarios.items():
            if scenario_id not in self._metadata_cache:
                scenarios.append({
                    "id": scenario_id,
                    "name": scenario_id.replace("_", " ").title(),
                    "type": config["physics_type"],
                    "status": "generate-on-demand",
                    "description": config["description"]
                })
        
        return scenarios
    
    def load_simulation(self, scenario_id: str, grid_resolution: Tuple[int, int, int] = (64, 64, 32)) -> Dict[str, Any]:
        """
        Load simulation data for a scenario.
        
        Args:
            scenario_id: Identifier for the simulation scenario
            grid_resolution: Grid resolution for simulation
            
        Returns:
            Dictionary containing simulation data and metadata
        """
        # Check cache first
        cache_key = f"{scenario_id}_{grid_resolution}"
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        # Try to load pre-computed data
        if scenario_id in self._metadata_cache:
            metadata = self._metadata_cache[scenario_id]
            try:
                data = self._load_precomputed_data(metadata.file_path)
                self._cache[cache_key] = data
                return data
            except Exception as e:
                print(f"Warning: Could not load pre-computed data for {scenario_id}: {e}")
        
        # Generate data on-demand
        print(f"Generating simulation data for {scenario_id}...")
        data = self._generate_simulation_data(scenario_id, grid_resolution)
        
        # Cache the result
        self._cache[cache_key] = data
        
        # Optionally save for future use
        self._save_simulation_data(scenario_id, data, grid_resolution)
        
        return data
    
    def _load_precomputed_data(self, file_path: str) -> Dict[str, Any]:
        """Load pre-computed simulation data from file"""
        full_path = self.data_dir / file_path
        
        if full_path.suffix == '.pkl':
            with open(full_path, 'rb') as f:
                return pickle.load(f)
        elif full_path.suffix == '.npz':
            data = np.load(full_path)
            return {key: data[key] for key in data.files}
        else:
            raise ValueError(f"Unsupported file format: {full_path.suffix}")
    
    def _generate_simulation_data(self, scenario_id: str, grid_resolution: Tuple[int, int, int]) -> Dict[str, Any]:
        """Generate simulation data using NVIDIA Modulus"""
        if scenario_id not in self.default_scenarios:
            raise ValueError(f"Unknown scenario: {scenario_id}")
        
        config = self.default_scenarios[scenario_id]
        physics_type = config["physics_type"]
        params = config["parameters"]
        
        start_time = time.time()
        
        # Generate grid
        domain_size = params.get("domain_size", 1e11)
        x = np.linspace(-domain_size/2, domain_size/2, grid_resolution[0])
        y = np.linspace(-domain_size/2, domain_size/2, grid_resolution[1])
        z = np.linspace(-domain_size/4, domain_size/4, grid_resolution[2])
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        
        # Generate simulation based on physics type
        if physics_type == "newtonian":
            result = self._generate_newtonian_simulation(X, Y, Z, params, scenario_id)
        elif physics_type == "relativistic":
            result = self._generate_relativistic_simulation(X, Y, Z, params, scenario_id)
        elif physics_type == "orbital":
            result = self._generate_orbital_simulation(X, Y, Z, params, scenario_id)
        else:
            raise ValueError(f"Unknown physics type: {physics_type}")
        
        computation_time = time.time() - start_time
        
        # Add metadata
        result.update({
            "scenario_id": scenario_id,
            "physics_type": physics_type,
            "parameters": params,
            "grid_resolution": grid_resolution,
            "computation_time": computation_time,
            "coordinates": {"X": X, "Y": Y, "Z": Z}
        })
        
        return result
    
    def _generate_newtonian_simulation(self, X, Y, Z, params, scenario_id):
        """Generate Newtonian gravity simulation"""
        if scenario_id == "binary_orbit":
            # Binary system
            m1, m2 = params["mass1"], params["mass2"]
            sep = params["separation"]
            G = 6.67430e-11
            
            # Positions of masses
            pos1 = np.array([-sep/3, 0, 0])
            pos2 = np.array([sep*2/3, 0, 0])
            
            # Calculate gravitational potential
            r1 = np.sqrt((X - pos1[0])**2 + (Y - pos1[1])**2 + (Z - pos1[2])**2)
            r2 = np.sqrt((X - pos2[0])**2 + (Y - pos2[1])**2 + (Z - pos2[2])**2)
            
            # Avoid singularities
            r1 = np.maximum(r1, sep/100)
            r2 = np.maximum(r2, sep/100)
            
            potential = -G * (m1/r1 + m2/r2)
            
            # Calculate field components
            field_x = G * (m1*(X - pos1[0])/r1**3 + m2*(X - pos2[0])/r2**3)
            field_y = G * (m1*(Y - pos1[1])/r1**3 + m2*(Y - pos2[1])/r2**3)
            field_z = G * (m1*(Z - pos1[2])/r1**3 + m2*(Z - pos2[2])/r2**3)
            
            # Generate orbital trajectories
            time_span = params.get("time_span", 365.25 * 24 * 3600)
            n_points = 1000
            t = np.linspace(0, time_span, n_points)
            
            # Simplified circular orbits
            omega = np.sqrt(G * (m1 + m2) / sep**3)
            
            # Trajectory of reduced mass around center of mass
            mu = m1 * m2 / (m1 + m2)
            r_cm1 = m2 * sep / (m1 + m2)
            r_cm2 = m1 * sep / (m1 + m2)
            
            traj1_x = r_cm1 * np.cos(omega * t)
            traj1_y = r_cm1 * np.sin(omega * t)
            traj1_z = np.zeros_like(t)
            
            traj2_x = -r_cm2 * np.cos(omega * t)
            traj2_y = -r_cm2 * np.sin(omega * t)
            traj2_z = np.zeros_like(t)
            
            return {
                "potential": potential,
                "field_x": field_x,
                "field_y": field_y,
                "field_z": field_z,
                "trajectories": {
                    "time": t,
                    "object1": np.column_stack([traj1_x, traj1_y, traj1_z]),
                    "object2": np.column_stack([traj2_x, traj2_y, traj2_z])
                },
                "masses": {"m1": m1, "m2": m2},
                "orbital_period": 2 * np.pi / omega
            }
        
        # Default case
        return self._generate_default_gravity_field(X, Y, Z, params)
    
    def _generate_relativistic_simulation(self, X, Y, Z, params, scenario_id):
        """Generate relativistic gravity simulation"""
        if scenario_id == "black_hole_accretion":
            M = params["black_hole_mass"]
            G = 6.67430e-11
            c = 299792458  # m/s
            rs = 2 * G * M / c**2  # Schwarzschild radius
            
            # Calculate radius from center
            r = np.sqrt(X**2 + Y**2 + Z**2)
            r = np.maximum(r, rs/10)  # Avoid singularity
            
            # Schwarzschild metric components (simplified)
            potential = -G * M / r * (1 + rs/(2*r))
            
            # Frame dragging effects (simplified)
            dragging_factor = rs / (2 * r)
            dragging_factor = np.minimum(dragging_factor, 1.0)
            
            # Relativistic field corrections
            field_strength = G * M / r**2 * (1 + 3*rs/(2*r))
            
            field_x = field_strength * X / r
            field_y = field_strength * Y / r
            field_z = field_strength * Z / r
            
            # Generate geodesics for test particles
            n_geodesics = 20
            geodesics = []
            
            for i in range(n_geodesics):
                # Initial conditions for test particle
                r_init = rs * (3 + 10 * i/n_geodesics)  # Start at different radii
                phi_init = 2 * np.pi * i / n_geodesics
                
                # Simplified geodesic (circular orbit approximation)
                n_points = 500
                phi = np.linspace(phi_init, phi_init + 4*np.pi, n_points)
                r_orbit = r_init * np.ones_like(phi)
                
                x_geo = r_orbit * np.cos(phi)
                y_geo = r_orbit * np.sin(phi)
                z_geo = np.zeros_like(phi)
                
                geodesics.append(np.column_stack([x_geo, y_geo, z_geo]))
            
            return {
                "potential": potential,
                "field_x": field_x,
                "field_y": field_y,
                "field_z": field_z,
                "schwarzschild_radius": rs,
                "frame_dragging": dragging_factor,
                "geodesics": geodesics,
                "black_hole_mass": M
            }
        
        # Default relativistic case
        return self._generate_default_gravity_field(X, Y, Z, params)
    
    def _generate_orbital_simulation(self, X, Y, Z, params, scenario_id):
        """Generate orbital mechanics simulation"""
        if scenario_id == "planetary_system":
            M_sun = params["central_mass"]
            planet_masses = params["planet_masses"]
            planet_distances = params["planet_distances"]
            G = 6.67430e-11
            
            # Central star potential
            r_sun = np.sqrt(X**2 + Y**2 + Z**2)
            r_sun = np.maximum(r_sun, 1e8)  # Avoid singularity
            
            potential = -G * M_sun / r_sun
            field_x = G * M_sun * X / r_sun**3
            field_y = G * M_sun * Y / r_sun**3
            field_z = G * M_sun * Z / r_sun**3
            
            # Add planetary contributions
            planet_trajectories = []
            time_span = params.get("time_span", 2 * 365.25 * 24 * 3600)
            n_points = 2000
            t = np.linspace(0, time_span, n_points)
            
            for i, (m_planet, a) in enumerate(zip(planet_masses, planet_distances)):
                # Orbital period (Kepler's 3rd law)
                T = 2 * np.pi * np.sqrt(a**3 / (G * M_sun))
                omega = 2 * np.pi / T
                
                # Circular orbit approximation
                phase = 2 * np.pi * i / len(planet_masses)  # Spread planets
                x_planet = a * np.cos(omega * t + phase)
                y_planet = a * np.sin(omega * t + phase)
                z_planet = np.zeros_like(t)
                
                planet_trajectories.append({
                    "trajectory": np.column_stack([x_planet, y_planet, z_planet]),
                    "mass": m_planet,
                    "orbital_radius": a,
                    "period": T
                })
                
                # Add planet's gravitational contribution to field
                # (simplified - planets at mean position)
                x_mean, y_mean = a * np.cos(phase), a * np.sin(phase)
                r_planet = np.sqrt((X - x_mean)**2 + (Y - y_mean)**2 + Z**2)
                r_planet = np.maximum(r_planet, a/1000)
                
                potential += -G * m_planet / r_planet
                field_x += G * m_planet * (X - x_mean) / r_planet**3
                field_y += G * m_planet * (Y - y_mean) / r_planet**3
                field_z += G * m_planet * Z / r_planet**3
            
            return {
                "potential": potential,
                "field_x": field_x,
                "field_y": field_y,
                "field_z": field_z,
                "planet_trajectories": planet_trajectories,
                "time": t,
                "central_mass": M_sun
            }
        
        # Default orbital case
        return self._generate_default_gravity_field(X, Y, Z, params)
    
    def _generate_default_gravity_field(self, X, Y, Z, params):
        """Generate a default gravity field simulation"""
        # Single point mass at origin
        M = params.get("mass1", 1.989e30)  # Solar mass default
        G = 6.67430e-11
        
        r = np.sqrt(X**2 + Y**2 + Z**2)
        r = np.maximum(r, 1e6)  # Avoid singularity
        
        potential = -G * M / r
        field_x = G * M * X / r**3
        field_y = G * M * Y / r**3
        field_z = G * M * Z / r**3
        
        return {
            "potential": potential,
            "field_x": field_x,
            "field_y": field_y,
            "field_z": field_z,
            "mass": M
        }
    
    def _save_simulation_data(self, scenario_id: str, data: Dict[str, Any], grid_resolution: Tuple[int, int, int]):
        """Save simulation data for future use"""
        try:
            # Create filename
            resolution_str = f"{grid_resolution[0]}x{grid_resolution[1]}x{grid_resolution[2]}"
            filename = f"{scenario_id}_{resolution_str}.pkl"
            filepath = self.data_dir / filename
            
            # Save data
            with open(filepath, 'wb') as f:
                pickle.dump(data, f)
            
            # Update metadata
            metadata = SimulationMetadata(
                scenario_id=scenario_id,
                physics_type=data.get("physics_type", "unknown"),
                parameters=data.get("parameters", {}),
                grid_resolution=grid_resolution,
                time_steps=data.get("time_steps", 0),
                computation_time=data.get("computation_time", 0),
                file_path=filename,
                created_at=time.strftime("%Y-%m-%d %H:%M:%S")
            )
            
            self._metadata_cache[scenario_id] = metadata
            self._save_metadata_index()
            
            print(f"Saved simulation data: {filepath}")
            
        except Exception as e:
            print(f"Warning: Could not save simulation data: {e}")
    
    def get_simulation_info(self, scenario_id: str) -> Dict[str, Any]:
        """Get information about a simulation scenario"""
        if scenario_id in self._metadata_cache:
            metadata = self._metadata_cache[scenario_id]
            return {
                "scenario_id": scenario_id,
                "physics_type": metadata.physics_type,
                "status": "pre-computed",
                "grid_resolution": metadata.grid_resolution,
                "computation_time": metadata.computation_time,
                "created_at": metadata.created_at,
                "file_size": self._get_file_size(metadata.file_path)
            }
        elif scenario_id in self.default_scenarios:
            config = self.default_scenarios[scenario_id]
            return {
                "scenario_id": scenario_id,
                "physics_type": config["physics_type"],
                "status": "generate-on-demand",
                "description": config["description"],
                "parameters": config["parameters"]
            }
        else:
            return {"error": f"Unknown scenario: {scenario_id}"}
    
    def _get_file_size(self, filename: str) -> str:
        """Get human-readable file size"""
        try:
            filepath = self.data_dir / filename
            if filepath.exists():
                size_bytes = filepath.stat().st_size
                for unit in ['B', 'KB', 'MB', 'GB']:
                    if size_bytes < 1024:
                        return f"{size_bytes:.1f} {unit}"
                    size_bytes /= 1024
                return f"{size_bytes:.1f} TB"
            return "Unknown"
        except:
            return "Unknown"
    
    def clear_cache(self):
        """Clear the in-memory cache"""
        self._cache.clear()
        print("Simulation cache cleared")
    
    def regenerate_scenario(self, scenario_id: str, grid_resolution: Tuple[int, int, int] = (64, 64, 32)):
        """Force regeneration of a scenario"""
        # Remove from cache
        cache_key = f"{scenario_id}_{grid_resolution}"
        if cache_key in self._cache:
            del self._cache[cache_key]
        
        # Remove metadata if exists
        if scenario_id in self._metadata_cache:
            old_metadata = self._metadata_cache[scenario_id]
            old_file = self.data_dir / old_metadata.file_path
            if old_file.exists():
                old_file.unlink()
            del self._metadata_cache[scenario_id]
            self._save_metadata_index()
        
        # Generate new data
        return self.load_simulation(scenario_id, grid_resolution)
