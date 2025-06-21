"""
CPU-based Physics Engine using Pre-trained Models
Replaces NVIDIA Modulus with CPU-optimized physics simulations
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
import logging
from pathlib import Path
import pickle

# Try to import optional ML libraries
try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import StandardScaler
    import joblib
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

logger = logging.getLogger(__name__)

class ClassicalPhysicsEngine:
    """
    Classical physics engine with CPU-optimized numerical methods
    """
    
    def __init__(self):
        self.G = 6.674e-11  # Gravitational constant
        
    def compute_gravity_force(self, pos1: np.ndarray, pos2: np.ndarray, 
                            mass1: float, mass2: float) -> np.ndarray:
        """Compute gravitational force between two bodies"""
        r_vec = pos2 - pos1
        r_mag = np.linalg.norm(r_vec) + 1e-10  # Softening
        force_mag = self.G * mass1 * mass2 / (r_mag**2)
        return force_mag * r_vec / r_mag
    
    def compute_orbital_velocity(self, central_mass: float, distance: float) -> float:
        """Compute circular orbital velocity"""
        return np.sqrt(self.G * central_mass / distance)
    
    def compute_escape_velocity(self, mass: float, radius: float) -> float:
        """Compute escape velocity from a massive body"""
        return np.sqrt(2 * self.G * mass / radius)

class PretrainedGravityModel:
    """
    Pre-trained machine learning models for gravity predictions
    Uses scikit-learn models for CPU efficiency
    """
    
    def __init__(self, model_cache_dir: str = "ml_models/pretrained"):
        self.model_cache_dir = Path(model_cache_dir)
        self.model_cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.gravity_field_model = None
        self.trajectory_model = None
        self.scaler_field = None
        self.scaler_trajectory = None
        
        self._load_or_create_models()
    
    def _load_or_create_models(self):
        """Load existing models or create new pre-trained ones"""
        # Load or create gravity field prediction model
        field_model_path = self.model_cache_dir / "gravity_field_model.pkl"
        field_scaler_path = self.model_cache_dir / "gravity_field_scaler.pkl"
        
        if field_model_path.exists() and field_scaler_path.exists():
            self.gravity_field_model = joblib.load(field_model_path)
            self.scaler_field = joblib.load(field_scaler_path)
            logger.info("Loaded gravity field model from cache")
        else:
            self._create_gravity_field_model()
        
        # Load or create trajectory prediction model
        traj_model_path = self.model_cache_dir / "trajectory_model.pkl"
        traj_scaler_path = self.model_cache_dir / "trajectory_scaler.pkl"
        
        if traj_model_path.exists() and traj_scaler_path.exists():
            self.trajectory_model = joblib.load(traj_model_path)
            self.scaler_trajectory = joblib.load(traj_scaler_path)
            logger.info("Loaded trajectory model from cache")
        else:
            self._create_trajectory_model()
    
    def _create_gravity_field_model(self):
        """Create and train a gravity field prediction model"""
        logger.info("Creating gravity field prediction model...")
        
        # Generate synthetic training data
        X_train, y_train = self._generate_gravity_training_data(n_samples=10000)
        
        # Create and train model
        self.gravity_field_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=20,
            random_state=42,
            n_jobs=-1
        )
        
        self.scaler_field = StandardScaler()
        X_train_scaled = self.scaler_field.fit_transform(X_train)
        
        self.gravity_field_model.fit(X_train_scaled, y_train)
        
        # Save models
        joblib.dump(self.gravity_field_model, self.model_cache_dir / "gravity_field_model.pkl")
        joblib.dump(self.scaler_field, self.model_cache_dir / "gravity_field_scaler.pkl")
        
        logger.info("Created and saved gravity field model")
    
    def _create_trajectory_model(self):
        """Create and train a trajectory prediction model"""
        logger.info("Creating trajectory prediction model...")
        
        # Generate synthetic training data
        X_train, y_train = self._generate_trajectory_training_data(n_samples=5000)
        
        # Create and train model
        self.trajectory_model = MLPRegressor(
            hidden_layer_sizes=(128, 64, 32),
            activation='relu',
            solver='adam',
            max_iter=500,
            random_state=42
        )
        
        self.scaler_trajectory = StandardScaler()
        X_train_scaled = self.scaler_trajectory.fit_transform(X_train)
        
        self.trajectory_model.fit(X_train_scaled, y_train)
        
        # Save models
        joblib.dump(self.trajectory_model, self.model_cache_dir / "trajectory_model.pkl")
        joblib.dump(self.scaler_trajectory, self.model_cache_dir / "trajectory_scaler.pkl")
        
        logger.info("Created and saved trajectory model")
    
    def _generate_gravity_training_data(self, n_samples: int) -> Tuple[np.ndarray, np.ndarray]:
        """Generate synthetic training data for gravity field prediction"""
        # Input: [x, y, z, mass, central_x, central_y, central_z]
        # Output: gravitational potential
        
        X = np.random.uniform(-10, 10, (n_samples, 7))
        y = np.zeros(n_samples)
        
        G = 6.674e-11
        
        for i in range(n_samples):
            x, y_coord, z = X[i, :3]
            mass = abs(X[i, 3]) * 1e24  # Ensure positive mass
            cx, cy, cz = X[i, 4:7]
            
            # Compute gravitational potential
            r = np.sqrt((x - cx)**2 + (y_coord - cy)**2 + (z - cz)**2) + 1e-10
            y[i] = -G * mass / r
        
        return X, y
    
    def _generate_trajectory_training_data(self, n_samples: int) -> Tuple[np.ndarray, np.ndarray]:
        """Generate synthetic training data for trajectory prediction"""
        # Input: [x, y, z, vx, vy, vz, mass, dt]
        # Output: [new_x, new_y, new_z, new_vx, new_vy, new_vz]
        
        X = np.random.uniform(-5, 5, (n_samples, 8))
        y = np.zeros((n_samples, 6))
        
        G = 6.674e-11
        central_mass = 5.972e24  # Earth mass
        
        for i in range(n_samples):
            pos = X[i, :3]
            vel = X[i, 3:6]
            mass = abs(X[i, 6]) * 1e20
            dt = abs(X[i, 7]) * 0.1  # Small time step
            
            # Simple Euler integration
            r = np.linalg.norm(pos) + 1e-10
            acc = -G * central_mass * pos / (r**3)
            
            new_vel = vel + acc * dt
            new_pos = pos + new_vel * dt
            
            y[i] = np.concatenate([new_pos, new_vel])
        
        return X, y
    
    def predict_gravity_field(self, positions: np.ndarray, masses: np.ndarray) -> np.ndarray:
        """Predict gravitational field using pre-trained model"""
        if self.gravity_field_model is None:
            raise ValueError("Gravity field model not loaded")
        
        # Prepare input features
        n_points = len(positions)
        features = []
        
        for pos in positions:
            for i, mass in enumerate(masses):
                # Feature: [x, y, z, mass, central_x, central_y, central_z]
                feature = np.concatenate([pos, [mass], pos])  # Simplified
                features.append(feature)
        
        X = np.array(features)
        X_scaled = self.scaler_field.transform(X)
        
        predictions = self.gravity_field_model.predict(X_scaled)
        return predictions.reshape(len(masses), -1)
    
    def predict_trajectory(self, initial_state: np.ndarray, time_steps: int, dt: float = 0.01) -> np.ndarray:
        """Predict trajectory using pre-trained model"""
        if self.trajectory_model is None:
            raise ValueError("Trajectory model not loaded")
        
        trajectory = np.zeros((time_steps, 6))
        current_state = initial_state[:6].copy()  # [x, y, z, vx, vy, vz]
        mass = 1e20  # Default mass
        
        for i in range(time_steps):
            trajectory[i] = current_state
            
            # Prepare input: [x, y, z, vx, vy, vz, mass, dt]
            X = np.array([[*current_state, mass, dt]])
            X_scaled = self.scaler_trajectory.transform(X)
            
            # Predict next state
            next_state = self.trajectory_model.predict(X_scaled)[0]
            current_state = next_state
        
        return trajectory

class ModulusGravityEngine:
    """
    CPU-based gravity engine that replaces NVIDIA Modulus
    Combines classical physics with pre-trained ML models
    """
    
    def __init__(self):
        self.classical_engine = ClassicalPhysicsEngine()
        self.ml_models = PretrainedGravityModel()
        self.device = "cpu"
        self.status = "initialized"
        
        logger.info("Initialized CPU-based gravity engine")
    
    def compute_gravity_field(self, positions: np.ndarray, masses: np.ndarray, 
                            grid_bounds: Tuple[float, float, float] = (-10, 10, 50),
                            use_ml: bool = True) -> np.ndarray:
        """
        Compute gravitational field using hybrid classical + ML approach
        """
        min_coord, max_coord, grid_size = grid_bounds
        
        # Create coordinate grid
        coords = np.linspace(min_coord, max_coord, grid_size)
        x_grid, y_grid, z_grid = np.meshgrid(coords, coords, coords, indexing='ij')
        
        # Flatten grid for processing
        grid_points = np.stack([x_grid.flatten(), y_grid.flatten(), z_grid.flatten()], axis=1)
        
        if use_ml and self.ml_models.gravity_field_model is not None:
            # Use ML prediction
            try:
                field_flat = self.ml_models.predict_gravity_field(grid_points, masses)
                field_flat = field_flat.mean(axis=0)  # Average over masses
            except:
                logger.warning("ML prediction failed, falling back to classical")
                field_flat = self._compute_classical_field(grid_points, positions, masses)
        else:
            # Use classical computation
            field_flat = self._compute_classical_field(grid_points, positions, masses)
        
        # Reshape back to grid
        field = field_flat.reshape(grid_size, grid_size, grid_size)
        return field
    
    def _compute_classical_field(self, grid_points: np.ndarray, positions: np.ndarray, 
                               masses: np.ndarray) -> np.ndarray:
        """Classical gravitational field computation"""
        G = self.classical_engine.G
        field = np.zeros(len(grid_points))
        
        for i, (pos, mass) in enumerate(zip(positions, masses)):
            r_vecs = grid_points - pos
            r_mags = np.linalg.norm(r_vecs, axis=1) + 1e-10
            field += -G * mass / r_mags
        
        return field
    
    def simulate_trajectory(self, initial_conditions: Dict[str, Any], 
                          time_span: Tuple[float, float], 
                          n_points: int = 1000) -> Dict[str, np.ndarray]:
        """
        Simulate orbital trajectory
        """
        t_start, t_end = time_span
        times = np.linspace(t_start, t_end, n_points)
        dt = (t_end - t_start) / (n_points - 1)
        
        # Extract initial conditions
        pos = np.array(initial_conditions.get('position', [1.0, 0.0, 0.0]))
        vel = np.array(initial_conditions.get('velocity', [0.0, 1.0, 0.0]))
        mass = initial_conditions.get('mass', 1e20)
        
        initial_state = np.concatenate([pos, vel])
        
        # Use ML prediction if available, otherwise classical
        try:
            trajectory = self.ml_models.predict_trajectory(initial_state, n_points, dt)
            positions = trajectory[:, :3]
            velocities = trajectory[:, 3:]
        except:
            logger.warning("ML trajectory prediction failed, using classical")
            positions, velocities = self._classical_trajectory(initial_state, times)
        
        return {
            'times': times,
            'positions': positions,
            'velocities': velocities,
            'energy': self._compute_energy(positions, velocities, mass)
        }
    
    def _classical_trajectory(self, initial_state: np.ndarray, times: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Classical trajectory integration"""
        n_points = len(times)
        positions = np.zeros((n_points, 3))
        velocities = np.zeros((n_points, 3))
        
        pos = initial_state[:3]
        vel = initial_state[3:]
        
        G = self.classical_engine.G
        central_mass = 5.972e24  # Earth mass
        
        for i, t in enumerate(times):
            positions[i] = pos
            velocities[i] = vel
            
            if i < n_points - 1:
                dt = times[i+1] - times[i]
                
                # Simple Euler integration
                r = np.linalg.norm(pos) + 1e-10
                acc = -G * central_mass * pos / (r**3)
                
                vel += acc * dt
                pos += vel * dt
        
        return positions, velocities
    
    def _compute_energy(self, positions: np.ndarray, velocities: np.ndarray, mass: float) -> np.ndarray:
        """Compute total energy over time"""
        G = self.classical_engine.G
        central_mass = 5.972e24
        
        kinetic = 0.5 * mass * np.sum(velocities**2, axis=1)
        
        r = np.linalg.norm(positions, axis=1) + 1e-10
        potential = -G * mass * central_mass / r
        
        return kinetic + potential
    
    def get_status(self) -> Dict[str, Any]:
        """Get engine status information"""
        return {
            'status': self.status,
            'device': self.device,
            'classical_engine': 'active',
            'ml_models': 'loaded' if self.ml_models.gravity_field_model else 'not_loaded',
            'framework': 'CPU-based scikit-learn + classical physics'
        }
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about loaded models"""
        info = {
            'framework': 'scikit-learn',
            'device': self.device,
            'models': {}
        }
        
        if self.ml_models.gravity_field_model:
            info['models']['gravity_field'] = {
                'type': 'RandomForestRegressor',
                'n_estimators': self.ml_models.gravity_field_model.n_estimators,
                'max_depth': self.ml_models.gravity_field_model.max_depth
            }
        
        if self.ml_models.trajectory_model:
            info['models']['trajectory'] = {
                'type': 'MLPRegressor',
                'hidden_layers': self.ml_models.trajectory_model.hidden_layer_sizes,
                'activation': self.ml_models.trajectory_model.activation
            }
        
        return info
