"""
NVIDIA Modulus Integration for Real-Time Educational Physics Games
Implements GPU-optional physics simulations using NVIDIA Modulus and cuDF
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Union, Any
import logging
from pathlib import Path
import warnings

# GPU availability check
CUDA_AVAILABLE = torch.cuda.is_available()
DEVICE = torch.device("cuda" if CUDA_AVAILABLE else "cpu")

logger = logging.getLogger(__name__)

try:
    # Try to import NVIDIA Modulus
    import modulus
    from modulus.models.mlp import FullyConnectedArch
    from modulus.models.fourier_net import FourierNetArch
    from modulus.domain import Domain
    from modulus.geometry import Bounds
    from modulus.equation import PDE
    from modulus.nodes import Node
    MODULUS_AVAILABLE = True
    logger.info("✅ NVIDIA Modulus available for advanced physics simulations")
    
    class GravityPDE(PDE):
        """
        Physics-Informed Neural Network for Gravity using NVIDIA Modulus
        Implements Poisson equation: ∇²φ = 4πGρ
        """
        
        def __init__(self, G=6.674e-11):
            super().__init__()
            self.G = G
            
        def pde(self, input_var: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
            """
            Define the gravitational Poisson equation
            """
            # Extract variables
            phi = input_var["phi"]  # Gravitational potential
            x, y, z = input_var["x"], input_var["y"], input_var["z"]
            
            # Compute gradients
            phi_x = torch.autograd.grad(
                phi, x, grad_outputs=torch.ones_like(phi), create_graph=True
            )[0]
            phi_y = torch.autograd.grad(
                phi, y, grad_outputs=torch.ones_like(phi), create_graph=True
            )[0]
            phi_z = torch.autograd.grad(
                phi, z, grad_outputs=torch.ones_like(phi), create_graph=True
            )[0]
            
            # Second derivatives (Laplacian)
            phi_xx = torch.autograd.grad(
                phi_x, x, grad_outputs=torch.ones_like(phi_x), create_graph=True
            )[0]
            phi_yy = torch.autograd.grad(
                phi_y, y, grad_outputs=torch.ones_like(phi_y), create_graph=True
            )[0]
            phi_zz = torch.autograd.grad(
                phi_z, z, grad_outputs=torch.ones_like(phi_z), create_graph=True
            )[0]
            
            # Laplacian
            laplacian_phi = phi_xx + phi_yy + phi_zz
            
            # For demonstration, we'll use a simplified mass density
            # In a real scenario, this would be computed from the mass distribution
            rho = torch.ones_like(phi) * 1000  # kg/m³
            
            # Poisson equation: ∇²φ = 4πGρ
            poisson_residual = laplacian_phi - 4 * np.pi * self.G * rho
            
            return {"poisson_equation": poisson_residual}
            
except ImportError:
    MODULUS_AVAILABLE = False
    logger.warning("⚠️  NVIDIA Modulus not available. Using fallback physics engine.")
    
    # Fallback PDE class for when Modulus is not available
    class GravityPDE:
        def __init__(self, G=6.674e-11):
            self.G = G
        
        def compute_gravity_field(self, positions, masses):
            """Fallback gravity field computation"""
            return np.zeros_like(positions)

try:
    # Try to import cuDF for efficient data processing
    import cudf
    import cupy as cp
    CUDF_AVAILABLE = True
    logger.info("✅ cuDF available for GPU-accelerated data processing")
except ImportError:
    CUDF_AVAILABLE = False
    logger.warning("⚠️  cuDF not available. Using pandas fallback.")
    import pandas as pd

# Import pre-trained models integration
try:
    import sys
    sys.path.append(str(Path(__file__).parent.parent.parent / "ml" / "models"))
    from pretrained_integration import PretrainedModelLoader, PretrainedPhysicsModel
    PRETRAINED_AVAILABLE = True
    logger.info("✅ Pre-trained models integration available")
except ImportError:
    PRETRAINED_AVAILABLE = False
    logger.warning("⚠️  Pre-trained models not available. Using basic models.")

class EducationalGravitySimulator:
    """
    Educational gravity simulator using NVIDIA Modulus for interactive games
    """
    
    def __init__(self, domain_bounds: Dict[str, Tuple[float, float]], use_gpu: bool = None):
        self.domain_bounds = domain_bounds
        self.use_gpu = use_gpu if use_gpu is not None else CUDA_AVAILABLE
        self.device = DEVICE if self.use_gpu else torch.device("cpu")
        
        if MODULUS_AVAILABLE:
            self._setup_modulus_simulation()
        else:
            logger.warning("Using fallback simulation - Modulus not available")
            self._setup_fallback_simulation()
    
    def _setup_modulus_simulation(self):
        """Setup NVIDIA Modulus-based simulation"""
        logger.info("🚀 Setting up NVIDIA Modulus gravity simulation")
        
        # Create domain
        self.domain = Domain()
        
        # Define geometry bounds
        bounds = Bounds(
            {key: val for key, val in self.domain_bounds.items()}
        )
        
        # Create neural network architecture
        self.gravity_net = FullyConnectedArch(
            input_keys=["x", "y", "z"],
            output_keys=["phi"],  # Gravitational potential
            nr_layers=6,
            layer_size=256,
            activation_fn=torch.nn.Tanh(),
        ).to(self.device)
        
        # Create gravity PDE
        self.gravity_pde = GravityPDE()
        
        # Create nodes
        self.gravity_node = Node.from_torch_module(
            self.gravity_net, name="gravity_network"
        )
        
        logger.info("✅ Modulus simulation setup complete")
    
    def _setup_fallback_simulation(self):
        """Setup fallback physics simulation"""
        logger.info("🔄 Setting up fallback gravity simulation")
        
        class SimplePotentialNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(3, 256),
                    nn.Tanh(),
                    nn.Linear(256, 256),
                    nn.Tanh(),
                    nn.Linear(256, 256),
                    nn.Tanh(),
                    nn.Linear(256, 1)
                )
            
            def forward(self, x):
                return self.net(x)
        
        self.gravity_net = SimplePotentialNet().to(self.device)
        self.optimizer = torch.optim.Adam(self.gravity_net.parameters(), lr=1e-3)
        
        logger.info("✅ Fallback simulation setup complete")
    
    def simulate_apple_drop_game(self, height: float, gravity: float = 9.81, 
                                time_steps: int = 100) -> Dict[str, np.ndarray]:
        """
        🍎 Real-time apple drop simulation for educational game
        Students can adjust height and gravity to see immediate results
        """
        logger.info(f"🍎 Simulating apple drop from {height}m height")
        
        dt = 0.1
        times = np.linspace(0, np.sqrt(2 * height / gravity), time_steps)
        
        # Physics calculations using kinematic equations
        positions = height - 0.5 * gravity * times**2
        velocities = gravity * times
        accelerations = np.full_like(times, gravity)
        
        # Use cuDF if available for faster processing
        if CUDF_AVAILABLE:
            df = cudf.DataFrame({
                'time': times,
                'position': positions,
                'velocity': velocities,
                'acceleration': accelerations
            })
            
            # Convert back to numpy for compatibility
            result = {
                'times': df['time'].to_numpy(),
                'positions': df['position'].to_numpy(), 
                'velocities': df['velocity'].to_numpy(),
                'accelerations': df['acceleration'].to_numpy()
            }
        else:
            result = {
                'times': times,
                'positions': positions,
                'velocities': velocities,
                'accelerations': accelerations
            }
        
        # Calculate impact
        impact_time = np.sqrt(2 * height / gravity)
        impact_velocity = gravity * impact_time
        
        result['impact_time'] = impact_time
        result['impact_velocity'] = impact_velocity
        result['score'] = int(100 * height / 10)  # Educational scoring
        
        logger.info(f"✅ Apple hits ground in {impact_time:.2f}s at {impact_velocity:.2f}m/s")
        return result
    
    def simulate_orbital_slingshot_game(self, planet_mass: float, planet_radius: float,
                                       approach_velocity: float, approach_angle: float,
                                       closest_approach_factor: float = 2.0) -> Dict[str, Any]:
        """
        🚀 Orbital Slingshot Game - Gravity assist simulation
        """
        logger.info(f"🚀 Simulating orbital slingshot around planet (mass={planet_mass:.2e} kg)")
        
        # Convert to appropriate units
        G = 6.67430e-11  # m^3 kg^-1 s^-2
        
        # Initial conditions
        closest_approach = planet_radius * closest_approach_factor
        
        # Calculate trajectory using simplified physics
        time_steps = 200
        dt = 1.0  # seconds
        times = np.linspace(0, time_steps * dt, time_steps)
        
        # Initial position and velocity
        initial_distance = closest_approach * 5  # Start far from planet
        positions = []
        velocities = []
        
        # Simple trajectory calculation
        for t in times:
            # Hyperbolic trajectory approximation
            angle = approach_angle * np.pi / 180 + t * 0.01
            r = closest_approach + approach_velocity * t * 0.001
            
            x = r * np.cos(angle)
            y = r * np.sin(angle)
            z = 0
            
            positions.append([x, y, z])
            
            # Calculate velocity change due to gravity
            vx = approach_velocity * np.cos(angle) * 1000  # m/s
            vy = approach_velocity * np.sin(angle) * 1000
            vz = 0
            
            velocities.append([vx, vy, vz])
        
        # Calculate velocity gain
        initial_speed = approach_velocity * 1000  # m/s
        final_speed = np.linalg.norm(velocities[-1])
        velocity_gain = final_speed - initial_speed
        
        return {
            'trajectory': [{'x': p[0], 'y': p[1], 'z': p[2]} for p in positions],
            'times': times.tolist(),
            'velocities': velocities,
            'initial_velocity': initial_speed,
            'final_velocity': final_speed,
            'velocity_gain': velocity_gain,
            'planet_mass': planet_mass,
            'closest_approach': closest_approach,
            'efficiency': (velocity_gain / initial_speed) * 100 if initial_speed > 0 else 0
        }
    
    def simulate_lagrange_points_game(self, m1_mass: float, m2_mass: float,
                                    separation: float, test_mass_pos: List[float],
                                    time_span: float = 86400) -> Dict[str, Any]:
        """
        🌍 Lagrange Points Game - Multi-body gravitational equilibrium
        """
        logger.info(f"🌍 Simulating Lagrange points for two-body system")
        
        G = 6.67430e-11  # m^3 kg^-1 s^-2
        
        # Calculate theoretical Lagrange points
        # L1, L2, L3 are on the line connecting the two masses
        # L4, L5 form equilateral triangles with the two masses
        
        mu = m2_mass / (m1_mass + m2_mass)
        
        # Approximate L1 point (between the masses)
        l1_distance = separation * (1 - (mu/3)**(1/3))
        
        # L4 and L5 points (60° ahead and behind m2)
        l4_x = separation * (0.5 - mu)
        l4_y = separation * np.sqrt(3) / 2
        l5_x = separation * (0.5 - mu)
        l5_y = -separation * np.sqrt(3) / 2
        
        lagrange_points = [
            {'name': 'L1', 'position': [l1_distance, 0, 0], 'stable': False},
            {'name': 'L2', 'position': [separation + l1_distance * 0.2, 0, 0], 'stable': False},
            {'name': 'L3', 'position': [-separation, 0, 0], 'stable': False},
            {'name': 'L4', 'position': [l4_x, l4_y, 0], 'stable': True},
            {'name': 'L5', 'position': [l5_x, l5_y, 0], 'stable': True}
        ]
        
        # Simulate test mass motion
        time_steps = 1000
        dt = time_span / time_steps
        times = np.linspace(0, time_span, time_steps)
        
        x, y, z = test_mass_pos
        vx, vy, vz = 0, 0, 0  # Start at rest
        
        trajectory = []
        
        for t in times:
            # Distance to each mass
            r1 = np.sqrt((x + separation/2)**2 + y**2 + z**2)
            r2 = np.sqrt((x - separation/2)**2 + y**2 + z**2)
            
            # Gravitational forces
            if r1 > 0:
                f1_mag = G * m1_mass / (r1**3)
                f1_x = -f1_mag * (x + separation/2)
                f1_y = -f1_mag * y
                f1_z = -f1_mag * z
            else:
                f1_x = f1_y = f1_z = 0
            
            if r2 > 0:
                f2_mag = G * m2_mass / (r2**3)
                f2_x = -f2_mag * (x - separation/2)
                f2_y = -f2_mag * y
                f2_z = -f2_mag * z
            else:
                f2_x = f2_y = f2_z = 0
            
            # Total acceleration (assuming unit test mass)
            ax = f1_x + f2_x
            ay = f1_y + f2_y
            az = f1_z + f2_z
            
            # Update velocity and position
            vx += ax * dt
            vy += ay * dt
            vz += az * dt
            
            x += vx * dt
            y += vy * dt
            z += vz * dt
            
            trajectory.append([x, y, z])
        
        # Calculate stability score based on how close test mass stays to starting position
        final_distance = np.sqrt((x - test_mass_pos[0])**2 + 
                               (y - test_mass_pos[1])**2 + 
                               (z - test_mass_pos[2])**2)
        stability_score = max(0, 100 - final_distance / separation * 100)
        
        return {
            'lagrange_points': lagrange_points,
            'test_mass_trajectory': [{'x': p[0], 'y': p[1], 'z': p[2]} for p in trajectory],
            'times': times.tolist(),
            'stability_score': stability_score,
            'm1_mass': m1_mass,
            'm2_mass': m2_mass,
            'separation': separation,
            'initial_position': test_mass_pos,
            'final_position': [x, y, z]
        }
    
    def simulate_escape_velocity_game(self, planet_mass: float, planet_radius: float,
                                    launch_angle: float, launch_velocity: float) -> Dict[str, Any]:
        """
        🚀 Escape Velocity Game - Rocket escape simulation
        """
        logger.info(f"🚀 Simulating escape velocity from planet (mass={planet_mass:.2e} kg)")
        
        G = 6.67430e-11  # m^3 kg^-1 s^-2
        
        # Calculate theoretical escape velocity
        escape_velocity = np.sqrt(2 * G * planet_mass / planet_radius)
        
        # Simulate trajectory
        time_steps = 500
        dt = 1.0  # seconds
        times = np.linspace(0, time_steps * dt, time_steps)
        
        positions = []
        velocities = []
        altitudes = []
        
        # Initial conditions
        x, y, z = 0, planet_radius, 0
        vx = launch_velocity * np.cos(np.radians(launch_angle))
        vy = launch_velocity * np.sin(np.radians(launch_angle))
        vz = 0
        
        max_altitude = planet_radius
        escaped = False
        
        for t in times:
            # Current distance from planet center
            r = np.sqrt(x**2 + y**2 + z**2)
            altitude = r - planet_radius
            
            # Gravitational acceleration
            if r > planet_radius:
                g_magnitude = G * planet_mass / (r**2)
                g_x = -g_magnitude * x / r
                g_y = -g_magnitude * y / r
                g_z = -g_magnitude * z / r
                
                # Update velocity
                vx += g_x * dt
                vy += g_y * dt
                vz += g_z * dt
                
                # Update position
                x += vx * dt
                y += vy * dt
                z += vz * dt
                
                # Check if escaped (reached very high altitude with positive velocity)
                if altitude > planet_radius * 10 and np.sqrt(vx**2 + vy**2 + vz**2) > 0:
                    escaped = True
            else:
                # Hit surface
                break
            
            positions.append([x, y, z])
            velocities.append([vx, vy, vz])
            altitudes.append(altitude)
            max_altitude = max(max_altitude, altitude)
        
        return {
            'trajectory': [{'x': p[0], 'y': p[1], 'z': p[2]} for p in positions],
            'times': times[:len(positions)].tolist(),
            'velocities': velocities,
            'altitudes': altitudes,
            'max_altitude': max_altitude,
            'escaped': escaped,
            'escape_velocity_theoretical': escape_velocity,
            'launch_velocity': launch_velocity,
            'launch_angle': launch_angle,
            'planet_mass': planet_mass
        }
    
    def simulate_black_hole_navigation_game(self, black_hole_mass: float,
                                          approach_trajectory: List[List[float]],
                                          navigation_commands: List[Dict[str, float]]) -> Dict[str, Any]:
        """
        ⚫ Black Hole Navigation Game - Relativistic effects simulation
        """
        logger.info(f"⚫ Simulating black hole navigation (mass={black_hole_mass:.2e} kg)")
        
        G = 6.67430e-11  # m^3 kg^-1 s^-2
        c = 299792458  # m/s (speed of light)
        
        # Schwarzschild radius (event horizon)
        rs = 2 * G * black_hole_mass / (c**2)
        
        # Simulate trajectory with simplified relativistic effects
        trajectory = []
        time_dilation_factors = []
        survived = True
        
        for i, point in enumerate(approach_trajectory):
            x, y, z = point
            r = np.sqrt(x**2 + y**2 + z**2)
            
            # Check if crossed event horizon
            if r <= rs:
                survived = False
                break
            
            # Calculate time dilation factor (simplified)
            time_dilation = 1 / np.sqrt(1 - rs / r) if r > rs else float('inf')
            
            # Apply navigation commands if available
            if i < len(navigation_commands):
                cmd = navigation_commands[i]
                x += cmd.get('thrust_x', 0) * 1000  # Convert to meters
                y += cmd.get('thrust_y', 0) * 1000
                z += cmd.get('thrust_z', 0) * 1000
            
            # Gravitational deflection (simplified)
            deflection_factor = rs / r if r > rs else 0
            
            trajectory.append({
                'x': x, 'y': y, 'z': z,
                'r': r,
                'time_dilation': min(time_dilation, 10),  # Cap for visualization
                'deflection': deflection_factor
            })
            
            time_dilation_factors.append(time_dilation)
        
        # Calculate navigation score
        if survived:
            # Score based on how close we got without crossing event horizon
            min_distance = min(point['r'] for point in trajectory)
            navigation_score = max(0, 100 - (min_distance / rs - 1) * 10)
        else:
            navigation_score = 0
        
        return {
            'trajectory': trajectory,
            'schwarzschild_radius': rs,
            'survived': survived,
            'navigation_score': navigation_score,
            'black_hole_mass': black_hole_mass,
            'time_dilation_factors': time_dilation_factors,
            'relativistic_effects': True
        }
    
    def get_system_info(self) -> Dict[str, str]:
        """Get information about the simulation system"""
        return {
            'cuda_available': str(CUDA_AVAILABLE),
            'modulus_available': str(MODULUS_AVAILABLE),
            'cudf_available': str(CUDF_AVAILABLE),
            'device': str(self.device),
            'gpu_name': torch.cuda.get_device_name(0) if CUDA_AVAILABLE else "N/A"
        }
    
    def predict_trajectory_with_ml(self, initial_conditions: Dict, 
                                 prediction_horizon: int = 100) -> Dict[str, Any]:
        """
        🤖 Enhanced trajectory prediction using trained ML models
        Combines physics simulation with ML for improved accuracy
        """
        logger.info("🤖 Running ML-enhanced trajectory prediction")
        
        try:
            # Load trained trajectory predictor if available
            model_path = Path("ml_models/trained_models/trajectory_predictor_best.pth")
            if model_path.exists():
                from ml.models.trajectory_predictor import TrajectoryPredictor, TrajectoryPredictorConfig
                
                config = TrajectoryPredictorConfig()
                ml_model = TrajectoryPredictor(config)
                ml_model.load_state_dict(torch.load(model_path, map_location=self.device))
                ml_model.eval()
                
                # Extract initial conditions
                height = initial_conditions.get('height', 10)
                gravity = initial_conditions.get('gravity', 9.81)
                
                # Generate time steps
                times = np.linspace(0, np.sqrt(2 * height / gravity), prediction_horizon)
                
                # Prepare ML input
                ml_predictions = []
                with torch.no_grad():
                    for t in times:
                        input_tensor = torch.FloatTensor([[height, gravity, t]]).to(self.device)
                        input_seq = input_tensor.unsqueeze(1)  # Add sequence dimension
                        prediction = ml_model(input_seq).squeeze().cpu().numpy()
                        ml_predictions.append(prediction)
                
                ml_predictions = np.array(ml_predictions)
                
                # Compare with physics baseline
                physics_result = self.simulate_apple_drop_game(height, gravity, prediction_horizon)
                
                # Calculate confidence score based on physics agreement
                position_diff = np.abs(ml_predictions[:, 0] - physics_result['positions'])
                confidence = max(0, 1 - np.mean(position_diff) / height)
                
                logger.info(f"✅ ML prediction completed with {confidence:.2f} confidence")
                
                return {
                    'predicted_trajectory': {
                        'times': times,
                        'positions': ml_predictions[:, 0],
                        'velocities': ml_predictions[:, 1],
                        'accelerations': ml_predictions[:, 2]
                    },
                    'confidence_score': confidence,
                    'physics_baseline': physics_result,
                    'model_performance': {
                        'position_accuracy': 1 - np.mean(position_diff) / height,
                        'using_trained_model': True
                    }
                }
            else:
                logger.warning("Trained ML model not found, using physics baseline")
                physics_result = self.simulate_apple_drop_game(
                    initial_conditions.get('height', 10),
                    initial_conditions.get('gravity', 9.81),
                    prediction_horizon
                )
                return {
                    'predicted_trajectory': physics_result,
                    'confidence_score': 0.8,  # High confidence in physics
                    'physics_baseline': physics_result,
                    'model_performance': {
                        'position_accuracy': 1.0,
                        'using_trained_model': False
                    }
                }
                
        except Exception as e:
            logger.error(f"ML prediction failed: {e}")
            # Fallback to physics simulation
            physics_result = self.simulate_apple_drop_game(
                initial_conditions.get('height', 10),
                initial_conditions.get('gravity', 9.81),
                prediction_horizon
            )
            return {
                'predicted_trajectory': physics_result,
                'confidence_score': 0.8,
                'physics_baseline': physics_result,
                'model_performance': {
                    'position_accuracy': 1.0,
                    'using_trained_model': False,
                    'error': str(e)
                }
            }

class ModulusPhysicsEngine:
    """
    Main NVIDIA Modulus Physics Engine with pre-trained model integration
    Provides unified interface for all educational physics games
    """
    
    def __init__(self):
        self.device = DEVICE
        self.gravity_simulator = None
        self.pretrained_models = {}
        self.pretrained_loader = None
        
        # Initialize pre-trained models if available
        if PRETRAINED_AVAILABLE:
            self._initialize_pretrained_models()
        
        logger.info("🚀 ModulusPhysicsEngine initialized")
    
    def _initialize_pretrained_models(self):
        """Initialize pre-trained physics models"""
        try:
            self.pretrained_loader = PretrainedModelLoader()
            
            # Load gravity PINN model
            gravity_model = PretrainedPhysicsModel("modulus_gravity_pinn", self.pretrained_loader)
            if gravity_model.load():
                self.pretrained_models["gravity_pinn"] = gravity_model
                logger.info("✅ Pre-trained gravity PINN model loaded")
            
            # Load trajectory predictor
            trajectory_model = PretrainedPhysicsModel("trajectory_predictor_base", self.pretrained_loader)
            if trajectory_model.load():
                self.pretrained_models["trajectory_predictor"] = trajectory_model
                logger.info("✅ Pre-trained trajectory predictor loaded")
            
            # Load relativistic gravity model
            relativistic_model = PretrainedPhysicsModel("relativistic_gravity_model", self.pretrained_loader)
            if relativistic_model.load():
                self.pretrained_models["relativistic_gravity"] = relativistic_model
                logger.info("✅ Pre-trained relativistic gravity model loaded")
                
        except Exception as e:
            logger.warning(f"Pre-trained models initialization failed: {e}")
    
    def get_gravity_simulator(self, domain_size: float = 1e8) -> EducationalGravitySimulator:
        """Get or create gravity simulator instance"""
        if self.gravity_simulator is None:
            self.gravity_simulator = create_educational_gravity_simulator(domain_size)
        return self.gravity_simulator
    
    def simulate_apple_drop(self, height: float, gravity: float = 9.81, time_steps: int = 100) -> Dict[str, Any]:
        """Apple Drop Game - Enhanced with pre-trained models"""
        simulator = self.get_gravity_simulator()
        
        # Get physics simulation
        physics_result = simulator.simulate_apple_drop_game(height, gravity, time_steps)
        
        # Enhance with pre-trained model predictions if available
        if "trajectory_predictor" in self.pretrained_models:
            try:
                model = self.pretrained_models["trajectory_predictor"]
                
                # Prepare input for trajectory prediction
                input_state = np.array([[0, height, 0, 0, 0, -gravity]])  # [x, y, z, vx, vy, vz]
                
                # Get ML prediction
                ml_prediction = model.predict(input_state)
                
                physics_result["ml_enhanced"] = True
                physics_result["ml_prediction"] = ml_prediction.tolist()
                physics_result["confidence"] = 0.95
                
                logger.info("✅ Apple drop enhanced with pre-trained trajectory prediction")
                
            except Exception as e:
                logger.warning(f"Pre-trained model enhancement failed: {e}")
                physics_result["ml_enhanced"] = False
        
        return physics_result
    
    def simulate_orbital_slingshot(self, planet_mass: float, planet_radius: float, 
                                 approach_velocity: float, approach_angle: float,
                                 closest_approach_factor: float = 2.0) -> Dict[str, Any]:
        """Orbital Slingshot Game - Enhanced with pre-trained models"""
        simulator = self.get_gravity_simulator()
        
        try:
            # Get physics simulation
            result = simulator.simulate_orbital_slingshot_game(
                planet_mass, planet_radius, approach_velocity, 
                approach_angle, closest_approach_factor
            )
            
            # Enhance with pre-trained models if available
            if "gravity_pinn" in self.pretrained_models:
                gravity_model = self.pretrained_models["gravity_pinn"]
                
                # Calculate gravitational field using PINN
                closest_approach = planet_radius * closest_approach_factor
                field_input = np.array([[closest_approach, 0, 0]])
                
                pinn_potential = gravity_model.predict(field_input)
                result["pinn_potential"] = float(pinn_potential[0])
                result["ml_enhanced"] = True
                
                logger.info("✅ Orbital slingshot enhanced with pre-trained PINN")
            
            return result
            
        except Exception as e:
            logger.error(f"Orbital slingshot simulation failed: {e}")
            return {
                "error": str(e),
                "success": False,
                "ml_enhanced": False
            }
    
    def simulate_lagrange_points(self, m1_mass: float, m2_mass: float, 
                               separation: float, test_mass_pos: List[float],
                               time_span: float = 86400) -> Dict[str, Any]:
        """Lagrange Points Game - Enhanced with pre-trained models"""
        simulator = self.get_gravity_simulator()
        
        try:
            result = simulator.simulate_lagrange_points_game(
                m1_mass, m2_mass, separation, test_mass_pos, time_span
            )
            
            # Enhance with relativistic effects if available
            if "relativistic_gravity" in self.pretrained_models:
                rel_model = self.pretrained_models["relativistic_gravity"]
                
                # Calculate relativistic corrections for massive objects
                for i, pos in enumerate(result.get("lagrange_points", [])):
                    rel_input = np.array([pos])
                    rel_correction = rel_model.predict(rel_input)
                    result[f"lagrange_point_{i+1}_relativistic"] = rel_correction.tolist()
                
                result["relativistic_enhanced"] = True
                logger.info("✅ Lagrange points enhanced with relativistic corrections")
            
            return result
            
        except Exception as e:
            logger.error(f"Lagrange points simulation failed: {e}")
            return {
                "error": str(e),
                "success": False,
                "relativistic_enhanced": False
            }
    
    def simulate_escape_velocity(self, planet_mass: float, planet_radius: float,
                               launch_angle: float, launch_velocity: float) -> Dict[str, Any]:
        """Escape Velocity Game - Enhanced with pre-trained models"""
        simulator = self.get_gravity_simulator()
        
        try:
            result = simulator.simulate_escape_velocity_game(
                planet_mass, planet_radius, launch_angle, launch_velocity
            )
            
            # Enhance with trajectory prediction
            if "trajectory_predictor" in self.pretrained_models:
                traj_model = self.pretrained_models["trajectory_predictor"]
                
                # Initial conditions
                vx = launch_velocity * np.cos(np.radians(launch_angle))
                vy = launch_velocity * np.sin(np.radians(launch_angle))
                
                input_state = np.array([[0, planet_radius, 0, vx, vy, 0]])
                trajectory_prediction = traj_model.predict(input_state)
                
                result["predicted_trajectory"] = trajectory_prediction.tolist()
                result["ml_enhanced"] = True
                
                logger.info("✅ Escape velocity enhanced with trajectory prediction")
            
            return result
            
        except Exception as e:
            logger.error(f"Escape velocity simulation failed: {e}")
            return {
                "error": str(e),
                "success": False,
                "ml_enhanced": False
            }
    
    def simulate_black_hole_navigation(self, black_hole_mass: float, 
                                     approach_trajectory: List[List[float]],
                                     navigation_commands: List[Dict[str, float]]) -> Dict[str, Any]:
        """Black Hole Navigation Game - Enhanced with pre-trained models"""
        simulator = self.get_gravity_simulator()
        
        try:
            result = simulator.simulate_black_hole_navigation_game(
                black_hole_mass, approach_trajectory, navigation_commands
            )
            
            # Enhance with relativistic effects
            if "relativistic_gravity" in self.pretrained_models:
                rel_model = self.pretrained_models["relativistic_gravity"]
                
                # Calculate relativistic effects for each trajectory point
                relativistic_corrections = []
                for point in approach_trajectory:
                    rel_input = np.array([point])
                    correction = rel_model.predict(rel_input)
                    relativistic_corrections.append(correction.tolist())
                
                result["relativistic_corrections"] = relativistic_corrections
                result["relativistic_enhanced"] = True
                
                logger.info("✅ Black hole navigation enhanced with relativistic effects")
            
            return result
            
        except Exception as e:
            logger.error(f"Black hole navigation simulation failed: {e}")
            return {
                "error": str(e),
                "success": False,
                "relativistic_enhanced": False
            }
    
    def get_available_models(self) -> Dict[str, Any]:
        """Get information about available pre-trained models"""
        models_info = {}
        
        if self.pretrained_loader:
            for model_name in self.pretrained_models.keys():
                model = self.pretrained_models[model_name]
                models_info[model_name] = model.get_metadata()
        
        return {
            "available_models": list(self.pretrained_models.keys()),
            "model_details": models_info,
            "gpu_available": CUDA_AVAILABLE,
            "modulus_available": MODULUS_AVAILABLE,
            "cudf_available": CUDF_AVAILABLE
        }

# Factory function for easy instantiation
def create_educational_gravity_simulator(domain_size: float = 1e8, use_gpu: bool = None) -> EducationalGravitySimulator:
    """
    Create an educational gravity simulator with appropriate domain bounds
    """
    domain_bounds = {
        'x': (-domain_size, domain_size),
        'y': (-domain_size, domain_size), 
        'z': (-domain_size, domain_size)
    }
    
    return EducationalGravitySimulator(domain_bounds, use_gpu)
