"""
Physics-Informed Neural Network (PINN) for gravitational field prediction
Combines neural networks with physics constraints for accurate gravity modeling
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class PINNConfig:
    """Configuration for PINN training"""
    input_dim: int = 3  # x, y, z coordinates
    hidden_dims: List[int] = None
    output_dim: int = 1  # gravitational potential
    activation: str = 'tanh'
    learning_rate: float = 1e-3
    epochs: int = 10000
    batch_size: int = 1024
    physics_weight: float = 1.0
    data_weight: float = 1.0
    boundary_weight: float = 0.1
    
    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [64, 64, 64, 64]

class GravityPINN(nn.Module):
    """
    Physics-Informed Neural Network for gravitational field modeling
    """
    
    def __init__(self, config: PINNConfig):
        super(GravityPINN, self).__init__()
        self.config = config
        self.G = 6.67430e-11  # Gravitational constant
        
        # Build network architecture
        layers = []
        input_dim = config.input_dim
        
        for hidden_dim in config.hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            if config.activation == 'tanh':
                layers.append(nn.Tanh())
            elif config.activation == 'relu':
                layers.append(nn.ReLU())
            elif config.activation == 'swish':
                layers.append(nn.SiLU())
            input_dim = hidden_dim
        
        layers.append(nn.Linear(input_dim, config.output_dim))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Xavier initialization for better convergence"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network
        
        Args:
            x: Input coordinates [batch_size, 3] (x, y, z)
            
        Returns:
            Gravitational potential [batch_size, 1]
        """
        return self.network(x)
    
    def compute_gradients(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute gradients of potential for physics constraints
        
        Args:
            x: Input coordinates [batch_size, 3]
            
        Returns:
            Tuple of (potential, gradient) tensors
        """
        x.requires_grad_(True)
        potential = self.forward(x)
        
        # Compute gradient (gravitational field)
        grad_outputs = torch.ones_like(potential)
        gradients = torch.autograd.grad(
            outputs=potential,
            inputs=x,
            grad_outputs=grad_outputs,
            create_graph=True,
            retain_graph=True
        )[0]
        
        return potential, gradients
    
    def compute_laplacian(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute Laplacian for Poisson equation constraint
        
        Args:
            x: Input coordinates [batch_size, 3]
            
        Returns:
            Laplacian of potential [batch_size, 1]
        """
        x.requires_grad_(True)
        potential = self.forward(x)
        
        # First derivatives
        grad_outputs = torch.ones_like(potential)
        first_grads = torch.autograd.grad(
            outputs=potential,
            inputs=x,
            grad_outputs=grad_outputs,
            create_graph=True,
            retain_graph=True
        )[0]
        
        # Second derivatives (Laplacian components)
        laplacian = torch.zeros_like(potential)
        
        for i in range(3):  # x, y, z components
            second_grad = torch.autograd.grad(
                outputs=first_grads[:, i].sum(),
                inputs=x,
                create_graph=True,
                retain_graph=True
            )[0][:, i]
            laplacian += second_grad.unsqueeze(1)
        
        return laplacian
    
    def physics_loss(self, x: torch.Tensor, mass_density: float = 0.0) -> torch.Tensor:
        """
        Compute physics-based loss using Poisson equation: ∇²φ = 4πGρ
        
        Args:
            x: Input coordinates [batch_size, 3]
            mass_density: Mass density at the points (default 0 for vacuum)
            
        Returns:
            Physics loss scalar
        """
        laplacian = self.compute_laplacian(x)
        
        # Poisson equation in vacuum: ∇²φ = 0
        # For mass density ρ: ∇²φ = 4πGρ
        target_laplacian = 4 * np.pi * self.G * mass_density
        
        physics_residual = laplacian - target_laplacian
        return torch.mean(physics_residual ** 2)
    
    def boundary_loss(self, x_boundary: torch.Tensor, boundary_values: torch.Tensor) -> torch.Tensor:
        """
        Compute boundary condition loss
        
        Args:
            x_boundary: Boundary coordinates [batch_size, 3]
            boundary_values: Expected potential values at boundary
            
        Returns:
            Boundary loss scalar
        """
        boundary_potential = self.forward(x_boundary)
        return torch.mean((boundary_potential - boundary_values) ** 2)
    

class GravityPINNTrainer:
    """
    Trainer for Physics-Informed Neural Network
    """
    
    def __init__(self, config: PINNConfig, device: str = 'cuda'):
        self.config = config
        self.device = device if torch.cuda.is_available() else 'cpu'
        
        # Initialize model
        self.model = GravityPINN(config).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.learning_rate)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, patience=1000, factor=0.8
        )
        
        # Loss history
        self.loss_history = {
            'total': [],
            'physics': [],
            'data': [],
            'boundary': []
        }
        
        self.G = 6.67430e-11
        
    def generate_training_data(self, masses: List[float], positions: List[Tuple],
                              domain_bounds: Tuple[float, float], n_points: int = 10000) -> Dict[str, torch.Tensor]:
        """
        Generate training data for the PINN
        
        Args:
            masses: List of mass values
            positions: List of (x, y, z) mass positions
            domain_bounds: (min_coord, max_coord) for domain
            n_points: Number of training points
            
        Returns:
            Dictionary with training data tensors
        """
        # Generate random points in domain
        x_min, x_max = domain_bounds
        coords = torch.rand(n_points, 3) * (x_max - x_min) + x_min
        
        # Compute analytical gravitational potential
        potential = torch.zeros(n_points, 1)
        
        for mass, pos in zip(masses, positions):
            pos_tensor = torch.tensor(pos, dtype=torch.float32)
            r_vec = coords - pos_tensor.unsqueeze(0)
            r_mag = torch.norm(r_vec, dim=1) + 1e-10  # Softening
            potential += (-self.G * mass / r_mag).unsqueeze(1)
        
        # Generate physics points (for PDE constraint)
        physics_coords = torch.rand(n_points // 2, 3) * (x_max - x_min) + x_min
        
        # Generate boundary points
        boundary_coords = self._generate_boundary_points(domain_bounds, n_points // 10)
        
        # Compute density field for Poisson equation
        density = self._compute_density_field(masses, positions, physics_coords)
        
        return {
            'data_coords': coords.to(self.device),
            'data_potential': potential.to(self.device),
            'physics_coords': physics_coords.to(self.device),
            'physics_density': density.to(self.device),
            'boundary_coords': boundary_coords.to(self.device)
        }
    
    def _generate_boundary_points(self, domain_bounds: Tuple[float, float], n_points: int) -> torch.Tensor:
        """Generate points on domain boundary"""
        x_min, x_max = domain_bounds
        boundary_points = []
        
        # Generate points on each face of the cube
        for i in range(3):  # x, y, z dimensions
            for bound in [x_min, x_max]:
                # Random points on face
                face_points = torch.rand(n_points // 12, 3) * (x_max - x_min) + x_min
                face_points[:, i] = bound
                boundary_points.append(face_points)
        
        return torch.cat(boundary_points, dim=0)
    
    def _compute_density_field(self, masses: List[float], positions: List[Tuple],
                              coords: torch.Tensor) -> torch.Tensor:
        """Compute mass density field for Poisson equation"""
        density = torch.zeros(len(coords), 1)
        
        # Add point masses as delta functions (approximated by Gaussians)
        for mass, pos in zip(masses, positions):
            pos_tensor = torch.tensor(pos, dtype=torch.float32)
            r_vec = coords - pos_tensor.unsqueeze(0)
            r_mag = torch.norm(r_vec, dim=1)
            
            # Gaussian approximation of delta function
            sigma = 1e6  # Width of Gaussian
            gaussian = mass * torch.exp(-0.5 * (r_mag / sigma)**2) / (sigma**3 * (2 * np.pi)**1.5)
            density += gaussian.unsqueeze(1)
        
        return density
    
    def compute_physics_loss(self, coords: torch.Tensor, density: torch.Tensor) -> torch.Tensor:
        """
        Compute physics-based loss using Poisson equation
        ∇²φ = 4πGρ
        
        Args:
            coords: Physics constraint points
            density: Mass density at constraint points
            
        Returns:
            Physics loss tensor
        """
        laplacian = self.model.compute_laplacian(coords)
        
        # Poisson equation constraint
        poisson_residual = laplacian - 4 * np.pi * self.G * density
        
        return torch.mean(poisson_residual**2)
    
    def compute_data_loss(self, coords: torch.Tensor, target_potential: torch.Tensor) -> torch.Tensor:
        """
        Compute data fitting loss
        
        Args:
            coords: Data coordinates
            target_potential: Target potential values
            
        Returns:
            Data loss tensor
        """
        predicted_potential = self.model(coords)
        return torch.mean((predicted_potential - target_potential)**2)
    
    def compute_boundary_loss(self, boundary_coords: torch.Tensor) -> torch.Tensor:
        """
        Compute boundary condition loss (potential goes to zero at infinity)
        
        Args:
            boundary_coords: Boundary coordinates
            
        Returns:
            Boundary loss tensor
        """
        boundary_potential = self.model(boundary_coords)
        
        # Potential should be small at domain boundaries
        # (approximating infinity condition)
        return torch.mean(boundary_potential**2)
    
    def train_step(self, training_data: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Perform one training step
        
        Args:
            training_data: Dictionary with training tensors
            
        Returns:
            Dictionary with loss values
        """
        self.optimizer.zero_grad()
        
        # Compute individual losses
        data_loss = self.compute_data_loss(
            training_data['data_coords'],
            training_data['data_potential']
        )
        
        physics_loss = self.compute_physics_loss(
            training_data['physics_coords'],
            training_data['physics_density']
        )
        
        boundary_loss = self.compute_boundary_loss(
            training_data['boundary_coords']
        )
        
        # Weighted total loss
        total_loss = (
            self.config.data_weight * data_loss +
            self.config.physics_weight * physics_loss +
            self.config.boundary_weight * boundary_loss
        )
        
        # Backward pass
        total_loss.backward()
        self.optimizer.step()
        
        return {
            'total': total_loss.item(),
            'data': data_loss.item(),
            'physics': physics_loss.item(),
            'boundary': boundary_loss.item()
        }
    
    def train(self, training_data: Dict[str, torch.Tensor], validation_data: Optional[Dict[str, torch.Tensor]] = None):
        """
        Train the PINN model
        
        Args:
            training_data: Training data dictionary
            validation_data: Optional validation data
        """
        logger.info(f"Starting PINN training for {self.config.epochs} epochs")
        
        for epoch in range(self.config.epochs):
            # Training step
            losses = self.train_step(training_data)
            
            # Record losses
            for key, value in losses.items():
                self.loss_history[key].append(value)
            
            # Update learning rate
            self.scheduler.step(losses['total'])
            
            # Logging
            if epoch % 1000 == 0:
                logger.info(f"Epoch {epoch}: Total Loss = {losses['total']:.6f}, "
                           f"Data Loss = {losses['data']:.6f}, "
                           f"Physics Loss = {losses['physics']:.6f}, "
                           f"Boundary Loss = {losses['boundary']:.6f}")
                
                # Validation
                if validation_data:
                    val_loss = self._validate(validation_data)
                    logger.info(f"Validation Loss: {val_loss:.6f}")
        
        logger.info("Training completed")
    
    def _validate(self, validation_data: Dict[str, torch.Tensor]) -> float:
        """Compute validation loss"""
        self.model.eval()
        with torch.no_grad():
            val_loss = self.compute_data_loss(
                validation_data['data_coords'],
                validation_data['data_potential']
            )
        self.model.train()
        return val_loss.item()
    
    def predict_field(self, coordinates: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict gravitational field at given coordinates
        
        Args:
            coordinates: Array of (x, y, z) coordinates [N, 3]
            
        Returns:
            Tuple of (potential, field) arrays
        """
        self.model.eval()
        
        coords_tensor = torch.tensor(coordinates, dtype=torch.float32).to(self.device)
        
        with torch.no_grad():
            potential, gradients = self.model.compute_gradients(coords_tensor)
            
            # Gravitational field is negative gradient of potential
            field = -gradients
        
        self.model.train()
        
        return potential.cpu().numpy(), field.cpu().numpy()
    
    def save_model(self, filepath: str):
        """Save trained model"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'loss_history': self.loss_history
        }, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load trained model"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.loss_history = checkpoint['loss_history']
        logger.info(f"Model loaded from {filepath}")
    
    def plot_training_history(self):
        """Plot training loss history"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        epochs = range(len(self.loss_history['total']))
        
        # Total loss
        axes[0, 0].plot(epochs, self.loss_history['total'])
        axes[0, 0].set_title('Total Loss')
        axes[0, 0].set_yscale('log')
        
        # Data loss
        axes[0, 1].plot(epochs, self.loss_history['data'])
        axes[0, 1].set_title('Data Loss')
        axes[0, 1].set_yscale('log')
        
        # Physics loss
        axes[1, 0].plot(epochs, self.loss_history['physics'])
        axes[1, 0].set_title('Physics Loss')
        axes[1, 0].set_yscale('log')
        
        # Boundary loss
        axes[1, 1].plot(epochs, self.loss_history['boundary'])
        axes[1, 1].set_title('Boundary Loss')
        axes[1, 1].set_yscale('log')
        
        for ax in axes.flat:
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.grid(True)
        
        plt.tight_layout()
        plt.show()

# Example usage
def train_earth_moon_pinn():
    """Example: Train PINN for Earth-Moon system"""
    config = PINNConfig(
        hidden_dims=[128, 128, 128, 128],
        learning_rate=1e-3,
        epochs=20000,
        physics_weight=1.0,
        data_weight=1.0,
        boundary_weight=0.1
    )
    
    trainer = GravityPINNTrainer(config)
    
    # Earth-Moon system
    masses = [5.972e24, 7.35e22]  # Earth, Moon
    positions = [(0, 0, 0), (384400000, 0, 0)]  # Earth at origin, Moon at distance
    domain_bounds = (-6e8, 6e8)  # 600,000 km domain
    
    # Generate training data
    training_data = trainer.generate_training_data(masses, positions, domain_bounds)
    
    # Train model
    trainer.train(training_data)
    
    # Save model
    trainer.save_model("earth_moon_pinn.pth")
    
    # Plot training history
    trainer.plot_training_history()
    
    return trainer

if __name__ == "__main__":
    # Train example Earth-Moon PINN
    trainer = train_earth_moon_pinn()
    
    # Test prediction
    test_coords = np.array([[1e8, 0, 0], [2e8, 0, 0], [3e8, 0, 0]])
    potential, field = trainer.predict_field(test_coords)
    
    print("Test predictions:")
    for i, (coord, pot, f) in enumerate(zip(test_coords, potential, field)):
        print(f"Point {i+1}: {coord} -> Potential: {pot[0]:.2e}, Field: {f}")
