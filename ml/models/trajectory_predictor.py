"""
Neural network model for predicting particle trajectories in gravitational fields
Uses LSTM and attention mechanisms for accurate long-term trajectory prediction
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class TrajectoryPredictorConfig:
    """Configuration for trajectory prediction model"""
    input_dim: int = 6  # position (3) + velocity (3)
    hidden_dim: int = 128
    num_layers: int = 3
    dropout: float = 0.1
    attention_heads: int = 8
    sequence_length: int = 50
    prediction_horizon: int = 20
    learning_rate: float = 1e-3
    batch_size: int = 32
    epochs: int = 1000
    use_attention: bool = True
    use_physics_constraints: bool = True

class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism for trajectory sequences"""
    
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = np.sqrt(self.d_k)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, _ = x.size()
        
        # Linear transformations
        Q = self.W_q(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = torch.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        context = torch.matmul(attention_weights, V)
        context = context.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model
        )
        
        return self.W_o(context)

class TrajectoryLSTM(nn.Module):
    """LSTM-based trajectory prediction network"""
    
    def __init__(self, config: TrajectoryPredictorConfig):
        super(TrajectoryLSTM, self).__init__()
        self.config = config
        
        # Input projection
        self.input_projection = nn.Linear(config.input_dim, config.hidden_dim)
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=config.hidden_dim,
            hidden_size=config.hidden_dim,
            num_layers=config.num_layers,
            batch_first=True,
            dropout=config.dropout if config.num_layers > 1 else 0
        )
        
        # Attention mechanism
        if config.use_attention:
            self.attention = MultiHeadAttention(config.hidden_dim, config.attention_heads, config.dropout)
            self.norm1 = nn.LayerNorm(config.hidden_dim)
            
        # Feed-forward network
        self.feed_forward = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim * 2, config.hidden_dim)
        )
        self.norm2 = nn.LayerNorm(config.hidden_dim)
        
        # Output projection
        self.output_projection = nn.Linear(config.hidden_dim, config.input_dim)
        
    def forward(self, x: torch.Tensor, hidden: Optional[Tuple] = None) -> Tuple[torch.Tensor, Tuple]:
        """
        Forward pass
        
        Args:
            x: Input trajectory sequence [batch, seq_len, input_dim]
            hidden: Optional hidden state from previous step
            
        Returns:
            Tuple of (output, hidden_state)
        """
        # Input projection
        x = self.input_projection(x)
        
        # LSTM processing
        lstm_out, hidden = self.lstm(x, hidden)
        
        # Attention mechanism
        if self.config.use_attention:
            attended = self.attention(lstm_out)
            lstm_out = self.norm1(lstm_out + attended)
        
        # Feed-forward processing
        ff_out = self.feed_forward(lstm_out)
        output = self.norm2(lstm_out + ff_out)
        
        # Output projection
        output = self.output_projection(output)
        
        return output, hidden

class PhysicsConstrainedLoss(nn.Module):
    """Loss function with physics constraints for trajectory prediction"""
    
    def __init__(self, masses: List[float], physics_weight: float = 0.1):
        super(PhysicsConstrainedLoss, self).__init__()
        self.masses = torch.tensor(masses, dtype=torch.float32)
        self.physics_weight = physics_weight
        self.G = 6.67430e-11
        self.mse_loss = nn.MSELoss()
        
    def forward(self, predicted: torch.Tensor, target: torch.Tensor, 
                positions: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute physics-constrained loss
        
        Args:
            predicted: Predicted trajectory [batch, seq_len, 6]
            target: Target trajectory [batch, seq_len, 6]
            positions: Current positions for physics constraints [batch, seq_len, n_bodies, 3]
            
        Returns:
            Tuple of (total_loss, loss_components)
        """
        # Basic MSE loss
        mse_loss = self.mse_loss(predicted, target)
        
        # Physics constraints
        if self.physics_weight > 0:
            physics_loss = self._compute_physics_loss(predicted, positions)
        else:
            physics_loss = torch.tensor(0.0)
        
        # Energy conservation constraint
        energy_loss = self._compute_energy_conservation_loss(predicted, positions)
        
        # Momentum conservation constraint
        momentum_loss = self._compute_momentum_conservation_loss(predicted)
        
        # Total loss
        total_loss = (
            mse_loss + 
            self.physics_weight * physics_loss +
            0.01 * energy_loss +
            0.01 * momentum_loss
        )
        
        loss_components = {
            'mse': mse_loss,
            'physics': physics_loss,
            'energy': energy_loss,
            'momentum': momentum_loss
        }
        
        return total_loss, loss_components
    
    def _compute_physics_loss(self, trajectory: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """Compute physics-based loss using gravitational force equations"""
        # Extract positions and velocities
        pred_positions = trajectory[:, :, :3]  # [batch, seq, 3]
        pred_velocities = trajectory[:, :, 3:]  # [batch, seq, 3]
        
        # Compute accelerations from positions (gravitational forces)
        gravitational_acc = self._compute_gravitational_acceleration(positions)
        
        # Compute predicted accelerations from velocity changes
        dt = 1.0  # Assume unit time step
        pred_acc = torch.diff(pred_velocities, dim=1) / dt
        
        # Physics loss: difference between predicted and gravitational accelerations
        if pred_acc.shape[1] > 0 and gravitational_acc.shape[1] > 0:
            min_len = min(pred_acc.shape[1], gravitational_acc.shape[1])
            physics_loss = torch.mean((pred_acc[:, :min_len] - gravitational_acc[:, :min_len])**2)
        else:
            physics_loss = torch.tensor(0.0)
        
        return physics_loss
    
    def _compute_gravitational_acceleration(self, positions: torch.Tensor) -> torch.Tensor:
        """Compute gravitational accelerations between bodies"""
        batch_size, seq_len, n_bodies, _ = positions.shape
        accelerations = torch.zeros_like(positions)
        
        for i in range(n_bodies):
            for j in range(n_bodies):
                if i != j:
                    r_vec = positions[:, :, j] - positions[:, :, i]  # [batch, seq, 3]
                    r_mag = torch.norm(r_vec, dim=-1, keepdim=True) + 1e-10  # [batch, seq, 1]
                    
                    # Gravitational acceleration
                    acc_mag = self.G * self.masses[j] / (r_mag**2)  # [batch, seq, 1]
                    accelerations[:, :, i] += acc_mag * r_vec / r_mag
        
        return accelerations
    
    def _compute_energy_conservation_loss(self, trajectory: torch.Tensor, 
                                        positions: torch.Tensor) -> torch.Tensor:
        """Compute energy conservation constraint"""
        # Extract positions and velocities
        pred_positions = trajectory[:, :, :3]
        pred_velocities = trajectory[:, :, 3:]
        
        # Compute kinetic energy
        kinetic_energy = 0.5 * torch.sum(self.masses[0] * torch.sum(pred_velocities**2, dim=-1), dim=-1)
        
        # Compute potential energy (simplified for single body)
        # For multiple bodies, this would need to be more complex
        potential_energy = torch.zeros_like(kinetic_energy)
        
        # Total energy
        total_energy = kinetic_energy + potential_energy
        
        # Energy should be conserved (constant over time)
        energy_variance = torch.var(total_energy, dim=1)
        energy_loss = torch.mean(energy_variance)
        
        return energy_loss
    
    def _compute_momentum_conservation_loss(self, trajectory: torch.Tensor) -> torch.Tensor:
        """Compute momentum conservation constraint"""
        velocities = trajectory[:, :, 3:]  # [batch, seq, 3]
        
        # Total momentum (for isolated system should be conserved)
        total_momentum = torch.sum(self.masses[0] * velocities, dim=-1)  # [batch, seq]
        
        # Momentum should be conserved (constant over time)
        momentum_variance = torch.var(total_momentum, dim=1)
        momentum_loss = torch.mean(momentum_variance)
        
        return momentum_loss

class TrajectoryPredictor:
    """Main trajectory prediction class"""
    
    def __init__(self, config: TrajectoryPredictorConfig, masses: List[float], device: str = 'cuda'):
        self.config = config
        self.masses = masses
        self.device = device if torch.cuda.is_available() else 'cpu'
        
        # Initialize model
        self.model = TrajectoryLSTM(config).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.learning_rate)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, patience=50, factor=0.8
        )
        
        # Loss function
        if config.use_physics_constraints:
            self.criterion = PhysicsConstrainedLoss(masses)
        else:
            self.criterion = nn.MSELoss()
        
        # Scalers for normalization
        self.position_scaler = StandardScaler()
        self.velocity_scaler = StandardScaler()
        
        # Training history
        self.training_history = {
            'loss': [],
            'val_loss': [],
            'mse': [],
            'physics': [],
            'energy': [],
            'momentum': []
        }
        
    def prepare_data(self, trajectory_data: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare trajectory data for training
        
        Args:
            trajectory_data: Dictionary with 'positions', 'velocities', 'times'
            
        Returns:
            Tuple of (sequences, targets) for training
        """
        positions = trajectory_data['positions']  # [n_steps, n_bodies, 3]
        velocities = trajectory_data['velocities']  # [n_steps, n_bodies, 3]
        
        # Combine positions and velocities
        state_vectors = np.concatenate([positions, velocities], axis=-1)  # [n_steps, n_bodies, 6]
        
        # For single body prediction, flatten
        if len(self.masses) == 1:
            state_vectors = state_vectors[:, 0, :]  # [n_steps, 6]
        else:
            # For multi-body, reshape to [n_steps, n_bodies * 6]
            state_vectors = state_vectors.reshape(state_vectors.shape[0], -1)
        
        # Normalize data
        state_vectors = self._normalize_data(state_vectors)
        
        # Create sequences
        sequences = []
        targets = []
        
        for i in range(len(state_vectors) - self.config.sequence_length - self.config.prediction_horizon + 1):
            seq = state_vectors[i:i + self.config.sequence_length]
            target = state_vectors[i + self.config.sequence_length:
                                 i + self.config.sequence_length + self.config.prediction_horizon]
            
            sequences.append(seq)
            targets.append(target)
        
        return np.array(sequences), np.array(targets)
    
    def _normalize_data(self, data: np.ndarray) -> np.ndarray:
        """Normalize trajectory data"""
        # Split into positions and velocities
        if data.shape[-1] == 6:  # Single body
            positions = data[:, :3]
            velocities = data[:, 3:]
        else:  # Multi-body
            mid_point = data.shape[-1] // 2
            positions = data[:, :mid_point]
            velocities = data[:, mid_point:]
        
        # Fit and transform
        positions_norm = self.position_scaler.fit_transform(positions)
        velocities_norm = self.velocity_scaler.fit_transform(velocities)
        
        return np.concatenate([positions_norm, velocities_norm], axis=-1)
    
    def _denormalize_data(self, data: np.ndarray) -> np.ndarray:
        """Denormalize trajectory data"""
        if data.shape[-1] == 6:  # Single body
            positions = data[:, :3]
            velocities = data[:, 3:]
        else:  # Multi-body
            mid_point = data.shape[-1] // 2
            positions = data[:, :mid_point]
            velocities = data[:, mid_point:]
        
        # Inverse transform
        positions_denorm = self.position_scaler.inverse_transform(positions)
        velocities_denorm = self.velocity_scaler.inverse_transform(velocities)
        
        return np.concatenate([positions_denorm, velocities_denorm], axis=-1)
    
    def train(self, training_data: Dict[str, np.ndarray], validation_data: Optional[Dict[str, np.ndarray]] = None):
        """
        Train the trajectory prediction model
        
        Args:
            training_data: Training trajectory data
            validation_data: Optional validation trajectory data
        """
        logger.info("Preparing training data...")
        train_sequences, train_targets = self.prepare_data(training_data)
        
        # Create data loaders
        train_dataset = TensorDataset(
            torch.tensor(train_sequences, dtype=torch.float32),
            torch.tensor(train_targets, dtype=torch.float32)
        )
        train_loader = DataLoader(train_dataset, batch_size=self.config.batch_size, shuffle=True)
        
        # Validation data
        val_loader = None
        if validation_data:
            val_sequences, val_targets = self.prepare_data(validation_data)
            val_dataset = TensorDataset(
                torch.tensor(val_sequences, dtype=torch.float32),
                torch.tensor(val_targets, dtype=torch.float32)
            )
            val_loader = DataLoader(val_dataset, batch_size=self.config.batch_size)
        
        logger.info(f"Starting training for {self.config.epochs} epochs...")
        
        for epoch in range(self.config.epochs):
            # Training
            self.model.train()
            train_loss = 0.0
            train_components = {'mse': 0, 'physics': 0, 'energy': 0, 'momentum': 0}
            
            for batch_sequences, batch_targets in train_loader:
                batch_sequences = batch_sequences.to(self.device)
                batch_targets = batch_targets.to(self.device)
                
                self.optimizer.zero_grad()
                
                # Forward pass
                predictions, _ = self.model(batch_sequences)
                
                # Compute loss
                if self.config.use_physics_constraints:
                    # Mock positions for physics constraints
                    positions = batch_sequences[:, :, :3].unsqueeze(2)  # Add body dimension
                    loss, components = self.criterion(predictions, batch_targets, positions)
                    
                    for key, value in components.items():
                        train_components[key] += value.item()
                else:
                    loss = self.criterion(predictions, batch_targets)
                
                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                
                train_loss += loss.item()
            
            # Average losses
            train_loss /= len(train_loader)
            for key in train_components:
                train_components[key] /= len(train_loader)
            
            # Validation
            val_loss = 0.0
            if val_loader:
                self.model.eval()
                with torch.no_grad():
                    for batch_sequences, batch_targets in val_loader:
                        batch_sequences = batch_sequences.to(self.device)
                        batch_targets = batch_targets.to(self.device)
                        
                        predictions, _ = self.model(batch_sequences)
                        
                        if self.config.use_physics_constraints:
                            positions = batch_sequences[:, :, :3].unsqueeze(2)
                            loss, _ = self.criterion(predictions, batch_targets, positions)
                        else:
                            loss = self.criterion(predictions, batch_targets)
                        
                        val_loss += loss.item()
                
                val_loss /= len(val_loader)
                self.scheduler.step(val_loss)
            
            # Record history
            self.training_history['loss'].append(train_loss)
            self.training_history['val_loss'].append(val_loss)
            
            if self.config.use_physics_constraints:
                for key, value in train_components.items():
                    self.training_history[key].append(value)
            
            # Logging
            if epoch % 100 == 0:
                logger.info(f"Epoch {epoch}: Train Loss = {train_loss:.6f}, Val Loss = {val_loss:.6f}")
                if self.config.use_physics_constraints:
                    logger.info(f"  MSE: {train_components['mse']:.6f}, "
                               f"Physics: {train_components['physics']:.6f}, "
                               f"Energy: {train_components['energy']:.6f}, "
                               f"Momentum: {train_components['momentum']:.6f}")
        
        logger.info("Training completed")
    
    def predict(self, initial_sequence: np.ndarray, prediction_steps: int) -> np.ndarray:
        """
        Predict trajectory from initial sequence
        
        Args:
            initial_sequence: Initial trajectory sequence [seq_len, state_dim]
            prediction_steps: Number of steps to predict
            
        Returns:
            Predicted trajectory [prediction_steps, state_dim]
        """
        self.model.eval()
        
        # Normalize input
        initial_sequence_norm = self._normalize_data(initial_sequence)
        
        # Convert to tensor
        sequence = torch.tensor(initial_sequence_norm, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        predictions = []
        hidden = None
        
        with torch.no_grad():
            for _ in range(prediction_steps):
                # Predict next step
                output, hidden = self.model(sequence, hidden)
                next_step = output[:, -1:, :]  # Take last prediction
                
                # Add to predictions
                predictions.append(next_step.cpu().numpy()[0, 0])
                
                # Update sequence for next prediction
                sequence = torch.cat([sequence[:, 1:, :], next_step], dim=1)
        
        # Denormalize predictions
        predictions = np.array(predictions)
        predictions_denorm = self._denormalize_data(predictions)
        
        return predictions_denorm
    
    def save_model(self, filepath: str):
        """Save trained model"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'masses': self.masses,
            'position_scaler': self.position_scaler,
            'velocity_scaler': self.velocity_scaler,
            'training_history': self.training_history
        }, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load trained model"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.position_scaler = checkpoint['position_scaler']
        self.velocity_scaler = checkpoint['velocity_scaler']
        self.training_history = checkpoint['training_history']
        logger.info(f"Model loaded from {filepath}")
    
    def plot_training_history(self):
        """Plot training history"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # Loss curves
        axes[0, 0].plot(self.training_history['loss'], label='Train Loss')
        if self.training_history['val_loss']:
            axes[0, 0].plot(self.training_history['val_loss'], label='Val Loss')
        axes[0, 0].set_title('Training Loss')
        axes[0, 0].set_yscale('log')
        axes[0, 0].legend()
        
        if self.config.use_physics_constraints:
            # Physics constraints
            axes[0, 1].plot(self.training_history['mse'], label='MSE')
            axes[0, 1].plot(self.training_history['physics'], label='Physics')
            axes[0, 1].set_title('Loss Components')
            axes[0, 1].set_yscale('log')
            axes[0, 1].legend()
            
            # Energy and momentum conservation
            axes[1, 0].plot(self.training_history['energy'])
            axes[1, 0].set_title('Energy Conservation Loss')
            axes[1, 0].set_yscale('log')
            
            axes[1, 1].plot(self.training_history['momentum'])
            axes[1, 1].set_title('Momentum Conservation Loss')
            axes[1, 1].set_yscale('log')
        
        for ax in axes.flat:
            ax.set_xlabel('Epoch')
            ax.grid(True)
        
        plt.tight_layout()
        plt.show()

# Example usage
def create_sample_trajectory_data() -> Dict[str, np.ndarray]:
    """Create sample trajectory data for testing"""
    # Simple circular orbit
    n_steps = 1000
    dt = 60  # 1 minute steps
    
    # Orbital parameters
    radius = 7e6  # 7000 km altitude
    velocity = np.sqrt(3.986e14 / radius)  # Orbital velocity
    
    times = np.arange(n_steps) * dt
    angles = velocity * times / radius
    
    positions = np.zeros((n_steps, 1, 3))
    velocities = np.zeros((n_steps, 1, 3))
    
    positions[:, 0, 0] = radius * np.cos(angles)
    positions[:, 0, 1] = radius * np.sin(angles)
    
    velocities[:, 0, 0] = -velocity * np.sin(angles)
    velocities[:, 0, 1] = velocity * np.cos(angles)
    
    return {
        'positions': positions,
        'velocities': velocities,
        'times': times
    }

if __name__ == "__main__":
    # Example usage
    config = TrajectoryPredictorConfig(
        hidden_dim=128,
        num_layers=3,
        sequence_length=50,
        prediction_horizon=20,
        use_attention=True,
        use_physics_constraints=True
    )
    
    # Create predictor for single satellite
    masses = [1000.0]  # 1000 kg satellite
    predictor = TrajectoryPredictor(config, masses)
    
    # Generate sample data
    trajectory_data = create_sample_trajectory_data()
    
    # Train model
    predictor.train(trajectory_data)
    
    # Save model
    predictor.save_model("trajectory_predictor.pth")
    
    # Plot training history
    predictor.plot_training_history()
    
    # Test prediction
    initial_sequence = np.concatenate([
        trajectory_data['positions'][:50, 0],
        trajectory_data['velocities'][:50, 0]
    ], axis=1)
    
    predicted_trajectory = predictor.predict(initial_sequence, 100)
    print(f"Predicted trajectory shape: {predicted_trajectory.shape}")
