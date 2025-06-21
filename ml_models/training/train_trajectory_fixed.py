#!/usr/bin/env python3
"""
Training script for trajectory predictor model
Usage: python train_trajectory_fixed.py --epochs 500 --sequence_length 100
"""

import argparse
import torch
import numpy as np
from pathlib import Path
import sys
import matplotlib.pyplot as plt

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from ml.models.trajectory_predictor import TrajectoryPredictor, TrajectoryPredictorConfig

def generate_orbital_trajectories(n_trajectories=1000, steps=200):
    """Generate synthetic orbital trajectory data"""
    trajectories = []
    
    for _ in range(n_trajectories):
        # Random orbital parameters
        r0 = np.random.uniform(6.37e6, 5e7)  # Initial distance (6.37e6 = Earth radius)
        v0 = np.random.uniform(1000, 12000)  # Initial velocity
        mass_central = 5.97e24  # Earth mass
        mass_orbiting = np.random.uniform(100, 10000)  # Satellite mass
        
        dt = 60.0  # 1 minute time steps
        G = 6.674e-11
        
        # Initialize arrays
        positions = np.zeros((steps, 3))
        velocities = np.zeros((steps, 3))
        
        # Initial conditions (circular orbit approximation)
        positions[0] = [r0, 0, 0]
        velocities[0] = [0, v0, 0]
        
        # Simple gravitational simulation
        for i in range(1, steps):
            r_vec = positions[i-1]
            r_mag = np.linalg.norm(r_vec)
            
            if r_mag < 6.37e6:  # Hit Earth
                break
                
            # Gravitational acceleration
            a_grav = -G * mass_central * r_vec / r_mag**3
            
            # Update velocity and position (Euler integration)
            velocities[i] = velocities[i-1] + a_grav * dt
            positions[i] = positions[i-1] + velocities[i] * dt
        
        # Combine position and velocity for state vector
        trajectory = np.concatenate([positions[:i], velocities[:i]], axis=1)
        
        if len(trajectory) > 50:  # Only keep trajectories with reasonable length
            trajectories.append(trajectory)
    
    return trajectories

def prepare_training_data(trajectories, sequence_length=50):
    """Prepare training data from trajectories"""
    sequences = []
    targets = []
    
    for trajectory in trajectories:
        if len(trajectory) < sequence_length + 10:
            continue
            
        for start_idx in range(0, len(trajectory) - sequence_length - 5, 5):
            # Input sequence
            input_seq = trajectory[start_idx:start_idx + sequence_length]
            # Target (next 5 steps)
            target_seq = trajectory[start_idx + sequence_length:start_idx + sequence_length + 5]
            
            sequences.append(input_seq)
            targets.append(target_seq)
    
    return np.array(sequences), np.array(targets)

def train_trajectory_model(epochs=500, sequence_length=50, hidden_size=256, learning_rate=0.001):
    """Train the trajectory predictor model"""
    print("Generating training data...")
    trajectories = generate_orbital_trajectories(n_trajectories=500, steps=200)
    
    print("Preparing training sequences...")
    train_sequences, train_targets = prepare_training_data(trajectories, sequence_length)
    
    print(f"Generated {len(train_sequences)} training sequences")
    print(f"Input shape: {train_sequences.shape}")
    print(f"Target shape: {train_targets.shape}")
      # Initialize model
    config = TrajectoryPredictorConfig(
        input_dim=6,  # position (3) + velocity (3)
        hidden_dim=hidden_size,
        sequence_length=sequence_length,
        prediction_horizon=5,
        num_layers=3,
        dropout=0.1,
        learning_rate=learning_rate,
        batch_size=32,
        epochs=epochs,
        use_attention=True,
        use_physics_constraints=True
    )
    
    masses = [5.97e24]  # Earth mass for physics constraints
    model = TrajectoryPredictor(config, masses, device='cpu')
      # Prepare training data dict
    training_data = {
        'positions': train_sequences[:, :, :3],  # First 3 components are positions
        'velocities': train_sequences[:, :, 3:],  # Last 3 components are velocities
        'times': np.arange(sequence_length) * 60.0  # Time steps in seconds
    }
    
    print(f"Starting training for {epochs} epochs...")
    model.train(training_data)
    
    return model, model.training_history

def main():
    parser = argparse.ArgumentParser(description='Train trajectory predictor model')
    parser.add_argument('--epochs', type=int, default=500, help='Number of training epochs')
    parser.add_argument('--sequence_length', type=int, default=50, help='Input sequence length')
    parser.add_argument('--hidden_size', type=int, default=256, help='Hidden layer size')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--output', type=str, default='../trained_models/trajectory_predictor_v1.pth', 
                       help='Output model path')
    
    args = parser.parse_args()
    
    # Train model
    model, training_history = train_trajectory_model(
        args.epochs, args.sequence_length, args.hidden_size, args.learning_rate
    )
    
    # Save model
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(output_path))
    
    print(f"Model saved to {output_path}")
    print(f"Final loss: {training_history['loss'][-1]:.6f}")
    
    # Plot training curves
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(training_history['loss'], label='Training Loss')
    if 'val_loss' in training_history and training_history['val_loss']:
        plt.plot(training_history['val_loss'], label='Validation Loss')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 3, 2)
    if 'mse' in training_history and training_history['mse']:
        plt.plot(training_history['mse'], label='MSE')
    if 'physics' in training_history and training_history['physics']:
        plt.plot(training_history['physics'], label='Physics Loss')
    plt.title('Loss Components')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 3, 3)
    if 'energy' in training_history and training_history['energy']:
        plt.plot(training_history['energy'], label='Energy Conservation')
    if 'momentum' in training_history and training_history['momentum']:
        plt.plot(training_history['momentum'], label='Momentum Conservation')
    plt.title('Physics Constraints')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(output_path.parent / 'trajectory_training_curves.png')
    print(f"Training curves saved to {output_path.parent / 'trajectory_training_curves.png'}")

if __name__ == "__main__":
    main()
