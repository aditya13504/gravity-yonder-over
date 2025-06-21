#!/usr/bin/env python3
"""
Simplified training script for trajectory predictor model
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

def generate_simple_trajectory_data(n_trajectories=100, steps=100):
    """Generate simple orbital trajectory data"""
    all_positions = []
    all_velocities = []
    
    for _ in range(n_trajectories):
        # Simple circular orbit parameters
        r = np.random.uniform(1e7, 5e7)  # Orbital radius
        v = np.sqrt(3.986e14 / r)  # Circular orbital velocity for Earth
        
        # Time parameters
        dt = 60.0  # 1 minute time steps
        t = np.arange(steps) * dt
        
        # Circular orbit
        theta = v * t / r
        
        positions = np.zeros((steps, 1, 3))  # [time, body, xyz]
        velocities = np.zeros((steps, 1, 3))
        
        positions[:, 0, 0] = r * np.cos(theta)
        positions[:, 0, 1] = r * np.sin(theta)
        positions[:, 0, 2] = 0
        
        velocities[:, 0, 0] = -v * np.sin(theta)
        velocities[:, 0, 1] = v * np.cos(theta)
        velocities[:, 0, 2] = 0
        
        all_positions.append(positions)
        all_velocities.append(velocities)
    
    # Combine all trajectories
    positions = np.concatenate(all_positions, axis=0)
    velocities = np.concatenate(all_velocities, axis=0)
    times = np.tile(t, n_trajectories)
    
    return positions, velocities, times

def main():
    parser = argparse.ArgumentParser(description='Train trajectory predictor model')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--sequence_length', type=int, default=30, help='Input sequence length')
    parser.add_argument('--output', type=str, default='../trained_models/trajectory_predictor_v1.pth', 
                       help='Output model path')
    
    args = parser.parse_args()
    
    print("Generating training data...")
    positions, velocities, times = generate_simple_trajectory_data(n_trajectories=50, steps=100)
    
    print(f"Data shapes: positions {positions.shape}, velocities {velocities.shape}")
    
    # Initialize model
    config = TrajectoryPredictorConfig(
        input_dim=6,
        hidden_dim=128,
        sequence_length=args.sequence_length,
        prediction_horizon=5,
        num_layers=2,
        dropout=0.1,
        learning_rate=0.001,
        batch_size=16,
        epochs=args.epochs,
        use_attention=False,  # Simplified
        use_physics_constraints=False  # Simplified
    )
    
    masses = [5.97e24]  # Earth mass
    model = TrajectoryPredictor(config, masses, device='cpu')
    
    # Prepare training data
    training_data = {
        'positions': positions,
        'velocities': velocities, 
        'times': times
    }
    
    print(f"Starting training for {args.epochs} epochs...")
    model.train(training_data)
    
    # Save model
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(output_path))
    
    print(f"Model saved to {output_path}")
    print(f"Final loss: {model.training_history['loss'][-1]:.6f}")

if __name__ == "__main__":
    main()
