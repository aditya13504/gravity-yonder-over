#!/usr/bin/env python3
"""
Training script for trajectory predictor model
Usage: python train_trajectory.py --epochs 500 --sequence_length 100
"""

import argparse
import torch
import numpy as np
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from ml.models.trajectory_predictor import TrajectoryPredictor, TrajectoryPredictorConfig

def generate_orbital_trajectories(n_trajectories=1000, steps=200):
    """Generate synthetic orbital trajectory data"""
    trajectories = []
    
    for _ in range(n_trajectories):
        # Random orbital parameters
        r0 = np.random.uniform(5, 50)  # Initial distance
        v0 = np.random.uniform(0.5, 3.0)  # Initial velocity
        mass = np.random.uniform(1e20, 1e30)  # Central mass
        
        dt = 0.1
        G = 6.674e-11
        
        positions = np.zeros((steps, 3))
        velocities = np.zeros((steps, 3))
        
        # Random initial orientation
        theta = np.random.uniform(0, 2*np.pi)
        phi = np.random.uniform(0, np.pi)
        
        positions[0] = [r0 * np.sin(phi) * np.cos(theta),
                       r0 * np.sin(phi) * np.sin(theta),
                       r0 * np.cos(phi)]
        
        # Perpendicular velocity for orbit
        velocities[0] = [v0 * np.cos(theta), v0 * np.sin(theta), 0]
        
        for i in range(1, steps):
            r = np.linalg.norm(positions[i-1])
            if r > 0:
                a = -G * mass / r**3 * positions[i-1]
                velocities[i] = velocities[i-1] + a * dt
                positions[i] = positions[i-1] + velocities[i] * dt
            else:
                break
        
        # Combine position and velocity
        trajectory = np.concatenate([positions, velocities], axis=1)
        trajectories.append(trajectory)
    
    return np.array(trajectories)

def train_trajectory_model(epochs=500, sequence_length=100, hidden_size=256, learning_rate=0.001):
    """Train the trajectory predictor model"""
    print("Generating training data...")
    trajectories = generate_orbital_trajectories(n_trajectories=2000, steps=sequence_length + 50)
    
    print("Initializing model...")
    model = TrajectoryPredictor(input_size=6, hidden_size=hidden_size, sequence_length=sequence_length)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.MSELoss()
    
    print(f"Starting training for {epochs} epochs...")
    losses = []
    
    for epoch in range(epochs):
        epoch_loss = 0
        n_batches = 0
        
        for trajectory in trajectories:
            if len(trajectory) < sequence_length + 10:
                continue
                
            # Random starting point
            start_idx = np.random.randint(0, len(trajectory) - sequence_length - 10)
            input_seq = trajectory[start_idx:start_idx + sequence_length]
            target = trajectory[start_idx + sequence_length:start_idx + sequence_length + 1]
            
            # Convert to tensors
            input_tensor = torch.FloatTensor(input_seq).unsqueeze(0)
            target_tensor = torch.FloatTensor(target)
            
            # Forward pass
            optimizer.zero_grad()
            output = model(input_tensor)
            loss = criterion(output.squeeze(), target_tensor.squeeze())
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            n_batches += 1
        
        avg_loss = epoch_loss / n_batches if n_batches > 0 else 0
        losses.append(avg_loss)
        
        if epoch % 50 == 0:
            print(f"Epoch {epoch}/{epochs}, Loss: {avg_loss:.6f}")
    
    return model, losses

def main():
    parser = argparse.ArgumentParser(description='Train trajectory predictor model')
    parser.add_argument('--epochs', type=int, default=500, help='Number of training epochs')
    parser.add_argument('--sequence_length', type=int, default=100, help='Input sequence length')
    parser.add_argument('--hidden_size', type=int, default=256, help='Hidden layer size')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--output', type=str, default='../trained_models/trajectory_predictor_v1.pth', 
                       help='Output model path')
    
    args = parser.parse_args()
    
    # Train model
    model, losses = train_trajectory_model(
        args.epochs, args.sequence_length, args.hidden_size, args.learning_rate
    )
    
    # Save model
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_path)
    
    print(f"Model saved to {output_path}")
    print(f"Final loss: {losses[-1]:.6f}")

if __name__ == "__main__":
    main()
