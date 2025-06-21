#!/usr/bin/env python3
"""
Training script for PINN gravity model
Usage: python train_pinn.py --epochs 1000 --batch_size 32
"""

import argparse
import torch
import numpy as np
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from ml.models.pinn_gravity import GravityPINN, PINNConfig

def generate_training_data(n_samples=10000):
    """Generate synthetic training data for gravity simulations"""
    # Random initial conditions
    positions = np.random.uniform(-50, 50, (n_samples, 3))
    velocities = np.random.uniform(-10, 10, (n_samples, 3))
    masses = np.random.uniform(1e20, 1e30, (n_samples, 1))
    
    return positions, velocities, masses

def train_pinn_model(epochs=1000, batch_size=32, learning_rate=0.001):
    """Train the PINN gravity model"""
    print("Generating training data...")
    train_pos, train_vel, train_mass = generate_training_data()
    
    print("Initializing model...")
    model = PINNGravity(input_size=7, hidden_size=128, output_size=3)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    print(f"Starting training for {epochs} epochs...")
    losses = []
    
    for epoch in range(epochs):
        # Sample batch
        idx = np.random.choice(len(train_pos), batch_size, replace=False)
        batch_pos = train_pos[idx]
        batch_vel = train_vel[idx]
        batch_mass = train_mass[idx]
        
        # Training step
        loss = model.train_step(batch_pos, batch_vel, batch_mass)
        losses.append(loss)
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch}/{epochs}, Loss: {loss:.6f}")
    
    return model, losses

def main():
    parser = argparse.ArgumentParser(description='Train PINN gravity model')
    parser.add_argument('--epochs', type=int, default=1000, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--output', type=str, default='../trained_models/pinn_gravity_v1.pth', 
                       help='Output model path')
    
    args = parser.parse_args()
    
    # Train model
    model, losses = train_pinn_model(args.epochs, args.batch_size, args.learning_rate)
    
    # Save model
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_path)
    
    print(f"Model saved to {output_path}")
    print(f"Final loss: {losses[-1]:.6f}")

if __name__ == "__main__":
    main()
