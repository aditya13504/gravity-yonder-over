#!/usr/bin/env python3
"""
Training script for PINN gravity model
Usage: python train_pinn_fixed.py --epochs 1000 --batch_size 32
"""

import argparse
import torch
import numpy as np
from pathlib import Path
import sys
import matplotlib.pyplot as plt

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from ml.models.pinn_gravity import GravityPINN, PINNConfig

def generate_gravity_training_data(n_samples=10000):
    """Generate synthetic training data for gravity simulations"""
    # Generate random spatial coordinates
    x = np.random.uniform(-100, 100, (n_samples, 1))
    y = np.random.uniform(-100, 100, (n_samples, 1))
    z = np.random.uniform(-100, 100, (n_samples, 1))
    
    # Combine into position vectors
    positions = np.concatenate([x, y, z], axis=1)
    
    # Calculate analytical gravitational potential (for validation)
    # For a point mass at origin: V = -GM/r
    r = np.sqrt(x**2 + y**2 + z**2)
    G = 6.674e-11
    M = 1e24  # Mass of central body
    potential = -G * M / r
    
    return torch.tensor(positions, dtype=torch.float32), torch.tensor(potential, dtype=torch.float32)

def train_pinn_model(epochs=1000, batch_size=32, learning_rate=0.001):
    """Train the PINN gravity model"""
    print("Generating training data...")
    train_coords, train_potential = generate_gravity_training_data()
    
    print("Initializing PINN model...")
    config = PINNConfig(
        input_dim=3,
        hidden_dims=[128, 128, 128, 128],
        output_dim=1,
        learning_rate=learning_rate,
        epochs=epochs,
        batch_size=batch_size
    )
    model = GravityPINN(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    print(f"Starting training for {epochs} epochs...")
    losses = []
    physics_losses = []
    data_losses = []
    
    for epoch in range(epochs):
        # Sample batch
        idx = torch.randperm(len(train_coords))[:batch_size]
        batch_coords = train_coords[idx]
        batch_potential = train_potential[idx]
        
        # Forward pass
        optimizer.zero_grad()
        predicted_potential = model(batch_coords)
        
        # Data loss
        data_loss = torch.nn.functional.mse_loss(predicted_potential, batch_potential)
        
        # Physics loss (Poisson equation: ∇²φ = 4πGρ)
        physics_loss = model.physics_loss(batch_coords)
        
        # Combined loss
        total_loss = config.data_weight * data_loss + config.physics_weight * physics_loss
        
        # Backward pass
        total_loss.backward()
        optimizer.step()
        
        losses.append(total_loss.item())
        physics_losses.append(physics_loss.item())
        data_losses.append(data_loss.item())
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch}/{epochs}")
            print(f"  Total Loss: {total_loss.item():.6f}")
            print(f"  Data Loss: {data_loss.item():.6f}")
            print(f"  Physics Loss: {physics_loss.item():.6f}")
    
    return model, losses, physics_losses, data_losses

def main():
    parser = argparse.ArgumentParser(description='Train PINN gravity model')
    parser.add_argument('--epochs', type=int, default=1000, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--output', type=str, default='../trained_models/pinn_gravity_v1.pth', 
                       help='Output model path')
    
    args = parser.parse_args()
    
    # Train model
    model, losses, physics_losses, data_losses = train_pinn_model(
        args.epochs, args.batch_size, args.learning_rate
    )
    
    # Save model
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': model.config,
        'losses': losses,
        'physics_losses': physics_losses,
        'data_losses': data_losses
    }, output_path)
    
    print(f"Model saved to {output_path}")
    print(f"Final total loss: {losses[-1]:.6f}")
    print(f"Final physics loss: {physics_losses[-1]:.6f}")
    print(f"Final data loss: {data_losses[-1]:.6f}")
    
    # Plot training curves
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.plot(losses)
    plt.title('Total Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    plt.subplot(1, 3, 2)
    plt.plot(data_losses)
    plt.title('Data Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    plt.subplot(1, 3, 3)
    plt.plot(physics_losses)
    plt.title('Physics Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    plt.tight_layout()
    plt.savefig(output_path.parent / 'pinn_training_curves.png')
    print(f"Training curves saved to {output_path.parent / 'pinn_training_curves.png'}")

if __name__ == "__main__":
    main()
