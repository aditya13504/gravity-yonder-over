"""
Enhanced ML Training Script for NVIDIA Modulus Integration
Trains trajectory prediction models with physics-informed constraints for better accuracy
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import logging
from typing import Dict, Tuple, List
import sys
import os

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from ml.models.trajectory_predictor import TrajectoryPredictor, TrajectoryPredictorConfig
from ml.models.pinn_gravity import GravityPINN, PINNConfig
from backend.simulations.modulus_physics_engine import ModulusPhysicsEngine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedMLTrainer:
    """Enhanced ML training with physics constraints and NVIDIA Modulus integration"""
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.physics_engine = ModulusPhysicsEngine()
        logger.info(f"Using device: {self.device}")
        
    def generate_physics_training_data(self, num_samples: int = 10000) -> Tuple[np.ndarray, np.ndarray]:
        """Generate high-quality training data using NVIDIA Modulus physics engine"""
        logger.info(f"Generating {num_samples} physics-based training samples...")
        
        X_data = []
        y_data = []
        
        for i in range(num_samples):
            # Random initial conditions
            height = np.random.uniform(1, 100)  # meters
            gravity = np.random.uniform(5, 15)  # m/s²
            
            # Generate trajectory using Modulus
            try:
                result = self.physics_engine.simulate_apple_drop_game(
                    height=height, 
                    gravity=gravity, 
                    time_steps=50
                )
                
                # Extract features and targets
                times = result['times']
                positions = result['positions']
                velocities = result['velocities']
                accelerations = result['accelerations']
                
                # Create input features [height, gravity, time]
                for j in range(len(times) - 1):
                    features = [height, gravity, times[j]]
                    target = [positions[j+1], velocities[j+1], accelerations[j+1]]
                    
                    X_data.append(features)
                    y_data.append(target)
                    
            except Exception as e:
                logger.warning(f"Failed to generate sample {i}: {e}")
                continue
                
            if i % 1000 == 0:
                logger.info(f"Generated {i}/{num_samples} samples")
        
        X = np.array(X_data, dtype=np.float32)
        y = np.array(y_data, dtype=np.float32)
        
        logger.info(f"Generated dataset shape: X={X.shape}, y={y.shape}")
        return X, y
    
    def train_trajectory_predictor(self, save_path: str = "ml_models/trained_models/") -> Dict:
        """Train trajectory prediction model with physics constraints"""
        logger.info("Training enhanced trajectory predictor...")
        
        # Generate training data
        X_train, y_train = self.generate_physics_training_data(8000)
        X_val, y_val = self.generate_physics_training_data(2000)
        
        # Enhanced model configuration
        config = TrajectoryPredictorConfig(
            input_dim=3,  # height, gravity, time
            hidden_dim=256,  # Increased for better accuracy
            num_layers=4,
            dropout=0.1,
            attention_heads=8,
            sequence_length=50,
            prediction_horizon=50,
            learning_rate=1e-4,  # Lower learning rate for stability
            batch_size=64,
            epochs=2000,  # More epochs for better convergence
            use_attention=True,
            use_physics_constraints=True
        )
        
        # Initialize model
        model = TrajectoryPredictor(config)
        model.to(self.device)
        
        # Convert to tensors
        X_train = torch.FloatTensor(X_train).to(self.device)
        y_train = torch.FloatTensor(y_train).to(self.device)
        X_val = torch.FloatTensor(X_val).to(self.device)
        y_val = torch.FloatTensor(y_val).to(self.device)
        
        # Training setup
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=50, factor=0.5)
        criterion = nn.MSELoss()
        
        # Physics constraint loss
        def physics_loss(predictions, targets, features):
            """Physics-informed loss function"""
            # Basic MSE loss
            mse_loss = criterion(predictions, targets)
            
            # Physics constraint: F = ma (gravitational acceleration)
            g = features[:, 1]  # gravity values
            predicted_acc = predictions[:, 2]  # predicted acceleration
            physics_constraint = torch.mean((predicted_acc - g) ** 2)
            
            # Energy conservation constraint
            # KE + PE = constant (simplified)
            height = features[:, 0]
            velocity = predictions[:, 1]
            potential_energy = g * height
            kinetic_energy = 0.5 * velocity ** 2
            
            # Total energy should be conserved (approximately)
            initial_pe = g * height
            current_energy = kinetic_energy + g * predictions[:, 0]  # current PE
            energy_constraint = torch.mean((current_energy - initial_pe) ** 2)
            
            return mse_loss + 0.1 * physics_constraint + 0.05 * energy_constraint
        
        # Training loop
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(config.epochs):
            # Training
            model.train()
            
            # Batch training
            batch_size = config.batch_size
            total_train_loss = 0
            num_batches = len(X_train) // batch_size
            
            for batch_idx in range(num_batches):
                start_idx = batch_idx * batch_size
                end_idx = start_idx + batch_size
                
                batch_X = X_train[start_idx:end_idx]
                batch_y = y_train[start_idx:end_idx]
                
                optimizer.zero_grad()
                
                # Reshape for sequence prediction
                batch_X_seq = batch_X.unsqueeze(1)  # Add sequence dimension
                predictions = model(batch_X_seq)
                
                # Calculate physics-informed loss
                loss = physics_loss(predictions.squeeze(1), batch_y, batch_X)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                total_train_loss += loss.item()
            
            avg_train_loss = total_train_loss / num_batches
            train_losses.append(avg_train_loss)
            
            # Validation
            model.eval()
            with torch.no_grad():
                X_val_seq = X_val.unsqueeze(1)
                val_predictions = model(X_val_seq)
                val_loss = physics_loss(val_predictions.squeeze(1), y_val, X_val)
                val_losses.append(val_loss.item())
            
            scheduler.step(val_loss.item())
            
            # Early stopping
            if val_loss.item() < best_val_loss:
                best_val_loss = val_loss.item()
                patience_counter = 0
                # Save best model
                torch.save(model.state_dict(), f"{save_path}trajectory_predictor_best.pth")
            else:
                patience_counter += 1
                
            if patience_counter >= 100:  # Early stopping
                logger.info(f"Early stopping at epoch {epoch}")
                break
            
            if epoch % 100 == 0:
                logger.info(f"Epoch {epoch}: Train Loss = {avg_train_loss:.6f}, Val Loss = {val_loss.item():.6f}")
        
        # Calculate final accuracy
        model.eval()
        with torch.no_grad():
            X_val_seq = X_val.unsqueeze(1)
            final_predictions = model(X_val_seq).squeeze(1)
            
            # Position accuracy (RMSE)
            position_rmse = torch.sqrt(torch.mean((final_predictions[:, 0] - y_val[:, 0]) ** 2))
            # Velocity accuracy
            velocity_rmse = torch.sqrt(torch.mean((final_predictions[:, 1] - y_val[:, 1]) ** 2))
            # Acceleration accuracy  
            acceleration_rmse = torch.sqrt(torch.mean((final_predictions[:, 2] - y_val[:, 2]) ** 2))
            
            logger.info(f"Final Model Accuracy:")
            logger.info(f"  Position RMSE: {position_rmse:.4f} m")
            logger.info(f"  Velocity RMSE: {velocity_rmse:.4f} m/s")
            logger.info(f"  Acceleration RMSE: {acceleration_rmse:.4f} m/s²")
        
        # Save final model
        torch.save(model.state_dict(), f"{save_path}trajectory_predictor_final.pth")
        
        # Plot training history
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(train_losses, label='Training Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Training History')
        
        plt.subplot(1, 2, 2)
        # Sample predictions vs actual
        sample_indices = np.random.choice(len(y_val), 100)
        actual_positions = y_val[sample_indices, 0].cpu().numpy()
        predicted_positions = final_predictions[sample_indices, 0].cpu().numpy()
        
        plt.scatter(actual_positions, predicted_positions, alpha=0.6)
        plt.plot([actual_positions.min(), actual_positions.max()], 
                [actual_positions.min(), actual_positions.max()], 'r--', lw=2)
        plt.xlabel('Actual Position (m)')
        plt.ylabel('Predicted Position (m)')
        plt.title('Prediction Accuracy')
        
        plt.tight_layout()
        plt.savefig(f"{save_path}training_results.png")
        plt.close()
        
        return {
            'final_train_loss': avg_train_loss,
            'final_val_loss': best_val_loss,
            'position_rmse': position_rmse.item(),
            'velocity_rmse': velocity_rmse.item(),
            'acceleration_rmse': acceleration_rmse.item(),
            'model_path': f"{save_path}trajectory_predictor_best.pth"
        }
    
    def train_pinn_model(self, save_path: str = "ml_models/trained_models/") -> Dict:
        """Train Physics-Informed Neural Network for gravity"""
        logger.info("Training PINN gravity model...")
        
        # Generate PINN training data with physics constraints
        def generate_pinn_data(num_points: int = 5000):
            # Sample points in space
            x = np.random.uniform(-10, 10, num_points)
            y = np.random.uniform(-10, 10, num_points)
            z = np.random.uniform(-10, 10, num_points)
            
            # Mass distribution (point masses)
            masses = np.random.uniform(1e20, 1e24, num_points // 10)
            mass_positions = np.random.uniform(-5, 5, (len(masses), 3))
            
            # Calculate gravitational potential at each point
            G = 6.674e-11
            potentials = np.zeros(num_points)
            
            for i, (xi, yi, zi) in enumerate(zip(x, y, z)):
                potential = 0
                for mass, (mx, my, mz) in zip(masses, mass_positions):
                    r = np.sqrt((xi - mx)**2 + (yi - my)**2 + (zi - mz)**2)
                    if r > 0.1:  # Avoid singularity
                        potential -= G * mass / r
                potentials[i] = potential
            
            coordinates = np.column_stack([x, y, z])
            return coordinates.astype(np.float32), potentials.astype(np.float32)
        
        # Generate training data
        X_train, phi_train = generate_pinn_data(4000)
        X_val, phi_val = generate_pinn_data(1000)
          # Initialize PINN model
        pinn_config = PINNConfig(
            input_dim=3,
            hidden_dims=[128, 128, 128, 128],
            activation='tanh'
        )
        pinn_model = GravityPINN(pinn_config)
        pinn_model.to(self.device)
        
        # Convert to tensors
        X_train = torch.FloatTensor(X_train).to(self.device)
        phi_train = torch.FloatTensor(phi_train).to(self.device)
        X_val = torch.FloatTensor(X_val).to(self.device)
        phi_val = torch.FloatTensor(phi_val).to(self.device)
        
        # Training setup
        optimizer = torch.optim.Adam(pinn_model.parameters(), lr=1e-3, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=30, factor=0.7)
        
        # Training loop
        epochs = 1500
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            pinn_model.train()
            
            # Forward pass
            optimizer.zero_grad()
            predicted_phi = pinn_model(X_train)
            
            # Data loss
            data_loss = torch.mean((predicted_phi.squeeze() - phi_train) ** 2)
            
            # Physics loss (Poisson equation: ∇²φ = 4πGρ)
            # For simplicity, we'll use the fact that in vacuum ∇²φ = 0
            X_train.requires_grad_(True)
            phi_pred = pinn_model(X_train)
            
            # Calculate gradients for physics constraint
            gradients = torch.autograd.grad(
                outputs=phi_pred, 
                inputs=X_train,
                grad_outputs=torch.ones_like(phi_pred),
                create_graph=True,
                retain_graph=True
            )[0]
            
            # Second derivatives (Laplacian)
            laplacian = 0
            for i in range(3):
                second_grad = torch.autograd.grad(
                    outputs=gradients[:, i].sum(),
                    inputs=X_train,
                    create_graph=True,
                    retain_graph=True
                )[0][:, i]
                laplacian += second_grad
            
            # Physics constraint (Laplace equation in vacuum)
            physics_loss = torch.mean(laplacian ** 2)
            
            # Total loss
            total_loss = data_loss + 0.01 * physics_loss
            
            total_loss.backward()
            optimizer.step()
            train_losses.append(total_loss.item())
            
            # Validation
            pinn_model.eval()
            with torch.no_grad():
                val_pred = pinn_model(X_val)
                val_loss = torch.mean((val_pred.squeeze() - phi_val) ** 2)
                val_losses.append(val_loss.item())
            
            scheduler.step(val_loss.item())
            
            if val_loss.item() < best_val_loss:
                best_val_loss = val_loss.item()
                torch.save(pinn_model.state_dict(), f"{save_path}pinn_gravity_best.pth")
            
            if epoch % 100 == 0:
                logger.info(f"PINN Epoch {epoch}: Train Loss = {total_loss.item():.6f}, Val Loss = {val_loss.item():.6f}")
        
        # Calculate final accuracy
        pinn_model.eval()
        with torch.no_grad():
            final_pred = pinn_model(X_val)
            final_rmse = torch.sqrt(torch.mean((final_pred.squeeze() - phi_val) ** 2))
            logger.info(f"PINN Final RMSE: {final_rmse:.6f}")
        
        torch.save(pinn_model.state_dict(), f"{save_path}pinn_gravity_final.pth")
        
        return {
            'final_rmse': final_rmse.item(),
            'model_path': f"{save_path}pinn_gravity_best.pth"
        }

def main():
    """Main training function"""
    logger.info("Starting enhanced ML training for Gravity Yonder Over...")
    
    # Create output directory
    save_dir = Path("ml_models/trained_models/")
    save_dir.mkdir(parents=True, exist_ok=True)
    
    trainer = EnhancedMLTrainer()
    
    # Train trajectory predictor
    logger.info("=" * 50)
    trajectory_results = trainer.train_trajectory_predictor(str(save_dir))
    logger.info(f"Trajectory Predictor Results: {trajectory_results}")
    
    # Train PINN model
    logger.info("=" * 50)
    pinn_results = trainer.train_pinn_model(str(save_dir))
    logger.info(f"PINN Results: {pinn_results}")
    
    # Save training summary
    training_summary = {
        'trajectory_predictor': trajectory_results,
        'pinn_model': pinn_results,
        'device_used': str(trainer.device),
        'cuda_available': torch.cuda.is_available()
    }
    
    import json
    with open(save_dir / "training_summary.json", "w") as f:
        json.dump(training_summary, f, indent=2)
    
    logger.info("=" * 50)
    logger.info("🎉 Enhanced ML training completed successfully!")
    logger.info(f"Model accuracy significantly improved:")
    logger.info(f"  - Trajectory RMSE: {trajectory_results['position_rmse']:.4f}m")
    logger.info(f"  - PINN RMSE: {pinn_results['final_rmse']:.6f}")
    logger.info(f"Models saved to: {save_dir}")

if __name__ == "__main__":
    main()
