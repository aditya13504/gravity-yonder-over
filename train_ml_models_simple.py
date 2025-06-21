"""
Simple Enhanced ML Training Script for Gravity Yonder Over
Trains trajectory prediction models with better accuracy
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import logging
import sys
import os

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleTrajectoryPredictor(nn.Module):
    """Simple but accurate trajectory prediction model"""
    
    def __init__(self, input_dim=3, hidden_dim=256, output_dim=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, x):
        return self.net(x)

def generate_physics_data(num_samples=10000):
    """Generate accurate physics training data"""
    logger.info(f"Generating {num_samples} physics training samples...")
    
    X_data = []
    y_data = []
    
    for i in range(num_samples):
        # Random parameters
        height = np.random.uniform(1, 100)  # meters
        gravity = np.random.uniform(5, 15)   # m/s²
        time = np.random.uniform(0, np.sqrt(2 * height / gravity))
        
        # Physics calculations
        position = height - 0.5 * gravity * time**2
        velocity = gravity * time
        acceleration = gravity
        
        # Only include valid data (above ground)
        if position >= 0:
            X_data.append([height, gravity, time])
            y_data.append([position, velocity, acceleration])
        
        if i % 1000 == 0:
            logger.info(f"Generated {i}/{num_samples} samples")
    
    return np.array(X_data, dtype=np.float32), np.array(y_data, dtype=np.float32)

def train_enhanced_model():
    """Train enhanced trajectory prediction model"""
    logger.info("Training enhanced trajectory predictor...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Generate training data
    X_train, y_train = generate_physics_data(8000)
    X_val, y_val = generate_physics_data(2000)
    
    # Convert to tensors
    X_train = torch.FloatTensor(X_train).to(device)
    y_train = torch.FloatTensor(y_train).to(device)
    X_val = torch.FloatTensor(X_val).to(device)
    y_val = torch.FloatTensor(y_val).to(device)
    
    # Initialize model
    model = SimpleTrajectoryPredictor(input_dim=3, hidden_dim=256, output_dim=3)
    model.to(device)
    
    # Training setup
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=50, factor=0.5)
    criterion = nn.MSELoss()
    
    # Physics-informed loss function
    def physics_loss(predictions, targets, features):
        # Basic MSE
        mse = criterion(predictions, targets)
        
        # Physics constraint: acceleration should equal gravity
        gravity = features[:, 1]
        predicted_acc = predictions[:, 2]
        physics_constraint = torch.mean((predicted_acc - gravity) ** 2)
        
        return mse + 0.1 * physics_constraint
    
    # Training loop
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience = 0
    
    epochs = 1000
    batch_size = 64
    
    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        num_batches = len(X_train) // batch_size
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = start_idx + batch_size
            
            batch_X = X_train[start_idx:end_idx]
            batch_y = y_train[start_idx:end_idx]
            
            optimizer.zero_grad()
            predictions = model(batch_X)
            loss = physics_loss(predictions, batch_y, batch_X)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_train_loss += loss.item()
        
        avg_train_loss = total_train_loss / num_batches
        train_losses.append(avg_train_loss)
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_predictions = model(X_val)
            val_loss = physics_loss(val_predictions, y_val, X_val)
            val_losses.append(val_loss.item())
        
        scheduler.step(val_loss.item())
        
        # Early stopping
        if val_loss.item() < best_val_loss:
            best_val_loss = val_loss.item()
            patience = 0
            # Save best model
            save_dir = project_root / "ml_models" / "trained_models"
            save_dir.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), save_dir / "enhanced_trajectory_predictor.pth")
        else:
            patience += 1
            
        if patience >= 100:
            logger.info(f"Early stopping at epoch {epoch}")
            break
        
        if epoch % 100 == 0:
            logger.info(f"Epoch {epoch}: Train Loss = {avg_train_loss:.6f}, Val Loss = {val_loss.item():.6f}")
    
    # Calculate final accuracy
    model.eval()
    with torch.no_grad():
        final_predictions = model(X_val)
        
        # Position accuracy (RMSE)
        position_rmse = torch.sqrt(torch.mean((final_predictions[:, 0] - y_val[:, 0]) ** 2))
        velocity_rmse = torch.sqrt(torch.mean((final_predictions[:, 1] - y_val[:, 1]) ** 2))
        acceleration_rmse = torch.sqrt(torch.mean((final_predictions[:, 2] - y_val[:, 2]) ** 2))
        
        logger.info(f"Final Enhanced Model Accuracy:")
        logger.info(f"  Position RMSE: {position_rmse:.4f} m")
        logger.info(f"  Velocity RMSE: {velocity_rmse:.4f} m/s")
        logger.info(f"  Acceleration RMSE: {acceleration_rmse:.4f} m/s²")
    
    # Create visualization
    plt.figure(figsize=(15, 5))
    
    # Training history
    plt.subplot(1, 3, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training History')
    
    # Position predictions
    plt.subplot(1, 3, 2)
    sample_indices = np.random.choice(len(y_val), 200)
    actual_pos = y_val[sample_indices, 0].cpu().numpy()
    predicted_pos = final_predictions[sample_indices, 0].cpu().numpy()
    
    plt.scatter(actual_pos, predicted_pos, alpha=0.6)
    plt.plot([actual_pos.min(), actual_pos.max()], [actual_pos.min(), actual_pos.max()], 'r--', lw=2)
    plt.xlabel('Actual Position (m)')
    plt.ylabel('Predicted Position (m)')
    plt.title('Position Accuracy')
    
    # Velocity predictions
    plt.subplot(1, 3, 3)
    actual_vel = y_val[sample_indices, 1].cpu().numpy()
    predicted_vel = final_predictions[sample_indices, 1].cpu().numpy()
    
    plt.scatter(actual_vel, predicted_vel, alpha=0.6)
    plt.plot([actual_vel.min(), actual_vel.max()], [actual_vel.min(), actual_vel.max()], 'r--', lw=2)
    plt.xlabel('Actual Velocity (m/s)')
    plt.ylabel('Predicted Velocity (m/s)')
    plt.title('Velocity Accuracy')
    
    plt.tight_layout()
    save_dir = project_root / "ml_models" / "trained_models"
    plt.savefig(save_dir / "enhanced_training_results.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # Save training summary
    import json
    summary = {
        'model_type': 'Enhanced Trajectory Predictor',
        'final_train_loss': avg_train_loss,
        'final_val_loss': best_val_loss,
        'position_rmse': position_rmse.item(),
        'velocity_rmse': velocity_rmse.item(),
        'acceleration_rmse': acceleration_rmse.item(),
        'epochs_trained': epoch + 1,
        'device_used': str(device),
        'model_file': 'enhanced_trajectory_predictor.pth'
    }
    
    with open(save_dir / "enhanced_training_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    return summary

def main():
    """Main training function"""
    logger.info("🚀 Starting Enhanced ML Training for Gravity Yonder Over")
    logger.info("=" * 60)
    
    try:
        # Train enhanced model
        results = train_enhanced_model()
        
        logger.info("\n" + "🎉" * 20)
        logger.info("✅ ENHANCED ML TRAINING COMPLETED SUCCESSFULLY!")
        logger.info("🎉" * 20)
        
        logger.info(f"\n📊 FINAL RESULTS:")
        logger.info(f"  🎯 Position RMSE: {results['position_rmse']:.4f} m")
        logger.info(f"  🏃 Velocity RMSE: {results['velocity_rmse']:.4f} m/s")
        logger.info(f"  ⚡ Acceleration RMSE: {results['acceleration_rmse']:.4f} m/s²")
        logger.info(f"  🔥 Final Loss: {results['final_val_loss']:.6f}")
        logger.info(f"  🎲 Device Used: {results['device_used']}")
        
        logger.info(f"\n💾 Models saved to: ml_models/trained_models/")
        logger.info(f"📈 Training visualization: enhanced_training_results.png")
        
        # Performance assessment
        if results['position_rmse'] < 1.0:
            logger.info("\n🏆 EXCELLENT ACCURACY ACHIEVED!")
            logger.info("   Model is ready for real-time educational games")
        elif results['position_rmse'] < 5.0:
            logger.info("\n✅ GOOD ACCURACY ACHIEVED!")
            logger.info("   Model suitable for educational demonstrations")
        else:
            logger.info("\n⚠️  Model accuracy could be improved")
            logger.info("   Consider increasing training time or data")
        
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    main()
