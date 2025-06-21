"""
Simple Enhanced ML Training Script for Gravity Yonder Over
Trains trajectory prediction models with CPU-only processing
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import logging
import sys
import os

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GravityTrajectoryPredictor(nn.Module):
    """Simple neural network for trajectory prediction (CPU-optimized)"""
    
    def __init__(self, input_size=6, hidden_sizes=[64, 32], output_size=6):
        super().__init__()
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, output_size))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

def generate_physics_data(n_samples=5000):
    """Generate synthetic physics data for training"""
    logger.info(f"Generating {n_samples} physics training samples...")
    
    # Input: [x, y, z, vx, vy, vz]
    # Output: [new_x, new_y, new_z, new_vx, new_vy, new_vz] after one time step
    
    X = np.random.uniform(-5, 5, (n_samples, 6))
    y = np.zeros((n_samples, 6))
    
    G = 6.674e-11
    central_mass = 5.972e24  # Earth mass
    dt = 0.01  # Time step
    
    for i in range(n_samples):
        pos = X[i, :3]
        vel = X[i, 3:]
        
        # Simple gravitational dynamics
        r = np.linalg.norm(pos) + 1e-10
        acc = -G * central_mass * pos / (r**3)
        
        # Euler integration
        new_vel = vel + acc * dt
        new_pos = pos + new_vel * dt
        
        y[i] = np.concatenate([new_pos, new_vel])
    
    return X, y

def train_scikit_models():
    """Train scikit-learn models for physics prediction"""
    logger.info("Training scikit-learn based physics models...")
    
    # Generate training data
    X, y = generate_physics_data(10000)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)
    y_train_scaled = scaler_y.fit_transform(y_train)
    
    models = {}
    
    # Random Forest model
    logger.info("Training Random Forest model...")
    rf_model = RandomForestRegressor(
        n_estimators=100,
        max_depth=20,
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train_scaled, y_train_scaled)
    
    # Predictions and evaluation
    rf_pred = rf_model.predict(X_test_scaled)
    rf_pred_unscaled = scaler_y.inverse_transform(rf_pred)
    
    rf_mse = mean_squared_error(y_test, rf_pred_unscaled)
    rf_r2 = r2_score(y_test, rf_pred_unscaled)
    
    logger.info(f"Random Forest - MSE: {rf_mse:.6f}, R²: {rf_r2:.4f}")
    
    models['random_forest'] = {
        'model': rf_model,
        'scaler_X': scaler_X,
        'scaler_y': scaler_y,
        'mse': rf_mse,
        'r2': rf_r2
    }
    
    # Neural Network model
    logger.info("Training MLP Neural Network...")
    mlp_model = MLPRegressor(
        hidden_layer_sizes=(128, 64, 32),
        activation='relu',
        solver='adam',
        max_iter=500,
        random_state=42
    )
    mlp_model.fit(X_train_scaled, y_train_scaled)
    
    # Predictions and evaluation
    mlp_pred = mlp_model.predict(X_test_scaled)
    mlp_pred_unscaled = scaler_y.inverse_transform(mlp_pred)
    
    mlp_mse = mean_squared_error(y_test, mlp_pred_unscaled)
    mlp_r2 = r2_score(y_test, mlp_pred_unscaled)
    
    logger.info(f"MLP Neural Network - MSE: {mlp_mse:.6f}, R²: {mlp_r2:.4f}")
    
    models['mlp'] = {
        'model': mlp_model,
        'scaler_X': scaler_X,
        'scaler_y': scaler_y,
        'mse': mlp_mse,
        'r2': mlp_r2
    }
    
    return models

def train_pytorch_model():
    """Train PyTorch model (CPU-only)"""
    logger.info("Training PyTorch model on CPU...")
    
    device = torch.device("cpu")
    logger.info(f"Using device: {device}")
    
    # Generate training data
    X_train, y_train = generate_physics_data(8000)
    X_val, y_val = generate_physics_data(2000)
    
    # Convert to tensors
    X_train_tensor = torch.FloatTensor(X_train).to(device)
    y_train_tensor = torch.FloatTensor(y_train).to(device)
    X_val_tensor = torch.FloatTensor(X_val).to(device)
    y_val_tensor = torch.FloatTensor(y_val).to(device)
    
    # Create model
    model = GravityTrajectoryPredictor().to(device)
    
    # Training parameters
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    epochs = 100
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        optimizer.zero_grad()
        
        train_pred = model(X_train_tensor)
        train_loss = criterion(train_pred, y_train_tensor)
        train_loss.backward()
        optimizer.step()
        
        # Validation phase
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val_tensor)
            val_loss = criterion(val_pred, y_val_tensor)
        
        train_losses.append(train_loss.item())
        val_losses.append(val_loss.item())
        
        if (epoch + 1) % 20 == 0:
            logger.info(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss.item():.6f}, Val Loss: {val_loss.item():.6f}")
    
    return model, train_losses, val_losses

def save_models():
    """Save all trained models"""
    logger.info("Training and saving all models...")
    
    # Create models directory
    models_dir = Path("ml_models/trained_models")
    models_dir.mkdir(parents=True, exist_ok=True)
    
    # Train and save scikit-learn models
    sklearn_models = train_scikit_models()
    
    for name, model_info in sklearn_models.items():
        model_path = models_dir / f"{name}_model.pkl"
        scaler_x_path = models_dir / f"{name}_scaler_x.pkl"
        scaler_y_path = models_dir / f"{name}_scaler_y.pkl"
        
        joblib.dump(model_info['model'], model_path)
        joblib.dump(model_info['scaler_X'], scaler_x_path)
        joblib.dump(model_info['scaler_y'], scaler_y_path)
        
        logger.info(f"Saved {name} model (MSE: {model_info['mse']:.6f}, R²: {model_info['r2']:.4f})")
    
    # Train and save PyTorch model
    pytorch_model, train_losses, val_losses = train_pytorch_model()
    torch_path = models_dir / "pytorch_trajectory_model.pth"
    torch.save(pytorch_model.state_dict(), torch_path)
    logger.info(f"Saved PyTorch model with final train loss: {train_losses[-1]:.6f}")
    
    # Save training history
    history = {
        'pytorch_train_losses': train_losses,
        'pytorch_val_losses': val_losses,
        'sklearn_models': {name: {'mse': info['mse'], 'r2': info['r2']} 
                          for name, info in sklearn_models.items()}
    }
    
    import json
    with open(models_dir / "training_history.json", 'w') as f:
        json.dump(history, f, indent=2)
    
    logger.info("All models saved successfully!")
    
    return sklearn_models, pytorch_model, history

def create_model_comparison_plot():
    """Create comparison plots for different models"""
    logger.info("Creating model comparison plots...")
    
    models_dir = Path("ml_models/trained_models")
    
    # Load training history
    try:
        import json
        with open(models_dir / "training_history.json", 'r') as f:
            history = json.load(f)
    except FileNotFoundError:
        logger.warning("Training history not found. Run training first.")
        return
    
    # Create plots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # PyTorch training curves
    epochs = range(1, len(history['pytorch_train_losses']) + 1)
    ax1.plot(epochs, history['pytorch_train_losses'], 'b-', label='Training Loss')
    ax1.plot(epochs, history['pytorch_val_losses'], 'r-', label='Validation Loss') 
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('PyTorch Model Training Curves')
    ax1.legend()
    ax1.grid(True)
    
    # Sklearn model comparison (MSE)
    sklearn_names = list(history['sklearn_models'].keys())
    sklearn_mse = [history['sklearn_models'][name]['mse'] for name in sklearn_names]
    
    ax2.bar(sklearn_names, sklearn_mse)
    ax2.set_ylabel('Mean Squared Error')
    ax2.set_title('Scikit-learn Models - MSE Comparison')
    ax2.tick_params(axis='x', rotation=45)
    
    # Sklearn model comparison (R²)
    sklearn_r2 = [history['sklearn_models'][name]['r2'] for name in sklearn_names]
    
    ax3.bar(sklearn_names, sklearn_r2)
    ax3.set_ylabel('R² Score')
    ax3.set_title('Scikit-learn Models - R² Comparison')
    ax3.tick_params(axis='x', rotation=45)
    
    # Sample prediction visualization
    # Generate test data for visualization
    X_test, y_test = generate_physics_data(100)
    
    # Load a model for prediction (Random Forest)
    try:
        rf_model = joblib.load(models_dir / "random_forest_model.pkl")
        scaler_x = joblib.load(models_dir / "random_forest_scaler_x.pkl")
        scaler_y = joblib.load(models_dir / "random_forest_scaler_y.pkl")
        
        X_test_scaled = scaler_x.transform(X_test)
        y_pred_scaled = rf_model.predict(X_test_scaled)
        y_pred = scaler_y.inverse_transform(y_pred_scaled)
        
        # Plot actual vs predicted for first component
        ax4.scatter(y_test[:, 0], y_pred[:, 0], alpha=0.6)
        ax4.plot([y_test[:, 0].min(), y_test[:, 0].max()], 
                [y_test[:, 0].min(), y_test[:, 0].max()], 'r--', lw=2)
        ax4.set_xlabel('Actual Position X')
        ax4.set_ylabel('Predicted Position X')
        ax4.set_title('Random Forest: Actual vs Predicted')
        ax4.grid(True)
        
    except FileNotFoundError:
        ax4.text(0.5, 0.5, 'Model not found\nRun training first', 
                ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title('Sample Predictions')
    
    plt.tight_layout()
    
    # Save plot
    plot_path = models_dir / "model_comparison.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved comparison plot to {plot_path}")
    
    plt.show()

def generate_model_summary():
    """Generate a summary report of trained models"""
    models_dir = Path("ml_models/trained_models")
    
    summary = {
        "timestamp": str(pd.Timestamp.now()),
        "framework": "CPU-based scikit-learn + PyTorch",
        "models_trained": [],
        "performance_metrics": {},
        "model_files": []
    }
    
    # Check for model files
    model_files = list(models_dir.glob("*.pkl")) + list(models_dir.glob("*.pth"))
    summary["model_files"] = [str(f.name) for f in model_files]
    
    # Load performance metrics if available
    try:
        import json
        with open(models_dir / "training_history.json", 'r') as f:
            history = json.load(f)
        
        summary["performance_metrics"] = history["sklearn_models"]
        summary["models_trained"] = list(history["sklearn_models"].keys()) + ["pytorch_trajectory"]
        
    except FileNotFoundError:
        logger.warning("Training history not found")
    
    # Save summary
    with open(models_dir / "model_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info("Generated model summary")
    
    return summary

def main():
    """Main training pipeline"""
    logger.info("Starting CPU-based ML model training pipeline...")
    
    try:
        # Train and save all models
        sklearn_models, pytorch_model, history = save_models()
        
        # Create comparison plots
        create_model_comparison_plot()
        
        # Generate summary
        summary = generate_model_summary()
        
        logger.info("Training pipeline completed successfully!")
        
        # Print summary
        print("\n" + "="*50)
        print("TRAINING SUMMARY")
        print("="*50)
        print(f"Framework: CPU-based scikit-learn + PyTorch")
        print(f"Models trained: {len(summary['models_trained'])}")
        print(f"Model files created: {len(summary['model_files'])}")
        
        if 'sklearn_models' in history:
            print("\nScikit-learn Model Performance:")
            for name, metrics in history['sklearn_models'].items():
                print(f"  {name}: MSE={metrics['mse']:.6f}, R²={metrics['r2']:.4f}")
        
        print(f"\nPyTorch Model: Final training loss = {history['pytorch_train_losses'][-1]:.6f}")
        print("\nAll models saved to ml_models/trained_models/")
        
    except Exception as e:
        logger.error(f"Training pipeline failed: {e}")
        raise

if __name__ == "__main__":
    main()
