"""
Comprehensive Gravity Yonder Over Setup and Runner
This script sets up the complete project with NVIDIA Modulus, cuDF, and trained ML models
"""

import subprocess
import sys
import os
import logging
import time
from pathlib import Path
import json
import platform

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GravityYonderSetup:
    """Complete setup and runner for Gravity Yonder Over educational platform"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.backend_dir = self.project_root / "backend"
        self.frontend_dir = self.project_root / "frontend"
        self.ml_dir = self.project_root / "ml_models"
        
        logger.info("🚀 Gravity Yonder Over - Educational Physics Platform")
        logger.info("=" * 60)
        
    def check_system_requirements(self):
        """Check system requirements for NVIDIA tools"""
        logger.info("🔍 Checking system requirements...")
        
        # Check Python version
        python_version = sys.version_info
        if python_version.major < 3 or python_version.minor < 8:
            logger.error("❌ Python 3.8+ required")
            return False
        
        logger.info(f"✅ Python {python_version.major}.{python_version.minor}")
        
        # Check for CUDA
        try:
            import torch
            cuda_available = torch.cuda.is_available()
            if cuda_available:
                logger.info(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
            else:
                logger.info("⚠️  CUDA not available - will use CPU fallback")
        except ImportError:
            logger.info("⚠️  PyTorch not installed yet")
        
        return True
    
    def install_dependencies(self):
        """Install all required dependencies"""
        logger.info("📦 Installing dependencies...")
        
        # Backend dependencies
        logger.info("Installing backend dependencies...")
        try:
            subprocess.run([
                sys.executable, "-m", "pip", "install", "-r", 
                str(self.project_root / "requirements.txt")
            ], check=True, cwd=self.project_root)
            logger.info("✅ Backend dependencies installed")
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Failed to install backend dependencies: {e}")
            logger.info("📝 Note: Some NVIDIA packages may not be available on all systems")
            logger.info("📝 The application will use fallback implementations")
        
        # Frontend dependencies
        if self.frontend_dir.exists() and (self.frontend_dir / "package.json").exists():
            logger.info("Installing frontend dependencies...")
            try:
                subprocess.run(["npm", "install"], check=True, cwd=self.frontend_dir)
                logger.info("✅ Frontend dependencies installed")
            except subprocess.CalledProcessError:
                logger.error("❌ Failed to install frontend dependencies")
                logger.info("📝 Make sure Node.js and npm are installed")
        
    def setup_directories(self):
        """Create necessary directories"""
        logger.info("📁 Setting up directories...")
        
        directories = [
            self.ml_dir / "trained_models",
            self.backend_dir / "data" / "precomputed" / "fields",
            self.backend_dir / "data" / "precomputed" / "orbits",
            self.backend_dir / "data" / "precomputed" / "trajectories",
            Path("logs"),
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            logger.info(f"✅ Created: {directory}")
    
    def train_ml_models(self):
        """Train ML models with enhanced accuracy"""
        logger.info("🤖 Training ML models for enhanced accuracy...")
        
        training_script = self.ml_dir / "training" / "train_enhanced_models.py"
        
        if training_script.exists():
            try:
                logger.info("🔬 Starting enhanced ML training (this may take a while)...")
                
                # Run the enhanced training script
                result = subprocess.run([
                    sys.executable, str(training_script)
                ], cwd=self.project_root, capture_output=True, text=True)
                
                if result.returncode == 0:
                    logger.info("✅ ML models trained successfully!")
                    logger.info("📊 Training output:")
                    for line in result.stdout.split('\n')[-10:]:  # Show last 10 lines
                        if line.strip():
                            logger.info(f"    {line}")
                else:
                    logger.error("❌ ML training failed")
                    logger.error(f"Error: {result.stderr}")
                    logger.info("📝 Application will use basic physics simulation")
                
            except Exception as e:
                logger.error(f"❌ Failed to run ML training: {e}")
                logger.info("📝 Application will use basic physics simulation")
        else:
            logger.warning("⚠️  Enhanced training script not found")
            # Run basic training
            try:
                basic_training = self.ml_dir / "training" / "train_trajectory.py"
                if basic_training.exists():
                    subprocess.run([sys.executable, str(basic_training)], cwd=self.project_root)
                    logger.info("✅ Basic ML models trained")
            except Exception as e:
                logger.warning(f"⚠️  Basic training also failed: {e}")
    
    def validate_setup(self):
        """Validate that everything is set up correctly"""
        logger.info("🔍 Validating setup...")
        
        # Check if models were trained
        models_dir = self.ml_dir / "trained_models"
        model_files = list(models_dir.glob("*.pth"))
        
        if model_files:
            logger.info(f"✅ Found {len(model_files)} trained models")
            for model in model_files:
                logger.info(f"    📄 {model.name}")
        else:
            logger.warning("⚠️  No trained models found")
        
        # Check backend files
        key_backend_files = [
            self.backend_dir / "simulations" / "modulus_physics_engine.py",
            self.backend_dir / "api" / "routes.py",
            self.backend_dir / "app.py"
        ]
        
        for file in key_backend_files:
            if file.exists():
                logger.info(f"✅ {file.name}")
            else:
                logger.error(f"❌ Missing: {file}")
        
        # Check frontend files
        if (self.frontend_dir / "src").exists():
            logger.info("✅ Frontend source files found")
        else:
            logger.warning("⚠️  Frontend source not found")
        
        logger.info("🎯 Setup validation complete!")
    
    def start_backend(self):
        """Start the FastAPI backend server"""
        logger.info("🚀 Starting backend server...")
        
        backend_app = self.backend_dir / "app.py"
        if not backend_app.exists():
            backend_app = self.project_root / "app.py"
        
        if backend_app.exists():
            try:
                # Start FastAPI server
                import uvicorn
                logger.info("🌐 Starting FastAPI server on http://localhost:8000")
                
                # Import the app
                sys.path.insert(0, str(self.project_root))
                
                if (self.backend_dir / "app.py").exists():
                    from backend.app import app
                else:
                    from app import app
                
                # Run server in a separate process
                uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
                
            except ImportError:
                logger.error("❌ FastAPI/Uvicorn not installed")
                logger.info("📝 Run: pip install fastapi uvicorn")
            except Exception as e:
                logger.error(f"❌ Failed to start backend: {e}")
        else:
            logger.error("❌ Backend app not found")
    
    def start_frontend(self):
        """Start the React frontend development server"""
        logger.info("🎨 Starting frontend development server...")
        
        if self.frontend_dir.exists() and (self.frontend_dir / "package.json").exists():
            try:
                logger.info("🌐 Starting React dev server on http://localhost:3000")
                subprocess.run(["npm", "start"], cwd=self.frontend_dir)
            except KeyboardInterrupt:
                logger.info("Frontend server stopped")
            except Exception as e:
                logger.error(f"❌ Failed to start frontend: {e}")
        else:
            logger.warning("⚠️  Frontend not found - backend only mode")
    
    def run_demo(self):
        """Run a quick demo of the physics engine"""
        logger.info("🎮 Running physics engine demo...")
        
        try:
            # Import and test the physics engine
            sys.path.insert(0, str(self.project_root))
            from backend.simulations.modulus_physics_engine import ModulusPhysicsEngine
            
            engine = ModulusPhysicsEngine()
            
            # Test apple drop simulation
            logger.info("🍎 Testing apple drop simulation...")
            result = engine.simulate_apple_drop_game(height=20, gravity=9.81, time_steps=50)
            logger.info(f"✅ Apple drop: {result['times'][-1]:.2f}s fall time")
            
            # Test orbital slingshot
            logger.info("🚀 Testing orbital slingshot...")
            slingshot_result = engine.simulate_orbital_slingshot_game(
                planet_mass=5.972e24,  # Earth mass
                planet_radius=6.371e6,  # Earth radius
                approach_velocity=11000,  # m/s
                approach_angle=45  # degrees
            )
            logger.info(f"✅ Slingshot: {'Success' if slingshot_result['slingshot_successful'] else 'Failed'}")
            
            logger.info("🎉 Physics engine demo completed successfully!")
            
        except Exception as e:
            logger.error(f"❌ Demo failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    def display_status(self):
        """Display current project status"""
        logger.info("\n" + "=" * 60)
        logger.info("🎓 GRAVITY YONDER OVER - PROJECT STATUS")
        logger.info("=" * 60)
        
        # Check components
        components = {
            "NVIDIA Modulus Physics Engine": self.backend_dir / "simulations" / "modulus_physics_engine.py",
            "Enhanced ML Models": self.ml_dir / "trained_models",
            "Interactive Games API": self.backend_dir / "api" / "routes.py",
            "React Frontend": self.frontend_dir / "src",
            "Educational Content": Path("curriculum") / "lessons"
        }
        
        for component, path in components.items():
            if path.exists():
                if path.is_dir() and any(path.iterdir()):
                    logger.info(f"✅ {component}")
                elif path.is_file():
                    logger.info(f"✅ {component}")
                else:
                    logger.info(f"⚠️  {component} (empty)")
            else:
                logger.info(f"❌ {component}")
        
        # Display features
        logger.info("\n🎮 AVAILABLE EDUCATIONAL GAMES:")
        games = [
            "🍎 Apple Drop - Real-time gravity simulation",
            "🚀 Orbital Slingshot - Gravity assist maneuvers",
            "🌌 Lagrange Points - Stability exploration",
            "🔥 Escape Velocity - Planetary escape challenges", 
            "🕳️ Black Hole Navigation - Relativistic physics"
        ]
        
        for game in games:
            logger.info(f"  {game}")
        
        logger.info("\n🔬 TECHNOLOGY STACK:")
        tech_stack = [
            "🧮 NVIDIA Modulus - GPU-optional physics simulations",
            "⚡ cuDF - GPU-accelerated data processing",
            "🤖 PyTorch - Enhanced ML model training",
            "⚛️ React + Three.js - Interactive 3D visualizations",
            "🚀 FastAPI - High-performance API backend"
        ]
        
        for tech in tech_stack:
            logger.info(f"  {tech}")
        
        logger.info("\n" + "=" * 60)

def main():
    """Main setup and run function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Gravity Yonder Over - Setup and Runner")
    parser.add_argument("--setup", action="store_true", help="Run full setup")
    parser.add_argument("--train", action="store_true", help="Train ML models only")
    parser.add_argument("--demo", action="store_true", help="Run physics demo")
    parser.add_argument("--backend", action="store_true", help="Start backend server")
    parser.add_argument("--frontend", action="store_true", help="Start frontend server")
    parser.add_argument("--status", action="store_true", help="Show project status")
    
    args = parser.parse_args()
    
    setup = GravityYonderSetup()
    
    if args.status:
        setup.display_status()
        return
    
    if args.demo:
        setup.run_demo()
        return
    
    if args.train:
        setup.train_ml_models()
        return
    
    if args.backend:
        setup.start_backend()
        return
    
    if args.frontend:
        setup.start_frontend()
        return
    
    if args.setup or len(sys.argv) == 1:
        # Full setup process
        setup.display_status()
        
        if not setup.check_system_requirements():
            return
        
        setup.setup_directories()
        setup.install_dependencies()
        setup.train_ml_models()
        setup.validate_setup()
        
        logger.info("\n" + "🎉" * 20)
        logger.info("🎓 GRAVITY YONDER OVER SETUP COMPLETE!")
        logger.info("🎉" * 20)
        
        logger.info("\n🚀 NEXT STEPS:")
        logger.info("1. Start backend: python run_gravity_yonder.py --backend")
        logger.info("2. Start frontend: python run_gravity_yonder.py --frontend")
        logger.info("3. Or run demo: python run_gravity_yonder.py --demo")
        
        setup.display_status()

if __name__ == "__main__":
    main()
