#!/usr/bin/env python3
"""
Gravity Yonder Over - Complete Application Runner
Runs both backend and frontend for the educational physics simulation platform
"""

import subprocess
import sys
import os
import time
import webbrowser
from pathlib import Path

def run_backend():
    """Run the FastAPI backend server"""
    print("🚀 Starting Gravity Yonder Backend (FastAPI + NVIDIA Modulus)...")
    
    backend_dir = Path(__file__).parent / "backend"
    os.chdir(backend_dir)
    
    # Start FastAPI with uvicorn
    backend_process = subprocess.Popen([
        sys.executable, "-m", "uvicorn", 
        "app:app", 
        "--host", "0.0.0.0", 
        "--port", "8000", 
        "--reload"
    ])
    
    print("✅ Backend started on http://localhost:8000")
    return backend_process

def run_frontend():
    """Run the React frontend development server"""
    print("🎨 Starting Gravity Yonder Frontend (React + Three.js)...")
    
    frontend_dir = Path(__file__).parent / "frontend"
    os.chdir(frontend_dir)
    
    # Install dependencies and start development server
    print("📦 Installing frontend dependencies...")
    subprocess.run(["npm", "install"], check=True)
    
    print("🌟 Starting React development server...")
    frontend_process = subprocess.Popen([
        "npm", "run", "dev"
    ])
    
    print("✅ Frontend started on http://localhost:3000")
    return frontend_process

def main():
    """Main application runner"""
    print("=" * 60)
    print("🌌 GRAVITY YONDER OVER - Educational Physics Platform")
    print("=" * 60)
    print("⚡ NVIDIA Modulus + cuDF Physics Simulations")
    print("🎮 Interactive Gravity Games & Visualizations")
    print("🎓 Real-time Educational Physics Engine")
    print("=" * 60)
    
    try:
        # Start backend
        backend_process = run_backend()
        time.sleep(3)  # Give backend time to start
        
        # Start frontend
        frontend_process = run_frontend()
        time.sleep(3)  # Give frontend time to start
        
        print("\n" + "=" * 60)
        print("🎉 GRAVITY YONDER OVER IS READY!")
        print("=" * 60)
        print("🌐 Frontend: http://localhost:3000")
        print("🔧 Backend API: http://localhost:8000")
        print("📚 API Docs: http://localhost:8000/docs")
        print("💫 Health Check: http://localhost:8000/health")
        print("=" * 60)
        print("\n🎮 Available Games:")
        print("  • 🍎 Apple Drop - Real gravity physics")
        print("  • 🚀 Orbital Slingshot - N-body mechanics")
        print("  • 🌍 Escape Velocity - Planetary physics")
        print("  • 🌐 Lagrange Explorer - Equilibrium points")
        print("  • ⚫ Black Hole Navigator - Relativistic effects")
        print("  • 🌌 Wormhole Navigator - Exotic spacetime")
        print("\n📊 Features:")
        print("  • Real-time NVIDIA Modulus simulations")
        print("  • GPU-accelerated with cuDF processing")
        print("  • Educational physics insights")
        print("  • Interactive 3D visualizations")
        print("  • Pre-trained ML model integration")
        print("\n🎯 Press Ctrl+C to stop all servers")
        
        # Open browser
        time.sleep(2)
        try:
            webbrowser.open("http://localhost:3000")
        except:
            pass
        
        # Wait for user to stop
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n\n🛑 Shutting down Gravity Yonder Over...")
            
            # Terminate processes
            try:
                frontend_process.terminate()
                frontend_process.wait(timeout=5)
            except:
                frontend_process.kill()
                
            try:
                backend_process.terminate()
                backend_process.wait(timeout=5)
            except:
                backend_process.kill()
                
            print("✅ All servers stopped. Thank you for using Gravity Yonder Over!")
            
    except Exception as e:
        print(f"❌ Error starting application: {e}")
        print("\n🔧 Troubleshooting:")
        print("  • Make sure you have Node.js and npm installed")
        print("  • Ensure Python dependencies are installed: pip install -r requirements.txt")
        print("  • Check if ports 3000 and 8000 are available")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
