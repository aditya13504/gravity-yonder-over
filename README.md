# 🌌 Gravity Yonder Over

> *From falling apples to orbital slingshots — learn gravity the cosmic way.*

An interactive educational platform that transforms complex gravitational physics into engaging, gamified learning experiences. Built with cutting-edge technologies including NVIDIA Modulus, Physics-Informed Neural Networks (PINNs), and real-time GPU-accelerated simulations.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://gravity-yonder-over.streamlit.app)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🚀 Features

### 🎮 Interactive Learning Modules
- **Beginner Adventures**: Intuitive gravity games for all ages
- **Advanced Simulations**: N-body orbital mechanics with real-time physics
- **Educational Content**: Progressive curriculum from Newton's laws to relativistic effects
- **Gamified Progress**: Points, badges, and achievement tracking

### 🤖 AI-Powered Physics Engine
- **Physics-Informed Neural Networks (PINNs)**: Learn gravitational fields from first principles
- **NVIDIA Modulus Integration**: GPU-accelerated physics simulations
- **Machine Learning Models**: Trajectory prediction and orbital analysis
- **Real-time Optimization**: Adaptive performance based on hardware capabilities

### 🎯 Visualization & Analysis
- **3D Interactive Plots**: Plotly-powered orbital visualizations
- **Real-time Animations**: Watch gravity in action
- **Heatmap Analysis**: Visualize gravitational field strength
- **Educational Diagrams**: Concept illustrations and explanations

### 🔧 Technical Capabilities
- **Multi-Platform**: Web app (Streamlit), React frontend, Flask API backend
- **GPU Acceleration**: CUDA support with cuDF and cuPy
- **Docker Deployment**: Containerized for easy scaling
- **Database Integration**: Progress tracking and user management

## 📋 Table of Contents

- [Quick Start](#-quick-start)
- [Installation](#-installation)
- [Usage](#-usage)
- [Architecture](#-architecture)
- [Development](#-development)
- [Docker Deployment](#-docker-deployment)
- [Machine Learning Models](#-machine-learning-models)
- [API Documentation](#-api-documentation)
- [Contributing](#-contributing)
- [License](#-license)

## 🚀 Quick Start

### Option 1: Streamlit Cloud (Easiest)
Visit the live demo: [gravity-yonder-over.streamlit.app](https://gravity-yonder-over.streamlit.app)

### Option 2: Local Installation
```bash
# Clone the repository
git clone https://github.com/aditya13504/gravity-yonder-over.git
cd gravity-yonder-over

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run app.py
```

### Option 3: Full Development Setup
```bash
# Clone and setup backend
git clone https://github.com/aditya13504/gravity-yonder-over.git
cd gravity-yonder-over

# Backend setup
pip install -r requirements.txt
pip install -r requirements_gpu.txt  # For GPU acceleration

# Frontend setup (optional)
cd frontend
npm install
npm run dev

# Run the full stack
python run_gravity_yonder_full.py
```

## 🛠 Installation

### Prerequisites
- Python 3.8+
- Node.js 16+ (for React frontend)
- CUDA 11.0+ (optional, for GPU acceleration)
- Docker (optional, for containerized deployment)

### Environment Setup

#### CPU-Only Installation (Recommended for beginners)
```bash
# Create virtual environment
python -m venv gravity_env
source gravity_env/bin/activate  # On Windows: gravity_env\Scripts\activate

# Install CPU dependencies
pip install -r requirements.txt
```

#### GPU-Accelerated Installation (Advanced users)
```bash
# Install GPU dependencies
pip install -r requirements_gpu.txt

# Verify CUDA installation
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

#### Development Environment
```bash
# Install development tools
pip install black isort pytest jupyter

# Setup pre-commit hooks
pip install pre-commit
pre-commit install
```

## 🎯 Usage

### 1. Educational Web App (Streamlit)
```bash
streamlit run app.py
```
Access at `http://localhost:8501`

**Features:**
- Interactive gravity simulations
- Step-by-step physics tutorials
- Real-time parameter adjustment
- Progress tracking and achievements

### 2. Advanced Physics Engine
```python
from backend.simulations.gravity_solver import GravitySolver
from backend.models.celestial_body import CelestialBody

# Create celestial bodies
earth = CelestialBody("Earth", mass=5.972e24, position=[0, 0], velocity=[0, 0])
moon = CelestialBody("Moon", mass=7.342e22, position=[384400000, 0], velocity=[0, 1022])

# Run simulation
solver = GravitySolver()
trajectory = solver.simulate([earth, moon], time_span=86400*30)  # 30 days
```

### 3. Machine Learning Models
```python
from ml.models.pinn_gravity import GravityPINN

# Train physics-informed neural network
pinn = GravityPINN()
pinn.train(training_data, physics_constraints)

# Predict gravitational field
field = pinn.predict_field(coordinates)
```

### 4. React Frontend (Optional)
```bash
cd frontend
npm start
```
Access at `http://localhost:3000`

## 🏗 Architecture

```
gravity-yonder-over/
├── 🌐 Web Applications
│   ├── app.py                 # Main Streamlit app
│   ├── streamlit_app.py       # Enhanced UI version
│   └── lightweight_app.py     # Minimal version
│
├── 🔧 Backend Services
│   ├── backend/
│   │   ├── api/              # REST API endpoints
│   │   ├── models/           # Data models
│   │   ├── simulations/      # Physics engines
│   │   └── visualizations/   # Plotting utilities
│   │
├── 🎨 Frontend (React)
│   ├── frontend/
│   │   ├── src/components/   # React components
│   │   ├── src/physics/      # Three.js physics
│   │   └── src/pages/        # Application pages
│   │
├── 🤖 Machine Learning
│   ├── ml/
│   │   └── models/           # PINN and trajectory models
│   ├── ml_models/
│   │   ├── notebooks/        # Training notebooks
│   │   ├── pretrained/       # Pre-trained models
│   │   └── training/         # Training scripts
│   │
├── 🐳 Deployment
│   ├── docker/               # Docker configuration
│   ├── scripts/              # Deployment scripts
│   └── requirements*.txt     # Dependencies
```

### Key Components

#### 1. Physics Engine (`backend/simulations/`)
- **GravitySolver**: Core gravitational simulation engine
- **ModulusPhysicsEngine**: NVIDIA Modulus integration
- **CudaAccelerator**: GPU-accelerated computations
- **PrecomputedSimulations**: Cached simulation results

#### 2. Machine Learning (`ml/models/`)
- **GravityPINN**: Physics-informed neural networks
- **TrajectoryPredictor**: Orbital path prediction
- **EnhancedModels**: Advanced ML integrations

#### 3. Visualization (`backend/visualizations/`)
- **PlotlyGraphs**: Interactive 3D plots
- **Animations**: Real-time physics animations
- **Heatmaps**: Field strength visualizations

#### 4. Educational Content (`src/`)
- **EducationalContent**: Structured learning modules
- **BeginnerFeatures**: Gamified introductory content
- **AdvancedSimulations**: Professional-grade tools

## 💻 Development

### Project Structure Philosophy
```
🎯 Modular Design: Each component is self-contained
🔄 Hot Reloading: Real-time development updates
🧪 Test-Driven: Comprehensive testing framework
📚 Documentation: Inline docs and examples
🚀 Performance: GPU acceleration where available
```

### Running Development Servers

#### Streamlit Development
```bash
streamlit run app.py --server.runOnSave true
```

#### React Development
```bash
cd frontend
npm run dev
```

#### Flask API Development
```bash
cd backend
python app.py
```

### Code Quality Tools
```bash
# Format code
black .
isort .

# Run tests
pytest

# Check GPU availability
python -c "from backend.simulations.cuda_accelerator import check_gpu; check_gpu()"
```

### Adding New Features

#### 1. New Physics Simulation
```python
# backend/simulations/your_simulation.py
from .gravity_solver import GravitySolver

class YourSimulation(GravitySolver):
    def simulate(self, bodies, **kwargs):
        # Your custom physics logic
        pass
```

#### 2. New Educational Module
```python
# Add to beginner_features.py or create new module
def your_educational_feature():
    st.markdown("# Your Feature")
    # Interactive components
```

#### 3. New ML Model
```python
# ml/models/your_model.py
from .pinn_gravity import BasePINN

class YourModel(BasePINN):
    def __init__(self):
        # Your model architecture
        pass
```

## 🐳 Docker Deployment

### Quick Deployment
```bash
# Build and run with Docker Compose
docker-compose -f docker/docker-compose.yml up --build
```

### Services
- **Frontend**: React app on port 3000
- **Backend**: Flask API on port 8000
- **Database**: PostgreSQL on port 5432

### Environment Variables
```bash
# Create .env file
FLASK_ENV=production
DATABASE_URL=postgresql://user:pass@localhost:5432/gravity_db
CUDA_ENABLED=true
ML_DEVICE=cuda
```

### Production Deployment
```bash
# Build optimized images
docker build -f docker/Dockerfile.backend -t gravity-backend .
docker build -f docker/Dockerfile.frontend -t gravity-frontend .

# Deploy to cloud
# (Add your specific cloud deployment commands)
```

## 🤖 Machine Learning Models

### Physics-Informed Neural Networks (PINNs)

#### Training a Gravity PINN
```python
from ml.models.pinn_gravity import GravityPINN, PINNConfig

# Configure model
config = PINNConfig(
    hidden_dims=[50, 50, 50],
    learning_rate=1e-3,
    physics_weight=1.0
)

# Initialize and train
pinn = GravityPINN(config)
pinn.train(training_data)

# Save model
pinn.save_model('ml_models/trained_models/my_pinn.pth')
```

#### Using Pre-trained Models
```python
# Load pre-trained PINN
pinn = GravityPINN.load_model('ml_models/pretrained/modulus_gravity_pinn.pth')

# Predict gravitational field
coordinates = np.array([[x, y, z] for x, y, z in coordinate_points])
potential = pinn.predict(coordinates)
```

### Trajectory Prediction
```python
from ml.models.trajectory_predictor import TrajectoryPredictor

# Load trajectory model
predictor = TrajectoryPredictor.load_model('ml_models/pretrained/trajectory_predictor_base.pth')

# Predict orbital path
initial_conditions = [position, velocity, mass]
future_trajectory = predictor.predict_trajectory(initial_conditions, time_steps=1000)
```

### Training Your Own Models
```bash
# Train PINN from scratch
python ml_models/training/train_pinn.py --config config.yaml

# Train trajectory predictor
python ml_models/training/train_trajectory.py --data_path data/trajectories/

# Enhanced model training
python ml_models/training/train_enhanced_models.py
```

## 📖 API Documentation

### REST API Endpoints

#### Simulation Endpoints
```http
POST /api/v1/simulate
Content-Type: application/json

{
  "bodies": [
    {"name": "Earth", "mass": 5.972e24, "position": [0, 0], "velocity": [0, 0]},
    {"name": "Moon", "mass": 7.342e22, "position": [384400000, 0], "velocity": [0, 1022]}
  ],
  "time_span": 86400,
  "time_steps": 1000
}
```

#### ML Model Endpoints
```http
POST /api/v1/predict/gravity
Content-Type: application/json

{
  "coordinates": [[0, 0, 0], [1000, 1000, 1000]],
  "model": "pinn_gravity"
}
```

#### Educational Content
```http
GET /api/v1/education/modules
GET /api/v1/education/module/{module_id}
POST /api/v1/progress/update
```

### Python API

#### Core Classes
```python
from backend.models import CelestialBody
from backend.simulations import GravitySolver
from backend.visualizations import GravityVisualizer

# Create simulation
solver = GravitySolver()
visualizer = GravityVisualizer()

# Run and visualize
results = solver.simulate(bodies)
fig = visualizer.plot_trajectories(results)
```

## 🎓 Educational Use Cases

### 1. High School Physics
- Newton's law of universal gravitation
- Orbital mechanics basics
- Energy conservation
- Kepler's laws

### 2. University Courses
- Classical mechanics
- Computational physics
- Numerical methods
- Machine learning in physics

### 3. Research Applications
- N-body simulations
- Gravitational wave studies
- Exoplanet detection
- Space mission planning

### 4. Professional Development
- GPU computing with CUDA
- Physics-informed ML
- Scientific visualization
- High-performance computing

## 🔬 Scientific Accuracy

### Physics Implementation
- **Newtonian Gravity**: Full N-body gravitational interactions
- **Numerical Integration**: RK4 and adaptive step-size methods
- **Relativistic Effects**: Optional GR corrections for high-precision simulations
- **Tidal Forces**: Detailed body deformation modeling

### Validation
- **Unit Tests**: Comprehensive physics validation
- **Benchmark Problems**: Two-body, three-body, and solar system tests
- **Literature Comparison**: Results validated against published research
- **Expert Review**: Reviewed by physics educators and researchers

## 🤝 Contributing

We welcome contributions from educators, developers, and physics enthusiasts!

### Quick Contribution Guide
1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Make your changes
4. Add tests and documentation
5. Submit a pull request

### Development Setup
```bash
# Fork and clone your fork
git clone https://github.com/YOUR_USERNAME/gravity-yonder-over.git
cd gravity-yonder-over

# Set up development environment
python -m venv dev_env
source dev_env/bin/activate
pip install -r requirements.txt -r requirements_gpu.txt

# Install development tools
pip install -e .
pre-commit install

# Run tests
pytest
```

### Contribution Areas
- 🎮 **Educational Content**: New tutorials and interactive modules
- 🧮 **Physics Engines**: Advanced simulation algorithms
- 🤖 **Machine Learning**: New ML models and training techniques
- 🎨 **Visualization**: Enhanced graphics and animations
- 📚 **Documentation**: Guides, examples, and API docs
- 🧪 **Testing**: Unit tests and validation scripts

### Code Style
- Python: Black formatter, isort, type hints
- JavaScript: ESLint, Prettier
- Documentation: Google-style docstrings

## 🌟 Roadmap

### Short-term (Next 3 months)
- [ ] Enhanced VR/AR visualization support
- [ ] Mobile app development
- [ ] Advanced PINN architectures
- [ ] Real-time multiplayer simulations
- [ ] Integration with physics curriculum standards

### Medium-term (6 months)
- [ ] Cloud-based model training
- [ ] Advanced relativity simulations
- [ ] Quantum gravity exploration modules
- [ ] Teacher dashboard and classroom management
- [ ] Multi-language support

### Long-term (1 year+)
- [ ] AI-powered personalized learning paths
- [ ] Professional space mission planning tools
- [ ] Integration with space agencies' data
- [ ] Advanced gravitational wave simulations
- [ ] Citizen science project integration

## 🏆 Acknowledgments

### Technologies Used
- **[NVIDIA Modulus](https://developer.nvidia.com/modulus)**: Physics-informed machine learning
- **[Streamlit](https://streamlit.io/)**: Interactive web applications
- **[Plotly](https://plotly.com/)**: Scientific visualization
- **[PyTorch](https://pytorch.org/)**: Deep learning framework
- **[cuDF](https://rapids.ai/cudf.html)**: GPU-accelerated dataframes
- **[React](https://reactjs.org/)**: Frontend framework
- **[Three.js](https://threejs.org/)**: 3D graphics

### Contributors
- **Aditya Sharma** - Project Lead & Physics Implementation
- **[Add other contributors]**

### Inspiration
This project was inspired by the need to make gravitational physics accessible and engaging for learners of all levels, from curious middle schoolers to graduate students in physics.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📞 Support & Contact

### Getting Help
- 📖 **Documentation**: Check this README and inline documentation
- 🐛 **Issues**: [GitHub Issues](https://github.com/aditya13504/gravity-yonder-over/issues)
- 💬 **Discussions**: [GitHub Discussions](https://github.com/aditya13504/gravity-yonder-over/discussions)

### Educational Partnerships
We're actively seeking partnerships with:
- Educational institutions
- Physics departments
- Science museums
- Online learning platforms

Contact us at: [your-email@example.com]

### Commercial Support
Professional support and custom development available for:
- Educational institutions
- Research organizations
- Space agencies
- Technology companies

---

<div align="center">

**🌌 Gravity Yonder Over - Making Physics Accessible Through Interactive Learning 🌌**

*Built with ❤️ for space enthusiasts, physics learners, and curious minds everywhere*

[⭐ Star us on GitHub](https://github.com/aditya13504/gravity-yonder-over) | [🚀 Try the Live Demo](https://gravity-yonder-over.streamlit.app) | [📚 Read the Docs](https://github.com/aditya13504/gravity-yonder-over/wiki)

</div>
