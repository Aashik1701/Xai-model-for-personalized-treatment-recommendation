# Hybrid Explainable AI Models for Personalized Treatment Recommendation

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A comprehensive framework for developing hybrid explainable AI models for personalized healthcare treatment recommendations. This project combines ensemble learning, multi-modal fusion, and state-of-the-art explainability techniques to create interpretable healthcare AI systems.

## 🏥 Overview

This project implements hybrid explainable AI models that:

- **Combine multiple ML approaches**: Ensemble methods, multi-modal fusion, attention mechanisms
- **Provide comprehensive explanations**: SHAP, LIME, attention visualization
- **Support personalized medicine**: Patient similarity networks, federated learning
- **Ensure clinical validation**: Bias assessment, fairness evaluation
- **Enable production deployment**: FastAPI, Docker, MLOps integration

## 📁 Project Structure

```
hybrid_explainable_healthcare_ai/
├── 📄 README.md                    # This file
├── 📄 requirements.txt             # Python dependencies
├── 📄 setup.py                     # Package configuration
├── 📄 pyproject.toml               # Modern Python config
├── 📄 Dockerfile                   # Container configuration
├── 📄 docker-compose.yml           # Multi-container setup
├── 📄 .env.example                 # Environment template
├── 📄 .gitignore                   # Git ignore rules
├── 📁 .github/workflows/           # CI/CD pipelines
├── 📁 .vscode/                     # VS Code configuration
│   ├── 📄 settings.json           # Workspace settings
│   ├── 📄 extensions.json         # Recommended extensions
│   ├── 📄 launch.json             # Debug configurations
│   └── 📄 tasks.json              # Build tasks
├── 📁 src/hybrid_xai_healthcare/   # Main source code
│   ├── 📁 config/                 # Configuration management
│   ├── 📁 data/                   # Data processing modules
│   ├── 📁 models/                 # ML model implementations
│   │   ├── 📁 ensemble/           # Ensemble methods
│   │   ├── 📁 hybrid/             # Hybrid approaches
│   │   └── 📁 federated/          # Federated learning
│   ├── 📁 explainability/         # Explainability modules
│   ├── 📁 personalization/        # Personalization features
│   ├── 📁 evaluation/             # Model evaluation
│   └── 📁 utils/                  # Utility functions
├── 📁 data/                       # Data storage
│   ├── 📁 raw/                    # Raw datasets
│   ├── 📁 processed/              # Processed data
│   └── 📁 external/               # External datasets
├── 📁 models/                     # Trained models
├── 📁 notebooks/                  # Jupyter notebooks
├── 📁 experiments/                # Experiment tracking
├── 📁 tests/                      # Unit and integration tests
├── 📁 api/                        # FastAPI application
├── 📁 scripts/                    # Training and utility scripts
├── 📁 config/                     # Configuration files
└── 📁 docs/                       # Documentation
```

## 🚀 Quick Start with VS Code

### Prerequisites

- **Python 3.8+** installed on your system
- **VS Code** with Python extension
- **Git** for version control
- **Docker** (optional, for containerized deployment)

### 1. Clone and Open in VS Code

```bash
# Clone the repository
git clone https://github.com/yourusername/hybrid-xai-healthcare.git
cd hybrid-xai-healthcare

# Open in VS Code
code .
```

### 2. Install Recommended Extensions

When you open the project, VS Code will recommend installing the following extensions:

- **Python** - Python language support
- **Pylance** - Fast, feature-rich language support
- **Jupyter** - Jupyter notebook support
- **Black Formatter** - Code formatting
- **Flake8** - Code linting
- **MyPy Type Checker** - Static type checking
- **Docker** - Container support
- **GitHub Copilot** - AI-powered coding assistant

### 3. Set Up Development Environment

#### Option A: Using VS Code Tasks (Recommended)

1. Open Command Palette (`Ctrl+Shift+P` / `Cmd+Shift+P`)
2. Run "Tasks: Run Task"
3. Select "Setup Virtual Environment"
4. Then run "Install Development Dependencies"

#### Option B: Manual Setup

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
pip install -e .[dev]
```

### 4. Configure Environment Variables

```bash
# Copy environment template
cp .env.example .env

# Edit .env file with your configuration
# Set database URLs, API keys, etc.
```

### 5. Initialize Data Version Control

```bash
# Initialize DVC for data versioning
dvc init

# Add data to DVC tracking
dvc add data/raw/
git add data/raw/.dvc .gitignore
git commit -m "Add raw data to DVC"
```

### 6. Start Development Services

#### Start MLflow Tracking Server
```bash
# Using VS Code task
Ctrl+Shift+P → "Tasks: Run Task" → "Start MLflow Server"

# Or manually
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --host 0.0.0.0
```

#### Start API Server (for development)
```bash
# Using VS Code debug configuration
F5 → Select "Python: Run API Server"

# Or manually
uvicorn hybrid_xai_healthcare.api.app:app --reload --host 0.0.0.0 --port 8000
```

## 🧪 Running Experiments

### 1. Data Preprocessing

```python
# Run data preprocessing notebook
# Open notebooks/02_preprocessing.ipynb in VS Code

# Or run script
python scripts/data_preprocessing.py --config config/data_config.yaml
```

### 2. Model Training

```python
# Using VS Code debug configuration
F5 → Select "Python: Train Model"

# Or run directly
python scripts/train_model.py --config config/model_config.yaml
```

### 3. Generate Explanations

```python
# Using debug configuration
F5 → Select "Python: Generate Explanations"

# Or run script
python scripts/generate_explanations.py --model-path models/trained/hybrid_model.pkl
```

## 🔬 Key Components

### Hybrid Models

```python
from hybrid_xai_healthcare.models.ensemble import VotingEnsemble
from hybrid_xai_healthcare.models.hybrid import MultiModalFusion

# Create ensemble model
ensemble = VotingEnsemble(
    models=['random_forest', 'xgboost', 'neural_network'],
    voting='soft'
)

# Multi-modal fusion
fusion_model = MultiModalFusion(
    modalities=['clinical', 'imaging', 'genomic'],
    fusion_strategy='attention'
)
```

### Explainability

```python
from hybrid_xai_healthcare.explainability import SHAPExplainer, LIMEExplainer

# SHAP explanations
shap_explainer = SHAPExplainer(model, explainer_type='auto')
explanations = shap_explainer.explain_instance(patient_data)

# LIME explanations
lime_explainer = LIMEExplainer(model, mode='classification')
lime_explanation = lime_explainer.explain_instance(patient_data)
```

### Personalization

```python
from hybrid_xai_healthcare.personalization import PatientSimilarity

# Find similar patients
similarity = PatientSimilarity(method='graph_neural_network')
similar_patients = similarity.find_similar(patient_id, k=10)
```

## 🧪 Testing

### Run All Tests

```bash
# Using VS Code
F5 → Select "Python: Run Tests"

# Or using task
Ctrl+Shift+P → "Tasks: Run Task" → "Run Tests"

# Or manually
pytest tests/ -v --cov=src --cov-report=html
```

### Test Coverage

After running tests, open `htmlcov/index.html` in your browser to view detailed coverage reports.

## 📊 Monitoring and Experiment Tracking

### MLflow Dashboard

Access MLflow UI at http://localhost:5000 to:
- Track experiments and metrics
- Compare model performance
- Manage model registry
- View artifacts and plots

### Weights & Biases Integration

```python
import wandb

# Initialize W&B logging
wandb.init(project="hybrid-xai-healthcare")

# Log metrics during training
wandb.log({"accuracy": 0.95, "loss": 0.05})
```

## 🚀 Deployment

### Docker Deployment

```bash
# Build container
docker build -t hybrid-xai-healthcare .

# Run container
docker run -p 8000:8000 hybrid-xai-healthcare
```

### Kubernetes Deployment

```bash
# Apply Kubernetes manifests
kubectl apply -f deployment/kubernetes/
```

## 📚 Documentation

### Generate Documentation

```bash
# Using VS Code task
Ctrl+Shift+P → "Tasks: Run Task" → "Build Documentation"

# Or manually
sphinx-build -b html docs/source docs/build
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests and linting (`pytest`, `black`, `flake8`)
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

### Code Style

This project uses:
- **Black** for code formatting
- **Flake8** for linting
- **MyPy** for type checking
- **pytest** for testing

VS Code is configured to automatically format and lint code on save.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- SHAP and LIME libraries for explainability
- MLflow for experiment tracking
- FastAPI for API development
- The healthcare AI research community

## 📞 Support

For questions and support:
- Open an issue on GitHub
- Email: your.email@example.com
- Documentation: https://hybrid-xai-healthcare.readthedocs.io/

---

**Happy coding! 🚀**