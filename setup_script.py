# Create a setup script for easy project initialization

setup_script = '''#!/usr/bin/env python3
"""
Setup script for Hybrid Explainable AI Healthcare Project

This script automates the initial setup process for the project,
including environment creation, dependency installation, and
configuration file generation.
"""

import os
import subprocess
import sys
import shutil
from pathlib import Path
import json
import yaml

class ProjectSetup:
    """Automated project setup class."""
    
    def __init__(self, project_root: Path = None):
        """Initialize setup with project root directory."""
        self.project_root = project_root or Path.cwd()
        self.venv_path = self.project_root / "venv"
        
    def print_step(self, step_name: str, description: str = ""):
        """Print formatted step information."""
        print(f"\\n🔧 {step_name}")
        if description:
            print(f"   {description}")
        print("-" * 50)
        
    def run_command(self, command: list, cwd: Path = None, check: bool = True):
        """Run shell command and handle errors."""
        try:
            result = subprocess.run(
                command, 
                cwd=cwd or self.project_root, 
                check=check,
                capture_output=True, 
                text=True
            )
            if result.stdout:
                print(result.stdout)
            return result
        except subprocess.CalledProcessError as e:
            print(f"❌ Error running command: {' '.join(command)}")
            print(f"   Error: {e.stderr}")
            return None
            
    def check_python_version(self):
        """Check Python version compatibility."""
        self.print_step("Checking Python Version")
        
        version = sys.version_info
        if version.major < 3 or (version.major == 3 and version.minor < 8):
            print("❌ Python 3.8+ is required")
            print(f"   Current version: {version.major}.{version.minor}")
            sys.exit(1)
        else:
            print(f"✅ Python {version.major}.{version.minor} is compatible")
            
    def create_virtual_environment(self):
        """Create Python virtual environment."""
        self.print_step("Creating Virtual Environment")
        
        if self.venv_path.exists():
            print("⚠️  Virtual environment already exists")
            response = input("   Do you want to recreate it? (y/N): ")
            if response.lower() == 'y':
                shutil.rmtree(self.venv_path)
            else:
                print("   Skipping virtual environment creation")
                return
        
        result = self.run_command([sys.executable, "-m", "venv", str(self.venv_path)])
        if result:
            print("✅ Virtual environment created successfully")
        else:
            print("❌ Failed to create virtual environment")
            sys.exit(1)
            
    def get_pip_command(self):
        """Get pip command for the virtual environment."""
        if os.name == 'nt':  # Windows
            return str(self.venv_path / "Scripts" / "pip")
        else:  # Unix/Linux/macOS
            return str(self.venv_path / "bin" / "pip")
            
    def get_python_command(self):
        """Get python command for the virtual environment."""
        if os.name == 'nt':  # Windows
            return str(self.venv_path / "Scripts" / "python")
        else:  # Unix/Linux/macOS
            return str(self.venv_path / "bin" / "python")
            
    def install_dependencies(self):
        """Install project dependencies."""
        self.print_step("Installing Dependencies")
        
        pip_cmd = self.get_pip_command()
        
        # Upgrade pip first
        print("   Upgrading pip...")
        self.run_command([pip_cmd, "install", "--upgrade", "pip"])
        
        # Install requirements
        requirements_file = self.project_root / "requirements.txt"
        if requirements_file.exists():
            print("   Installing requirements.txt...")
            result = self.run_command([pip_cmd, "install", "-r", str(requirements_file)])
            if result:
                print("✅ Dependencies installed successfully")
            else:
                print("❌ Failed to install dependencies")
                return False
        else:
            print("⚠️  requirements.txt not found, skipping dependency installation")
            
        # Install package in development mode
        print("   Installing package in development mode...")
        result = self.run_command([pip_cmd, "install", "-e", ".[dev]"])
        if result:
            print("✅ Package installed in development mode")
        else:
            print("⚠️  Failed to install package in development mode")
            
        return True
        
    def create_config_files(self):
        """Create default configuration files."""
        self.print_step("Creating Configuration Files")
        
        config_dir = self.project_root / "config"
        config_dir.mkdir(exist_ok=True)
        
        # Model configuration
        model_config = {
            "ensemble_model": {
                "model_type": "ensemble",
                "hyperparameters": {
                    "n_estimators": 100,
                    "random_state": 42
                },
                "ensemble_config": {
                    "voting": "soft",
                    "models": ["random_forest", "xgboost", "neural_network"]
                }
            },
            "hybrid_model": {
                "model_type": "multimodal",
                "hyperparameters": {
                    "fusion_strategy": "attention",
                    "dropout_rate": 0.2
                }
            }
        }
        
        # Data configuration
        data_config = {
            "data_path": "data/processed/healthcare_data.csv",
            "features": ["age", "gender", "symptoms", "lab_results"],
            "target_column": "treatment_outcome",
            "preprocessing_steps": ["normalize", "encode_categorical", "handle_missing"],
            "validation_split": 0.2,
            "test_split": 0.1
        }
        
        # Training configuration
        training_config = {
            "epochs": 100,
            "batch_size": 32,
            "learning_rate": 0.001,
            "early_stopping": True,
            "patience": 10,
            "cross_validation_folds": 5
        }
        
        # Explainability configuration
        explainability_config = {
            "methods": ["shap", "lime"],
            "shap_config": {
                "explainer_type": "auto",
                "max_evals": 1000
            },
            "lime_config": {
                "num_features": 10,
                "num_samples": 1000
            },
            "attention_config": {
                "visualize_heads": True,
                "save_attention_maps": True
            }
        }
        
        configs = [
            ("model_config.yaml", model_config),
            ("data_config.yaml", data_config),
            ("training_config.yaml", training_config),
            ("explainability_config.yaml", explainability_config)
        ]
        
        for filename, config in configs:
            config_path = config_dir / filename
            if not config_path.exists():
                with open(config_path, 'w') as f:
                    yaml.dump(config, f, default_flow_style=False)
                print(f"   ✅ Created {filename}")
            else:
                print(f"   ⚠️  {filename} already exists, skipping")
                
    def create_directory_structure(self):
        """Create project directory structure."""
        self.print_step("Creating Directory Structure")
        
        directories = [
            "data/raw",
            "data/processed", 
            "data/external",
            "data/interim",
            "models/trained",
            "models/checkpoints",
            "models/ensemble_weights",
            "experiments/configs",
            "experiments/results",
            "experiments/plots",
            "logs",
            "docs/source",
            "docs/build",
            "docs/images"
        ]
        
        for dir_path in directories:
            full_path = self.project_root / dir_path
            full_path.mkdir(parents=True, exist_ok=True)
            
            # Create .gitkeep file to track empty directories
            gitkeep_path = full_path / ".gitkeep"
            if not gitkeep_path.exists():
                gitkeep_path.touch()
                
        print("✅ Directory structure created")
        
    def initialize_dvc(self):
        """Initialize DVC for data version control."""
        self.print_step("Initializing DVC")
        
        try:
            # Check if DVC is installed
            self.run_command([self.get_pip_command(), "show", "dvc"])
            
            # Initialize DVC
            result = self.run_command(["dvc", "init"], check=False)
            if result and result.returncode == 0:
                print("✅ DVC initialized successfully")
                
                # Add data directory to DVC
                data_dir = self.project_root / "data" / "raw"
                if data_dir.exists() and any(data_dir.iterdir()):
                    self.run_command(["dvc", "add", "data/raw"])
                    print("✅ Data directory added to DVC")
            else:
                print("⚠️  DVC initialization skipped (may already be initialized)")
                
        except Exception as e:
            print(f"⚠️  DVC initialization failed: {e}")
            print("   Install DVC manually: pip install dvc")
            
    def create_env_file(self):
        """Create .env file from template."""
        self.print_step("Creating Environment File")
        
        env_example = self.project_root / ".env.example"
        env_file = self.project_root / ".env"
        
        if env_example.exists() and not env_file.exists():
            shutil.copy(env_example, env_file)
            print("✅ Created .env file from template")
            print("   📝 Please edit .env with your configuration")
        else:
            if env_file.exists():
                print("⚠️  .env file already exists")
            else:
                print("⚠️  .env.example not found, skipping")
                
    def setup_git_hooks(self):
        """Set up git hooks for code quality."""
        self.print_step("Setting up Git Hooks")
        
        try:
            # Install pre-commit if available
            pip_cmd = self.get_pip_command()
            result = self.run_command([pip_cmd, "show", "pre-commit"], check=False)
            
            if result and result.returncode == 0:
                # Install git hooks
                python_cmd = self.get_python_command()
                result = self.run_command([python_cmd, "-m", "pre_commit", "install"])
                if result:
                    print("✅ Git hooks installed successfully")
                else:
                    print("⚠️  Failed to install git hooks")
            else:
                print("⚠️  pre-commit not installed, skipping git hooks")
                
        except Exception as e:
            print(f"⚠️  Git hooks setup failed: {e}")
            
    def display_next_steps(self):
        """Display next steps for the user."""
        self.print_step("Setup Complete! 🎉")
        
        print("Next steps:")
        print("1. Open the project in VS Code:")
        print("   code .")
        print()
        print("2. Install recommended VS Code extensions when prompted")
        print()
        print("3. Select the Python interpreter:")
        print(f"   {self.get_python_command()}")
        print()
        print("4. Configure your .env file with your settings")
        print()
        print("5. Start exploring the notebooks:")
        print("   - notebooks/01_data_exploration.ipynb")
        print("   - notebooks/02_preprocessing.ipynb")
        print()
        print("6. Start MLflow server:")
        print("   mlflow server --backend-store-uri sqlite:///mlflow.db")
        print()
        print("7. Run your first training:")
        print("   python scripts/train_model.py")
        print()
        print("Happy coding! 🚀")
        
    def run_setup(self):
        """Run the complete setup process."""
        print("🏥 Hybrid Explainable AI Healthcare - Project Setup")
        print("=" * 55)
        
        try:
            self.check_python_version()
            self.create_virtual_environment()
            self.install_dependencies()
            self.create_directory_structure()
            self.create_config_files()
            self.create_env_file()
            self.initialize_dvc()
            self.setup_git_hooks()
            self.display_next_steps()
            
        except KeyboardInterrupt:
            print("\\n\\n⚠️  Setup interrupted by user")
            sys.exit(1)
        except Exception as e:
            print(f"\\n\\n❌ Setup failed with error: {e}")
            sys.exit(1)


if __name__ == "__main__":
    setup = ProjectSetup()
    setup.run_setup()
'''

# Create a simple batch setup script for Windows
batch_setup_script = '''@echo off
echo 🏥 Hybrid Explainable AI Healthcare - Quick Setup
echo ================================================

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python is not installed or not in PATH
    echo Please install Python 3.8+ from python.org
    pause
    exit /b 1
)

echo ✅ Python found
echo.

REM Run the Python setup script
echo Running setup script...
python setup.py

echo.
echo Setup complete! Run "code ." to open in VS Code.
pause
'''

# Create shell setup script for Unix/Linux/macOS
shell_setup_script = '''#!/bin/bash

echo "🏥 Hybrid Explainable AI Healthcare - Quick Setup"
echo "================================================"

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed"
    echo "Please install Python 3.8+ using your system package manager"
    exit 1
fi

echo "✅ Python found"
echo

# Run the Python setup script
echo "Running setup script..."
python3 setup.py

echo
echo "Setup complete! Run 'code .' to open in VS Code."
'''

print("🛠️ Setup Scripts Created:")
print("=" * 30)
print("✅ setup.py - Main Python setup script")
print("✅ setup.bat - Windows batch setup script") 
print("✅ setup.sh - Unix/Linux/macOS shell setup script")
print("\n🎯 Setup Features:")
print("- Automated virtual environment creation")
print("- Dependency installation and package setup")
print("- Directory structure creation")
print("- Configuration file generation")
print("- DVC initialization for data versioning")
print("- Git hooks setup with pre-commit")
print("- Environment file creation from template")
print("- Comprehensive next steps guide")