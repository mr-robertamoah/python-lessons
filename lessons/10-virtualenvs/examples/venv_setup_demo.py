"""
Lesson 08 Example 1: Virtual Environment Setup Demo
Demonstrates virtual environment concepts and commands
"""

import sys
import subprocess
import os

print("=== Virtual Environment Demo ===\n")

print("1. CURRENT PYTHON ENVIRONMENT INFO")
print("=" * 40)
print(f"Python executable: {sys.executable}")
print(f"Python version: {sys.version}")
print(f"Python path: {sys.path[0]}")

# Check if in virtual environment
in_venv = hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)
print(f"In virtual environment: {in_venv}")

if 'VIRTUAL_ENV' in os.environ:
    print(f"Virtual environment path: {os.environ['VIRTUAL_ENV']}")
print()

print("2. PACKAGE MANAGEMENT COMMANDS")
print("=" * 40)
print("# Create virtual environment:")
print("python -m venv myproject_env")
print()
print("# Activate (Windows):")
print("myproject_env\\Scripts\\activate")
print()
print("# Activate (macOS/Linux):")
print("source myproject_env/bin/activate")
print()
print("# Install packages:")
print("pip install pandas numpy matplotlib")
print()
print("# Save requirements:")
print("pip freeze > requirements.txt")
print()
print("# Install from requirements:")
print("pip install -r requirements.txt")
print()
print("# Deactivate:")
print("deactivate")
print()

print("3. SAMPLE PROJECT STRUCTURE")
print("=" * 40)
project_structure = """
my_data_project/
├── data_env/              # Virtual environment (don't commit)
├── data/                  # Data files
│   ├── raw/              # Original data
│   ├── processed/        # Cleaned data
│   └── external/         # External datasets
├── notebooks/            # Jupyter notebooks
├── src/                  # Source code
│   ├── __init__.py
│   ├── data_processing.py
│   └── analysis.py
├── tests/                # Test files
├── requirements.txt      # Dependencies
├── README.md            # Project description
└── .gitignore           # Git ignore file
"""
print(project_structure)

print("4. SAMPLE REQUIREMENTS.TXT")
print("=" * 40)
sample_requirements = """
# Core data science packages
pandas>=1.5.0
numpy>=1.20.0
matplotlib>=3.5.0
seaborn>=0.11.0

# File handling
openpyxl>=3.0.0
xlrd>=2.0.0

# Web and APIs
requests>=2.28.0

# Development tools
jupyter>=1.0.0
pytest>=7.0.0

# Optional: Machine learning
scikit-learn>=1.2.0
"""
print(sample_requirements)

print("5. ENVIRONMENT SETUP CHECKLIST")
print("=" * 40)
checklist = [
    "✓ Create project directory",
    "✓ Create virtual environment",
    "✓ Activate environment", 
    "✓ Install required packages",
    "✓ Create requirements.txt",
    "✓ Set up project structure",
    "✓ Create .gitignore file",
    "✓ Initialize Git repository",
    "✓ Create README.md",
    "✓ Start coding!"
]

for item in checklist:
    print(f"  {item}")
print()

print("6. COMMON COMMANDS REFERENCE")
print("=" * 40)
commands = {
    "Create environment": "python -m venv env_name",
    "Activate (Windows)": "env_name\\Scripts\\activate",
    "Activate (Unix)": "source env_name/bin/activate",
    "Check packages": "pip list",
    "Install package": "pip install package_name",
    "Install specific version": "pip install package_name==1.2.3",
    "Upgrade package": "pip install --upgrade package_name",
    "Uninstall package": "pip uninstall package_name",
    "Save requirements": "pip freeze > requirements.txt",
    "Install requirements": "pip install -r requirements.txt",
    "Deactivate": "deactivate"
}

for description, command in commands.items():
    print(f"{description:20}: {command}")
print()

print("=== Best Practices ===")
print("• Always use virtual environments for projects")
print("• Keep requirements.txt updated")
print("• Don't commit virtual environment folders")
print("• Use descriptive environment names")
print("• Document setup instructions in README")
