# Lesson 08: Virtual Environments and Package Management

## Learning Objectives
By the end of this lesson, you will be able to:
- Understand why virtual environments are essential
- Create and manage Python virtual environments
- Install and manage packages with pip
- Use requirements.txt files for project dependencies
- Set up professional Python project structures
- Avoid common dependency conflicts

## Why Virtual Environments Matter

### Real-World Problem: Dependency Conflicts
Imagine you're working on multiple Python projects:

**Project A**: Needs pandas version 1.3.0 for compatibility with legacy code
**Project B**: Needs pandas version 2.0.0 for new features
**Your System**: Can only have one version of pandas installed globally

```bash
# This creates a conflict!
pip install pandas==1.3.0  # For Project A
pip install pandas==2.0.0  # Overwrites 1.3.0, breaks Project A
```

### The Solution: Virtual Environments
Virtual environments create isolated Python installations for each project:

```
Your Computer
├── System Python (3.11)
├── Project A Environment
│   ├── Python 3.11
│   └── pandas 1.3.0
└── Project B Environment
    ├── Python 3.11
    └── pandas 2.0.0
```

### Benefits of Virtual Environments
- **Isolation**: Each project has its own dependencies
- **Reproducibility**: Share exact package versions with others
- **Cleanliness**: Keep system Python clean
- **Flexibility**: Test different package versions safely
- **Professional**: Industry standard practice

## Understanding Python Package Management

### What is pip?
**pip** (Pip Installs Packages) is Python's package installer:
- Downloads packages from PyPI (Python Package Index)
- Manages dependencies automatically
- Installs packages system-wide or in virtual environments

### PyPI - The Python Package Index
- Central repository of Python packages
- Over 400,000 packages available
- Includes data science libraries: pandas, numpy, matplotlib, scikit-learn

```bash
# Examples of popular packages
pip install pandas          # Data manipulation
pip install numpy           # Numerical computing
pip install matplotlib      # Data visualization
pip install requests        # HTTP library
pip install jupyter         # Interactive notebooks
```

## Creating Virtual Environments

### Method 1: Using venv (Recommended)
Python 3.3+ includes `venv` module built-in:

```bash
# Create a virtual environment
python -m venv myproject_env

# On Windows, activate it:
myproject_env\Scripts\activate

# On macOS/Linux, activate it:
source myproject_env/bin/activate

# Your prompt will change to show the environment:
(myproject_env) C:\Users\YourName>
```

### Method 2: Using virtualenv
Older method, still widely used:

```bash
# Install virtualenv first
pip install virtualenv

# Create environment
virtualenv myproject_env

# Activate (same as venv)
myproject_env\Scripts\activate  # Windows
source myproject_env/bin/activate  # macOS/Linux
```

### Virtual Environment Structure
```
myproject_env/
├── Scripts/          # Windows executables
│   ├── activate.bat  # Activation script
│   ├── python.exe    # Python interpreter
│   └── pip.exe       # Package installer
├── Lib/              # Installed packages
│   └── site-packages/
├── Include/          # Header files
└── pyvenv.cfg        # Configuration
```

## Working with Virtual Environments

### Basic Workflow
```bash
# 1. Create project directory
mkdir my_data_project
cd my_data_project

# 2. Create virtual environment
python -m venv data_env

# 3. Activate environment
data_env\Scripts\activate  # Windows
# source data_env/bin/activate  # macOS/Linux

# 4. Install packages
pip install pandas numpy matplotlib

# 5. Work on your project
python my_analysis.py

# 6. Deactivate when done
deactivate
```

### Checking Your Environment
```bash
# See which Python you're using
where python  # Windows
which python  # macOS/Linux

# See installed packages
pip list

# See pip version
pip --version

# Check if in virtual environment
echo $VIRTUAL_ENV  # macOS/Linux
echo %VIRTUAL_ENV%  # Windows
```

## Package Management with pip

### Installing Packages
```bash
# Install latest version
pip install pandas

# Install specific version
pip install pandas==1.5.0

# Install minimum version
pip install pandas>=1.4.0

# Install from requirements file
pip install -r requirements.txt

# Install multiple packages
pip install pandas numpy matplotlib seaborn
```

### Managing Packages
```bash
# List installed packages
pip list

# Show package information
pip show pandas

# Check for outdated packages
pip list --outdated

# Upgrade a package
pip install --upgrade pandas

# Uninstall a package
pip uninstall pandas
```

### Package Information
```bash
# Detailed package info
pip show pandas
# Name: pandas
# Version: 1.5.3
# Summary: Powerful data structures for data analysis
# Dependencies: numpy, python-dateutil, pytz

# List package files
pip show --files pandas
```

## Requirements Files

### Creating requirements.txt
```bash
# Generate requirements from current environment
pip freeze > requirements.txt

# This creates a file like:
# numpy==1.24.3
# pandas==1.5.3
# matplotlib==3.7.1
# seaborn==0.12.2
```

### Example requirements.txt
```txt
# Core data science packages
pandas>=1.5.0
numpy>=1.20.0
matplotlib>=3.5.0
seaborn>=0.11.0

# Additional utilities
requests>=2.28.0
openpyxl>=3.0.0  # For Excel file support

# Development tools
jupyter>=1.0.0
pytest>=7.0.0
```

### Using requirements.txt
```bash
# Install all requirements
pip install -r requirements.txt

# Install with specific index
pip install -r requirements.txt --index-url https://pypi.org/simple/

# Upgrade all packages in requirements
pip install -r requirements.txt --upgrade
```

## Professional Project Structure

### Recommended Directory Layout
```
my_data_project/
├── data_env/              # Virtual environment
├── data/                  # Data files
│   ├── raw/              # Original data
│   ├── processed/        # Cleaned data
│   └── external/         # External datasets
├── notebooks/            # Jupyter notebooks
├── src/                  # Source code
│   ├── __init__.py
│   ├── data_processing.py
│   ├── analysis.py
│   └── visualization.py
├── tests/                # Test files
├── docs/                 # Documentation
├── requirements.txt      # Dependencies
├── README.md            # Project description
└── .gitignore           # Git ignore file
```

### Setting Up a New Project
```bash
# 1. Create project structure
mkdir my_analysis_project
cd my_analysis_project

# 2. Create virtual environment
python -m venv analysis_env

# 3. Activate environment
analysis_env\Scripts\activate

# 4. Create project directories
mkdir data notebooks src tests docs
mkdir data\raw data\processed data\external

# 5. Create initial files
echo "# My Analysis Project" > README.md
echo "analysis_env/" > .gitignore
echo "*.pyc" >> .gitignore
echo "__pycache__/" >> .gitignore

# 6. Install base packages
pip install pandas numpy matplotlib jupyter

# 7. Create requirements file
pip freeze > requirements.txt
```

## Data Science Environment Setup

### Essential Packages for Data Analysis
```bash
# Activate your environment first
analysis_env\Scripts\activate

# Core data science stack
pip install pandas numpy matplotlib seaborn

# Statistical analysis
pip install scipy statsmodels

# Machine learning
pip install scikit-learn

# Jupyter notebooks
pip install jupyter ipykernel

# Data visualization
pip install plotly bokeh

# File handling
pip install openpyxl xlrd

# Web scraping and APIs
pip install requests beautifulsoup4

# Create requirements file
pip freeze > requirements.txt
```

### Jupyter Notebook Integration
```bash
# Install Jupyter in your virtual environment
pip install jupyter ipykernel

# Add your environment as a Jupyter kernel
python -m ipykernel install --user --name=analysis_env --display-name="Python (Analysis)"

# Start Jupyter
jupyter notebook

# In Jupyter, select "Python (Analysis)" kernel
```

## Common Workflows and Best Practices

### Starting a New Data Project
```bash
# 1. Project setup
mkdir sales_analysis
cd sales_analysis
python -m venv sales_env
sales_env\Scripts\activate

# 2. Install packages
pip install pandas numpy matplotlib seaborn jupyter
pip freeze > requirements.txt

# 3. Create structure
mkdir data notebooks src
echo "# Sales Analysis Project" > README.md

# 4. Start working
jupyter notebook
```

### Sharing Your Project
```bash
# 1. Create clean requirements file
pip freeze > requirements.txt

# 2. Create setup instructions in README.md
echo "## Setup Instructions" >> README.md
echo "1. Create virtual environment: python -m venv project_env" >> README.md
echo "2. Activate: project_env\Scripts\activate" >> README.md
echo "3. Install dependencies: pip install -r requirements.txt" >> README.md

# 3. Share project (without virtual environment)
# Include: src/, data/, notebooks/, requirements.txt, README.md
# Exclude: project_env/ (add to .gitignore)
```

### Collaborating with Others
```bash
# When someone shares a project with you:

# 1. Clone/download the project
# 2. Create your own virtual environment
python -m venv project_env
project_env\Scripts\activate

# 3. Install their exact dependencies
pip install -r requirements.txt

# 4. Start working with same package versions
```

## Troubleshooting Common Issues

### Environment Not Activating
```bash
# Problem: activation script not found
# Solution: Check path and use full path
C:\path\to\project\myenv\Scripts\activate

# Problem: execution policy on Windows
# Solution: Change PowerShell execution policy
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Package Installation Errors
```bash
# Problem: pip not found
# Solution: Ensure environment is activated and pip is installed
python -m ensurepip --upgrade

# Problem: permission denied
# Solution: Don't use sudo with virtual environments
# Make sure you're in activated environment

# Problem: package conflicts
# Solution: Create fresh environment
deactivate
rm -rf old_env  # or rmdir /s old_env on Windows
python -m venv new_env
```

### Version Conflicts
```bash
# Check what's installed
pip list

# Check specific package version
pip show package_name

# Install specific version
pip install package_name==1.2.3

# Uninstall and reinstall
pip uninstall package_name
pip install package_name
```

## Advanced Virtual Environment Management

### Using Different Python Versions
```bash
# If you have multiple Python versions installed
python3.9 -m venv env_39
python3.11 -m venv env_311

# Check Python version in environment
python --version
```

### Environment Variables
```bash
# Set environment variables for your project
# Create .env file:
DATABASE_URL=sqlite:///data.db
API_KEY=your_secret_key

# Load in Python:
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv('API_KEY')
```

### Conda Alternative
```bash
# Conda is another environment manager
# Install Miniconda or Anaconda first

# Create conda environment
conda create --name myproject python=3.11

# Activate
conda activate myproject

# Install packages
conda install pandas numpy matplotlib

# Export environment
conda env export > environment.yml

# Create from file
conda env create -f environment.yml
```

## Practical Examples

### Example 1: Setting Up Data Analysis Environment
```bash
# Create project for customer analysis
mkdir customer_analysis
cd customer_analysis

# Set up environment
python -m venv customer_env
customer_env\Scripts\activate

# Install data science packages
pip install pandas numpy matplotlib seaborn
pip install jupyter scikit-learn
pip install openpyxl requests

# Create project structure
mkdir data notebooks src results
mkdir data\raw data\processed

# Save dependencies
pip freeze > requirements.txt

# Create initial notebook
jupyter notebook
```

### Example 2: Machine Learning Project Setup
```bash
# ML project setup
mkdir ml_project
cd ml_project
python -m venv ml_env
ml_env\Scripts\activate

# Install ML stack
pip install pandas numpy matplotlib seaborn
pip install scikit-learn tensorflow
pip install jupyter plotly

# Create ML project structure
mkdir data models notebooks src
mkdir data\train data\test data\validation

pip freeze > requirements.txt
```

### Example 3: Web Scraping Project
```bash
# Web scraping project
mkdir web_scraper
cd web_scraper
python -m venv scraper_env
scraper_env\Scripts\activate

# Install scraping tools
pip install requests beautifulsoup4
pip install pandas lxml
pip install selenium  # For dynamic content

mkdir src data output
pip freeze > requirements.txt
```

## Best Practices Summary

### Do's
- ✅ Always use virtual environments for projects
- ✅ Create requirements.txt for every project
- ✅ Use descriptive environment names
- ✅ Keep environments project-specific
- ✅ Document setup instructions in README
- ✅ Add virtual environment folders to .gitignore

### Don'ts
- ❌ Don't install packages globally unless necessary
- ❌ Don't commit virtual environment folders to Git
- ❌ Don't share environments between unrelated projects
- ❌ Don't forget to activate environment before installing
- ❌ Don't use sudo/administrator with virtual environments

## Key Terminology

- **Virtual Environment**: Isolated Python installation for a project
- **pip**: Python package installer
- **PyPI**: Python Package Index (package repository)
- **requirements.txt**: File listing project dependencies
- **Activation**: Making a virtual environment active
- **Site-packages**: Directory where packages are installed
- **Dependency**: Package that another package needs to work
- **Package**: Reusable Python code distributed via PyPI

## Looking Ahead

In Lesson 09, we'll learn about:
- **NumPy**: Numerical computing with arrays
- **Array operations**: Mathematical operations on large datasets
- **Broadcasting**: Efficient operations on different-sized arrays
- **Performance**: Why NumPy is faster than pure Python
- **Data analysis foundations**: Building blocks for pandas
