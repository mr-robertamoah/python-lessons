# Lesson 08 Exercises: Virtual Environments and Package Management

## Guided Exercises (Do with Instructor)

### Exercise 1: Create Your First Virtual Environment
**Goal**: Set up a basic virtual environment and understand the workflow

**Steps**:
1. Create a new directory called `test_project`
2. Navigate into the directory
3. Create a virtual environment called `test_env`
4. Activate the environment
5. Check that you're in the virtual environment
6. Install pandas
7. Create a requirements.txt file
8. Deactivate the environment

**Commands to use**:
```bash
mkdir test_project
cd test_project
python -m venv test_env
# Activation command depends on your OS
pip list
pip install pandas
pip freeze > requirements.txt
deactivate
```

---

### Exercise 2: Package Management Practice
**Goal**: Learn to install, upgrade, and manage packages

**Tasks**:
1. Activate your test environment from Exercise 1
2. Install numpy version 1.24.0 specifically
3. Check what version is installed
4. Upgrade numpy to the latest version
5. Install multiple packages at once: matplotlib, seaborn
6. List all installed packages
7. Show detailed information about pandas
8. Update your requirements.txt

**Verification**: Your environment should have pandas, numpy, matplotlib, and seaborn installed

---

### Exercise 3: Working with Requirements Files
**Goal**: Practice creating and using requirements files

**Create a requirements.txt file with**:
```txt
pandas>=1.5.0
numpy>=1.20.0
matplotlib>=3.5.0
requests>=2.28.0
```

**Tasks**:
1. Create a new virtual environment called `req_test_env`
2. Activate it
3. Install packages from your requirements file
4. Verify all packages are installed with correct versions
5. Add jupyter to the environment
6. Update the requirements file

---

## Independent Exercises (Try on Your Own)

### Exercise 4: Data Analysis Project Setup
**Goal**: Set up a complete data analysis project structure

**Requirements**:
Create a project called `sales_analysis` with:
- Virtual environment named `sales_env`
- Proper directory structure (data/, notebooks/, src/)
- Requirements file with data science packages
- README.md with setup instructions
- .gitignore file

**Directory structure should look like**:
```
sales_analysis/
├── sales_env/           # Virtual environment
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
├── src/
├── requirements.txt
├── README.md
└── .gitignore
```

**Packages to include**:
- pandas, numpy, matplotlib, seaborn, jupyter

---

### Exercise 5: Environment Comparison
**Goal**: Understand isolation between environments

**Tasks**:
1. Create two virtual environments:
   - `env_old`: Install pandas version 1.3.0
   - `env_new`: Install pandas version 2.0.0 (or latest)

2. For each environment:
   - Activate it
   - Check pandas version
   - Create a simple script that prints the pandas version
   - Run the script

3. Document the differences you observe

**Test script** (`check_pandas.py`):
```python
import pandas as pd
print(f"Pandas version: {pd.__version__}")
```

---

### Exercise 6: Collaborative Project Simulation
**Goal**: Practice sharing and recreating environments

**Part A - Create Project**:
1. Set up a project called `team_project`
2. Install these packages: pandas, matplotlib, requests, beautifulsoup4
3. Create a simple data analysis script
4. Generate requirements.txt
5. Create setup instructions in README.md

**Part B - Simulate Teammate**:
1. Create a new directory `team_project_copy`
2. Copy only the source files and requirements.txt (not the virtual environment)
3. Create a new virtual environment
4. Install from requirements.txt
5. Run the analysis script

**Verification**: The script should run identically in both environments

---

### Exercise 7: Troubleshooting Practice
**Goal**: Learn to diagnose and fix common environment issues

**Scenarios to practice**:

1. **Missing Package Error**:
   - Create environment and try to run script without installing dependencies
   - Practice reading error messages
   - Fix by installing correct packages

2. **Wrong Python Version**:
   - Check what Python version your environment uses
   - Compare with system Python

3. **Environment Not Activated**:
   - Try installing packages without activating environment
   - Observe where packages get installed
   - Fix by properly activating environment

**Create test scenarios and document solutions**

---

## Challenge Exercises (If You Finish Early)

### Challenge 1: Multi-Environment Project
**Goal**: Manage a project that needs different environments for different tasks

**Scenario**: Create a web scraping project that needs:
- `scraping_env`: For data collection (requests, beautifulsoup4, selenium)
- `analysis_env`: For data analysis (pandas, numpy, matplotlib, scikit-learn)
- `web_env`: For web dashboard (flask, plotly, dash)

**Requirements**:
1. Create three separate environments
2. Create requirements files for each
3. Create scripts that work in each environment
4. Document when to use each environment

---

### Challenge 2: Environment Automation
**Goal**: Create scripts to automate environment setup

**Create setup scripts**:

1. **setup_project.py**: Script that:
   - Creates project directory structure
   - Creates virtual environment
   - Installs packages from requirements.txt
   - Creates initial files (README, .gitignore)

2. **check_environment.py**: Script that:
   - Verifies all required packages are installed
   - Checks package versions
   - Reports any missing dependencies

**Test your scripts** by setting up a new project entirely through automation

---

### Challenge 3: Package Version Management
**Goal**: Handle complex version requirements

**Scenario**: You need to work with packages that have conflicting requirements:
- Package A requires numpy>=1.20.0,<1.24.0
- Package B requires numpy>=1.23.0
- Package C requires scipy>=1.9.0 (which needs numpy>=1.21.0)

**Tasks**:
1. Research actual package dependencies
2. Find compatible versions
3. Create a requirements.txt that satisfies all constraints
4. Test the installation
5. Document your solution process

---

## Troubleshooting Guide

### Common Issues and Solutions

**Problem**: Virtual environment won't activate
- **Windows**: Try using full path to activate script
- **PowerShell**: Set execution policy: `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`
- **Check**: Make sure you're in the right directory

**Problem**: pip not found after activation
- **Solution**: Make sure environment is properly activated
- **Alternative**: Use `python -m pip` instead of `pip`

**Problem**: Packages installing globally instead of in environment
- **Cause**: Environment not activated
- **Solution**: Always activate before installing packages

**Problem**: ImportError even after installing package
- **Check**: Are you in the right environment?
- **Check**: Is the package actually installed? (`pip list`)
- **Check**: Are you using the right Python? (`which python`)

**Problem**: Requirements.txt installation fails
- **Solution**: Check for typos in package names
- **Solution**: Try installing packages one by one to identify problematic ones
- **Solution**: Update pip: `pip install --upgrade pip`

---

## Self-Assessment Checklist

After completing these exercises, you should be able to:

- [ ] Create virtual environments using `python -m venv`
- [ ] Activate and deactivate virtual environments
- [ ] Install packages with pip in virtual environments
- [ ] Create and use requirements.txt files
- [ ] Set up professional project directory structures
- [ ] Understand why virtual environments are important
- [ ] Troubleshoot common environment issues
- [ ] Share projects with proper dependency management

## Best Practices Learned

### Environment Management
- Always use virtual environments for projects
- Use descriptive names for environments
- Keep one environment per project
- Don't commit virtual environment folders to Git

### Package Management
- Pin important package versions in requirements.txt
- Use `pip freeze` to capture exact versions
- Regularly update requirements.txt
- Install packages only in activated environments

### Project Organization
- Create consistent directory structures
- Document setup instructions in README
- Use .gitignore to exclude environment folders
- Keep requirements.txt in project root

## Git Reminder

Save your work:
1. Create `lesson-08-virtualenvs` folder in your repository
2. Save project structures and requirements files
3. Document your learning in notes
4. Commit and push:
   ```bash
   git add .
   git commit -m "Complete Lesson 08: Virtual Environments"
   git push
   ```

## Next Lesson Preview

In Lesson 09, we'll learn about:
- **NumPy**: The foundation of scientific computing in Python
- **Arrays**: Efficient storage and manipulation of numerical data
- **Mathematical operations**: Vectorized computations
- **Broadcasting**: Operations on arrays of different shapes
- **Performance**: Why NumPy is much faster than pure Python
