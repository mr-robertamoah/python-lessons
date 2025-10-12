# Student: Windows 11 setup

Steps for the student to install Python and set up a working environment:

1. Install Python 3.11+ from the Microsoft Store or python.org. During installation, check "Add Python to PATH".
2. Install Visual Studio Code: https://code.visualstudio.com/
3. In VS Code install the Python extension (ms-python.python).
4. Open the lesson folder in VS Code and create a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install pandas numpy matplotlib seaborn jupyter
```

5. Run Jupyter (if needed):

```powershell
jupyter notebook
```

If PowerShell blocks scripts, run `Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned` as admin.

Troubleshooting:
- If commands fail, confirm Python is installed and the PATH is correct by running `python --version`.
