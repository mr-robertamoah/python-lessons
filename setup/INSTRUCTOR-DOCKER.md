# Instructor: Docker setup (Ubuntu)

This file explains how the instructor can use Docker to run Python and notebooks for demonstrations.

Prerequisites:
- Docker installed on your Ubuntu machine.

Quick start:
1. Create a directory to mount as a workspace (e.g., the repository root).
2. Run a Python container with a volume mount:

```bash
docker run --rm -it -p 8888:8888 -v "$(pwd)":/workspace -w /workspace python:3.11 bash
```

From inside the container you can run Python, install packages with pip, and start a Jupyter server:

```bash
pip install jupyter pandas matplotlib seaborn numpy
jupyter notebook --ip=0.0.0.0 --no-browser --allow-root
```

Open `http://localhost:8888` in your browser on the host.

Notes:
- Use the same Python and package versions as the student where possible to avoid confusion.
- Keep exercises simple and reproducible.
