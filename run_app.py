#!/usr/bin/env python3
"""
Convenience script to run the Forensic RAG Streamlit application.
Usage: python run_app.py
"""
import sys
import subprocess
import os
from pathlib import Path

# Get project root directory
project_root = Path(__file__).parent.resolve()

# Set PYTHONPATH to include project root
env = os.environ.copy()
env['PYTHONPATH'] = str(project_root)

if __name__ == "__main__":
    # Run Streamlit app with proper environment
    app_path = project_root / "src" / "app.py"
    subprocess.run(["streamlit", "run", str(app_path)], env=env)

