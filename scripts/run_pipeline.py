"""Run validation, training, and unscored answer comparison.

Run with: .venv/bin/python scripts/run_pipeline.py
"""

import subprocess
import sys
from pathlib import Path


scripts_folder = Path(__file__).resolve().parent

# Training already runs validate_data.py before updating any weights.
for script in ("training.py", "compare_answers.py"):
    print(f"Running {script}...", flush=True)
    subprocess.run([sys.executable, str(scripts_folder / script)], check=True)
