"""Reproduce every table, figure and number of the manuscript from the saved predictions (about 30 s on a laptop).

    pip install -r requirements.txt
    python run_all.py
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
steps = ['01_results_tables.py', '02_tracking_stats.py', '03_figures.py', '04_refit_sex_classifier.py', '05_check_against_manuscript.py']
for s in steps:
    print(f'\n{"=" * 70}\n{s}\n{"=" * 70}', flush=True)
    r = subprocess.run([sys.executable, str(ROOT / 'scripts' / s)], cwd=ROOT)
    if r.returncode:
        sys.exit(f'{s} failed (exit code {r.returncode})')
print('\nDone. Tables: results/tables/ and results/RESULTS.md   Figures: results/figures/')
