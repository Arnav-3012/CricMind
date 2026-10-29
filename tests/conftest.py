"""Test-session setup."""

import os

# LightGBM (test_baselines) and torch (test_cric2vec) each bring their own OpenMP runtime on macOS; once
# LightGBM has trained in the process, a multi-threaded torch op deadlocks. One thread avoids it.
# Set before either import.
os.environ.setdefault("OMP_NUM_THREADS", "1")
