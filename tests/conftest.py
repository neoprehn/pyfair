"""Pytest configuration for FAIR simulator tests."""

import matplotlib

# Use a non-interactive backend for all tests.
# This avoids Tkinter / GUI issues on Windows and CI environments.
matplotlib.use("Agg")