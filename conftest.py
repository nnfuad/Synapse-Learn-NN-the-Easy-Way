"""
Project Synapse: Global Pytest Configuration
Automatically adds python directories to sys.path so tests can import modules cleanly.
"""

import sys
import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
PYTHON_DIR = os.path.join(ROOT_DIR, "python")

for item in os.listdir(PYTHON_DIR):
    item_path = os.path.join(PYTHON_DIR, item)
    if os.path.isdir(item_path) and not item.startswith("."):
        if item_path not in sys.path:
            sys.path.insert(0, item_path)

if PYTHON_DIR not in sys.path:
    sys.path.insert(0, PYTHON_DIR)
