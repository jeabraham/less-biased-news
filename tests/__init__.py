"""
Test package for less-biased-news.

This file ensures that tests can import modules from the parent directory.
"""
import sys
from pathlib import Path

# Add the parent directory to sys.path to allow importing project modules
parent_dir = Path(__file__).parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))
